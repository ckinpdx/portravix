"""The face: a client for workers/face_worker.py, which renders MiniMax H3 in ComfyUI's Python.

Each render() hands the worker one chunk of 48 kHz audio and gets that chunk's frames back: 51 frames (2.125 s)
at 24 fps with the default 56-frame window, 34 (1.417 s) with a 39-frame window.
"""
import json
import os
import struct
import subprocess
import time
from multiprocessing import shared_memory

import numpy as np

from . import config as C


class Face:
    AUDIO_RATE = 48000  # the worker's input rate, needed before it starts (the SpeechBuffer is built first)

    def __init__(self, cfg, log=print):
        self.log = log
        os.makedirs(os.path.join(C.ROOT, "logs"), exist_ok=True)
        self._log = open(os.path.join(C.ROOT, "logs", "face_worker.log"), "ab")
        python = C.comfy_python(cfg)
        if not os.path.exists(python):
            raise RuntimeError(f"ComfyUI's Python not found at {python} (set [comfyui] dir or python)")
        t = time.perf_counter()
        self._proc = subprocess.Popen([python, os.path.join(C.ROOT, "workers", "face_worker.py"), "--config", cfg["_path"]],
                                      stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log)
        if self._read(1) != b"R":
            raise RuntimeError("the face worker failed to start, see logs\\face_worker.log")
        (n,) = struct.unpack("<I", self._read(4))
        header = json.loads(self._read(n))
        self.fps, self.audio_rate = header["fps"], header["audio_rate"]
        self.frames_per_chunk = header["frames_per_chunk"]
        self.chunk_seconds = self.frames_per_chunk / self.fps
        self.warmup = 0  # the worker renders its own warm-up window before it reports ready
        self._samples = header["samples_per_chunk"]
        self._shape = (self.frames_per_chunk, header["height"], header["width"], 3)
        self._shm = shared_memory.SharedMemory(name=header["shm"])
        self._shm_frames = np.ndarray(self._shape, np.uint8, buffer=self._shm.buf)
        self.last_timing = None
        self.timings = []  # the worker's own timing per chunk, plus send / wait / read on this side
        log(f"[face] ready in {time.perf_counter() - t:.1f}s: {self.frames_per_chunk} frames ({self.chunk_seconds:.3f}s) "
            f"per chunk at {header['width']}x{header['height']}, {self.fps} fps")

    def _read(self, n):
        buf = b""
        while len(buf) < n:
            part = self._proc.stdout.read(n - len(buf))
            if not part:
                raise RuntimeError("the face worker exited, see logs\\face_worker.log")
            buf += part
        return buf

    def render(self, face_audio):
        """face_audio: chunk_seconds of 48 kHz mono. Returns uint8 RGB frames [frames_per_chunk, H, W, 3]."""
        audio = np.asarray(face_audio, "<f4")
        if len(audio) != self._samples:
            audio = np.pad(audio[:self._samples], (0, max(0, self._samples - len(audio))))
        t0 = time.perf_counter()
        body = audio.tobytes()
        self._proc.stdin.write(b"C" + struct.pack("<I", len(body)) + body)
        self._proc.stdin.flush()
        t1 = time.perf_counter()
        if self._read(1) != b"F":
            raise RuntimeError("face worker protocol error")
        t2 = time.perf_counter()
        (m,) = struct.unpack("<I", self._read(4))
        self.last_timing = json.loads(self._read(m))
        (nbytes,) = struct.unpack("<I", self._read(4))
        if nbytes:
            frames = np.frombuffer(bytearray(self._read(nbytes)), np.uint8).reshape(self._shape)
        else:
            frames = self._shm_frames.copy()  # copied out before the next request lets the worker overwrite it
        self.last_timing.update({"send_s": t1 - t0, "wait_s": t2 - t1, "read_s": time.perf_counter() - t2})
        self.timings.append(self.last_timing)
        return frames

    def close(self):
        del self._shm_frames
        self._shm.close()
        if self._proc.poll() is None:
            try:
                self._proc.stdin.write(b"Q")
                self._proc.stdin.flush()
                self._proc.wait(timeout=30)
            except Exception:
                self._proc.kill()
        self._log.close()
