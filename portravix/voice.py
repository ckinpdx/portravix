"""Client for workers/tts_worker.py: sends sentences, pushes the streamed audio into the SpeechBuffer."""
import collections
import json
import os
import struct
import subprocess
import sys
import threading
import time

import numpy as np

from . import config as C


class Voice:
    def __init__(self, speech, cfg):
        """cfg: the [voice] settings. The worker runs in the app's own Python unless [voice] python says otherwise."""
        self.speech = speech
        os.makedirs(os.path.join(C.ROOT, "logs"), exist_ok=True)
        self._log = open(os.path.join(C.ROOT, "logs", "tts_worker.log"), "ab")
        voice = cfg["voice"]
        if os.path.exists(C.resolve(voice)):
            voice = C.resolve(voice)  # a file to clone; otherwise a preset name
        cmd = [C.resolve(cfg["python"]) or sys.executable, "-X", "utf8", os.path.join(C.ROOT, "workers", "tts_worker.py"),
               "--voice", voice, "--threads", str(cfg["threads"]), "--language", cfg["language"]]
        if cfg["config"]:
            cmd += ["--config", C.resolve(cfg["config"])]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log)
        kind = self._proc.stdout.read(1)
        if kind != b"R":
            raise RuntimeError(f"TTS worker failed to start, see logs\\tts_worker.log (got {kind!r})")
        self.sample_rate = struct.unpack("<I", self._proc.stdout.read(4))[0]
        if self.sample_rate != speech.in_rate:
            raise ValueError(f"TTS rate {self.sample_rate} != SpeechBuffer in_rate {speech.in_rate}")
        self._lock = threading.Lock()
        self._next_id = 0
        self._pending = collections.deque()  # ids sent but not finished, in order
        self._done = {}
        self.first_audio_time = {}           # request id -> when its first audio reached the SpeechBuffer
        threading.Thread(target=self._reader, daemon=True).start()

    def say(self, text):
        """Queue one sentence; the worker speaks requests in order. Returns its request id."""
        with self._lock:
            rid = self._next_id
            self._next_id += 1
            self._done[rid] = threading.Event()
            self._pending.append(rid)
        self._proc.stdin.write((json.dumps({"id": rid, "text": text}) + "\n").encode("utf-8"))
        self._proc.stdin.flush()
        return rid

    def wait(self, rid, timeout=None):
        return self._done[rid].wait(timeout)

    def _reader(self):
        stream = self._proc.stdout
        while True:
            kind = stream.read(1)
            if not kind:
                return
            (value,) = struct.unpack("<I", stream.read(4))
            if kind == b"A":
                audio = np.frombuffer(stream.read(value), "<f4")
                with self._lock:
                    rid = self._pending[0]
                self.first_audio_time.setdefault(rid, time.perf_counter())
                self.speech.push(audio)
            elif kind == b"E":
                self.speech.push(np.zeros(0, np.float32), last=True)  # flush the resamplers at the end of the sentence
                with self._lock:
                    self._pending.popleft()
                self._done[value].set()

    def close(self):
        if self._proc.poll() is None:
            self._proc.stdin.close()
            self._proc.wait(timeout=10)
        self._log.close()
