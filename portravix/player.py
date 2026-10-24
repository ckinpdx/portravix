"""Synced audio/video player. The 48 kHz output stream is the clock: the window shows the
frame the audio has reached, and if rendering falls behind, both stall together."""
import collections
import threading
import time

import cv2
import numpy as np
import sounddevice as sd

from .speech_buffer import PLAYBACK_RATE

FPS = 24  # MiniMax H3; the Player takes the face's rate


class Player:
    """headless=True: no window and no local audio; a real-time virtual clock consumes the audio
    instead, so rendering stays paced while the video goes only to listeners (e.g. the phone page)."""

    def __init__(self, output_device=None, window="Portravix", scale=1.5, recorder=None, headless=False, fps=FPS):
        self.window, self.scale, self.headless = window, scale, headless
        self.fps = fps
        self.samples_per_frame = PLAYBACK_RATE // fps  # 2000 at 24 fps
        self.listeners = [recorder.add] if recorder is not None else []  # called with (frames, audio, speech)
        self._lock = threading.Lock()
        self._audio = collections.deque()   # float32 blocks waiting to play
        self._audio_head = 0                # samples already consumed from _audio[0]
        self._queued_samples = 0
        self._played = 0                    # samples played = the clock
        self._frames = collections.deque()  # (global_index, frame)
        self._next_frame_index = 0
        self._chunk_starts = collections.deque()  # (sample position, speech flag) of enqueued chunks
        self.speaking = False               # True while a chunk that contains speech is playing
        self.events = []                    # (wall time, sample position, speech flag) when a chunk starts playing
        self.underruns = 0
        self.underrun_log = []              # (wall time, missing samples)
        self.stop_event = threading.Event()
        self._stream = None if headless else sd.OutputStream(
            samplerate=PLAYBACK_RATE, channels=1, dtype="float32", device=output_device,
            blocksize=1024, latency="high", callback=self._callback)

    # ---- producer side (renderer thread)
    def enqueue(self, frames, audio, speech=False):
        assert len(audio) == len(frames) * self.samples_per_frame, (len(audio), len(frames))
        with self._lock:
            self._chunk_starts.append((self._queued_samples + self._played, speech))
            for frame in frames:
                self._frames.append((self._next_frame_index, frame))
                self._next_frame_index += 1
            self._audio.append(audio.astype(np.float32))
            self._queued_samples += len(audio)
        for listener in self.listeners:
            listener(frames, audio, speech)

    def lead_seconds(self):
        """Audio queued but not yet played."""
        with self._lock:
            return self._queued_samples / PLAYBACK_RATE

    # ---- audio thread
    def _callback(self, outdata, n, time_info, status):
        out = outdata[:, 0]
        filled = 0
        with self._lock:
            while filled < n and self._audio:
                block = self._audio[0]
                take = min(n - filled, len(block) - self._audio_head)
                out[filled:filled + take] = block[self._audio_head:self._audio_head + take]
                filled += take
                self._audio_head += take
                if self._audio_head == len(block):
                    self._audio.popleft()
                    self._audio_head = 0
            self._queued_samples -= filled
            self._played += filled
            while self._chunk_starts and self._chunk_starts[0][0] <= self._played:
                pos, speech = self._chunk_starts.popleft()
                self.speaking = speech
                self.events.append((time.perf_counter(), pos, speech))
        if filled < n:
            out[filled:] = 0
            if self._played > 0:
                self.underruns += 1
                self.underrun_log.append((time.perf_counter(), n - filled))

    # ---- main thread
    def _current_frame(self):
        index = self._played // self.samples_per_frame
        with self._lock:
            while len(self._frames) > 1 and self._frames[1][0] <= index:
                self._frames.popleft()
            return self._frames[0][1] if self._frames else None

    def run(self, until=None):
        """Display loop; returns when the window closes, Esc is pressed, stop() is called, or until() is true."""
        if self.headless:
            return self._run_virtual_clock(until)
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        shown = None
        started = False
        try:
            while not self.stop_event.is_set():
                if not started and self.lead_seconds() > 0:
                    self._stream.start()
                    started = True
                frame = self._current_frame()
                if frame is not None and frame is not shown:
                    img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    if self.scale != 1:
                        img = cv2.resize(img, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_LANCZOS4)
                    cv2.imshow(self.window, img)
                    shown = frame
                if (cv2.waitKey(5) & 0xFF) == 27 or cv2.getWindowProperty(self.window, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if until is not None and until():
                    break
        finally:
            self.stop_event.set()
            if started:
                self._stream.stop()
            self._stream.close()
            cv2.destroyWindow(self.window)

    def _run_virtual_clock(self, until):
        """Headless: consume queued audio in real time, as a sound card would, without playing it."""
        block = 1024
        scratch = np.zeros((block, 1), np.float32)
        start = None
        consumed = 0
        try:
            while not self.stop_event.is_set():
                if until is not None and until():
                    break
                if start is None:
                    if self.lead_seconds() == 0:
                        time.sleep(0.01)
                        continue
                    start = time.perf_counter()
                due = int((time.perf_counter() - start) * PLAYBACK_RATE)
                while consumed + block <= due:
                    self._callback(scratch, block, None, None)
                    consumed += block
                time.sleep(0.005)
        finally:
            self.stop_event.set()

    def stop(self):
        self.stop_event.set()
