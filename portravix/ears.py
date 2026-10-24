"""Ears: finds utterances in a live 16 kHz mic stream with Silero VAD, and writes them down with Parakeet.
Both run on the CPU (ONNX), so they never compete with the face for the GPU.
One UtteranceDetector per listening page (the VAD session is shared); one Transcriber per avatar.

models/silero_vad_v6.onnx is the copy faster-whisper 1.2.1 bundles (Silero VAD, MIT license): 512-sample
frames with 64 samples of the previous frame as context, and an LSTM state (h, c) carried between calls."""
import collections
import os
import threading

import numpy as np
import onnxruntime

RATE = 16000
FRAME = 512  # 32 ms
CONTEXT = 64
MODEL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "silero_vad_v6.onnx")

_session = None
_session_lock = threading.Lock()


def _vad_session():
    global _session
    with _session_lock:
        if _session is None:
            opts = onnxruntime.SessionOptions()
            opts.inter_op_num_threads = opts.intra_op_num_threads = 1
            opts.log_severity_level = 4
            _session = onnxruntime.InferenceSession(MODEL, providers=["CPUExecutionProvider"], sess_options=opts)
    return _session


class UtteranceDetector:
    """feed() 16 kHz float32 audio; get back ("start", None) when speech begins and ("end", audio) when it
    has been followed by end_ms of silence. A start needs start_ms of speech; the utterance keeps preroll_ms
    from before the start (first syllable) and tail_ms after it. Shorter than min_ms of speech: ("discard", None).
    Longer than max_s: cut off and returned."""

    def __init__(self, start=0.5, stop=0.35, start_ms=96, end_ms=700, preroll_ms=300, tail_ms=200, min_ms=250, max_s=30):
        frames = lambda ms: max(1, round(ms / 1000 * RATE / FRAME))  # noqa: E731
        self.start_p, self.stop_p = start, stop
        self.start_frames, self.end_frames = frames(start_ms), frames(end_ms)
        self.preroll_frames, self.tail_frames = frames(preroll_ms), frames(tail_ms)
        self.min_frames, self.max_frames = frames(min_ms), frames(max_s * 1000)
        self.session = _vad_session()
        self.reset()

    def reset(self):
        """Forget everything, e.g. when the mic reopens after she has spoken."""
        self.h = np.zeros((1, 1, 128), np.float32)
        self.c = np.zeros((1, 1, 128), np.float32)
        self.context = np.zeros(CONTEXT, np.float32)
        self.pending = np.zeros(0, np.float32)
        self.pre = collections.deque(maxlen=self.preroll_frames)
        self.utterance = None  # frames, while speech is on
        self.run = self.voiced = self.quiet = 0

    def _prob(self, frame):
        x = np.concatenate([self.context, frame])[None]
        self.context = frame[-CONTEXT:]
        p, self.h, self.c = self.session.run(None, {"input": x, "h": self.h, "c": self.c})
        return float(p.reshape(-1)[0])

    def feed(self, audio):
        events = []
        self.pending = np.concatenate([self.pending, np.asarray(audio, np.float32)])
        while len(self.pending) >= FRAME:
            frame, self.pending = self.pending[:FRAME], self.pending[FRAME:]
            event = self._step(frame, self._prob(frame))
            if event:
                events.append(event)
        return events

    def _step(self, frame, p):
        if self.utterance is None:
            self.pre.append(frame)
            self.run = self.run + 1 if p >= self.start_p else 0
            if self.run >= self.start_frames:
                self.utterance, self.voiced, self.quiet = list(self.pre), self.run, 0
                return ("start", None)
            return None
        self.utterance.append(frame)
        if p >= self.start_p:
            self.voiced += 1
        self.quiet = self.quiet + 1 if p < self.stop_p else 0
        if self.quiet < self.end_frames and len(self.utterance) < self.max_frames:
            return None
        frames = self.utterance[:len(self.utterance) - max(0, self.quiet - self.tail_frames)]
        voiced = self.voiced
        self.utterance, self.run = None, 0
        self.pre.clear()
        if voiced < self.min_frames:
            return ("discard", None)
        return ("end", np.concatenate(frames))


class Transcriber:
    """Writes down speech: NVIDIA Parakeet TDT 0.6B v2 (English, CC-BY-4.0), the int8 ONNX export by istupakov, run
    by onnx-asr on the CPU. Punctuated, capitalised text. Measured on a Ryzen 9 7950X: 0.19 s for 5 s of speech on 4
    threads, every test clip word for word (Gemma 4 E4B writing the same clips down itself got about half the words wrong)."""

    def __init__(self, model_dir=None, threads=4):
        """model_dir: a local copy of istupakov/parakeet-tdt-0.6b-v2-onnx; None downloads it (int8 set, ~660 MB)."""
        import onnx_asr  # only the avatar needs it, not the launcher that imports this module

        opts = onnxruntime.SessionOptions()
        opts.intra_op_num_threads = threads  # leave cores for the TTS worker and the H.264 encoder
        opts.log_severity_level = 3
        self.model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v2", model_dir or None, quantization="int8",
                                         providers=["CPUExecutionProvider"], sess_options=opts)
        self.model.recognize(np.zeros(RATE, np.float32))  # warm-up

    def transcribe(self, audio):
        """16 kHz float32 mono -> text ("" for no words)."""
        return self.model.recognize(np.asarray(audio, np.float32)).strip()
