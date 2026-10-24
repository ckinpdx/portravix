"""Queue of speech audio waiting to be rendered, held at the two rates the pipeline needs.

TTS audio (24 kHz) goes in once; the face pops slices at its own rate (48 kHz for the H3 face worker)
and the player gets the matching 48 kHz slice, so both stay on the same timeline.
"""
import threading

import numpy as np
import soxr

FACE_RATE = 48000      # the face worker's input rate
PLAYBACK_RATE = 48000  # output device rate


class SpeechBuffer:
    def __init__(self, in_rate=24000, face_rate=FACE_RATE):
        self.in_rate, self.face_rate = in_rate, face_rate
        self._lock = threading.Lock()
        self._face = np.zeros(0, np.float32)
        self._playback = np.zeros(0, np.float32)
        self._new_streams()

    def _new_streams(self):
        self._to_face = soxr.ResampleStream(self.in_rate, self.face_rate, 1, dtype="float32")
        self._to_playback = soxr.ResampleStream(self.in_rate, PLAYBACK_RATE, 1, dtype="float32")

    def push(self, audio, last=False):
        """Append TTS audio. last=True flushes the resamplers at the end of an utterance."""
        audio = np.asarray(audio, np.float32)
        with self._lock:
            self._face = np.concatenate([self._face, self._to_face.resample_chunk(audio, last=last)])
            self._playback = np.concatenate([self._playback, self._to_playback.resample_chunk(audio, last=last)])
            if last:
                self._new_streams()

    def pop(self, seconds):
        """Take `seconds` of audio at both rates, padding with silence. Returns (face, playback, speech_seconds)."""
        n_face, n_play = round(seconds * self.face_rate), round(seconds * PLAYBACK_RATE)
        with self._lock:
            face, self._face = self._face[:n_face], self._face[n_face:]
            play, self._playback = self._playback[:n_play], self._playback[n_play:]
        speech_seconds = len(face) / self.face_rate
        face = np.pad(face, (0, n_face - len(face)))
        play = np.pad(play, (0, n_play - len(play)))
        return face, play, speech_seconds

    def pending_seconds(self):
        with self._lock:
            return len(self._face) / self.face_rate

    def clear(self):
        with self._lock:
            self._face = np.zeros(0, np.float32)
            self._playback = np.zeros(0, np.float32)
            self._new_streams()
