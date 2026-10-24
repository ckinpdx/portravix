"""Keeps the player fed: renders the next chunk (speech if queued, else silence) whenever the player's lead drops
below one render time plus a margin, so the face never runs far ahead of the voice."""
import threading
import time

import numpy as np


class Renderer(threading.Thread):
    def __init__(self, face, speech, player, margin=0.25, warmup=None, mood=None, talk_mood=None, log=print):
        """mood: returns what the conversation is doing (idle / listen / think) for chunks without speech.
        talk_mood: returns the latest reply's mood, for chunks with speech."""
        super().__init__(daemon=True)
        self.face, self.speech, self.player = face, speech, player
        self.mood = mood or (lambda: "idle")
        self.talk_mood = talk_mood or (lambda: None)
        self.margin = margin
        self.warmup = face.warmup if warmup is None else warmup
        self.log = log
        self.render_seconds = face.chunk_seconds  # refined from measurements
        self.ready = threading.Event()
        self.stop_event = threading.Event()
        self.stats = []  # (wall time, render seconds, speech seconds, lead before, lead after)

    def run(self):
        silence = np.zeros(round(self.face.chunk_seconds * self.face.audio_rate), np.float32)
        for _ in range(self.warmup):
            self.face.render(silence)
        self.ready.set()
        while not self.stop_event.is_set():
            lead_before = self.player.lead_seconds()
            if lead_before > self.render_seconds + self.margin:
                time.sleep(0.005)
                continue
            face_audio, playback_audio, speech_seconds = self.speech.pop(self.face.chunk_seconds)
            t = time.perf_counter()
            if speech_seconds > 0:
                frames = self.face.render(face_audio, mode="talk", mood=self.talk_mood())
            else:
                frames = self.face.render(face_audio, mode=self.mood())
            took = time.perf_counter() - t
            self.render_seconds = 0.8 * self.render_seconds + 0.2 * took
            if took > lead_before and self.player._played > 0:
                self.log(f"[face] render took {took:.2f}s with {lead_before:.2f}s queued: playback stalled "
                         f"{took - lead_before:.2f}s")
            self.player.enqueue(frames, playback_audio, speech=speech_seconds > 0)
            self.stats.append((time.perf_counter(), took, speech_seconds, lead_before, self.player.lead_seconds()))
