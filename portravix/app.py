"""Wires the pieces together: Brain (LLM) -> Voice (TTS) -> SpeechBuffer -> Renderer (face) -> Player."""
import threading
import time

from . import config as C
from .brain import Brain
from .ears import Transcriber
from .face import Face
from .player import Player
from .recorder import Recorder
from .renderer import Renderer
from .speech_buffer import SpeechBuffer
from .voice import Voice


class Avatar:
    def __init__(self, cfg=None, record_path=None, headless=False, log=print):
        """cfg: settings from config.load() (loaded here when None). headless=True: no window or local audio on this
        machine, e.g. when only the phone page watches."""
        self.cfg = cfg = cfg or C.load()
        self.log = log
        t = time.perf_counter()
        self.speech = SpeechBuffer(in_rate=24000, face_rate=Face.AUDIO_RATE)
        self.voice = Voice(self.speech, cfg["voice"])
        b = cfg["brain"]
        self.brain = Brain(b["url"], b["model"], C.resolve(b["persona"]), max_turns=int(b["history_turns"]),
                           max_tokens=int(b["max_tokens"]), audio=bool(b["audio"]),
                           moods=[m for m, _ in C.moods(cfg)], log=log)
        self.talk_mood = None  # the mood the latest reply was tagged with: picks the face's talking prompt
        self.ears = None
        self._ears_ready = threading.Event()
        threading.Thread(target=self._load_ears, daemon=True).start()  # ~2 s, in the shadow of the face
        # The face first, then the LLM: their load peaks never coincide on the GPU. The LLM is warmed up before the
        # avatar goes live, or a message sent right after Start waits for it to load (8 s with Gemma 4 E4B).
        self.face = Face(cfg, log=log)
        try:
            self.brain.warm_up()
        except Exception as e:
            log(f"[brain] warm-up failed ({type(e).__name__}: {e}); replies will fail until {b['url']} answers")
        log(f"[avatar] models loaded in {time.perf_counter() - t:.1f}s")
        self.recorder = Recorder(record_path, fps=self.face.fps) if record_path else None
        self.player = Player(recorder=self.recorder, headless=headless, fps=self.face.fps)
        self._mood, self._mood_until = "idle", 0.0
        self.renderer = Renderer(self.face, self.speech, self.player, mood=self.mood,
                                 talk_mood=lambda: self.talk_mood, log=log)
        self.turns = []
        self._turn_lock = threading.Lock()  # one conversation turn at a time, whoever sends it

    def set_mood(self, mood, seconds=30.0):
        """What the face shows while nobody is speaking: "listen" (you are talking), "think" (her reply is being
        written) or "idle". listen and think expire after `seconds`, so a lost event can't leave her stuck."""
        self._mood, self._mood_until = mood, time.perf_counter() + seconds

    def mood(self):
        return self._mood if self._mood == "idle" or time.perf_counter() < self._mood_until else "idle"

    def _load_ears(self):
        e = self.cfg["ears"]
        self.ears = Transcriber(C.resolve(e["parakeet_dir"]) or None, threads=int(e["threads"]))
        self._ears_ready.set()

    def respond(self, text, images=(), on_sentence=None, audio=None, on_heard=None):
        """One turn: stream the LLM reply into TTS sentence by sentence. Returns once all audio is queued.
        on_sentence(text) is called as each sentence is handed to the voice.
        audio: a voice message instead of text (16 kHz float32). Parakeet writes down the words, passed to
        on_heard(text); the LLM then gets the words (and the audio, if [brain] audio is on). No words, no reply."""
        with self._turn_lock:
            t0 = time.perf_counter()
            self.set_mood("think")
            try:
                if audio is not None:
                    self._ears_ready.wait()
                    text = self.ears.transcribe(audio)
                    self.log(f"[ears] {len(audio) / 16000:.1f}s of speech heard in {time.perf_counter() - t0:.2f}s: {text!r}")
                    if on_heard:
                        on_heard(text)
                    if not text:
                        return {"t0": t0, "user": "", "reply": "", "voice": True}
                return self._respond(text, images, on_sentence, audio, t0)
            finally:
                self.set_mood("idle")  # her speech is queued by now; chunks with speech talk regardless

    def _respond(self, text, images, on_sentence, audio=None, t0=None):
        t0 = t0 or time.perf_counter()
        sentences, ids = [], []
        for sentence in self.brain.reply(text, images, audio=audio):
            if not sentences:  # the tag came before the first sentence: set the face's mood before its audio
                self.talk_mood = self.brain.mood
                if self.brain.moods:
                    self.log(f"[brain] mood: {self.brain.mood or 'none (default talking prompt)'}")
            sentences.append(sentence)
            ids.append(self.voice.say(sentence))
            if on_sentence:
                on_sentence(sentence)
        if ids:
            self.voice.wait(ids[-1])
        turn = {"t0": t0, "user": text, "reply": " ".join(sentences), "voice": audio is not None, "mood": self.brain.mood,
                "llm_first_sentence": self.brain.timing.get("first_sentence", t0) - t0,
                "tts_first_audio": self.voice.first_audio_time[ids[0]] - t0 if ids else None}
        self.turns.append(turn)
        return turn

    def mouth_start(self, turn, timeout=10):
        """Seconds from the message to the first chunk with her speech starting to play."""
        end = time.perf_counter() + timeout
        while time.perf_counter() < end:
            for t, _, speech in list(self.player.events):
                if speech and t > turn["t0"]:
                    return t - turn["t0"]
            time.sleep(0.01)
        return None

    def wait_until_quiet(self, settle=0.5, timeout=120):
        """Block until all queued speech has been played."""
        end = time.perf_counter() + timeout
        quiet_since = None
        while time.perf_counter() < end:
            quiet = self.speech.pending_seconds() == 0 and not self.player.speaking
            quiet_since = (quiet_since or time.perf_counter()) if quiet else None
            if quiet_since and time.perf_counter() - quiet_since >= settle:
                return True
            time.sleep(0.02)
        return False

    def wait_ready(self, timeout=600):
        """Block until rendering has started and playback is under way."""
        if not self.renderer.ready.wait(timeout):
            return False
        end = time.perf_counter() + 30
        while self.player._played == 0 and time.perf_counter() < end:
            time.sleep(0.05)
        return True

    def run(self, until=None):
        """Start rendering and show the window (main thread). Returns when the window closes."""
        t = time.perf_counter()
        threading.Thread(target=lambda: self.renderer.ready.wait() and self.log(
            f"[avatar] live after {time.perf_counter() - t:.1f}s"), daemon=True).start()
        self.renderer.start()
        try:
            self.player.run(until=until)
        finally:
            self.renderer.stop_event.set()
            self.renderer.join(timeout=10)
            self.voice.close()
            self.face.close()
            if self.recorder:
                self.log(f"[avatar] recording: {self.recorder.close()}")
