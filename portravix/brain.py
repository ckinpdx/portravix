"""Streaming chat with a local LLM (any OpenAI-compatible server: llama.cpp, llama-swap, ...), yielded one spoken
sentence at a time. Voice turns can also send the audio, so the reply can follow tone; the words come from Parakeet
(portravix/ears.py)."""
import base64
import io
import json
import os
import re
import time
import unicodedata
import urllib.request
import wave

import numpy as np

# The persona (system prompt) is re-read whenever its file changes, so it can be edited while the avatar runs.
PERSONA_SOFT_TOKENS = 2000    # above this a 4B model follows the persona less reliably
PERSONA_HARD_TOKENS = 12000   # above this it may not fit the 16k context with history and reply

# Used only when the persona file is missing or empty.
DEFAULT_PERSONA = (
    "You are talking with the user face to face through a live video avatar, and everything you write is "
    "spoken aloud. Reply the way a person talks: one to three short sentences, warm and natural. "
    "Never use markdown, lists, emoji, stage directions or asterisks."
)

# Voice turns: Parakeet writes down the words (portravix/ears.py), and the reply gets the audio (tone) plus those
# words when the model takes audio. Gemma 4 E4B writing the words down itself got about half of them wrong.
VOICE_PROMPT = '(Voice message. My words: "{}")'

# Moods: when the face has a talking prompt per mood, the model starts each reply with a tag naming one. The tag is
# taken off before speech and picks the face's talking prompt for that reply.
MOOD_INSTRUCTION = (
    "Your face shows a mood while you speak. Begin every reply with exactly one mood tag in square brackets, chosen "
    "from: {tags}. It sets your expression and is never spoken; pick the one that fits how you feel about what you "
    "are saying. Then reply as usual.")

_SENTENCE_END = re.compile(r'([.!?…]+["\')\]]*)\s+')
_MARKUP = re.compile(r"[*_#`~>|\[\]{}]")
_TAG = re.compile(r"\[\s*[A-Za-z][A-Za-z _-]{0,23}\s*\]")
_LEADING_TAG = re.compile(r"\s*\[\s*([A-Za-z][A-Za-z _-]{0,23})\s*\]\s*")


def clean_for_speech(text):
    """Strip what a TTS would read as symbols: markdown, emoji, stage directions, stray mood tags."""
    text = re.sub(r"\*[^*]+\*", " ", text)  # *smiles*
    text = _TAG.sub(" ", text)              # [warm]
    text = _MARKUP.sub(" ", text)
    text = text.replace("&", " and ").replace("%", " percent")
    text = "".join(ch for ch in text if unicodedata.category(ch) not in ("So", "Sk", "Cs", "Co", "Cf"))
    return re.sub(r"\s+", " ", text).strip()


def audio_part(audio, rate=16000):
    """float32 mono samples -> an OpenAI-style input_audio content part (16-bit WAV, base64)."""
    pcm = (np.clip(np.asarray(audio, np.float32), -1, 1) * 32767).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return {"type": "input_audio", "input_audio": {"data": base64.b64encode(buf.getvalue()).decode(), "format": "wav"}}


class Brain:
    def __init__(self, url, model, persona_file, max_turns=20, max_tokens=300, audio=True, moods=(), log=print):
        """audio: send voice turns' audio to the model as well as the words (needs an audio-capable model).
        moods: the moods the face has a talking prompt for; the model tags each reply with one (self.mood)."""
        self.url, self.model, self.max_tokens, self.max_turns, self.audio = url, model, max_tokens, max_turns, audio
        self.persona_file, self.log = persona_file, log
        self.moods = [m.lower() for m in moods]
        self.mood = None  # the latest reply's mood, when it tagged one the face has
        self.system = DEFAULT_PERSONA
        self._persona_mtime = None
        self.history = []  # alternating user/assistant messages, text only (voice turns: what was heard)
        self.timing = {}
        self.load_persona()

    def load_persona(self):
        """(Re)read the persona file if it changed. Missing or empty keeps the previous persona."""
        try:
            mtime = os.path.getmtime(self.persona_file)
        except OSError:
            if self._persona_mtime is None:
                self.log(f"[brain] no persona file at {self.persona_file}; using the built-in default")
                self._persona_mtime = -1
            return
        if mtime == self._persona_mtime:
            return
        self._persona_mtime = mtime
        with open(self.persona_file, encoding="utf-8") as f:
            text = f.read().strip()
        if not text:
            self.log("[brain] persona file is empty; keeping the previous persona")
            return
        self.system = text
        tokens = len(text) // 4  # rough estimate for English
        note = ""
        if tokens > PERSONA_HARD_TOKENS:
            note = f" WARNING: over ~{PERSONA_HARD_TOKENS} tokens, may not fit the 16k context"
        elif tokens > PERSONA_SOFT_TOKENS:
            note = f" (above ~{PERSONA_SOFT_TOKENS} tokens a 4B model follows it less reliably)"
        self.log(f"[brain] persona loaded: {len(text)} chars, ~{tokens} tokens{note}")

    def _post(self, body, timeout=180):
        body = {"model": self.model, "chat_template_kwargs": {"enable_thinking": False}, **body}
        request = urllib.request.Request(self.url, json.dumps(body).encode(), {"Content-Type": "application/json"})
        return urllib.request.urlopen(request, timeout=timeout)

    def warm_up(self):
        """Make the server load the model now, and run audio through it once (with Gemma 4 E4B the first audio after a
        load costs ~4 s extra), so the first real reply waits for neither."""
        self._post({"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 1}).read()
        if not self.audio:
            return
        noise = np.random.default_rng(0).standard_normal(16000).astype(np.float32) * 0.001
        self._post({"messages": [{"role": "user", "content": [audio_part(noise), {"type": "text", "text": "Hi"}]}],
                    "max_tokens": 1}).read()

    def _messages(self, user_content):
        system = self.system
        if self.moods:
            system += "\n\n" + MOOD_INSTRUCTION.format(tags=", ".join(f"[{m}]" for m in self.moods))
        return [{"role": "system", "content": system}, *self.history[-2 * self.max_turns:],
                {"role": "user", "content": user_content}]

    def _take_mood(self, text):
        """Take a leading [mood] tag off the start of a reply: (rest, done). done is False while the start could
        still be a tag that has not arrived whole."""
        stripped = text.lstrip()
        if not stripped or (stripped.startswith("[") and "]" not in stripped and len(stripped) < 32):
            return text, False
        match = _LEADING_TAG.match(text)
        if not match:
            return text, True
        name = match.group(1).strip().lower().replace(" ", "_").replace("-", "_")
        self.mood = name if name in self.moods else None
        return text[match.end():], True

    def reply(self, user_text, images=(), audio=None):
        """Stream the reply; yields cleaned sentences as soon as each one is complete.
        images: data URLs (e.g. webcam JPEGs) attached to this turn only.
        audio: the voice message user_text was heard from (16 kHz float32), attached to this turn only."""
        self.load_persona()
        if audio is not None and self.audio:
            content = [audio_part(audio), *({"type": "image_url", "image_url": {"url": url}} for url in images),
                       {"type": "text", "text": VOICE_PROMPT.format(user_text)}]
        elif images:
            content = [*({"type": "image_url", "image_url": {"url": url}} for url in images),
                       {"type": "text", "text": user_text}]
        else:
            content = user_text
        body = {"model": self.model, "messages": self._messages(content), "stream": True,
                "max_tokens": self.max_tokens, "chat_template_kwargs": {"enable_thinking": False}}
        request = urllib.request.Request(self.url, json.dumps(body).encode(), {"Content-Type": "application/json"})
        t0 = time.perf_counter()
        self.timing = {"start": t0}
        self.mood = None
        full, pending = [], ""
        head = bool(self.moods)  # still looking for the mood tag at the start
        with urllib.request.urlopen(request, timeout=120) as response:
            for raw in response:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                delta = json.loads(line[5:])["choices"][0]["delta"].get("content") or ""
                if not delta:
                    continue
                self.timing.setdefault("first_token", time.perf_counter())
                full.append(delta)
                pending += delta
                if head:
                    pending, done = self._take_mood(pending)
                    if not done:
                        continue
                    head = False
                while (match := _SENTENCE_END.search(pending)):
                    sentence, pending = clean_for_speech(pending[:match.end(1)]), pending[match.end():]
                    if sentence:
                        self.timing.setdefault("first_sentence", time.perf_counter())
                        yield sentence
        if head:
            pending, _ = self._take_mood(pending + " ")
        tail = clean_for_speech(pending)
        if tail:
            self.timing.setdefault("first_sentence", time.perf_counter())
            yield tail
        self.timing["done"] = time.perf_counter()
        self.history += [{"role": "user", "content": user_text}, {"role": "assistant", "content": "".join(full)}]
