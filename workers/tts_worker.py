"""Pocket TTS worker (Kyutai Pocket TTS, CPU): sentences in, streamed 24 kHz audio out.

    python -X utf8 tts_worker.py --voice <name or file> [--language english | --config <yaml>] [--threads 4]

-X utf8 is required on Windows: pocket_tts opens its YAML config without an encoding.

--voice is one of Pocket TTS's preset voices (eve, mary, jean, ...) or a WAV / voice-state file to clone. Cloning
needs the voice-cloning weights, which are gated: accept the terms at huggingface.co/kyutai/pocket-tts and log in
with `hf auth login`. Without them Pocket TTS loads the preset-voices model, and only presets work.

Requests arrive on stdin as JSON lines {"id": n, "text": "..."}; they are spoken in order.
Replies go to stdout as binary frames:
    b"R" + uint32 sample_rate      once, when the model and voice are loaded
    b"A" + uint32 nbytes + float32  audio as it is generated
    b"E" + uint32 id               end of that request
"""
import argparse
import json
import struct
import sys

out = sys.stdout.buffer
sys.stdout = sys.stderr  # anything a library prints must not land in the audio stream

import torch  # noqa: E402
from pocket_tts import TTSModel  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--voice", required=True)
parser.add_argument("--language", default="english")
parser.add_argument("--config", default="")
parser.add_argument("--threads", type=int, default=4)
args = parser.parse_args()

torch.set_num_threads(args.threads)
model = TTSModel.load_model(config=args.config) if args.config else TTSModel.load_model(language=args.language)
voice = model.get_state_for_audio_prompt(args.voice)
model.generate_audio(voice, "Ready.")  # warm-up: first call pays one-off costs


def send(kind, payload=b""):
    out.write(kind + payload)
    out.flush()


send(b"R", struct.pack("<I", model.sample_rate))
for line in sys.stdin:
    if not line.strip():
        continue
    request = json.loads(line)
    for chunk in model.generate_audio_stream(voice, request["text"]):
        audio = chunk.numpy().astype("<f4").tobytes()
        send(b"A", struct.pack("<I", len(audio)) + audio)
    send(b"E", struct.pack("<I", request["id"]))
