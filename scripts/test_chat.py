"""End-to-end conversation test: typed prompts in, spoken replies out, with per-stage latency and render health.

    python scripts\\test_chat.py [--config config.toml] [--headless] [--no-record]

--headless: no window or local audio (a virtual clock plays the audio); the recording is the output.
--no-record: skip the session recording (its video encoding uses CPU and adds ~0.1 s per chunk).
"""
import argparse
import os
import subprocess
import sys
import threading
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from portravix import config as C  # noqa: E402
from portravix.app import Avatar  # noqa: E402

PROMPTS = [
    "Hi! It's nice to see you. How are you doing today?",
    "What's a good way to wind down after a long day?",
    "Tell me something surprising you learned recently.",
]

parser = argparse.ArgumentParser()
parser.add_argument("--config", default=None)
parser.add_argument("--headless", action="store_true")
parser.add_argument("--no-record", action="store_true")
args = parser.parse_args()

os.makedirs(os.path.join(ROOT, "recordings"), exist_ok=True)
avatar = Avatar(C.load(args.config), headless=args.headless,
                record_path=None if args.no_record else
                os.path.join(ROOT, "recordings", f"chat_{time.strftime('%Y%m%d-%H%M%S')}.mp4"))
done = threading.Event()
vram = []


def conversation():
    avatar.wait_ready()
    time.sleep(4)  # idle first
    for prompt in PROMPTS:
        turn = avatar.respond(prompt)
        turn["mouth_start"] = avatar.mouth_start(turn)
        vram.append(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True).stdout.strip())
        avatar.wait_until_quiet()
        time.sleep(2)
    time.sleep(2)
    done.set()


threading.Thread(target=conversation, daemon=True).start()
avatar.run(until=done.is_set)

for turn in avatar.turns:
    mouth = f"{turn['mouth_start']:.2f} s" if turn.get("mouth_start") is not None else "n/a"
    print(f"\n> {turn['user']}\n< {turn['reply']}")
    print(f"  LLM first sentence {turn['llm_first_sentence'] * 1000:.0f} ms | TTS first audio "
          f"{turn['tts_first_audio'] * 1000:.0f} ms | mouth starts {mouth} after the message")
renders = np.array([s[1] for s in avatar.renderer.stats])
budget = avatar.face.chunk_seconds
print(f"\nrender median {np.median(renders) * 1000:.0f} ms, max {renders.max() * 1000:.0f} ms over {len(renders)} chunks "
      f"of {budget:.3f} s; underruns {avatar.player.underruns}; GPU memory during turns (MiB): {', '.join(vram)}")
slow = [(i, round(s[1], 2), round(s[2], 2)) for i, s in enumerate(avatar.renderer.stats) if s[1] > 0.9 * budget]
print("slow chunks, over 90% of the budget (index, seconds, speech seconds):", slow)
if avatar.face.timings:
    w = np.array([t["total_s"] for t in avatar.face.timings[1:]] or [avatar.face.timings[0]["total_s"]])
    print(f"face worker's own time per chunk: median {np.median(w) * 1000:.0f} ms, max {w.max() * 1000:.0f} ms; the "
          f"app side adds {(np.median(renders) - np.median(w)) * 1000:.0f} ms")
print(f"lead before each render: min {min(s[3] for s in avatar.renderer.stats):.2f}s")
for t, missing in avatar.player.underrun_log:
    prior = [i for i, turn in enumerate(avatar.turns) if turn['t0'] <= t]
    where = f"turn {prior[-1] + 1} +{t - avatar.turns[prior[-1]]['t0']:.2f}s" if prior else "before first turn"
    print(f"  underrun {missing} samples at {where}")
