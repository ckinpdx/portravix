"""Memory end to end with the real LLM, on a TEMPORARY memory file (the configured one is never touched): a short
conversation that mentions a few facts, a curator pass, then a fresh brain (the next session) asked about them.
No face or voice: the brain only.

    python scripts\\test_memory.py [--config config.toml]
"""
import argparse
import os
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from portravix import config as C  # noqa: E402
from portravix.brain import Brain  # noqa: E402
from portravix.memory import Memory  # noqa: E402

FIRST_SESSION = [
    "I finally named the new dog. She's called Biscuit.",
    "Also, I started learning the cello last week. It's going terribly.",
    "What's your favourite season, and why?",
]
NEXT_SESSION = [
    "Hey, it's me again. What's my dog called?",
    "What's your favourite season again?",
]

parser = argparse.ArgumentParser()
parser.add_argument("--config", default=None)
args = parser.parse_args()
cfg = C.load(args.config)
b, m = cfg["brain"], cfg["memory"]
user = m["user_name"] or "the user"

with tempfile.TemporaryDirectory() as tmp:
    def memory():
        return Memory(os.path.join(tmp, "memory.md"), url=b["url"], model=b["model"], user=user,
                      budget_tokens=m["budget_tokens"], summaries=m["summaries"], keep_summaries=m["keep_summaries"],
                      gap_minutes=m["quiet_minutes"], state_dir=tmp)

    def brain(mem):
        return Brain(b["url"], b["model"], C.resolve(b["persona"]), max_turns=int(b["history_turns"]),
                     max_tokens=int(b["max_tokens"]), audio=False, memory=mem, user=user)

    def talk(br, mem, prompts):
        for prompt in prompts:
            reply = " ".join(br.reply(prompt))
            mem.record(prompt, reply)
            print(f"> {prompt}\n< {reply}")

    print("--- first session")
    mem = memory()
    talk(brain(mem), mem, FIRST_SESSION)
    t = time.perf_counter()
    mem.curate()
    print(f"\n--- curated in {time.perf_counter() - t:.1f}s; the memory file:\n")
    with open(mem.path, encoding="utf-8") as f:
        print(f.read())

    print("--- next session (a fresh brain; its first message carries this state block):")
    mem2 = memory()
    mem2.previous_turn -= 2 * 86400  # as if the first session were two days ago
    br2 = brain(mem2)
    print(br2.state_block())
    talk(br2, mem2, NEXT_SESSION)
    first = br2.history[1]["content"].lower()
    print("\nremembered the dog:", "yes" if "biscuit" in first else "NO")
