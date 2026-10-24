"""Typed chat with the avatar on this machine: type in this console, she answers out loud in a window.

    python scripts\\chat.py [--config config.toml] [--record]

Type a message and press Enter. 'quit' (or closing the window, or Esc) ends the session.
"""
import argparse
import os
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from portravix import config as C  # noqa: E402
from portravix.app import Avatar  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--config", default=None)
parser.add_argument("--record", action="store_true", help="save the session to recordings\\")
args = parser.parse_args()

record = None
if args.record:
    os.makedirs(os.path.join(ROOT, "recordings"), exist_ok=True)
    record = os.path.join(ROOT, "recordings", f"session_{time.strftime('%Y%m%d-%H%M%S')}.mp4")
avatar = Avatar(C.load(args.config), record_path=record)
done = threading.Event()


def console():
    print("\nReady. Type a message and press Enter ('quit' to exit).", flush=True)
    while not done.is_set():
        try:
            text = input("you> ").strip()
        except EOFError:
            break
        if text.lower() in ("quit", "exit"):
            break
        if text:
            turn = avatar.respond(text)
            print(f"avatar> {turn['reply']}", flush=True)
            avatar.wait_until_quiet()
    done.set()


threading.Thread(target=console, daemon=True).start()
avatar.run(until=done.is_set)
