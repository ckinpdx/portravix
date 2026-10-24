"""Serve the avatar to a phone (or any browser): chat box, mic, and the live video and voice.

    python scripts\\serve.py [--config config.toml] [--local] [--record] [--host ...] [--port ...]

Normally the launcher (scripts\\launcher.py) starts this on 127.0.0.1 and relays the page to it. Run on its own, it
binds to this machine's Tailscale address (the page has no login), so it is reachable from your tailnet and nowhere
else; open http://<this machine>:<port> on the phone and tap the picture. Plain HTTP means JPEG video; H.264 and the
mic need the launcher's HTTPS. --local also shows the window and plays the audio on this machine. Ctrl+C stops.
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(line_buffering=True)  # under the launcher stdout is a log file; keep it current
from portravix import config as C  # noqa: E402
from portravix.app import Avatar  # noqa: E402
from portravix.web import ChunkStreamer, create_app, start_server, tailscale_ip  # noqa: E402


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}")


parser = argparse.ArgumentParser()
parser.add_argument("--config", default=None)
parser.add_argument("--port", type=int, default=None, help="default: [server] port")
parser.add_argument("--host", default=None, help="default: [server] host, else this machine's Tailscale IPv4 address")
parser.add_argument("--local", action="store_true", help="also show the window and play audio on this machine")
parser.add_argument("--record", action="store_true", help="save the session to recordings\\")
args = parser.parse_args()
cfg = C.load(args.config)

host = args.host or cfg["server"]["host"] or tailscale_ip()
if not host:
    sys.exit("No Tailscale address found (is Tailscale running and signed in?). Pass --host to override.")
port = args.port or int(cfg["server"]["port"])

record = None
if args.record:
    os.makedirs(os.path.join(ROOT, "recordings"), exist_ok=True)
    record = os.path.join(ROOT, "recordings", f"phone_{time.strftime('%Y%m%d-%H%M%S')}.mp4")

avatar = Avatar(cfg, record_path=record, headless=not args.local, log=log)
streamer = ChunkStreamer(fps=avatar.face.fps, log=log)
avatar.player.listeners.append(streamer)
start_server(create_app(avatar, streamer, log=log), host, port)
print(f"[serve] page: http://{host}:{port}; live once the first chunk renders", flush=True)
try:
    avatar.run()
except KeyboardInterrupt:
    pass
