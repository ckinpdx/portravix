"""Always-on launcher for the phone page: start and stop the avatar from the phone.

    pythonw scripts\\launcher.py [--config config.toml]

Serves the page on this machine's Tailscale address ([server] port): over HTTPS (https://<machine>.<tailnet>.ts.net:port)
when the tailnet has HTTPS certificates enabled, otherwise plain http://<machine>:port. Waits for Tailscale at startup,
so it can run from a logon task. Logs to logs\\launcher.log.
"""
import argparse
import datetime
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
_log_file = open(os.path.join(ROOT, "logs", "launcher.log"), "a", encoding="utf-8", buffering=1)
if sys.stdout is None:  # pythonw has no console
    sys.stdout = sys.stderr = _log_file


def log(msg):
    _log_file.write(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} {msg}\n")


from portravix import config as C  # noqa: E402
from portravix.launcher import run  # noqa: E402
from portravix.web import tailscale_cert, tailscale_ip  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--config", default=None)
args = parser.parse_args()
cfg = C.load(args.config)
PORT = int(cfg["server"]["port"])
CERTS = os.path.join(ROOT, "certs")

host = cfg["server"]["host"]
for _ in range(300):  # Tailscale may come up after logon
    host = host or tailscale_ip()
    if host:
        break
    time.sleep(2)
if not host:
    log("[launcher] no Tailscale address after 10 minutes; giving up")
    sys.exit(1)

tls, why = tailscale_cert(CERTS)  # HTTPS lets the page use H.264 (WebCodecs) and the phone's mic
if tls:
    domain, cert, key = tls
    log(f"[launcher] serving https://{domain}:{PORT} on {host}")
    run(cfg, host, PORT, log=log, tls=(cert, key), renew=lambda: tailscale_cert(CERTS))
else:
    log(f"[launcher] serving http://{host}:{PORT} (no HTTPS: {why})")
    run(cfg, host, PORT, log=log)
