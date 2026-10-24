"""Always-on launcher behind the phone page: serves the page, starts and stops the avatar process on request, and
relays the page's WebSocket to it while it runs. Uses no GPU and loads no models.

The avatar itself runs as `scripts/serve.py --host 127.0.0.1 --port <avatar_port>`, reachable only through here.

Before Start it makes room on the GPU the way a person would: a running ComfyUI is asked to unload its models
([comfyui] free_before_start), and, if [gpu] llama_swap points at a llama-swap server, its idle models are unloaded
too. It refuses to evict anything that looks mid-task, because an evicted model that is still working loads
straight back beside the avatar and slows both.
"""
import asyncio
import collections
import contextlib
import datetime
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request

import psutil
import uvicorn
import websockets
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse

from . import config as C

STATIC = os.path.join(C.ROOT, "static")
LOGS = os.path.join(C.ROOT, "logs")
PYTHON = os.path.join(os.path.dirname(sys.executable), "python.exe" if os.name == "nt" else "python")  # not pythonw
BUSY_WINDOW = 180    # s since another model last finished a request: treat it as mid-task (tool calls, image jobs)
FULL_VRAM_MB = 1500  # free VRAM below this with another model loaded = Windows is spilling into system RAM
RELAY_BACKLOG = 2    # media chunks held for a slow phone before the oldest is dropped
NO_WINDOW = 0x08000000 if os.name == "nt" else 0


def _post(url, body=None, timeout=10):
    data = json.dumps(body).encode() if body is not None else b""
    req = urllib.request.Request(url, data=data, method="POST", headers={"Content-Type": "application/json"})
    with contextlib.suppress(Exception):
        urllib.request.urlopen(req, timeout=timeout).read()


def _get(url, timeout=5):
    return json.load(urllib.request.urlopen(url, timeout=timeout))


def free_vram_mb():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, creationflags=NO_WINDOW).stdout
    return int(out.split()[0])


class Gpu:
    """What else is on the GPU, as far as llama-swap and ComfyUI can say."""

    def __init__(self, cfg):
        self.llama_swap = (cfg["gpu"]["llama_swap"] or "").rstrip("/")
        self.comfy = (cfg["comfyui"]["url"] or "").rstrip("/") if cfg["comfyui"]["free_before_start"] else ""
        self.own_model = cfg["brain"]["model"]

    def users(self):
        """Other llama-swap models loaded now, plus "ComfyUI" if it is running a job."""
        users = []
        if self.llama_swap:
            with contextlib.suppress(Exception):
                users += [r["model"] for r in _get(f"{self.llama_swap}/running")["running"] if r["model"] != self.own_model]
        if self.comfy:
            with contextlib.suppress(Exception):
                if _get(f"{self.comfy}/queue")["queue_running"]:
                    users.append("ComfyUI")
        return users

    def busy_reason(self):
        """Why evicting the GPU's other users now would collide with work in progress, or ""."""
        for user in self.users():
            if user == "ComfyUI":
                return "ComfyUI is generating"
            with contextlib.suppress(Exception):
                if any(slot.get("is_processing") for slot in _get(f"{self.llama_swap}/upstream/{user}/slots")):
                    return f"{user} is in the middle of a request"
            with contextlib.suppress(Exception):
                for row in _get(f"{self.llama_swap}/api/metrics/activity?limit=50")["data"]:  # newest first
                    if row["model"] == user:
                        ago = time.time() - datetime.datetime.fromisoformat(row["timestamp"]).timestamp()
                        if ago < BUSY_WINDOW:
                            return f"{user} finished a request {ago:.0f} s ago and is probably mid-task"
                        break
        return ""

    def make_room(self):
        if self.llama_swap:
            _post(f"{self.llama_swap}/api/models/unload")
        if self.comfy:
            _post(f"{self.comfy}/free", {"unload_models": True, "free_memory": True})

    def release_brain(self):
        if self.llama_swap:
            _post(f"{self.llama_swap}/api/models/unload/{self.own_model}")  # give its VRAM back now, not after a ttl


class AvatarProcess:
    """One avatar child process at most. State: stopped | busy | starting | live | stopping | error.
    busy: a Start was refused because another model or ComfyUI is working; nothing was touched."""

    def __init__(self, cfg, log):
        self.cfg, self.log = cfg, log
        self.gpu = Gpu(cfg)
        self.avatar_port = int(cfg["server"]["avatar_port"])
        self.needed_mb = int(float(cfg["gpu"]["needed_vram_gb"]) * 1024)
        self.state, self.detail = "stopped", ""
        self.since = time.time()
        self.proc = None  # psutil.Process
        self.log_path = None
        self._lock = threading.Lock()
        self._adopt_running()
        threading.Thread(target=self._monitor, daemon=True).start()

    def _set(self, state, detail=""):
        self.state, self.detail, self.since = state, detail, time.time()
        self.log(f"[launcher] {state}{': ' + detail if detail else ''}")

    def _adopt_running(self):
        """If an avatar is already listening on its port (e.g. the launcher restarted), take it over."""
        try:
            conns = psutil.net_connections(kind="tcp")
        except psutil.Error:
            return
        for conn in conns:
            if conn.laddr and conn.laddr.port == self.avatar_port and conn.status == psutil.CONN_LISTEN and conn.pid:
                self.proc = psutil.Process(conn.pid)
                self._set("live", "adopted a running avatar")
                return

    def status(self):
        return {"state": self.state, "detail": self.detail, "since": self.since}

    def start(self):
        with self._lock:
            if self.state in ("starting", "live", "stopping"):
                return
            free = free_vram_mb()
            if free < self.needed_mb:
                reason = self.gpu.busy_reason()
                if reason:
                    self._set("busy", f"Not starting: {reason}. Pushing it off the GPU now would only make it load "
                                      f"straight back next to the avatar and slow both. Try again once it has been "
                                      f"idle for {BUSY_WINDOW // 60} minutes.")
                    return
                self.log(f"[launcher] {free} MB VRAM free; asking idle models to unload")
                self.gpu.make_room()
                time.sleep(3)
                free = free_vram_mb()
                if free < self.needed_mb:
                    self._set("error", f"Not enough GPU memory: {free / 1024:.1f} GB free, about "
                                       f"{self.needed_mb / 1024:.0f} GB needed. Something else is using the GPU.")
                    return
            os.makedirs(LOGS, exist_ok=True)
            self.log_path = os.path.join(LOGS, f"avatar_{datetime.datetime.now():%Y%m%d-%H%M%S}.log")
            out = open(self.log_path, "wb")
            popen = subprocess.Popen([PYTHON, os.path.join(C.ROOT, "scripts", "serve.py"), "--host", "127.0.0.1",
                                      "--port", str(self.avatar_port), "--config", self.cfg["_path"]], cwd=C.ROOT,
                                     stdout=out, stderr=subprocess.STDOUT, creationflags=NO_WINDOW)
            self.proc = psutil.Process(popen.pid)
            self._set("starting", "loading the face, then the voice model (about 40 s)")

    def stop(self):
        with self._lock:
            if self.proc is None:
                self._set("stopped")
                return
            self._set("stopping")
            procs = [self.proc] + self.proc.children(recursive=True)  # serve.py + its face and TTS workers
            for p in procs:
                with contextlib.suppress(psutil.Error):
                    p.terminate()
            _, alive = psutil.wait_procs(procs, timeout=8)
            for p in alive:
                with contextlib.suppress(psutil.Error):
                    p.kill()
            self.proc = None
            self.gpu.release_brain()
            self._set("stopped")

    def _last_log_lines(self, n=4):
        if not self.log_path or not os.path.exists(self.log_path):
            return ""
        with open(self.log_path, encoding="utf-8", errors="replace") as f:
            lines = [re.sub(r"\x1b\[[0-9;]*m", "", ln).strip() for ln in f if ln.strip()]
        return " | ".join(lines[-n:])

    def _check_crowding(self):
        """While live: warn on the page if another model (or ComfyUI) has loaded next to the avatar and filled the GPU."""
        users = self.gpu.users()
        crowded = bool(users) and free_vram_mb() < FULL_VRAM_MB
        detail = (f"{', '.join(users)} also loaded and the GPU is full: she will stutter until it unloads when "
                  f"idle, or you press Stop" if crowded else "")
        with self._lock:
            if self.state == "live" and detail != self.detail:
                self.detail = detail  # the state and its 'since' stay as they are
                self.log(f"[launcher] {detail or 'GPU no longer crowded'}")

    def _monitor(self):
        last_crowding_check = 0
        while True:
            time.sleep(1)
            with self._lock:
                proc = self.proc
                if proc is not None and (not proc.is_running() or proc.status() == psutil.STATUS_ZOMBIE):
                    self.proc = None
                    self._set("error", "the avatar process exited: " + self._last_log_lines())
                    continue
                if self.state == "starting":
                    with contextlib.suppress(Exception):
                        s = json.load(urllib.request.urlopen(f"http://127.0.0.1:{self.avatar_port}/status", timeout=2))
                        if s.get("state") == "live":
                            self._set("live")
                live = self.state == "live"
            if live and time.time() - last_crowding_check > 10:
                last_crowding_check = time.time()
                with contextlib.suppress(Exception):
                    self._check_crowding()


def create_launcher_app(avatar):
    app = FastAPI()

    @app.get("/")
    async def index():
        return FileResponse(os.path.join(STATIC, "index.html"), headers={"Cache-Control": "no-store"})

    @app.get("/api/status")
    async def status():
        return avatar.status()

    @app.post("/api/start")
    async def start():
        await asyncio.to_thread(avatar.start)
        return avatar.status()

    @app.post("/api/stop")
    async def stop():
        await asyncio.to_thread(avatar.stop)
        return avatar.status()

    @app.websocket("/ws")
    async def relay(ws: WebSocket):
        await ws.accept()
        if avatar.state != "live":
            await ws.close(code=1013)  # try again later
            return
        try:
            async with websockets.connect(f"ws://127.0.0.1:{avatar.avatar_port}/ws", max_size=None) as upstream:
                media, texts = collections.deque(), collections.deque()
                wake = asyncio.Event()
                dropped = 0

                async def from_avatar():
                    # Read the avatar as fast as it sends, so a slow phone backs up here and drops old chunks,
                    # instead of backing up in socket buffers where the backlog only turns into delay.
                    nonlocal dropped
                    async for msg in upstream:
                        if isinstance(msg, bytes):
                            if len(media) >= RELAY_BACKLOG:
                                media.popleft()
                                dropped += 1
                                avatar.log(f"[relay] phone connection too slow: dropped a chunk ({dropped} this session)")
                            media.append(msg)
                        else:
                            texts.append(msg)  # chat text is small and never dropped
                        wake.set()

                async def to_phone():
                    while True:
                        await wake.wait()
                        wake.clear()
                        while texts or media:
                            if texts:
                                await ws.send_text(texts.popleft())
                            else:
                                await ws.send_bytes(media.popleft())

                async def upstream_pump():
                    while True:
                        msg = await ws.receive()
                        if msg["type"] == "websocket.disconnect":
                            return
                        if msg.get("text") is not None:
                            await upstream.send(msg["text"])
                        elif msg.get("bytes") is not None:  # mic audio
                            await upstream.send(msg["bytes"])

                tasks = [asyncio.create_task(from_avatar()), asyncio.create_task(to_phone()),
                         asyncio.create_task(upstream_pump())]
                _, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for t in pending:
                    t.cancel()
        except Exception:
            pass
        finally:
            with contextlib.suppress(Exception):
                await ws.close()

    return app


def run(cfg, host, port, log=print, tls=None, renew=None):
    """tls: (certfile, keyfile) to serve HTTPS. renew: refreshes those files; called daily, and the server picks up a
    changed certificate for new connections without restarting."""
    avatar = AvatarProcess(cfg, log)
    config = uvicorn.Config(create_launcher_app(avatar), host=host, port=port, log_level="warning",
                            ssl_certfile=tls[0] if tls else None, ssl_keyfile=tls[1] if tls else None)
    if tls and renew:
        def renew_daily():
            while True:
                time.sleep(24 * 3600)
                with open(tls[0], "rb") as f:
                    before = f.read()
                renew()
                with open(tls[0], "rb") as f:
                    changed = f.read() != before
                if changed and config.ssl is not None:
                    config.ssl.load_cert_chain(*tls)
                    log("[launcher] HTTPS certificate renewed")
        threading.Thread(target=renew_daily, daemon=True).start()
    uvicorn.Server(config).run()
