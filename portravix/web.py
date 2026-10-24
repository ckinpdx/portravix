"""Phone page: serves static/index.html and a WebSocket that streams the avatar's video and voice, chunk by
chunk, and takes typed and spoken messages. Bind it to the Tailscale address only: the page has no login."""
import asyncio
import contextlib
import itertools
import json
import os
import queue
import shutil
import struct
import subprocess
import threading
import time
from fractions import Fraction

import av
import cv2
import numpy as np
import soxr
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from .ears import RATE, UtteranceDetector
from .speech_buffer import PLAYBACK_RATE

STATIC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")
STREAM_RATE = 24000
TAILSCALE = shutil.which("tailscale") or r"C:\Program Files\Tailscale\tailscale.exe"
NO_WINDOW = 0x08000000 if os.name == "nt" else 0  # CREATE_NO_WINDOW: no console flash when run from pythonw


def tailscale_ip():
    try:
        return subprocess.run([TAILSCALE, "ip", "-4"], capture_output=True, text=True, timeout=5,
                              creationflags=NO_WINDOW).stdout.split()[0]
    except Exception:
        return None


def tailscale_cert(folder):
    """Fetch or renew this machine's Tailscale HTTPS certificate into `folder`.
    Returns ((domain, certfile, keyfile), "") or (None, reason)."""
    try:
        status = json.loads(subprocess.run([TAILSCALE, "status", "--json"], capture_output=True, text=True,
                                           timeout=10, creationflags=NO_WINDOW).stdout)
        domain = status["Self"]["DNSName"].rstrip(".")
        if not status.get("CertDomains"):
            return None, "HTTPS certificates are off for this tailnet (Tailscale admin console > DNS > HTTPS Certificates)"
        os.makedirs(folder, exist_ok=True)
        cert, key = os.path.join(folder, domain + ".crt"), os.path.join(folder, domain + ".key")
        r = subprocess.run([TAILSCALE, "cert", "--cert-file", cert, "--key-file", key, domain],
                           capture_output=True, text=True, timeout=120, creationflags=NO_WINDOW)
        if r.returncode != 0:
            return None, ((r.stderr or r.stdout).strip().splitlines() or ["tailscale cert failed"])[-1]
        return (domain, cert, key), ""
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def avc_codec(access_unit):
    """RFC 6381 codec string (avc1.PPCCLL) read from the SPS of an H.264 access unit."""
    i = access_unit.find(b"\x00\x00\x01\x67")
    return "avc1." + access_unit[i + 4:i + 7].hex() if i >= 0 else "avc1.42c01e"


class Client:
    _ids = itertools.count(1)

    def __init__(self):
        self.id = next(self._ids)
        self.queue = asyncio.Queue()
        self.codec = None  # set by the page's hello; nothing is sent before it
        self.sent = self.dropped = 0
        self.since = time.time()


class ChunkStreamer:
    """Player listener: encodes each rendered chunk once per video format in use and fans it out to
    every connected page.

    Wire format, one binary WebSocket message per chunk:
        uint32 header length | JSON header | int16 PCM at 24 kHz | video frames back to back
    The header has seq (gaps = dropped chunks), codec ("h264" or "jpeg"), the byte size of the audio
    and of each frame, and the server's send time in ms. H.264 frames are Annex B access units; each
    chunk starts with an IDR frame carrying SPS/PPS, so every chunk decodes on its own.
    """

    def __init__(self, jpeg_quality=65, h264_crf=23, max_backlog=4, fps=24, log=print):
        self.quality, self.crf, self.max_backlog, self.log = jpeg_quality, h264_crf, max_backlog, log
        self.fps = fps  # the face's frame rate; the page reads it from each chunk's header
        self.loop = None  # the server's event loop, set at startup
        self._in = queue.Queue(maxsize=8)
        self._clients = []
        self._lock = threading.Lock()
        self._resampler = soxr.ResampleStream(PLAYBACK_RATE, STREAM_RATE, 1, dtype="float32")
        self._seq = 0
        self._h264 = None  # persistent encoder, created on first use
        self._pts = 0
        threading.Thread(target=self._encode_loop, daemon=True).start()

    def __call__(self, frames, audio, speech):
        # Runs on the renderer thread: hand off and return, never block rendering.
        with contextlib.suppress(queue.Full):
            self._in.put_nowait((frames, audio, speech))

    def add_client(self):
        client = Client()
        with self._lock:
            self._clients.append(client)
        return client

    def remove_client(self, client):
        with self._lock:
            self._clients.remove(client)

    def _encode_jpeg(self, frames):
        return [cv2.imencode(".jpg", cv2.cvtColor(f, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, self.quality])[1].tobytes()
                for f in frames]

    def _encode_h264(self, frames):
        if self._h264 is None:
            ctx = av.CodecContext.create("libx264", "w")
            ctx.height, ctx.width = frames[0].shape[:2]
            ctx.pix_fmt = "yuv420p"
            ctx.time_base, ctx.framerate = Fraction(1, self.fps), Fraction(self.fps)
            ctx.gop_size, ctx.max_b_frames = 250, 0  # keyframes come from the forced IDR at each chunk start
            ctx.options = {"preset": "veryfast", "tune": "zerolatency", "profile": "baseline",
                           "crf": str(self.crf), "forced-idr": "1"}
            self._h264 = ctx
        out = []
        for i, f in enumerate(frames):
            vf = av.VideoFrame.from_ndarray(f, format="rgb24").reformat(format="yuv420p")
            vf.pts = self._pts
            self._pts += 1
            if i == 0:
                vf.pict_type = av.video.frame.PictureType.I
            out.append(b"".join(bytes(p) for p in self._h264.encode(vf)))  # zerolatency: one packet per frame, now
        return out

    def _encode_loop(self):
        while True:
            frames, audio, speech = self._in.get()
            with self._lock:
                clients = [c for c in self._clients if c.codec]
            seq = self._seq
            self._seq += 1
            if not clients or self.loop is None:
                continue
            pcm = self._resampler.resample_chunk(np.asarray(audio, np.float32))
            pcm = (np.clip(pcm, -1, 1) * 32767).astype("<i2").tobytes()
            payloads = {}
            for codec in {c.codec for c in clients}:
                video = self._encode_h264(frames) if codec == "h264" else self._encode_jpeg(frames)
                header = {"seq": seq, "codec": codec, "fps": self.fps, "rate": STREAM_RATE, "speech": bool(speech),
                          "audio": len(pcm), "frames": [len(v) for v in video], "sent": round(time.time() * 1000)}
                if codec == "h264":
                    header["avc"] = avc_codec(video[0])
                header = json.dumps(header).encode()
                payloads[codec] = struct.pack("<I", len(header)) + header + pcm + b"".join(video)
            for c in clients:
                self.loop.call_soon_threadsafe(self._offer, c, payloads[c.codec])

    def _offer(self, client, payload):
        while client.queue.qsize() >= self.max_backlog:  # a slow connection: drop the oldest, the page resyncs
            client.queue.get_nowait()
            client.dropped += 1
            self.log(f"[web] client {client.id}: dropped a chunk ({self.max_backlog} already queued for it)")
        client.queue.put_nowait(payload)


def create_app(avatar, streamer, log=print):
    app = FastAPI()
    started = time.time()

    @app.on_event("startup")
    async def _startup():
        streamer.loop = asyncio.get_running_loop()

    @app.get("/")
    async def index():
        return FileResponse(os.path.join(STATIC, "index.html"), headers={"Cache-Control": "no-store"})

    @app.get("/status")
    async def status():
        live = avatar.renderer.ready.is_set() and avatar.player._played > 0
        return {"state": "live" if live else "compiling", "turns": len(avatar.turns)}

    @app.get("/api/status")
    async def page_status():  # the launcher's API, for the page when serve.py runs on its own
        live = avatar.renderer.ready.is_set() and avatar.player._played > 0
        return {"state": "live" if live else "starting", "detail": "", "since": started, "managed": False}

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        loop = asyncio.get_running_loop()
        send_lock = asyncio.Lock()
        client = streamer.add_client()

        async def send_json(obj):
            async with send_lock:
                await ws.send_json(obj)

        async def pump():
            while True:
                payload = await client.queue.get()
                async with send_lock:
                    await ws.send_bytes(payload)
                client.sent += 1

        ears = None    # this page's utterance detector, made when its mic first sends audio
        busy = False   # a voice turn is being heard and answered: mic audio is ignored meanwhile

        async def chat(text=None, audio=None):
            nonlocal busy

            def on_sentence(sentence):
                asyncio.run_coroutine_threadsafe(send_json({"type": "sentence", "text": sentence}), loop)

            async def heard_messages(heard):
                await send_json({"type": "heard", "text": heard})
                if heard:
                    await send_json({"type": "thinking"})

            def on_heard(heard):
                asyncio.run_coroutine_threadsafe(heard_messages(heard), loop)

            try:
                if audio is None:
                    await send_json({"type": "thinking"})
                turn = await asyncio.to_thread(avatar.respond, text, (), on_sentence, audio, on_heard)
                await send_json({"type": "reply", "text": turn["reply"]})
            except Exception as e:
                log(f"[web] client {client.id}: turn failed: {type(e).__name__}: {e}")
                with contextlib.suppress(Exception):
                    await send_json({"type": "reply", "text": ""})
            finally:
                if audio is not None:
                    busy = False

        def start_chat(**kwargs):
            task = asyncio.create_task(chat(**kwargs))  # keep reading the page (mic, events) meanwhile
            chats.add(task)
            task.add_done_callback(chats.discard)

        pump_task = asyncio.create_task(pump())
        chats = set()
        try:
            while True:
                message = await ws.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if message.get("bytes") is not None:  # mic audio: int16 mono 16 kHz
                    if busy:
                        continue
                    ears = ears or UtteranceDetector()
                    pcm = np.frombuffer(message["bytes"], "<i2").astype(np.float32) / 32768
                    for kind, audio in ears.feed(pcm):
                        if kind == "start":
                            await send_json({"type": "hearing"})
                        elif kind == "discard":  # too short to be words (a knock, a breath)
                            await send_json({"type": "listening"})
                        elif kind == "end":
                            busy = True
                            await send_json({"type": "heard_audio", "seconds": round(len(audio) / RATE, 1)})
                            start_chat(audio=audio)
                            break
                    continue
                msg = json.loads(message.get("text") or "{}")
                kind = msg.get("type")
                if kind == "hello":  # the page says which video it can decode
                    client.codec = "h264" if msg.get("video") == "h264" else "jpeg"
                    log(f"[web] client {client.id} connected: {client.codec}, buffer {msg.get('buffer')} s")
                elif kind == "mic":  # the page opened, reopened (after her reply) or closed its mic
                    if ears:
                        ears.reset()
                    if msg.get("toggle"):
                        log(f"[web] client {client.id}: mic {'on' if msg.get('on') else 'off'}")
                elif kind in ("event", "stats"):  # the page's own view of the stream: refills, gaps, bitrate
                    log(f"[web] client {client.id} {kind}: " + json.dumps({k: v for k, v in msg.items() if k != "type"}))
                elif kind == "chat" and (msg.get("text") or "").strip():
                    start_chat(text=msg["text"].strip())
        except WebSocketDisconnect:
            pass
        finally:
            pump_task.cancel()
            streamer.remove_client(client)
            log(f"[web] client {client.id} disconnected after {(time.time() - client.since) / 60:.1f} min: "
                f"{client.sent} chunks sent, {client.dropped} dropped here")

    return app


def start_server(app, host, port):
    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="warning", ws_max_size=1 << 20))
    threading.Thread(target=server.run, daemon=True).start()
    return server
