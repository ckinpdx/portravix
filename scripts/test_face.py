"""The face on its own: stream a speech WAV through the face worker chunk by chunk, as the app does, time every chunk
against real time, and write the result as an MP4 with the audio.

    python scripts\\test_face.py <speech.wav> [--config config.toml] [--lead 2] [--tail 2] [--listen N] [--think N]
        [--out recordings\\face_test.mp4]

--lead / --tail add seconds of silence before and after the speech (the idle face). --listen / --think insert N silent
chunks in those states before the speech, as when you talk to her and she works out a reply. Needs the GPU free; the
worker's log is logs\\face_worker.log.
"""
import argparse
import os
import sys
import time

import av
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from portravix import config as C  # noqa: E402
from portravix.face import Face  # noqa: E402


def load_wav(path, rate):
    with av.open(path) as c:
        res = av.AudioResampler(format="flt", layout="mono", rate=rate)
        return np.concatenate([r.to_ndarray().reshape(-1) for f in c.decode(audio=0) for r in res.resample(f)])


def write_mp4(path, frames, audio, fps, rate):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with av.open(path, "w") as c:
        vs = c.add_stream("libx264", rate=fps)
        vs.width, vs.height, vs.pix_fmt = frames.shape[2], frames.shape[1], "yuv420p"
        vs.options = {"crf": "18"}
        aus = c.add_stream("aac", rate=rate)
        aus.layout = "mono"
        for f in frames:
            for p in vs.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
                c.mux(p)
        for p in vs.encode():
            c.mux(p)
        for s in range(0, len(audio), 1024):
            af = av.AudioFrame.from_ndarray(audio[s:s + 1024].reshape(1, -1).astype(np.float32), format="flt", layout="mono")
            af.sample_rate = rate
            for p in aus.encode(af):
                c.mux(p)
        for p in aus.encode():
            c.mux(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--config", default=None)
    ap.add_argument("--lead", type=float, default=2.0)
    ap.add_argument("--tail", type=float, default=2.0)
    ap.add_argument("--listen", type=int, default=0, help="silent chunks in the listen state before the speech")
    ap.add_argument("--think", type=int, default=0, help="silent chunks in the think state before the speech")
    ap.add_argument("--out", default=os.path.join(ROOT, "recordings", "face_test.mp4"))
    a = ap.parse_args()

    t = time.perf_counter()
    face = Face(C.load(a.config))
    print(f"face ready in {time.perf_counter() - t:.1f}s", flush=True)
    rate, spc = face.audio_rate, round(face.chunk_seconds * face.audio_rate)

    def chunks(samples, mode):
        samples = np.pad(samples, (0, (-len(samples)) % spc))
        return [(c, mode) for c in samples.reshape(-1, spc)]

    plan = (chunks(np.zeros(int(a.lead * rate), np.float32), "idle")
            + chunks(np.zeros(a.listen * spc, np.float32), "listen")
            + chunks(np.zeros(a.think * spc, np.float32), "think")
            + chunks(load_wav(a.wav, rate), "talk")
            + chunks(np.zeros(int(a.tail * rate), np.float32), "idle"))
    audio = np.concatenate([c for c, _ in plan])

    frames, trips = [], []
    try:
        for i, (chunk, mode) in enumerate(plan):
            t = time.perf_counter()
            frames.append(face.render(chunk, mode=mode))
            trips.append(time.perf_counter() - t)
            w = face.last_timing
            print(f"chunk {i + 1:3d}: {trips[-1]:.2f}s ({trips[-1] / face.chunk_seconds:.2f}x real time) = audio "
                  f"{w['audio_s']:.2f} + sample {w['sample_s']:.2f} + decode {w['decode_s']:.2f} + app side "
                  f"{trips[-1] - w['total_s']:.2f} | {mode} -> {w.get('prompt', '?')}", flush=True)
    finally:
        face.close()
    trips = np.array(trips[1:] or trips)
    print(f"\n{len(frames)} chunks ({len(frames) * face.chunk_seconds:.1f}s of video): median {np.median(trips):.2f}s, "
          f"max {trips.max():.2f}s per {face.chunk_seconds:.3f}s chunk; {int((trips > face.chunk_seconds).sum())} over budget",
          flush=True)
    write_mp4(a.out, np.concatenate(frames), audio, face.fps, rate)
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
