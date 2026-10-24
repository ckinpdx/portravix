"""The face: MiniMax H3 (FastH3) rendering one chunk at a time, in ComfyUI's Python, using ComfyUI as a library.

    <ComfyUI venv python> face_worker.py [--config config.toml]

Needs ComfyUI with core MiniMax H3 support and the MMH3Tools node pack >= 0.107.0 (its `mmh3tools.live` module does
the rendering; see its docs/live.md). The face image and prompt come from the [face] section of the config. The
face/prompt conditioning is encoded once with H3's text encoder (~26 GB of VRAM for a moment) and cached in cache\;
it is only rebuilt when the image, prompt, size, window or text encoder changes.

Protocol on stdin/stdout (binary; stdout carries nothing else):
    worker -> b"R" + uint32 n + JSON header (fps, frames_per_chunk, samples_per_chunk, audio_rate, width, height, shm)
    client -> b"C" + uint32 nbytes + float32 mono audio, exactly samples_per_chunk samples at audio_rate
    worker -> b"F" + uint32 n + JSON timings + uint32 0; the frames, uint8 RGB [frames_per_chunk, H, W, 3], are in the
              shared memory block named "shm" in the header, and the client copies them out before its next request
    client -> b"Q" (or EOF) to quit
"""
import argparse
import hashlib
import json
import logging
import os
import struct
import sys
import time

out = sys.stdout.buffer
sys.stdout = sys.stderr  # anything a library prints must not land in the protocol stream

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from portravix import config as C  # noqa: E402  (stdlib only: safe before ComfyUI is set up)

parser = argparse.ArgumentParser()
parser.add_argument("--config", default=None)
opts = parser.parse_args()
cfg = C.load(opts.config)
comfy_dir = C.resolve(cfg["comfyui"]["dir"])
face_cfg = cfg["face"]

# ---- ComfyUI as a library: the parts of its main.py that matter here, with dynamic VRAM on -----------------------
# Dynamic VRAM stays ON: with --disable-dynamic-vram the legacy patcher overflowed a 32 GB card into system RAM.
# The Comfy compiler stays OFF: it gained nothing here (the forwards are compute-bound) and it has failed on some graphs.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")
os.chdir(comfy_dir)
sys.path.insert(0, comfy_dir)
sys.argv = [sys.argv[0], "--use-pytorch-cross-attention", "--disable-comfy-compiler"]
logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="[face] %(message)s")

import comfy.options  # noqa: E402
comfy.options.enable_args_parsing()
from comfy.cli_args import args, enables_dynamic_vram  # noqa: E402
import utils.extra_config  # noqa: E402
if os.path.exists(os.path.join(comfy_dir, "extra_model_paths.yaml")):
    utils.extra_config.load_extra_path_config(os.path.join(comfy_dir, "extra_model_paths.yaml"))
import comfy_aimdo.control  # noqa: E402
if enables_dynamic_vram():
    comfy_aimdo.control.init(simple_vram_headroom=None if args.reserve_vram is None else int(args.reserve_vram * 1024 ** 3),
                             nvml_pressure=not args.disable_nvml_pressure)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import comfy.memory_management  # noqa: E402
import comfy.model_management as mm  # noqa: E402
import comfy.model_patcher  # noqa: E402
if args.enable_dynamic_vram or (enables_dynamic_vram() and mm.is_nvidia()):
    try:
        ok = comfy_aimdo.control.init_devices((d.index, int(args.vram_headroom * 1024 ** 3))
                                              for d in mm.get_all_torch_devices())
    except TypeError:
        ok = comfy_aimdo.control.init_devices(d.index for d in mm.get_all_torch_devices())
    if ok:
        comfy_aimdo.control.set_log_warning()
        comfy.model_patcher.CoreModelPatcher = comfy.model_patcher.ModelPatcherDynamic
        comfy.memory_management.aimdo_enabled = True


def find_pack():
    """The MMH3Tools folder in ComfyUI's custom_nodes (whatever it was cloned as)."""
    nodes_dir = os.path.join(comfy_dir, "custom_nodes")
    for name in sorted(os.listdir(nodes_dir)):
        if os.path.isfile(os.path.join(nodes_dir, name, "mmh3tools", "live.py")):
            return os.path.join(nodes_dir, name)
    raise SystemExit("MMH3Tools >= 0.107.0 not found in %s (it needs mmh3tools/live.py)" % nodes_dir)


sys.path.insert(0, find_pack())
import nodes  # noqa: E402
from mmh3tools import live  # noqa: E402


def send(kind, payload=b""):
    out.write(kind + payload)
    out.flush()


def read_exact(n):
    buf = b""
    while len(buf) < n:
        part = sys.stdin.buffer.read(n - len(buf))
        if not part:
            return None
        buf += part
    return buf


def load_image(path):
    from PIL import Image
    return torch.from_numpy(np.asarray(Image.open(path).convert("RGB"), np.float32) / 255.0)[None]


def conditioning(vae, audio_vae, config):
    """Face + prompt conditioning, from the cache when nothing that goes into it has changed."""
    image_path, prompt_path = C.resolve(face_cfg["image"]), C.resolve(face_cfg["prompt"])
    for what, path in (("face image", image_path), ("prompt", prompt_path)):
        if not os.path.exists(path):
            raise SystemExit(f"{what} not found: {path} (set [face] in {cfg['_path']})")
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    with open(prompt_path, encoding="utf-8") as f:
        prompt = f.read().strip()
    key = hashlib.sha256(image_bytes + prompt.encode() + f"|{config.size}|{config.window_frames}|"
                         f"{cfg['comfyui']['text_encoder']}".encode()).hexdigest()[:16]
    cache = os.path.join(ROOT, "cache", f"face_{key}.pt")
    if os.path.exists(cache):
        logging.info("conditioning from %s", cache)
        saved = torch.load(cache, weights_only=False)
        return saved["cond"], saved["template"]
    t = time.perf_counter()
    clip = nodes.CLIPLoader().load_clip(cfg["comfyui"]["text_encoder"], type="minimax", device="default")[0]
    cond, template = live.build_conditioning(clip, vae, audio_vae, load_image(image_path), prompt, config)
    del clip
    mm.unload_all_models()
    mm.soft_empty_cache()
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    torch.save({"cond": cond, "template": template}, cache)
    logging.info("conditioning encoded in %.1fs, cached to %s", time.perf_counter() - t, cache)
    return cond, template


def main():
    t = time.perf_counter()
    config = live.LiveConfig(size=int(face_cfg["size"]), window_frames=int(face_cfg["window"]),
                             steps=int(face_cfg["steps"]), seed=int(face_cfg["seed"]))
    vae, audio_vae = live.load_vae(cfg["comfyui"]["video_vae"]), live.load_vae(cfg["comfyui"]["audio_vae"])
    cond, template = conditioning(vae, audio_vae, config)
    model = live.load_model(cfg["comfyui"]["model"])
    face = live.LiveH3(model, vae, audio_vae, cond, template, config)
    load_s = time.perf_counter() - t
    _, warm = face.render(np.zeros(face.samples_per_chunk, np.float32))   # warm-up: colour anchor + first carry
    logging.info("loaded in %.1fs, warm-up window %.1fs", load_s, warm["total_s"])

    # Everything loaded so far lives for the whole session. Freeze it out of the garbage collector's reach: a full
    # collection walking ComfyUI's object graph took 0.24 s every few chunks, enough to stall playback.
    import gc
    gc.collect()
    gc.freeze()
    # Frames go back through shared memory: pushing ~40 MB a chunk through the pipe cost ~0.19 s.
    from multiprocessing import shared_memory
    shape = (face.frames_per_chunk, config.size, config.size, 3)
    shm = shared_memory.SharedMemory(create=True, size=int(np.prod(shape)))
    shm_frames = np.ndarray(shape, np.uint8, buffer=shm.buf)
    header = {"fps": face.fps, "frames_per_chunk": face.frames_per_chunk, "samples_per_chunk": face.samples_per_chunk,
              "audio_rate": config.audio_rate, "width": config.size, "height": config.size, "load_s": load_s,
              "shm": shm.name}
    body = json.dumps(header).encode()
    send(b"R", struct.pack("<I", len(body)) + body)
    try:
        while True:
            kind = sys.stdin.buffer.read(1)
            if not kind or kind == b"Q":
                return
            (nbytes,) = struct.unpack("<I", read_exact(4))
            audio = np.frombuffer(read_exact(nbytes), "<f4")
            frames, timing = face.render(audio)
            shm_frames[:] = frames
            meta = json.dumps(timing).encode()
            send(b"F", struct.pack("<I", len(meta)) + meta + struct.pack("<I", 0))
    finally:
        del shm_frames
        shm.close()
        shm.unlink()


if __name__ == "__main__":
    main()
