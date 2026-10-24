"""The face: MiniMax H3 (FastH3) rendering one chunk at a time, in ComfyUI's Python, using ComfyUI as a library.

    <ComfyUI venv python> face_worker.py [--config config.toml]

Needs ComfyUI with core MiniMax H3 support and the MMH3Tools node pack >= 0.108.0 (its `mmh3tools.live` module does
the rendering; see its docs/live.md). The face image and prompts come from the [face] section of the config: talk
(any window with speech), and for silent windows listen, think, or one of the idle variations (see Prompter), so the
face does not mouth along to nothing. Each is encoded once with H3's text encoder (~26 GB of VRAM for a moment) and
cached in cache\; it is only rebuilt when the image, that prompt, the subject, size, window or text encoder changes.

Protocol on stdin/stdout (binary; stdout carries nothing else):
    worker -> b"R" + uint32 n + JSON header (fps, frames_per_chunk, samples_per_chunk, audio_rate, width, height, shm)
    client -> b"C" + uint8 mode + uint8 mood + uint32 nbytes + float32 mono audio, exactly samples_per_chunk samples at
              audio_rate; mode 0 auto, 1 talk, 2 idle, 3 listen (the user is speaking), 4 think (the reply is being
              written); mood 0 none, else 1 + its index in the header's "moods". Speech in the window always gets a
              talking prompt: the mood's, if it has one.
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
    raise SystemExit("MMH3Tools >= 0.108.0 not found in %s (it needs mmh3tools/live.py)" % nodes_dir)


sys.path.insert(0, find_pack())
import nodes  # noqa: E402
from mmh3tools import live  # noqa: E402
if not hasattr(live, "build_conditionings"):
    raise SystemExit("MMH3Tools is too old: Portravix needs 0.108.0 or newer (a per-window prompt). Update the pack.")


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


MODES = {0: "auto", 1: "talk", 2: "idle", 3: "listen", 4: "think"}   # the byte the client sends with each chunk


def conditionings(vae, audio_vae, config):
    """Every prompt's conditioning for this face, {name: cond}: "talk", "talk_<mood>" per mood, "idle0".."idleN",
    "listen", "think" (the optional ones only when their files exist). Prompt files may say {subject}, filled from
    [face] subject_file or subject. Each comes from the cache when nothing that goes into it has changed; the missing
    ones are encoded together in one text-encoder pass. Also returns the window's latent template."""
    image_path = C.resolve(face_cfg["image"])
    if not os.path.exists(image_path):
        raise SystemExit(f"face image not found: {image_path} (set [face] image in {cfg['_path']})")
    talk = C.prompt_paths(face_cfg["prompt"])
    if not talk:
        raise SystemExit(f"prompt not found: {C.resolve(face_cfg['prompt'])} (set [face] prompt in {cfg['_path']})")
    paths = {"talk": talk[0]}
    paths.update({f"talk_{mood}": p for mood, p in C.moods(cfg)})
    paths.update({f"idle{i}": p for i, p in enumerate(C.prompt_paths(face_cfg.get("idle_prompt", "")))})
    for kind in ("listen", "think"):
        found_paths = C.prompt_paths(face_cfg.get(f"{kind}_prompt", ""))
        if found_paths:
            paths[kind] = found_paths[0]
    subject = C.subject(cfg)
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    prompts, caches, found = {}, {}, {}
    for kind, path in paths.items():
        with open(path, encoding="utf-8") as f:
            prompts[kind] = f.read().replace("{subject}", subject).strip()
        key = hashlib.sha256(image_bytes + prompts[kind].encode() + f"|{config.size}|{config.window_frames}|"
                             f"{cfg['comfyui']['text_encoder']}".encode()).hexdigest()[:16]
        caches[kind] = os.path.join(ROOT, "cache", f"face_{key}.pt")
        if os.path.exists(caches[kind]):
            found[kind] = torch.load(caches[kind], weights_only=False)
    missing = [k for k in paths if k not in found]
    if missing:
        t = time.perf_counter()
        clip = nodes.CLIPLoader().load_clip(cfg["comfyui"]["text_encoder"], type="minimax", device="default")[0]
        conds, template = live.build_conditionings(clip, vae, audio_vae, load_image(image_path),
                                                   [prompts[k] for k in missing], config)
        del clip
        mm.unload_all_models()
        mm.soft_empty_cache()
        os.makedirs(os.path.join(ROOT, "cache"), exist_ok=True)
        for kind, cond in zip(missing, conds):
            found[kind] = {"cond": cond, "template": template}
            torch.save(found[kind], caches[kind])
        logging.info("encoded %s in %.1fs and cached", ", ".join(missing), time.perf_counter() - t)
    logging.info("prompts: %s (cached: %s)", ", ".join(paths), ", ".join(k for k in paths if k not in missing) or "none")
    return {k: v["cond"] for k, v in found.items()}, found["talk"]["template"]


def is_silent(audio):
    """The renderer pads windows with no speech with exact zeros; a TTS sentence is never this quiet."""
    return not np.any(np.abs(audio) > 1e-4)


class Prompter:
    """Picks each window's prompt. Speech in the window: talk. Otherwise the client's mode: listen (you are
    talking) or think (her reply is being written), when those prompts exist; else idle, rotating through the idle
    variations, each held for [face] idle_hold windows so she doesn't flit between moods. No idle prompt at all:
    talk (the behaviour before idle prompts)."""

    def __init__(self, conds, hold, seed):
        self.conds = conds
        self.idle = sorted(k for k in conds if k.startswith("idle"))
        self.hold = max(1, int(hold))
        self.rng = np.random.default_rng(seed)
        self.current, self.held = (self.idle[0] if self.idle else None), 0

    def _next_idle(self):
        if not self.idle:
            return "talk"
        if self.held >= self.hold and len(self.idle) > 1:
            others = [k for k in self.idle if k != self.current]
            self.current, self.held = others[int(self.rng.integers(len(others)))], 0
        self.held += 1
        return self.current

    def pick(self, mode, silent, mood=None):
        if not silent or mode == "talk":
            name = f"talk_{mood}" if mood and f"talk_{mood}" in self.conds else "talk"
        elif mode in ("listen", "think") and mode in self.conds:
            name = mode
        else:
            name = self._next_idle()
        return name, self.conds[name]


def main():
    t = time.perf_counter()
    config = live.LiveConfig(size=int(face_cfg["size"]), window_frames=int(face_cfg["window"]),
                             steps=int(face_cfg["steps"]), seed=int(face_cfg["seed"]))
    vae, audio_vae = live.load_vae(cfg["comfyui"]["video_vae"]), live.load_vae(cfg["comfyui"]["audio_vae"])
    conds, template = conditionings(vae, audio_vae, config)
    prompter = Prompter(conds, face_cfg.get("idle_hold", 2), int(face_cfg["seed"]))
    model = live.load_model(cfg["comfyui"]["model"])
    face = live.LiveH3(model, vae, audio_vae, conds["talk"], template, config)
    load_s = time.perf_counter() - t
    # warm-up: colour anchor + first carry, with the prompt a silent window will get
    silence = np.zeros(face.samples_per_chunk, np.float32)
    _, warm = face.render(silence, cond=prompter.pick("idle", True)[1])
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
    moods = sorted(k[5:] for k in conds if k.startswith("talk_"))   # the client sends 1 + index, 0 for none
    header = {"fps": face.fps, "frames_per_chunk": face.frames_per_chunk, "samples_per_chunk": face.samples_per_chunk,
              "audio_rate": config.audio_rate, "width": config.size, "height": config.size, "load_s": load_s,
              "shm": shm.name, "moods": moods}
    body = json.dumps(header).encode()
    send(b"R", struct.pack("<I", len(body)) + body)
    try:
        while True:
            kind = sys.stdin.buffer.read(1)
            if not kind or kind == b"Q":
                return
            mode_byte, mood_byte, nbytes = struct.unpack("<BBI", read_exact(6))
            mode = MODES.get(mode_byte, "auto")
            mood = moods[mood_byte - 1] if 0 < mood_byte <= len(moods) else None
            audio = np.frombuffer(read_exact(nbytes), "<f4")
            # A talking prompt over silence makes the idle face mouth along to nothing: pick per window.
            name, cond = prompter.pick(mode, is_silent(audio), mood)
            frames, timing = face.render(audio, cond=cond)
            timing["mode"], timing["prompt"] = mode, name
            shm_frames[:] = frames
            meta = json.dumps(timing).encode()
            send(b"F", struct.pack("<I", len(meta)) + meta + struct.pack("<I", 0))
    finally:
        del shm_frames
        shm.close()
        shm.unlink()


if __name__ == "__main__":
    main()
