"""Settings: config.toml in the project folder (start from config.example.toml), merged over these defaults.
Relative paths resolve against the project folder."""
import copy
import os
import tomllib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PATH = os.path.join(ROOT, "config.toml")

DEFAULTS = {
    "comfyui": {
        "dir": "C:/ComfyUI",
        "python": "",                 # empty: <dir>/venv/Scripts/python.exe (Windows) or <dir>/venv/bin/python
        "url": "http://127.0.0.1:8188",
        "free_before_start": True,    # the launcher asks a running ComfyUI to unload its models before Start
        "model": "fasth3_omniref_qad_nvfp4_comfy.safetensors",
        "text_encoder": "qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
        "video_vae": "minimax_h3_video_vae_int8_convrot.safetensors",
        "audio_vae": "minimax_h3_audio_vae_fp32.safetensors",
    },
    "face": {
        "image": "my/face.png",
        "prompt": "prompts/face.txt",
        "subject": "the person in <Picture 1>",  # fills {subject} in the prompt files
        "subject_file": "",                 # or read the subject from a file (overrides subject)
        "mood_prompts": "prompts/talk_*.txt",  # talk_<mood>.txt: one talking prompt per mood the LLM can tag
        "idle_prompt": "prompts/idle*.txt",  # paths, a list, or wildcards; rotated while nothing happens
        "idle_hold": 2,                     # windows each idle variation lasts
        "listen_prompt": "prompts/listen.txt",
        "think_prompt": "prompts/think.txt",
        "size": 512,
        "window": 56,
        "steps": 2,
        "seed": 1,
    },
    "voice": {
        "voice": "eve",
        "language": "english",
        "config": "",                 # optional custom Pocket TTS YAML; overrides language
        "python": "",                 # empty: the interpreter running the app
        "threads": 4,
    },
    "brain": {
        "url": "http://127.0.0.1:8080/v1/chat/completions",
        "model": "gemma-4-e4b",
        "persona": "my/persona.md",
        "max_tokens": 300,
        "history_turns": 20,
        "audio": True,
    },
    "memory": {
        "file": "my/memory.md",       # what the avatar remembers across sessions; empty: no memory
        "user_name": "",              # what the memory and the app's instructions call you (empty: "the user")
        "budget_tokens": 1200,        # memory given to the LLM (newest lines first), all but the summaries
        "summaries": 5,               # conversation summaries given to the LLM
        "keep_summaries": 20,         # kept in the file; older ones move to <file>_archive.md
        "quiet_minutes": 10,          # quiet this long: curate, and the next exchange starts a new conversation
    },
    "debug": {
        "transcript": False,          # write every exchange to logs/transcript_<session>.jsonl
    },
    "ears": {
        "parakeet_dir": "",
        "threads": 4,
    },
    "server": {
        "host": "",
        "port": 8765,
        "avatar_port": 8766,
    },
    "gpu": {
        "needed_vram_gb": 24,
        "llama_swap": "",
    },
}


def _merge(base, override):
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def load(path=None):
    """The settings as a nested dict. A missing config.toml means all defaults."""
    cfg = copy.deepcopy(DEFAULTS)
    path = path or os.environ.get("PORTRAVIX_CONFIG") or DEFAULT_PATH
    if os.path.exists(path):
        with open(path, "rb") as f:
            _merge(cfg, tomllib.load(f))
    cfg["_path"] = os.path.abspath(path)
    return cfg


def resolve(path):
    """A configured path, relative to the project folder unless absolute. Empty stays empty."""
    if not path:
        return ""
    path = os.path.expanduser(str(path))
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def prompt_paths(value):
    """A prompt setting -- one path, a list, or wildcards like "face/idle*.txt" -- as the existing files, sorted."""
    import glob
    out = []
    for v in (value if isinstance(value, list) else [value]):
        if not v:
            continue
        path = resolve(v)
        out += sorted(glob.glob(path)) if any(ch in path for ch in "*?[") else ([path] if os.path.exists(path) else [])
    return out


def subject(cfg):
    """Who the prompts describe ({subject}): [face] subject_file if set and present, else [face] subject."""
    face = cfg["face"]
    path = resolve(face.get("subject_file", ""))
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            text = " ".join(f.read().split())
        if text:
            return text.rstrip(".")
    return str(face.get("subject") or "the person in <Picture 1>")


def moods(cfg):
    """[(mood, prompt path)] from the talk_<mood>.txt files [face] mood_prompts finds, in name order."""
    found = {}
    for path in prompt_paths(cfg["face"].get("mood_prompts", "")):
        stem = os.path.splitext(os.path.basename(path))[0]
        name = stem[5:] if stem.lower().startswith("talk_") else stem
        if name and name.replace("_", "").isalnum():
            found.setdefault(name.lower(), path)
    return sorted(found.items())


def comfy_python(cfg):
    if cfg["comfyui"]["python"]:
        return resolve(cfg["comfyui"]["python"])
    venv = os.path.join(resolve(cfg["comfyui"]["dir"]), "venv")
    return os.path.join(venv, "Scripts", "python.exe") if os.name == "nt" else os.path.join(venv, "bin", "python")
