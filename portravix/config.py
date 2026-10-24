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


def comfy_python(cfg):
    if cfg["comfyui"]["python"]:
        return resolve(cfg["comfyui"]["python"])
    venv = os.path.join(resolve(cfg["comfyui"]["dir"]), "venv")
    return os.path.join(venv, "Scripts", "python.exe") if os.name == "nt" else os.path.join(venv, "bin", "python")
