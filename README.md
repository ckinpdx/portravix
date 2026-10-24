# Portravix

A live talking avatar that runs on your own machine: you talk to it from your phone, a local LLM answers, a local
voice speaks the answer, and a face rendered by **MiniMax H3** says it, streamed chunk by chunk as it is generated.

- **Face:** MiniMax H3 (FastVideo's FastH3 Omni NVFP4), animated from one portrait and lip-synced to the voice, rendered
  2.1 s at a time and colour-stabilised so it holds up over a long conversation.
- **Voice:** Kyutai's Pocket TTS on the CPU: a preset voice, or a clone of a voice you have the right to use.
- **Brain:** any OpenAI-compatible LLM server (llama.cpp, llama-swap, LM Studio, ...). Gemma 4 E4B is a good fit: it
  is fast, sees images, and hears your voice clip as well as the words.
- **Ears:** Silero VAD finds when you speak; NVIDIA Parakeet writes it down, on the CPU.
- **Phone page:** a web page served over [Tailscale](https://tailscale.com) only (it has no login), with chat, a
  hands-free mic, and an always-on launcher so you can start and stop the avatar from the phone.

It contains no model weights; you download those yourself (see Setup).

## Read this first: licences and consent

- **MiniMax H3's licence governs everything the face does.** H3 and FastH3 are under the
  [MiniMax H3 Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE). It
  applies only in the territories it lists, which **exclude some countries**: check whether it covers yours before
  you run Portravix. Its Acceptable Use Policy (Exhibit A) applies to everything Portravix generates.
- **Only animate a face and clone a voice you have the right to use.** Impersonating someone without their consent
  breaks MiniMax's policy and, in many places, the law. If you share anything Portravix makes, say that it is
  machine-generated.
- Portravix's own code is GPL-3.0 (it uses ComfyUI, which is GPL-3.0). Other components: NOTICE and
  THIRD_PARTY_NOTICES.md.

## What you need

- **An NVIDIA RTX 50-series GPU** (Blackwell: FastH3 here is NVFP4) with ~24 GB of VRAM free for the face plus the
  LLM. Developed and measured on an RTX 5090 (32 GB), Ryzen 9 7950X, Windows 11.
- **Windows.** The code mostly avoids Windows-only calls, but it has only been run there.
- **ComfyUI** with core MiniMax H3 support and the **[MMH3Tools](https://github.com/ckinpdx/ComfyUI-MMH3Tools) node
  pack, 0.108.0 or newer**, in `custom_nodes`. Portravix runs H3 through ComfyUI's code, in ComfyUI's own Python,
  without starting its server.
- **Python 3.11+** for the app (CPU only).
- **An OpenAI-compatible LLM server** running locally.
- **Tailscale** on the computer and the phone, for the phone page. HTTPS certificates enabled for the tailnet lets the
  page use H.264 video and the phone's mic.

## Setup

1. **Models**, into ComfyUI's model folders:

   | file | folder | from |
   |---|---|---|
   | `fasth3_omniref_qad_nvfp4_comfy.safetensors` | `diffusion_models` | [FastVideo/FastH3-Omni-NVFP4-QAD-Comfy](https://huggingface.co/FastVideo/FastH3-Omni-NVFP4-QAD-Comfy) |
   | `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` | `text_encoders` | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) |
   | `minimax_h3_video_vae_int8_convrot.safetensors` | `vae` | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) |
   | `minimax_h3_audio_vae_fp32.safetensors` | `vae` | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) |

   Parakeet (~660 MB) and Pocket TTS download themselves on first use.

2. **The app's Python environment**, in the Portravix folder:

   ```
   python -m venv .venv
   .venv\Scripts\pip install -r requirements.txt
   ```

3. **Your settings:** copy `config.example.toml` to `config.toml` and set at least `[comfyui] dir`, the `[brain]`
   server and model, and your face. Personal files go in `my/`, which git ignores:
   - `my/face.png`: a clear, front-facing portrait, ideally square.
   - `my/persona.md`: who the avatar is (start from `persona.example.md`). It is re-read when it changes.
   - optionally a voice WAV to clone; set `[voice] voice` to its path. Cloning needs the gated weights: accept the
     terms at [kyutai/pocket-tts](https://huggingface.co/kyutai/pocket-tts) and run `hf auth login` once.
   - `[face] subject`: a description of the person (age, look, hair). It fills `{subject}` in every prompt, and the
     prompts carry it into each scene.
   - optionally your own copies of the prompts in `prompts/`, in H3's structured prompt format (see Expressions
     below). Leave out what they say: the words come from the voice.

The first start encodes every prompt once with H3's text encoder (which briefly needs ~26 GB of VRAM) and caches
them in `cache\`; later starts skip that until the face, a prompt, the subject, the size or the window changes.

## Run it

**On this computer, with a window:**

```
.venv\Scripts\python scripts\chat.py [--record]
```

Type in the console; the avatar answers in the window. Ready in ~40 s.

**From your phone:** run the launcher, ideally from a Windows logon task so it is always there:

```
.venv\Scripts\pythonw scripts\launcher.py
```

Open `https://<computer>.<tailnet>.ts.net:8765` on the phone (with Tailscale on), tap the picture (phones only allow
sound after a tap), and press Start. Talk or type. Stop gives the GPU back.

- **Start** makes room first: it asks a running ComfyUI to unload its models, and, if `[gpu] llama_swap` is set,
  unloads llama-swap's idle models. It refuses to evict anything that is mid-task.
- **The line under the video** shows the stream's health: format, bitrate, how low the buffer got, refills (picture
  and voice paused together) and dropped chunks (a slow link).
- **The mic button** turns on hands-free voice: it listens, notices when you stop, and stays shut while the avatar
  speaks.
- `scripts\serve.py` runs the avatar directly, without the launcher (`--local` also shows the window here).

## How it works

```
you (voice) -> phone mic -> Silero VAD -> Parakeet (CPU): your words
you (typed or heard) -> LLM (streamed, sentence by sentence) -> Pocket TTS (CPU) -> 24 kHz audio
    -> SpeechBuffer --48 kHz--> Renderer -> face worker (H3, GPU) --frames--+
               +-------48 kHz-------------------------------------------------> Player (audio clock) -> phone / window
```

- **The face worker** (`workers/face_worker.py`) runs in ComfyUI's Python and uses ComfyUI as a library. It renders
  with MMH3Tools' `mmh3tools.live`: each chunk is a 56-frame window whose first 5 frames carry the end of the last
  window, so motion continues across seams; the carry is colour-matched to the first window, so a long session does
  not drift. The speech audio is pinned, so the face follows it. Details:
  [docs/live.md](https://github.com/ckinpdx/ComfyUI-MMH3Tools/blob/main/docs/live.md).
- **The renderer** keeps about one chunk queued ahead of playback, rendering silence while nobody is talking, so idle
  flows into speech without a seam.
- **Expressions:** each chunk gets the prompt that fits the moment, all encoded once at start, so switching costs
  nothing:

  | state | when | prompt (`[face]`) |
  |---|---|---|
  | talk, in a mood | the chunk has speech, and the reply was tagged with a mood the face has | `mood_prompts`: `talk_<mood>.txt` |
  | talk | the chunk has speech | `prompt` |
  | listen | the phone's mic hears you talking | `listen_prompt` |
  | think | the reply is being written | `think_prompt` |
  | idle | nothing is happening; rotates through the variations, each held `idle_hold` chunks | `idle_prompt` |

  **Moods** tie the words to the face: when there are `talk_<mood>.txt` prompts, the LLM is asked to start each reply
  with one mood tag (`[warm]`, `[playful]`, ...). The tag is never spoken or shown; it picks that reply's talking
  prompt. Shipped: warm, playful, thoughtful, excited. Adding `talk_<name>.txt` adds a mood.

  A talking prompt over silence makes the face mouth along to nothing, which is why the other states exist. A missing
  prompt falls back to idle, then to talk. States change at chunk boundaries, so the face reacts a chunk or two
  (2-4 s) after the moment. Prompt settings take wildcards, so the prompts can live in a folder someone else
  maintains (another agent, say) and new files become new variations at the next start.
- **The player** uses the audio as its clock: if rendering ever falls behind, picture and voice pause together
  rather than drift apart.
- **Frames** come back from the worker through shared memory; the pipe carries only small messages.

## Performance (RTX 5090, 512x512, 2 steps)

- A 56-frame window renders 2.125 s of video in ~1.7 s in the worker, ~1.8 s in the app with the LLM replying alongside:
  about 15% headroom, and one 0.02 s hiccup in a three-turn test conversation.
- Her mouth starts ~4 s after your message: the chunk already queued plays out, then one chunk renders. `window = 39`
  starts ~0.7 s sooner but leaves almost no headroom.
- 512x512 is the largest size that keeps up; render time grows from 544 up.
- VRAM: ~17 GB for the face worker, plus the LLM.

## Tests

- `scripts\test_face.py <speech.wav>`: the face on its own. Streams the WAV (with silence before and after) through the
  worker chunk by chunk, prints each chunk's time against real time, and writes `recordings\face_test.mp4`.
- `scripts\test_chat.py [--headless] [--no-record]`: three typed prompts answered end to end, with per-stage latency,
  render health and audio underruns.

## Known limits

- A faint colour step can show at chunk seams every ~2 s.
- One face per session, square output, English speech recognition.
- Not tested on Linux or macOS, or on GPUs without FP4 support (a non-NVFP4 H3 model would need more VRAM and time).
