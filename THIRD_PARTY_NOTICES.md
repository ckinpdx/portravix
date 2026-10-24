# Third-party notices

## Included in this repository

### Silero VAD (`models/silero_vad_v6.onnx`)

The ONNX export of Silero VAD v6, as bundled by faster-whisper 1.2.1. https://github.com/snakers4/silero-vad

```
MIT License

Copyright (c) 2020-present Silero Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Not included: downloaded or installed by you

| Component | What Portravix uses it for | Terms |
|---|---|---|
| [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) and [FastH3](https://huggingface.co/FastVideo) (FastVideo's distilled H3) | the face | [MiniMax H3 Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE) and its Acceptable Use Policy (see NOTICE) |
| Qwen3-VL-32B, as H3's text encoder | encoding the face prompt once | Apache-2.0 |
| [ComfyUI](https://github.com/comfyanonymous/ComfyUI) | runs H3 (used as a library) | GPL-3.0 |
| [MMH3Tools](https://github.com/ckinpdx/ComfyUI-MMH3Tools) | the chunk-by-chunk renderer (`mmh3tools.live`) | MIT |
| [Pocket TTS](https://github.com/kyutai-labs/pocket-tts) by Kyutai | the voice | code MIT; weights CC-BY-4.0 (the voice-cloning weights are gated) |
| [NVIDIA Parakeet TDT 0.6B v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2), ONNX export by [istupakov](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v2-onnx), run by [onnx-asr](https://github.com/istupakov/onnx-asr) | writing down what you say | model CC-BY-4.0; onnx-asr MIT |
| The LLM you choose (e.g. Gemma 4 E4B) | the conversation | that model's licence |
