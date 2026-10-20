# Local LLM NPU Translator

A local translator that runs on your Intel NPU.

Run **TranslateGemma** on the **Intel NPU** and serve it through an **Ollama-compatible API** on `localhost:11434`.
Anything that already talks to Ollama can then translate locally on the NPU, leaving the GPU and CPU free:

- [Llama Franca](https://github.com/iNViTiON/llama-franca) (fork, branch `npu`): translates web pages in the browser.
- [OCR Translate](https://github.com/iNViTiON/dms-ocr-translate): a DankMaterialShell plugin that reads text from the
  screen with Tesseract and translates it.
- `curl`, scripts, or any other Ollama or OpenAI client.

Nothing leaves the machine: no account, no API key, no telemetry.

## Features

- **Ollama API** (`/api/tags`, `/api/ps`, `/api/generate`, `/api/chat`), so Ollama clients work unchanged, and an
  **OpenAI API** (`/v1/models`, `/v1/chat/completions`).
- **Streaming**: `/api/chat` with `"stream": true` returns text as it is generated (Ollama's NDJSON format);
  `/v1/chat/completions` streams Server-Sent Events.
- **Long text on the NPU**: the NPU pipeline has a fixed prompt window, so long input is cut into sentence groups
  (never inside Llama Franca's `[N:...]` link markers, keeping line breaks) and translated piece by piece.
- **Several devices**: serve the model on the NPU, GPU and CPU at once; clients pick one with names like
  `translategemma:4b@GPU`.
- **Frees memory when idle** (after 5 minutes by default) and reloads in about 2 s from the compile cache.
- **Speed in every answer**: Ollama's `eval_count` / `eval_duration`, or OpenAI's `usage` and `timings`.
- **Safe on localhost**: web pages cannot use it. Only browser extensions and non-browser clients may call it (see
  [Security](#security)).

## Requirements

- Linux x86_64 with an Intel NPU ("AI Boost"), the `intel_vpu` kernel driver and the NPU user-space driver
  (Level Zero). Developed and tested on a **Core Ultra 7 258V (Lunar Lake)** on NixOS. Other NPU generations are
  untested.
- [Nix](https://nixos.org/download) with flakes. The dev shell provides Python 3.12, `uv` and Intel's NPU compiler,
  which the OpenVINO pip packages no longer ship. The NPU user-space driver itself comes from the system
  (`/run/opengl-driver/lib` on NixOS) so that it matches the kernel firmware.
- A Hugging Face account: TranslateGemma is gated, so accept its license at
  [huggingface.co/google/translategemma-4b-it](https://huggingface.co/google/translategemma-4b-it) and create a
  read-only token.
- Disk: 3.3 GB for the converted model. The one-time conversion downloads about 8.6 GB and needs about 1 GB of extra
  Python packages; both can be deleted afterwards ([docs/cleanup.md](docs/cleanup.md)).

## Quick start

```bash
git clone https://github.com/iNViTiON/local-llm-npu-translator
cd local-llm-npu-translator
nix develop                        # or `direnv allow`

# One-time: convert TranslateGemma to OpenVINO INT4 for the NPU
uv sync --group convert            # export-time packages (torch, optimum-intel, nncf)
uv run hf auth login               # paste your read-only token; stored in .cache/hf
uv run python -m scripts.convert_translation_models --model-id google/translategemma-4b-it --quant int4
uv sync                            # drop the export-time packages again

./run.sh                           # serve on 127.0.0.1:11434 (NPU)
```

The first request compiles the model for the NPU (40-70 s); after that, compiled blobs in `.cache/ov` make restarts
take a few seconds. `./run.sh` passes its flags to the server, for example `./run.sh --device NPU,GPU`.

Port 11434 is Ollama's default, so stop Ollama or move one of them (`--port`).

Check that it works:

```bash
curl -s localhost:11434/api/tags
curl -s localhost:11434/api/chat -d '{
  "model": "translategemma:4b", "stream": false,
  "messages": [{"role": "user", "content": "You are a professional Estonian (et) to English (en) translator. Your goal is to accurately convey the meaning and nuances of the original Estonian text while adhering to English grammar, vocabulary, and cultural sensitivities.\nProduce only the English translation, without any additional explanations or commentary. Please translate the following Estonian text into English:\n\n\nTere hommikust! Kuidas sul läheb?"}]
}' | jq -r .message.content
```

## Using it

**Prompt.** Send TranslateGemma's own prompt template (above, from
[ollama.com/library/translategemma](https://ollama.com/library/translategemma)) as one user message. The server
recognises it, takes the languages and the text from it, and builds the model input itself; that is what enables the
long-text splitting. Any other prompt is passed to the model as a plain chat. On the OpenAI endpoint, `src_lang` and
`tgt_lang` (top level or in `extra_body`) add language hints.

**Llama Franca.** Its default model name, `translategemma:4b`, is what this server lists, so it works without
settings. Browser extension origins are allowed, so no `OLLAMA_ORIGINS` is needed. The fork's model picker also lists
the per-device names.

**OCR Translate.** Its server setting defaults to `http://localhost:11434`; pick a model in its settings.

### Streaming

`"stream": true` on `/api/chat` answers like Ollama: `Content-Type: application/x-ndjson`, one JSON object per line
with the next bit of text in `message.content`, then a last object with `"done": true` and the timings. The
long-text pieces stream one after another. A client that closes the connection stops the generation.

### Devices

```bash
./run.sh --device NPU,GPU,CPU --default-device NPU
```

| Model name | Runs on |
|---|---|
| `translategemma:4b` | the `--default-device` (NPU by default; `any` = whichever served device is idle) |
| `translategemma:4b@NPU`, `@GPU`, `@CPU` | that device only |

A device loads the model the first time a request needs it. Measured on the 258V: NPU ~21 tokens/s, GPU ~37, CPU ~24.
The NPU is the quiet, power-efficient choice and keeps the GPU free; the GPU is the fastest.

### Options

| Flag | Default | |
|---|---|---|
| `--device` | `NPU` | devices to serve, comma-separated |
| `--default-device` | `NPU` | where a plain model name runs |
| `--fallback` | `none` | `gpu`, `cpu` or `all`: devices to try when the NPU compile fails |
| `--idle-unload` | `300` | seconds without requests before the model is freed (`0` = keep it loaded) |
| `--preload` | off | compile at startup instead of on the first request |
| `--host` / `--port` | `127.0.0.1` / `11434` | there is no authentication, keep it on loopback |
| `--allow-origin` | | extra browser origin allowed to call the API, e.g. a local web UI |
| `--cache-dir` / `--no-cache` | `.cache/ov` | compiled-model cache |

### Running as a service

On NixOS, the flake's module runs the server as a systemd user service from your checkout, started at login and
restarted on failure:

```nix
# flake.nix
inputs.local-llm-npu-translator.url = "github:iNViTiON/local-llm-npu-translator";

# configuration
imports = [ inputs.local-llm-npu-translator.nixosModules.default ];
services.npu-translator = {
  enable = true;
  user = "alice";
  directory = "/home/alice/local-llm-npu-translator";   # the default; must hold the converted model
  devices = "all";                                       # "npu" (default) or NPU, GPU and CPU
  idleUnloadSeconds = 300;
};
```

The module only starts `run.sh`, so the checkout, its model conversion and flakes for that user are still needed.
Elsewhere, any user service that runs `./run.sh` works.

## Security

The server has no authentication and listens on loopback. Because any website open in the browser can send requests
to `localhost`, it refuses every request whose `Origin` is a web page (only browser extensions, `--allow-origin`
entries and clients that send no `Origin`, such as `curl`, are served) and every request whose `Host` is not a
loopback name (DNS rebinding). Request size, input length and generation length are capped. The review, with
measurements and the prompt-injection findings, is in [docs/security.md](docs/security.md).

## Development

```bash
uv run pytest                      # no NPU or model needed: tests use a stub backend
uv run python -m server.main --stub
```

`server/app.py` holds the HTTP API, `server/backends.py` the OpenVINO pipelines (TranslateGemma runs through
`VLMPipeline`; an NLLB and a decoder-only LLM backend are kept and tested), `server/registry.py` the served models.

Measurements, NPU limits and why the code works around them: [docs/notes.md](docs/notes.md).
Freeing disk space: [docs/cleanup.md](docs/cleanup.md).

## License

MIT, see [LICENSE](LICENSE). The TranslateGemma weights are not part of this repository and come under Google's
[Gemma terms of use](https://ai.google.dev/gemma/terms).
