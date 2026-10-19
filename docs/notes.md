# Notes and measurements

Measured on a Core Ultra 7 258V (Lunar Lake), OpenVINO 2026.4.1.

## Memory when idle

After `--idle-unload` seconds without a request (default 300, like Ollama's `keep_alive`; `0` keeps the model loaded) the
server frees the model, so an idle server holds almost nothing (~0.2 GB RSS). The next request reloads it in about 2 s
thanks to the compile cache. `/api/ps` is empty while unloaded. Measured on a Core Ultra 7 258V: the NPU memory goes back to
0 and the GPU's to ~0.2 GB (per-process DRM counters), and free system RAM returns to its starting value.

## Tokens per second

Every request is logged with numbers only (never text) and returned in the response, always on: collecting it costs
nothing measurable (GenAI records the metrics regardless; our part is ~2 microseconds per request, and 2699.0 ms vs
2699.3 ms per generation with and without it), so there is no switch.

```
INFO translategemma:4b: 51 tokens in 1.15s = 43.4 tok/s (prompt 148 tokens in 0.56s)
```

- Ollama `/api/chat`: `eval_count`, `eval_duration` (ns), `prompt_eval_count`, `prompt_eval_duration`; `/api/generate`
  preload returns `load_duration`. Decode speed: `curl ... | jq '.eval_count / (.eval_duration / 1e9)'`.
- OpenAI `/v1/chat/completions`: real `usage` plus `timings` (`predicted_per_second`, `prompt_ms`, `predicted_ms`); when
  streaming they come in the final chunk.
- Decode speed excludes prefill.

## NPU status (Lunar Lake, OpenVINO 2026.4.1)

- NPU LLMs prefill in 128-token chunks (`NPU_LLM_CONFIG` in `server/backends.py`) instead of one pass padded to
  `MAX_PROMPT_LEN`. A typical 90-150 token request now reaches its first token in ~0.3 s instead of ~0.7-0.8 s
  (TranslateGemma), at the same decode speed: 16-27% less time per typical
  NPU request (measured interleaved in one process). Output is identical for prompts that fit one chunk; longer ones
  can differ slightly in wording. A chunk of 64 does not compile. Prefix caching nets out (0.26 s faster first token on
  two-chunk prompts, ~3% slower decode). Long custom prompts prefill slower (688 tokens: ~2.1 s instead of ~0.7 s).
- `translategemma:4b` (Gemma-3-4B INT4, run text-only through `VLMPipeline`): **loads and runs on the NPU** (~13 s first
  compile, ~2 s cached, ~21 tok/s), with limits:
  - `MAX_PROMPT_LEN` must be <= ~896: at 1024 or more the NPU output is garbage (Gemma-3's sliding window is 1024
    tokens). The registry uses 768. The KV cache holds prompt + output (768 + 128 = 896 tokens), so a long paragraph used
    to come back cut off mid-sentence, and output quality drops on long prompts (garbled Estonian with Cyrillic at
    ~480 prompt tokens). On the NPU the server therefore translates text over 128 tokens in sentence groups
    (`npu_piece_tokens`; never splitting inside the extension's `[N:...]` markers, keeping line breaks): the same
    370-word paragraph now comes back complete and clean. A custom (unparsed) prompt over 768 tokens still gets a clear
    "input too long" 400.
  - An INT8 export compiled but stalled on a ~270-token input.
  - Same IR on other devices is correct: GPU ~37 tok/s, CPU ~24 tok/s. Like every model it runs on `--device` (default
    NPU); `--device GPU` is the way to get the best quality and speed from it.
  - The model's own chat template does not parse in GenAI's Jinja engine, so the server writes Gemma's turn markup itself
    (`VLMBackend`). Its Gemma license terms apply to the weights.

- The flake builds Intel's NPU compiler (`npu_compiler` release, pinned hash) and links it into the venv's `openvino/libs` on shell entry; pip OpenVINO >= 2026 does not ship it. The Level Zero loader must be on `LD_LIBRARY_PATH` (flake does it) or no NPU backend is found. Do not set `NPU_PLATFORM`: it forces compile-only mode.
- NPU userland driver comes from the system (`/run/opengl-driver/lib`), matching your firmware.

Other notes: `optimum-cli export` can hang after finishing; files are complete, kill it.
