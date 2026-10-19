# Disk cleanup

Most of the disk is not needed to serve translations. Measured on a Core Ultra 7 258V:

| What | Size | Needed to serve? | Comes back via |
|---|---|---|---|
| `.cache/hf` (original HF checkpoints, only used for conversion; also holds your HF token) | 8.3 GB, 16-17 GB with TranslateGemma | no | automatic re-download on the next convert |
| `.venv` conversion-only packages (torch, nncf, optimum-intel + dependents) | ~0.9 GB | no | `uv sync --group convert` |
| `.venv` runtime packages (openvino, genai, transformers, fastapi ...) | 0.4 GB | yes | `uv sync` |
| `models/` (converted INT4 IR: TranslateGemma 3.3 GB) | 5.1 GB | **yes** | `scripts.convert_translation_models` (minutes, plus the 8 GB download) |
| uv cache `~/.cache/uv` (shared by all uv projects) | 1.4 GB | no | re-downloaded on demand |
| `.cache/ov` (compiled NPU blobs, makes restarts fast; ~1.7 GB) | 1.7 GB | no, but a restart then recompiles 40-70 s per model | rebuilt automatically on the next start |
| `__pycache__`, `.pytest_cache`, `.*.log` | < 1 MB | no | regenerated |

`scripts/cleanup.py` is stdlib-only, so it runs without the venv. **Dry run by default**: it prints each step, its size,
the paths and how to restore, and deletes nothing until you pass `--yes`.

```bash
python3 -m scripts.cleanup                          # report only
python3 -m scripts.cleanup --all --yes              # safe set, keeps everything needed to serve (8.3 GB cache, +~0.9 GB if convert deps installed)
python3 -m scripts.cleanup --everything --yes       # also .venv, converted models, compile cache, uv cache
python3 -m scripts.cleanup --conversion-cache --yes # single steps: see --help
python3 -m scripts.cleanup --nix-gc --yes           # nix-collect-garbage, system-wide, never part of --all/--everything
```

| Flag | Removes |
|---|---|
| `--stale-models` | dirs in `models/` the registry does not use (old exports, experiments) |
| `--conversion-cache` | `.cache/hf`, plus `~/.cache/huggingface` copies of the two source models if you converted outside the flake shell |
| `--conversion-deps` | runs `uv sync`, which uninstalls the `convert` dependency group (the size shown is a lower bound) |
| `--ov-cache` | `.cache/ov`, the compiled-model cache (not in `--all`; stale blobs pile up after a model is re-exported, clear it then) |
| `--junk` | `__pycache__`, `.pytest_cache`, `*.egg-info`, stray root `.*.log` |
| `--venv` | the whole `.venv` |
| `--models` | converted models the registry loads |
| `--uv-cache` | `uv cache clean` (affects every uv project, which re-download as needed) |
| `--nix-gc` | `nix-collect-garbage` |
| `--all` | stale-models, conversion-cache, conversion-deps, junk |
| `--everything` | `--all` plus venv, models, ov-cache, uv-cache |

Safety: it refuses to touch anything outside the project (or the two HF cache dirs above), never touches `server/`,
`scripts/`, `tests/`, `docs/`, `.git`, `flake.*`, `pyproject.toml` or `uv.lock`, removes symlinks without following
them, and refuses to run while something answers on `127.0.0.1:11434` unless you pass `--force`.

## Slim runtime vs conversion

Conversion dependencies live in their own uv group, so serving never pulls torch:

```bash
uv sync                       # runtime + dev only (what you want day to day; also removes the convert group)
uv sync --group convert       # before running scripts.convert_translation_models
```

After the venv is recreated (`--venv`, or deleting `.venv` yourself), run `direnv reload` (or leave and re-enter
`nix develop`): the flake's shell hook is what links Intel's NPU compiler libraries into `openvino/libs`, and without
them the NPU plugin fails to compile models.

`--nix-gc` can remove the NPU compiler build and Python from the Nix store if nothing roots them. While `.direnv/`
exists the dev shell is rooted, so they stay; otherwise the next `nix develop` fetches them again (hash-pinned).
