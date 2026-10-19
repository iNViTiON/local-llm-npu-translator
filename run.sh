#!/usr/bin/env bash
# Start the translation server inside the flake dev shell (NPU libs, compiler links, uv env).
# Flags pass straight through:  ./run.sh   ./run.sh --device GPU   ./run.sh --device NPU,GPU --preload
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
[ -d .venv ] || nix develop -c uv sync   # first run only; the next nix develop links the NPU compiler into the new venv
exec nix develop -c uv run --no-sync python -m server.main "$@"
