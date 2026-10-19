"""Export translation models to OpenVINO IR with NPU-friendly INT4.

  uv sync --group convert        # export-time deps (torch, optimum-intel, nncf); `uv sync` removes them again
  uv run python -m scripts.convert_translation_models --model-id google/translategemma-4b-it --quant int4
    (gated: accept the Gemma license on the model page, then `hf auth login`; ~8.6 GB download into .cache/hf)
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys

from server.registry import ROOT

TARGETS = {
    # Gated: accept the Gemma license on huggingface.co/google/translategemma-4b-it and log in (`hf auth login`) first.
    "google/translategemma-4b-it": ("image-text-to-text", "translategemma-4b-it"),
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", required=True, choices=list(TARGETS))
    p.add_argument("--quant", default="int4", choices=["int4", "int8", "fp16"])
    args = p.parse_args()
    if shutil.which("optimum-cli") is None:
        sys.exit("optimum-cli not found: run `uv sync --group convert` first (export-time dependencies)")

    task, stem = TARGETS[args.model_id]
    out = ROOT / "models" / f"{stem}-{args.quant}-sym-ov"
    cmd = [
        "optimum-cli", "export", "openvino", "--model", args.model_id, "--task", task,
        "--weight-format", args.quant,
    ]
    if args.quant == "int4":
        # NPU 3720/4000 cannot dequantise asymmetric INT4; int8_sym backs off layers that can't take 4-bit.
        cmd += ["--sym", "--group-size", "128", "--backup-precision", "int8_sym", "--ratio", "1.0"]
    cmd.append(str(out))
    print("+", " ".join(cmd), flush=True)
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
