"""Free disk space. Dry run unless --yes: shows what each step removes, its size and how to get it back.

  python3 -m scripts.cleanup                      # report only (stdlib: no venv needed, safe to run from anywhere)
  python3 -m scripts.cleanup --all --yes          # everything not needed to serve (keeps converted models + venv)
  python3 -m scripts.cleanup --everything --yes   # also venv, converted models, shared uv cache
  python3 -m scripts.cleanup --conversion-cache --yes   # or pick single steps (see --help)

Stop the server first. After removing .venv or running `uv sync`, re-enter the shell (`direnv reload`) so the flake
re-links the NPU compiler into the new venv.
"""

from __future__ import annotations

import argparse
import re
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from scripts.convert_translation_models import TARGETS
from server.registry import REGISTRY, ROOT

SKIP_WALK = {".venv", ".git", ".direnv", ".cache", "models", "node_modules"}
JUNK_DIRS = {"__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache"}
PROTECTED = {".git", "server", "scripts", "tests", "docs", "flake.nix", "flake.lock", "pyproject.toml", "uv.lock"}


def usage(path: Path) -> int:
    """Allocated bytes under path; hardlinked files (uv installs) are counted once."""
    seen: set[tuple[int, int]] = set()
    total = 0
    for p in [path, *path.rglob("*")] if path.is_dir() and not path.is_symlink() else [path]:
        try:
            st = p.lstat()
        except OSError:
            continue
        if p.is_symlink() or p.is_dir() or (st.st_dev, st.st_ino) in seen:
            continue
        seen.add((st.st_dev, st.st_ino))
        total += st.st_blocks * 512
    return total


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return ""


def remove(path: Path) -> None:
    home_hf = Path.home() / ".cache" / "huggingface"
    resolved = path.resolve() if not path.is_symlink() else path.parent.resolve() / path.name
    inside = resolved.is_relative_to(ROOT) or resolved.is_relative_to(home_hf)
    if not inside or resolved == ROOT or path.name in PROTECTED and path.parent == ROOT:
        raise RuntimeError(f"refusing to delete {path}")
    if path.is_symlink() or path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)


@dataclass
class Step:
    key: str
    title: str
    restore: str
    paths: Callable[[], list[Path]] = lambda: []
    run: Callable[[], None] | None = None  # for steps that are a command, not a path removal
    tool: str | None = None  # executable `run` needs
    estimate: Callable[[], int] | None = None
    groups: set[str] = field(default_factory=set)  # "all" and/or "everything"


def venv_site() -> Path | None:
    return next(iter((ROOT / ".venv" / "lib").glob("python3*/site-packages")), None)


def stale_models() -> list[Path]:
    keep = {e.path.resolve() for e in REGISTRY.values()}
    models = ROOT / "models"
    return [p for p in sorted(models.iterdir()) if p.resolve() not in keep] if models.is_dir() else []


def junk() -> list[Path]:
    found: list[Path] = []
    stack = [ROOT]
    while stack:
        for p in stack.pop().iterdir():
            if p.is_symlink():
                continue
            if p.is_dir():
                if p.name in JUNK_DIRS or p.name.endswith(".egg-info"):
                    found.append(p)
                elif p.name not in SKIP_WALK:
                    stack.append(p)
            elif p.parent == ROOT and re.fullmatch(r"\.[\w-]*\.log", p.name):
                found.append(p)
    return found


def conversion_cache() -> list[Path]:
    """HF checkpoints downloaded only to be converted. HF_HOME is the project's .cache/hf under the flake."""
    paths = [ROOT / ".cache" / "hf"]
    hub = Path.home() / ".cache" / "huggingface" / "hub"  # if you converted outside the flake shell
    paths += [hub / ("models--" + model_id.replace("/", "--")) for model_id in TARGETS]
    return [p for p in paths if p.exists()]


def convert_deps_plan() -> list[str]:
    """Packages a plain `uv sync` would uninstall (the `convert` group: torch, nncf, optimum-intel + dependents)."""
    if not shutil.which("uv") or not (ROOT / ".venv").is_dir():
        return []
    out = subprocess.run(["uv", "sync", "--dry-run"], cwd=ROOT, capture_output=True, text=True).stderr
    return re.findall(r"^\s*-\s+([\w.-]+)==", out, re.M)


def convert_deps_estimate() -> int:
    site = venv_site()
    if site is None:
        return 0
    dirs = {n.lower().replace("-", "_") for n in convert_deps_plan()}
    return sum(usage(p) for p in site.iterdir() if p.is_dir() and p.name.lower() in dirs)  # lower bound


def convert_deps_run() -> None:
    subprocess.run(["uv", "sync"], cwd=ROOT, check=True)


def run_cmd(*cmd: str) -> Callable[[], None]:
    return lambda: subprocess.run(cmd, check=True)


def uv_cache_size() -> int:
    if not shutil.which("uv"):
        return 0
    path = subprocess.run(["uv", "cache", "dir"], capture_output=True, text=True).stdout.strip()
    return usage(Path(path)) if path else 0


STEPS = [
    Step("stale-models", "Model dirs in models/ that the registry does not use (old exports, experiments)",
         "nothing to restore, they are unused", stale_models, groups={"all", "everything"}),
    Step("conversion-cache", "Original HF checkpoints downloaded only for conversion (.cache/hf)",
         "re-downloaded automatically on the next convert", conversion_cache, groups={"all", "everything"}),
    Step("conversion-deps", "torch, nncf, optimum-intel and dependents in .venv (convert group)",
         "uv sync --group convert", run=convert_deps_run, tool="uv", estimate=convert_deps_estimate, groups={"all"}),
    Step("ov-cache", "Compiled NPU blobs (.cache/ov); stale ones pile up after a model is re-exported",
         "next start recompiles (~40-70 s per model), then re-caches", lambda: [ROOT / ".cache" / "ov"] if (ROOT / ".cache" / "ov").exists() else [],
         groups={"everything"}),
    Step("junk", "__pycache__, .pytest_cache, *.egg-info, stray .*.log files",
         "regenerated on use", junk, groups={"all", "everything"}),
    Step("venv", "The whole .venv (everything, incl. runtime packages)",
         "uv sync, then `direnv reload`", lambda: [ROOT / ".venv"] if (ROOT / ".venv").exists() else [],
         groups={"everything"}),
    Step("models", "Converted models used by the registry (what the server loads)",
         "scripts.convert_translation_models (needs conversion deps + ~8 GB download again)",
         lambda: [e.path for e in REGISTRY.values() if e.path.exists()], groups={"everything"}),
    Step("uv-cache", "uv's download/wheel cache, shared by ALL your uv projects",
         "re-downloaded on demand", run=run_cmd("uv", "cache", "clean"), tool="uv", estimate=uv_cache_size, groups={"everything"}),
    Step("nix-gc", "Unreferenced Nix store paths, system-wide (nix-collect-garbage)",
         "re-fetched/rebuilt on demand (e.g. the NPU compiler)", run=run_cmd("nix-collect-garbage"), tool="nix-collect-garbage", groups=set()),
]


def server_running(port: int = 11434) -> bool:
    with socket.socket() as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def main() -> int:
    p = argparse.ArgumentParser(prog="scripts.cleanup", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--all", action="store_true", help="every step marked 'all' (keeps what is needed to serve)")
    p.add_argument("--everything", action="store_true", help="all + venv, converted models and the uv cache")
    for s in STEPS:
        p.add_argument(f"--{s.key}", action="store_true", help=s.title)
    p.add_argument("--yes", action="store_true", help="actually delete (default is a dry run)")
    p.add_argument("--force", action="store_true", help="proceed even if a server answers on :11434")
    args = p.parse_args()

    chosen = [
        s for s in STEPS
        if getattr(args, s.key.replace("-", "_")) or (args.all and "all" in s.groups)
        or (args.everything and ("all" in s.groups or "everything" in s.groups))
    ]
    if any(s.key == "venv" for s in chosen):  # nothing left to sync
        chosen = [s for s in chosen if s.key != "conversion-deps"]

    print(f"{'step':18} {'size':>10}  what / how to restore")
    total = 0
    for s in STEPS:
        paths = s.paths()
        size = s.estimate() if s.estimate else sum(usage(x) for x in paths)
        mark = "*" if s in chosen else " "
        print(f"{mark}{s.key:17} {human(size):>10}  {s.title}\n{'':29}restore: {s.restore}")
        for x in paths[:6]:
            print(f"{'':29}- {x.relative_to(ROOT) if x.is_relative_to(ROOT) else x}")
        if len(paths) > 6:
            print(f"{'':29}- ... {len(paths) - 6} more")
        total += size if s in chosen else 0
    if not chosen:
        print("\nnothing selected (* marks selected steps): pass --all, --everything or single step flags")
        return 0
    print(f"\nselected (*) steps free about {human(total)}")
    if not args.yes:
        print("dry run: nothing deleted. Re-run with --yes")
        return 0
    if server_running() and not args.force:
        print("a server is answering on 127.0.0.1:11434: stop it first (or --force)", file=sys.stderr)
        return 1

    for s in chosen:
        print(f"-> {s.key}")
        if s.run:
            if not shutil.which(s.tool):
                print(f"   {s.tool} not found, skipped")
                continue
            s.run()
        for path in s.paths():
            remove(path)
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
