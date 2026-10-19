"""CLI: uv run python -m server.main --model translategemma:4b --device NPU"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import uvicorn

from .app import create_app
from .registry import REGISTRY, ROOT

log = logging.getLogger("npu-translator")


def main() -> None:
    p = argparse.ArgumentParser(prog="server.main")
    p.add_argument("--model", action="append", choices=list(REGISTRY),
                   help="model to serve (repeatable); default: all. Only one is loaded at a time.")
    p.add_argument("--device", default="NPU", type=lambda v: [d.strip().upper() for d in v.split(",") if d.strip()],
                   help="NPU, GPU or CPU, or a comma list such as NPU,GPU: one replica of the model per device, all "
                        "fed from one queue (more RAM; measured slower than GPU alone, see README)")
    p.add_argument("--default-device", default="NPU", type=str.upper, choices=["NPU", "GPU", "CPU", "ANY"],
                   help="device for a plain model name (clients pick others as model@GPU); ANY = whichever is "
                        "idle. Default NPU; when it isn't in --device, the first --device is used")
    p.add_argument("--fallback", default="none", choices=["none", "gpu", "cpu", "all"],
                   help="devices to try when --device fails to compile (default: none, NPU only)")
    p.add_argument("--cache-dir", type=Path, default=ROOT / ".cache" / "ov", metavar="DIR",
                   help="compiled-model cache; a restart then skips the NPU compile (default: .cache/ov)")
    p.add_argument("--no-cache", action="store_true", help="always recompile (~40-70 s per model on the NPU)")
    p.add_argument("--host", default="127.0.0.1", help="no authentication: keep it on loopback")
    p.add_argument("--allow-origin", action="append", default=[], metavar="ORIGIN",
                   help="extra browser origin allowed to call the API, e.g. http://localhost:3000 (repeatable). "
                        "Browser extensions are always allowed; ordinary web pages are refused otherwise.")
    p.add_argument("--allow-host", action="append", default=[], metavar="HOST",
                   help="extra Host header value accepted besides localhost/127.0.0.1/[::1] (repeatable)")
    p.add_argument("--port", type=int, default=11434, help="default matches Ollama so the extension works unchanged")
    p.add_argument("--no-vision", action="store_true", help="accepted for compatibility; vision is always off")
    p.add_argument("--idle-unload", type=float, default=300.0, metavar="SECONDS",
                   help="free the model's memory after this long without a request; the next request reloads it "
                        "(seconds with the compile cache). 0 keeps it loaded. Default 300, like Ollama's keep_alive")
    p.add_argument("--preload", action="store_true", help="compile first model at startup instead of first request")
    p.add_argument("--stub", action="store_true", help="echo backend, no model (wiring test)")
    args = p.parse_args()

    if not args.device or any(d not in ("NPU", "GPU", "CPU") for d in args.device) or len(set(args.device)) != len(args.device):
        p.error("--device: use NPU, GPU or CPU, or a comma list without repeats, e.g. NPU,GPU")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    hosts = list(args.allow_host)
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        log.warning("listening on %s without authentication: anyone who can reach the port can run your NPU", args.host)
        if args.host not in ("0.0.0.0", "::"):
            hosts.append(args.host)
    default_device = None if args.default_device == "ANY" else (
        args.default_device if args.default_device in args.device else args.device[0]
    )
    cache_dir = None if args.no_cache or args.stub else args.cache_dir
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    app = create_app(args.model, args.device, args.stub, args.fallback, args.allow_origin, hosts, cache_dir, args.idle_unload, default_device)
    if args.preload:
        mgr = app.state.mgr
        mgr.preload(mgr.entry(mgr.served[0]))
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
