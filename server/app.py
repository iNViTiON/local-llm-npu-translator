"""FastAPI app: OpenAI-compatible /v1 plus an Ollama-compatible /api shim (what the llama-franca extension speaks)."""

from __future__ import annotations

import asyncio
import contextlib
import gc
import json
import logging
import math
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Iterator, Sequence

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from .backends import BackendError, ModelBackend, Stats, make_backend
from .prompts import parse_extension_prompt, system_prompt
from .registry import REGISTRY, ModelEntry, find_model
from .security import EXTENSION_ORIGIN, MAX_BODY_BYTES, OriginGuard

log = logging.getLogger("npu-translator")

MAX_INPUT_CHARS = 20_000  # all message text of one request; bounds time spent holding the NPU lock
MAX_NEW_TOKENS = 2048
LANG_CODE = re.compile(r"^[A-Za-z]{2,3}(?:-[A-Za-z]{2,8})?$")
ROLES = {"system", "user", "assistant"}


class HTTPError(Exception):
    def __init__(self, status: int, message: str):
        self.status, self.message = status, message


def get_param(body: dict, name: str, default: Any = None) -> Any:
    """Root-level field wins, then OpenAI-style `extra_body`."""
    if body.get(name) is not None:
        return body[name]
    extra = body.get("extra_body")
    if isinstance(extra, dict) and extra.get(name) is not None:
        return extra[name]
    return default


def lang_param(body: dict, name: str) -> str | None:
    value = get_param(body, name)
    if value is None:
        return None
    if not isinstance(value, str) or not LANG_CODE.match(value):
        raise HTTPError(400, f"{name} must be a language code such as 'et' or 'zh-Hans'")
    return value


def number(value: Any, name: str, cast, default, lo, hi):
    """Parse a client-supplied numeric option; garbage is a 400, out-of-range values are clamped."""
    if value is None:
        return default
    try:
        n = cast(value)
        if not math.isfinite(n):
            raise ValueError
    except (TypeError, ValueError, OverflowError):
        raise HTTPError(400, f"{name} must be a number") from None
    return min(max(n, lo), hi)


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict) and p.get("type") in ("image_url", "image"):
                raise HTTPError(400, "vision is disabled for this server")
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(p.get("text", ""))
        return "".join(parts)
    return ""


def normalize_messages(raw: Any) -> list[dict]:
    if not isinstance(raw, list) or not raw:
        raise HTTPError(400, "messages must be a non-empty list")
    out = []
    for m in raw:
        if not isinstance(m, dict) or m.get("role", "user") not in ROLES:
            raise HTTPError(400, "each message must be an object with role system, user or assistant")
        out.append({"role": m.get("role", "user"), "content": _text_of(m.get("content"))})
    if sum(len(m["content"]) for m in out) > MAX_INPUT_CHARS:
        raise HTTPError(413, f"input too long (max {MAX_INPUT_CHARS} characters)")
    return out


class Manager:
    """One loaded model at a time (NPU memory is tight), with one replica per device in `devices`.

    A request may name a device (`model@GPU`) or not (`model`: any device). Either way it takes whichever matching
    replica is idle and waits when all are busy. Replicas compile lazily, in parallel, the first time a request
    can use them. Asking for another model waits until every replica is idle, then rebuilds the pool.
    """

    def __init__(
        self, served: list[str], devices: Sequence[str], stub: bool = False, fallback: str = "none",
        cache_dir: Path | None = None, idle_unload: float = 0.0, default_device: str | None = None,
    ):
        self.served, self.devices = served, list(devices)
        # Where a plain `model` (no @DEVICE) runs; None = whichever replica is idle on any device.
        self.default_device = default_device if default_device in self.devices else None
        self.stub, self.fallback, self.cache_dir = stub, fallback, cache_dir
        self.cond = threading.Condition()
        self.current: ModelEntry | None = None
        self.seq = 0  # arrival tickets: a pending model switch only holds back requests that arrived after it
        self.want: ModelEntry | None = None
        self.want_ticket = 0
        self.backends: dict[str, ModelBackend] = {}
        self.state: dict[str, str] = {}  # per device: new | loading | free | busy | failed
        self.errors: dict[str, str] = {}
        self.idle_unload, self.last_used = idle_unload, time.monotonic()
        if idle_unload > 0:
            threading.Thread(target=self._reaper, daemon=True).start()

    def _reaper(self) -> None:
        """Free the model's memory after `idle_unload` seconds without a request; the next request reloads it."""
        while True:
            time.sleep(min(5.0, max(0.05, self.idle_unload / 4)))
            with self.cond:
                idle = time.monotonic() - self.last_used
                if self.current is None or self.busy or self.pending or idle < self.idle_unload:
                    continue
                log.info("unloading %s after %.0fs idle", self.current.short_name, idle)
                self.current, self.want = None, None
                self.backends, self.state, self.errors = {}, {}, {}
            gc.collect()  # drop the pipelines now
            try:
                import ctypes

                ctypes.CDLL("libc.so.6").malloc_trim(0)  # hand freed heap pages back to the OS
            except OSError:
                pass

    busy = property(lambda self: sum(v == "busy" for v in self.state.values()))
    pending = property(lambda self: sum(v == "loading" for v in self.state.values()))
    alive = property(lambda self: sum(v in ("free", "busy") for v in self.state.values()))

    # ---- names -------------------------------------------------------------

    def resolve(self, name: Any) -> tuple[ModelEntry, list[str] | None]:
        """`model@GPU` -> (entry, ["GPU"]); `model` -> (entry, [default device]), or None (any) without one."""
        if not isinstance(name, str):
            raise HTTPError(400, "model must be a string")
        devices = [self.default_device] if self.default_device else None
        base, sep, dev = name.rpartition("@")
        if sep:
            dev = dev.strip().upper()
            if dev not in self.devices:
                raise HTTPError(404, f"model '{name}' not found: device {dev} is not served (served: {', '.join(self.devices)})")
            name, devices = base, [dev]
        entry = find_model(name)
        if entry is None or entry.short_name not in self.served:
            raise HTTPError(404, f"model '{name}' not found (served: {', '.join(self.served)})")
        return entry, devices

    def entry(self, name: Any) -> ModelEntry:
        return self.resolve(name)[0]

    def served_names(self) -> list[str]:
        """Each model, plus one `model@DEVICE` option per device when more than one device is served."""
        return [n for m in self.served for n in [m, *(f"{m}@{d}" for d in self.devices if len(self.devices) > 1)]]

    def loaded_names(self) -> list[str]:
        if self.current is None or not self.alive:
            return []
        name = self.current.short_name
        up = [d for d in self.devices if self.state[d] in ("free", "busy")]
        # A plain name is loaded once the device it runs on is (any device, without a default).
        plain = [name] if not self.default_device or self.default_device in up else []
        return plain + [f"{name}@{d}" for d in up if len(self.devices) > 1]

    # ---- replica pool ------------------------------------------------------

    def _switch(self, entry: ModelEntry) -> None:
        """Caller holds self.cond and nothing is busy or compiling."""
        self.current, self.want, self.last_used = entry, None, time.monotonic()
        self.backends, self.state, self.errors = {}, {}, {}
        for d in self.devices:
            be = make_backend(entry, d, self.stub, self.fallback, self.cache_dir)
            be.slot = d
            self.backends[d], self.state[d] = be, "new"

    def _start(self, device: str) -> None:
        self.state[device] = "loading"
        threading.Thread(target=self._load, args=(device,), daemon=True).start()

    def _load(self, device: str) -> None:
        err = None
        try:
            self.backends[device].load()
        except Exception as e:  # BackendError, or whatever the runtime raised
            err = str(e)
            log.warning("replica on %s failed: %s", device, err)
        with self.cond:
            self.state[device] = "failed" if err else "free"
            if err:
                self.errors[device] = err
            self.cond.notify_all()

    def acquire(self, entry: ModelEntry, devices: Sequence[str] | None = None) -> ModelBackend:
        """Blocks until a replica of `entry` on one of `devices` (default: any) is idle. Pair with release()."""
        t0 = time.perf_counter()
        with self.cond:
            self.seq += 1
            ticket = self.seq
            while True:
                if self.current is not entry:
                    if self.busy == 0 and self.pending == 0:
                        self._switch(entry)
                    else:
                        if self.want is None or ticket < self.want_ticket:
                            self.want, self.want_ticket = entry, ticket
                        self.cond.wait()
                    continue
                if self.want is not None and self.want is not entry and self.want_ticket < ticket:
                    self.cond.wait()  # an earlier request for another model is waiting for the pool to drain
                    continue
                wanted = list(devices or self.devices)
                for d in wanted:
                    if self.state[d] == "new":
                        self._start(d)
                free = [d for d in wanted if self.state[d] == "free"]
                if free:
                    self.state[free[0]] = "busy"
                    be = self.backends[free[0]]
                    be.queued = time.perf_counter() - t0  # time spent waiting for a free replica (shown in the log)
                    self.last_used = time.monotonic()
                    return be
                if all(self.state[d] == "failed" for d in wanted):
                    err = "; ".join(self.errors.get(d, "?") for d in wanted)
                    for d in wanted:  # forget the failure so the next request tries again (e.g. model converted since)
                        self.state[d] = "new"
                    raise BackendError(err)
                self.cond.wait()

    def release(self, be: ModelBackend) -> None:
        with self.cond:
            self.state[be.slot] = "free"
            self.last_used = time.monotonic()
            self.cond.notify_all()

    @contextlib.contextmanager
    def lease(self, entry: ModelEntry, devices: Sequence[str] | None = None) -> Iterator[ModelBackend]:
        be = self.acquire(entry, devices)
        try:
            yield be
        finally:
            self.release(be)

    def preload(self, entry: ModelEntry, devices: Sequence[str] | None = None) -> None:
        """Returns once every wanted replica has finished compiling (or failed)."""
        self.release(self.acquire(entry, devices))
        wanted = list(devices or self.devices)
        with self.cond:
            while any(self.state[d] == "loading" for d in wanted):
                self.cond.wait()


def create_app(
    served: list[str] | None = None,
    device: str | Sequence[str] = "NPU",
    stub: bool = False,
    fallback: str = "none",
    origins: Sequence[str] = (),
    hosts: Sequence[str] = (),
    cache_dir: Path | None = None,
    idle_unload: float = 0.0,
    default_device: str | None = None,
) -> FastAPI:
    devices = [device] if isinstance(device, str) else device
    mgr = Manager(served or list(REGISTRY), devices, stub, fallback, cache_dir, idle_unload, default_device)
    app = FastAPI(title="npu-translator")
    app.state.mgr = mgr
    # No auth: only browser extensions (plus --allow-origin) may call us, and only under a loopback Host.
    # Added last = outermost, so CORS answers preflights and OriginGuard refuses everything else.
    app.add_middleware(OriginGuard, origins=origins, hosts=hosts)
    app.add_middleware(
        CORSMiddleware, allow_origins=list(origins), allow_origin_regex=EXTENSION_ORIGIN,
        allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["*"],
    )

    @app.exception_handler(HTTPError)
    async def _http_error(_: Request, e: HTTPError):
        return JSONResponse({"error": e.message}, status_code=e.status)

    @app.exception_handler(BackendError)
    async def _backend_error(_: Request, e: BackendError):
        return JSONResponse({"error": str(e)}, status_code=400)

    async def body_of(request: Request) -> dict:
        chunks, size = [], 0
        async for chunk in request.stream():  # the extension sends no Content-Type, so parse JSON ourselves
            size += len(chunk)
            if size > MAX_BODY_BYTES:
                raise HTTPError(413, "request body too large")
            chunks.append(chunk)
        try:
            body = json.loads(b"".join(chunks))
        except ValueError:
            raise HTTPError(400, "invalid JSON body") from None
        if not isinstance(body, dict):
            raise HTTPError(400, "JSON body must be an object")
        return body

    # ---- shared inference -------------------------------------------------

    def resolve(entry: ModelEntry, messages: list[dict], body: dict, parsed=None):
        """Returns (kind, payload): ("translate", (text, src, tgt)) or ("chat", messages)."""
        src, tgt = lang_param(body, "src_lang"), lang_param(body, "tgt_lang")
        if parsed is not None:
            src, tgt = src or parsed.src, tgt or parsed.tgt
        if entry.backend == "openvino_seq2seq":
            if not tgt:
                raise HTTPError(400, "tgt_lang is required for this model")
            if not src:
                raise HTTPError(400, "src_lang is required for this model")
            text = parsed.text if parsed else next(
                (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
            )
            return "translate", (text, src, tgt)
        if parsed is not None:
            return "translate", (parsed.text, src, tgt)
        if not src and not tgt:
            log.info("request to %s without src_lang/tgt_lang hints", entry.short_name)
        if entry.backend == "openvino" and messages[0]["role"] != "system":  # TranslateGemma has its own template
            messages = [{"role": "system", "content": system_prompt(src, tgt)}, *messages]
        return "chat", messages

    def max_tokens_of(text_len: int, requested: Any) -> int:
        return number(requested or None, "max tokens", int, min(text_len * 2 + 64, MAX_NEW_TOKENS), 1, MAX_NEW_TOKENS)

    def run(backend: ModelBackend, kind: str, payload, max_new: int, temperature: float) -> tuple[str, Stats | None]:
        """Caller leased `backend`, so its last_stats belongs to this request."""
        if kind == "translate":
            text, src, tgt = payload
            out = backend.translate(text, src, tgt or "en", max_new)
        else:
            out = backend.chat(payload, max_new, temperature)
        return out, report(backend)

    def report(backend: ModelBackend) -> Stats | None:
        stats = backend.last_stats
        if stats is not None:  # numbers only: request text is never logged
            log.info("%s on %s: %d tokens in %.2fs = %.1f tok/s (prompt %d tokens in %.2fs), queued %.2fs",
                     backend.entry.short_name, getattr(backend, "active_device", "?"), stats.new_tokens,
                     stats.decode_seconds, stats.tokens_per_second, stats.prompt_tokens, stats.prompt_seconds,
                     getattr(backend, "queued", 0.0))
        return stats

    def ollama_timing(stats: Stats | None) -> dict:
        """Ollama's own fields (durations in ns); eval_count / eval_duration = decode tokens per second."""
        if stats is None:
            return {}
        return {
            "prompt_eval_count": stats.prompt_tokens, "prompt_eval_duration": int(stats.prompt_seconds * 1e9),
            "eval_count": stats.new_tokens, "eval_duration": int(stats.eval_seconds * 1e9),
        }

    def openai_timing(stats: Stats | None) -> dict:
        if stats is None:
            return {"usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}
        return {
            "usage": {"prompt_tokens": stats.prompt_tokens, "completion_tokens": stats.new_tokens,
                      "total_tokens": stats.prompt_tokens + stats.new_tokens},
            "timings": {"prompt_ms": stats.prompt_seconds * 1e3, "predicted_ms": stats.decode_seconds * 1e3,
                        "predicted_per_second": stats.tokens_per_second},  # llama.cpp server's names
        }

    # ---- discovery ---------------------------------------------------------

    @app.get("/")
    def root():
        return {
            "status": "ok", "models": mgr.served, "device": ",".join(mgr.devices), "devices": mgr.devices,
            "default_device": mgr.default_device,
        }

    @app.get("/v1/models")
    def v1_models():
        data = []
        for name in mgr.served_names():
            base, _, dev = name.partition("@")
            e = REGISTRY[base]
            data.append({
                "id": name, "object": "model", "created": 0, "owned_by": e.owned_by,
                **({"device": dev} if dev else {}),
                "capabilities": {
                    "translation": True, "chat": e.chat, "languages_supported": e.languages_supported,
                },
            })
        return {"object": "list", "data": data}

    @app.get("/api/tags")
    def api_tags():
        return {"models": [{"name": n, "model": n} for n in mgr.served_names()]}

    @app.get("/api/ps")
    def api_ps():
        return {"models": [{"name": n, "model": n} for n in mgr.loaded_names()]}

    @app.post("/api/generate")
    async def api_generate(request: Request):
        """Extension calls this with no prompt to preload. Compiles the model (blocks until ready)."""
        body = await body_of(request)
        entry, devs = mgr.resolve(body.get("model", ""))
        def load() -> float:
            t0 = time.perf_counter()
            mgr.preload(entry, devs)
            return time.perf_counter() - t0

        try:
            seconds = await run_in_threadpool(load)
        except BackendError as e:
            raise HTTPError(400, str(e)) from None
        return {"model": entry.short_name, "response": "", "done": True, "load_duration": int(seconds * 1e9)}

    # ---- Ollama chat shim --------------------------------------------------

    async def ndjson_chat(request: Request, entry: ModelEntry, devs, kind: str, payload, max_new: int,
                          temperature: float) -> AsyncIterator[str]:
        """Ollama's streaming /api/chat: one JSON object per line, the last with done=true and the timings. Same
        disconnect rules as the SSE stream below."""
        t0 = time.time()
        be = pieces = stats = None
        acquiring = asyncio.ensure_future(run_in_threadpool(mgr.acquire, entry, devs))

        def line(content: str, done: bool = False, extra: dict | None = None) -> str:
            obj = {
                "model": entry.short_name, "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "message": {"role": "assistant", "content": content}, "done": done, **(extra or {}),
            }
            return json.dumps(obj, ensure_ascii=False) + "\n"

        def finish() -> None:
            try:
                if pieces is not None:
                    pieces.close()  # stops and joins the generation thread
            finally:
                mgr.release(be)

        try:
            be = await acquiring
            if kind == "translate" and hasattr(be, "translate_stream"):
                text, src, tgt = payload
                pieces = be.translate_stream(text, src, tgt or "en", max_new)
            elif kind == "chat":
                pieces = be.chat_stream(payload, max_new, temperature)
            if pieces is not None:
                while (piece := await run_in_threadpool(next, pieces, None)) is not None:
                    if await request.is_disconnected():
                        return
                    yield line(piece)
                stats = report(be)  # set once the stream generator is exhausted
            else:  # seq2seq: no incremental callback, emit the whole result
                text, stats = await run_in_threadpool(run, be, kind, payload, max_new, temperature)
                yield line(text)
        except BackendError as e:
            yield json.dumps({"error": str(e)}, ensure_ascii=False) + "\n"
            return
        finally:
            if be is not None:
                await asyncio.shield(asyncio.get_running_loop().run_in_executor(None, finish))
            else:  # cancelled while queued: hand the replica back if the acquire still completes
                acquiring.add_done_callback(
                    lambda t: not t.cancelled() and t.exception() is None and mgr.release(t.result())
                )
        yield line("", True, {"done_reason": "stop", "total_duration": int((time.time() - t0) * 1e9),
                              **ollama_timing(stats)})

    @app.post("/api/chat")
    async def api_chat(request: Request):
        body = await body_of(request)
        entry, devs = mgr.resolve(body.get("model", ""))
        messages = normalize_messages(body.get("messages"))
        opts = body.get("options") or {}
        if not isinstance(opts, dict):
            raise HTTPError(400, "options must be an object")
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        parsed = parse_extension_prompt(last_user)
        if parsed is None and entry.backend == "openvino_seq2seq":
            raise HTTPError(400, "custom prompt not supported by translation-only model; use the default prompt")
        kind, payload = resolve(entry, messages, body, parsed)
        max_new = max_tokens_of(len(parsed.text if parsed else last_user), opts.get("num_predict"))
        temperature = number(opts.get("temperature"), "temperature", float, 0.0, 0.0, 2.0)

        if body.get("stream"):
            return StreamingResponse(ndjson_chat(request, entry, devs, kind, payload, max_new, temperature),
                                     media_type="application/x-ndjson")

        def work() -> tuple[str, Stats | None]:
            with mgr.lease(entry, devs) as be:
                return run(be, kind, payload, max_new, temperature)

        t0 = time.time()
        try:
            content, stats = await run_in_threadpool(work)
        except BackendError as e:
            raise HTTPError(400, str(e)) from None
        return {
            "model": entry.short_name,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "message": {"role": "assistant", "content": content},
            "done": True,
            "total_duration": int((time.time() - t0) * 1e9),
            **ollama_timing(stats),
        }

    # ---- OpenAI chat -------------------------------------------------------

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await body_of(request)
        entry, devs = mgr.resolve(body.get("model", ""))
        messages = normalize_messages(body.get("messages"))
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        kind, payload = resolve(entry, messages, body)
        max_new = max_tokens_of(len(last_user), body.get("max_tokens") or body.get("max_completion_tokens"))
        temperature = number(body.get("temperature"), "temperature", float, 0.0, 0.0, 2.0)
        cid, created = f"chatcmpl-{uuid.uuid4().hex[:24]}", int(time.time())

        def chunk(delta: dict, finish: str | None = None, extra: dict | None = None) -> str:
            obj = {
                "id": cid, "object": "chat.completion.chunk", "created": created, "model": entry.short_name,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}], **(extra or {}),
            }
            return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

        if body.get("stream"):
            # Async on purpose: Starlette never closes a sync generator on client disconnect and uvicorn drops
            # writes to a dead socket, so only request.is_disconnected() reveals it. The lock must not be released
            # while the generation thread may still be running on the pipeline.
            async def sse() -> AsyncIterator[str]:
                be = pieces = stats = None
                acquiring = asyncio.ensure_future(run_in_threadpool(mgr.acquire, entry, devs))

                def finish() -> None:
                    # Worker-thread job, shielded below: is_disconnected() can leave a stray cancellation that hits
                    # the next await, and stopping the stream, then freeing the replica, must happen regardless.
                    try:
                        if pieces is not None:
                            pieces.close()  # stops and joins the generation thread
                    finally:
                        mgr.release(be)

                try:
                    be = await acquiring
                    yield chunk({"role": "assistant"})
                    if kind == "chat":
                        pieces = be.chat_stream(payload, max_new, temperature)
                        while (piece := await run_in_threadpool(next, pieces, None)) is not None:
                            if await request.is_disconnected():
                                return
                            yield chunk({"content": piece})
                        stats = report(be)  # set once the stream generator is exhausted
                    else:  # seq2seq: no incremental callback, emit whole result
                        text, stats = await run_in_threadpool(run, be, kind, payload, max_new, temperature)
                        yield chunk({"content": text})
                except BackendError as e:
                    yield f"data: {json.dumps({'error': {'message': str(e)}})}\n\n"
                    return
                finally:
                    if be is not None:
                        await asyncio.shield(asyncio.get_running_loop().run_in_executor(None, finish))
                    else:  # cancelled while queued: hand the replica back if the acquire still completes
                        acquiring.add_done_callback(
                            lambda t: not t.cancelled() and t.exception() is None and mgr.release(t.result())
                        )
                yield chunk({}, "stop", openai_timing(stats))
                yield "data: [DONE]\n\n"

            return StreamingResponse(sse(), media_type="text/event-stream")

        def work() -> tuple[str, Stats | None]:
            with mgr.lease(entry, devs) as be:
                return run(be, kind, payload, max_new, temperature)

        try:
            content, stats = await run_in_threadpool(work)
        except BackendError as e:
            raise HTTPError(400, str(e)) from None
        return {
            "id": cid, "object": "chat.completion", "created": created, "model": entry.short_name,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            **openai_timing(stats),
        }

    return app
