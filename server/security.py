"""Request policy. The server has no auth, so it must refuse anything a web page could send."""

from __future__ import annotations

import re
from typing import Iterable

from starlette.responses import JSONResponse

# Browser extensions only. Ordinary pages (http/https, `null`, file://) are refused.
EXTENSION_ORIGIN = r"^(chrome|moz|safari-web)-extension://[\w.-]+$"
LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "[::1]"})
MAX_BODY_BYTES = 1 << 20


def host_of(header: str) -> str:
    """Host header without its port (IPv6 literals keep their brackets)."""
    header = header.strip().lower()
    return header.split("]")[0] + "]" if header.startswith("[") else header.split(":")[0]


class OriginGuard:
    """Pure-ASGI middleware, installed inside CORSMiddleware so refusals carry no CORS headers.

    A web page can send a "simple" POST (text/plain body, no preflight) to localhost, and this API parses JSON
    whatever the Content-Type is, so CORS alone does not stop a site from running inference on your NPU. Browsers
    always set `Origin` on cross-origin POSTs and pages cannot forge it, so refuse by Origin; refuse by Host too,
    against DNS rebinding. Requests without Origin (curl, SDKs) pass.
    """

    def __init__(self, app, origins: Iterable[str] = (), hosts: Iterable[str] = ()):
        self.app = app
        self.origin_re = re.compile(EXTENSION_ORIGIN)
        self.origins = frozenset(origins)
        self.hosts = LOOPBACK_HOSTS | {h.lower() for h in hosts}

    def refusal(self, headers: dict[str, str]) -> str | None:
        host = headers.get("host", "")
        if host_of(host) not in self.hosts:
            return f"host '{host}' not allowed (DNS-rebinding protection); see --allow-host"
        origin = headers.get("origin")
        if origin is not None and origin not in self.origins and not self.origin_re.match(origin):
            return f"origin '{origin}' not allowed: only browser extensions may use this server; see --allow-origin"
        return None

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]}
            if (why := self.refusal(headers)) is not None:
                await JSONResponse({"error": why}, status_code=403)(scope, receive, send)
                return
        await self.app(scope, receive, send)
