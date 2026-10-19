"""Real uvicorn + raw socket: a client that vanishes mid-stream must stop generation and free the NPU lock.

Found live on the NPU: Starlette never closes a sync generator on disconnect and uvicorn drops writes to a dead
socket, so the first version kept generating (lock held) until max_tokens. TestClient cannot reproduce this.
"""

import json
import socket
import threading
import time

import uvicorn

from server import app as app_module
from server.backends import StubBackend


class SlowBackend(StubBackend):
    closed = threading.Event()

    def chat_stream(self, messages, max_new_tokens, temperature):
        try:
            for i in range(10_000):  # ~200 s if nobody stops us
                time.sleep(0.02)
                yield f"t{i} "
        finally:
            SlowBackend.closed.set()


def test_client_disconnect_stops_stream_and_frees_lock(monkeypatch):
    monkeypatch.setattr(app_module, "make_backend", lambda entry, *args, **kw: SlowBackend(entry))
    SlowBackend.closed.clear()
    app = app_module.create_app(stub=True)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time.time() + 10
        while not server.started and time.time() < deadline:
            time.sleep(0.02)
        port = server.servers[0].sockets[0].getsockname()[1]

        body = json.dumps({"model": "eurollm-1.7b-it", "stream": True,
                           "messages": [{"role": "user", "content": "story"}]}).encode()
        sock = socket.create_connection(("127.0.0.1", port))
        sock.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n"
                     b"Content-Length: %d\r\n\r\n" % len(body) + body)
        sock.settimeout(10)
        data = b""
        while b"t0" not in data:
            piece = sock.recv(4096)
            assert piece, "server closed the stream before the first token"
            data += piece
        assert app.state.mgr.busy == 1
        sock.close()  # client navigates away

        assert SlowBackend.closed.wait(5), "generation kept running after the client disconnected"
        deadline = time.time() + 2
        while app.state.mgr.busy and time.time() < deadline:
            time.sleep(0.02)
        assert app.state.mgr.busy == 0
    finally:
        server.should_exit = True
        thread.join(5)
