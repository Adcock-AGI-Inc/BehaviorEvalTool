"""A minimal Chrome DevTools Protocol client, standard library only: enough to attach to a
Chrome started with ``--remote-debugging-port``, open or pick a tab, navigate it and run
JavaScript in the page (``Runtime.evaluate`` with promises awaited). Local, unencrypted
WebSocket on 127.0.0.1; nothing here is general-purpose."""
from __future__ import annotations

import base64
import json
import os
import socket
import struct
import urllib.request
from urllib.parse import urlparse


def _http(port: int, path: str, method: str = "GET") -> object:
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method=method)
    with urllib.request.urlopen(req, timeout=5) as r:
        body = r.read()
    try:
        return json.loads(body)
    except ValueError:
        return body.decode(errors="replace")


def up(port: int) -> bool:
    try:
        return isinstance(_http(port, "/json/version"), dict)
    except Exception:
        return False


def targets(port: int) -> list[dict]:
    t = _http(port, "/json/list")
    return [x for x in t if isinstance(x, dict) and x.get("type") == "page"] if isinstance(t, list) else []


def new_target(port: int, url: str = "about:blank") -> dict:
    t = _http(port, f"/json/new?{url}", method="PUT")
    if not isinstance(t, dict):
        raise RuntimeError(f"Chrome did not open a tab: {str(t)[:120]}")
    return t


# ── websocket frames (client side: masked) ───────────────────────────────────

def encode_frame(payload: bytes, opcode: int = 1) -> bytes:
    head = bytes([0x80 | opcode])
    n = len(payload)
    if n < 126:
        head += bytes([0x80 | n])
    elif n < 65536:
        head += bytes([0x80 | 126]) + struct.pack(">H", n)
    else:
        head += bytes([0x80 | 127]) + struct.pack(">Q", n)
    mask = os.urandom(4)
    return head + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(payload))


def _read_exact(sock: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("websocket closed")
        buf += chunk
    return buf


def read_frame(sock: socket.socket) -> tuple[int, bytes]:
    """(opcode, payload) of the next complete message (continuation frames joined)."""
    payload, opcode = b"", None
    while True:
        b0, b1 = _read_exact(sock, 2)
        fin, op = b0 & 0x80, b0 & 0x0F
        masked, n = b1 & 0x80, b1 & 0x7F
        if n == 126:
            n = struct.unpack(">H", _read_exact(sock, 2))[0]
        elif n == 127:
            n = struct.unpack(">Q", _read_exact(sock, 8))[0]
        mask = _read_exact(sock, 4) if masked else b""
        data = _read_exact(sock, n)
        if mask:
            data = bytes(b ^ mask[i % 4] for i, b in enumerate(data))
        if opcode is None:
            opcode = op
        payload += data
        if fin:
            return opcode, payload


class Tab:
    """One page target over its WebSocket debugger URL."""

    def __init__(self, ws_url: str, timeout_s: float = 120):
        u = urlparse(ws_url)
        self.sock = socket.create_connection((u.hostname, u.port), timeout=timeout_s)
        key = base64.b64encode(os.urandom(16)).decode()
        self.sock.sendall((f"GET {u.path} HTTP/1.1\r\nHost: {u.hostname}:{u.port}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                           f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode())
        head = b""
        while b"\r\n\r\n" not in head:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("no websocket handshake")
            head += chunk
        if b" 101 " not in head.split(b"\r\n", 1)[0]:
            raise ConnectionError(f"websocket handshake refused: {head[:80]!r}")
        self.n = 0

    def call(self, method: str, **params) -> dict:
        self.n += 1
        self.sock.sendall(encode_frame(json.dumps({"id": self.n, "method": method, "params": params}).encode()))
        while True:
            op, data = read_frame(self.sock)
            if op == 8:
                raise ConnectionError("websocket closed by Chrome")
            if op == 9:   # ping → pong
                self.sock.sendall(encode_frame(data, 10))
                continue
            if op != 1:
                continue
            msg = json.loads(data)
            if msg.get("id") == self.n:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error'].get('message', msg['error'])}")
                return msg.get("result") or {}

    def js(self, expression: str):
        """Evaluate in the page; a promise is awaited; the value comes back by value."""
        r = self.call("Runtime.evaluate", expression=expression, awaitPromise=True, returnByValue=True)
        if r.get("exceptionDetails"):
            ex = r["exceptionDetails"]
            raise RuntimeError("page script failed: " + str((ex.get("exception") or {}).get("description") or ex.get("text") or ex)[:300])
        return (r.get("result") or {}).get("value")

    def navigate(self, url: str) -> None:
        self.call("Page.navigate", url=url)

    def front(self) -> None:
        try:
            self.call("Page.bringToFront")
        except Exception:
            pass

    def close(self) -> None:
        try:
            self.sock.close()
        except Exception:
            pass


def attach(port: int, prefer_url: str = "") -> Tab:
    """A tab to drive: one already on ``prefer_url`` when given, else a new blank one."""
    for t in targets(port):
        if prefer_url and str(t.get("url", "")).startswith(prefer_url) and t.get("webSocketDebuggerUrl"):
            return Tab(t["webSocketDebuggerUrl"])
    t = new_target(port)
    return Tab(t["webSocketDebuggerUrl"])
