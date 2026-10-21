"""Any assistant that speaks /v1/chat/completions (a bare model, a lane, an agent behind an
OpenAI-compatible front) — no account, no product. The whole transcript goes with every
message; a reply's tool_calls are recorded as calls with no result."""
from __future__ import annotations

import json
import os
import time
import urllib.request

from behavioreval.core.config import Config
from behavioreval.core.endpoint import AssistantTurn


class OpenAIChatEndpoint:
    def __init__(self, cfg: Config, out=print, system: str = ""):
        if not cfg.endpoint_url:
            raise ValueError("the openai_chat endpoint needs --endpoint-url (the /v1 base)")
        self.cfg, self.out = cfg, out
        self.url = cfg.endpoint_url.rstrip("/") + "/chat/completions"
        self.model = cfg.lane or "default"
        self.key = os.environ.get("BEHAVIOREVAL_ENDPOINT_KEY", "")
        self.messages: list[dict] = [{"role": "system", "content": system}] if system else []

    def open(self, account, lane, title, profile) -> str:
        if profile:
            self.messages.insert(0, {"role": "system", "content": "About the user: " + ", ".join(f"{k}: {v}" for k, v in profile.items())})
        return f"chat-{int(time.time())}"

    def turn(self, text: str, timeout_s: float) -> AssistantTurn:
        self.messages.append({"role": "user", "content": text})
        body = {"model": self.model, "messages": self.messages}
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json", **({"Authorization": "Bearer " + self.key} if self.key else {})})
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as r:
                resp = json.loads(r.read())
        except Exception as exc:
            raise TimeoutError(f"{type(exc).__name__}: {str(exc)[:200]}") from None
        msg = ((resp.get("choices") or [{}])[0].get("message") or {})
        reply = msg.get("content") or ""
        calls = [{"name": (c.get("function") or {}).get("name", "?"), "input": (c.get("function") or {}).get("arguments", ""), "result": "", "is_error": False}
                 for c in msg.get("tool_calls") or []]
        self.messages.append({"role": "assistant", "content": reply})
        return AssistantTurn(reply=reply, reason=str((resp.get("choices") or [{}])[0].get("finish_reason") or ""), calls=calls, duration_s=round(time.monotonic() - t0, 1))

    def close(self) -> None:
        pass


def create(name: str, cfg: Config, out=print):
    return OpenAIChatEndpoint(cfg, out)
