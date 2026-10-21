"""A scripted assistant for tests and dry runs: one (reply, reason) per user message."""
from __future__ import annotations

from usersim.core.endpoint import AssistantTurn


class FakeEndpoint:
    def __init__(self, turns=None, calls=None):
        self.turns = list(turns or [])
        self.calls = calls or []
        self.sent: list[str] = []
        self.opened = None
        self.closed = False

    def open(self, account, lane, title, profile) -> str:
        self.opened = {"account": account, "lane": lane, "title": title, "profile": profile}
        return "conv-fake-0001"

    def turn(self, text, timeout_s) -> AssistantTurn:
        self.sent.append(text)
        if not self.turns:
            raise TimeoutError("the scripted assistant has no more turns")
        reply, reason = self.turns.pop(0)
        return AssistantTurn(reply=reply, reason=reason, calls=list(self.calls), duration_s=1.0)

    def close(self) -> None:
        self.closed = True


def create(name, cfg, out=print):
    return FakeEndpoint([("(fake assistant) tell me more about that", "awaiting_input")] * cfg.max_turns,
                        calls=[{"name": "web_search", "input": {"query": "example"}, "result": "{\"results\": []}", "is_error": False}])
