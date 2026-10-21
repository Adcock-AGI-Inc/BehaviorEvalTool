"""Where the user's messages go. An endpoint opens one conversation as one account, takes a
message and returns the assistant's turn once it ended, and closes. Implementations live
under ``usersim/endpoints``; ``make`` loads one by name so the core never imports them."""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Protocol

from usersim.core.config import Config

NAMES = {"evalsystem": "usersim.endpoints.evalsystem", "openai_chat": "usersim.endpoints.openai_chat", "fake": "usersim.endpoints.fake"}


@dataclass
class AssistantTurn:
    reply: str = ""
    reason: str = ""                               # end_turn, awaiting_input, max_turns, error, ...; "*" suffix = inferred
    calls: list[dict] = field(default_factory=list)   # {name, input, result, is_error}
    duration_s: float = 0.0
    error: str = ""
    note: str = ""


class Endpoint(Protocol):
    def open(self, account: str, lane: str, title: str, profile: dict) -> str: ...
    def turn(self, text: str, timeout_s: float) -> AssistantTurn: ...
    def close(self) -> None: ...


def make(name: str, cfg: Config, out=print) -> Endpoint:
    if name not in NAMES:
        raise ValueError(f"unknown endpoint {name!r}; have: {', '.join(NAMES)}")
    mod = importlib.import_module(NAMES[name])
    return mod.create(name, cfg, out)
