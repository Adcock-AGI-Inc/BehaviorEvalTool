"""Optional scoring of a finished conversation. A scorer takes the Conversation and the
(rendered) Scenario and returns a dict with at least ``verdict`` (PASS / FAIL / UNREADABLE / -)
and ``lines`` (what to print in the report). Implementations live under ``behavioreval/scorers``."""
from __future__ import annotations

import importlib
from typing import Protocol

from behavioreval.core.config import Config
from behavioreval.core.harness import Conversation
from behavioreval.core.scenario import Scenario

NAMES = {"judge": "behavioreval.scorers.judge", "none": "behavioreval.scorers.none"}


class Scorer(Protocol):
    def score(self, conv: Conversation, sc: Scenario) -> dict: ...


def make(name: str, cfg: Config, key: str = "", out=print, limits=None) -> Scorer:
    if name not in NAMES:
        raise ValueError(f"unknown scorer {name!r}; have: {', '.join(NAMES)}")
    return importlib.import_module(NAMES[name]).create(cfg, key, out, limits=limits)
