"""What our own model calls cost, and the caps that stop a run.

Every OpenRouter call in refine mode goes through ``Meter.post``, which asks OpenRouter to include the
call's cost (``usage.include``) and adds it up per role (user, judge, rubric, pairwise, diagnose, editor).
Hark's own model cost on dev is not visible to this tool; ``max_conversations`` is the cap on that side."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field

from behavioreval.core import openrouter


class BudgetExceeded(RuntimeError):
    """Raised before a call or a conversation that would go past a cap. The loop saves its state and stops."""


@dataclass
class Meter:
    max_cost_usd: float
    max_conversations: int
    spent_usd: float = 0.0
    calls_without_cost: int = 0
    conversations: int = 0
    by_role: dict = field(default_factory=dict)
    send: object = openrouter.post
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def post_for(self, role: str):
        """A drop-in for ``openrouter.post`` that records the call's cost under ``role``."""
        def post(body: dict, key: str, timeout_s: float = 240) -> dict:
            self.check_cost()
            resp = self.send({**body, "usage": {"include": True}}, key, timeout_s=timeout_s)
            usage = resp.get("usage") or {}
            with self._lock:
                cost = usage.get("cost")
                if isinstance(cost, (int, float)):
                    self.spent_usd += float(cost)
                    self.by_role[role] = self.by_role.get(role, 0.0) + float(cost)
                else:
                    self.calls_without_cost += 1
            return resp
        return post

    def check_cost(self) -> None:
        if self.spent_usd >= self.max_cost_usd:
            raise BudgetExceeded(f"spend cap reached: ${self.spent_usd:.2f} of ${self.max_cost_usd:.2f}")

    def start_conversation(self) -> None:
        with self._lock:
            if self.conversations >= self.max_conversations:
                raise BudgetExceeded(f"conversation cap reached: {self.conversations} of {self.max_conversations}")
            self.conversations += 1
        self.check_cost()

    def to_dict(self) -> dict:
        return {"spent_usd": round(self.spent_usd, 4), "conversations": self.conversations, "calls_without_cost": self.calls_without_cost,
                "by_role": {k: round(v, 4) for k, v in self.by_role.items()}}

    def restore(self, d: dict) -> None:
        self.spent_usd = float(d.get("spent_usd") or 0)
        self.conversations = int(d.get("conversations") or 0)
        self.calls_without_cost = int(d.get("calls_without_cost") or 0)
        self.by_role = dict(d.get("by_role") or {})
