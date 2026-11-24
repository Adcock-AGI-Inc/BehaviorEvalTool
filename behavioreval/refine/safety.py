"""The halt check: stop the whole run when Hark reaches a pay, book or send step.

The simulated user never approves those steps, and the test account has no saved card. This is the
backstop: after every conversation, its tool calls are matched against the halt rules in refine.json.
A match stops the run and is logged as "would have completed it". A reply that only claims such a step
is a warning, not a halt (a false halt would end the night), and is listed in the report.
Only the main agent's calls are visible (the trajectory viewer export
carries no browser-subagent steps and no tool results), so a purchase inside a browser subagent is
caught only by the browser instruction or by Hark's own reply."""
from __future__ import annotations

import json
import re

CLAIMS = re.compile(r"(?i)\bI(?:'ve| have)? (?:just )?(?:placed (?:the|your) order|completed (?:the|your) (?:purchase|checkout|booking)|"
                    r"booked (?:it|the|your)|paid (?:for|the)|sent (?:the|your|it to|an? (?:email|text|message) to))")


class Halt(RuntimeError):
    """Hark reached a pay, book or send step. Carries the evidence."""


def _input_text(call: dict) -> str:
    raw = call.get("input")
    return raw if isinstance(raw, str) else json.dumps(raw or {}, ensure_ascii=False)


def check(conv, rules: list[dict]) -> tuple[list[str], list[str]]:
    """(halts, warnings): tool calls that stop the run, and replies that claim a completed step."""
    hits, claims = [], []
    for t in conv.turns:
        for c in t.calls or []:
            name = str(c.get("name") or "")
            for r in rules:
                if re.search(r["tool"], name) and (not r.get("input") or re.search(r["input"], _input_text(c))):
                    hits.append(f"turn {t.n}: tool `{name}` ({r.get('why', 'halt rule')}): {_input_text(c)[:200]}")
                    break
        m = CLAIMS.search(t.assistant or "")
        if m:
            claims.append(f"turn {t.n}: Hark's reply claims a completed step: {m.group(0)!r}")
    return hits, claims
