"""The simulated user: facts (what ``{placeholders}`` in scenarios draw on), a typing style,
limits they never cross, and the profile the assistant should know about them. One default
user lives in ``user.json``; a scenario may override facts with its own ``user`` block."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

# What the assistant's profile page accepts (eval_system SeedProfile): anything else is refused.
PROFILE_FIELDS = ("name", "nickname", "phone", "home", "employers", "roles", "pets", "hobbies", "family")


@dataclass
class User:
    facts: dict = field(default_factory=dict)
    style: str = ""
    limits: list[str] = field(default_factory=list)
    profile: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return str(self.facts.get("name") or "the user")

    def has(self, key: str) -> bool:
        v = self.facts.get(key)
        return v is not None and str(v).strip() != ""

    def brief(self) -> str:
        return "\n".join(f"- {k}: {v}" for k, v in self.facts.items() if v not in (None, ""))

    def with_overrides(self, overrides: dict | None) -> "User":
        """This user with a scenario's ``user`` block laid over the facts (``style`` / ``limits`` keys override those too)."""
        if not overrides:
            return self
        o = dict(overrides)
        style = str(o.pop("style", self.style) or self.style)
        limits = [str(x) for x in o.pop("limits", self.limits)] if "limits" in o else list(self.limits)
        return replace(self, facts={**self.facts, **o}, style=style, limits=limits)


def from_dict(d: dict, where: str = "user.json") -> User:
    if not isinstance(d, dict):
        raise ValueError(f"{where}: not an object")
    facts = d.get("facts") if isinstance(d.get("facts"), dict) else {}
    if not facts.get("name"):
        raise ValueError(f"{where}: needs facts.name")
    profile = {k: str(v) for k, v in (d.get("profile") or {}).items() if v not in (None, "")}
    bad = sorted(set(profile) - set(PROFILE_FIELDS))
    if bad:
        raise ValueError(f"{where}: profile has fields the assistant's profile page refuses: {', '.join(bad)} (allowed: {', '.join(PROFILE_FIELDS)})")
    return User(facts=facts, style=str(d.get("style") or ""), limits=[str(x) for x in d.get("limits") or []], profile=profile)


def load(path: Path) -> User:
    if not path.exists():
        raise ValueError(f"no user file {path}")
    return from_dict(json.loads(path.read_text(encoding="utf-8")), path.name)
