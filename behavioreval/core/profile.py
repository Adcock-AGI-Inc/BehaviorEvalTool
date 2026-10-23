"""A profile is the simulated person a tester runs as: facts (what ``{placeholders}`` in
scenarios draw on), a typing style, limits they never cross, the profile page the assistant
should know, and **the test accounts bound to them**. A bound account hosts only its own
profile, so one person's memory never leaks into another's conversations. One file per
profile under ``profiles/<id>.json``; a scenario may override facts with its own ``user`` block."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path

# What the assistant's profile page accepts (eval_system SeedProfile): anything else is refused.
PROFILE_FIELDS = ("name", "nickname", "phone", "home", "employers", "roles", "pets", "hobbies", "family")
SUGGESTED_FACTS = ("name", "age", "job", "home", "car", "household", "food", "travel", "entertainment", "health", "subscriptions", "habits", "routine", "calendar")
ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,40}$")


@dataclass
class Profile:
    id: str = "default"
    facts: dict = field(default_factory=dict)
    style: str = ""
    limits: list[str] = field(default_factory=list)
    profile: dict = field(default_factory=dict)      # written into the account's profile page before a conversation
    accounts: list[str] = field(default_factory=list)   # evalsystem endpoint: test accounts that belong to this person (passwords stay in accounts.tsv)
    email: str = ""                                   # chrome endpoint: the dev account this person is signed in as (informational, recorded on every conversation)
    chrome_profile: str = ""                          # chrome endpoint: the Chrome profile directory ("Default", "Profile 3") signed in as that account

    @property
    def name(self) -> str:
        return str(self.facts.get("name") or ("me" if self.id == "default" else self.id))

    def has(self, key: str) -> bool:
        v = self.facts.get(key)
        return v is not None and str(v).strip() != ""

    def brief(self) -> str:
        return "\n".join(f"- {k}: {v}" for k, v in self.facts.items() if v not in (None, "") and not (k == "name" and str(v).lower() == "me"))

    def with_overrides(self, overrides: dict | None) -> "Profile":
        """This person with a scenario's ``user`` block laid over the facts (``style`` / ``limits`` keys override those too)."""
        if not overrides:
            return self
        o = dict(overrides)
        style = str(o.pop("style", self.style) or self.style)
        limits = [str(x) for x in o.pop("limits", self.limits)] if "limits" in o else list(self.limits)
        return replace(self, facts={**self.facts, **o}, style=style, limits=limits)

    def to_dict(self) -> dict:
        return {"id": self.id, "facts": self.facts, "style": self.style, "limits": self.limits, "profile": self.profile, "accounts": self.accounts,
                "email": self.email, "chrome_profile": self.chrome_profile}


def from_dict(d: dict, where: str = "profile") -> Profile:
    if not isinstance(d, dict):
        raise ValueError(f"{where}: not an object")
    pid = str(d.get("id") or "").strip()
    if not ID.match(pid):
        raise ValueError(f"{where}: id must be lowercase letters, digits, - or _ (got {pid!r})")
    facts = d.get("facts") if isinstance(d.get("facts"), dict) else {}
    profile = {k: str(v) for k, v in (d.get("profile") or {}).items() if v not in (None, "")}
    bad = sorted(set(profile) - set(PROFILE_FIELDS))
    if bad:
        raise ValueError(f"{where}: profile has fields the assistant's profile page refuses: {', '.join(bad)} (allowed: {', '.join(PROFILE_FIELDS)})")
    accounts = d.get("accounts") or []
    if not isinstance(accounts, list) or any(not isinstance(a, str) or "@" not in a for a in accounts):
        raise ValueError(f"{where}: accounts must be a list of emails")
    return Profile(id=pid, facts=facts, style=str(d.get("style") or ""), limits=[str(x) for x in d.get("limits") or []], profile=profile,
                   accounts=[a.strip() for a in accounts], email=str(d.get("email") or "").strip(), chrome_profile=str(d.get("chrome_profile") or "").strip())


def load_file(path: Path) -> Profile:
    p = from_dict(json.loads(path.read_text(encoding="utf-8")), path.name)
    if p.id != path.stem:
        raise ValueError(f"{path.name}: id {p.id!r} must equal the file name")
    return p


def load_all(directory: Path) -> dict[str, Profile]:
    if not directory.exists():
        return {}
    return {p.stem: load_file(p) for p in sorted(directory.glob("*.json"))}


def load(directory: Path, pid: str) -> Profile:
    all_ = load_all(directory)
    if pid not in all_:
        raise ValueError(f"no profile {pid!r} under {directory} (have: {', '.join(sorted(all_)) or 'none'})")
    return all_[pid]


def save(directory: Path, p: Profile) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{p.id}.json"
    path.write_text(json.dumps(p.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def delete(directory: Path, pid: str) -> bool:
    path = directory / f"{pid}.json"
    if pid == "default" or not path.exists():
        return False
    path.unlink()
    return True


def bound_accounts(profiles: dict[str, Profile]) -> dict[str, str]:
    """email → the profile it belongs to."""
    return {a: p.id for p in profiles.values() for a in p.accounts}


def accounts_for(p: Profile, all_accounts: list[str], profiles: dict[str, Profile]) -> list[str]:
    """The accounts this profile may run on: its own, or — when it has none — every account no profile claims."""
    if p.accounts:
        return [a for a in p.accounts if a in all_accounts]
    taken = bound_accounts(profiles)
    return [a for a in all_accounts if a not in taken]


def lint(p: Profile, all_accounts: list[str], profiles: dict[str, Profile]) -> list[str]:
    """Why this profile cannot run as written. Empty = fine."""
    problems = []
    taken = bound_accounts({k: v for k, v in profiles.items() if k != p.id})
    for a in p.accounts:
        if all_accounts and a not in all_accounts:
            problems.append(f"account {a} is not in accounts.tsv")
        if a in taken:
            problems.append(f"account {a} already belongs to profile {taken[a]!r}")
    return problems
