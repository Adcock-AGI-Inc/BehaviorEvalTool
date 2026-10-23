"""A scenario is one task: the opening sent verbatim, what is going on, what the person wants,
what they hold back until asked, when they are done. Facts come from the user through
``{placeholders}`` (``{home}``, ``{car}`` …) so the same task can be lived by any user;
a scenario may add or override facts in its own ``user`` block. ``lint`` checks that every
placeholder resolves."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from behavioreval.core.profile import Profile

PLACEHOLDER = re.compile(r"(?<!\{)\{([a-z_][a-z0-9_]*)\}(?!\})")   # {home}; not {{today}} (a judge placeholder)
DEFAULT_REVEAL = "only when the assistant asks, or when it clearly cannot proceed without it; one fact per reply"
RUNTIME_PREFIX = "prev_"   # placeholders the harness fills at run time (cleanup scenarios)

FIELD_HELP = {   # what each field means, for the UI form and the README
    "id": "unique name, <set>/<number>_<slug>, e.g. tests/03_dry_cleaner",
    "title": "a short name for the card",
    "brief": "free text for the simulated user and the judge: what is going on, what the person wants, details they give only when asked, when it counts as done",
    "opening": "the first message, sent to the assistant exactly as written",
    "situation": "what is going on in the person's life right now (the user model reads this, the assistant never does)",
    "goal": "what the person wants out of the conversation",
    "hidden": "facts the person knows but did not put in the opening; given out one at a time when asked",
    "reveal": "how hidden facts are given out",
    "beats": "ordered sub-goals; the next is raised only once the current one is settled",
    "done_when": "when the person considers the job done — also the judge's hard requirement",
    "max_turns": "user messages in this conversation (overrides config.json)",
    "tags": "free labels for filtering (local, calendar, shopping, …)",
    "leaves": "what running it creates on the account (calendar entries, reminders, pages) — tells the cleanup step what to undo",
    "user": "facts for this scenario that differ from the profile, e.g. {\"car\": \"2016 Subaru Outback\"}",
    "scoring": "optional: extra judge criteria, {\"criteria\": [\"...\"]}",
}


@dataclass
class Scenario:
    id: str
    opening: str
    title: str = ""
    brief: str = ""
    situation: str = ""
    goal: str = ""
    hidden: list[str] = field(default_factory=list)
    reveal: str = DEFAULT_REVEAL
    done_when: str = ""
    beats: list[str] = field(default_factory=list)
    max_turns: int | None = None
    scoring: dict = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    leaves: str = ""
    user: dict = field(default_factory=dict)      # fact overrides for this scenario

    def texts(self) -> list[str]:
        out = [self.opening, self.brief, self.situation, self.goal, self.done_when, self.reveal, *self.hidden, *self.beats]
        if self.scoring:
            out.append(json.dumps(self.scoring, ensure_ascii=False))
        return out

    def placeholders(self) -> set[str]:
        return {m for t in self.texts() for m in PLACEHOLDER.findall(t)}

    def to_dict(self) -> dict:
        d = {"id": self.id, "opening": self.opening}
        for k in ("title", "brief", "situation", "goal", "hidden", "reveal", "done_when", "beats", "max_turns", "scoring", "tags", "leaves", "user"):
            v = getattr(self, k)
            if v in (None, "", [], {}) or (k == "reveal" and v == DEFAULT_REVEAL):
                continue
            d[k] = v
        return d


def from_dict(d: dict, where: str = "") -> Scenario:
    if not isinstance(d, dict) or not str(d.get("id") or "").strip() or not str(d.get("opening") or "").strip():
        raise ValueError(f"{where}: a scenario needs at least id and opening")
    sid = str(d["id"]).strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", sid):
        raise ValueError(f"{where}: id {sid!r} must look like <set>/<name> with letters, digits, _ . - only")
    for key in ("hidden", "tags", "beats"):
        if key in d and not isinstance(d[key], list):
            raise ValueError(f"{where}: {sid}: {key} must be a list")
    turns = d.get("max_turns")
    if turns is not None and (not isinstance(turns, int) or turns < 1):
        raise ValueError(f"{where}: {sid}: max_turns must be a positive integer")
    if "user" in d and not isinstance(d["user"], dict):
        raise ValueError(f"{where}: {sid}: user must be an object of facts")
    unknown = sorted(set(d) - set(FIELD_HELP) - {"persona", "requires", "real"})   # the last three: accepted and ignored (older files)
    if unknown:
        raise ValueError(f"{where}: {sid}: unknown field(s) {', '.join(unknown)}")
    return Scenario(id=sid, opening=str(d["opening"]).strip(), title=str(d.get("title") or ""), brief=str(d.get("brief") or "").strip(), situation=str(d.get("situation") or ""),
                    goal=str(d.get("goal") or ""), hidden=[str(x) for x in d.get("hidden") or []],
                    reveal=str(d.get("reveal") or DEFAULT_REVEAL), done_when=str(d.get("done_when") or ""), beats=[str(x) for x in d.get("beats") or []],
                    max_turns=turns, scoring=d.get("scoring") if isinstance(d.get("scoring"), dict) else {}, tags=[str(x) for x in d.get("tags") or []],
                    leaves=str(d.get("leaves") or ""), user=d.get("user") or {})


def load_file(path: Path) -> list[Scenario]:
    out: list[Scenario] = []
    seen: set[str] = set()
    for n, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            d = json.loads(line)
        except ValueError as exc:
            raise ValueError(f"{path.name}:{n}: not JSON: {exc}") from None
        sc = from_dict(d, f"{path.name}:{n}")
        if sc.id in seen:
            raise ValueError(f"{path.name}:{n}: duplicate id {sc.id!r}")
        seen.add(sc.id)
        out.append(sc)
    return out


def sets(directory: Path) -> list[str]:
    return sorted(p.stem for p in directory.glob("*.jsonl") if not p.name.startswith("_"))


def set_label(directory: Path, stem: str) -> str:
    """The human name of a set: the first comment line of its file up to ' — ' (else the stem)."""
    path = directory / f"{stem}.jsonl"
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("#"):
                return line.lstrip("# ").split(" — ")[0].split(" - ")[0].strip() or stem
            if line.strip():
                break
    except OSError:
        pass
    return stem


def resolve(spec: str, only: list[str] | None = None, directory: Path | None = None) -> list[Scenario]:
    """``spec`` is a path, a file stem under the scenarios dir, or ``all``; ``only`` keeps ids."""
    scs: list[Scenario] = []
    path = Path(spec).expanduser()
    if path.is_file():
        scs = load_file(path)
    elif directory is not None and spec == "all":
        for stem in sets(directory):
            scs += load_file(directory / f"{stem}.jsonl")
    elif directory is not None and (directory / f"{spec}.jsonl").exists():
        scs = load_file(directory / f"{spec}.jsonl")
    else:
        have = sets(directory) if directory is not None else []
        raise ValueError(f"no scenario file {spec!r}; have: {', '.join(have) or 'none'}" + (f" under {directory}" if directory else ""))
    if only:
        by_id = {s.id: s for s in scs}
        missing = [i for i in only if i not in by_id]
        if missing:
            raise ValueError(f"no scenario {', '.join(missing)} in {spec}")
        scs = [by_id[i] for i in only]
    return scs


def save(directory: Path, set_name: str, sc: Scenario) -> Path:
    """Add or replace one scenario in ``<directory>/<set>.jsonl``, keeping comment lines."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", set_name):
        raise ValueError(f"bad set name {set_name!r}")
    path = directory / f"{set_name}.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    out, replaced = [], False
    for raw in lines:
        s = raw.strip()
        if s and not s.startswith("#"):
            try:
                if json.loads(s).get("id") == sc.id:
                    out.append(json.dumps(sc.to_dict(), ensure_ascii=False))
                    replaced = True
                    continue
            except ValueError:
                pass
        out.append(raw)
    if not replaced:
        out.append(json.dumps(sc.to_dict(), ensure_ascii=False))
    directory.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out).rstrip("\n") + "\n", encoding="utf-8")
    return path


def delete(directory: Path, set_name: str, scenario_id: str) -> bool:
    path = directory / f"{set_name}.jsonl"
    if not path.exists():
        return False
    out, found = [], False
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if s and not s.startswith("#"):
            try:
                if json.loads(s).get("id") == scenario_id:
                    found = True
                    continue
            except ValueError:
                pass
        out.append(raw)
    path.write_text("\n".join(out).rstrip("\n") + "\n", encoding="utf-8")
    return found


def render_text(text: str, facts: dict) -> str:
    return PLACEHOLDER.sub(lambda m: str(facts[m.group(1)]) if m.group(1) in facts else m.group(0), text)


def profile_for(sc: Scenario, user: Profile) -> Profile:
    return user.with_overrides(sc.user)


def render(sc: Scenario, user: Profile) -> Scenario:
    """The scenario as this user lives it: every placeholder filled (the scenario's own user block applied first)."""
    facts = profile_for(sc, user).facts
    r = lambda t: render_text(t, facts)   # noqa: E731
    scoring = json.loads(render_text(json.dumps(sc.scoring, ensure_ascii=False), facts)) if sc.scoring else {}
    return Scenario(id=sc.id, opening=r(sc.opening), title=sc.title, brief=r(sc.brief), situation=r(sc.situation), goal=r(sc.goal), hidden=[r(h) for h in sc.hidden],
                    reveal=r(sc.reveal), done_when=r(sc.done_when), beats=[r(b) for b in sc.beats], max_turns=sc.max_turns, scoring=scoring,
                    tags=sc.tags, leaves=sc.leaves, user=sc.user)


def lint(sc: Scenario, user: Profile) -> list[str]:
    """Why this scenario cannot be lived by this user. Empty = fine."""
    problems: list[str] = []
    u = profile_for(sc, user)
    unknown = sorted(p for p in sc.placeholders() if not u.has(p) and not p.startswith(RUNTIME_PREFIX))
    if unknown:
        problems.append("placeholder(s) with no fact in the profile or the scenario's user block: " + ", ".join("{" + p + "}" for p in unknown))
    if not sc.done_when and not sc.brief:
        problems.append("the notes are empty: the simulated user does not know what it wants or when it is done")
    if sc.max_turns is not None and sc.beats and sc.max_turns < len(sc.beats):
        problems.append(f"max_turns {sc.max_turns} is smaller than the number of beats ({len(sc.beats)})")
    return problems


def lint_all(scenarios: list[Scenario], user: Profile) -> list[tuple[str, str]]:
    return [(sc.id, p) for sc in scenarios for p in lint(sc, user)]
