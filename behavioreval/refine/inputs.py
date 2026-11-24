"""Everything a person supplies before a refine run, and the check that refuses to start until it is all there.

Shared across skills (repo root):
    rubric/general.md               what a good skill conversation looks like, for every skill
    rubric/skill_best_practices.md  how a good skill is written (the editor follows it)
One folder per skill, ``skills/<name>/``:
    goal.md         what the skill must achieve; what success looks like
    criteria.md     this skill's own rubric lines, one per line starting with "- "
    personality.md  how Hark should sound when doing this
    personas.jsonl  the simulated users: one profile per line (id, facts, style, limits)
    cards.jsonl     the test cards: the usual card fields plus "split" (train | holdout | final) and "persona"
    start.md        the skill text to start from; absent = write the first version from the goal
    refine.json     budgets, models, how the skill reaches Hark (see Settings)
A file that still contains the word PLACEHOLDER counts as not filled in."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, fields
from pathlib import Path

from behavioreval.core import profile as profile_mod, scenario
from behavioreval.core.config import ROOT

PLACEHOLDER = "PLACEHOLDER"
SPLITS = ("train", "holdout", "final")
SKILL_NAME = re.compile(r"^[a-z0-9-]{1,64}$")   # what Hark's skill tool accepts (services/agent/internal/tools/skill.go)
DELIVERIES = ("personal_skill", "paste", "fake")
SKILL_TEXT_LIMIT_CHARS = 48_000                 # skills service cap on name + description + content (services/skills handler.go:49)

DEFAULT_HALT = [
    {"tool": r"^google_gmail_(send|reply|forward)_message$", "why": "sent an email"},
    {"tool": r"^send_texts?$", "why": "sent a text"},
    {"tool": r"(pay|purchase|checkout)", "why": "a payment tool"},
    {"tool": r"^google_calendar_(create|update)_event$", "input": r"attendee", "why": "a calendar invite to other people"},
    {"tool": r"^browser$", "input": r"(?i)(place (the )?order|complete (the )?(purchase|checkout|booking)|pay now|submit payment|confirm (the )?booking)",
     "why": "told the browser to complete a purchase or booking"},
]


@dataclass
class Settings:
    """refine.json. Every key is optional; these are the defaults."""
    skill_name: str = ""                 # the personal skill's name on the test account; "" = the folder name
    skill_description: str = ""          # the one-line description Hark sees in its skill list (personal_skill delivery needs it)
    delivery: str = "personal_skill"     # personal_skill | paste | fake
    rounds: int = 8
    max_cost_usd: float = 25.0           # our own model calls (simulated user, judges, editor); Hark's cost on dev is not visible here
    max_conversations: int = 300         # the practical cap on Hark's side
    rollouts_train: int = 2
    rollouts_holdout: int = 3
    rollouts_final: int = 3
    screen_rollouts: int = 1             # per train card, when screening candidate edits
    candidates_per_round: int = 3
    noise_margin: float = 0.10           # a quality change smaller than this is noise
    speed_margin: float = 0.15           # a speed gain counts only if it is at least this fraction
    tries_per_failure: int = 2           # rejected edits on one failure before moving to the next
    plateau: int = 3                     # rejected rounds in a row before stopping
    max_diff_lines: int = 6            # changed lines per edit: the small learning rate
    max_skill_chars: int = 20_000
    parallel: int = 4
    editor_model: str = "anthropic/claude-opus-5.5"
    diagnose_model: str = "anthropic/claude-sonnet-5.5"
    judge_model: str = ""                # "" = config.json's judge_model; must differ from editor_model
    max_user_turns: int = 0              # speed target; 0 = none
    max_total_s: float = 0.0             # speed target per conversation, seconds; 0 = none
    halt: list = field(default_factory=lambda: [dict(r) for r in DEFAULT_HALT])


@dataclass
class Card:
    split: str
    persona: str
    scenario: scenario.Scenario


@dataclass
class SkillInputs:
    name: str
    folder: Path
    general_rubric: str
    best_practices: str
    goal: str
    criteria: list[str]
    personality: str
    personas: dict[str, profile_mod.Profile]
    cards: list[Card]
    start: str | None
    settings: Settings
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def split(self, name: str) -> list[Card]:
        return [c for c in self.cards if c.split == name]

    @property
    def personal_name(self) -> str:
        return self.settings.skill_name or self.name


def _text(path: Path, problems: list[str], required: bool = True) -> str:
    if not path.exists():
        if required:
            problems.append(f"missing {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")
        return ""
    text = path.read_text(encoding="utf-8").strip()
    if PLACEHOLDER in text:
        problems.append(f"{path.name} is still a placeholder")
    elif required and not text:
        problems.append(f"{path.name} is empty")
    return text


def _jsonl(path: Path, problems: list[str]) -> list[dict]:
    rows = []
    if not path.exists():
        problems.append(f"missing {path.name}")
        return rows
    for n, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if PLACEHOLDER in line:
            problems.append(f"{path.name}:{n} is still a placeholder")
            continue
        try:
            rows.append((n, json.loads(line)))
        except ValueError as exc:
            problems.append(f"{path.name}:{n}: not JSON: {exc}")
    return rows


def load_settings(path: Path, problems: list[str]) -> Settings:
    s = Settings()
    if not path.exists():
        return s
    data = json.loads(path.read_text(encoding="utf-8"))
    known = {f.name: f for f in fields(Settings)}
    for k, v in data.items():
        if k.startswith("_"):
            continue
        if k not in known:
            problems.append(f"refine.json: unknown key {k!r}")
            continue
        setattr(s, k, v)
    return s


def _personas(folder: Path, problems: list[str]) -> dict[str, profile_mod.Profile]:
    out = {}
    for n, d in _jsonl(folder / "personas.jsonl", problems):
        try:
            p = profile_mod.from_dict(d, f"personas.jsonl:{n}")
        except ValueError as exc:
            problems.append(str(exc))
            continue
        if p.id in out:
            problems.append(f"personas.jsonl:{n}: duplicate persona {p.id!r}")
        out[p.id] = p
    return out


def _cards(folder: Path, personas: dict, problems: list[str]) -> list[Card]:
    out, seen = [], set()
    for n, d in _jsonl(folder / "cards.jsonl", problems):
        where = f"cards.jsonl:{n}"
        split, persona = str(d.pop("split", "")), str(d.get("persona") or "")
        if split not in SPLITS:
            problems.append(f"{where}: split must be one of {', '.join(SPLITS)} (got {split!r})")
            continue
        if persona not in personas:
            problems.append(f"{where}: persona {persona!r} is not in personas.jsonl")
            continue
        try:
            sc = scenario.from_dict(d, where)
        except ValueError as exc:
            problems.append(str(exc))
            continue
        if sc.id in seen:
            problems.append(f"{where}: card {sc.id!r} appears twice (a card belongs to one split only)")
            continue
        seen.add(sc.id)
        problems += [f"{where}: {p}" for p in scenario.lint(sc, personas[persona])]
        out.append(Card(split, persona, sc))
    return out


def load(name: str, skills_dir: Path | None = None, rubric_dir: Path | None = None) -> SkillInputs:
    """Read every input. Never raises for a content problem: those land in ``problems`` so ``check`` can list them all."""
    skills_dir = skills_dir or ROOT / "skills"
    rubric_dir = rubric_dir or ROOT / "rubric"
    folder = skills_dir / name
    problems: list[str] = []
    if not folder.is_dir():
        return SkillInputs(name, folder, "", "", "", [], "", {}, [], None, Settings(), [f"no skill folder {folder}; create one with `refine new {name}`"])
    settings = load_settings(folder / "refine.json", problems)
    criteria_text = _text(folder / "criteria.md", problems)
    personas = _personas(folder, problems)
    start_path = folder / "start.md"
    inputs = SkillInputs(
        name=name, folder=folder,
        general_rubric=_text(rubric_dir / "general.md", problems),
        best_practices=_text(rubric_dir / "skill_best_practices.md", problems),
        goal=_text(folder / "goal.md", problems),
        criteria=[l[2:].strip() for l in criteria_text.splitlines() if l.startswith("- ") and l[2:].strip()],
        personality=_text(folder / "personality.md", problems),
        personas=personas, cards=_cards(folder, personas, problems),
        start=_text(start_path, problems) if start_path.exists() else None,
        settings=settings, problems=problems)
    _check(inputs)
    return inputs


def _check(i: SkillInputs) -> None:
    s, p, w = i.settings, i.problems, i.warnings
    if not i.criteria and not any("criteria.md" in x for x in p):
        p.append("criteria.md has no lines starting with '- '")
    for split, need in (("train", 1), ("holdout", 1)):
        if len(i.split(split)) < need:
            p.append(f"cards.jsonl needs at least {need} {split} card")
    if not i.split("final"):
        w.append("no final cards: the morning report will have no untouched score")
    if i.split("holdout") and len(i.split("holdout")) < 0.2 * (len(i.split("holdout")) + len(i.split("train"))):
        w.append("held-out is under a fifth of the cards; about a third keeps the keep/reject decision honest")
    if s.delivery not in DELIVERIES:
        p.append(f"refine.json: delivery must be one of {', '.join(DELIVERIES)}")
    if not SKILL_NAME.match(i.personal_name):
        p.append(f"skill name {i.personal_name!r}: Hark accepts lowercase letters, digits and hyphens, at most 64")
    if PLACEHOLDER in json.dumps(vars(s)):
        p.append("refine.json still has a PLACEHOLDER value")
    elif s.delivery == "personal_skill" and not s.skill_description.strip():
        p.append("refine.json: skill_description is required for personal_skill delivery (it is what Hark sees in its skill list)")
    if s.max_skill_chars + len(s.skill_description) + len(i.personal_name) > SKILL_TEXT_LIMIT_CHARS:
        p.append(f"refine.json: max_skill_chars plus the description must stay under Hark's {SKILL_TEXT_LIMIT_CHARS} characters")
    if i.start is not None and len(i.start) > s.max_skill_chars:
        p.append(f"start.md is {len(i.start)} characters; max_skill_chars is {s.max_skill_chars}")
    if i.start is None:
        w.append("no start.md: round 0 writes the first version from the goal")
    for n, r in enumerate(s.halt):
        try:
            re.compile(r.get("tool", "")), re.compile(r.get("input", ""))
        except re.error as exc:
            p.append(f"refine.json: halt rule {n}: {exc}")
