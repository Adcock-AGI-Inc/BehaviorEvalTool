"""The optimizer step: one small, explained change to the skill per candidate.

The editor sees the current skill, the goal, the best-practices doc, the rubric, the personality, the one
failure group it should fix (train conversations only, with the judge's reasons and the diagnosis), and the
edits already tried. It never sees a held-out or final card. It answers with a few candidate edits, each an
exact find/replace (Hark's own skill-edit shape), an addition after an anchor, or a deletion, with a
one-line reason. Every candidate then passes the guards before any conversation runs:
  applies   the find text occurs exactly once (an addition with no anchor goes at the end)
  small     at most max_diff_lines changed lines, and the skill stays under max_skill_chars
  general   no specifics from the train cards or personas (names, places, amounts, dates) that the
            skill, goal, criteria or best practices do not already contain."""
from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass

from behavioreval.refine.decide import failed_lines
from behavioreval.tools import failures_digest

KINDS = ("replace", "add", "delete")
_NUMBER = re.compile(r"\$?\d[\d,.:/-]*\d|\$\d")
_PROPER = re.compile(r"(?<![.!?]\s)(?<!^)\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*")
_CAPITALIZED = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*")
_QUOTED = re.compile(r"[\"“]([^\"”]{4,60})[\"”]")


@dataclass
class Edit:
    kind: str
    find: str
    text: str
    reason: str
    targets: str = ""

    def to_dict(self) -> dict:
        return {"kind": self.kind, "find": self.find, "text": self.text, "reason": self.reason, "targets": self.targets}


def apply(skill: str, e: Edit) -> str:
    """The skill with the edit applied. Raises ValueError when it does not apply cleanly."""
    if e.kind not in KINDS:
        raise ValueError(f"unknown edit kind {e.kind!r}")
    if e.kind == "add" and not e.find:
        return skill.rstrip() + "\n\n" + e.text.strip() + "\n"
    n = skill.count(e.find) if e.find else 0
    if n != 1:
        raise ValueError(f"the find text occurs {n} times; it must occur exactly once")
    if e.kind == "replace":
        return skill.replace(e.find, e.text, 1)
    if e.kind == "delete":
        return skill.replace(e.find, "", 1)
    return skill.replace(e.find, e.find.rstrip("\n") + "\n" + e.text.strip("\n") + "\n", 1)


def diff(old: str, new: str, old_name: str = "before", new_name: str = "after") -> str:
    lines = lambda t: (t if t.endswith("\n") else t + "\n").splitlines(keepends=True)   # noqa: E731
    return "".join(difflib.unified_diff(lines(old), lines(new), old_name, new_name))


def changed_lines(old: str, new: str) -> int:
    return sum(1 for l in diff(old, new).splitlines() if (l.startswith("+") or l.startswith("-")) and not l.startswith(("+++", "---")))


def specifics(texts: list[str], skip_sentence_starts: bool = True) -> set[str]:
    """The concrete tokens in some text: numbers and amounts, capitalized names, quoted phrases. Lowercased.
    With skip_sentence_starts, a capitalized word that opens a line or sentence is not taken for a name."""
    out: set[str] = set()
    proper = _PROPER if skip_sentence_starts else _CAPITALIZED
    for t in texts:
        for line in (t or "").splitlines():
            out |= {m.group(0).lower().strip(".,") for m in _NUMBER.finditer(line)}
            out |= {m.group(0).lower() for m in proper.finditer(line) if len(m.group(0)) > 2}
            out |= {m.group(1).lower() for m in _QUOTED.finditer(line)}
    return out


def leaks(added: str, train_texts: list[str], allowed_texts: list[str]) -> list[str]:
    """Train-card specifics in the added text that the allowed texts (skill, goal, criteria, best practices) do not already carry."""
    allowed = " ".join(allowed_texts).lower()
    found = specifics([added]) & specifics(train_texts, skip_sentence_starts=False)
    return sorted(s for s in found if s not in allowed)


def guard(skill: str, e: Edit, max_diff_lines: int, max_chars: int, train_texts: list[str], allowed_texts: list[str]) -> tuple[str | None, str]:
    """(new skill text, '') when the edit passes every guard, else (None, why)."""
    try:
        new = apply(skill, e)
    except ValueError as exc:
        return None, f"does not apply: {exc}"
    n = changed_lines(skill, new)
    if n == 0:
        return None, "changes nothing"
    if n > max_diff_lines:
        return None, f"too big: {n} changed lines (cap {max_diff_lines})"
    if len(new) > max_chars:
        return None, f"too long: {len(new)} characters (cap {max_chars})"
    bad = leaks(e.text, train_texts, allowed_texts)
    if bad:
        return None, "names train-card specifics: " + ", ".join(bad[:6])
    return new, ""


PROMPT = """You improve the text of a skill: instructions Hark, a personal AI assistant, follows for one kind of task. Each round makes ONE small, explained change, so a reviewer can tell what helped.

<goal_of_this_skill>
{goal}
</goal_of_this_skill>

<how_good_skills_are_written>
{best_practices}
</how_good_skills_are_written>

<rubric>
{rubric}
</rubric>

<personality>
{personality}
</personality>

<current_skill>
{skill}
</current_skill>

<failure_to_fix>
{failure}
</failure_to_fix>

<edits_already_tried_for_this_skill>
{tried}
</edits_already_tried_for_this_skill>

Propose {k} different candidate edits, each one alone enough to fix this failure in general, for every user, not just the conversations shown. Rules:
- Each edit is one of: "replace" (find = exact text from the current skill, occurring once; text = its replacement), "add" (find = an exact anchor line to add after, or empty to add at the end; text = the new lines), "delete" (find = exact text to remove; text empty). Deleting or tightening is welcome when the skill is long or contradicts itself.
- At most {max_lines} changed lines per edit. Plain sentences.
- Never write anything specific to the conversations shown: no names, places, products, prices, dates or amounts from them. Write the general rule.
- Do not repeat an edit that was already tried and rejected.
- reason: one line saying what the change does and which failure it targets.

Answer with ONE JSON object:
{{"candidates": [{{"kind": "replace", "find": "...", "text": "...", "reason": "...", "targets": "the failure group key"}}]}}"""


DRAFT_PROMPT = """Write the first version of a skill: instructions Hark, a personal AI assistant, follows for one kind of task.

<goal_of_this_skill>
{goal}
</goal_of_this_skill>

<how_good_skills_are_written>
{best_practices}
</how_good_skills_are_written>

<rubric>
{rubric}
</rubric>

<personality>
{personality}
</personality>

Keep it under {max_chars} characters, in markdown. Write general rules for every user; no example names, places or amounts.
Answer with ONE JSON object: {{"skill": "the full skill text"}}"""


def failure_block(group_key: str, results: list[dict], records: dict, chars: int = 380, examples: int = 3) -> str:
    """The failure group the editor targets: its key, how often, and a few train conversations in the failures digest's compact form."""
    parts = [f"group: {group_key} ({len(results)} train conversation(s))"]
    for r in results[:examples]:
        d = records.get(r["file"]) or {}
        dg = r.get("diagnosis") or {}
        parts.append(f"\n--- {r['card']} rollout {r['rollout']}: verdict {r.get('verdict')}, cause {dg.get('cause', '?')}")
        parts.append("judge: " + str(((d.get("score") or {}).get("judge") or [{}])[0].get("reasoning", ""))[:600])
        for l in failed_lines(r):
            parts.append(f"rubric {l.get('id')} not met ({l.get('severity') or '?'}): {l.get('evidence', '')}")
        for k in ("evidence", "skill_line", "slowest_step", "fix_hint"):
            if dg.get(k):
                parts.append(f"diagnosis {k}: {dg[k]}")
        parts += failures_digest.transcript(d, chars) if d else []
    return "\n".join(parts)


def parse_candidates(d: dict) -> list[Edit]:
    out = []
    for c in d.get("candidates") or []:
        if isinstance(c, dict) and str(c.get("kind")) in KINDS:
            out.append(Edit(str(c["kind"]), str(c.get("find") or ""), str(c.get("text") or ""), str(c.get("reason") or "").strip()[:300], str(c.get("targets") or "")))
    return out


def propose(ask, key: str, model: str, post, *, goal: str, best_practices: str, rubric: str, personality: str, skill: str,
            failure: str, tried: list[dict], k: int, max_lines: int) -> tuple[list[Edit], dict]:
    tried_text = "\n".join(f"- [{t.get('outcome')}] {t.get('reason')}" for t in tried) or "(none yet)"
    raw = ask(PROMPT.format(goal=goal, best_practices=best_practices, rubric=rubric, personality=personality, skill=skill, failure=failure,
                            tried=tried_text, k=k, max_lines=max_lines), model, key, post=post, reasoning_tokens=4000)
    return parse_candidates(raw), raw


def draft(ask, key: str, model: str, post, *, goal: str, best_practices: str, rubric: str, personality: str, max_chars: int) -> str:
    raw = ask(DRAFT_PROMPT.format(goal=goal, best_practices=best_practices, rubric=rubric, personality=personality, max_chars=max_chars),
              model, key, post=post, reasoning_tokens=4000)
    text = str(raw.get("skill") or "").strip()
    if not text:
        raise ValueError("the editor returned no skill text: " + json.dumps(raw)[:200])
    return text
