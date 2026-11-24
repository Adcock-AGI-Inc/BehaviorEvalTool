"""From graded train conversations to the one failure the editor should fix next.

diagnose_all  gives every failed or slow train conversation a cause (judges.CAUSES). Two causes are settled
              without a model call: a conversation that never loaded the candidate is not_loaded, and one the
              product broke is product_or_tool.
pick          groups the failures whose cause the skill text can fix (ignored, silent, caused_it) by the rubric
              line or judge failure kind they concern, weights each group by severity × count, and returns the
              heaviest group that has not had tries_per_failure rounds without a kept edit. Slow conversations form their own group."""
from __future__ import annotations

import json
from pathlib import Path

from behavioreval.core import records, scenario
from behavioreval.refine import decide, judges
from behavioreval.tools import failures_digest


def _save(r: dict, path: Path) -> None:
    path.write_text(json.dumps(r, indent=1, ensure_ascii=False), encoding="utf-8")


def diagnose_all(results: list[dict], paths: dict, inputs, graders, skill_text: str, delivery_kind: str) -> dict:
    """Adds ``diagnosis`` to each failed or slow result (and saves it). Returns counts per cause."""
    s = inputs.settings
    counts: dict = {}
    cards = {c.scenario.id: c for c in inputs.cards}
    for r in results:
        if not (decide.is_failure(r) or decide.too_slow(r, s.max_user_turns, s.max_total_s) or str(r.get("stop", "")).startswith("error")):
            continue
        if not r.get("diagnosis"):
            r["diagnosis"] = _diagnose(r, cards, inputs, graders, skill_text, delivery_kind)
            _save(r, paths[(r["card"], r["rollout"])])
        cause = r["diagnosis"].get("cause", "unreadable")
        counts[cause] = counts.get(cause, 0) + 1
    return counts


def _diagnose(r: dict, cards: dict, inputs, graders, skill_text: str, delivery_kind: str) -> dict:
    if delivery_kind == "personal_skill" and r.get("loaded") is False:
        return {"cause": "not_loaded", "evidence": r.get("loaded_detail", "")}
    if str(r.get("stop", "")).startswith("error") or r.get("stop") == "assistant_error":
        return {"cause": "product_or_tool", "evidence": r.get("stop", "")}
    card = cards[r["card"]]
    persona = inputs.personas[card.persona]
    conv, _ = records.load_conversation(Path(r["file"]))
    sc = scenario.render(card.scenario, persona)
    grades = {"verdict": r.get("verdict"), "judge": r.get("judge_reasoning"), "rubric_not_met": decide.failed_lines(r)}
    st = inputs.settings
    targets = f"targets: at most {st.max_user_turns or 'any'} user turns, {st.max_total_s or 'any'} s per conversation"
    speed = json.dumps({**r.get("speed", {}), "per_turn_s": [t.duration_s for t in conv.turns if t.user]}) + f"; {targets}"
    return graders.diagnose(conv, sc, persona, skill_text, grades, speed)


def group_key(r: dict) -> str:
    dg = r.get("diagnosis") or {}
    if dg.get("criterion"):
        return f"rubric {dg['criterion']}"
    if r.get("verdict") == "FAIL":
        try:
            d = json.loads(Path(r["file"]).read_text(encoding="utf-8"))
            return "hard requirement: " + failures_digest.kind_of(d)
        except (OSError, ValueError):
            return "hard requirement"
    return "speed"


def _weight(r: dict) -> int:
    lines = decide.failed_lines(r)
    crit = (r.get("diagnosis") or {}).get("criterion")
    sev = [judges.SEVERITY_WEIGHT.get(l.get("severity"), 1) for l in lines if not crit or l.get("id") == crit]
    return max(sev) if sev else (3 if r.get("verdict") == "FAIL" else 1)


def pick(results: list[dict], tried: list[dict], tries_per_failure: int) -> tuple[str, list[dict]] | None:
    groups: dict = {}
    for r in results:
        if (r.get("diagnosis") or {}).get("cause") in judges.SKILL_CAUSES:
            groups.setdefault(group_key(r), []).append(r)
    rounds_without_keep: dict = {}
    for t in tried:
        if t.get("outcome") != "kept":
            rounds_without_keep.setdefault(t.get("target"), set()).add(t.get("round"))
    spent = {k: len(v) for k, v in rounds_without_keep.items()}
    ranked = sorted(groups.items(), key=lambda kv: -sum(_weight(r) for r in kv[1]))
    for key, rs in ranked:
        if spent.get(key, 0) < tries_per_failure:
            return key, rs
    return None
