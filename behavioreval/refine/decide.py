"""Numbers per conversation and per split, and the rule that keeps or rejects a candidate.

Quality of one conversation = half the existing judge's PASS, half the share of applicable rubric lines met.
A conversation the product broke (error stop) or the judge could not read is counted but not scored.
The keep rule, in order:
  nothing scorable                                        → reject
  a held-out card that always passed now always fails      → reject (regression)
  pairwise: more "worse" than "better"                     → reject
  quality up by more than noise_margin                     → keep
  quality within noise and one of: faster by speed_margin with the pass rate held, at least 10% shorter text,
  or preferred in at least half of the pairwise pairs      → keep
  otherwise                                                → reject (no change beyond noise)
A version cannot win by doing less: speed and length only count when quality and pass rate held."""
from __future__ import annotations

import json
from statistics import mean


def speed(conv) -> dict:
    seen, repeated, calls = set(), 0, 0
    for t in conv.turns:
        for c in t.calls or []:
            calls += 1
            k = (c.get("name"), json.dumps(c.get("input") or {}, sort_keys=True, ensure_ascii=False))
            repeated += k in seen
            seen.add(k)
    asked = [t for t in conv.turns if t.user]
    return {"user_turns": len(asked), "total_s": round(sum(t.duration_s or 0 for t in asked), 1),
            "first_reply_s": round(asked[0].duration_s or 0, 1) if asked else 0.0, "tool_calls": calls, "repeated_calls": repeated,
            "slowest_turn": max(asked, key=lambda t: t.duration_s or 0).n if asked else 0}


def scorable(r: dict) -> bool:
    return r.get("verdict") in ("PASS", "FAIL") and not str(r.get("stop", "")).startswith("error") and r.get("stop") != "assistant_error"


def rubric_share(r: dict) -> float | None:
    lines = [l for l in (r.get("rubric") or {}).get("lines") or [] if l.get("met") is not None]
    return sum(bool(l["met"]) for l in lines) / len(lines) if lines else None


def quality(r: dict) -> float:
    share = rubric_share(r)
    passed = 1.0 if r.get("verdict") == "PASS" else 0.0
    return passed if share is None else 0.5 * passed + 0.5 * share


def failed_lines(r: dict) -> list[dict]:
    return [l for l in (r.get("rubric") or {}).get("lines") or [] if l.get("met") is False]


def is_failure(r: dict) -> bool:
    return r.get("verdict") == "FAIL" or any(l.get("severity") in ("high", "medium") for l in failed_lines(r))


def too_slow(r: dict, max_user_turns: int, max_total_s: float) -> bool:
    s = r.get("speed") or {}
    return bool((max_user_turns and s.get("user_turns", 0) > max_user_turns) or (max_total_s and s.get("total_s", 0) > max_total_s))


def summary(results: list[dict]) -> dict:
    scored = [r for r in results if scorable(r)]
    by_card: dict = {}
    for r in scored:
        by_card.setdefault(r["card"], []).append(r)
    return {
        "conversations": len(results), "scored": len(scored),
        "pass_rate": round(mean(r["verdict"] == "PASS" for r in scored), 3) if scored else None,
        "rubric": round(mean(s for s in (rubric_share(r) for r in scored) if s is not None), 3) if any(rubric_share(r) is not None for r in scored) else None,
        "quality": round(mean(quality(r) for r in scored), 3) if scored else None,
        "mean_total_s": round(mean((r.get("speed") or {}).get("total_s", 0) for r in scored), 1) if scored else None,
        "mean_user_turns": round(mean((r.get("speed") or {}).get("user_turns", 0) for r in scored), 2) if scored else None,
        "card_pass": {c: round(mean(r["verdict"] == "PASS" for r in rs), 3) for c, rs in sorted(by_card.items())},
    }


def regressions(best: dict, cand: dict) -> list[str]:
    return [c for c, p in best.get("card_pass", {}).items() if p == 1.0 and cand.get("card_pass", {}).get(c) == 0.0]


def decide(best: dict, cand: dict, pairwise: list[str], best_chars: int, cand_chars: int, noise_margin: float, speed_margin: float) -> tuple[bool, str]:
    if not cand.get("scored"):
        return False, "nothing scorable on held-out (every conversation errored or was unreadable)"
    lost = regressions(best, cand)
    if lost:
        return False, "regression: always passed before, always fails now: " + ", ".join(lost)
    better, worse = pairwise.count("better"), pairwise.count("worse")
    if worse > better:
        return False, f"pairwise net worse ({better} better, {worse} worse, {len(pairwise) - better - worse} equal)"
    dq = (cand["quality"] or 0) - (best.get("quality") or 0)
    if dq > noise_margin:
        return True, f"held-out quality up {dq:+.2f} (over the {noise_margin:.2f} noise margin)"
    if dq < -noise_margin:
        return False, f"held-out quality down {dq:+.2f}"
    held = (cand.get("pass_rate") or 0) >= (best.get("pass_rate") or 0)
    if held and best.get("mean_total_s") and cand.get("mean_total_s") is not None and cand["mean_total_s"] <= best["mean_total_s"] * (1 - speed_margin):
        return True, f"same quality ({dq:+.2f}), faster: {best['mean_total_s']:.0f}s → {cand['mean_total_s']:.0f}s per conversation"
    if held and cand_chars <= best_chars * 0.9:
        return True, f"same quality ({dq:+.2f}), shorter skill: {best_chars} → {cand_chars} characters"
    if held and pairwise and better >= max(1, len(pairwise) / 2):
        return True, f"same quality ({dq:+.2f}), preferred in pairwise ({better} of {len(pairwise)} better)"
    return False, f"no change beyond noise (quality {dq:+.2f}; pairwise {better} better, {worse} worse)"
