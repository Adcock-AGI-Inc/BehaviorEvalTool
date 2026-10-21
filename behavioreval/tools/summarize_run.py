"""Summarize a run directory: per column pass rate, stops, judge flags, per-scenario pass rate, and which
conversations count as usable traces (the assistant replied at least once, no fallback line from the user
model went out, the conversation did not end in an error before any reply).
    python3 python3 -m behavioreval.tools.summarize_run <run dir> [--md out.md]"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path

FALLBACKS = ("ok, go ahead", "hold on, can you say that again in one line?")


def load(run_dir: Path) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(run_dir.glob("*.conversation.json"))]


def usable(d: dict) -> tuple[bool, str]:
    turns = d.get("turns") or []
    if not any(t.get("user") and (t.get("assistant") or t.get("calls")) for t in turns):
        return False, "no assistant reply"
    if any((t.get("user") or "") in FALLBACKS for t in turns):
        return False, "user-model fallback line sent"
    if str(d.get("stop", "")).startswith("error"):
        return False, "harness error"
    return True, ""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--md", help="write the summary as markdown here")
    a = ap.parse_args(argv)
    rows = [d for d in load(Path(a.run_dir).expanduser()) if not str(d.get("scenario", "")).startswith("maint/")]
    by_col = collections.defaultdict(list)
    for d in rows:
        by_col[d.get("column") or "default"].append(d)
    out = [f"# {Path(a.run_dir).name}: {len(rows)} conversations", ""]
    out += ["| column | n | usable | PASS | FAIL | unreadable | pass rate (judged) | mean score | done | max_turns | assistant_error | timeout | fabrication | violations | asked first (when relevant) |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for col, ds in sorted(by_col.items()):
        v = collections.Counter((d.get("score") or {}).get("verdict", "-") for d in ds)
        stops = collections.Counter(d.get("stop") for d in ds)
        judged = [(d.get("score") or {}).get("judge", [{}])[0] for d in ds if (d.get("score") or {}).get("judge")]
        scores = [j.get("score") for j in judged if isinstance(j.get("score"), (int, float))]
        fab = sum(1 for j in judged if j.get("fabrication") is True)
        viol = sum(1 for j in judged if j.get("violations"))
        rel = [j for j in judged if j.get("asked_before_acting") in (True, False)]
        asked = sum(1 for j in rel if j.get("asked_before_acting") is True)
        us = sum(1 for d in ds if usable(d)[0])
        pf = v["PASS"] + v["FAIL"]
        out.append(f"| {col} | {len(ds)} | {us} | {v['PASS']} | {v['FAIL']} | {v['UNREADABLE'] + v['-']} | {v['PASS'] / pf:.0%} | {statistics.mean(scores):.2f} | "
                   f"{stops['done']} | {stops['max_turns']} | {stops['assistant_error']} | {stops['timeout']} | {fab} | {viol} | {asked}/{len(rel)} |" if pf and scores else
                   f"| {col} | {len(ds)} | {us} | {v['PASS']} | {v['FAIL']} | {v['UNREADABLE'] + v['-']} | - | - | {stops['done']} | {stops['max_turns']} | {stops['assistant_error']} | {stops['timeout']} | {fab} | {viol} | {asked}/{len(rel)} |")
    out.append("")
    # per scenario per column
    cols = sorted(by_col)
    out += ["| scenario | " + " | ".join(f"{c} PASS/judged" for c in cols) + " |", "|---|" + "---|" * len(cols)]
    scen = sorted({d["scenario"] for d in rows})
    for s in scen:
        cells = []
        for c in cols:
            ds = [d for d in by_col[c] if d["scenario"] == s]
            v = collections.Counter((d.get("score") or {}).get("verdict", "-") for d in ds)
            cells.append(f"{v['PASS']}/{v['PASS'] + v['FAIL']}" if v["PASS"] + v["FAIL"] else "-")
        out.append(f"| {s} | " + " | ".join(cells) + " |")
    out.append("")
    reasons = collections.Counter()
    for d in rows:
        ok, why = usable(d)
        if not ok:
            reasons[why] += 1
    out.append("Not usable: " + (", ".join(f"{k} = {n}" for k, n in reasons.items()) or "none"))
    text = "\n".join(out)
    print(text)
    if a.md:
        Path(a.md).expanduser().write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
