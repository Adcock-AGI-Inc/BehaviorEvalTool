"""A readable digest of the failed conversations of a run: grouped by column and failure kind,
each with the judge's reasoning, the evidence it cited (violations, fabrications, feasibility),
and a compact transcript (user lines, assistant replies, tool names per turn).
    python3 python3 -m usersim.tools.failures_digest <run dir> --out digest.md [--max-per-kind N] [--chars N]"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def kind_of(d: dict) -> str:
    stop = d.get("stop", "")
    j = ((d.get("score") or {}).get("judge") or [{}])[0]
    if stop.startswith("error") or stop == "assistant_error":
        return "assistant run failed (product/lane error)"
    if j.get("violations"):
        return "violation (acted against the user's limits or without permission)"
    if j.get("fabrication") and j.get("fabrication_examples"):
        return "fabricated specifics"
    if stop == "timeout":
        return "turn timed out"
    if stop == "max_turns":
        return "ran out of turns before the hard requirement"
    return "delivered but not what was asked"


def transcript(d: dict, chars: int) -> list[str]:
    out = []
    for t in d.get("turns") or []:
        if not t.get("user"):
            continue
        out.append(f"- **user {t['n']}:** {t['user'][:chars]}")
        names = collections.Counter(c.get("name", "?") for c in t.get("calls") or [])
        tools = ", ".join(f"{k}×{v}" if v > 1 else k for k, v in names.items())
        reply = (t.get("assistant") or "(no reply)")[:chars]
        err = f" — error: {t['error'][:120]}" if t.get("error") else ""
        out.append(f"  - assistant ({t.get('reason') or '?'}; {tools or 'no tools'}; {t.get('duration_s', 0):.0f}s): {reply}{err}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-per-kind", type=int, default=4)
    ap.add_argument("--chars", type=int, default=420)
    a = ap.parse_args(argv)
    run = Path(a.run_dir).expanduser()
    fails = []
    for f in sorted(run.glob("*.conversation.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        if d.get("scenario", "").startswith("maint/"):
            continue
        v = (d.get("score") or {}).get("verdict")
        if v == "FAIL" or str(d.get("stop", "")).startswith("error") or d.get("stop") == "assistant_error":
            fails.append(d)
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for d in fails:
        by[d.get("column") or "default"][kind_of(d)].append(d)
    lines = [f"# Failed conversations — {run.name}", "", f"{len(fails)} failures. Per column and kind (count), then up to {a.max_per_kind} examples each.", ""]
    for col in sorted(by):
        lines += [f"## {col}", "", "| kind | count |", "|---|---|"]
        for k, ds in sorted(by[col].items(), key=lambda kv: -len(kv[1])):
            lines.append(f"| {k} | {len(ds)} |")
        lines.append("")
        for k, ds in sorted(by[col].items(), key=lambda kv: -len(kv[1])):
            lines += [f"### {col}: {k}", ""]
            for d in ds[: a.max_per_kind]:
                j = ((d.get("score") or {}).get("judge") or [{}])[0]
                lines += [f"#### {d['scenario']} as {d.get('account','')} — stop={d['stop']}, score={j.get('score', '?')}, steps={j.get('steps_completed', '?')}, asked_first={j.get('asked_before_acting', '?')}", ""]
                lines.append(f"**Judge:** {j.get('reasoning', '')[:900]}")
                for key in ("violations", "fabrication_examples", "feasibility_issues"):
                    if j.get(key):
                        lines.append(f"- *{key}:* " + "; ".join(str(x) for x in j[key])[:600])
                lines.append("")
                lines += transcript(d, a.chars)
                lines.append("")
            rest = len(ds) - a.max_per_kind
            if rest > 0:
                lines.append(f"_… {rest} more of this kind: " + ", ".join(f"{d['scenario'].split('/')[-1]}/{d.get('account','')}" for d in ds[a.max_per_kind:]) + "_")
                lines.append("")
    Path(a.out).expanduser().write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(fails)} failures → {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
