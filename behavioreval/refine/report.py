"""The morning output, written after every run (also a stopped or halted one) into the run directory:

changelog.md  every version tried, kept or rejected: the edit as a diff, its reason, the failure it targeted,
              scores before and after, the pairwise tally, and a link to every held-out conversation behind it
report.md     scores by round, the final-test numbers, causes, the final skill text, the whole diff from v0,
              the 3 worst conversations of the best version, cost, and what blocked or needs a human"""
from __future__ import annotations

from pathlib import Path

from behavioreval.refine import decide, editor, rollouts


def _fmt(x, pct: bool = False) -> str:
    if x is None:
        return "-"
    return f"{x:.0%}" if pct else (f"{x:.2f}" if isinstance(x, float) else str(x))


def _summary_row(label: str, s: dict) -> str:
    s = s or {}
    return f"| {label} | {s.get('scored', 0)}/{s.get('conversations', 0)} | {_fmt(s.get('pass_rate'), True)} | {_fmt(s.get('rubric'))} | {_fmt(s.get('quality'))} | {_fmt(s.get('mean_total_s'))} | {_fmt(s.get('mean_user_turns'))} |"


SUMMARY_HEAD = ["| | scored | pass | rubric | quality | mean s | mean turns |", "|---|---|---|---|---|---|---|"]


def _links(run_dir: Path, version: str, split: str) -> list[str]:
    out = []
    for r in rollouts.load_results(run_dir, version, split):
        link = f"[open on dev]({r['link']})" if r.get("link") else "(no link)"
        cause = (r.get("diagnosis") or {}).get("cause", "")
        out.append(f"  - {r['card']} r{r['rollout']}: {r.get('verdict')} quality {decide.quality(r):.2f}, {r.get('speed', {}).get('total_s', '?')}s"
                   + (f", cause {cause}" if cause else "") + f" · {link} · `{r['file']}`")
    return out


def changelog(run_dir: Path, state: dict) -> str:
    v = state["versions"]
    lines = [f"# Skill changelog: {state['skill']}", "", f"Run `{run_dir.name}` · status **{state['status']}** · skill delivered as: {state['delivery']}",
             f"Judge `{state['judge_model']}`, editor `{state['editor_model']}`, account {state['account'] or '-'}.", ""]
    if "v0" in v:
        lines += [f"## v0 ({v['v0'].get('outcome', 'start')})", "", *SUMMARY_HEAD]
        for split, s in (v["v0"].get("summary") or {}).items():
            lines.append(_summary_row(split, s))
        lines += ["", f"Causes of train failures: {v['v0'].get('causes') or '-'}", ""]
    for rnd in state["rounds"]:
        d = rnd.get("decision") or {}
        lines += [f"## Round {rnd['n']}: {'KEPT' if d.get('keep') else ('rejected' if d else 'in progress')}", "",
                  f"Target: **{rnd['target']}** ({rnd['group_size']} train conversation(s)); best before: {rnd['best_before']}.",
                  f"Decision: {d.get('why', '(not decided)')}", ""]
        for c in rnd["candidates"]:
            e = c["edit"]
            ver = c.get("version")
            outcome = v.get(ver, {}).get("outcome", "") if ver else f"dropped by a guard: {c['guard']}"
            lines += [f"### {ver or '(no version)'}: {e['reason']}", "", f"Outcome: {outcome or 'screened, not chosen'}", ""]
            p = run_dir / "versions" / f"{ver}.diff" if ver else None
            if p and p.exists():
                lines += ["```diff", p.read_text(encoding="utf-8").rstrip(), "```", ""]
            elif not ver:
                lines += ["```", f"{e['kind']} {e['find'][:300]!r}\n→ {e['text'][:600]}", "```", ""]
            if c.get("train"):
                lines += ["Train screening:", "", *SUMMARY_HEAD, _summary_row(ver, c["train"]), ""]
        if rnd.get("holdout"):
            best_h = (v.get(rnd["best_before"], {}).get("summary") or {}).get("holdout")
            pw = [p["verdict"] for p in rnd.get("pairwise") or []]
            lines += ["Held-out:", "", *SUMMARY_HEAD, _summary_row(f"{rnd['best_before']} (best before)", best_h), _summary_row(f"{rnd['chosen']} (candidate)", rnd["holdout"]), "",
                      f"Pairwise vs best: {pw.count('better')} better, {pw.count('equal')} equal, {pw.count('worse')} worse.", "",
                      "Held-out conversations:", *_links(run_dir, rnd["chosen"], "holdout"), ""]
    return "\n".join(lines) + "\n"


def _worst(run_dir: Path, version: str, n: int = 3) -> list[dict]:
    res = [r for s in ("holdout", "train") for r in rollouts.load_results(run_dir, version, s) if decide.scorable(r)]
    return sorted(res, key=decide.quality)[:n]


def morning(run_dir: Path, state: dict, inputs) -> str:
    v, best = state["versions"], state.get("best") or "v0"
    kept = [r for r in state["rounds"] if (r.get("decision") or {}).get("keep")]
    lines = [f"# Refine report: {state['skill']}", "", f"Status: **{state['status']}**. Best version: **{best}** after {len(state['rounds'])} round(s), {len(kept)} kept.",
             f"Skill delivered as: {state['delivery']}. Started {state['started']}.", ""]
    if state.get("halt"):
        lines += [f"**HALTED:** {state['halt']}", ""]
    lines += ["## Scores by round", "", "| round | target | candidate | train (screen) | held-out best → candidate | pass | mean s | pairwise b/e/w | decision |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in state["rounds"]:
        bh = (v.get(r["best_before"], {}).get("summary") or {}).get("holdout") or {}
        h = r.get("holdout") or {}
        tr = next((c["train"] for c in r["candidates"] if c.get("version") == r.get("chosen")), {}) or {}
        pw = [p["verdict"] for p in r.get("pairwise") or []]
        d = r.get("decision") or {}
        lines.append(f"| {r['n']} | {r['target']} | {r.get('chosen', '-')} | {_fmt(tr.get('quality'))} | {_fmt(bh.get('quality'))} → {_fmt(h.get('quality'))} | "
                     f"{_fmt(h.get('pass_rate'), True)} | {_fmt(h.get('mean_total_s'))} | {pw.count('better')}/{pw.count('equal')}/{pw.count('worse')} | "
                     f"{'KEPT' if d.get('keep') else 'rejected'}: {d.get('why', '-')} |")
    lines += ["", "## Final test (scored once, never used to choose)", ""]
    if state.get("final"):
        lines += [*SUMMARY_HEAD, *(_summary_row(k, s) for k, s in sorted(state["final"].items())), ""]
    else:
        lines += ["Not run (no final cards, or the run stopped early).", ""]
    lines += ["## Why train conversations failed", ""] + [f"- {x}: {v.get(x, {}).get('causes') or '-'}" for x in dict.fromkeys(["v0", best])] + [""]
    lines += ["## The 3 worst conversations of the best version", ""]
    for r in _worst(run_dir, best):
        lines += [f"- **{r['card']}** ({r['split']} r{r['rollout']}): {r.get('verdict')}, quality {decide.quality(r):.2f}. "
                  + (f"[open on dev]({r['link']})" if r.get("link") else "") + f" · `{r['file']}`", f"  - judge: {r.get('judge_reasoning', '')[:400]}"]
    lines += ["", "## Cost", "", f"Our model calls: ${state['meter'].get('spent_usd', 0):.2f} ({state['meter'].get('by_role') or {}}); "
              f"{state['meter'].get('conversations', 0)} conversations with Hark. Hark's own model cost on dev is not visible to this tool."
              + (f" {state['meter']['calls_without_cost']} call(s) came back without a cost." if state["meter"].get("calls_without_cost") else ""), ""]
    lines += ["## Blockers and things a human should look at", ""]
    blockers = list(state.get("notes") or [])
    if state["status"] != "done":
        blockers.append(f"the run did not finish: {state['status']}")
    if "paste" in state["delivery"]:
        blockers.append("the skill was pasted, not loaded through Hark's skill list, so results may overstate how well it does live")
    if state.get("cleanup"):
        blockers.append(f"cleanup: {state['cleanup']}")
    claims = [c for s in ("train", "holdout", "final") for vv in v for r in rollouts.load_results(run_dir, vv, s) for c in r.get("claims") or []]
    if claims:
        blockers.append(f"{len(claims)} reply(ies) claimed a completed pay/book/send step; check them: " + "; ".join(claims[:3]))
    lines += [f"- {b}" for b in blockers] or ["- none"]
    lines += ["", f"## Final skill text ({best})", "", "```markdown", (run_dir / "versions" / f"{best}.md").read_text(encoding="utf-8").rstrip() if (run_dir / "versions" / f"{best}.md").exists() else "", "```", ""]
    if best != "v0" and (run_dir / "versions" / "v0.md").exists():
        lines += ["## Whole diff, v0 → best", "", "```diff",
                  editor.diff((run_dir / "versions" / "v0.md").read_text(encoding="utf-8"), (run_dir / "versions" / f"{best}.md").read_text(encoding="utf-8"), "v0.md", f"{best}.md").rstrip(),
                  "```", ""]
    lines += ["See changelog.md for every version tried, with its reason and the conversations behind it.", ""]
    return "\n".join(lines)


def write(run_dir: Path, state: dict, inputs) -> Path:
    (run_dir / "changelog.md").write_text(changelog(run_dir, state), encoding="utf-8")
    path = run_dir / "report.md"
    path.write_text(morning(run_dir, state, inputs), encoding="utf-8")
    return path
