"""What a run leaves behind, under ``<data_root>/runs/<time>-<tag>/``: one
``<column>__<account>__<scenario>.conversation.json`` per conversation (turns, the user
model's side, the score, any human annotation), ``report.md`` and ``summary.json``.
Results never enter the repo."""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict
from pathlib import Path

from behavioreval.core.harness import Conversation, Turn


def run_dir(data_root: Path, tag: str) -> Path:
    tag = re.sub(r"[^A-Za-z0-9_.-]+", "-", tag or "run").strip("-") or "run"
    d = data_root / "runs" / f"{time.strftime('%Y%m%d-%H%M')}-{tag}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def file_name(column: str, account: str, scenario: str, attempt: int = 0) -> str:
    acct = (account or "noaccount").split("@")[0]
    return f"{(column or 'default').replace('/', '_')}__{acct}__" + scenario.replace("/", "_") + (f".run{attempt}" if attempt else "") + ".conversation.json"


def save_conversation(d: Path, conv: Conversation, score: dict | None = None, attempt: int = 0) -> Path:
    path = d / file_name(conv.column, conv.account, conv.scenario, attempt)
    payload = asdict(conv)
    payload["score"] = score or {}
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    return path


def load_conversation(path: Path) -> tuple[Conversation, dict]:
    d = json.loads(path.read_text(encoding="utf-8"))
    conv = Conversation(**{k: d.get(k, "") for k in ("scenario", "user", "account", "profile", "lane", "column", "conversation_id", "stop", "started", "ended", "note")})
    conv.tags = list(d.get("tags") or [])
    conv.turns = [Turn(**{k: t.get(k) for k in ("n", "user", "assistant", "reason", "calls", "duration_s", "error", "sim")}) for t in d.get("turns") or []]
    return conv, d


def _verdict(score: dict) -> str:
    return str(score.get("verdict") or "-")


def write_report(d: Path, rows: list[tuple[Conversation, dict]], cfg_summary: dict) -> Path:
    lines = [f"# behavioreval run {d.name}", "", "Settings: " + ", ".join(f"{k}={v}" for k, v in cfg_summary.items()), ""]
    cols = sorted({c.column or "default" for c, _ in rows})
    if rows:
        lines += ["| column | conversations | done | max_turns | timeout | error | PASS | FAIL | pass rate |", "|---|---|---|---|---|---|---|---|---|"]
        for col in cols:
            rs = [(c, s) for c, s in rows if (c.column or "default") == col]
            n = len(rs)
            stops = {k: sum(1 for c, _ in rs if c.stop == k) for k in ("done", "max_turns", "timeout")}
            errs = sum(1 for c, _ in rs if c.stop.startswith("error") or c.stop == "assistant_error")
            p = sum(1 for _, s in rs if _verdict(s) == "PASS"); f = sum(1 for _, s in rs if _verdict(s) == "FAIL")
            rate = f"{p / (p + f):.0%}" if p + f else "-"
            lines.append(f"| {col} | {n} | {stops['done']} | {stops['max_turns']} | {stops['timeout']} | {errs} | {p} | {f} | {rate} |")
        lines.append("")
    lines += ["| column | scenario | account | user turns | stop | score | note |", "|---|---|---|---|---|---|---|"]
    for conv, score in rows:
        lines.append(f"| {conv.column or 'default'} | {conv.scenario} | {conv.account or '-'} | {conv.user_turns} | {conv.stop} | {_verdict(score)} | {conv.note[:60] or '-'} |")
    lines.append("")
    for conv, score in rows:
        lines += [f"## [{conv.column or 'default'}] {conv.scenario} on {conv.account or '-'}", ""]
        for t in conv.turns:
            why = t.sim.get("why", "") if t.sim else ""
            lines.append(f"- **user {t.n}:** {t.user or '(done, nothing sent)'}" + (f"  _why: {why}_" if why else ""))
            if t.user:
                names = ", ".join(str(c.get("name", "?")) for c in t.calls)
                lines.append(f"  - assistant ended `{t.reason or '?'}` ({len(t.calls)} calls{': ' + names if names else ''}, {t.duration_s:.0f} s): {t.assistant[:300] or '(no reply)'}" + (f"; error: {t.error}" if t.error else ""))
        for line in score.get("lines") or []:
            lines.append(f"- {line}")
        lines.append("")
    path = d / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    (d / "summary.json").write_text(json.dumps({"run": d.name, "settings": cfg_summary,
        "conversations": [{"column": c.column or "default", "scenario": c.scenario, "account": c.account, "conversation_id": c.conversation_id,
                           "user_turns": c.user_turns, "stop": c.stop, "verdict": _verdict(s), "score": s} for c, s in rows]},
        indent=1, ensure_ascii=False), encoding="utf-8")
    return path
