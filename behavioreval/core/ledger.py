"""What each account has done: one entry per conversation (scenario, column, when, how it
ended). Lives under the data root, never in the repo. Used to tell the user model what
this account already asked the assistant for (``history``) and to pick a rested account."""
from __future__ import annotations

import json
import time
from pathlib import Path


def path_for(data_root: Path) -> Path:
    return data_root / "ledger.json"


def load(path: Path) -> dict[str, list[dict]]:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except ValueError:
        return {}


def record(path: Path, account: str, entry: dict) -> None:
    led = load(path)
    led.setdefault(account, []).append({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), **entry})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(led, indent=1, ensure_ascii=False), encoding="utf-8")


def history(led: dict[str, list[dict]], account: str, limit: int = 5) -> list[str]:
    rows = led.get(account, [])[-limit:]
    return [f"{e.get('time', '')[:10]}: asked for \"{e.get('opening', e.get('scenario', '?'))[:80]}\"" + (f" (ended: {e['stop']})" if e.get("stop") else "")
            for e in rows]


def least_used_first(led: dict[str, list[dict]], accounts: list[str]) -> list[str]:
    return sorted(accounts, key=lambda a: max((e.get("time", "") for e in led.get(a, [])), default=""))
