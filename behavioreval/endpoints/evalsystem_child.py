"""One assistant conversation as one account, driven line by line, from inside mvp's
eval_system uv project (``uv run python <this file>``, cwd = services/eval_system).
Started by behavioreval.endpoints.evalsystem. Why a child: eval_system's TextEvalClient follows
the conversation's SSE stream, so it knows when a run ended and which tools it called
without the trajectory viewer (which indexes only a few configured accounts).

Env: EVAL_SESSION_COOKIE (a pre-minted session). Stdin: one JSON per line,
{"op": "send", "text": ...} or {"op": "close"}. Stdout: one JSON per op — "open" first
(conversation_id, lane note, profile note), a "turn" per send (reply, reason, calls,
duration_s, held_s, error), "closed" last. Diagnostics on stderr. At close the client's
own trajectory (judge-shaped events) goes to --out.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time



def _tool_call(tc: dict) -> dict:
    name = tc.get("toolName", tc.get("name", "unknown"))
    result = tc.get("result")
    if not isinstance(result, str):
        result = json.dumps(result, ensure_ascii=False) if result is not None else ""
    return {"name": name, "input": tc.get("input", tc.get("arguments", {})) or {}, "result": result[:4000],
            "is_error": bool(tc.get("error")) or str(tc.get("status", "")).lower() in ("error", "failed")}


async def amain(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="evalsystem_child")
    ap.add_argument("--env", default="dev")
    ap.add_argument("--email", required=True)
    ap.add_argument("--model", default="", help="per-message model override (agent-llm for a lane, claude-opus-5-5, ...); empty = the product default")
    ap.add_argument("--provider", default="", help="modelProvider for the override (hark-slotN for a lane)")
    ap.add_argument("--title", default="")
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--out", required=True)
    ap.add_argument("--profile", default="", help="JSON of profile fields for USER.md (seed_profile)")
    a = ap.parse_args(argv)
    cookie = os.environ.get("EVAL_SESSION_COOKIE", "").strip()
    if not cookie:
        print("evalsystem_child: EVAL_SESSION_COOKIE is not set", file=sys.stderr)
        return 2
    from eval_system.clients import TextEvalClient
    from eval_system.environments import get_environment
    env = get_environment(a.env)

    def emit(obj: dict) -> None:
        print(json.dumps(obj, ensure_ascii=False), flush=True)

    async with TextEvalClient(api_url=env.web_url, session_cookie=cookie, account_email=a.email, scenario_name="behavioreval",
                              response_timeout=a.timeout, model=a.model or None, model_provider=a.provider or None) as client:
        await client.rotate_conversation()   # a fresh main conversation (local mvp patch); memory is per account and stays
        note = (f"model={a.model}" + (f" modelProvider={a.provider}" if a.provider else "") + " on every message") if a.model else "endpoint default model"
        profile_note = ""
        if a.profile:
            fields = {k: str(v) for k, v in json.loads(a.profile).items() if v}
            try:
                await client.seed_profile(fields)
                profile_note = "seeded " + ", ".join(fields)
            except Exception as exc:
                profile_note = f"NOT seeded: {type(exc).__name__}: {str(exc)[:160]}"
        emit({"op": "open", "conversation_id": client.conversation_id, "note": note, "profile": profile_note})
        for raw in sys.stdin:
            raw = raw.strip()
            if not raw:
                continue
            try:
                req = json.loads(raw)
            except ValueError:
                emit({"op": "error", "detail": f"not JSON: {raw[:80]}"})
                continue
            if req.get("op") == "close":
                break
            if req.get("op") != "send":
                emit({"op": "error", "detail": f"unknown op {req.get('op')!r}"})
                continue
            t0 = time.monotonic()
            try:
                turn = await client.send_and_wait(str(req.get("text") or ""), timeout=a.timeout)
            except Exception as exc:
                emit({"op": "turn", "reply": "", "reason": "", "calls": [], "duration_s": round(time.monotonic() - t0, 1), "held_s": 0,
                      "error": f"{type(exc).__name__}: {str(exc)[:300]}"})
                continue
            # The client returns when the reply landed; the run may still wrap up (links, ledger,
            # a trailing update) and a message sent now would join it. Hold until it closed.
            try:
                held = await client.hold_for_open_run(bound_seconds=a.timeout)
            except Exception as exc:
                print(f"evalsystem_child: hold_for_open_run: {type(exc).__name__}: {exc}", file=sys.stderr)
                held = 0.0
            finishes = [f for f in client.result.loop_finishes if not turn.agent_id or f.get("agent_id") == turn.agent_id]
            reason = str((finishes or client.result.loop_finishes or [{}])[-1].get("reason") or "")
            if not reason:   # dev's stream showed no loop_finished (2026-10-20): say what the turn looked like, starred
                reason = "error" if turn.error else ("end_turn*" if turn.assistant_text else ("tools_only*" if turn.tool_calls else "no_reply*"))
            emit({"op": "turn", "reply": turn.assistant_text or "", "reason": reason,
                  "calls": [_tool_call(tc) for tc in turn.tool_calls if isinstance(tc, dict)],
                  "duration_s": round(turn.duration_seconds or (time.monotonic() - t0), 1), "held_s": round(held, 1), "error": turn.error or ""})
        try:
            with open(a.out, "w", encoding="utf-8") as f:
                json.dump(client.result.build_trajectory(), f, ensure_ascii=False)
        except Exception as exc:
            print(f"evalsystem_child: trajectory not written: {type(exc).__name__}: {exc}", file=sys.stderr)
        emit({"op": "closed", "conversation_id": client.conversation_id, "trajectory": a.out})
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(amain()))
