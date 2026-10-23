"""The trajectory viewer (behind Cloudflare Access) is the clock and the source of tool calls:
a conversation's runs appear there within ~10 s of ending, with their reason and provider;
an export gives every tool call of a run. ``cloudflared access login <viewer>`` once."""
from __future__ import annotations

import json
import shutil
import subprocess
import time
import urllib.parse
import urllib.request

BROWSER_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"


def headers(viewer_url: str) -> dict[str, str]:
    cf = shutil.which("cloudflared")
    if not cf:
        raise RuntimeError("cloudflared is not installed (brew install cloudflared); the trajectory viewer sits behind Cloudflare Access")
    token = subprocess.run([cf, "access", "token", "--app", viewer_url], capture_output=True, text=True).stdout.strip()
    if not token or token.startswith("Unable"):
        raise RuntimeError(f"no Cloudflare Access token for the viewer; run once: cloudflared access login {viewer_url}")
    return {"cf-access-token": token, "User-Agent": BROWSER_UA}


def get(viewer_url: str, path: str, hdrs: dict[str, str]) -> str:
    req = urllib.request.Request(viewer_url + path, headers=hdrs)
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.read().decode(errors="replace")
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"viewer {path} answered HTTP {exc.code}: {exc.read()[:160]!r}") from None


def conversation_runs(viewer_url: str, env: str, conv: str, hdrs: dict[str, str], folder: str = "mine") -> list[dict]:
    """``folder`` is the viewer's account folder: ``mine`` = the account the viewer was signed in with; a dev account
    that differs from it (anderson1@hark.com while signed in as anderson.breazeale@) is its own folder, named by its email."""
    return json.loads(get(viewer_url, f"/api/{env}/conversations/{folder}/{conv}/runs", hdrs)).get("runs", [])


def folder_for(viewer_url: str, env: str, conv: str, hdrs: dict[str, str], email: str) -> str:
    """The folder this conversation's runs will appear in: the account's own folder when the viewer knows it, else ``mine``."""
    if email:
        try:
            get(viewer_url, f"/api/{env}/conversations/{email}/{conv}/runs", hdrs)
            return email
        except RuntimeError:
            pass
    return "mine"


def wait_for_runs(viewer_url: str, env: str, conv: str, expected: int, hdrs: dict[str, str], timeout_s: float,
                  sleep=time.sleep, now=time.monotonic, fetch=conversation_runs, folder: str = "mine") -> list[dict]:
    """The conversation's runs once ``expected`` of them have ended (a non-empty reason)."""
    t0 = now()
    while True:
        runs = fetch(viewer_url, env, conv, hdrs, folder)
        done = [r for r in runs if r.get("reason")]
        if len(done) >= expected:
            return sorted(runs, key=lambda r: r.get("start", 0))
        if now() - t0 > timeout_s:
            raise TimeoutError(f"{len(done)} of {expected} runs ended within {timeout_s:.0f} s (viewer shows {len(runs)} run(s))")
        sleep(10)


def settled(viewer_url: str, env: str, conv: str, hdrs: dict[str, str], timeout_s: float, sleep=time.sleep, now=time.monotonic, fetch=conversation_runs, folder: str = "mine") -> list[dict]:
    """Hold until every run of the conversation has ended and nothing new started for 8 s (a browser
    subagent or a wake-up run may still be working on the last request)."""
    t0 = now()
    runs: list[dict] = []
    while now() - t0 < timeout_s:
        runs = fetch(viewer_url, env, conv, hdrs, folder)
        if runs and all(r.get("reason") for r in runs):
            sleep(8)
            again = fetch(viewer_url, env, conv, hdrs, folder)
            if len(again) == len(runs) and all(r.get("reason") for r in again):
                return sorted(again, key=lambda r: r.get("start", 0))
        sleep(10)
    return sorted(runs, key=lambda r: r.get("start", 0))


def rollouts(viewer_url: str, env: str, refs: list[str], hdrs: dict[str, str]) -> list[dict]:
    body = urllib.parse.urlencode({"format": "rollouts", "runs": ",".join(refs), "subagents": "0", "prompt": "0"}).encode()
    req = urllib.request.Request(f"{viewer_url}/api/{env}/export", data=body, method="POST", headers={**hdrs, "Content-Type": "application/x-www-form-urlencoded"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        text = resp.read().decode(errors="replace")
    return [json.loads(l) for l in text.splitlines() if l.strip()]


def calls_of(lines: list[dict]) -> list[dict]:
    """Tool calls (name, input) of the main lane of an exported run; results are not in the export."""
    calls = []
    for line in lines:
        lane_msgs = next((l for l in line.get("lanes") or [] if not l.get("dispatch_path")), {}).get("messages") or []
        for m in lane_msgs:
            for b in m.get("content") or []:
                if m.get("part") == "run" and isinstance(b, dict) and b.get("type") == "tool_use":
                    calls.append({"name": b.get("name", ""), "input": b.get("input") or {}, "result": "", "is_error": False})
    return calls
