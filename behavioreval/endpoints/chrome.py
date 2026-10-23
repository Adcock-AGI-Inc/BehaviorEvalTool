"""The product on dev in the tester's own Chrome, as their own account. The browser is the
session holder: every API call is a same-origin fetch inside the page (cookies ride along,
no token leaves the browser), the window is navigated to the conversation so the person
can watch, and the trajectory viewer says when a run ended and which tools it called.
Nothing is written to the account's profile page (it is a real person's account)."""
from __future__ import annotations

import json
import time
import uuid

from behavioreval.browser import cdp, viewer
from behavioreval.core.config import Config, column
from behavioreval.core.endpoint import AssistantTurn

ENVS = {"dev": "https://dev.hark.com", "staging": "https://staging.hark.com", "prod": "https://app.hark.com"}


def web_url(cfg: Config) -> str:
    return cfg.web_url or ENVS.get(cfg.env, ENVS["dev"])


def fetch_js(url: str, method: str, body=None) -> str:
    return ("(async () => { const r = await fetch(%s, {method: %s, credentials: 'include', headers: {'Content-Type': 'application/json'}, body: %s}); "
            "const t = await r.text(); let j = null; try { j = JSON.parse(t); } catch (e) {} return {status: r.status, json: j, text: t.slice(0, 400)}; })()"
            % (json.dumps(url), json.dumps(method), "null" if body is None else json.dumps(json.dumps(body))))


def text_of(row: dict) -> str:
    c = row.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return " ".join(str(p.get("text", "")) for p in c if isinstance(p, dict))
    return str(row.get("text") or "")


def delivered(row: dict) -> bool:
    return row.get("role") == "assistant" and row.get("source") == "send_bubble" and bool(text_of(row).strip())


class ChromeEndpoint:
    def __init__(self, cfg: Config, out=print, tab=None, hdrs=None):
        self.cfg, self.out = cfg, out
        self.web = web_url(cfg)
        self.port = cfg.chrome_port
        self.tab = tab
        self.hdrs = hdrs
        self.conv = ""
        self.seen: set = set()
        self.sent = 0
        self.account = ""
        self.folder = "mine"

    # ── page side ────────────────────────────────────────────────────────
    def api(self, method: str, path: str, body=None):
        out = self.tab.js(fetch_js(self.web + path, method, body))
        if not isinstance(out, dict):
            raise RuntimeError(f"fetch {path} gave no result (not on a Hark page?)")
        return out.get("status"), (out.get("json") if out.get("json") is not None else out.get("text"))

    def _signed_in(self) -> None:
        """Load the app in our tab and prove the session: a Cloudflare / Google page means the copied profile is not
        signed in; a slow first load gets 90 s. On failure the tab is left open so the person can see what is on it."""
        self.tab.navigate(self.web + "/")
        last = {}
        for _ in range(180):
            time.sleep(0.5)
            last = self.tab.js("({url: location.href, title: document.title})") or {}
            url = str(last.get("url", ""))
            if "cloudflareaccess.com" in url or "accounts.google.com" in url:
                self.tab = None
                raise RuntimeError(f"this Chrome is not signed in to {self.web}: the page went to {url.split('?')[0]}. Sign in to dev in your everyday Chrome, then Start my Chrome again (it re-copies the profile)")
            if url.startswith(self.web):
                status, _ = self.api("GET", "/api/connectors")
                if status == 200:
                    return
                if status in (401, 403):
                    self.tab = None
                    raise RuntimeError(f"{self.web} answered HTTP {status}: not signed in. Sign in to dev in your everyday Chrome, then Start my Chrome again")
        self.tab = None
        raise RuntimeError(f"the app did not load within 90 s (tab shows {str(last.get('url', ''))[:80]!r}, {str(last.get('title', ''))[:60]!r}); the tab was left open")

    def rows(self) -> list[dict]:
        status, data = self.api("GET", f"/api/messages/page?conversationId={self.conv}&limit=100")
        msgs = data.get("messages") if status == 200 and isinstance(data, dict) else None
        return msgs if isinstance(msgs, list) else []

    def open(self, account: str, lane: str, title: str, profile: dict) -> str:
        self.account = account or ""
        if profile:
            self.out("  (chrome endpoint: the profile page of a real account is never written)")
        if self.tab is None:
            if not cdp.up(self.port):
                raise RuntimeError(f"no Chrome answering CDP on 127.0.0.1:{self.port}: start it with `python3 -m behavioreval chrome` (or the Setup tab)")
            self.tab = cdp.attach(self.port, self.web)
        if self.hdrs is None:
            self.hdrs = viewer.headers(self.cfg.viewer_url)
        self._signed_in()
        status, data = self.api("POST", "/api/conversations", {"title": title, "timezone": self.cfg.timezone, "autoTitle": False})
        if status != 200 or not isinstance(data, dict) or not data.get("conversationId"):
            raise RuntimeError(f"conversations.create HTTP {status}: {str(data)[:200]}")
        self.conv = data["conversationId"]
        try:
            self.tab.navigate(f"{self.web}/project/{self.conv}")
        except Exception:
            pass
        # Which viewer folder this account's runs land in: its own (anderson1@hark.com) or 'mine' (the viewer login itself).
        self.folder = viewer.folder_for(self.cfg.viewer_url, self.cfg.env, self.conv, self.hdrs, self.account)
        if self.folder != "mine":
            self.out(f"  viewer folder: {self.folder}")
        label, model, provider = column(self.cfg)
        self.note = f"model={model}" + (f" modelProvider={provider}" if provider else "") + " on every message" if model else "the product's default model"
        if self.cfg.lane:
            # Step 1 for a lane: the /model slash command, the same switch a person types (the worker handles it before
            # the agent loop; the account needs the clear_command_allowlist flag). Its reply is not stored on the page
            # and it starts no run, so it is not counted. The per-message override below is sent as well.
            cmd = f"/model hark-{self.cfg.lane}"
            status, data = self.api("POST", "/api/messages/send", {"conversationId": self.conv, "message": cmd, "idempotencyKey": str(uuid.uuid4())})
            if status != 200:
                raise RuntimeError(f"{cmd} was refused: HTTP {status}: {str(data)[:160]}")
            self.out(f"  sent {cmd}")
            self.note = f"sent {cmd}; " + self.note
            time.sleep(3)
        self.seen = {r.get("id") for r in self.rows() if r.get("id")}
        return self.conv

    def turn(self, text: str, timeout_s: float) -> AssistantTurn:
        label, model, provider = column(self.cfg)
        body = {"conversationId": self.conv, "message": text, "idempotencyKey": str(uuid.uuid4())}
        if model:
            body["model"] = model
            if provider:
                body["modelProvider"] = provider
        status, data = self.api("POST", "/api/messages/send", body)
        if status != 200:
            raise RuntimeError(f"messages.send HTTP {status}: {str(data)[:200]}")
        self.sent += 1
        t0 = time.monotonic()
        if self.sent == 1:   # the viewer must list this conversation at all; otherwise every turn would wait the full timeout
            seen_any = False
            while time.monotonic() - t0 < 150:
                if viewer.conversation_runs(self.cfg.viewer_url, self.cfg.env, self.conv, self.hdrs, self.folder):
                    seen_any = True
                    break
                time.sleep(10)
            if not seen_any:
                raise RuntimeError(f"the trajectory viewer lists no runs for this conversation after 150 s (looked in folder '{self.folder}'). Either the dev account "
                                   f"{self.account or '(none set)'} is not recorded by the viewer, or the viewer login is a different account: set your dev email in "
                                   "Set up, and run `cloudflared access login https://trajectory.tools.hark.com`. (The message did reach the assistant; only the tool "
                                   "cannot see when it finished.)")
        runs = viewer.wait_for_runs(self.cfg.viewer_url, self.cfg.env, self.conv, self.sent, self.hdrs, timeout_s, folder=self.folder)
        runs = viewer.settled(self.cfg.viewer_url, self.cfg.env, self.conv, self.hdrs, max(30.0, timeout_s - (time.monotonic() - t0)), folder=self.folder)
        last = runs[-1] if runs else {}
        calls: list[dict] = []
        if last.get("calls") and last.get("job"):
            try:
                calls = viewer.calls_of(viewer.rollouts(self.cfg.viewer_url, self.cfg.env, [f"{self.folder}/{self.conv}/{last['job']}"], self.hdrs))
            except Exception as exc:
                self.out(f"    (rollout not read: {type(exc).__name__})")
        current = self.rows()
        reply = " ".join(text_of(r) for r in current if r.get("id") not in self.seen and delivered(r))
        self.seen = {r.get("id") for r in current if r.get("id")}
        served = ",".join(last.get("providers") or [])
        reason = str(last.get("reason") or "")
        return AssistantTurn(reply=reply, reason=reason, calls=calls, duration_s=round((last.get("durMs") or 0) / 1000, 1),
                             error="" if reply or calls else f"run ended as {reason or 'unknown'} with no reply",
                             note=(self.note + (f"; served by {served}" if served else "")) if self.sent == 1 else "")

    def close(self) -> None:
        if self.tab:
            tid = getattr(self.tab, "target_id", "")
            self.tab.close()
            if tid:
                cdp.close_target(self.port, tid)   # the tab was ours; the window stays


def check(cfg: Config) -> list[tuple[str, bool, str]]:
    import shutil
    from pathlib import Path
    from behavioreval.browser import chrome
    cf = shutil.which("cloudflared")
    token_ok = False
    if cf:
        import subprocess
        t = subprocess.run([cf, "access", "token", "--app", cfg.viewer_url], capture_output=True, text=True).stdout.strip()
        token_ok = bool(t) and not t.startswith("Unable")
    profiles = chrome.list_profiles(cfg.chrome_dir)
    return [("Google Chrome", Path(chrome.CHROME_BIN).is_file(), chrome.CHROME_BIN),
            ("Chrome profiles found", bool(profiles), ", ".join(f"{p['dir']} ({p['email'] or p['name']})" for p in profiles) or f"nothing under {cfg.chrome_dir}"),
            ("the tool's Chrome is up", cdp.up(cfg.chrome_port), f"CDP 127.0.0.1:{cfg.chrome_port}" if cdp.up(cfg.chrome_port) else "start it: python3 -m behavioreval chrome, or the Setup tab"),
            ("cloudflared", bool(cf), cf or "brew install cloudflared"),
            ("viewer login", token_ok, "ok" if token_ok else f"run once: cloudflared access login {cfg.viewer_url}")]


def create(name: str, cfg: Config, out=print):
    return ChromeEndpoint(cfg, out)
