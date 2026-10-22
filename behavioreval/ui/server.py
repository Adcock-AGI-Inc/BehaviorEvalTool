"""The web UI: a small local server (stdlib only) over the same files the command line uses.
Profiles are ``profiles/*.json``, scenarios ``scenarios/*.jsonl``; runs are started as
``python3 -m behavioreval run`` subprocesses; results are read from ``<data_root>/runs/<run>/``
and human annotations are written back into each ``*.conversation.json`` (``annotation``).
Binds 127.0.0.1 only."""
from __future__ import annotations

import csv
import io
import json
import os
import re
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from behavioreval.core import config as config_mod, ledger, openrouter, profile as profile_mod, scenario, sync
from behavioreval.core.live import LiveChat
from behavioreval.core.config import ROOT, Config

HERE = Path(__file__).resolve().parent
SAFE = re.compile(r"^[A-Za-z0-9_.@-]+$")
ISSUES = ["fabricated specifics", "acted without asking", "ignored a limit", "wrong or missing deliverable", "asked unnecessary questions",
          "loop / repeated tool calls", "run failed (product error)", "user model misbehaved", "judge wrong"]


class State:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.procs: dict[str, dict] = {}   # run tag → {proc, log, cmd, started}
        self.lock = threading.Lock()
        self.live: LiveChat | None = None   # one live conversation at a time (one browser)

    # ── live chat ────────────────────────────────────────────────────────
    def live_start(self, req: dict) -> dict:
        if self.live and self.live.state not in ("finished", "error"):
            raise ValueError("a live conversation is still going; end it first")
        pid = str(req.get("profile") or self.cfg.profile)
        person = self.person(pid)
        sc = next((s for s in scenario.resolve("all", directory=self.cfg.scenarios_dir) if s.id == req.get("scenario")), None)
        if sc is None:
            raise ValueError(f"no scenario {req.get('scenario')!r}")
        problems = scenario.lint(sc, person)
        if problems:
            raise ValueError("; ".join(problems))
        cfg = config_mod.with_column(self.cfg, str(req.get("column") or "default"))
        if req.get("max_turns"):
            from dataclasses import replace
            cfg = replace(cfg, max_turns=int(req["max_turns"]))
            sc = scenario.from_dict({**sc.to_dict(), "max_turns": int(req["max_turns"])}, "live")
        if not openrouter.key():
            raise ValueError("no OPENROUTER_API_KEY: the user model and the judge need it")
        self.live = LiveChat(cfg, sc, person, endpoint_name=str(req.get("endpoint") or cfg.endpoint))
        self.live.auto = bool(req.get("auto"))
        self.live.start()
        return self.live.snapshot()

    def live_state(self) -> dict:
        return self.live.snapshot() if self.live else {"state": "idle"}

    def live_act(self, req: dict) -> dict:
        if not self.live:
            raise ValueError("no live conversation")
        act = str(req.get("action") or "")
        if act == "send":
            self.live.send(str(req.get("text") or ""), bool(req.get("end")))
        elif act == "end":
            self.live.send("", True)
        elif act == "auto":
            self.live.set_auto(bool(req.get("on", True)))
        else:
            raise ValueError(f"unknown action {act!r}")
        return self.live.snapshot()

    # ── profiles ─────────────────────────────────────────────────────────
    def profiles(self) -> dict[str, profile_mod.Profile]:
        return profile_mod.load_all(self.cfg.profiles_dir)

    def person(self, pid: str = "") -> profile_mod.Profile:
        return profile_mod.load(self.cfg.profiles_dir, pid or self.cfg.profile)

    def all_accounts(self) -> list[str]:
        return list(config_mod.load_accounts(self.cfg.accounts_file))

    def profile_rows(self) -> list[dict]:
        profiles = self.profiles()
        accounts = self.all_accounts()
        out = []
        for p in profiles.values():
            d = p.to_dict()
            d["name"], d["problems"] = p.name, profile_mod.lint(p, accounts, profiles)
            d["usable_accounts"] = profile_mod.accounts_for(p, accounts, profiles)
            out.append(d)
        return out

    def save_profile(self, d: dict) -> dict:
        p = profile_mod.from_dict(d, "form")
        profiles = self.profiles()
        problems = profile_mod.lint(p, self.all_accounts(), profiles)
        if any("already belongs" in x for x in problems):
            raise ValueError("; ".join(x for x in problems if "already belongs" in x))
        profile_mod.save(self.cfg.profiles_dir, p)
        return {"id": p.id, "problems": problems}

    # ── scenarios ────────────────────────────────────────────────────────
    def scenarios(self, pid: str = "") -> list[dict]:
        u = self.person(pid)
        out = []
        for s in scenario.sets(self.cfg.scenarios_dir):
            try:
                scs = scenario.resolve(s, directory=self.cfg.scenarios_dir)
            except ValueError as exc:
                out.append({"set": s, "id": f"{s}/(file error)", "opening": "", "problems": [str(exc)]})
                continue
            for sc in scs:
                d = sc.to_dict()
                d["set"], d["problems"] = s, scenario.lint(sc, u)
                out.append(d)
        return out

    def save_scenario(self, set_name: str, d: dict, pid: str = "") -> dict:
        sc = scenario.from_dict(d, "form")
        if not sc.id.startswith(set_name + "/"):
            raise ValueError(f"the id must start with the set name: {set_name}/…")
        problems = scenario.lint(sc, self.person(pid))
        scenario.save(self.cfg.scenarios_dir, set_name, sc)
        return {"id": sc.id, "problems": problems}

    # ── runs ─────────────────────────────────────────────────────────────
    def runs_dir(self) -> Path:
        return self.cfg.data_root / "runs"

    def logs_dir(self) -> Path:
        d = self.cfg.data_root / "logs"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def start_run(self, req: dict) -> dict:
        if not req.get("scenarios"):
            raise ValueError("pick at least one scenario")
        pid = str(req.get("profile") or self.cfg.profile)
        self.person(pid)   # must exist
        tag = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(req.get("tag") or pid)).strip("-") or pid
        tag = f"{tag}-{time.strftime('%H%M%S')}"
        cmd = [sys.executable, "-m", "behavioreval", "run", "--yes", "--tag", tag, "--scenarios", "all", "--profile", pid]
        for sid in req["scenarios"]:
            cmd += ["--only", sid]
        cols = [c.strip() for c in (req.get("columns") or []) if c.strip()]
        if cols:
            cmd += ["--columns", ",".join(cols)]
        for a in req.get("accounts") or []:
            cmd += ["--account", a]
        for key, flag in (("runs", "--runs"), ("parallel", "--parallel"), ("max_turns", "--max-turns"), ("score", "--score"), ("endpoint", "--endpoint")):
            if req.get(key) not in (None, "", 0):
                cmd += [flag, str(req[key])]
        if req.get("calibrate_first"):
            cmd.append("--calibrate-first")
        if req.get("cleanup"):
            cmd.append("--cleanup")
        log = self.logs_dir() / f"{tag}.log"
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=open(log, "w"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        with self.lock:
            self.procs[tag] = {"proc": proc, "log": log, "cmd": " ".join(cmd), "started": time.strftime("%Y-%m-%d %H:%M:%S")}
        return {"tag": tag, "log": str(log), "cmd": " ".join(cmd)}

    def stop_run(self, tag: str) -> bool:
        p = self.procs.get(tag)
        if not p or p["proc"].poll() is not None:
            return False
        p["proc"].terminate()
        return True

    def _dir_summary(self, d: Path) -> dict:
        files = sorted(d.glob("*.conversation.json"))
        verdicts: dict[str, int] = {}
        cols: set[str] = set()
        profiles: set[str] = set()
        annotated = 0
        for f in files:
            try:
                j = json.loads(f.read_text(encoding="utf-8"))
            except ValueError:
                continue
            v = (j.get("score") or {}).get("verdict") or "-"
            verdicts[v] = verdicts.get(v, 0) + 1
            cols.add(j.get("column") or "default")
            profiles.add(j.get("profile") or j.get("persona") or "")
            annotated += bool(j.get("annotation"))
        tag = d.name.split("-", 2)[2] if d.name.count("-") >= 2 else d.name
        live = self.procs.get(tag)
        status = "running" if live and live["proc"].poll() is None else ("finished" if live else "")
        return {"name": d.name, "tag": tag, "conversations": len(files), "verdicts": verdicts, "columns": sorted(cols), "profiles": sorted(p for p in profiles if p),
                "annotated": annotated, "status": status, "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(d.stat().st_mtime)),
                "log": str(live["log"]) if live else str(self.logs_dir() / f"{tag}.log")}

    def runs(self) -> list[dict]:
        rd = self.runs_dir()
        out = [self._dir_summary(d) for d in sorted(rd.iterdir(), reverse=True) if d.is_dir()] if rd.exists() else []
        for tag, p in self.procs.items():
            if not any(r["tag"] == tag for r in out):
                out.insert(0, {"name": "(starting)", "tag": tag, "conversations": 0, "verdicts": {}, "columns": [], "profiles": [], "annotated": 0,
                               "status": "running" if p["proc"].poll() is None else "finished", "modified": p["started"], "log": str(p["log"])})
        return out

    def run_detail(self, name: str) -> list[dict]:
        d = self.runs_dir() / name
        rows = []
        for f in sorted(d.glob("*.conversation.json")):
            try:
                j = json.loads(f.read_text(encoding="utf-8"))
            except ValueError:
                continue
            judge = ((j.get("score") or {}).get("judge") or [{}])[0]
            rows.append({"file": f.name, "scenario": j.get("scenario"), "column": j.get("column") or "default", "account": j.get("account"),
                         "profile": j.get("profile") or j.get("persona") or "", "user_turns": sum(1 for t in j.get("turns") or [] if t.get("user")), "stop": j.get("stop"),
                         "verdict": (j.get("score") or {}).get("verdict") or "-", "score": judge.get("score"), "steps": judge.get("steps_completed"),
                         "fabrication": judge.get("fabrication"), "violations": len(judge.get("violations") or []),
                         "asked_first": judge.get("asked_before_acting"), "annotation": j.get("annotation") or {}, "started": j.get("started")})
        return rows

    def conversation(self, name: str, file: str) -> dict:
        return json.loads((self.runs_dir() / name / file).read_text(encoding="utf-8"))

    def annotate(self, name: str, file: str, ann: dict) -> dict:
        f = self.runs_dir() / name / file
        j = json.loads(f.read_text(encoding="utf-8"))
        clean = {"verdict": str(ann.get("verdict") or ""), "issues": [str(x) for x in ann.get("issues") or [] if x],
                 "comment": str(ann.get("comment") or "")[:4000], "by": str(ann.get("by") or "")[:80], "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        if clean["verdict"] not in ("", "PASS", "FAIL", "UNSURE"):
            raise ValueError("verdict must be PASS, FAIL or UNSURE")
        j["annotation"] = clean
        f.write_text(json.dumps(j, indent=1, ensure_ascii=False), encoding="utf-8")
        return clean

    def export_csv(self, name: str) -> str:
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["file", "scenario", "column", "profile", "account", "user_turns", "stop", "judge_verdict", "judge_score", "human_verdict", "issues", "comment", "by", "annotated_at"])
        for r in self.run_detail(name):
            a = r["annotation"]
            w.writerow([r["file"], r["scenario"], r["column"], r["profile"], r["account"], r["user_turns"], r["stop"], r["verdict"], r["score"],
                        a.get("verdict", ""), "; ".join(a.get("issues") or []), a.get("comment", ""), a.get("by", ""), a.get("time", "")])
        return buf.getvalue()

    def preview(self, req: dict) -> dict:
        from behavioreval.core.simulator import Exchange, UserSimulator, system_prompt
        person = self.person(str(req.get("profile") or ""))
        sc = scenario.from_dict(req["scenario"], "form") if req.get("scenario") else next(s for s in scenario.resolve("all", directory=self.cfg.scenarios_dir) if s.id == req["id"])
        problems = scenario.lint(sc, person)
        u = scenario.profile_for(sc, person)
        r = scenario.render(sc, person)
        today = time.strftime("%Y-%m-%d")
        out = {"problems": problems, "opening": r.opening, "system_prompt": system_prompt(u, r, [], today)}
        lines = [l for l in (req.get("assistant") or []) if str(l).strip()]
        if lines and req.get("call_model"):
            key = openrouter.key()
            if not key:
                out["error"] = "no OPENROUTER_API_KEY"
            else:
                transcript, line = [], r.opening
                for a in lines:
                    transcript.append(Exchange(user=line, assistant=str(a), ended="end_turn"))
                    line = "(earlier line)"
                reply = UserSimulator(self.cfg, key).next_message(u, r, transcript, [], today)
                out["next"] = {"say": reply.say, "done": reply.done, "why": reply.why, "unreadable": reply.unreadable}
        return out

    def chrome_start(self, profile_directory: str) -> dict:
        from behavioreval.browser import chrome
        lines: list[str] = []
        chrome.start(self.cfg.chrome_dir, profile_directory, self.cfg.chrome_profile_dir, self.cfg.chrome_port, lines.append)
        return {"ok": True, "log": lines}

    def state(self) -> dict:
        from behavioreval.browser import cdp, chrome
        from behavioreval.core.config import secrets
        if self.cfg.endpoint == "chrome":
            from behavioreval.endpoints.chrome import check
        else:
            from behavioreval.endpoints.evalsystem import check
        led = ledger.load(ledger.path_for(self.cfg.data_root))
        profiles = self.profiles()
        bound = profile_mod.bound_accounts(profiles)
        return {"config": {k: str(getattr(self.cfg, k)) for k in ("user_model", "judge_model", "max_turns", "endpoint", "env", "viewer_url", "chrome_port", "chrome_dir", "data_root", "scenarios_dir", "profiles_dir")},
                "checks": [{"what": w, "ok": ok, "detail": d} for w, ok, d in check(self.cfg)] + [{"what": "OpenRouter key", "ok": bool(secrets(self.cfg).get("OPENROUTER_API_KEY")), "detail": f"OPENROUTER_API_KEY in {self.cfg.secrets_file.name}"}],
                "endpoint": self.cfg.endpoint, "chrome": {"up": cdp.up(self.cfg.chrome_port), "port": self.cfg.chrome_port, "profiles": chrome.list_profiles(self.cfg.chrome_dir)},
                "accounts": [{"email": a, "uses": len(led.get(a, [])), "last": (led.get(a) or [{}])[-1].get("time", ""), "profile": bound.get(a, "")} for a in self.all_accounts()],
                "profiles": [{"id": p.id, "name": p.name, "accounts": p.accounts, "email": p.email, "chrome_profile": p.chrome_profile} for p in profiles.values()],
                "default_profile": self.cfg.profile, "columns": self.cfg.columns, "suggested_facts": list(profile_mod.SUGGESTED_FACTS), "profile_fields": list(profile_mod.PROFILE_FIELDS),
                "sets": scenario.sets(self.cfg.scenarios_dir), "issues": ISSUES, "default_column": config_mod.column(self.cfg)[0], "root": str(ROOT)}


def make_handler(state: State):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):   # quiet
            pass

        def _send(self, code: int, body, ctype="application/json"):
            data = body if isinstance(body, (bytes, bytearray)) else (json.dumps(body, ensure_ascii=False) if ctype == "application/json" else body).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype + ("; charset=utf-8" if ctype.startswith("text/") or ctype == "application/json" else ""))
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _q(self) -> dict:
            return {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}

        def _safe(self, s: str) -> str:
            if not s or not SAFE.match(s) or ".." in s:
                raise ValueError(f"bad name {s!r}")
            return s

        def do_GET(self):
            try:
                path = urlparse(self.path).path
                q = self._q()
                if path in ("/", "/index.html"):
                    return self._send(200, (HERE / "app.html").read_bytes(), "text/html")
                if path == "/api/state":
                    return self._send(200, state.state())
                if path == "/api/profiles":
                    return self._send(200, state.profile_rows())
                if path == "/api/scenarios":
                    return self._send(200, state.scenarios(q.get("profile", "")))
                if path == "/api/runs":
                    return self._send(200, state.runs())
                if path == "/api/run":
                    return self._send(200, state.run_detail(self._safe(q.get("name", ""))))
                if path == "/api/conversation":
                    return self._send(200, state.conversation(self._safe(q.get("run", "")), self._safe(q.get("file", ""))))
                if path == "/api/live":
                    return self._send(200, state.live_state())
                if path == "/api/sync":
                    return self._send(200, sync.status(fetch=q.get("fetch", "1") != "0"))
                if path == "/api/log":
                    log = state.logs_dir() / f"{self._safe(q.get('tag', ''))}.log"
                    text = log.read_text(encoding="utf-8", errors="replace")[-int(q.get("chars", 20000)):] if log.exists() else "(no log yet)"
                    return self._send(200, text, "text/plain")
                if path == "/api/export":
                    name = self._safe(q.get("name", ""))
                    data = state.export_csv(name).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/csv; charset=utf-8")
                    self.send_header("Content-Disposition", f'attachment; filename="{name}.csv"')
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                return self._send(404, {"error": "not found"})
            except (ValueError, FileNotFoundError, StopIteration) as exc:
                return self._send(400, {"error": str(exc) or type(exc).__name__})
            except Exception as exc:
                return self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

        def do_POST(self):
            try:
                path = urlparse(self.path).path
                n = int(self.headers.get("Content-Length") or 0)
                req = json.loads(self.rfile.read(n) or b"{}")
                if path == "/api/profiles":
                    return self._send(200, state.save_profile(req))
                if path == "/api/profiles/delete":
                    return self._send(200, {"deleted": profile_mod.delete(state.cfg.profiles_dir, str(req.get("id") or ""))})
                if path == "/api/scenarios":
                    return self._send(200, state.save_scenario(str(req.get("set") or ""), req.get("scenario") or {}, str(req.get("profile") or "")))
                if path == "/api/scenarios/delete":
                    return self._send(200, {"deleted": scenario.delete(state.cfg.scenarios_dir, str(req.get("set") or ""), str(req.get("id") or ""))})
                if path == "/api/preview":
                    return self._send(200, state.preview(req))
                if path == "/api/runs/start":
                    return self._send(200, state.start_run(req))
                if path == "/api/live/start":
                    return self._send(200, state.live_start(req))
                if path == "/api/live/act":
                    return self._send(200, state.live_act(req))
                if path == "/api/sync/pull":
                    return self._send(200, {"result": sync.pull()})
                if path == "/api/sync/push":
                    return self._send(200, {"result": sync.push(str(req.get("who") or ""), str(req.get("message") or ""))})
                if path == "/api/chrome/start":
                    return self._send(200, state.chrome_start(str(req.get("profile_directory") or "")))
                if path == "/api/runs/stop":
                    return self._send(200, {"stopped": state.stop_run(str(req.get("tag") or ""))})
                if path == "/api/annotate":
                    return self._send(200, state.annotate(self._safe(str(req.get("run") or "")), self._safe(str(req.get("file") or "")), req.get("annotation") or {}))
                return self._send(404, {"error": "not found"})
            except (ValueError, FileNotFoundError, StopIteration, RuntimeError) as exc:
                return self._send(400, {"error": str(exc) or type(exc).__name__})
            except Exception as exc:
                return self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

    return Handler


def make_server(cfg: Config, port: int = 0) -> ThreadingHTTPServer:
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(State(cfg)))
    srv.daemon_threads = True
    return srv


def serve(cfg: Config, port: int, open_browser: bool = True) -> int:
    srv = make_server(cfg, port)
    url = f"http://127.0.0.1:{srv.server_address[1]}/"
    print(f"Behavior Eval Tool at {url}  (Ctrl-C to stop)")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0
