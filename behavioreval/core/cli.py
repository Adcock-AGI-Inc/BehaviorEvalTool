"""``python3 -m behavioreval {ui, run, lint, scenarios, profiles, simulate, accounts, check, config, rescore, audit}``."""
from __future__ import annotations

import argparse
import json
import sys
import time

from behavioreval.core import config, ledger, openrouter, profile as profile_mod, records, scenario, scoring
from behavioreval.core.endpoint import NAMES as ENDPOINTS, make as make_endpoint
from behavioreval.core.harness import Conversation, run as run_conversation
from behavioreval.core.simulator import Exchange, UserSimulator


def _overrides(args) -> dict:
    return {k: getattr(args, k, None) for k in ("user_model", "temperature", "max_tokens", "max_turns", "turn_timeout_s", "lane", "model", "endpoint",
                                                "endpoint_url", "env", "judge_model", "judge_samples", "data_root", "mvp_dir", "profile")}


def _add_config_flags(p) -> None:
    p.add_argument("--profile", help="which simulated person runs (profiles/<id>.json; config: default)")
    p.add_argument("--user-model", dest="user_model", help="OpenRouter model that plays the user")
    p.add_argument("--temperature", type=float)
    p.add_argument("--max-tokens", dest="max_tokens", type=int)
    p.add_argument("--max-turns", dest="max_turns", type=int, help="user messages per conversation, opening included (config: 5)")
    p.add_argument("--turn-timeout", dest="turn_timeout_s", type=float)
    p.add_argument("--lane", help="hark-slotN / slotN for a self-hosted lane; default = the product's default model")
    p.add_argument("--model", help="explicit per-message model override, e.g. claude-opus-5-5 (overrides --lane)")
    p.add_argument("--endpoint", choices=tuple(ENDPOINTS))
    p.add_argument("--endpoint-url", dest="endpoint_url", help="openai_chat: the /v1 base")
    p.add_argument("--env")
    p.add_argument("--judge-model", dest="judge_model")
    p.add_argument("--judge-samples", dest="judge_samples", type=int)
    p.add_argument("--data-root", dest="data_root")
    p.add_argument("--mvp-dir", dest="mvp_dir")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="behavioreval", description="Behavior Eval Tool: a simulated user (an LLM playing a person from a scenario) talks to the Hark assistant for a few turns.")
    sub = p.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("ui", help="the web UI: profiles, scenarios, runs, results, annotation (http://127.0.0.1:<ui_port>)")
    u.add_argument("--port", type=int)
    u.add_argument("--no-browser", dest="no_browser", action="store_true")
    r = sub.add_parser("run", help="run scenarios as one profile on its accounts and judge the conversations")
    r.add_argument("--scenarios", default="all", help="a set name under scenarios/ (daily, complex, ...), a jsonl path, or all")
    r.add_argument("--only", action="append", help="scenario id; repeatable")
    r.add_argument("--account", action="append", help="use only these accounts (repeatable); default: the profile's accounts, else every unbound account")
    r.add_argument("--runs", type=int, default=1, help="conversations per scenario per column")
    r.add_argument("--score", choices=tuple(scoring.NAMES), default="judge")
    r.add_argument("--columns", help="assistant columns, comma-separated: default, slot20, claude-opus-5-5 (default: the config's lane/model)")
    r.add_argument("--parallel", type=int, default=4, help="conversations side by side (never two on one account)")
    r.add_argument("--tag", help="run directory name suffix")
    r.add_argument("--resume", help="an existing run directory: skip what it already holds, add the rest")
    r.add_argument("--calibrate-first", dest="calibrate_first", action="store_true", help="before an account's first scenario, audit and correct the assistant's memory (maint/01; recorded, not scored)")
    r.add_argument("--cleanup", action="store_true", help="after a scenario that leaves things on the account, ask the assistant to undo them (maint/02; recorded, not scored)")
    r.add_argument("--yes", action="store_true", help="skip the confirmation")
    _add_config_flags(r)
    l = sub.add_parser("lint", help="every scenario against a profile: unresolved placeholders, missing done_when; and every profile's accounts")
    l.add_argument("--scenarios", action="append", help="set name or path; default every set under scenarios/")
    _add_config_flags(l)
    sc = sub.add_parser("scenarios", help="list the scenarios")
    sc.add_argument("--scenarios", default="all")
    sub.add_parser("profiles", help="the simulated people and the accounts bound to them")
    s = sub.add_parser("simulate", help="only the user model: given a scenario and assistant lines, print the next user message")
    s.add_argument("--scenarios", default="all")
    s.add_argument("--scenario", required=True, help="scenario id")
    s.add_argument("--assistant", action="append", default=[], help="an assistant reply; repeatable to build a transcript")
    s.add_argument("--show-prompt", action="store_true")
    _add_config_flags(s)
    sub.add_parser("accounts", help="the test accounts, who they belong to and how often each was used")
    sub.add_parser("check", help="is everything in place to run on dev? (uv, mvp checkout, secrets, accounts)")
    sub.add_parser("config", help="the effective settings and where each came from")
    rs = sub.add_parser("rescore", help="re-judge the saved conversations of a run directory")
    rs.add_argument("run_dir")
    rs.add_argument("--score", choices=tuple(scoring.NAMES), default="judge")
    rs.add_argument("--only-unreadable", dest="only_unreadable", action="store_true")
    rs.add_argument("--parallel", type=int, default=6)
    _add_config_flags(rs)
    au = sub.add_parser("audit", help="judge reliability: re-judge a sample of a run's conversations and report agreement")
    au.add_argument("run_dir")
    au.add_argument("--sample", type=int, default=10)
    au.add_argument("--repeats", type=int, default=2)
    au.add_argument("--seed", type=int, default=1)
    _add_config_flags(au)
    return p


def _profile(cfg, pid: str | None = None) -> profile_mod.Profile:
    return profile_mod.load(cfg.profiles_dir, pid or cfg.profile)


# ── run ──────────────────────────────────────────────────────────────────────

def plan_jobs(cfg, scenarios, columns, runs, accounts, led, person):
    """[(rendered scenario, person as the scenario sees them, account, column)]: scenarios × columns × runs,
    spread over the accounts (least recently used first, round robin)."""
    problems = scenario.lint_all(scenarios, person)
    if problems:
        raise ValueError("lint first: " + "; ".join(f"{i}: {p}" for i, p in problems[:5]))
    order = ledger.least_used_first(led, accounts) if accounts else [""]
    jobs, k = [], 0
    for col in columns:
        for sc in scenarios:
            for _ in range(runs):
                jobs.append((scenario.render(sc, person), scenario.profile_for(sc, person), order[k % len(order)], col))
                k += 1
    return jobs


def run_jobs(cfg, jobs, args, key, out_dir, person, say=print) -> list:
    """Run the planned conversations (parallel over accounts), scoring and recording each; returns [(conv, score)]."""
    import threading
    from collections import defaultdict, deque
    from concurrent.futures import ThreadPoolExecutor
    led_path = ledger.path_for(cfg.data_root)
    led = ledger.load(led_path)
    rows: list = []
    sim_cache: dict = {}
    scorers: dict = {}
    locks: dict = {}
    io_lock = threading.Lock()
    counter = {"done": 0}
    maint = {m.id: m for m in scenario.resolve("maint", directory=cfg.scenarios_dir)} if (args.calibrate_first or args.cleanup) and (cfg.scenarios_dir / "maint.jsonl").exists() else {}
    calibrated: set = set()
    settings = {"profile": person.id, "columns": ",".join(sorted({j[3] for j in jobs})), "user_model": cfg.user_model, "max_turns": cfg.max_turns,
                "score": args.score, "env": cfg.env, "judge": cfg.judge_model}

    def log(prefix, msg):
        with io_lock:
            say(f"[{prefix}] {msg}")

    def converse(sc, u, account, jcfg, label, prefix, score_it):
        history = ledger.history(led, account) if jcfg.history_in_briefing and account else []
        endpoint = make_endpoint(args.endpoint or jcfg.endpoint, jcfg, lambda m, _p=prefix: log(_p, m.strip()))
        try:
            try:
                conv = run_conversation(sc, u, account, sim_cache.setdefault(jcfg.user_model, UserSimulator(jcfg, key)), endpoint, jcfg, history,
                                        lambda m, _p=prefix: log(_p, m.strip()))
            finally:
                endpoint.close()
        except (RuntimeError, ValueError, TimeoutError, OSError) as exc:
            log(prefix, f"error: {exc}")
            conv = Conversation(scenario=sc.id, user=u.name, account=account, profile=u.id, lane=jcfg.lane, column=label, tags=list(sc.tags), stop=f"error: {str(exc)[:120]}")
            score = {"verdict": "-", "lines": [f"error: {exc}"]}
        else:
            if score_it:
                try:
                    scorer = scorers.setdefault(label, scoring.make(args.score, jcfg, key, lambda m, _p=prefix: log(_p, m.strip()), limits=u.limits))
                    score = scorer.score(conv, sc)
                except Exception as exc:
                    score = {"verdict": "UNREADABLE", "lines": [f"scorer failed: {type(exc).__name__}: {str(exc)[:200]}"]}
            else:
                score = {"verdict": "-", "lines": ["maintenance conversation, not scored"]}
            if account:
                with io_lock:
                    ledger.record(led_path, account, {"scenario": sc.id, "profile": u.id, "opening": sc.opening, "conversation": conv.conversation_id,
                                                      "column": label, "stop": conv.stop, "verdict": score.get("verdict", "-"), "tags": list(sc.tags)})
        path = records.save_conversation(out_dir, conv, score)
        with io_lock:
            rows.append((conv, score))
            counter["done"] += 1
            say(f"[{prefix}] {conv.stop:10s} score={score.get('verdict', '-')} turns={conv.user_turns}  ({counter['done']}/{len(jobs)})  {path.name}")
            records.write_report(out_dir, rows, settings)
        return conv, score

    def one(job):
        sc, u, account, col = job
        jcfg = config.with_column(cfg, col)
        label = config.column(jcfg)[0]
        prefix = f"{label}|{(account or 'noaccount').split('@')[0]}|{sc.id.split('/')[-1][:24]}"
        try:
            with locks.setdefault(account or "-", threading.Lock()):
                log(prefix, "start")
                if args.calibrate_first and account not in calibrated and "maint/01_calibrate_memory" in maint:
                    calibrated.add(account)
                    converse(scenario.render(maint["maint/01_calibrate_memory"], person), person, account, jcfg, label, prefix + "|calibrate", False)
                conv, _ = converse(sc, u, account, jcfg, label, prefix, True)
                if args.cleanup and sc.leaves and not sc.leaves.lower().startswith("nothing") and "maint/02_cleanup_last_task" in maint and not conv.stop.startswith("error"):
                    prev = person.with_overrides({"prev_opening": sc.opening.replace('"', "'")[:300], "prev_leaves": sc.leaves})
                    converse(scenario.render(maint["maint/02_cleanup_last_task"], prev), prev, account, jcfg, label, prefix + "|cleanup", False)
        except Exception as exc:
            log(prefix, f"UNHANDLED {type(exc).__name__}: {str(exc)[:200]}")

    by_acct: dict = defaultdict(deque)
    for job in jobs:
        by_acct[job[2]].append(job)
    ordered = []
    while any(by_acct.values()):
        for a in list(by_acct):
            if by_acct[a]:
                ordered.append(by_acct[a].popleft())
    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as ex:
        list(ex.map(one, ordered))
    return rows


def cmd_run(args) -> int:
    from pathlib import Path
    cfg = config.load(overrides=_overrides(args))
    key = openrouter.key()
    if not key:
        print(f"the user model needs OPENROUTER_API_KEY (environment or {cfg.secrets_file})")
        return 2
    profiles = profile_mod.load_all(cfg.profiles_dir)
    person = _profile(cfg)
    scs = scenario.resolve(args.scenarios, args.only, directory=cfg.scenarios_dir)
    scs = [s for s in scs if not s.id.startswith("maint/")] if args.scenarios == "all" else scs
    if not scs:
        print("no scenarios")
        return 2
    columns = [c.strip() for c in (args.columns or "").split(",") if c.strip()] or [config.column(cfg)[0]]
    endpoint_name = args.endpoint or cfg.endpoint
    all_accounts = list(config.load_accounts(cfg.accounts_file))
    accounts = args.account or profile_mod.accounts_for(person, all_accounts, profiles)
    bad = profile_mod.lint(person, all_accounts, profiles)
    if bad:
        print(f"profile {person.id}: " + "; ".join(bad))
        return 2
    if endpoint_name == "evalsystem" and not accounts:
        print(f"no accounts for profile {person.id}: bind some in profiles/{person.id}.json, or add unbound ones to {cfg.accounts_file.name}")
        return 2
    led = ledger.load(ledger.path_for(cfg.data_root))
    jobs = plan_jobs(cfg, scs, columns, args.runs, accounts if endpoint_name == "evalsystem" else [], led, person)
    print(f"behavioreval run as {person.name} ({person.id}): {len(scs)} scenario(s) × {len(columns)} column(s) × {args.runs} = {len(jobs)} conversation(s) on "
          f"{len(set(j[2] for j in jobs))} account(s); columns {', '.join(columns)}; user model {cfg.user_model}; max_turns {cfg.max_turns}; parallel {args.parallel}; "
          f"score {args.score}; endpoint {endpoint_name}")
    for sc, u, account, col in jobs[:40]:
        print(f"  - {sc.id} [{col}] on {account or 'no account'}; opening {sc.opening[:60]!r}")
    if len(jobs) > 40:
        print(f"  … {len(jobs) - 40} more")
    if not args.yes and input("Continue? [y/N] ").strip().lower() not in ("y", "yes"):
        print("aborted; nothing was sent")
        return 2
    out_dir = Path(args.resume).expanduser() if args.resume else records.run_dir(cfg.data_root, args.tag or f"{person.id}-" + "-".join(columns))
    rows: list = []
    if args.resume:
        done = {p.name for p in out_dir.glob("*.conversation.json")}
        before = len(jobs)
        jobs = [j for j in jobs if records.file_name(config.column(config.with_column(cfg, j[3]))[0], j[2], j[0].id) not in done]
        print(f"resume {out_dir.name}: {before - len(jobs)} already done, {len(jobs)} to run")
        for f in sorted(out_dir.glob("*.conversation.json")):
            conv, d = records.load_conversation(f)
            rows.append((conv, d.get("score") or {}))
    print(f"run dir: {out_dir}", flush=True)
    rows += run_jobs(cfg, jobs, args, key, out_dir, person, say=lambda m: print(m, flush=True))
    report = records.write_report(out_dir, rows, {"profile": person.id, "columns": ",".join(columns), "user_model": cfg.user_model, "max_turns": cfg.max_turns,
                                                   "score": args.score, "env": cfg.env, "judge": cfg.judge_model})
    print(f"\nreport: {report}")
    return 0


# ── the rest ─────────────────────────────────────────────────────────────────

def cmd_lint(args) -> int:
    cfg = config.load(overrides=_overrides(args))
    profiles = profile_mod.load_all(cfg.profiles_dir)
    person = _profile(cfg)
    all_accounts = list(config.load_accounts(cfg.accounts_file))
    bad = 0
    for pid, p in profiles.items():
        problems = profile_mod.lint(p, all_accounts, profiles)
        print(f"profile {pid}: {len(p.accounts)} account(s), {len(problems)} problem(s)")
        for why in problems:
            print(f"  {why}")
        bad += len(problems)
    files = args.scenarios or scenario.sets(cfg.scenarios_dir)
    for f in files:
        try:
            scs = scenario.resolve(f, directory=cfg.scenarios_dir)
        except ValueError as exc:
            print(f"{f}: {exc}")
            bad += 1
            continue
        problems = scenario.lint_all(scs, person)
        print(f"{f}: {len(scs)} scenario(s), {len(problems)} problem(s) as {person.id}")
        for sid, why in problems:
            print(f"  {sid}: {why}")
        bad += len(problems)
    return 1 if bad else 0


def cmd_scenarios(args) -> int:
    cfg = config.load()
    for sc in scenario.resolve(args.scenarios, directory=cfg.scenarios_dir):
        print(f"{sc.id:42s} turns={sc.max_turns or cfg.max_turns:2d} beats={len(sc.beats)} tags={','.join(sc.tags):24s} {sc.opening[:70]!r}")
    return 0


def cmd_profiles(args) -> int:
    cfg = config.load()
    profiles = profile_mod.load_all(cfg.profiles_dir)
    all_accounts = list(config.load_accounts(cfg.accounts_file))
    for pid, p in profiles.items():
        accts = profile_mod.accounts_for(p, all_accounts, profiles)
        print(f"{pid:12s} {p.name[:24]:24s} home={str(p.facts.get('home', ''))[:24]:24s} accounts={len(p.accounts) or f'none bound ({len(accts)} unbound available)'}"
              + (f": {', '.join(p.accounts)}" if p.accounts else ""))
    return 0


def cmd_simulate(args) -> int:
    cfg = config.load(overrides=_overrides(args))
    person = _profile(cfg)
    sc = scenario.resolve(args.scenarios, [args.scenario], directory=cfg.scenarios_dir)[0]
    problems = scenario.lint(sc, person)
    if problems:
        print(f"{sc.id}: " + "; ".join(problems))
        return 2
    u = scenario.profile_for(sc, person)
    sc = scenario.render(sc, person)
    today = time.strftime("%Y-%m-%d")
    if args.show_prompt:
        from behavioreval.core.simulator import system_prompt
        print(system_prompt(u, sc, [], today))
        print("-" * 60)
    transcript, user_line = [], sc.opening
    for a in args.assistant:
        transcript.append(Exchange(user=user_line, assistant=a, ended="end_turn"))
        user_line = "(earlier line)"
    print(f"user {u.name} ({u.id}); opening: {sc.opening!r}")
    if not args.assistant:
        print("no --assistant given: the first message is the opening, sent verbatim")
        return 0
    key = openrouter.key()
    if not key:
        print("needs OPENROUTER_API_KEY")
        return 2
    r = UserSimulator(cfg, key).next_message(u, sc, transcript, [], today)
    print(json.dumps({"say": r.say, "done": r.done, "why": r.why, "unreadable": r.unreadable, "usage": r.usage}, ensure_ascii=False, indent=1))
    return 0


def cmd_accounts(args) -> int:
    cfg = config.load()
    led = ledger.load(ledger.path_for(cfg.data_root))
    accounts = config.load_accounts(cfg.accounts_file)
    bound = profile_mod.bound_accounts(profile_mod.load_all(cfg.profiles_dir))
    if not accounts:
        print(f"no accounts in {cfg.accounts_file} (copy accounts.example.tsv)")
        return 2
    print(f"{'account':34s} {'profile':12s} uses  last used          last scenario")
    for a in accounts:
        uses = led.get(a, [])
        print(f"{a:34s} {bound.get(a, '-'):12s} {len(uses):4d}  {uses[-1]['time'][:16] if uses else '-':18s} {uses[-1].get('scenario', '') if uses else ''}")
    return 0


def cmd_check(args) -> int:
    from behavioreval.endpoints.evalsystem import check
    cfg = config.load()
    bad = 0
    for what, ok, detail in check(cfg):
        print(f"{'OK ' if ok else 'MISSING'}  {what:24s} {detail}")
        bad += not ok
    try:
        profiles = profile_mod.load_all(cfg.profiles_dir)
        person = _profile(cfg)
        n = sum(len(scenario.resolve(s, directory=cfg.scenarios_dir)) for s in scenario.sets(cfg.scenarios_dir))
        print(f"OK   profiles                 {len(profiles)} ({', '.join(profiles)}); default runs as {person.name}")
        print(f"OK   scenarios                {n} in {cfg.scenarios_dir}")
    except ValueError as exc:
        print(f"MISSING  profiles / scenarios  {exc}")
        bad += 1
    return 1 if bad else 0


def cmd_config(args) -> int:
    cfg = config.load()
    for k, src in sorted(cfg.source.items()):
        print(f"{k:22s} {str(getattr(cfg, k))[:70]:70s} ({src})")
    return 0


def _scenario_index(cfg) -> dict:
    out = {}
    for s in scenario.sets(cfg.scenarios_dir):
        for sc in scenario.resolve(s, directory=cfg.scenarios_dir):
            out[sc.id] = sc
    return out


def _person_of(conv, cfg, profiles):
    return profiles.get(conv.profile) or _profile(cfg)


def cmd_rescore(args) -> int:
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path
    cfg = config.load(overrides=_overrides(args))
    key = openrouter.key()
    run_dir = Path(args.run_dir).expanduser()
    files = sorted(run_dir.glob("*.conversation.json"))
    if not files:
        print(f"no conversations under {run_dir}")
        return 2
    index = _scenario_index(cfg)
    profiles = profile_mod.load_all(cfg.profiles_dir)
    scorers: dict = {}
    lock = threading.Lock()

    def one(f):
        try:
            conv, d = records.load_conversation(f)
            sc = index.get(conv.scenario)
            if conv.scenario.startswith("maint/") or sc is None or conv.stop.startswith("error") or \
               (args.only_unreadable and (d.get("score") or {}).get("verdict") not in ("UNREADABLE", "-", None)):
                return conv, d.get("score") or {"verdict": "-", "lines": []}
            person = _person_of(conv, cfg, profiles)
            u = scenario.profile_for(sc, person)
            with lock:
                scorer = scorers.setdefault(person.id, scoring.make(args.score, cfg, key, lambda m: None, limits=u.limits))
            score = scorer.score(conv, scenario.render(sc, person))
            if d.get("score"):
                d.setdefault("score_history", []).append(d["score"])
            d["score"] = score
            f.write_text(json.dumps(d, indent=1, ensure_ascii=False), encoding="utf-8")
            print(f"{score.get('verdict', '-'):10s} {f.name[:90]}", flush=True)
            return conv, score
        except Exception as exc:
            print(f"skip {f.name}: {type(exc).__name__}: {str(exc)[:160]}", flush=True)
            return None

    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as ex:
        rows = [r for r in ex.map(one, files) if r is not None]
    print(f"report: {records.write_report(run_dir, rows, {'rescored_with': args.score, 'judge': cfg.judge_model})}")
    return 0


def cmd_audit(args) -> int:
    import random
    from pathlib import Path
    from behavioreval.scorers import judge as J
    cfg = config.load(overrides=_overrides(args))
    key = openrouter.key()
    run_dir = Path(args.run_dir).expanduser()
    files = [f for f in sorted(run_dir.glob("*.conversation.json")) if (json.loads(f.read_text()).get("score") or {}).get("verdict") in ("PASS", "FAIL")]
    if not files:
        print("no judged conversations yet")
        return 2
    random.seed(args.seed)
    index = _scenario_index(cfg)
    profiles = profile_mod.load_all(cfg.profiles_dir)
    agree = disagree = 0
    rows = []
    for f in random.sample(files, min(args.sample, len(files))):
        conv, d = records.load_conversation(f)
        sc = index.get(conv.scenario)
        if sc is None:
            continue
        person = _person_of(conv, cfg, profiles)
        u = scenario.profile_for(sc, person)
        jcfg = config.with_column(cfg, conv.column or "default")
        jcfg.judge_samples = args.repeats
        res = J.UsersimJudge(jcfg, key, out=lambda m: None, limits=u.limits).score(conv, scenario.render(sc, person))
        votes = [s.get("success") for s in res.get("judge", [])]
        same = all(v == (d["score"]["verdict"] == "PASS") for v in votes if v is not None)
        agree += same; disagree += (not same)
        rows.append({"file": f.name, "stored": d["score"]["verdict"], "votes": votes, "scores": [s.get("score") for s in res.get("judge", [])]})
        print(f"{'agree' if same else 'DISAGREE':8s} {f.name[:70]:70s} stored={d['score']['verdict']} re-judged={votes}")
    print(f"\nagreement: {agree}/{agree + disagree} ({args.repeats} re-judgement(s) each, judge {cfg.judge_model})")
    (run_dir / "judge_audit.json").write_text(json.dumps({"judge": cfg.judge_model, "repeats": args.repeats, "agree": agree, "disagree": disagree, "rows": rows}, indent=1), encoding="utf-8")
    return 0


def cmd_ui(args) -> int:
    from behavioreval.ui.server import serve
    cfg = config.load()
    return serve(cfg, port=args.port or cfg.ui_port, open_browser=not args.no_browser)


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass
    args = build_parser().parse_args(argv)
    try:
        return {"ui": cmd_ui, "run": cmd_run, "lint": cmd_lint, "scenarios": cmd_scenarios, "profiles": cmd_profiles, "simulate": cmd_simulate,
                "accounts": cmd_accounts, "check": cmd_check, "config": cmd_config, "rescore": cmd_rescore, "audit": cmd_audit}[args.cmd](args)
    except ValueError as exc:
        print(f"error: {exc}")
        return 2
