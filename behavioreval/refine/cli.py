"""``python3 -m behavioreval refine <new|check|run|resume|report>``.

    refine new <skill>                     copy skills/_template to skills/<skill> (every file a placeholder)
    refine check <skill>                   list what is missing, and how many conversations a night would take
    refine run <skill> [--rounds N] [--max-cost X] [--max-conversations N] [--endpoint chrome|fake] [--yes]
    refine resume <run dir> [--yes]        continue a stopped run where it left off
    refine report <run dir>                rewrite changelog.md and report.md from state.json"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from behavioreval.core import config, openrouter, profile as profile_mod
from behavioreval.core.endpoint import make as make_endpoint
from behavioreval.core.config import ROOT
from behavioreval.refine import delivery as delivery_mod, inputs as inputs_mod, judges, report, rollouts
from behavioreval.refine.loop import Refiner
from behavioreval.refine.meter import Meter


def add_parser(sub) -> None:
    p = sub.add_parser("refine", help="improve one skill automatically: rollouts, judge, one small edit per round, held-out check")
    rs = p.add_subparsers(dest="refine_cmd", required=True)
    rs.add_parser("new", help="start a skill folder from the template").add_argument("skill")
    ck = rs.add_parser("check", help="what is missing before a run")
    ck.add_argument("skill")
    r = rs.add_parser("run", help="run the refine loop on one skill")
    r.add_argument("skill")
    re_ = rs.add_parser("resume", help="continue a stopped run")
    re_.add_argument("run_dir")
    for q in (ck, r, re_):
        q.add_argument("--skills-dir", dest="skills_dir", help="where skill folders live (default: skills/ in the repo)")
        q.add_argument("--rubric-dir", dest="rubric_dir", help="where general.md and skill_best_practices.md live (default: rubric/)")
    for q in (r, re_):
        q.add_argument("--rounds", type=int)
        q.add_argument("--max-cost", dest="max_cost", type=float, help="US dollars for our own model calls")
        q.add_argument("--max-conversations", dest="max_conversations", type=int)
        q.add_argument("--endpoint", choices=("chrome", "fake"), help="fake = a scripted assistant, for a dry run of the plumbing")
        q.add_argument("--profile", help="the profile whose dev account and Chrome run the conversations")
        q.add_argument("--yes", action="store_true")
    rs.add_parser("report", help="rewrite the report of a run").add_argument("run_dir")


def estimate(i: inputs_mod.SkillInputs) -> tuple[int, int]:
    """(conversations for v0, conversations per round)."""
    s, tr, ho = i.settings, len(i.split("train")), len(i.split("holdout"))
    return tr * s.rollouts_train + ho * s.rollouts_holdout, s.candidates_per_round * tr * s.screen_rollouts + ho * s.rollouts_holdout + tr * max(0, s.rollouts_train - s.screen_rollouts)


def cmd_new(skill: str) -> int:
    dst = ROOT / "skills" / skill
    if dst.exists():
        print(f"{dst} already exists")
        return 2
    shutil.copytree(ROOT / "skills" / "_template", dst)
    print(f"created {dst}; fill in every file that says PLACEHOLDER, then: python3 -m behavioreval refine check {skill}")
    return 0


def _dirs(args) -> tuple[Path | None, Path | None]:
    return tuple(Path(x).expanduser().resolve() if x else None for x in (getattr(args, "skills_dir", None), getattr(args, "rubric_dir", None)))


def cmd_check(skill: str, args) -> int:
    i = inputs_mod.load(skill, *_dirs(args))
    for w in i.warnings:
        print(f"warning: {w}")
    for p in i.problems:
        print(f"MISSING: {p}")
    if i.problems:
        print(f"\n{len(i.problems)} thing(s) to fill in before `refine run {skill}`.")
        return 2
    first, per_round = estimate(i)
    print(f"ready: {len(i.split('train'))} train, {len(i.split('holdout'))} held-out, {len(i.split('final'))} final card(s), {len(i.personas)} persona(s); "
          f"delivery {i.settings.delivery} as `{i.personal_name}`.\nConversations: {first} for v0, about {per_round} per round "
          f"(cap {i.settings.max_conversations}); at 2-10 minutes each and {i.settings.parallel} at a time.")
    return 0


def _build(skill: str, args, run_dir: Path, out=print):
    cfg = config.load(overrides={"profile": args.profile})
    i = inputs_mod.load(skill, *_dirs(args))
    if i.problems:
        raise SystemExit("refine check fails:\n  " + "\n  ".join(i.problems))
    s = i.settings
    for k in ("rounds", "max_cost", "max_conversations"):
        if getattr(args, k, None) is not None:
            setattr(s, "max_cost_usd" if k == "max_cost" else k, getattr(args, k))
    endpoint = args.endpoint or "chrome"
    if endpoint == "chrome" and cfg.env != "dev":
        raise SystemExit(f"refine runs on dev only (config env is {cfg.env!r})")
    cfg.judge_model = s.judge_model or cfg.judge_model
    if cfg.judge_model == s.editor_model:
        raise SystemExit("the judge and the editor must be different models (refine.json judge_model / editor_model)")
    key = openrouter.key()
    if not key:
        raise SystemExit("refine needs OPENROUTER_API_KEY (environment or .env)")
    person = profile_mod.load(cfg.profiles_dir, cfg.profile)
    account = person.email if endpoint == "chrome" else "dry-run@example.invalid"
    if endpoint == "chrome":
        from behavioreval.browser import cdp
        if not person.email:
            raise SystemExit(f"profile {person.id} has no dev account email; set it in the UI's Set up page")
        if not cdp.up(cfg.chrome_port):
            raise SystemExit(f"no Chrome on 127.0.0.1:{cfg.chrome_port}: start it with `python3 -m behavioreval chrome` (or Home → Start my Chrome)")
    kind = s.delivery if endpoint == "chrome" else "fake"
    meter = Meter(s.max_cost_usd, s.max_conversations)
    make = lambda: make_endpoint(endpoint, cfg, lambda m: None)   # noqa: E731
    dlv = delivery_mod.make(kind, make, account, i.personal_name, s.skill_description, cfg.turn_timeout_s, out)
    graders = judges.Graders(key, cfg.judge_model, s.diagnose_model, meter, cfg.judge_reasoning_tokens, cfg.judge_provider)
    from behavioreval.endpoints.chrome import web_url
    ctx = rollouts.Context(cfg=cfg, key=key, inputs=i, delivery=dlv, meter=meter, graders=graders, make_endpoint=make, account=account,
                           rubric_lines=judges.rubric_lines(i.general_rubric, i.criteria, i.personality), web_url=web_url(cfg) if endpoint == "chrome" else "", out=out)
    return ctx, i


def _go(skill: str, args, run_dir: Path) -> int:
    ctx, i = _build(skill, args, run_dir, out=lambda m: print(m, flush=True))
    first, per_round = estimate(i)
    s = i.settings
    print(f"refine {skill} on {ctx.account} ({ctx.delivery.label}); {s.rounds} round(s) max; v0 {first} conversations, ~{per_round} per round; "
          f"caps ${s.max_cost_usd:.2f} of our model calls, {s.max_conversations} conversations.\nrun dir: {run_dir}")
    if not args.yes and input("Start? [y/N] ").strip().lower() not in ("y", "yes"):
        print("aborted; nothing was sent")
        return 2
    refiner = Refiner(ctx, run_dir)
    refiner.state["skills_dir"], refiner.state["rubric_dir"] = str(i.folder.parent), str(_dirs(args)[1] or ROOT / "rubric")
    status = refiner.run()
    print(f"\n{status}\nreport: {run_dir / 'report.md'}\nchangelog: {run_dir / 'changelog.md'}")
    return 0 if status == "done" else 1


def main(args) -> int:
    c = args.refine_cmd
    if c == "new":
        return cmd_new(args.skill)
    if c == "check":
        return cmd_check(args.skill, args)
    if c == "run":
        cfg = config.load(overrides={"profile": args.profile})
        return _go(args.skill, args, cfg.data_root / "runs" / "refine" / f"{time.strftime('%Y%m%d-%H%M')}-{args.skill}")
    run_dir = Path(args.run_dir).expanduser()
    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    if c == "resume":
        args.skills_dir = args.skills_dir or state.get("skills_dir")
        args.rubric_dir = args.rubric_dir or state.get("rubric_dir")
        return _go(state["skill"], args, run_dir)
    print(report.write(run_dir, state, None))
    return 0
