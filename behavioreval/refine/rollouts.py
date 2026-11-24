"""Run one skill version on one split: every card × N rollouts through the existing harness, then the
existing judge, the rubric grader, the halt check and the speed numbers. One result file per rollout
(``<version>/<split>/<card>__r<i>.result.json``) next to the usual conversation record; a rollout whose
result file exists is not run again, which is what makes a run resumable."""
from __future__ import annotations

import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

from behavioreval.core import records, scenario
from behavioreval.core.harness import Conversation, run as run_conversation
from behavioreval.core.simulator import UserSimulator
from behavioreval.refine import decide, safety
from behavioreval.refine.meter import BudgetExceeded
from behavioreval.scorers import judge as base


@dataclass
class Context:
    cfg: object               # core Config, judge_model already set to the refine judge
    key: str
    inputs: object            # SkillInputs
    delivery: object
    meter: object
    graders: object
    make_endpoint: object     # () -> Endpoint
    account: str
    rubric_lines: list
    web_url: str = ""
    out: object = print
    ask: object = base._ask   # the JSON-answering model call; tests pass a fake


def result_path(run_dir: Path, version: str, split: str, card_id: str, i: int) -> Path:
    return run_dir / version / split / (re.sub(r"[^A-Za-z0-9_.-]+", "_", card_id) + f"__r{i}.result.json")


def load_results(run_dir: Path, version: str, split: str) -> list[dict]:
    d = run_dir / version / split
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(d.glob("*.result.json"))] if d.exists() else []


def _judge(ctx: Context, conv, sc, limits) -> dict:
    ask = lambda p, m, k, **kw: ctx.ask(p, m, k, post=ctx.meter.post_for("judge"), **kw)   # noqa: E731
    return base.UsersimJudge(ctx.cfg, ctx.key, lambda m: None, ask=ask, limits=limits).score(conv, sc)


def one(ctx: Context, run_dir: Path, version: str, split: str, card, i: int, sim: UserSimulator) -> dict:
    path = result_path(run_dir, version, split, card.scenario.id, i)
    persona = ctx.inputs.personas[card.persona]
    u = replace(scenario.profile_for(card.scenario, persona), email=ctx.account)
    sc = scenario.render(card.scenario, persona)
    ctx.meter.start_conversation()
    tag = f"{version}|{split}|{sc.id.split('/')[-1][:20]}|r{i}"
    endpoint = ctx.delivery.wrap(ctx.make_endpoint())
    try:
        try:
            conv = run_conversation(sc, u, ctx.account, sim, endpoint, ctx.cfg, [], lambda m: ctx.out(f"[{tag}] {m.strip()}"))
        finally:
            endpoint.close()
    except (RuntimeError, ValueError, TimeoutError, OSError) as exc:
        conv = Conversation(scenario=sc.id, user=u.name, account=ctx.account, profile=u.id, column=version, stop=f"error: {str(exc)[:160]}")
    conv.column = version
    halts, claims = safety.check(conv, ctx.inputs.settings.halt)
    if conv.stop.startswith("error"):
        score, rubric = {"verdict": "-", "lines": [conv.stop]}, {}
    else:
        score = _judge(ctx, conv, sc, u.limits)
        rubric = ctx.graders.rubric(conv, sc, persona, ctx.rubric_lines, ctx.inputs.goal)
    path.parent.mkdir(parents=True, exist_ok=True)
    conv_file = records.save_conversation(path.parent, conv, {**score, "rubric": rubric}, attempt=i + 1)
    loaded, loaded_detail = ctx.delivery.loaded(conv)
    r = {"version": version, "split": split, "card": sc.id, "persona": persona.id, "rollout": i, "file": str(conv_file),
         "conversation_id": conv.conversation_id, "link": f"{ctx.web_url}/project/{conv.conversation_id}" if ctx.web_url and conv.conversation_id else "",
         "stop": conv.stop, "verdict": score.get("verdict", "-"), "judge_reasoning": str(((score.get("judge") or [{}])[0]).get("reasoning", ""))[:900],
         "rubric": rubric, "loaded": loaded, "loaded_detail": loaded_detail, "speed": decide.speed(conv), "halts": halts, "claims": claims,
         "delivery": ctx.delivery.label, "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    path.write_text(json.dumps(r, indent=1, ensure_ascii=False), encoding="utf-8")
    ctx.out(f"[{tag}] {conv.stop} verdict={r['verdict']} quality={decide.quality(r):.2f} total={r['speed']['total_s']}s loaded={loaded}")
    if halts:
        raise safety.Halt(f"{sc.id} rollout {i}: " + "; ".join(halts))
    return r


def run_split(ctx: Context, run_dir: Path, version: str, split: str, cards: list, rollouts: int) -> list[dict]:
    """Every card × rollouts for this version; returns all results for the split, old and new. Raises Halt or BudgetExceeded."""
    todo = [(c, i) for c in cards for i in range(rollouts) if not result_path(run_dir, version, split, c.scenario.id, i).exists()]
    sim = UserSimulator(ctx.cfg, ctx.key, post=ctx.meter.post_for("user"))
    stop = threading.Event()
    errors: list[BaseException] = []

    def job(ci):
        if stop.is_set():
            return
        try:
            one(ctx, run_dir, version, split, ci[0], ci[1], sim)
        except (safety.Halt, BudgetExceeded) as exc:
            stop.set()
            errors.append(exc)

    with ThreadPoolExecutor(max_workers=max(1, ctx.inputs.settings.parallel)) as ex:
        list(ex.map(job, todo))
    halts = [e for e in errors if isinstance(e, safety.Halt)]
    if halts or errors:
        raise (halts or errors)[0]
    wanted = {(c.scenario.id, i) for c in cards for i in range(rollouts)}
    return [r for r in load_results(run_dir, version, split) if (r["card"], r["rollout"]) in wanted]
