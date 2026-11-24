"""The refine loop for one skill, resumable at any point.

v0 (start.md, or drafted from the goal) is installed and run on train and held-out, then each round:
  1. diagnose the best version's failed or slow train conversations; pick the heaviest skill-fixable group
  2. the editor proposes candidate edits for that group; the guards drop leaky, oversized or broken ones
  3. each surviving candidate is installed and screened on train (screen_rollouts per card)
  4. the best screened candidate runs on held-out and is compared pairwise with the best version so far
  5. decide.decide keeps it (it becomes the best; its train runs are topped up) or rejects it
Stops on max rounds, `plateau` rejected rounds in a row, nothing left to fix, a cap, or a halt.
At the end the final split runs once on v0 and on the best version, and the report is written.
State lives in <run dir>/state.json and is saved after every step; `refine resume` picks it up."""
from __future__ import annotations

import json
import time
from pathlib import Path

from behavioreval.core import records, scenario
from behavioreval.refine import decide, delivery as delivery_mod, editor, judges, report, rollouts, safety, targets
from behavioreval.refine.meter import BudgetExceeded


class Refiner:
    def __init__(self, ctx: rollouts.Context, run_dir: Path, ask=None):
        self.ctx, self.run_dir = ctx, run_dir
        self.s = ctx.inputs.settings
        self.ask = ask or ctx.ask
        self.state_path = run_dir / "state.json"
        self.state = json.loads(self.state_path.read_text(encoding="utf-8")) if self.state_path.exists() else self._new_state()
        ctx.meter.restore(self.state.get("meter") or {})
        self.installed = ""

    def _new_state(self) -> dict:
        i = self.ctx.inputs
        return {"skill": i.name, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "status": "running", "delivery": self.ctx.delivery.label,
                "account": self.ctx.account, "judge_model": self.ctx.cfg.judge_model, "editor_model": self.s.editor_model,
                "settings": {k: v for k, v in vars(self.s).items() if k != "halt"}, "cards": {c.scenario.id: c.split for c in i.cards},
                "best": "", "versions": {}, "rounds": [], "tried": [], "notes": [], "meter": {}, "final": {}}

    # ── bookkeeping ──────────────────────────────────────────────────────────
    def save(self) -> None:
        self.state["meter"] = self.ctx.meter.to_dict()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self.state, indent=1, ensure_ascii=False), encoding="utf-8")

    def text(self, version: str) -> str:
        return (self.run_dir / "versions" / f"{version}.md").read_text(encoding="utf-8")

    def add_version(self, text: str, parent: str, edit: dict | None) -> str:
        v = f"v{len(self.state['versions'])}"
        d = self.run_dir / "versions"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{v}.md").write_text(text, encoding="utf-8")
        if parent:
            (d / f"{v}.diff").write_text(editor.diff(self.text(parent), text, f"{parent}.md", f"{v}.md"), encoding="utf-8")
        self.state["versions"][v] = {"parent": parent, "edit": edit, "chars": len(text), "created": time.strftime("%H:%M:%S")}
        self.save()
        return v

    def install(self, version: str) -> None:
        if self.installed == version:
            return
        rec = self.ctx.delivery.install(self.text(version), version)
        self.state["versions"][version].setdefault("installs", []).append({"ok": rec.ok, "detail": rec.detail, "conversation": rec.conversation_id})
        self.save()
        if not rec.ok:
            raise delivery_mod.DeliveryFailed(f"{version}: {rec.detail}")
        self.installed = version

    def evaluate(self, version: str, split: str, rollouts_n: int) -> list[dict]:
        cards = self.ctx.inputs.split(split)
        if any(not rollouts.result_path(self.run_dir, version, split, c.scenario.id, i).exists() for c in cards for i in range(rollouts_n)):
            self.install(version)
        res = rollouts.run_split(self.ctx, self.run_dir, version, split, cards, rollouts_n)
        self.state["versions"][version].setdefault("summary", {})[split] = decide.summary(res)
        self.save()
        return res

    def paths(self, version: str, split: str) -> dict:
        return {(r["card"], r["rollout"]): rollouts.result_path(self.run_dir, version, split, r["card"], r["rollout"])
                for r in rollouts.load_results(self.run_dir, version, split)}

    # ── steps ────────────────────────────────────────────────────────────────
    def start(self) -> str:
        if self.state["best"]:
            return self.state["best"]
        i = self.ctx.inputs
        if "v0" not in self.state["versions"]:
            text = i.start if i.start is not None else editor.draft(
                self.ask, self.ctx.key, self.s.editor_model, self.ctx.meter.post_for("editor"), goal=i.goal, best_practices=i.best_practices,
                rubric=judges.rubric_text(self.ctx.rubric_lines), personality=i.personality, max_chars=self.s.max_skill_chars)
            self.add_version(text, "", None)
            self.state["versions"]["v0"]["outcome"] = "start" if i.start is not None else "drafted from the goal"
        v0 = "v0"
        self.evaluate(v0, "train", self.s.rollouts_train)
        self.evaluate(v0, "holdout", self.s.rollouts_holdout)
        self.state["best"] = v0
        self.save()
        return v0

    def diagnose(self, version: str) -> list[dict]:
        res = rollouts.load_results(self.run_dir, version, "train")
        counts = targets.diagnose_all(res, self.paths(version, "train"), self.ctx.inputs, self.ctx.graders, self.text(version), self.s.delivery)
        self.state["versions"][version]["causes"] = counts
        self.save()
        if counts.get("not_loaded", 0) * 2 > len(res):
            raise delivery_mod.DeliveryFailed(f"{version}: Hark did not load the candidate in {counts['not_loaded']} of {len(res)} train conversations; "
                                              "fix the skill description or the delivery before refining the text")
        return res

    def round(self) -> bool:
        """One round. False when there is nothing left to try."""
        best = self.state["best"]
        rnd = self.state["rounds"][-1] if self.state["rounds"] and "decision" not in self.state["rounds"][-1] else None
        if rnd is None:
            train = self.diagnose(best)
            picked = targets.pick(train, self.state["tried"], self.s.tries_per_failure)
            if picked is None:
                self.state["notes"].append(f"round {len(self.state['rounds']) + 1}: no skill-fixable failure left to try")
                return False
            key, group = picked
            rnd = {"n": len(self.state["rounds"]) + 1, "best_before": best, "target": key, "group_size": len(group), "candidates": []}
            self.state["rounds"].append(rnd)
            self._propose(rnd, best, group)
        self._screen(rnd)
        self._holdout(rnd)
        return True

    def _propose(self, rnd: dict, best: str, group: list[dict]) -> None:
        i = self.ctx.inputs
        skill = self.text(best)
        recs = {r["file"]: json.loads(Path(r["file"]).read_text(encoding="utf-8")) for r in group[:3]}
        edits, raw = editor.propose(self.ask, self.ctx.key, self.s.editor_model, self.ctx.meter.post_for("editor"), goal=i.goal,
                                    best_practices=i.best_practices, rubric=judges.rubric_text(self.ctx.rubric_lines), personality=i.personality,
                                    skill=skill, failure=editor.failure_block(rnd["target"], group, recs), tried=self.state["tried"],
                                    k=self.s.candidates_per_round, max_lines=self.s.max_diff_lines)
        if not edits:
            rnd["editor_error"] = json.dumps(raw)[:400]
        train_texts = [t for c in i.split("train") for t in c.scenario.texts()] + [str(v) for c in i.split("train") for v in i.personas[c.persona].facts.values()]
        allowed = [skill, i.goal, "\n".join(i.criteria), i.best_practices, i.general_rubric, i.personality]
        for e in edits[: self.s.candidates_per_round]:
            new, why = editor.guard(skill, e, self.s.max_diff_lines, self.s.max_skill_chars, train_texts, allowed)
            cand = {"edit": e.to_dict(), "guard": why or "ok"}
            if new is not None:
                cand["version"] = self.add_version(new, best, e.to_dict())
            else:
                self._tried(rnd, e.reason, f"guard: {why}")
            rnd["candidates"].append(cand)
        self.save()

    def _screen(self, rnd: dict) -> None:
        for cand in rnd["candidates"]:
            if cand.get("version") and "train" not in cand:
                cand["train"] = decide.summary(self.evaluate(cand["version"], "train", self.s.screen_rollouts))
                self.save()

    def _holdout(self, rnd: dict) -> None:
        best = rnd["best_before"]
        screened = [c for c in rnd["candidates"] if c.get("train")]
        best_train = decide.summary([r for r in rollouts.load_results(self.run_dir, best, "train") if r["rollout"] < self.s.screen_rollouts])
        if not screened:
            return self._decide(rnd, None, False, "no candidate passed the guards")
        chosen = max(screened, key=lambda c: c["train"].get("quality") or 0)
        for c in screened:
            if c is not chosen:
                self._tried(rnd, c["edit"]["reason"], "lost screening to another candidate")
        if (chosen["train"].get("quality") or 0) < (best_train.get("quality") or 0) - self.s.noise_margin:
            return self._decide(rnd, chosen, False, f"screening: train quality {chosen['train'].get('quality')} vs {best_train.get('quality')} for {best}")
        v = chosen["version"]
        cand_res = self.evaluate(v, "holdout", self.s.rollouts_holdout)
        rnd["chosen"], rnd["holdout"] = v, decide.summary(cand_res)
        rnd["pairwise"] = self._pairwise(v, best, cand_res)
        keep, why = decide.decide(self.state["versions"][best]["summary"]["holdout"], rnd["holdout"], [p["verdict"] for p in rnd["pairwise"]],
                                  len(self.text(best)), len(self.text(v)), self.s.noise_margin, self.s.speed_margin)
        return self._decide(rnd, chosen, keep, why)

    def _pairwise(self, v: str, best: str, cand_res: list[dict]) -> list[dict]:
        best_by = {(r["card"], r["rollout"]): r for r in rollouts.load_results(self.run_dir, best, "holdout")}
        cards = {c.scenario.id: c for c in self.ctx.inputs.cards}
        out = []
        for r in cand_res:
            b = best_by.get((r["card"], r["rollout"]))
            if not b or not decide.scorable(r) or not decide.scorable(b):
                continue
            card = cards[r["card"]]
            persona = self.ctx.inputs.personas[card.persona]
            sc = scenario.render(card.scenario, persona)
            pw = self.ctx.graders.pairwise(records.load_conversation(Path(r["file"]))[0], records.load_conversation(Path(b["file"]))[0],
                                           sc, persona, self.ctx.rubric_lines, self.ctx.inputs.goal)
            out.append({"card": r["card"], "rollout": r["rollout"], "verdict": pw["verdict"], "orders": pw["orders"]})
        return out

    def _decide(self, rnd: dict, chosen: dict | None, keep: bool, why: str) -> None:
        rnd["decision"] = {"keep": keep, "why": why, "at": time.strftime("%H:%M:%S")}
        if chosen:
            self.state["versions"][chosen["version"]]["outcome"] = "kept" if keep else "rejected"
            self._tried(rnd, chosen["edit"]["reason"], "kept" if keep else f"rejected: {why}")
        if keep:
            self.state["best"] = chosen["version"]
            self.save()
            self.evaluate(chosen["version"], "train", self.s.rollouts_train)
        self.save()

    def _tried(self, rnd: dict, reason: str, outcome: str) -> None:
        self.state["tried"].append({"round": rnd["n"], "target": rnd["target"], "reason": reason, "outcome": outcome})

    def final(self) -> None:
        if not self.ctx.inputs.split("final"):
            return
        for v in sorted({"v0", self.state["best"]}):
            res = self.evaluate(v, "final", self.s.rollouts_final)
            self.state["final"][v] = decide.summary(res)
        self.save()

    def rejected_in_a_row(self) -> int:
        n = 0
        for r in reversed(self.state["rounds"]):
            if (r.get("decision") or {}).get("keep") is False:
                n += 1
            elif "decision" in r:
                break
        return n

    # ── the whole run ────────────────────────────────────────────────────────
    def run(self) -> str:
        self.state["status"] = "running"
        self.save()
        try:
            self.start()
            while len([r for r in self.state["rounds"] if "decision" in r]) < self.s.rounds and self.rejected_in_a_row() < self.s.plateau:
                if not self.round():
                    break
            self.final()
            self.state["status"] = "done"
        except safety.Halt as exc:
            self.state["status"], self.state["halt"] = "halted", f"Hark reached a pay, book or send step and would have completed it: {exc}"
        except BudgetExceeded as exc:
            self.state["status"] = f"stopped: {exc}"
        except delivery_mod.DeliveryFailed as exc:
            self.state["status"] = f"failed: the skill did not reach Hark: {exc}"
        finally:
            self._cleanup()
            self.save()
            report.write(self.run_dir, self.state, self.ctx.inputs)
        return self.state["status"]

    def _cleanup(self) -> None:
        if not self.installed:
            return
        try:
            self.state["cleanup"] = self.ctx.delivery.remove()
        except Exception as exc:   # the account keeps the candidate; the report says so
            self.state["cleanup"] = f"could not remove the candidate skill: {type(exc).__name__}: {str(exc)[:160]}"
