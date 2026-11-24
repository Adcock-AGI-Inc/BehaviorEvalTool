import json
import tempfile
import unittest
from pathlib import Path

from behavioreval.core import config
from behavioreval.core.endpoint import AssistantTurn
from behavioreval.core.harness import Conversation, Turn
from behavioreval.refine import decide, delivery, editor, inputs, judges, rollouts, safety, targets
from behavioreval.refine.loop import Refiner
from behavioreval.refine.meter import BudgetExceeded, Meter

RULE = "Ask one question at a time."
GOOD = "FOLLOWS-RULE"


def write_skill(root: Path, start: str | None = "# Gift helper\nHelp the person pick a gift.\n", final: bool = True) -> Path:
    """A filled-in skill folder under root/skills/demo and root/rubric, with fixture text only (no real skill)."""
    (root / "rubric").mkdir(parents=True, exist_ok=True)
    (root / "rubric" / "general.md").write_text("- Asks a little at a time.\n- Uses real details.\n")
    (root / "rubric" / "skill_best_practices.md").write_text("Short plain rules.\n")
    d = root / "skills" / "demo"
    d.mkdir(parents=True, exist_ok=True)
    (d / "goal.md").write_text("Help pick a gift.\n")
    (d / "criteria.md").write_text("- Asks the budget before suggesting.\n")
    (d / "personality.md").write_text("Warm and brief.\n")
    (d / "personas.jsonl").write_text(json.dumps({"id": "sam", "facts": {"name": "Sam Rivera", "home": "Sunnyvale"}, "style": "short"}) + "\n")
    cards = [{"id": "demo/01_train", "split": "train", "persona": "sam", "opening": "help me find a gift for my sister", "hidden": ["budget $80", "sister likes Ceramics"], "done_when": "a gift is suggested"},
             {"id": "demo/02_hold", "split": "holdout", "persona": "sam", "opening": "HOLDOUT-OPENING gift for my dad", "hidden": ["budget $50"], "done_when": "a gift is suggested"}]
    if final:
        cards.append({"id": "demo/03_final", "split": "final", "persona": "sam", "opening": "gift for a coworker", "hidden": [], "done_when": "a gift is suggested"})
    (d / "cards.jsonl").write_text("".join(json.dumps(c) + "\n" for c in cards))
    (d / "refine.json").write_text(json.dumps({"skill_description": "helps pick gifts", "delivery": "fake", "rounds": 2, "rollouts_train": 1, "rollouts_holdout": 1,
                                                "rollouts_final": 1, "candidates_per_round": 2, "parallel": 1, "plateau": 2, "judge_model": "judge/x", "editor_model": "editor/y"}))
    if start is not None:
        (d / "start.md").write_text(start)
    return root


class InputsTests(unittest.TestCase):
    def test_filled_folder_passes_and_placeholders_fail(self):
        root = write_skill(Path(tempfile.mkdtemp()))
        i = inputs.load("demo", root / "skills", root / "rubric")
        self.assertEqual(i.problems, [])
        self.assertEqual([len(i.split(s)) for s in inputs.SPLITS], [1, 1, 1])
        (root / "skills" / "demo" / "goal.md").write_text("PLACEHOLDER: later")
        self.assertIn("goal.md is still a placeholder", inputs.load("demo", root / "skills", root / "rubric").problems)

    def test_bad_split_unknown_persona_and_duplicate_card(self):
        root = write_skill(Path(tempfile.mkdtemp()))
        p = root / "skills" / "demo" / "cards.jsonl"
        p.write_text(p.read_text() + json.dumps({"id": "demo/09", "split": "dev", "persona": "sam", "opening": "x", "done_when": "y"}) + "\n"
                     + json.dumps({"id": "demo/10", "split": "train", "persona": "nobody", "opening": "x", "done_when": "y"}) + "\n"
                     + json.dumps({"id": "demo/01_train", "split": "holdout", "persona": "sam", "opening": "x", "done_when": "y"}) + "\n")
        probs = " | ".join(inputs.load("demo", root / "skills", root / "rubric").problems)
        self.assertIn("split must be one of", probs)
        self.assertIn("persona 'nobody'", probs)
        self.assertIn("appears twice", probs)


class ShippedFilesTests(unittest.TestCase):
    ROOT = Path(__file__).resolve().parent.parent

    def test_dry_run_fixture_is_ready_and_template_is_not(self):
        fx = self.ROOT / "tests" / "fixtures" / "refine"
        self.assertEqual(inputs.load("refine-dryrun", fx / "skills", fx / "rubric").problems, [])
        problems = inputs.load("_template", self.ROOT / "skills", self.ROOT / "rubric").problems
        self.assertTrue(any("goal.md is still a placeholder" in p for p in problems))
        self.assertTrue(any("general.md is still a placeholder" in p for p in problems))


class EditorTests(unittest.TestCase):
    SKILL = "# Gift helper\nHelp the person pick a gift.\nSuggest three ideas.\n"

    def test_apply_kinds_and_exactly_once(self):
        self.assertIn("Suggest two ideas.", editor.apply(self.SKILL, editor.Edit("replace", "Suggest three ideas.", "Suggest two ideas.", "r")))
        self.assertTrue(editor.apply(self.SKILL, editor.Edit("add", "", RULE, "r")).rstrip().endswith(RULE))
        self.assertEqual(editor.apply(self.SKILL, editor.Edit("add", "Help the person pick a gift.", RULE, "r")).splitlines()[2], RULE)
        self.assertNotIn("three", editor.apply(self.SKILL, editor.Edit("delete", "Suggest three ideas.\n", "", "r")))
        with self.assertRaises(ValueError):
            editor.apply(self.SKILL + "Suggest three ideas.\n", editor.Edit("replace", "Suggest three ideas.", "x", "r"))

    def test_guard_rejects_leaks_size_and_noops(self):
        train = ["my sister likes Ceramics", "budget $80", "lives in Sunnyvale"]
        allowed = [self.SKILL, "Help pick a gift."]
        new, why = editor.guard(self.SKILL, editor.Edit("add", "", "If they mention Ceramics, suggest a $80 mug.", "r"), 12, 5000, train, allowed)
        self.assertIsNone(new)
        self.assertIn("ceramics", why)
        self.assertIn("$80", why)
        new, why = editor.guard(self.SKILL, editor.Edit("add", "", RULE, "r"), 12, 5000, train, allowed)
        self.assertEqual(why, "")
        self.assertIn(RULE, new)
        self.assertIn("too big", editor.guard(self.SKILL, editor.Edit("add", "", "\n".join(f"line {i}" for i in range(20)), "r"), 12, 5000, train, allowed)[1])
        self.assertIn("too long", editor.guard(self.SKILL, editor.Edit("add", "", RULE, "r"), 12, 40, train, allowed)[1])
        self.assertIn("does not apply", editor.guard(self.SKILL, editor.Edit("replace", "absent", "x", "r"), 12, 5000, train, allowed)[1])

    def test_a_specific_already_in_the_skill_is_allowed(self):
        skill = self.SKILL + "Never pass the budget.\n"
        new, why = editor.guard(skill, editor.Edit("replace", "Never pass the budget.", "Never pass the budget, and say it back.", "r"), 12, 5000, ["budget $80"], [skill])
        self.assertEqual(why, "")


def res(card, rollout, verdict="PASS", met=(True,), total=60.0, cause=""):
    r = {"card": card, "rollout": rollout, "verdict": verdict, "stop": "done", "speed": {"total_s": total, "user_turns": 3},
         "rubric": {"lines": [{"id": f"S{i}", "met": m, "severity": "" if m else "high"} for i, m in enumerate(met, 1)]}}
    if cause:
        r["diagnosis"] = {"cause": cause, "criterion": "S1"}
    return r


class DecideTests(unittest.TestCase):
    def setUp(self):
        self.best = decide.summary([res("a", 0, "FAIL", (False,)), res("b", 0, "PASS", (True,))])

    def test_quality_up_keeps(self):
        cand = decide.summary([res("a", 0), res("b", 0)])
        self.assertTrue(decide.decide(self.best, cand, ["better"], 100, 100, 0.1, 0.15)[0])

    def test_regression_and_pairwise_rejects(self):
        cand = decide.summary([res("a", 0), res("b", 0, "FAIL", (False,))])
        keep, why = decide.decide(self.best, cand, [], 100, 100, 0.1, 0.15)
        self.assertFalse(keep)
        self.assertIn("regression", why)
        up = decide.summary([res("a", 0), res("b", 0)])
        self.assertIn("pairwise net worse", decide.decide(self.best, up, ["worse", "worse", "better"], 100, 100, 0.1, 0.15)[1])

    def test_speed_counts_only_when_quality_and_pass_rate_hold(self):
        same_fast = decide.summary([res("a", 0, "FAIL", (False,), total=30), res("b", 0, total=30)])
        keep, why = decide.decide(self.best, same_fast, ["equal"], 100, 100, 0.1, 0.15)
        self.assertTrue(keep)
        self.assertIn("faster", why)
        worse_fast = decide.summary([res("a", 0, "FAIL", (False,), total=5), res("b", 0, "FAIL", (False,), total=5)])
        self.assertFalse(decide.decide(self.best, worse_fast, ["equal"], 100, 100, 0.1, 0.15)[0])

    def test_unscorable_rejects(self):
        self.assertFalse(decide.decide(self.best, decide.summary([{**res("a", 0), "stop": "error: x"}]), [], 1, 1, 0.1, 0.15)[0])


class PairwiseTests(unittest.TestCase):
    def verdict(self, first, second):
        answers = iter([{"winner": first}, {"winner": second}])
        g = judges.Graders("k", "judge/x", "diag/y", Meter(10, 10), ask=lambda *a, **k: next(answers))
        conv = Conversation("demo/01", "Sam", "")
        from behavioreval.core.scenario import Scenario
        from behavioreval.core.profile import Profile
        return g.pairwise(conv, conv, Scenario("demo/01", "hi"), Profile("sam"), [("S1", "x")], "goal")["verdict"]

    def test_both_orders_must_agree(self):
        self.assertEqual(self.verdict("A", "B"), "better")
        self.assertEqual(self.verdict("B", "A"), "worse")
        self.assertEqual(self.verdict("A", "A"), "equal")   # position bias: the first slot won both times
        self.assertEqual(self.verdict("tie", "B"), "equal")


class SafetyTests(unittest.TestCase):
    def test_send_tool_halts_and_a_claim_only_warns(self):
        rules = inputs.Settings().halt
        conv = Conversation("s/1", "u", "a", turns=[Turn(1, "hi", "I've placed the order for you.", calls=[{"name": "web_search", "input": {}}])])
        self.assertEqual(safety.check(conv, rules)[0], [])
        self.assertEqual(len(safety.check(conv, rules)[1]), 1)
        conv.turns[0].calls.append({"name": "google_gmail_send_message", "input": {"to": "x@y.z"}})
        self.assertEqual(len(safety.check(conv, rules)[0]), 1)
        conv.turns[0].calls = [{"name": "google_calendar_create_event", "input": {"title": "dentist"}}]
        self.assertEqual(safety.check(conv, rules)[0], [])
        conv.turns[0].calls = [{"name": "google_calendar_create_event", "input": {"attendees": ["x@y.z"]}}]
        self.assertEqual(len(safety.check(conv, rules)[0]), 1)


class MeterTests(unittest.TestCase):
    def test_cost_is_added_and_caps_raise(self):
        m = Meter(1.0, 1, send=lambda body, key, timeout_s=0: {"usage": {"cost": 0.6}, "echo": body})
        resp = m.post_for("judge")({"model": "x"}, "k")
        self.assertEqual(resp["echo"]["usage"], {"include": True})
        self.assertAlmostEqual(m.spent_usd, 0.6)
        m.post_for("judge")({}, "k")
        with self.assertRaises(BudgetExceeded):
            m.post_for("judge")({}, "k")
        Meter(10, 1).start_conversation()
        with self.assertRaises(BudgetExceeded):
            Meter(10, 1, conversations=1).start_conversation()


class ScriptedEndpoint:
    """An assistant that answers by script and records what it was sent."""

    def __init__(self, script):
        self.script, self.sent = list(script), []

    def open(self, account, lane, title, profile):
        return "conv-1"

    def turn(self, text, timeout_s):
        self.sent.append(text)
        return self.script.pop(0)

    def close(self):
        pass


class PersonalSkillDeliveryTests(unittest.TestCase):
    TEXT = "# Gift helper\nAsk the budget first.\nThen suggest three ideas."

    def test_exact_create_and_read_back_is_ok(self):
        ep = ScriptedEndpoint([AssistantTurn("saved", calls=[{"name": "skill", "input": {"action": "delete", "name": "gift"}},
                                                             {"name": "skill", "input": {"action": "create", "name": "gift", "content": self.TEXT + "\n"}}]),
                               AssistantTurn("# Gift helper\nThen suggest three ideas.", calls=[{"name": "skill", "input": {"name": "gift"}}])])
        rec = delivery.PersonalSkillDelivery(lambda: ep, "a@b.c", "gift", "helps", 60).install(self.TEXT, "v1")
        self.assertTrue(rec.ok, rec.detail)
        self.assertIn(delivery.START, ep.sent[0])

    def test_changed_text_is_retried_then_fails(self):
        wrong = AssistantTurn("saved", calls=[{"name": "skill", "input": {"action": "create", "name": "gift", "content": "# Gift helper\nSomething else."}}])
        ep = ScriptedEndpoint([wrong, wrong, wrong])
        rec = delivery.PersonalSkillDelivery(lambda: ep, "a@b.c", "gift", "helps", 60).install(self.TEXT, "v1")
        self.assertFalse(rec.ok)
        self.assertIn("not exactly", ep.sent[1])

    def test_loaded_tells_candidate_from_shared(self):
        d = delivery.PersonalSkillDelivery(None, "", "gift", "", 1)
        conv = Conversation("s/1", "u", "a", turns=[Turn(1, "hi", calls=[{"name": "skill", "input": {"id": "_system/gift"}}])])
        self.assertEqual(d.loaded(conv)[0], False)
        conv.turns[0].calls.append({"name": "skill", "input": {"action": "load", "name": "gift"}})
        self.assertEqual(d.loaded(conv)[0], True)


class TargetTests(unittest.TestCase):
    def test_only_skill_causes_and_tries_count_per_round(self):
        rs = [res("a", 0, "FAIL", (False,), cause="silent"), res("b", 0, "FAIL", (False,), cause="product_or_tool")]
        key, group = targets.pick(rs, [], 2)
        self.assertEqual((key, len(group)), ("rubric S1", 1))
        tried = [{"round": 1, "target": "rubric S1", "outcome": "guard: x"}, {"round": 1, "target": "rubric S1", "outcome": "rejected: y"}]
        self.assertIsNotNone(targets.pick(rs, tried, 2))
        self.assertIsNone(targets.pick(rs, tried + [{"round": 2, "target": "rubric S1", "outcome": "rejected"}], 2))


def fake_ask(log):
    """One fake for every model call, answering by which prompt it got. A conversation passes when Hark's reply carries GOOD."""
    def ask(prompt, model, key, post=None, reasoning_tokens=0, provider=""):
        log.append((model, prompt))
        ok = GOOD in prompt
        if "<judge_task>" in prompt:
            return {"success": ok, "score": 1.0 if ok else 0.0, "reasoning": "fake"}
        if prompt.startswith("You grade one conversation"):
            return {"lines": [{"id": "S1", "met": ok, "evidence": "x", "severity": "" if ok else "high"}], "personalization": {}}
        if prompt.startswith("Two conversations"):
            a, b = prompt.split("<conversation_B>")
            return {"winner": "A" if GOOD in a and GOOD not in b else ("B" if GOOD in b and GOOD not in a else "tie")}
        if prompt.startswith("A conversation with Hark"):
            return {"cause": "silent", "criterion": "S1", "fix_hint": "ask less at once"}
        if prompt.startswith("You improve the text"):
            return {"candidates": [{"kind": "add", "find": "", "text": "Mention Sunnyvale shops.", "reason": "leaky", "targets": "rubric S1"},
                                   {"kind": "add", "find": "", "text": RULE, "reason": "one question at a time", "targets": "rubric S1"}]}
        if prompt.startswith("Write the first version"):
            return {"skill": "# Drafted\nHelp pick a gift.\n"}
        raise AssertionError("unexpected prompt: " + prompt[:80])
    return ask


class LoopTests(unittest.TestCase):
    def build(self, root: Path, opened: list):
        i = inputs.load("demo", root / "skills", root / "rubric")
        self.assertEqual(i.problems, [])
        cfg = config.load(Path("/nonexistent"), {})
        cfg.judge_model, cfg.max_turns, cfg.seed_profile = "judge/x", 2, False
        dlv = delivery.FakeDelivery()

        def make():
            text = dlv.installed[-1][1] if dlv.installed else ""
            class EP(ScriptedEndpoint):
                def open(self, *a):
                    opened.append(text)
                    return f"conv-{len(opened)}"
            return EP([AssistantTurn(("Here are ideas. " + GOOD) if RULE in text else "Here are ideas.", "end_turn", duration_s=2.0)] * 3)
        sim_says = lambda body, key, timeout_s=0: {"choices": [{"message": {"content": json.dumps({"say": "", "done": True, "why": "done"})}}]}   # noqa: E731
        meter = Meter(5.0, 200, send=sim_says)
        log: list = []
        ask = fake_ask(log)
        ctx = rollouts.Context(cfg=cfg, key="test", inputs=i, delivery=dlv, meter=meter, graders=judges.Graders("test", "judge/x", "diag/y", meter, ask=ask),
                               make_endpoint=make, account="dry-run@example.invalid", rubric_lines=judges.rubric_lines(i.general_rubric, i.criteria, i.personality),
                               out=lambda m: None, ask=ask)
        return ctx, log, dlv

    def test_full_loop_keeps_the_good_edit_and_never_shows_holdout_to_the_editor(self):
        root = write_skill(Path(tempfile.mkdtemp()))
        opened: list = []
        ctx, log, dlv = self.build(root, opened)
        run_dir = root / "run"
        status = Refiner(ctx, run_dir).run()
        state = json.loads((run_dir / "state.json").read_text())
        self.assertEqual(status, "done")
        r1 = state["rounds"][0]
        self.assertTrue(r1["decision"]["keep"], r1["decision"])
        self.assertIn("train-card specifics", r1["candidates"][0]["guard"])   # the leaky edit never ran
        self.assertIn(RULE, (run_dir / "versions" / f"{state['best']}.md").read_text())
        editor_prompts = [p for m, p in log if m == "editor/y"]
        self.assertTrue(editor_prompts)
        self.assertFalse(any("HOLDOUT-OPENING" in p for p in editor_prompts))
        self.assertEqual(set(state["final"]), {"v0", state["best"]})
        self.assertTrue(dlv.removed)
        rep = (run_dir / "report.md").read_text()
        for part in ("Scores by round", "Final test", "worst conversations", "Cost", "Final skill text", "Whole diff"):
            self.assertIn(part, rep)
        self.assertIn("one question at a time", (run_dir / "changelog.md").read_text())

    def test_resume_runs_nothing_twice(self):
        root = write_skill(Path(tempfile.mkdtemp()))
        opened: list = []
        ctx, _, _ = self.build(root, opened)
        Refiner(ctx, root / "run").run()
        n = len(opened)
        ctx2, _, _ = self.build(root, opened)
        Refiner(ctx2, root / "run").run()
        self.assertEqual(len(opened), n)

    def test_halt_stops_the_run_and_says_why(self):
        root = write_skill(Path(tempfile.mkdtemp()))
        ctx, _, _ = self.build(root, [])
        ctx.make_endpoint = lambda: ScriptedEndpoint([AssistantTurn("sent", "end_turn", calls=[{"name": "google_gmail_send_message", "input": {"to": "x"}}])] * 3)
        status = Refiner(ctx, root / "run").run()
        self.assertEqual(status, "halted")
        self.assertIn("would have completed it", (root / "run" / "report.md").read_text())

    def test_no_start_drafts_v0_from_the_goal(self):
        root = write_skill(Path(tempfile.mkdtemp()), start=None)
        ctx, _, _ = self.build(root, [])
        r = Refiner(ctx, root / "run")
        r.start()
        self.assertEqual(r.text("v0"), "# Drafted\nHelp pick a gift.")


if __name__ == "__main__":
    unittest.main()
