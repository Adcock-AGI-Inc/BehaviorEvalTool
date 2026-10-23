import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from behavioreval.core import config, ledger, records, scenario, profile as profile_mod
from behavioreval.core.harness import run
from behavioreval.core.simulator import Exchange, UserSimulator, parse, system_prompt
from behavioreval.endpoints.fake import FakeEndpoint

ALEX = profile_mod.Profile("alex", {"name": "Alex Morgan", "home": "Sunnyvale, CA", "car": "2019 Honda Civic"}, "short, casual", ["never signs in"], {"name": "Alex Morgan"})
MECH = {"id": "local/01_mechanic", "opening": "find me a mechanic", "situation": "the {car} has a cracked mirror", "goal": "a shop near {home}",
        "hidden": ["lives in {home}", "drives a {car}"], "done_when": "a shop near {home} is named", "scoring": {"criteria": ["Today is {{today}}. Near {home}."]}}


def write_jsonl(rows, name="s.jsonl") -> Path:
    p = Path(tempfile.mkdtemp()) / name
    p.write_text("# c\n" + "".join((json.dumps(r) if isinstance(r, dict) else r) + "\n" for r in rows))
    return p


class ConfigTests(unittest.TestCase):
    def test_file_env_cli_precedence_and_coercion(self):
        f = Path(tempfile.mkdtemp()) / "config.json"
        f.write_text(json.dumps({"max_turns": 4, "lane": "hark-slot4", "data_root": "~/x"}))
        cfg = config.load(f, {"BEHAVIOREVAL_MAX_TURNS": "3", "BEHAVIOREVAL_SEED_PROFILE": "false"}, {"temperature": 0.2, "lane": None})
        self.assertEqual((cfg.max_turns, cfg.source["max_turns"], cfg.lane, cfg.seed_profile, cfg.temperature), (3, "env", "slot4", False, 0.2))
        self.assertTrue(cfg.data_root.is_absolute())
        self.assertEqual(config.load(f, {}, {"lane": "claude"}).lane, "")

    def test_unknown_key_and_bad_turns_fail(self):
        f = Path(tempfile.mkdtemp()) / "config.json"
        f.write_text(json.dumps({"nope": 1}))
        with self.assertRaises(ValueError):
            config.load(f, {})
        f.write_text(json.dumps({"max_turns": 0}))
        with self.assertRaises(ValueError):
            config.load(f, {})

    def test_the_checked_in_config_and_default_profile_load(self):
        cfg = config.load(environ={})
        self.assertEqual((cfg.max_turns, cfg.endpoint, cfg.profile), (10, "chrome", "default"))
        self.assertEqual([c["id"] for c in cfg.columns][:1], ["default"])
        self.assertEqual(config.load(None, {"BEHAVIOREVAL_COLUMNS": "default, slot7"}).columns[1], {"id": "slot7", "label": "slot7"})
        u = profile_mod.load(cfg.profiles_dir, "default")
        self.assertTrue(u.has("home") and u.limits and u.style and u.id == "default")

    def test_column_specs(self):
        cfg = config.load(environ={})
        self.assertEqual(config.column(config.with_column(cfg, "slot20")), ("slot20", "agent-llm", "hark-slot20"))
        self.assertEqual(config.column(config.with_column(cfg, "claude-opus-5-5")), ("claude-opus-5-5", "claude-opus-5-5", ""))
        self.assertEqual(config.column(config.with_column(cfg, "default")), ("default", "", ""))

    def test_secrets_and_accounts_files(self):
        d = Path(tempfile.mkdtemp())
        (d / ".env").write_text("OPENROUTER_API_KEY=abc   # comment\nCF_ACCESS_CLIENT_ID='id1'\nOTHER=x\n")
        (d / "accounts.tsv").write_text("# head\na@x\tpw1\nb@x\tpw2\nbad line\n")
        cfg = config.load(None, {}, {"secrets_file": str(d / ".env"), "accounts_file": str(d / "accounts.tsv")})
        self.assertEqual(config.secrets(cfg, {}), {"OPENROUTER_API_KEY": "abc", "CF_ACCESS_CLIENT_ID": "id1"})
        self.assertEqual(config.load_accounts(cfg.accounts_file), {"a@x": "pw1", "b@x": "pw2"})


class UserTests(unittest.TestCase):
    def test_overrides_and_refused_profile_fields(self):
        u = ALEX.with_overrides({"car": "2016 Subaru Outback", "style": "terse"})
        self.assertEqual((u.facts["car"], u.facts["home"], u.style, u.limits), ("2016 Subaru Outback", "Sunnyvale, CA", "terse", ["never signs in"]))
        self.assertIs(ALEX.with_overrides({}), ALEX)
        with self.assertRaises(ValueError) as cm:
            profile_mod.from_dict({"id": "x", "facts": {"name": "X"}, "profile": {"car": "y"}}, "t")
        self.assertIn("car", str(cm.exception))
        with self.assertRaises(ValueError):
            profile_mod.from_dict({"id": "x", "facts": {}}, "t")


class ScenarioTests(unittest.TestCase):
    def test_load_render_and_placeholders(self):
        sc = scenario.load_file(write_jsonl([MECH]))[0]
        self.assertEqual(sc.placeholders(), {"car", "home"})
        r = scenario.render(sc, ALEX)
        self.assertEqual(r.hidden, ["lives in Sunnyvale, CA", "drives a 2019 Honda Civic"])
        self.assertEqual(r.scoring["criteria"], ["Today is {{today}}. Near Sunnyvale, CA."])

    def test_scenario_user_block_overrides_the_default_user(self):
        sc = scenario.from_dict({**MECH, "user": {"car": "2016 Subaru Outback"}}, "t")
        self.assertEqual(scenario.render(sc, ALEX).hidden[1], "drives a 2016 Subaru Outback")
        self.assertEqual(scenario.profile_for(sc, ALEX).facts["car"], "2016 Subaru Outback")

    def test_lint(self):
        sc = scenario.from_dict(MECH, "t")
        self.assertEqual(scenario.lint(sc, ALEX), [])
        bad = scenario.from_dict({**MECH, "goal": "near {office}"}, "t")
        self.assertIn("{office}", scenario.lint(bad, ALEX)[0])
        nodone = scenario.from_dict({**MECH, "done_when": ""}, "t")
        self.assertTrue(any("done_when" in p for p in scenario.lint(nodone, ALEX)))
        few = scenario.from_dict({**MECH, "beats": ["a", "b", "c"], "max_turns": 2}, "t")
        self.assertTrue(any("beats" in p for p in scenario.lint(few, ALEX)))
        runtime = scenario.from_dict({**MECH, "opening": "undo {prev_opening}"}, "t")
        self.assertEqual(scenario.lint(runtime, ALEX), [])   # prev_* is filled at run time

    def test_bad_input_fails_clearly(self):
        with self.assertRaises(ValueError) as cm:
            scenario.load_file(write_jsonl([MECH, "{bad"]))
        self.assertIn(":3:", str(cm.exception))
        with self.assertRaises(ValueError):
            scenario.load_file(write_jsonl([MECH, MECH]))
        with self.assertRaises(ValueError):
            scenario.from_dict({**MECH, "id": "no-slash"}, "t")
        with self.assertRaises(ValueError) as cm:
            scenario.from_dict({**MECH, "goals": "typo"}, "t")
        self.assertIn("goals", str(cm.exception))

    def test_checked_in_scenarios_lint_clean(self):
        cfg = config.load(environ={})
        u = profile_mod.load(cfg.profiles_dir, "default")
        self.assertTrue(scenario.sets(cfg.scenarios_dir))
        for s in scenario.sets(cfg.scenarios_dir):
            self.assertEqual(scenario.lint_all(scenario.resolve(s, directory=cfg.scenarios_dir), u), [], s)
        self.assertGreaterEqual(len(scenario.resolve("all", directory=cfg.scenarios_dir)), 16)

    def test_save_delete_keep_comments_and_replace_in_place(self):
        d = Path(tempfile.mkdtemp())
        a = scenario.from_dict(MECH, "t")
        scenario.save(d, "local", a)
        (d / "local.jsonl").write_text("# keep me\n" + (d / "local.jsonl").read_text())
        b = scenario.from_dict({**MECH, "id": "local/02_other", "done_when": "x"}, "t")
        scenario.save(d, "local", b)
        scenario.save(d, "local", scenario.from_dict({**MECH, "goal": "changed"}, "t"))
        text = (d / "local.jsonl").read_text()
        self.assertTrue(text.startswith("# keep me\n"))
        scs = scenario.load_file(d / "local.jsonl")
        self.assertEqual([s.id for s in scs], ["local/01_mechanic", "local/02_other"])
        self.assertEqual(scs[0].goal, "changed")
        self.assertTrue(scenario.delete(d, "local", "local/01_mechanic"))
        self.assertEqual([s.id for s in scenario.load_file(d / "local.jsonl")], ["local/02_other"])
        with self.assertRaises(ValueError):
            scenario.save(d, "../x", a)


class SimulatorTests(unittest.TestCase):
    def setUp(self):
        self.sc = scenario.render(scenario.from_dict(MECH, "t"), ALEX)
        self.cfg = config.load(environ={})

    def test_system_prompt_carries_user_scenario_limits_history_and_date(self):
        s = system_prompt(ALEX, self.sc, ["2026-10-19: asked for \"a dentist\""], "2026-10-20")
        for part in ("Alex Morgan", "short, casual", "never signs in", "lives in Sunnyvale, CA", "a dentist", "2026-10-20", '"say"'):
            self.assertIn(part, s)

    def test_parse_shapes(self):
        self.assertEqual(parse('{"say": "sunnyvale", "done": false, "why": "asked"}').say, "sunnyvale")
        self.assertTrue(parse('```json\n{"say": "", "done": true, "why": "ok"}\n```').done)
        self.assertEqual(parse('{".say": "sunnyvale", "done": false}').say, "sunnyvale")
        self.assertTrue(parse('{"say": "", "done": false}').unreadable)
        self.assertTrue(parse("I am an AI").unreadable)

    def test_next_message_posts_without_thinking_and_retries_once(self):
        answers = ['{"say": "", "done": false, "why": "oops"}', '{"say": "sunnyvale", "done": false, "why": "ok"}']
        bodies = []

        def post(body, key):
            bodies.append(body)
            return {"choices": [{"message": {"content": answers.pop(0)}}], "usage": {"total_tokens": 5}}
        r = UserSimulator(self.cfg, "k", post).next_message(ALEX, self.sc, [Exchange("find me a mechanic", "which city?")], [], "2026-10-20")
        self.assertEqual((r.say, len(bodies)), ("sunnyvale", 2))
        self.assertEqual(bodies[0]["reasoning"], {"enabled": False})
        self.assertEqual([m["role"] for m in bodies[0]["messages"]], ["system", "assistant", "user"])
        r = UserSimulator(self.cfg, "k", lambda b, k: (_ for _ in ()).throw(OSError("down"))).next_message(ALEX, self.sc, [], [], "2026-10-20")
        self.assertEqual(r.say, "hold on, can you say that again in one line?")


def scripted(answers):
    answers = list(answers)
    return lambda body, key: {"choices": [{"message": {"content": json.dumps(answers.pop(0))}}], "usage": {}}


class HarnessTests(unittest.TestCase):
    def setUp(self):
        self.cfg = config.load(environ={}, overrides={"max_turns": 3, "lane": "slot4"})
        self.sc = scenario.render(scenario.from_dict(MECH, "t"), ALEX)
        self.out = []

    def go(self, ep, answers):
        return run(self.sc, ALEX, "a@x", UserSimulator(self.cfg, "k", scripted(answers)), ep, self.cfg, ["earlier"], self.out.append, "2026-10-20")

    def test_opening_then_model_until_done(self):
        ep = FakeEndpoint([("which city?", "awaiting_input"), ("Joe's Auto, 123 Main", "end_turn")])
        conv = self.go(ep, [{"say": "sunnyvale", "done": False, "why": "asked"}, {"say": "", "done": True, "why": "named"}])
        self.assertEqual(ep.sent, ["find me a mechanic", "sunnyvale"])
        self.assertEqual((conv.stop, conv.user_turns, len(conv.turns), conv.user, conv.profile), ("done", 2, 3, "Alex Morgan", "alex"))
        self.assertEqual(ep.opened["profile"], {"name": "Alex Morgan"})

    def test_closing_line_is_sent_then_stops(self):
        ep = FakeEndpoint([("two shops", "end_turn"), ("booked", "end_turn")])
        conv = self.go(ep, [{"say": "ok book it", "done": True, "why": "fine"}])
        self.assertEqual((ep.sent, conv.stop, conv.turns[1].assistant), (["find me a mechanic", "ok book it"], "done", "booked"))

    def test_turn_cap_timeout_and_repeated_errors(self):
        from behavioreval.core.endpoint import AssistantTurn
        conv = self.go(FakeEndpoint([("a", "end_turn"), ("b", "end_turn"), ("c", "end_turn")]), [{"say": "more", "done": False, "why": "x"}] * 2)
        self.assertEqual((conv.stop, conv.user_turns), ("max_turns", 3))
        self.assertEqual(self.go(FakeEndpoint([]), []).stop, "timeout")

        class Broken(FakeEndpoint):
            def turn(self, text, timeout_s):
                self.sent.append(text)
                return AssistantTurn(reply="", reason="error", error="Agent run failed", duration_s=3.0)
        ep = Broken()
        self.assertEqual(self.go(ep, [{"say": "hello?", "done": False, "why": "x"}] * 3).stop, "assistant_error")
        self.assertEqual(len(ep.sent), 2)

    def test_records_round_trip_and_report(self):
        d = Path(tempfile.mkdtemp())
        conv = self.go(FakeEndpoint([("a", "end_turn")]), [{"say": "", "done": True, "why": "x"}])
        p = records.save_conversation(d, conv, {"verdict": "PASS", "lines": ["judge 0: ok"]})
        self.assertEqual(p.name, "slot4__a__local_01_mechanic.conversation.json")
        back, raw = records.load_conversation(p)
        self.assertEqual((back.scenario, back.user_turns, raw["score"]["verdict"]), ("local/01_mechanic", 1, "PASS"))
        records.write_report(d, [(back, raw["score"])], {"x": 1})
        self.assertIn("| slot4 | 1 |", (d / "report.md").read_text())
        self.assertEqual(json.loads((d / "summary.json").read_text())["conversations"][0]["verdict"], "PASS")


class LedgerAndPlanTests(unittest.TestCase):
    def test_record_history_and_least_used(self):
        p = Path(tempfile.mkdtemp()) / "ledger.json"
        ledger.record(p, "a@x", {"scenario": "s1", "opening": "find me a mechanic", "stop": "done"})
        led = ledger.load(p)
        self.assertIn('asked for "find me a mechanic" (ended: done)', ledger.history(led, "a@x")[0])
        self.assertEqual(ledger.least_used_first(led, ["a@x", "b@x"]), ["b@x", "a@x"])

    def test_plan_spreads_jobs_over_accounts_and_lints_first(self):
        from behavioreval.core.cli import plan_jobs
        cfg = config.load(environ={})
        scs = [scenario.from_dict({**MECH, "id": f"t/{i}"}, "t") for i in range(4)]
        jobs = plan_jobs(cfg, scs, ["default", "slot20"], 1, ["a@x", "b@x", "c@x"], {}, ALEX)
        self.assertEqual(len(jobs), 8)
        self.assertEqual({j[2] for j in jobs}, {"a@x", "b@x", "c@x"})
        self.assertEqual((jobs[0][0].goal, jobs[0][1].id), ("a shop near Sunnyvale, CA", "alex"))
        with self.assertRaises(ValueError):
            plan_jobs(cfg, [scenario.from_dict({**MECH, "goal": "{office}"}, "t")], ["default"], 1, ["a@x"], {}, ALEX)
        long = scenario.from_dict({**MECH, "max_turns": 8}, "t")
        self.assertEqual(plan_jobs(cfg, [long], ["default"], 1, [], {}, ALEX)[0][0].max_turns, 8)                       # the scenario's own
        capped = config.load(None, {}, {"max_turns": 2})
        self.assertEqual(plan_jobs(capped, [long], ["default"], 1, [], {}, ALEX)[0][0].max_turns, 2)                   # an explicit --max-turns caps it


class UITests(unittest.TestCase):
    def setUp(self):
        from behavioreval.ui.server import make_server
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "scenarios").mkdir()
        (self.tmp / "scenarios" / "local.jsonl").write_text(json.dumps(MECH) + "\n")
        (self.tmp / "profiles").mkdir()
        (self.tmp / "profiles" / "default.json").write_text(json.dumps({"id": "default", "facts": ALEX.facts, "style": ALEX.style, "limits": ALEX.limits, "profile": ALEX.profile}))
        (self.tmp / "accounts.tsv").write_text("a1@x\tpw\na2@x\tpw\nfree@x\tpw\n")
        self.cfg = config.load(None, {}, {"data_root": str(self.tmp / "data"), "scenarios_dir": str(self.tmp / "scenarios"), "profiles_dir": str(self.tmp / "profiles"),
                                           "accounts_file": str(self.tmp / "accounts.tsv"), "secrets_file": str(self.tmp / "none.env")})
        self.srv = make_server(self.cfg, 0)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.srv.server_address[1]}"

    def tearDown(self):
        self.srv.shutdown()

    def call(self, path, body=None):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode() if body is not None else None, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read())

    def test_page_scenarios_save_runs_annotate_export(self):
        with urllib.request.urlopen(self.base + "/") as r:
            self.assertIn(b"Behavior Eval Tool", r.read())
        st = self.call("/api/state")
        self.assertEqual((st["default_profile"], st["sets"], [p["id"] for p in st["profiles"]]), ("default", ["local"], ["default"]))
        self.assertTrue(any(not c["ok"] for c in st["checks"]))
        # profiles: bind accounts, exclusivity, usable accounts of an unbound profile
        r = self.call("/api/profiles", {"id": "anderson", "facts": {"name": "Anderson", "home": "Oakland, CA", "car": "2016 Subaru"}, "accounts": ["a1@x", "a2@x"],
                                        "email": "anderson@hark.com", "chrome_profile": "Profile 3"})
        self.assertEqual((r["id"], r["problems"]), ("anderson", []))
        self.assertEqual([(p["email"], p["chrome_profile"]) for p in self.call("/api/state")["profiles"] if p["id"] == "anderson"], [("anderson@hark.com", "Profile 3")])
        self.assertIn("chrome", self.call("/api/state"))
        with self.assertRaises(urllib.error.HTTPError):
            self.call("/api/profiles", {"id": "ben", "facts": {"name": "Ben", "home": "x"}, "accounts": ["a1@x"]})   # a1 belongs to anderson
        rows = {p["id"]: p for p in self.call("/api/profiles")}
        self.assertEqual((rows["anderson"]["usable_accounts"], rows["default"]["usable_accounts"]), (["a1@x", "a2@x"], ["free@x"]))
        self.assertEqual({a["email"]: a["profile"] for a in self.call("/api/state")["accounts"]}, {"a1@x": "anderson", "a2@x": "anderson", "free@x": ""})
        self.assertEqual(self.call("/api/scenarios?profile=anderson")[0]["problems"], [])
        scs = self.call("/api/scenarios")
        self.assertEqual((scs[0]["id"], scs[0]["problems"]), ("local/01_mechanic", []))
        r = self.call("/api/scenarios", {"set": "local", "scenario": {**MECH, "id": "local/02_new", "goal": "near {office}"}})
        self.assertIn("{office}", r["problems"][0])
        self.assertEqual(len(self.call("/api/scenarios")), 2)
        with self.assertRaises(urllib.error.HTTPError):
            self.call("/api/scenarios", {"set": "other", "scenario": {**MECH, "id": "local/03"}})   # id must start with the set
        pv = self.call("/api/preview", {"id": "local/01_mechanic", "assistant": [], "call_model": False})
        self.assertIn("lives in Sunnyvale, CA", pv["system_prompt"])
        pv = self.call("/api/preview", {"id": "local/01_mechanic", "assistant": [], "call_model": False, "profile": "anderson"})
        self.assertIn("lives in Oakland, CA", pv["system_prompt"])
        self.assertEqual(self.call("/api/runs"), [])
        # a finished run on disk → results, annotation, export
        rd = self.cfg.data_root / "runs" / "20261021-1200-test"
        rd.mkdir(parents=True)
        conv = {"scenario": "local/01_mechanic", "user": "Alex Morgan", "account": "a@x", "profile": "default", "lane": "", "column": "default", "tags": [], "conversation_id": "c1",
                "turns": [{"n": 1, "user": "find me a mechanic", "assistant": "Joe's", "reason": "end_turn", "calls": [], "duration_s": 2, "error": "", "sim": {}}],
                "stop": "done", "started": "", "ended": "", "note": "", "score": {"verdict": "PASS", "judge": [{"success": True, "score": 0.9}]}}
        (rd / "default__a__local_01_mechanic.conversation.json").write_text(json.dumps(conv))
        runs = self.call("/api/runs")
        self.assertEqual((runs[0]["name"], runs[0]["verdicts"], runs[0]["profiles"]), ("20261021-1200-test", {"PASS": 1}, ["default"]))
        rows = self.call("/api/run?name=20261021-1200-test")
        self.assertEqual((rows[0]["verdict"], rows[0]["score"], rows[0]["annotation"]), ("PASS", 0.9, {}))
        ann = self.call("/api/annotate", {"run": "20261021-1200-test", "file": rows[0]["file"], "annotation": {"verdict": "FAIL", "issues": ["fabricated specifics"], "comment": "no such shop", "by": "anderson"}})
        self.assertEqual(ann["verdict"], "FAIL")
        self.assertEqual(self.call("/api/run?name=20261021-1200-test")[0]["annotation"]["issues"], ["fabricated specifics"])
        with urllib.request.urlopen(self.base + "/api/export?name=20261021-1200-test") as r:
            csv_text = r.read().decode()
        self.assertIn("no such shop", csv_text)
        with self.assertRaises(urllib.error.HTTPError):
            self.call("/api/run?name=../etc")


if __name__ == "__main__":
    unittest.main()
