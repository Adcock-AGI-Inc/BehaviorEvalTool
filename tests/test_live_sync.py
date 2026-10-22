import json
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

from behavioreval.core import config, profile as profile_mod, scenario
from behavioreval.core.live import LiveChat
from behavioreval.core.simulator import UserReply
from behavioreval.endpoints.fake import FakeEndpoint

ALEX = profile_mod.Profile("alex", {"name": "Alex Morgan", "home": "Sunnyvale, CA", "car": "2019 Honda Civic"}, "short", ["never signs in"], email="alex@x")
MECH = {"id": "local/01_mechanic", "opening": "find me a mechanic", "goal": "a shop near {home}", "hidden": ["lives in {home}"], "done_when": "a shop is named", "max_turns": 4}


class ScriptedSim:
    def __init__(self, replies):
        self.replies = list(replies)

    def next_message(self, *a, **k):
        r = self.replies.pop(0)
        return UserReply(say=r[0], done=r[1], why="scripted")


class Scorer:
    def score(self, conv, sc):
        return {"verdict": "PASS", "lines": ["judge: fine"], "judge": [{"success": True, "score": 0.9, "reasoning": "ok"}]}


def wait_state(chat, states, timeout=5):
    t0 = time.time()
    while chat.state not in states and time.time() - t0 < timeout:
        time.sleep(0.02)
    return chat.state


class LiveTests(unittest.TestCase):
    def setUp(self):
        self.cfg = config.load(None, {}, {"data_root": tempfile.mkdtemp(), "endpoint": "fake"})
        self.sc = scenario.from_dict(MECH, "t")

    def test_person_edits_a_proposal_then_ends(self):
        ep = FakeEndpoint([("which city?", "awaiting_input"), ("Joe's Auto", "end_turn"), ("ok booked", "end_turn")])
        chat = LiveChat(self.cfg, self.sc, ALEX, endpoint=ep, sim=ScriptedSim([("sunnyvale", False), ("book it", True)]), scorer=Scorer())
        chat.start()
        self.assertEqual(wait_state(chat, ("your_turn",)), "your_turn")
        self.assertEqual((chat.proposal.say, chat.proposal.done), ("sunnyvale", False))
        chat.send("Sunnyvale, near Fair Oaks")                 # edited
        self.assertEqual(wait_state(chat, ("your_turn",)), "your_turn")
        self.assertEqual(chat.proposal.say, "book it")
        chat.send("book it", end=True)                          # as proposed, as the closing line
        self.assertEqual(wait_state(chat, ("finished", "error")), "finished")
        self.assertEqual(ep.sent, ["find me a mechanic", "Sunnyvale, near Fair Oaks", "book it"])
        snap = chat.snapshot()
        self.assertEqual((snap["stop"], snap["score"]["verdict"], snap["turns"][1]["sim"]["edited"], snap["turns"][2]["sim"]["edited"]), ("done", "PASS", True, False))
        self.assertTrue(Path(snap["path"]).exists() and "live-alex" in snap["path"])
        self.assertEqual(json.loads(Path(snap["path"]).read_text())["profile"], "alex")

    def test_auto_runs_to_the_end_and_a_person_can_stop(self):
        ep = FakeEndpoint([("a", "end_turn")] * 6)
        chat = LiveChat(self.cfg, self.sc, ALEX, endpoint=ep, sim=ScriptedSim([("more", False)] * 6), scorer=Scorer())
        chat.auto = True
        chat.start()
        self.assertEqual(wait_state(chat, ("finished", "error")), "finished")
        self.assertEqual((chat.conv.stop, len(ep.sent)), ("max_turns", 4))
        ep2 = FakeEndpoint([("a", "end_turn")] * 6)
        chat2 = LiveChat(self.cfg, self.sc, ALEX, endpoint=ep2, sim=ScriptedSim([("more", False)] * 6), scorer=Scorer())
        chat2.start()
        wait_state(chat2, ("your_turn",))
        chat2.send("", end=True)                               # end without sending
        self.assertEqual(wait_state(chat2, ("finished", "error")), "finished")
        self.assertEqual((chat2.conv.stop, len(ep2.sent)), ("done", 1))


class SyncTests(unittest.TestCase):
    def test_status_pull_push_against_a_bare_origin(self):
        from behavioreval.core import sync
        tmp = Path(tempfile.mkdtemp())
        origin = tmp / "origin.git"
        subprocess.run(["git", "init", "--quiet", "--bare", str(origin)], check=True)
        work = tmp / "work"
        subprocess.run(["git", "clone", "--quiet", str(origin), str(work)], check=True)
        (work / "scenarios").mkdir(); (work / "profiles").mkdir()
        (work / "scenarios" / "daily.jsonl").write_text("{}\n"); (work / "config.json").write_text("{}\n")
        subprocess.run(["git", "-C", str(work), "-c", "user.name=t", "-c", "user.email=t@x", "add", "-A"], check=True)
        subprocess.run(["git", "-C", str(work), "-c", "user.name=t", "-c", "user.email=t@x", "commit", "--quiet", "-m", "init"], check=True)
        subprocess.run(["git", "-C", str(work), "push", "--quiet", "-u", "origin", "HEAD"], check=True)
        sync.ROOT = work
        st = sync.status()
        self.assertEqual((st["changed"], st["ahead"], st["behind"], st["error"]), ([], 0, 0, ""))
        (work / "scenarios" / "anderson_shop.jsonl").write_text('{"id": "x"}\n')
        self.assertEqual(sync.status(fetch=False)["changed"], ["scenarios/anderson_shop.jsonl"])
        self.assertIn("anderson: 3 shop scenarios", sync.push("anderson", "3 shop scenarios"))
        self.assertEqual(sync.status()["ahead"], 0)
        self.assertEqual(sync.push("anderson", "again"), "nothing to share: no changed scenarios or profiles")
        # someone else pushes; pull brings it in while keeping my uncommitted edit
        other = tmp / "other"
        subprocess.run(["git", "clone", "--quiet", str(origin), str(other)], check=True)
        (other / "scenarios" / "ben_travel.jsonl").write_text('{"id": "y"}\n')
        subprocess.run(["git", "-C", str(other), "-c", "user.name=b", "-c", "user.email=b@x", "add", "-A"], check=True)
        subprocess.run(["git", "-C", str(other), "-c", "user.name=b", "-c", "user.email=b@x", "commit", "--quiet", "-m", "ben"], check=True)
        subprocess.run(["git", "-C", str(other), "push", "--quiet"], check=True)
        (work / "scenarios" / "anderson_shop.jsonl").write_text('{"id": "x2"}\n')
        self.assertEqual(sync.status()["behind"], 1)
        sync.pull()
        self.assertTrue((work / "scenarios" / "ben_travel.jsonl").exists())
        self.assertEqual((work / "scenarios" / "anderson_shop.jsonl").read_text(), '{"id": "x2"}\n')


if __name__ == "__main__":
    unittest.main()
