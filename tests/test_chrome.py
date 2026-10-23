import json
import socket
import tempfile
import threading
import unittest
from pathlib import Path

from behavioreval.browser import cdp, chrome, viewer
from behavioreval.core import config
from behavioreval.core.endpoint import make
from behavioreval.endpoints.chrome import ChromeEndpoint, delivered, fetch_js


class FrameTests(unittest.TestCase):
    def test_frames_round_trip_all_sizes(self):
        for n in (0, 5, 126, 70000):
            payload = bytes(i % 251 for i in range(n))
            a, b = socket.socketpair()
            t = threading.Thread(target=a.sendall, args=(cdp.encode_frame(payload),))   # a big frame needs a reader on the other end
            t.start()
            op, got = cdp.read_frame(b)
            t.join()
            self.assertEqual((op, got), (1, payload))
            a.close(); b.close()


class ChromeProfileTests(unittest.TestCase):
    def test_list_profiles_reads_local_state(self):
        d = Path(tempfile.mkdtemp())
        (d / "Local State").write_text(json.dumps({"profile": {"info_cache": {"Default": {"name": "Person 1", "user_name": "a@hark.com"}, "Profile 3": {"name": "Work"}}}}))
        self.assertEqual(chrome.list_profiles(d), [{"dir": "Default", "name": "Person 1", "email": "a@hark.com"}, {"dir": "Profile 3", "name": "Work", "email": ""}])
        self.assertEqual(chrome.list_profiles(d / "nope"), [])
        with self.assertRaises(ValueError):
            chrome.trust_profile(d, "Profile 9", d / "copy")


class ViewerTests(unittest.TestCase):
    def test_wait_settled_and_calls(self):
        seq = [[{"start": 2, "reason": ""}], [{"start": 2, "reason": "end_turn", "job": "j1", "calls": 3}]]
        fetch = lambda *a, **k: seq.pop(0) if len(seq) > 1 else seq[0]   # noqa: E731
        runs = viewer.wait_for_runs("v", "dev", "c", 1, {}, 100, sleep=lambda s: None, fetch=fetch)
        self.assertEqual(runs[0]["job"], "j1")
        with self.assertRaises(TimeoutError):
            viewer.wait_for_runs("v", "dev", "c", 2, {}, 1, sleep=lambda s: None, now=iter([0, 5]).__next__, fetch=fetch)
        self.assertEqual(viewer.settled("v", "dev", "c", {}, 100, sleep=lambda s: None, fetch=fetch)[0]["reason"], "end_turn")
        lines = [{"lanes": [{"dispatch_path": "sub", "messages": []}, {"messages": [{"part": "run", "content": [{"type": "tool_use", "name": "web_search", "input": {"q": 1}}, {"type": "text"}]}, {"part": "prompt", "content": [{"type": "tool_use", "name": "x"}]}]}]}]
        self.assertEqual(viewer.calls_of(lines), [{"name": "web_search", "input": {"q": 1}, "result": "", "is_error": False}])


class FakeTab:
    """Answers the page-side fetches the endpoint makes, in order."""

    def __init__(self):
        self.urls, self.sent = [], []
        self.rows = []

    def js(self, expr):
        if expr.startswith("({url"):
            return {"url": "https://dev.hark.com/", "title": "Hark"}
        if "/api/connectors" in expr:
            return {"status": 200, "json": {}}
        if "/api/conversations" in expr and "POST" in expr:
            return {"status": 200, "json": {"conversationId": "conv-1"}}
        if "/api/messages/send" in expr:
            body = json.loads(json.loads(expr.split("body: ")[1].split("}); const t")[0]))
            self.sent.append(body)
            if str(body.get("message", "")).startswith("/model"):   # a slash command: no stored reply, no run
                return {"status": 200, "json": {}}
            self.rows.append({"id": f"a{len(self.sent)}", "role": "assistant", "source": "send_bubble", "content": [{"text": f"reply {len(self.sent)}"}]})
            return {"status": 200, "json": {}}
        if "/api/messages/page" in expr:
            return {"status": 200, "json": {"messages": list(self.rows)}}
        return None

    def navigate(self, url):
        self.urls.append(url)

    def front(self):
        pass

    def close(self):
        pass


class ChromeEndpointTests(unittest.TestCase):
    def setUp(self):
        self.saved = {k: getattr(viewer, k) for k in ("conversation_runs", "wait_for_runs", "settled", "rollouts", "folder_for")}
        viewer.folder_for = lambda *a, **k: "me@hark.com" if a[-1] else "mine"

    def tearDown(self):
        for k, v in self.saved.items():
            setattr(viewer, k, v)

    def test_open_send_wait_read(self):
        cfg = config.load(None, {}, {"endpoint": "chrome", "lane": "slot20"})
        tab = FakeTab()
        ep = ChromeEndpoint(cfg, out=lambda m: None, tab=tab, hdrs={"cf-access-token": "t"})
        runs = [{"start": 1, "reason": "end_turn", "job": "j1", "calls": 2, "providers": ["hark-slot20"], "durMs": 12000}]
        viewer.conversation_runs = lambda *a, **k: runs
        viewer.wait_for_runs = lambda *a, **k: runs
        viewer.settled = lambda *a, **k: runs
        refs = []
        viewer.rollouts = lambda v, e, r, h: (refs.extend(r) or [{"lanes": [{"messages": [{"part": "run", "content": [{"type": "tool_use", "name": "web_search", "input": {}}]}]}]}])
        import behavioreval.endpoints.chrome as C
        C.time.sleep = lambda s: None
        conv = ep.open("me@hark.com", "slot20", "t", {"name": "x"})
        self.assertEqual(conv, "conv-1")
        self.assertEqual(tab.sent[0]["message"], "/model hark-slot20")   # step 1 on a lane: the slash command
        self.assertIn("https://dev.hark.com/project/conv-1", tab.urls)
        t = ep.turn("find me a mechanic", 60)
        self.assertEqual(tab.sent[1]["model"], "agent-llm")
        self.assertEqual(tab.sent[1]["modelProvider"], "hark-slot20")
        self.assertEqual((t.reply, t.reason, t.calls[0]["name"], t.duration_s), ("reply 2", "end_turn", "web_search", 12.0))
        self.assertIn("served by hark-slot20", t.note)
        self.assertEqual((ep.folder, refs[0]), ("me@hark.com", "me@hark.com/conv-1/j1"))   # the account's own viewer folder
        t2 = ep.turn("sunnyvale", 60)
        self.assertEqual((t2.reply, t2.note), ("reply 3", ""))   # only the new row, no note after the first turn
        self.assertEqual(ep.sent, 2)
        self.assertTrue(delivered({"role": "assistant", "source": "send_bubble", "content": "hi"}))
        self.assertFalse(delivered({"role": "assistant", "source": "tool", "content": "hi"}))
        self.assertIn("credentials: 'include'", fetch_js("u", "GET"))

    def test_registry_and_default_column_sends_no_override(self):
        cfg = config.load(None, {}, {"endpoint": "chrome"})
        self.assertIsInstance(make("chrome", cfg, lambda m: None), ChromeEndpoint)
        tab = FakeTab()
        ep = ChromeEndpoint(cfg, out=lambda m: None, tab=tab, hdrs={})
        runs = [{"start": 1, "reason": "end_turn"}]
        viewer.conversation_runs = lambda *a, **k: runs
        viewer.wait_for_runs = lambda *a, **k: runs
        viewer.settled = lambda *a, **k: runs
        import behavioreval.endpoints.chrome as C
        C.time.sleep = lambda s: None
        ep.open("", "", "t", {})
        ep.turn("hi", 10)
        self.assertNotIn("model", tab.sent[0])
        self.assertEqual(ep.folder, "mine")


if __name__ == "__main__":
    unittest.main()
