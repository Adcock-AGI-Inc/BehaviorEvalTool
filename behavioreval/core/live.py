"""A live conversation: the same user simulation, one turn at a time, with a person watching.
After each assistant reply the user model proposes the next line; the person sends it as is,
edits it, types their own, lets it run on its own (``auto``), or ends it. The record is the
same ``*.conversation.json`` a batch run writes (run tag ``live-<profile>``), judged at the end."""
from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass, field

from behavioreval.core import config as config_mod, ledger, openrouter, records, scenario, scoring
from behavioreval.core.config import Config, column
from behavioreval.core.endpoint import make as make_endpoint
from behavioreval.core.harness import Conversation, Turn
from behavioreval.core.simulator import UserReply, UserSimulator
from behavioreval.core.profile import Profile


@dataclass
class Proposal:
    say: str = ""
    done: bool = False
    why: str = ""
    unreadable: bool = False


class LiveChat:
    """States: starting → waiting (assistant working) → proposing (user model thinking) → your_turn
    (a proposal is ready) → … → judging → finished | error."""

    def __init__(self, cfg: Config, sc: scenario.Scenario, person: Profile, endpoint_name: str = "", key: str = "", out=None,
                 endpoint=None, sim=None, scorer=None):
        self.cfg, self.sc, self.person = cfg, scenario.render(sc, person), scenario.profile_for(sc, person)
        self.key = key or openrouter.key()
        self.endpoint = endpoint or make_endpoint(endpoint_name or cfg.endpoint, cfg, self.log)
        self.sim = sim or UserSimulator(cfg, self.key)
        self.scorer = scorer
        self.auto = False
        self.state = "starting"
        self.proposal: Proposal | None = None
        self.log_lines: list[str] = []
        self.error = ""
        self.score: dict = {}
        self.path = ""
        self.max_turns = self.sc.max_turns or cfg.max_turns
        label, _, _ = column(cfg)
        self.conv = Conversation(scenario=self.sc.id, user=self.person.name, account=person.email or "", profile=person.id, lane=cfg.lane, column=label,
                                 tags=list(self.sc.tags), started=time.strftime("%Y-%m-%d %H:%M:%S"))
        self._next: list = []           # the person's answer to a proposal: [(text, end)]
        self._wake = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._out = out

    def log(self, msg: str) -> None:
        self.log_lines.append(msg.strip())
        if self._out:
            self._out(msg)

    def start(self) -> None:
        self._thread.start()

    def snapshot(self) -> dict:
        return {"state": self.state, "auto": self.auto, "scenario": self.sc.id, "column": self.conv.column, "profile": self.person.id,
                "conversation_id": self.conv.conversation_id, "turns": [asdict(t) for t in self.conv.turns], "max_turns": self.max_turns,
                "proposal": asdict(self.proposal) if self.proposal else None, "error": self.error, "score": self.score, "path": self.path,
                "log": self.log_lines[-30:], "stop": self.conv.stop}

    # ── what the person can do ───────────────────────────────────────────
    def send(self, text: str, end: bool = False) -> None:
        """Send this as the user's next line (empty text + end: stop without sending)."""
        self._next.append((text.strip(), end))
        if self.state == "your_turn":
            self.state = "waiting"   # taken: the loop picks it up next
        self._wake.set()

    def set_auto(self, on: bool) -> None:
        self.auto = on
        if on and self.state == "your_turn" and self.proposal:
            self.send(self.proposal.say, self.proposal.done)

    # ── the loop ─────────────────────────────────────────────────────────
    def _assistant(self, turn: Turn) -> bool:
        self.state = "waiting"
        self.log(f"user {turn.n}/{self.max_turns}: {turn.user[:90]!r}")
        try:
            t = self.endpoint.turn(turn.user, self.cfg.turn_timeout_s)
        except TimeoutError as exc:
            turn.error = f"timeout: {exc}"
            self.log(f"timeout: {exc}")
            return False
        turn.assistant, turn.reason, turn.calls, turn.duration_s, turn.error = t.reply, t.reason, t.calls, t.duration_s, t.error
        if t.note:
            self.conv.note = t.note
        self.log(f"assistant ended {t.reason or '?'} ({len(t.calls)} calls, {t.duration_s:.0f} s)")
        return True

    def _run(self) -> None:
        try:
            self.conv.conversation_id = self.endpoint.open(self.conv.account, self.cfg.lane, f"behavioreval live {self.sc.id} {self.conv.started}", {})
            self.log(f"conversation {self.conv.conversation_id[:8]} as {self.person.name} on {self.conv.column}")
            say = self.sc.opening
            self.conv.stop = "max_turns"
            failed = 0
            n = 0
            while True:
                n += 1
                turn = Turn(n=n, user=say)
                if self.proposal:
                    turn.sim = {"why": self.proposal.why, "unreadable": self.proposal.unreadable, "model": self.cfg.user_model, "proposed": self.proposal.say,
                                "edited": self.proposal.say != say}
                self.conv.turns.append(turn)
                if not self._assistant(turn):
                    self.conv.stop = "timeout"
                    break
                failed = failed + 1 if (turn.error and not turn.assistant) else 0
                if failed >= self.cfg.max_consecutive_errors:
                    self.conv.stop = "assistant_error"
                    break
                if n >= self.max_turns:
                    break
                self.state = "proposing"
                r: UserReply = self.sim.next_message(self.person, self.sc, self.conv.exchanges(), [], time.strftime("%Y-%m-%d"))
                self.proposal = Proposal(say=r.say, done=r.done, why=r.why, unreadable=r.unreadable)
                self.log(f"user model proposes: {r.say[:90]!r} (done={r.done})")
                if self.auto:
                    text, end = r.say, r.done
                else:
                    self.state = "your_turn"
                    self._wake.wait()
                    self._wake.clear()
                    text, end = self._next.pop(0)
                if not text:
                    self.conv.stop = "done"
                    self.conv.turns.append(Turn(n=n + 1, user="", sim={"why": self.proposal.why}))
                    break
                say = text
                if end:   # a closing line: sent, then the conversation ends
                    turn = Turn(n=n + 1, user=say, sim={"why": self.proposal.why, "proposed": self.proposal.say, "edited": self.proposal.say != say})
                    self.conv.turns.append(turn)
                    self.conv.stop = "done" if self._assistant(turn) else "timeout"
                    break
            self.conv.ended = time.strftime("%Y-%m-%d %H:%M:%S")
            self.state = "judging"
            try:
                scorer = self.scorer or scoring.make("judge", self.cfg, self.key, self.log, limits=self.person.limits)
                self.score = scorer.score(self.conv, self.sc)
            except Exception as exc:
                self.score = {"verdict": "UNREADABLE", "lines": [f"judge failed: {type(exc).__name__}: {str(exc)[:200]}"]}
            out_dir = self.cfg.data_root / "runs" / f"{time.strftime('%Y%m%d-%H%M')}-live-{self.person.id}"
            out_dir.mkdir(parents=True, exist_ok=True)
            self.path = str(records.save_conversation(out_dir, self.conv, self.score))
            records.write_report(out_dir, [(self.conv, self.score)], {"live": True, "profile": self.person.id, "column": self.conv.column})
            if self.conv.account:
                ledger.record(ledger.path_for(self.cfg.data_root), self.conv.account, {"scenario": self.sc.id, "profile": self.person.id, "opening": self.sc.opening,
                                                                                       "conversation": self.conv.conversation_id, "column": self.conv.column, "stop": self.conv.stop,
                                                                                       "verdict": self.score.get("verdict", "-"), "tags": list(self.sc.tags) + ["live"]})
            self.state = "finished"
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {str(exc)[:300]}"
            self.conv.stop = self.conv.stop or f"error: {self.error[:100]}"
            self.state = "error"
            self.log(self.error)
        finally:
            try:
                self.endpoint.close()
            except Exception:
                pass
