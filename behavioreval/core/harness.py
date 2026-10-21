"""The loop. opening → the assistant's turn → the user model's next line → … until the user
is done, the turn cap is reached, or an assistant turn never ends. Returns a Conversation;
records.py writes it, scorers read it."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from behavioreval.core.config import Config, column
from behavioreval.core.endpoint import AssistantTurn, Endpoint
from behavioreval.core.scenario import Scenario
from behavioreval.core.simulator import Exchange, UserReply, UserSimulator
from behavioreval.core.profile import Profile


@dataclass
class Turn:
    n: int
    user: str
    assistant: str = ""
    reason: str = ""
    calls: list[dict] = field(default_factory=list)
    duration_s: float = 0.0
    error: str = ""
    sim: dict = field(default_factory=dict)       # the user model's side: why, raw, usage, unreadable, model


@dataclass
class Conversation:
    scenario: str
    user: str                                       # the simulated user's name
    account: str
    profile: str = ""                               # the profile id the user came from
    lane: str = ""
    column: str = ""                                # the assistant column: slot20 / claude-opus-5-5 / default
    tags: list[str] = field(default_factory=list)
    conversation_id: str = ""
    turns: list[Turn] = field(default_factory=list)
    stop: str = ""                                  # done | max_turns | timeout | assistant_error | error:<what>
    started: str = ""
    ended: str = ""
    note: str = ""

    @property
    def user_turns(self) -> int:
        return sum(1 for t in self.turns if t.user)

    def exchanges(self) -> list[Exchange]:
        return [Exchange(user=t.user, assistant=t.assistant, ended=t.reason or t.error) for t in self.turns if t.user]


def _sim_fields(r: UserReply) -> dict:
    return {"why": r.why, "raw": r.raw, "usage": r.usage, "unreadable": r.unreadable, "model": r.model}


def run(sc: Scenario, user: Profile, account: str, sim: UserSimulator, endpoint: Endpoint, cfg: Config,
        history: list[str] | None = None, out=print, today: str | None = None) -> Conversation:
    """``sc`` must already be rendered for ``user`` (scenario.render); ``user`` is the scenario's user (scenario.user_for)."""
    today = today or time.strftime("%Y-%m-%d")
    max_turns = sc.max_turns or cfg.max_turns
    label, _, _ = column(cfg)
    conv = Conversation(scenario=sc.id, user=user.name, account=account, profile=user.id, lane=cfg.lane, column=label, tags=list(sc.tags), started=time.strftime("%Y-%m-%d %H:%M:%S"))
    profile = user.profile if cfg.seed_profile else {}
    conv.conversation_id = endpoint.open(account, cfg.lane, f"behavioreval {sc.id} {conv.started}", profile)
    out(f"  conversation {conv.conversation_id[:8]} as {user.name} ({account or 'no account'}) on {label}")

    def assistant(turn: Turn) -> bool:
        out(f"  user {turn.n}/{max_turns}: {turn.user[:90]!r}")
        try:
            t: AssistantTurn = endpoint.turn(turn.user, cfg.turn_timeout_s)
        except TimeoutError as exc:
            out(f"    timeout: {exc}")
            turn.error = f"timeout: {exc}"
            return False
        turn.assistant, turn.reason, turn.calls, turn.duration_s, turn.error = t.reply, t.reason, t.calls, t.duration_s, t.error
        if t.note:
            conv.note = t.note
        out(f"    assistant ended {t.reason or '?'} ({len(t.calls)} calls, {t.duration_s:.0f} s): {t.reply[:110]!r}" + (f"; error: {t.error[:80]}" if t.error else ""))
        return True

    say, reply = sc.opening, None
    conv.stop = "max_turns"
    failed_in_a_row = 0
    for n in range(1, max_turns + 1):
        turn = Turn(n=n, user=say, sim=_sim_fields(reply) if reply else {})
        conv.turns.append(turn)
        if not assistant(turn):
            conv.stop = "timeout"
            break
        failed_in_a_row = failed_in_a_row + 1 if (turn.error and not turn.assistant) else 0
        if failed_in_a_row >= cfg.max_consecutive_errors:
            out(f"    the assistant failed {failed_in_a_row} turns in a row; stopping")
            conv.stop = "assistant_error"
            break
        if n == max_turns:
            break
        reply = sim.next_message(user, sc, conv.exchanges(), history or [], today)
        out(f"    user model: done={reply.done} why={reply.why[:100]!r}")
        if not reply.say:
            conv.stop = "done"
            conv.turns.append(Turn(n=n + 1, user="", sim=_sim_fields(reply)))
            break
        say = reply.say
        if reply.done:
            turn = Turn(n=n + 1, user=say, sim=_sim_fields(reply))
            conv.turns.append(turn)
            conv.stop = "done" if assistant(turn) else "timeout"
            break
    conv.ended = time.strftime("%Y-%m-%d %H:%M:%S")
    return conv
