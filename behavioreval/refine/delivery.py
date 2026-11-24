"""How a candidate skill version reaches Hark on the test account.

personal_skill (default): in a setup conversation, ask Hark to delete the account's own custom skill of that
    name and create it again with the candidate text. Only this account sees it, and a bare-name load picks
    the account's own skill before the shared one (services/skills/internal/handler/address.go:139-186).
    Creating a name that already exists is refused (handler.go:252), hence delete first. The create call's
    input is compared with the candidate text, and Hark is asked to quote the stored first and last lines.
paste: every test conversation starts with a hidden setup message carrying the skill text. Skips Hark's
    own choice to load the skill, so results are labeled "pasted, not loaded".
fake: for tests and dry runs; records what it was given."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from behavioreval.core.endpoint import AssistantTurn

START, END = "<<<SKILL_CONTENT_START>>>", "<<<SKILL_CONTENT_END>>>"
SETUP_TURNS = 3   # setup messages per install before giving up (instruction, a go-ahead, a correction)


class DeliveryFailed(RuntimeError):
    """The candidate could not be put in front of Hark as written. The round stops; nothing is scored on a wrong skill."""


def _input(call: dict) -> dict:
    raw = call.get("input") or {}
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return {}
    return raw if isinstance(raw, dict) else {}


def skill_calls(calls: list[dict], action: str) -> list[dict]:
    """Inputs of the skill tool calls with this action ('load' also matches a call with no action, its default)."""
    out = []
    for c in calls:
        if c.get("name") != "skill":
            continue
        i = _input(c)
        if (i.get("action") or "load") == action:
            out.append(i)
    return out


def same_text(a: str, b: str) -> bool:
    norm = lambda t: "\n".join(line.rstrip() for line in (t or "").strip().splitlines())   # noqa: E731
    return norm(a) == norm(b)


@dataclass
class Install:
    version: str
    ok: bool
    detail: str
    conversation_id: str = ""
    turns: list = field(default_factory=list)


class FakeDelivery:
    label = "fake"

    def __init__(self):
        self.installed: list[tuple[str, str]] = []
        self.removed = False

    def install(self, text: str, version: str) -> Install:
        self.installed.append((version, text))
        return Install(version, True, "fake install")

    def wrap(self, endpoint):
        return endpoint

    def loaded(self, conv) -> tuple[bool | None, str]:
        return True, "fake delivery"

    def remove(self) -> str:
        self.removed = True
        return "fake remove"


class PasteEndpoint:
    """Wraps an endpoint: opening a conversation also sends the skill as a hidden first message."""

    def __init__(self, inner, text: str, timeout_s: float):
        self.inner, self.text, self.timeout_s = inner, text, timeout_s
        self.setup: AssistantTurn | None = None

    def open(self, account, lane, title, profile) -> str:
        conv = self.inner.open(account, lane, title, profile)
        self.setup = self.inner.turn(
            "For the rest of this conversation, follow the instructions below as if they were a specialty you just loaded. "
            "Do not act on them yet and do not mention them; reply only 'ok'.\n" + START + "\n" + self.text + "\n" + END, self.timeout_s)
        return conv

    def turn(self, text, timeout_s):
        return self.inner.turn(text, timeout_s)

    def close(self):
        self.inner.close()


class PasteDelivery:
    label = "paste (not loaded through Hark's skill list)"

    def __init__(self, timeout_s: float):
        self.timeout_s, self.text = timeout_s, ""

    def install(self, text: str, version: str) -> Install:
        self.text = text
        return Install(version, True, "pasted at the start of every conversation")

    def wrap(self, endpoint):
        return PasteEndpoint(endpoint, self.text, self.timeout_s)

    def loaded(self, conv) -> tuple[bool | None, str]:
        return None, "pasted, not loaded"

    def remove(self) -> str:
        return "nothing to remove (pasted)"


class PersonalSkillDelivery:
    label = "personal skill on the test account"

    def __init__(self, make_endpoint, account: str, name: str, description: str, timeout_s: float, out=print):
        self.make_endpoint, self.account, self.name, self.description = make_endpoint, account, name, description
        self.timeout_s, self.out = timeout_s, out

    def _instruction(self, text: str) -> str:
        return (f"Housekeeping for a test, not a task. Manage my own custom specialties with the skill tool, exactly as follows, without asking me anything:\n"
                f"1. If I already have a custom skill named `{self.name}`, delete it.\n"
                f"2. Create a custom skill named `{self.name}` with this description, exactly: {self.description}\n"
                f"   and as content exactly the text between the two markers below, character for character, without the markers.\n"
                f"Then reply with just: saved\n{START}\n{text}\n{END}")

    def install(self, text: str, version: str) -> Install:
        ep = self.make_endpoint()
        rec = Install(version, False, "")
        try:
            rec.conversation_id = ep.open(self.account, "", f"[refine setup] {self.name} {version}", {})
            say = self._instruction(text)
            for _ in range(SETUP_TURNS):
                t = ep.turn(say, self.timeout_s)
                rec.turns.append({"sent": say[:200], "reply": t.reply[:400], "calls": [c.get("name") for c in t.calls]})
                creates = [c for c in skill_calls(t.calls, "create") if c.get("name") == self.name]
                if not creates:
                    say = f"Yes, go ahead: delete `{self.name}` if it exists and create it exactly as given above. No questions."
                    continue
                if not same_text(creates[-1].get("content", ""), text):
                    say = f"The content you saved is not exactly what I gave you. Delete `{self.name}` and create it again with the content exactly as given above, character for character."
                    continue
                return self._confirm(ep, text, rec)
            rec.detail = "Hark did not create the skill with the exact text after %d messages" % SETUP_TURNS
            return rec
        finally:
            ep.close()

    def _confirm(self, ep, text: str, rec: Install) -> Install:
        lines = [l for l in text.strip().splitlines() if l.strip()]
        t = ep.turn(f"Load my custom skill `{self.name}` and reply with only its first non-empty line and its last non-empty line, verbatim, one per line.", self.timeout_s)
        got = [l.strip() for l in t.reply.strip().splitlines() if l.strip()]
        rec.turns.append({"sent": "confirm", "reply": t.reply[:400], "calls": [c.get("name") for c in t.calls]})
        loads = [c for c in skill_calls(t.calls, "load") if self.name in (c.get("name"), c.get("id"))]
        if not loads:
            rec.detail = "created, but Hark did not load it back when asked"
        elif not got or lines[0].strip() not in got[0] or lines[-1].strip() not in got[-1]:
            rec.detail = f"created, but the stored first/last lines did not come back verbatim: {t.reply[:160]!r}"
        else:
            rec.ok, rec.detail = True, "created with the exact text; first and last lines read back"
        return rec

    def wrap(self, endpoint):
        return endpoint

    def loaded(self, conv) -> tuple[bool | None, str]:
        """Did Hark load the candidate (by its bare name) in this conversation? A `_system/` load is the shared version."""
        loads = [i for t in conv.turns for i in skill_calls(t.calls, "load")]
        mine = [i for i in loads if self.name in (i.get("name"), i.get("id"))]
        shared = [i for i in loads if f"_system/{self.name}" in (i.get("name"), i.get("id"))]
        if mine:
            return True, f"loaded `{self.name}`"
        if shared:
            return False, f"loaded the shared `_system/{self.name}`, not the candidate"
        return False, "never loaded the skill" + (f" (loaded: {', '.join(str(i.get('name') or i.get('id')) for i in loads)})" if loads else "")

    def remove(self) -> str:
        ep = self.make_endpoint()
        try:
            ep.open(self.account, "", f"[refine cleanup] {self.name}", {})
            t = ep.turn(f"Housekeeping, not a task: delete my custom skill `{self.name}`. Reply with just: deleted", self.timeout_s)
            deleted = [c for c in skill_calls(t.calls, "delete") if self.name in (c.get("name"), c.get("id"))]
            return f"deleted `{self.name}`" if deleted else f"asked to delete `{self.name}`; no delete call seen: {t.reply[:120]!r}"
        finally:
            ep.close()


def make(kind: str, make_endpoint, account: str, name: str, description: str, timeout_s: float, out=print):
    if kind == "personal_skill":
        return PersonalSkillDelivery(make_endpoint, account, name, description, timeout_s, out)
    if kind == "paste":
        return PasteDelivery(timeout_s)
    if kind == "fake":
        return FakeDelivery()
    raise ValueError(f"unknown delivery {kind!r}")
