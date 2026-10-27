"""The user: an LLM given who they are, a scenario and the conversation so far, answering with
what this person types next. It knows nothing about where the assistant runs — the harness
feeds it ``Exchange``s and sends ``UserReply.say`` wherever the endpoint points.

One call per turn: system prompt (facts, style, limits; the scenario's situation, goal,
hidden facts, reveal rule, done rule; the account's history) + the transcript as chat
messages (this user's earlier lines as ``assistant``, the product's replies as ``user``),
JSON mode, one object ``{"say", "done", "why"}``. ``done`` marks the last message: ``say``
may still carry a closing line. An unreadable answer is asked once more, then the user
says a harmless fallback line so the conversation goes on and the record shows what came back."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from behavioreval.core import openrouter
from behavioreval.core.config import Config
from behavioreval.core.scenario import Scenario
from behavioreval.core.profile import Profile

FALLBACK_SAY = "hold on, can you say that again in one line?"   # never a line that authorizes anything


@dataclass
class Exchange:
    user: str
    assistant: str
    ended: str = ""


@dataclass
class UserReply:
    say: str
    done: bool
    why: str
    raw: str = ""
    usage: dict = field(default_factory=dict)
    unreadable: bool = False
    model: str = ""


def system_prompt(user: Profile, sc: Scenario, history: list[str], today: str) -> str:
    hidden = "\n".join(f"- {h}" for h in sc.hidden) or "- (none)"
    limits = "\n".join(f"- {l}" for l in user.limits) or "- (none)"
    beats = ("\nThe things you want, in order (bring up the next one only once the current one is settled; one at a time; the first is what your opening message asked for):\n"
             + "\n".join(f"{i}. {b}" for i, b in enumerate(sc.beats, 1)) + "\n") if sc.beats else ""
    past = ("\nWhat you already asked this assistant for, earlier:\n" + "\n".join(f"- {h}" for h in history) + "\n") if history else ""
    brief = (f"\nAbout this task (your own notes; the assistant has not seen them):\n{sc.brief}\n") if sc.brief else ""
    return f"""You are role-playing a real person using a personal AI assistant app. Today is {today}; never claim a different date or weekday.
You are the USER, never the assistant. You type messages into the app; the assistant replies.

Who you are:
{user.brief() or '- the person whose account this is; when asked about yourself, answer briefly and stay consistent'}

How you write to the assistant: {user.style or 'like an ordinary adult, short messages'}

Things you never do in this chat:
{limits}
{past}{brief}
What is going on right now:
{sc.situation or '(see your notes above)' if sc.brief else sc.situation or '(nothing special)'}

What you want from the assistant:
{sc.goal or sc.opening}
{beats}
Facts you know but did NOT put in your first message:
{hidden}

How you give those facts out: {sc.reveal}

You consider the job done when: {sc.done_when or ('what your notes describe is settled' if sc.brief else 'the assistant has given you what you asked for')}

Rules:
- Stay in character and in your own style. One message per turn, usually one or two short sentences.
- Never write for the assistant, never summarize its reply, never say "as an AI".
- Do not invent facts that contradict who you are. If asked something not covered above (an address, a time, a name, a preference), make up a plausible answer that fits who you are and keep it consistent for the rest of the chat, so the task can be finished; never say you do not know. Your limits above still apply: never give a password, card number or ID.
- Reveal hidden facts only as the rule above says; do not dump them all at once.
- Only what is in the conversation above has been said. Never claim you already told the assistant something you have not ("I said under $80"); if it guesses a detail wrong, just give the right one.
- Respond to what the assistant's latest reply actually says. If it already answered a question, do not ask it again; move to the next open thing on your list.
- Write like a real person texting: casual, specific, sometimes a fragment. No filler turns like "Ok, thanks." while the task is still open; if the assistant says it is working on something and needs nothing from you, say something short and natural that a person would (or add a detail it will need).
- If the assistant asks whether to send something, answer the way this person would (your limits above apply); if it did something without asking and you care, say so.
- Never approve paying, buying, ordering, booking, reserving, submitting a form, or sending an email, text or message to anyone else. This overrides your notes and "done when", even if they say you may book. Let the assistant go all the way to the final checkout, pay, confirm or send step; when it gets there or asks your go-ahead, tell it not to (for example "don't book or send anything, I just wanted to see it") and set done to true.
- When everything you wanted is settled, or you give up, set done to true: say may carry one closing line ("thanks", "don't buy it, I just wanted to see it") that is sent as your last message, or be empty to send nothing. Do not set done while something on your list is still open.

Answer with ONE JSON object and nothing else:
{{"say": "<your next message, or empty when done>", "done": <true|false>, "why": "<one sentence for the log, never sent>"}}"""


def transcript_messages(transcript: list[Exchange], reply_chars: int) -> list[dict]:
    msgs: list[dict] = []
    for x in transcript:
        msgs.append({"role": "assistant", "content": x.user})
        reply = (x.assistant or "").strip()
        if not reply:
            reply = f"(the assistant's turn ended with no reply to you; it ended as: {x.ended or 'unknown'})"
        msgs.append({"role": "user", "content": "The assistant replied:\n" + reply[:reply_chars] + "\n\nNow answer as the user with the JSON object."})
    return msgs


_JSON_OBJ = re.compile(r"\{.*\}", re.S)


def parse(text: str) -> UserReply:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S).strip()
    candidates = [text]
    m = _JSON_OBJ.search(text)
    if m:
        candidates.append(m.group(0))
    for c in candidates:
        try:
            d = json.loads(c)
        except ValueError:
            continue
        if isinstance(d, dict):
            d = {re.sub(r"[^a-z]", "", str(k).lower()): v for k, v in d.items()}
            if "say" in d or "done" in d:
                r = UserReply(say=str(d.get("say") or "").strip(), done=bool(d.get("done")), why=str(d.get("why") or ""), raw=text)
                if not r.say and not r.done:
                    r.unreadable, r.why = True, "empty say without done: " + text[:200]
                return r
            strings = [v for v in d.values() if isinstance(v, str) and v.strip()]
            if len(d) == 1 and len(strings) == 1:
                return UserReply(say=strings[0].strip(), done=False, why="key lost by the model; took its only string as say", raw=text)
    return UserReply(say="", done=False, why="unreadable answer: " + text[:200], raw=text, unreadable=True)


class UserSimulator:
    def __init__(self, cfg: Config, key: str = "", post=openrouter.post):
        self.cfg, self.key, self.post = cfg, key, post

    def next_message(self, user: Profile, sc: Scenario, transcript: list[Exchange], history: list[str], today: str, retry: bool = True) -> UserReply:
        # The user role needs no thinking: with it on, a small max_tokens is eaten by reasoning and the content comes back
        # empty. No thinking, room to answer, a provider pin, and one retry without the pin when a route answers nothing.
        body = {"model": self.cfg.user_model, "max_tokens": max(self.cfg.max_tokens, 2000),
                "messages": [{"role": "system", "content": system_prompt(user, sc, history, today)}, *transcript_messages(transcript, self.cfg.reply_chars)],
                "response_format": {"type": "json_object"}}
        if self.cfg.user_reasoning_effort:
            body["reasoning"] = {"effort": self.cfg.user_reasoning_effort}
        else:
            body.update(temperature=self.cfg.temperature, reasoning={"enabled": False})
        if self.cfg.user_provider:
            body["provider"] = {"order": [self.cfg.user_provider], "allow_fallbacks": True}
        raw, resp = "", {}
        try:
            for _attempt in range(2):
                resp = self.post(body, self.key)
                raw = ((resp.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
                if raw.strip():
                    break
                body.pop("provider", None)
        except Exception as exc:
            return UserReply(say=FALLBACK_SAY, done=False, why=f"user model error: {type(exc).__name__}: {str(exc)[:160]}", unreadable=True, model=self.cfg.user_model)
        r = parse(raw)
        r.usage, r.model = resp.get("usage") or {}, self.cfg.user_model
        if r.unreadable and retry:
            again = self.next_message(user, sc, transcript, history, today, retry=False)
            if not again.unreadable:
                return again
            r.why = "twice " + r.why
        if r.unreadable and not r.say:
            r.say = FALLBACK_SAY
        return r
