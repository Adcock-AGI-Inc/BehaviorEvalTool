"""The three graders refine mode adds on top of the existing judge (scorers/judge.py, which keeps the hard
requirement, fabrication and permission checks):

rubric    scores each rubric line (general rubric, the skill's criteria, the personality) with a quoted piece
          of evidence, and checks personalization against the persona's hidden facts, so a confident answer
          that invents facts cannot pass.
pairwise  candidate vs the best version so far on the same card, blind (A/B, no version names) and in both
          orders; the two orders must agree, otherwise the result is "equal".
diagnose  why a failed or slow conversation went wrong, as one cause from CAUSES, so only failures the
          skill text can fix ever reach the editor."""
from __future__ import annotations

import json

from behavioreval.refine.meter import BudgetExceeded
from behavioreval.scorers import judge as base

CAUSES = {
    "not_loaded": "Hark never loaded the candidate skill (a delivery problem, not a skill problem)",
    "ignored": "the skill covers this, but Hark did not follow it",
    "silent": "the skill says nothing about this situation",
    "caused_it": "an instruction in the skill led to the failure",
    "product_or_tool": "a blocked site, a failed tool, a product or lane error",
    "simulator": "the simulated user broke character, leaked hidden facts early, or approved something",
    "judge": "the verdict does not match the evidence",
    "none": "on a closer read, nothing went wrong",
}
SKILL_CAUSES = ("ignored", "silent", "caused_it")
SEVERITY_WEIGHT = {"high": 3, "medium": 2, "low": 1}


def bullets(text: str) -> list[str]:
    return [l.strip()[2:].strip() for l in text.splitlines() if l.strip().startswith("- ") and l.strip()[2:].strip()]


def rubric_lines(general: str, criteria: list[str], personality: str) -> list[tuple[str, str]]:
    """[(id, text)]: G1.. from the general rubric's bullets (the whole file if it has none), S1.. the skill's own, V the voice."""
    g = bullets(general) or ([general.strip()] if general.strip() else [])
    out = [(f"G{i}", t) for i, t in enumerate(g, 1)] + [(f"S{i}", t) for i, t in enumerate(criteria, 1)]
    if personality.strip():
        out.append(("V", "Hark sounds the way this personality describes: " + personality.strip()))
    return out


def hidden_facts(sc, persona) -> list[str]:
    return list(sc.hidden) + [f"{k}: {v}" for k, v in (persona.facts or {}).items() if v not in (None, "")]


RUBRIC_PROMPT = """You grade one conversation between a simulated user and Hark, a personal AI assistant, against a rubric. Grade HARK only.
You see the whole conversation with every tool call. You also see facts the user held back, which Hark could only learn by asking or from its own notes.

<goal_of_this_skill>
{goal}
</goal_of_this_skill>

<card>
- what the user wanted: {want}
- done when: {done_when}
- facts the user held back (personalization must come from these, not from guesses): {hidden}
</card>

<rubric>
{rubric}
</rubric>

<conversation>
{conversation}
</conversation>

For every rubric line: met true, false, or null when it does not apply to this conversation. Quote the exact words from the conversation that decide it (under 30 words). For a line not met, give its severity: high (the user's task suffers), medium (noticeably worse), low (polish).
Personalization: list the held-back facts Hark asked for, the ones it used, and any personal fact it stated that is not among them and not said by the user (invented).

Answer with ONE JSON object:
{{"lines": [{{"id": "G1", "met": true, "evidence": "...", "severity": ""}}],
  "personalization": {{"asked": ["..."], "used": ["..."], "invented": ["..."]}},
  "summary": "2-4 sentences"}}"""


PAIRWISE_PROMPT = """Two conversations, A and B, handled the same request from the same user. Decide which one a careful reviewer would rather the user had, judged by the goal and rubric below. Ignore length and style unless the rubric asks about them. Same quality is a tie.
Judge only the ASSISTANT's turns. The user's lines differ between A and B by chance (a simulated user improvises); never prefer a conversation because its user said more, said it sooner, or was easier to help.

<goal_of_this_skill>
{goal}
</goal_of_this_skill>

<what_the_user_wanted>
{want}; done when: {done_when}; facts held back until asked: {hidden}
</what_the_user_wanted>

<rubric>
{rubric}
</rubric>

<conversation_A>
{a}
</conversation_A>

<conversation_B>
{b}
</conversation_B>

Answer with ONE JSON object: {{"winner": "A" | "B" | "tie", "why": "one or two sentences with the decisive evidence"}}"""


DIAGNOSE_PROMPT = """A conversation with Hark, a personal AI assistant, failed or was slow. Find the cause, so the right thing gets fixed. Hark was following the skill text below.

<skill_text>
{skill}
</skill_text>

<what_the_user_wanted>
{want}; done when: {done_when}; facts held back until asked: {hidden}
</what_the_user_wanted>

<grades>
{grades}
</grades>

<speed>
{speed}
</speed>

<conversation>
{conversation}
</conversation>

Pick ONE cause:
{causes}

Rules: "ignored" and "caused_it" must quote the skill line involved. "simulator" and "judge" need the quote that shows it. A failed tool or blocked site is product_or_tool even when Hark could have handled it better, unless the skill told it how.
If the conversation was slow, name the step that took the most time or was repeated (from the per-turn times and tool calls).
fix_hint: one sentence about what the SKILL should say differently, in general terms; never name this user's specifics (names, products, places, dates, amounts).

Answer with ONE JSON object:
{{"cause": "...", "criterion": "the rubric line id it concerns, or empty", "evidence": "quote", "skill_line": "quoted skill line or empty", "slowest_step": "or empty", "fix_hint": "..."}}"""


def _want(sc) -> str:
    return (sc.goal or sc.opening) + (f" (notes: {sc.brief})" if sc.brief else "")


def rubric_text(lines: list[tuple[str, str]]) -> str:
    return "\n".join(f"{i}: {t}" for i, t in lines)


class Graders:
    """Holds the models and the metered post functions; every method returns a plain dict."""

    def __init__(self, key: str, judge_model: str, diagnose_model: str, meter, reasoning_tokens: int = 2000, provider: str = "", ask=None):
        self.key, self.judge_model, self.diagnose_model, self.meter = key, judge_model, diagnose_model, meter
        self.reasoning_tokens, self.provider = reasoning_tokens, provider
        self.ask = ask or base._ask

    def _call(self, role: str, prompt: str, model: str) -> dict:
        try:
            return self.ask(prompt, model, self.key, post=self.meter.post_for(role), reasoning_tokens=self.reasoning_tokens,
                            provider=self.provider if model == self.judge_model else "")
        except BudgetExceeded:
            raise
        except Exception as exc:   # a network hiccup stays visible on the record and is re-gradable
            return {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}

    def rubric(self, conv, sc, persona, lines: list[tuple[str, str]], goal: str) -> dict:
        d = self._call("rubric", RUBRIC_PROMPT.format(goal=goal, want=_want(sc), done_when=sc.done_when or "(see notes)",
                                                       hidden="; ".join(hidden_facts(sc, persona)) or "(none)", rubric=rubric_text(lines),
                                                       conversation=base.transcript(conv)), self.judge_model)
        known = {i for i, _ in lines}
        d["lines"] = [l for l in d.get("lines") or [] if isinstance(l, dict) and l.get("id") in known]
        return d

    def pairwise(self, cand_conv, best_conv, sc, persona, lines, goal: str) -> dict:
        """'better' | 'equal' | 'worse' for the candidate, from two blind orders."""
        def ask(a, b):
            return self._call("pairwise", PAIRWISE_PROMPT.format(goal=goal, want=_want(sc), done_when=sc.done_when or "(see notes)",
                                                                 hidden="; ".join(hidden_facts(sc, persona)) or "(none)", rubric=rubric_text(lines),
                                                                 a=base.transcript(a), b=base.transcript(b)), self.judge_model)
        first, second = ask(cand_conv, best_conv), ask(best_conv, cand_conv)
        w1, w2 = str(first.get("winner", "")).strip().upper(), str(second.get("winner", "")).strip().upper()
        if w1 == "A" and w2 == "B":
            verdict = "better"
        elif w1 == "B" and w2 == "A":
            verdict = "worse"
        else:
            verdict = "equal"
        return {"verdict": verdict, "orders": [{"candidate_is": "A", **first}, {"candidate_is": "B", **second}]}

    def diagnose(self, conv, sc, persona, skill_text: str, grades: dict, speed: str) -> dict:
        d = self._call("diagnose", DIAGNOSE_PROMPT.format(skill=skill_text, want=_want(sc), done_when=sc.done_when or "(see notes)",
                                                           hidden="; ".join(hidden_facts(sc, persona)) or "(none)",
                                                           grades=json.dumps(grades, ensure_ascii=False)[:6000], speed=speed,
                                                           conversation=base.transcript(conv),
                                                           causes="\n".join(f"- {k}: {v}" for k, v in CAUSES.items() if k != "not_loaded")),
                       self.diagnose_model)
        if d.get("cause") not in CAUSES:
            d["cause_raw"], d["cause"] = d.get("cause"), "unreadable"
        return d
