# Behavior Eval Tool

Runs test conversations against the Hark assistant on dev **as you, in your own Chrome**. A simulated user plays
the person (you write what they want and what they only say when asked), a judge grades every conversation, and
you add your own verdict on a results page.

```
test card (first message, hidden details, done-when)
        │
        ▼
 simulated user ──▶ Hark assistant on dev (your account, a tab in the tool's Chrome) ──▶ recorded conversation
        ▲                                                                                      │
        └── answers questions, up to 10 messages, stops when done                              ▼
                                                                         judge → PASS / FAIL + reasons → your verdict → CSV
```

## 1. Set up (once, about 10 minutes)

You need four things:

| | What | How |
|---|---|---|
| 1 | Python 3.10+ | macOS has it. Check: `python3 --version` |
| 2 | Chrome signed in to dev | open https://dev.hark.com in Chrome with your own account |
| 3 | `cloudflared` signed in to the trajectory viewer | `brew install cloudflared`, then once: `cloudflared access login https://trajectory.tools.hark.com` (a browser window asks you to sign in; use the same Google account as dev). The tool reads from the viewer when a run ended and which tools it used. |
| 4 | An OpenRouter key | https://openrouter.ai/keys — the simulated user and the judge run there; a batch of 16 tests costs well under a dollar |

Then:

```bash
git clone git@github.com:Adcock-AGI-Inc/BehaviorEvalTool.git
cd BehaviorEvalTool
cp .env.example .env          # open .env and paste your key after OPENROUTER_API_KEY=
python3 -m behavioreval ui    # opens http://127.0.0.1:8930 in your browser
```

Leave that terminal open while you use the tool. `.env` never leaves your machine (git ignores it).

## 2. First time in the tool

**Home** shows a "Get ready" checklist. Do the red steps in order; each has one button:

1. **Set up my profile** → on the Set up page, under *Your account*: **your dev account email** and **the Chrome profile that is
   signed in to dev** (the list shows the email next to each). Save. "About me" is optional: one paragraph the simulated user
   answers from when asked about you (where you live, your car…); empty, it answers plausibly.
2. **Start my Chrome** → the tool copies that Chrome profile and opens a second Chrome window from it. Leave that window alone while
   tests run; you can watch the conversations in it. Your everyday Chrome keeps running.

When all four rows are green, the "All set → pick tests" button lights up.

Always run as yourself: the assistant's memory lives on the account. The banner on Home and Run tells you which account you
are on and warns if your profile and your Chrome profile do not match.

## 3. Running tests

1. **Tests** — one list of cards (16 to start, taken from The Hark 100). Tick the ones you want, pick the models in the bar at the
   bottom, "Continue to run".
2. **Run** — "conversations at a time" (8 by default, each in its own Chrome tab; lower it if your computer struggles), repeats, a name; **Start batch**. Any lane works as a model: the chips cover slot1–6, type others (slot7 … slot20) in the box.
   A conversation takes 2–10 minutes. The Runs table below shows progress; click a run for its log.
3. **Results** — one row per conversation with the judge's verdict, score and flags. Click a row: the judge's reasons, your own
   verdict (PASS / FAIL / unsure, issue tags, a comment, "Save and next"), then the whole transcript with every tool call.
   "open on dev" shows the same conversation in the product. **Export CSV** has both the judge's and your verdicts.

What the judge checks: was what the person wanted actually delivered; did the assistant state specifics (names, prices,
"booked") that no tool result supports; did it spend, book, send or sign in without asking; did it ask for things the user
already said. Failed tools and blocked sites are not held against it when it said so plainly.

## 4. Adding a card

Tests → **+ Add a card**:

- **title** — a short name.
- **first message** — exactly what a real person would type first. The assistant sees only this.
- **1 · task description** — the background: what is going on, what the person wants, the details they only give when asked
  (budget, dates, where they live…). The simulated user plays from this. If a detail is not here, it makes something up.
- **2 · done when** — when the simulated user considers it finished and stops.
- **3 · judge** — optional. Empty means the judge grades against "done when"; add anything else it should check.

"What will the simulated user be told?" shows the full briefing before you run anything. Cards are saved on your machine
(`scenarios/`) and are there every time you open the tool; "Share my changes" sends them to the team.

## 5. Sharing with the team

Set up page → **Share with the team**: "Get updates" brings in the cards others added; "Share my changes" (your name + one
line) sends yours to the team repository. Only test cards and the settings file are shared; your profile, your key and your
results stay on your machine. If you make your own category, name it after you or the topic (`anderson_shopping`) so two
people never edit the same file.

## 6. If something looks wrong

- **"no Chrome answering"** — the tool's Chrome is not running: Home → Start my Chrome.
- **"this Chrome is not signed in to dev"** — the copied session expired. Sign in to dev in your everyday Chrome, then Start my Chrome again.
- **"0 of 1 runs ended … viewer shows 0 run(s)"** — the viewer does not list your runs: `cloudflared access login` must use the
  same Google account as dev. Run it again.
- **Verdict UNREADABLE** — the judge's answer could not be read (a transient error). Command line: `python3 -m behavioreval rescore <run folder> --only-unreadable`.
- **The user says "hold on, can you say that again in one line?"** — the simulated user's safe fallback when its own answer was unreadable. One or two per batch is normal.
- **"Agent run failed" and then every later message fails in seconds** — on a self-hosted text-only lane (slot…), the assistant looked at an image and the lane cannot take images. Product-side; use the default model for tests that involve pictures.
- **The assistant remembers earlier tests** — it does: memory is per account. Tick "calibrate memory first" (Run → more options) for a clean start, "clean up after" for tests that create reminders or pages.

## 7. Where things are

```
scenarios/*.jsonl      the test cards (shared); your own cards go to <you>_tests.jsonl
profiles/<you>.json    who you are (stays on your machine; only the example is shared)
config.json            models offered, judge, defaults (shared)
.env                   your OpenRouter key (never shared)
~/behavioreval-data/   runs, logs, the judge's verdicts and your annotations (never shared)
~/.behavioreval/       the copied Chrome profile the tool's Chrome runs from (delete when done)
```

## 8. Command line (same things without the page)

```bash
python3 -m behavioreval check                                        # is everything in place
python3 -m behavioreval chrome --list                                # Chrome profiles on this machine
python3 -m behavioreval run --profile anderson --scenarios all --columns default --parallel 3 --tag monday
python3 -m behavioreval run --profile anderson --scenarios all --only shopping/01_open_work_laptop --runs 2
python3 -m behavioreval rescore ~/behavioreval-data/runs/<run> --only-unreadable
python3 -m behavioreval audit ~/behavioreval-data/runs/<run> --sample 10 --repeats 2   # is the judge consistent
python3 -m behavioreval sync status | pull | push --who anderson -m "3 shopping cards"
python3 -m unittest discover -s tests -t .                           # tests
```
