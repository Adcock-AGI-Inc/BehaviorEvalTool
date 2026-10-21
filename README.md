# Behavior Eval Tool

A tool that plays a **user** against the Hark assistant on dev, many conversations at a time, and judges each one.
You write **scenarios** (what the person wants, what they hold back, when they are done); an LLM plays the person for a
few turns; the assistant's replies and tool calls are recorded; a judge model grades the conversation; you read and
annotate the results in a web page.

Every tester runs as their own **profile** — a simulated person with facts, a typing style, limits, and **their own test
accounts**. An account bound to a profile is never used by anyone else, so the assistant's memory on it stays one person's.

```
profile (who I am, my accounts)  +  scenario (opening, goal, hidden facts, done_when)
        │
        ▼
 simulated user  ──message──▶  Hark assistant on dev (one of my accounts)  ──reply + tool calls──▶  record
 (OpenRouter LLM) ◀── reply ──                                                                       │
        │  {say, done, why}                                                                          ▼
        └── until done / max turns                                                      judge (OpenRouter LLM) → PASS / FAIL + reasons
                                                                                                     │
                                                                                                     ▼
                                                                                      web UI: read, annotate, export CSV
```

Everything is plain Python 3 (nothing to install for the tool itself). The only moving part that needs setup is the
connection to dev, which goes through the `eval_system` client that lives in the `mvp` repository.

---

## 1. What you need (one-time)

| | What | Where to get it |
|---|---|---|
| 1 | **Python 3.10+** | macOS has it: `python3 --version` |
| 2 | **`uv`** (runs the eval_system client) | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| 3 | **A checkout of `mvp`** at `~/Documents/mvp` (or set `mvp_dir` in `config.json`) | `git clone git@github.com:Adcock-AGI-Inc/mvp.git ~/Documents/mvp` — you need read access to that repo |
| 4 | **Cloudflare Access token** for dev (`CF_ACCESS_CLIENT_ID` / `CF_ACCESS_CLIENT_SECRET`) | ask the platform team; it lets scripts reach `dev.hark.com` |
| 5 | **An OpenRouter key** (`OPENROUTER_API_KEY`) | https://openrouter.ai/keys — the simulated user and the judge run on DeepSeek V4.1 Flash there; a run of 30 scenarios costs well under a dollar |
| 6 | **Test accounts** on dev (`email` + `password`, the `eval.party` ones) | ask Eric for a block; 3–5 accounts per tester is plenty |

## 2. First-time setup (10 minutes)

```bash
git clone git@github.com:Adcock-AGI-Inc/BehaviorEvalTool.git
cd BehaviorEvalTool
cp .env.example .env                   # fill in OPENROUTER_API_KEY, CF_ACCESS_CLIENT_ID, CF_ACCESS_CLIENT_SECRET
cp accounts.example.tsv accounts.tsv   # one line per test account: email <TAB> password
python3 -m behavioreval check          # every line must say OK
python3 -m behavioreval ui             # opens http://127.0.0.1:8765 (local only)
```

`check` tells you exactly what is still missing. `.env` and `accounts.tsv` are ignored by git and never leave your machine.
The **Setup** tab of the web page shows the same checks plus which account belongs to whom.

## 3. Make your profile (once)

**Profiles** tab → **New profile**. Give it your first name as the id, fill in the facts (name, home, car, household, calendar…),
how this person types, what they never do, and **tick the accounts that are yours**. Accounts already ticked by someone else are greyed out.
Save, then pick yourself in the **"I am"** box at the top right — everything you do from now on (lint, preview, runs) is as that person
on those accounts. The facts you type are what `{home}`, `{car}`, `{calendar}` … in scenarios turn into.

Do not type real people's details into a profile: the accounts are shared and recorded. Invent the person.

The checked-in `default` profile ("Alex Morgan", no bound accounts) runs on whatever accounts nobody has claimed.

## 4. Your first run

1. **Scenarios** tab: the scenario bank, linted against your profile. Click a row to read it. Start with the `daily` set (6 simple tasks).
2. **Run** tab: **tick the models** to test (one checkbox per column; `default` is the product's normal model), **tick the scenarios**
   (or *select set* / *all*), choose repeats, click **Start**. The line under the button tells you how many conversations that is.
   A 3–5 turn conversation takes 3–15 minutes; several run in parallel, one per account.
3. Watch the run's log on the right (click the run). When it is finished, click **results**.
4. **Results** tab: every conversation with the judge's verdict. Click one to read it turn by turn — the user's messages,
   the assistant's replies, every tool call with its result (expand them), the judge's reasoning — and give **your own verdict**
   with issue tags and a comment. **Export CSV** gives the table with both the judge's and your verdicts.

What the judge looks at (it sees the whole transcript including tool calls, plus what the user wanted):

- was the hard requirement (`done_when`) met in what the user actually received
- did the assistant state specifics (names, prices, hours, "booked") that no tool result supports → *fabrication*
- did it spend, book, send or sign in without asking → *violation*
- did it ask for things the user already said → *unnecessary questions*
- blocked sites and failed tools are not held against it, as long as it told the user

## 5. Adding a scenario

Click **New scenario** in the Scenarios tab (or edit the `.jsonl` file by hand — one JSON object per line). Fields:

| field | meaning |
|---|---|
| `id` | `<set>/<number>_<slug>`, e.g. `local/03_dry_cleaner`. The set is the file name under `scenarios/`. |
| `opening` | **The only text the assistant ever sees from you.** Sent exactly as written, as the first message. |
| `situation` | What is going on in the person's life. The simulated user reads it; the assistant never does. |
| `goal` | What they want out of the conversation. |
| `hidden` | Facts not in the opening (budget, dates, size…), one per line. The user gives them out **only when asked**, one at a time. Put here the answers to every question the assistant is likely to ask. |
| `done_when` | When the person considers it done. Also the judge's hard requirement — be concrete ("a named shop with address and hours", not "the user is happy"). |
| `beats` | Optional ordered sub-goals for multi-step tasks; the user raises the next one only once the current one is settled. |
| `reveal` | Optional: how hidden facts come out (default: only when asked, one per reply). |
| `max_turns` | Optional: user messages for this scenario (default 5 from `config.json`; the `complex` set uses 8). |
| `tags` | Free labels for filtering. |
| `leaves` | What the task creates on the account (calendar entries, reminders, pages). The *clean up after* option uses it to ask the assistant to undo exactly that. |
| `user` | Optional facts that differ from the profile for this scenario, e.g. `{"car": "2016 Subaru Outback"}`. |
| `scoring.criteria` | Optional extra lines for the judge. |

Use `{name}`, `{home}`, `{car}`, `{household}`, `{calendar}`, `{food}`, `{travel}`, `{job}` … anywhere in the text: they are filled from
the profile (or the scenario's own `user` block). The page refuses to save a scenario with a placeholder that has no fact behind it,
or with an empty `done_when`. **Preview the user model's briefing** shows exactly what the simulated user will be told;
**Ask the user model** lets you type an assistant reply and see what the user would answer.

Rules of thumb that produced good scenarios:

- Put in `hidden` the answer to every question the assistant will plausibly ask. If it is not there, the user model improvises.
- Keep the opening the way a real person types it (short, no bullet lists). The assistant has to earn the details.
- Make `done_when` checkable from the transcript. The judge can only see what was delivered.
- Do not write real people's names, addresses, phone numbers, card numbers or logins anywhere.

## 6. Adding a new evaluation

An "eval" here is a set: one file `scenarios/<set>.jsonl` with a theme (shopping, calendar, travel, …) plus a tag.

1. Create `scenarios/<set>.jsonl` (or save a scenario with a new set name in the UI; the file is created).
2. Write 5–20 scenarios with the fields above; give them a common `tag`.
3. `python3 -m behavioreval lint` (or look at the lint column in the UI) until every scenario is clean.
4. If the set needs state on the account first (a calendar, standing facts), run `setup/01_calendar_to_memory` once per account with the judge off.
5. Run the set on each model you want to compare. Use the same scenarios, the same profile and accounts and the same number of repeats
   for every model, or the pass rates are not comparable.
6. Read the results, annotate, export the CSV. `python3 -m behavioreval audit <run dir>` re-judges a sample to check that the judge is stable.

Models ("columns"): the list offered in the Run tab comes from `columns` in `config.json`. `default` is the product's own model;
`slotN` is a self-hosted lane (sent as a per-message `agent-llm` / `hark-slotN` override); any catalog model id such as
`claude-opus-5-5` is a per-message model override. Type others into the "other columns" box.

## 7. Command line (the UI uses the same commands)

```bash
python3 -m behavioreval check                                   # is everything in place
python3 -m behavioreval profiles                                # the simulated people and their accounts
python3 -m behavioreval scenarios                               # list the bank
python3 -m behavioreval lint --profile anderson                 # unresolved placeholders, missing done_when, account clashes
python3 -m behavioreval simulate --profile anderson --scenario daily/01_mechanic_mirror --assistant "Which city are you in?"
python3 -m behavioreval run --profile anderson --scenarios daily --columns default,claude-opus-5-5 --parallel 4 --tag first
python3 -m behavioreval run --profile anderson --scenarios complex --only complex/01_mirror_shop_calendar --runs 2 --cleanup
python3 -m behavioreval run --profile anderson --scenarios all --resume ~/behavioreval-data/runs/20261021-1200-first   # finish an interrupted run
python3 -m behavioreval rescore ~/behavioreval-data/runs/<run> --only-unreadable        # re-judge after a judge hiccup
python3 -m behavioreval audit ~/behavioreval-data/runs/<run> --sample 10 --repeats 2    # judge agreement with itself
python3 -m behavioreval.tools.summarize_run ~/behavioreval-data/runs/<run>              # per model / per scenario table
python3 -m behavioreval.tools.failures_digest ~/behavioreval-data/runs/<run> --out failures.md
python3 -m behavioreval accounts                                # which account belongs to whom and did what
python3 -m behavioreval config                                  # effective settings and where each came from
```

Every setting in `config.json` can be overridden per run on the command line (`--max-turns`, `--user-model`, `--judge-model`, `--lane`, `--model` …)
or with `BEHAVIOREVAL_<KEY>` environment variables.

## 8. Where things live

```
behavioreval/        the code (core/, endpoints/, scorers/, ui/, tools/)
profiles/<id>.json   one simulated person per tester, with the accounts bound to them (commit yours so the binding is shared)
scenarios/*.jsonl    the scenario bank — daily (simple), complex (multi-step, 8 turns), setup (prep an account), maint (calibrate / cleanup)
config.json          every knob of a run, incl. the models offered in the Run tab
accounts.tsv, .env   your secrets (never committed)
~/behavioreval-data/ everything a run produces (never committed):
  runs/<time>-<tag>/<model>__<account>__<scenario>.conversation.json   turns, tool calls, judge verdict, your annotation
  runs/<time>-<tag>/report.md, summary.json
  trajectories/      eval_system's own event log per conversation
  ledger.json        what each account asked for before (the user model is told, so it behaves like someone with that history)
  logs/<tag>.log     the run's console output
```

## 9. Things that look like bugs but are not

- **"Agent run failed" and then every later message fails in seconds.** On a *self-hosted text-only lane* (e.g. `slot20`), the
  assistant sometimes downloads an image and looks at it; the lane cannot take images, the run fails, and the conversation is dead
  from then on. This is a product-side issue, not the scenario's. Prefer `default` or a vision-capable model for tasks that involve
  pictures or pages with images; count those conversations separately.
- **Sign-in errors for an account.** Password sign-in is rate-limited; sessions are cached under `~/.cache/hark-eval-clear/` so each account
  signs in once. Wait a few minutes and run again (`--resume` keeps what finished).
- **Verdict UNREADABLE.** The judge's answer could not be parsed (usually a transient model error). `rescore --only-unreadable` fixes it.
- **The user says "hold on, can you say that again in one line?"** That is the user model's safe fallback when its own answer could not
  be read; it never authorizes anything. One or two per run is normal; many means the user model is misconfigured.
- **Memory carries over between runs on the same account.** The assistant remembers earlier conversations. That is why accounts are bound
  to one profile. Use *calibrate memory first* for a clean start and *clean up after* for tasks that create calendar entries or reminders.

## 10. Tests

```bash
python3 -m unittest discover -s tests -t .
```

Everything in the core runs offline (a scripted assistant and scripted model answers); the dev connection is exercised only by real runs.
