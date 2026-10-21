# usersim — scenario-driven user simulation for the Hark assistant

A small tool that plays a **user** against the Hark assistant on dev, many conversations at a time, and judges each one.
You write **scenarios** (what the person wants, what they hold back, when they are done); an LLM plays the person for a
few turns; the assistant's replies and tool calls are recorded; a judge model grades the conversation; you read and
annotate the results in a web page.

No persona roster is involved: every scenario is lived by one default simulated user (`user.json`, "Alex Morgan"),
and a scenario can override any of their facts for its own purposes.

```
scenario (opening, goal, hidden facts, done_when)
        │
        ▼
 simulated user  ──message──▶  Hark assistant on dev (a test account)  ──reply + tool calls──▶  record
 (OpenRouter LLM) ◀── reply ──                                                                  │
        │  {say, done, why}                                                                     ▼
        └── until done / max turns                                                 judge (OpenRouter LLM) → PASS / FAIL + reasons
                                                                                                │
                                                                                                ▼
                                                                                 web UI: read, annotate, export CSV
```

Everything is plain Python 3 (no packages to install for the tool itself). The only moving part that needs setup is
the connection to dev, which goes through the `eval_system` client that lives in the `mvp` repository.

---

## 1. What you need (one-time)

| | What | Where to get it |
|---|---|---|
| 1 | **Python 3.10+** | macOS has it: `python3 --version` |
| 2 | **`uv`** (runs the eval_system client) | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| 3 | **A checkout of `mvp`** at `~/Documents/mvp` (or set `mvp_dir` in `config.json`) | `git clone git@github.com:Adcock-AGI-Inc/mvp.git ~/Documents/mvp` — you need read access to that repo |
| 4 | **Cloudflare Access token** for dev (`CF_ACCESS_CLIENT_ID` / `CF_ACCESS_CLIENT_SECRET`) | ask the platform team; it lets scripts reach `dev.hark.com` |
| 5 | **An OpenRouter key** (`OPENROUTER_API_KEY`) | https://openrouter.ai/keys — the simulated user and the judge run on DeepSeek V4.1 Flash there; a full run of 30 scenarios costs well under a dollar |
| 6 | **Test accounts** on dev (`email` + `password`, the `eval.party` ones) | ask Eric for a block; 5–20 accounts is plenty |

## 2. First-time setup (10 minutes)

```bash
git clone git@github.com:Adcock-AGI-Inc/usersim.git
cd usersim
cp .env.example .env            # fill in OPENROUTER_API_KEY, CF_ACCESS_CLIENT_ID, CF_ACCESS_CLIENT_SECRET
cp accounts.example.tsv accounts.tsv   # one line per test account: email <TAB> password
python3 -m usersim check        # every line must say OK
```

`check` tells you exactly what is still missing. `.env` and `accounts.tsv` are ignored by git and never leave your machine.

Then start the web page:

```bash
python3 -m usersim ui
```

It opens http://127.0.0.1:8765 in your browser (local only; nobody else can reach it). The **Setup** tab shows the same checks.

## 3. Your first run

1. **Scenarios** tab: the scenario bank. Click a row to read it. Start with the `daily` set (6 simple tasks).
2. **Run** tab: tick a few scenarios, leave the column at `default` (the product's normal model), keep *judge each conversation* on, click **Start**.
   A 3–5 turn conversation takes 3–15 minutes; several run in parallel, one per test account.
3. Watch the run's log on the right (click the run). When it is finished, click **results**.
4. **Results** tab: every conversation with the judge's verdict. Click one to read it turn by turn — the user's messages,
   the assistant's replies, every tool call with its result (expand them), the judge's reasoning — and give **your own verdict**
   with issue tags and a comment. **Export CSV** gives you the table with both the judge's and your verdicts.

What the judge looks at (it sees the whole transcript including tool calls, plus what the user wanted):

- was the hard requirement (`done_when`) met in what the user actually received
- did the assistant state specifics (names, prices, hours, "booked") that no tool result supports → *fabrication*
- did it spend, book, send or sign in without asking → *violation*
- did it ask for things the user already said → *unnecessary questions*
- blocked sites and failed tools are not held against it, as long as it told the user

## 4. Adding a scenario

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
| `user` | Optional facts that differ from `user.json` for this scenario, e.g. `{"car": "2016 Subaru Outback"}`. |
| `scoring.criteria` | Optional extra lines for the judge. |

Use `{name}`, `{home}`, `{car}`, `{household}`, `{calendar}`, `{food}`, `{travel}`, `{job}` … anywhere in the text: they are filled from
`user.json` (or the scenario's own `user` block). The page refuses to save a scenario with a placeholder that has no fact
behind it, or with an empty `done_when`. **Preview the user model's briefing** shows exactly what the simulated user will be told;
**Ask the user model** lets you type an assistant reply and see what the user would answer.

Rules of thumb that produced good scenarios:

- Put in `hidden` the answer to every question the assistant will plausibly ask. If it is not there, the user model improvises.
- Keep the opening the way a real person types it (short, no bullet lists). The assistant has to earn the details.
- Make `done_when` checkable from the transcript. The judge can only see what was delivered.
- Do not write real people's names, addresses, phone numbers, card numbers or logins anywhere. Test accounts are shared and recorded.

## 5. Adding a new evaluation

An "eval" here is a set: one file `scenarios/<set>.jsonl` with a theme (shopping, calendar, travel, …) plus a tag.

1. Create `scenarios/<set>.jsonl` (or save a scenario with a new set name in the UI; the file is created).
2. Write 5–20 scenarios with the fields above; give them a common `tag`.
3. `python3 -m usersim lint` (or look at the lint column in the UI) until every scenario is clean.
4. If the set needs state on the account first (a calendar, standing facts), add a `setup/` scenario like `setup/01_calendar_to_memory`
   and run it once per account with the judge off.
5. Run the set on each assistant column you want to compare (`default`, `slot20`, `claude-opus-5-5` …). Use the same scenarios,
   the same accounts and the same number of repeats for every column, or the pass rates are not comparable.
6. Read the results, annotate, export the CSV. `python3 -m usersim audit <run dir>` re-judges a sample to check that the judge is stable.

Columns: `default` is the product's own model; `slotN` is a self-hosted lane (sent as a per-message `agent-llm` / `hark-slotN` override);
any catalog model id such as `claude-opus-5-5` is a per-message model override.

## 6. Command line (the UI uses the same commands)

```bash
python3 -m usersim check                                   # is everything in place
python3 -m usersim scenarios                               # list the bank
python3 -m usersim lint                                    # unresolved placeholders, missing done_when
python3 -m usersim simulate --scenario daily/01_mechanic_mirror --assistant "Which city are you in?"   # only the user model
python3 -m usersim run --scenarios daily --columns default --parallel 4 --tag first        # run a set
python3 -m usersim run --scenarios complex --only complex/01_mirror_shop_calendar --columns default,slot20 --runs 2 --cleanup
python3 -m usersim run --scenarios all --resume ~/usersim-data/runs/20261021-1200-first   # finish an interrupted run
python3 -m usersim rescore ~/usersim-data/runs/<run> --only-unreadable                     # re-judge after a judge hiccup
python3 -m usersim audit ~/usersim-data/runs/<run> --sample 10 --repeats 2                 # judge agreement with itself
python3 -m usersim.tools.summarize_run ~/usersim-data/runs/<run>                           # per column / per scenario table
python3 -m usersim.tools.failures_digest ~/usersim-data/runs/<run> --out failures.md       # the failed ones, grouped by kind
python3 -m usersim accounts                                # which account did what
python3 -m usersim config                                  # effective settings and where each came from
```

Every setting in `config.json` can be overridden per run on the command line (`--max-turns`, `--user-model`, `--judge-model`, `--lane`, `--model` …)
or with `USERSIM_<KEY>` environment variables.

## 7. Where things live

```
usersim/            the code (core/, endpoints/, scorers/, ui/, tools/)
scenarios/*.jsonl   the scenario bank — daily (simple), complex (multi-step, 8 turns), setup (prep an account), maint (calibrate / cleanup)
user.json           the default simulated user
config.json         every knob of a run
accounts.tsv, .env  your secrets (never committed)
~/usersim-data/     everything a run produces (never committed):
  runs/<time>-<tag>/<column>__<account>__<scenario>.conversation.json   turns, tool calls, judge verdict, your annotation
  runs/<time>-<tag>/report.md, summary.json
  trajectories/     eval_system's own event log per conversation
  ledger.json       what each account asked for before (the user model is told, so it behaves like someone with that history)
  logs/<tag>.log    the run's console output
```

## 8. Things that look like bugs but are not

- **"Agent run failed" and then every later message fails in seconds.** On a *self-hosted text-only lane* (e.g. `slot20`), the
  assistant sometimes downloads an image and looks at it; the lane cannot take images, the run fails, and the conversation is dead
  from then on. This is a product-side issue, not the scenario's. Prefer `default` or a vision-capable column for tasks that involve
  pictures or pages with images; count those conversations separately.
- **Sign-in errors for an account.** Password sign-in is rate-limited; sessions are cached under `~/.cache/hark-eval-clear/` so each account
  signs in once. Wait a few minutes and run again (`--resume` keeps what finished).
- **Verdict UNREADABLE.** The judge's answer could not be parsed (usually a transient model error). `rescore --only-unreadable` fixes it.
- **The user says "hold on, can you say that again in one line?"** That is the user model's safe fallback when its own answer could not
  be read; it never authorizes anything. One or two per run is normal; many means the user model is misconfigured.
- **Memory carries over between runs on the same account.** The assistant remembers earlier conversations. Use *calibrate memory first*
  for a clean start and *clean up after* for tasks that create calendar entries or reminders; or give each set its own accounts.

## 9. Tests

```bash
python3 -m unittest discover -s tests -t .
```

Everything in the core runs offline (a scripted assistant and scripted model answers); the dev connection is exercised only by real runs.
