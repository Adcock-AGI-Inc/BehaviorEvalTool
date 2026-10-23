"""The outer layer: every knob of a run in one place, read in this order —
``config.json`` (repo root) → ``BEHAVIOREVAL_<KEY>`` environment variables → command-line overrides.
Nothing else in the package reads a file or the environment for a parameter."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent      # the repository root (config.json, profiles/, scenarios/ live here)
CONFIG_FILE = ROOT / "config.json"


@dataclass
class Config:
    user_model: str = "deepseek/deepseek-v4.1-flash-1017"
    user_provider: str = "deepinfra/fp8"   # OpenRouter provider pin for the user model ("" = any)
    temperature: float = 0.7
    max_tokens: int = 300
    max_turns: int = 10                # at most this many user messages per conversation (the user usually stops earlier)
    turn_timeout_s: float = 600.0      # how long one assistant turn may take
    reply_chars: int = 2000            # of the assistant's reply the user model gets to read
    max_consecutive_errors: int = 2    # assistant turns that fail with no reply in a row before the conversation stops
    lane: str = ""                     # hark-slotN for a self-hosted lane; "" = the product's default model
    model: str = ""                    # explicit per-message model override (e.g. claude-opus-5-5); empty = derived from lane
    model_provider: str = ""           # provider for the override; empty = none (lane sugar sets hark-slotN)
    endpoint: str = "chrome"           # chrome (your own account in your own Chrome) | evalsystem (test accounts, needs mvp) | openai_chat | fake
    endpoint_url: str = ""             # openai_chat: the /v1 base
    env: str = "dev"
    web_url: str = ""                  # the app's URL; empty = derived from env (dev → https://dev.hark.com)
    timezone: str = "America/Los_Angeles"
    viewer_url: str = "https://trajectory.tools.hark.com"   # the trajectory viewer: run end + tool calls (chrome endpoint)
    chrome_port: int = 9343            # the DevTools port of the tool's own Chrome
    chrome_dir: Path = Path("~/Library/Application Support/Google/Chrome")   # where Chrome keeps its profiles
    chrome_profile_dir: Path = Path("~/.behavioreval/chrome")               # the copied profile the tool's Chrome runs from
    judge_model: str = "deepseek/deepseek-v4.1-flash-1017"
    judge_provider: str = "deepinfra/fp8"
    judge_samples: int = 1
    judge_reasoning_tokens: int = 3000   # thinking budget for the judge (OpenRouter `reasoning.max_tokens`); 0 = disabled
    seed_profile: bool = True          # write the user's profile into the account before the opening
    history_in_briefing: bool = True   # tell the user model what this account asked for before
    data_root: Path = Path("~/behavioreval-data")
    mvp_dir: Path = Path("~/Documents/mvp")
    accounts_file: Path = Path("accounts.tsv")     # email<TAB>password per line (never committed)
    profiles_dir: Path = Path("profiles")          # one <id>.json per simulated person, with the accounts bound to them
    profile: str = "default"                       # which profile runs when none is chosen
    columns: list = field(default_factory=lambda: [{"id": "default", "label": "Product default model"}])   # the assistant columns the UI offers
    scenarios_dir: Path = Path("scenarios")
    session_cache: Path = Path("~/.cache/hark-eval-clear")
    secrets_file: Path = Path(".env")              # KEY=VALUE lines; only SECRET_KEYS are read from it
    ui_port: int = 8930
    source: dict = field(default_factory=dict)   # where each value came from, for `behavioreval config`

    @property
    def eval_dir(self) -> Path:
        return self.mvp_dir / "services" / "eval_system"


_PATHS = {"data_root", "mvp_dir", "accounts_file", "profiles_dir", "scenarios_dir", "session_cache", "secrets_file", "chrome_dir", "chrome_profile_dir"}
SECRET_KEYS = ("CF_ACCESS_CLIENT_ID", "CF_ACCESS_CLIENT_SECRET", "GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET", "OPENROUTER_API_KEY")
_BOOLS = {"seed_profile", "history_in_briefing"}


def _coerce(key: str, value):
    if key in _PATHS:
        p = Path(str(value)).expanduser()
        return p if p.is_absolute() else (ROOT / p)
    if key in _BOOLS:
        return value if isinstance(value, bool) else str(value).strip().lower() in ("1", "true", "yes", "on")
    kind = {f.name: f.type for f in fields(Config)}[key]
    if kind == "list":
        if isinstance(value, str):
            value = [{"id": v.strip(), "label": v.strip()} for v in value.split(",") if v.strip()]
        return [x if isinstance(x, dict) else {"id": str(x), "label": str(x)} for x in (value or [])]
    if kind == "int":
        return int(value)
    if kind == "float":
        return float(value)
    return "" if value is None else str(value)


def load(path: Path = CONFIG_FILE, environ=None, overrides: dict | None = None) -> Config:
    """File, then BEHAVIOREVAL_<KEY> env vars, then overrides whose value is not None."""
    environ = os.environ if environ is None else environ
    cfg = Config()
    known = {f.name for f in fields(Config)} - {"source"}
    for name in known:
        setattr(cfg, name, _coerce(name, getattr(cfg, name)))
        cfg.source[name] = "default"
    if path and path.exists():
        for k, v in json.loads(path.read_text(encoding="utf-8")).items():
            if k not in known:
                raise ValueError(f"{path.name}: unknown key {k!r}")
            setattr(cfg, k, _coerce(k, v))
            cfg.source[k] = path.name
    for k in known:
        v = environ.get("BEHAVIOREVAL_" + k.upper())
        if v is not None and v != "":
            setattr(cfg, k, _coerce(k, v))
            cfg.source[k] = "env"
    for k, v in (overrides or {}).items():
        if k in known and v is not None:
            setattr(cfg, k, _coerce(k, v))
            cfg.source[k] = "cli"
    if cfg.lane in ("claude", "default", "none"):
        cfg.lane = ""
    cfg.lane = cfg.lane.removeprefix("hark-")
    if cfg.max_turns < 1:
        raise ValueError("max_turns must be at least 1")
    return cfg


LANE_MODEL = "agent-llm"   # the per-message override the product accepts for a self-hosted lane


def column(cfg: Config) -> tuple[str, str, str]:
    """(label, model, provider) the assistant runs as: an explicit model (claude-opus-5-5,
    no provider), else the lane (agent-llm on hark-slotN), else the product's default."""
    if cfg.model:
        return cfg.model, cfg.model, cfg.model_provider
    if cfg.lane:
        return f"slot{cfg.lane}" if cfg.lane.isdigit() else cfg.lane, LANE_MODEL, f"hark-{cfg.lane}"
    return "default", "", ""


def with_column(cfg: Config, spec: str) -> Config:
    """A copy of cfg set to one column spec: ``slot20``, ``hark-slot20``, ``claude-opus-5-5``, ``default``."""
    from dataclasses import replace
    spec = spec.strip()
    if spec in ("", "default", "claude"):
        return replace(cfg, lane="", model="", model_provider="")
    if spec.removeprefix("hark-").removeprefix("slot").isdigit():
        return replace(cfg, lane=spec.removeprefix("hark-"), model="", model_provider="")
    return replace(cfg, lane="", model=spec, model_provider="")


def secrets(cfg: Config, environ=None) -> dict[str, str]:
    """SECRET_KEYS from the environment, else from ``cfg.secrets_file``. Never printed."""
    environ = os.environ if environ is None else environ
    out = {k: environ[k] for k in SECRET_KEYS if environ.get(k)}
    if cfg.secrets_file.exists():
        for line in cfg.secrets_file.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                v = v.split("#", 1)[0].strip().strip("'\"")   # a trailing "# comment" is not part of the value
                if k.strip() in SECRET_KEYS and k.strip() not in out and v:
                    out[k.strip()] = v
    return out


def load_accounts(path: Path) -> dict[str, str]:
    """email → password from the accounts file; empty when the file is missing."""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("#") and "\t" in line:
            email, pw = line.split("\t", 1)
            if email.strip() and pw.strip():
                out[email.strip()] = pw.strip()
    return out
