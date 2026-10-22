"""Sharing without knowing git: the repo checkout is the shared bank of scenarios and profiles.
``status`` says what you changed and whether others pushed; ``pull`` brings their additions
in (rebase); ``push`` commits your scenarios/profiles with a one-line message and pushes.
Only ``scenarios/``, ``profiles/`` and ``config.json`` are ever committed from here."""
from __future__ import annotations

import subprocess

from behavioreval.core.config import ROOT

SHARED = ("scenarios", "profiles", "config.json")


def _shared() -> list[str]:
    """The shared paths that hold something (git refuses a pathspec that matches no file, e.g. an empty profiles/)."""
    out = []
    for p in SHARED:
        f = ROOT / p
        if f.is_file() or (f.is_dir() and any(x.is_file() for x in f.rglob("*"))):
            out.append(p)
    return out


def git(*args: str, check: bool = True) -> str:
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout).strip()[-400:] or f"git {' '.join(args)} failed")
    return r.stdout


def status(fetch: bool = True) -> dict:
    out = {"repo": str(ROOT), "branch": "", "changed": [], "ahead": 0, "behind": 0, "remote": "", "error": ""}
    try:
        out["branch"] = git("rev-parse", "--abbrev-ref", "HEAD").strip()
        out["remote"] = git("remote", "get-url", "origin", check=False).strip()
        if fetch and out["remote"]:
            git("fetch", "--quiet", "origin", check=False)
        out["changed"] = [l[3:] for l in git("status", "--porcelain", "--", *_shared()).splitlines() if l.strip()]
        counts = git("rev-list", "--left-right", "--count", f"HEAD...origin/{out['branch']}", check=False).split()
        if len(counts) == 2:
            out["ahead"], out["behind"] = int(counts[0]), int(counts[1])
    except Exception as exc:
        out["error"] = str(exc)[:300]
    return out


def pull() -> str:
    """Others' additions, keeping your uncommitted scenario edits (stash around the rebase)."""
    stashed = bool(git("status", "--porcelain", "--", *_shared()).strip())
    if stashed:
        git("stash", "push", "--quiet", "--", *_shared())
    try:
        return git("pull", "--rebase", "--quiet", "origin", status(fetch=False)["branch"]) or "up to date"
    finally:
        if stashed:
            git("stash", "pop", "--quiet", check=False)


def push(who: str, message: str) -> str:
    who = (who or "tester").strip()[:40]
    message = (message or "add scenarios").strip()[:120]
    git("add", "--", *_shared())
    if not git("diff", "--cached", "--name-only").strip():
        return "nothing to share: no changed scenarios or profiles"
    git("-c", f"user.name={who}", "-c", f"user.email={who.replace(' ', '.').lower()}@behavioreval.local", "commit", "--quiet", "-m", f"{who}: {message}")
    branch = status(fetch=False)["branch"]
    try:
        git("pull", "--rebase", "--quiet", "origin", branch)
    except RuntimeError as exc:
        git("rebase", "--abort", check=False)
        raise RuntimeError("could not merge with what others pushed: " + str(exc)[:200]) from None
    git("push", "--quiet", "origin", branch)
    return f"shared as '{who}: {message}'"
