"""Print one better-auth session token for one eval.party (password) account, from inside
mvp's eval_system uv project. ``DATACAT_CACHED_SESSION``-style reuse: with ``USERSIM_CACHED_SESSION``
set and still valid, no sign-in happens (password sign-in is rate-limited hard)."""
from __future__ import annotations

import argparse
import asyncio
import os
import sys


async def amain(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="evalsystem_mint")
    ap.add_argument("--email", required=True)
    ap.add_argument("--accounts", required=True)
    ap.add_argument("--env", default="dev")
    a = ap.parse_args(argv)
    os.environ.pop("EVAL_SESSION_COOKIE", None)
    cached = os.environ.pop("USERSIM_CACHED_SESSION", "")
    from eval_system.clients import resolve_session_cookie, resolve_user_id
    from eval_system.core.account_pool import create_account_pool
    from eval_system.environments import get_environment
    env = get_environment(a.env)
    pool = create_account_pool(a.accounts, environment=env, use_ssm=False)
    acct = next((x for x in pool.accounts if x.email == a.email), None)
    if acct is None:
        print(f"evalsystem_mint: {a.email} not in {a.accounts}", file=sys.stderr)
        return 2
    if cached and await resolve_user_id(env.api_url, cached):
        print("evalsystem_mint: reused the cached session", file=sys.stderr)
        print(cached)
        return 0
    creds = pool._resolve(acct)
    cookie = await resolve_session_cookie(env.auth_url, google_refresh_token=creds.google_refresh_token, google_client_id=creds.google_client_id,
                                          google_client_secret=creds.google_client_secret, email=creds.email, password=creds.password)
    if not cookie or not await resolve_user_id(env.api_url, cookie):
        print(f"evalsystem_mint: auth failed for {a.email}", file=sys.stderr)
        return 1
    print(cookie)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(amain()))
