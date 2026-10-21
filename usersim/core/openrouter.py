"""One POST to OpenRouter's chat completions, and where the key comes from."""
from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

from usersim.core.config import ROOT

URL = "https://openrouter.ai/api/v1/chat/completions"
ENV_FILES = (ROOT / ".env",)   # read only for OPENROUTER_API_KEY


def post(body: dict, key: str, timeout_s: float = 240) -> dict:
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), method="POST",
                                 headers={"Authorization": "Bearer " + key, "Content-Type": "application/json",
                                          "HTTP-Referer": "https://github.com/Adcock-AGI-Inc/usersim", "X-Title": "usersim"})
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        return json.loads(r.read())


def key(environ=None, files: tuple[Path, ...] = ENV_FILES) -> str:
    environ = os.environ if environ is None else environ
    for name in ("USERSIM_OPENROUTER_KEY", "OPENROUTER_API_KEY"):
        if environ.get(name):
            return environ[name].strip()
    for f in files:
        if f.exists():
            for line in f.read_text(encoding="utf-8").splitlines():
                if line.startswith("OPENROUTER_API_KEY="):
                    return line.split("=", 1)[1].split("#", 1)[0].strip().strip("'\"")
    return ""
