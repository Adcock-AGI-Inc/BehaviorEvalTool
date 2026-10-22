"""The tester's own Chrome, signed in to dev, as the browser the tool drives. Chrome only
opens a DevTools port for a non-default user-data-dir, so the tool copies the signed-in
profile (cookies and all; the person's own action) into its own directory and starts a
second Chrome from it with ``--remote-debugging-port``. The everyday Chrome keeps running."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from behavioreval.browser import cdp

CHROME_BIN = {"darwin": "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
              "linux": "/usr/bin/google-chrome", "win32": r"C:\Program Files\Google\Chrome\Application\chrome.exe"}.get(sys.platform, "google-chrome")
CHROME_DIR = {"darwin": "~/Library/Application Support/Google/Chrome", "linux": "~/.config/google-chrome",
              "win32": r"~\AppData\Local\Google\Chrome\User Data"}.get(sys.platform, "~/.config/google-chrome")
CACHE_SKIP = ("Cache", "Code Cache", "GPUCache", "DawnCache", "DawnGraphiteCache", "DawnWebGPUCache", "ShaderCache", "GrShaderCache",
              "Service Worker", "Session Storage", "IndexedDB", "File System", "blob_storage", "Download Service", "optimization_guide_*",
              "*.ldb", "LOG", "LOG.old")


def list_profiles(chrome_dir: Path) -> list[dict]:
    """[{dir, name, email}] from Chrome's Local State, so a person can pick the one signed in to dev."""
    f = chrome_dir / "Local State"
    if not f.is_file():
        return []
    try:
        cache = json.loads(f.read_text(encoding="utf-8")).get("profile", {}).get("info_cache", {})
    except ValueError:
        return []
    return [{"dir": d, "name": str(v.get("name") or d), "email": str(v.get("user_name") or "")} for d, v in sorted(cache.items())]


def trust_profile(chrome_dir: Path, profile_directory: str, dst: Path, out=print) -> None:
    """Copy one Chrome profile (minus caches) into ``dst`` so a second Chrome can start from it."""
    src = chrome_dir / profile_directory
    if not src.is_dir():
        raise ValueError(f"{src} not found — pick one of: " + ", ".join(p["dir"] for p in list_profiles(chrome_dir)))
    if not shutil.which("rsync"):
        raise RuntimeError("rsync is not on PATH")
    dst.mkdir(parents=True, exist_ok=True)
    dst.chmod(0o700)
    if (chrome_dir / "Local State").is_file():
        subprocess.run(["rsync", "-a", str(chrome_dir / "Local State"), str(dst) + "/"], check=True)
    excludes = [x for name in CACHE_SKIP for x in ("--exclude", name)]
    out(f"copying {src} → {dst / profile_directory} (skipping caches)…")
    r = subprocess.run(["rsync", "-a", "--delete"] + excludes + [str(src), str(dst) + "/"])
    if r.returncode not in (0, 24):   # 24 = files vanished mid-copy (live Chrome); harmless
        raise RuntimeError(f"rsync exit {r.returncode}")


def launch(dst: Path, port: int, profile_directory: str, out=print) -> bool:
    if cdp.up(port):
        out(f"Chrome already answering on 127.0.0.1:{port}")
        return True
    if not Path(CHROME_BIN).is_file():
        raise RuntimeError(f"{CHROME_BIN} not found")
    cmd = [CHROME_BIN, f"--user-data-dir={dst}", f"--remote-debugging-port={port}", "--no-first-run", "--no-default-browser-check",
           "--accept-lang=en-US", "--window-size=1200,900", "--disable-backgrounding-occluded-windows", "--disable-renderer-backgrounding",
           "--disable-background-timer-throttling", f"--profile-directory={profile_directory}", "--disable-extensions", "about:blank"]
    subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    for _ in range(40):
        if cdp.up(port):
            out(f"Chrome up: profile copy {dst}, CDP 127.0.0.1:{port}. Leave the window alone while a run drives it.")
            return True
        time.sleep(0.5)
    raise RuntimeError(f"Chrome did not open 127.0.0.1:{port} within 20 s")


def start(chrome_dir: Path, profile_directory: str, dst: Path, port: int, out=print) -> bool:
    """Copy (when not yet copied today) and launch. Re-copying refreshes the sign-in cookies."""
    if cdp.up(port):
        out(f"Chrome already answering on 127.0.0.1:{port}")
        return True
    trust_profile(chrome_dir, profile_directory, dst, out)
    return launch(dst, port, profile_directory, out)
