#!/usr/bin/env python3
"""Push a one-shot Kaggle GPU kernel that builds kronos-cmaa-cache for BTC (and optional XAU).

Prereq: private dataset `kronos-cmaa-raw` already uploaded with:
  btc_1h.csv, btc_news.csv [, xau_1h.csv, xau_news.csv]

Usage:
  python scripts/upload_raw_dataset.py          # creates/updates kronos-cmaa-raw
  python scripts/build_cache_on_kaggle.py --asset BTC --dry-run
  python scripts/build_cache_on_kaggle.py --asset BTC
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.load_env import load_env  # noqa: E402

SLUG = "kronos-cmaa-cache-builder"
RAW_SLUG = "kronos-cmaa-raw"
REPO = "https://github.com/PrinceAikinsBaidoo/kronos-research.git"
POLL_S = 30
MAX_WAIT_MIN = 180


def run(cmd, check=True, capture=False):
    return subprocess.run(cmd, cwd=ROOT, check=check, text=True, capture_output=capture)


def kaggle_user():
    load_env()
    u = os.environ.get("KAGGLE_USERNAME")
    if u:
        return u
    p = Path.home() / ".kaggle" / "kaggle.json"
    if p.exists():
        return json.loads(p.read_text())["username"]
    # OAuth login: ask CLI
    r = run(["kaggle", "config", "view"], check=False, capture=True)
    for line in (r.stdout or "").splitlines():
        if "username" in line.lower():
            return line.split(":")[-1].strip().strip("'\"")
    # fallback from kernels list
    r = run(["kaggle", "kernels", "list", "--mine", "-p", "1"], check=False, capture=True)
    for line in (r.stdout or "").splitlines():
        if "/" in line and not line.startswith("ref"):
            return line.split("/")[0].strip()
    sys.exit("Cannot resolve Kaggle username; set KAGGLE_USERNAME or kaggle auth login")


def head_sha():
    r = run(["git", "rev-parse", "HEAD"], capture=True)
    return r.stdout.strip()


def render_kernel(user: str, asset: str, dry_run: bool, sha: str) -> Path:
    d = ROOT / ".kaggle_push_cache"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir()
    dry = "1" if dry_run else "0"
    code = f'''
import os, subprocess, sys, shutil
from pathlib import Path

REPO = "{REPO}"
COMMIT = "{sha}"
ASSET = "{asset}"
DRY = "{dry}" == "1"
WORK = Path("/kaggle/working")
REPO_DIR = WORK / "repo"
OUT = WORK / "cache"
OUT.mkdir(parents=True, exist_ok=True)

# find raw dataset
raw = None
for p in Path("/kaggle/input").rglob("btc_1h.csv" if ASSET == "BTC" else "xau_1h.csv"):
    raw = p.parent
    break
if raw is None:
    raise SystemExit("raw CSVs not found under /kaggle/input (need dataset kronos-cmaa-raw)")

token = ""
try:
    from kaggle_secrets import UserSecretsClient
    token = UserSecretsClient().get_secret("GITHUB_TOKEN")
except Exception:
    pass
url = REPO.replace("https://", f"https://{{token}}@") if token else REPO
subprocess.run(["git", "clone", "--quiet", url, str(REPO_DIR)], check=True)
subprocess.run(["git", "checkout", "--quiet", COMMIT], cwd=REPO_DIR, check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], cwd=REPO_DIR)

prices = raw / ("btc_1h.csv" if ASSET == "BTC" else "xau_1h.csv")
news = raw / ("btc_news.csv" if ASSET == "BTC" else "xau_news.csv")
cmd = [sys.executable, "scripts/build_cache.py", "--asset", ASSET,
       "--prices", str(prices), "--news", str(news), "--out", str(OUT)]
if ASSET == "BTC":
    cmd.append("--regular-grid")
if DRY:
    cmd.append("--dry-run")
print("Running:", " ".join(cmd), flush=True)
subprocess.run(cmd, cwd=REPO_DIR, check=True)
print("Cache build finished. Output under", OUT)
'''
    (d / "build_cache_kernel.py").write_text(code, encoding="utf-8")
    meta = {
        "id": f"{user}/{SLUG}",
        "title": SLUG,
        "code_file": "build_cache_kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [f"{user}/{RAW_SLUG}"],
        "competition_sources": [],
        "kernel_sources": [],
    }
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--asset", choices=["BTC", "XAU"], default="BTC")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-wait", action="store_true")
    a = ap.parse_args()

    user = kaggle_user()
    sha = head_sha()
    push_dir = render_kernel(user, a.asset, a.dry_run, sha)
    ref = f"{user}/{SLUG}"
    print(f"Pushing cache builder for {a.asset} (dry_run={a.dry_run}) at {sha[:8]}")
    run(["kaggle", "kernels", "push", "-p", str(push_dir)])
    if a.no_wait:
        print(f"Pushed. Check: kaggle kernels status {ref}")
        return

    t0 = time.time()
    seen = False
    while True:
        r = run(["kaggle", "kernels", "status", ref], check=False, capture=True)
        out = (r.stdout + r.stderr).upper()
        state = next((k for k in ("COMPLETE", "ERROR", "CANCEL", "RUNNING", "QUEUED") if k in out), "UNKNOWN")
        if state in {"RUNNING", "QUEUED"}:
            seen = True
        elapsed = time.time() - t0
        print(f"  [{int(elapsed // 60):02d}:{int(elapsed % 60):02d}] {state}")
        if state in {"COMPLETE", "ERROR", "CANCEL"} and (seen or elapsed > 90):
            break
        if elapsed > MAX_WAIT_MIN * 60:
            sys.exit("Timed out waiting for cache kernel")
        time.sleep(POLL_S)

    out_dir = ROOT / "experiments" / "_cache_build" / "kaggle_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    run(["kaggle", "kernels", "output", ref, "-p", str(out_dir)], check=False)
    print(f"Downloaded output to {out_dir}")
    if state != "COMPLETE":
        sys.exit(f"Kernel finished as {state}")


if __name__ == "__main__":
    main()
