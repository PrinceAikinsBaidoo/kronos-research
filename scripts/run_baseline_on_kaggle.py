#!/usr/bin/env python3
"""Push a Kaggle GPU kernel that locks the price-only BTC baseline JSON.

Prereq: private dataset kronos-cmaa-cache with BTC/ present.

  python scripts/run_baseline_on_kaggle.py
  python scripts/run_baseline_on_kaggle.py --no-wait
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

SLUG = "kronos-cmaa-baseline"
CACHE_SLUG = "kronos-cmaa-cache"
BUILDER_SLUG = "kronos-cmaa-cache-builder"
REPO = "https://github.com/PrinceAikinsBaidoo/kronos-research.git"
POLL_S = 30
MAX_WAIT_MIN = 90


def run(cmd, check=True, capture=False):
    return subprocess.run(cmd, cwd=ROOT, check=check, text=True, capture_output=capture)


def kaggle_user() -> str:
    load_env()
    u = os.environ.get("KAGGLE_USERNAME")
    if u:
        return u
    r = run(["kaggle", "config", "view"], check=False, capture=True)
    for line in (r.stdout or "").splitlines():
        if "username" in line.lower():
            return line.split(":")[-1].strip().strip("'\"")
    sys.exit("Cannot resolve Kaggle username")


def head_sha() -> str:
    return run(["git", "rev-parse", "HEAD"], capture=True).stdout.strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-wait", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    load_env()
    user = kaggle_user()
    sha = head_sha()
    gh = json.dumps((os.environ.get("GITHUB_TOKEN") or "").strip())
    smoke = "1" if a.smoke else "0"

    d = ROOT / ".kaggle_push_baseline"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir()
    code = f'''
import os, subprocess, sys, shutil, json
from pathlib import Path

REPO = "{REPO}"
COMMIT = "{sha}"
SMOKE = "{smoke}" == "1"
GH = {gh}
WORK = Path("/kaggle/working")
REPO_DIR = WORK / "repo"

cache = None
for p in Path("/kaggle/input").rglob("meta.json"):
    if p.parent.name == "BTC":
        cache = p.parent.parent
        break
if cache is None:
    raise SystemExit("kronos-cmaa-cache BTC/meta.json not found")

token = GH
try:
    from kaggle_secrets import UserSecretsClient
    token = UserSecretsClient().get_secret("GITHUB_TOKEN") or token
except Exception:
    pass
url = REPO.replace("https://", f"https://{{token}}@") if token else REPO
subprocess.run(["git", "clone", "--quiet", url, str(REPO_DIR)], check=True)
subprocess.run(["git", "checkout", "--quiet", COMMIT], cwd=REPO_DIR, check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], cwd=REPO_DIR)

out = WORK / "BTC.json"
cmd = [sys.executable, "scripts/run_price_only_baseline.py", "--asset", "BTC",
       "--cache-dir", str(cache), "--out", str(out)]
if SMOKE:
    cmd.append("--smoke")
print("Running:", " ".join(cmd), flush=True)
subprocess.run(cmd, cwd=REPO_DIR, check=True)
print(out.read_text()[:2000], flush=True)
'''
    (d / "baseline_kernel.py").write_text(code, encoding="utf-8")
    meta = {
        "id": f"{user}/{SLUG}",
        "title": SLUG,
        "code_file": "baseline_kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        # Prefer builder kernel output until kronos-cmaa-cache dataset is published
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [f"{user}/{BUILDER_SLUG}"],
    }
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    ref = f"{user}/{SLUG}"
    print(f"Pushing {ref} at {sha[:8]}")
    run(["kaggle", "kernels", "push", "-p", str(d)])
    if a.no_wait:
        print(f"Pushed. Status: kaggle kernels status {ref}")
        return

    t0 = time.time()
    seen = False
    state = "UNKNOWN"
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
            sys.exit("Timed out")
        time.sleep(POLL_S)

    out_dir = ROOT / "experiments" / "_baseline" / "kaggle_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    run(["kaggle", "kernels", "output", ref, "-p", str(out_dir)], check=False)
    src = out_dir / "BTC.json"
    if src.exists():
        dest = ROOT / "locked" / "baselines" / "BTC.json"
        dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Locked {dest}")
    if state != "COMPLETE":
        sys.exit(f"Kernel finished as {state}")


if __name__ == "__main__":
    main()
