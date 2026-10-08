#!/usr/bin/env python3
"""Publish kronos-cmaa-cache from a completed cache-builder kernel (no local download).

Uses kernel_sources so files stay on Kaggle's side (avoids flaky CDN downloads).
Needs Kaggle API user/key available as User Secrets, or OAuth credentials.json
injected at push time into the private kernel.

  python scripts/publish_cache_from_kernel.py
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

SLUG = "kronos-cmaa-cache-publish"
BUILDER = "kronos-cmaa-cache-builder"
CACHE_SLUG = "kronos-cmaa-cache"
POLL_S = 20
MAX_WAIT_MIN = 45


def run(cmd, check=True, capture=False):
    return subprocess.run(cmd, cwd=ROOT, check=check, text=True, capture_output=capture)


def kaggle_user() -> str:
    load_env()
    r = run(["kaggle", "config", "view"], check=False, capture=True)
    for line in (r.stdout or "").splitlines():
        if "username" in line.lower():
            return line.split(":")[-1].strip().strip("'\"")
    sys.exit("Cannot resolve Kaggle username")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-wait", action="store_true")
    a = ap.parse_args()

    load_env()
    user = kaggle_user()

    d = ROOT / ".kaggle_push_cache_publish"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir()
    code = f'''
import json, os, shutil, subprocess, sys
from pathlib import Path

USER = "{user}"
CACHE_SLUG = "{CACHE_SLUG}"
WORK = Path("/kaggle/working")

src = None
for p in Path("/kaggle/input").rglob("meta.json"):
    if p.parent.name == "BTC":
        src = p.parent
        break
if src is None:
    for p in Path("/kaggle/input").rglob("*"):
        if p.is_file():
            print("INPUT", p, p.stat().st_size, flush=True)
    raise SystemExit("BTC cache not found under /kaggle/input")

print("SRC", src, flush=True)
for p in sorted(src.iterdir()):
    print(f"  {{p.name}}: {{p.stat().st_size}}", flush=True)

required = ["meta.json", "s1_ids.npy", "s2_ids.npy", "y.npy", "text_mask.npy"]
missing = [n for n in required if not (src / n).exists() or (src / n).stat().st_size < 64]
if missing:
    raise SystemExit(f"Missing/corrupt in builder output: {{missing}}")

shard_meta = src / "text_states_shards.json"
if shard_meta.exists():
    sm = json.loads(shard_meta.read_text())
    for n in sm.get("shards", []):
        p = src / n
        if not p.exists() or p.stat().st_size < 1024:
            raise SystemExit(f"Bad shard {{n}} size={{p.stat().st_size if p.exists() else 0}}")
else:
    ts = src / "text_states.npy"
    if not ts.exists() or ts.stat().st_size < 1024:
        raise SystemExit("No usable text_states")

# Dataset API auth: Add notebook secrets KAGGLE_USERNAME + KAGGLE_KEY
ku = USER
kk = ""
try:
    from kaggle_secrets import UserSecretsClient
    usc = UserSecretsClient()
    ku = usc.get_secret("KAGGLE_USERNAME") or ku
    kk = usc.get_secret("KAGGLE_KEY") or ""
except Exception as e:
    print("secrets:", e, flush=True)
if not kk:
    # Still verify builder output for diagnostics
    (WORK / "verify_ok.txt").write_text("builder output looks intact; add KAGGLE_KEY secret to publish\\n")
    raise SystemExit(
        "Add Kaggle User Secrets KAGGLE_USERNAME and KAGGLE_KEY on this notebook, then re-run publish"
    )

os.environ["KAGGLE_USERNAME"] = ku
os.environ["KAGGLE_KEY"] = kk
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kaggle"], check=True)
stage = WORK / "stage"
shutil.rmtree(stage, ignore_errors=True)
stage.mkdir(parents=True)
shutil.copytree(src, stage / "BTC")
meta = {{"title": CACHE_SLUG, "id": f"{{ku}}/{{CACHE_SLUG}}", "licenses": [{{"name": "CC0-1.0"}}]}}
(stage / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))

st = subprocess.run(["kaggle", "datasets", "status", f"{{ku}}/{{CACHE_SLUG}}"],
                    capture_output=True, text=True)
blob = (st.stdout or "") + (st.stderr or "")
exists = st.returncode == 0 and "404" not in blob and "403" not in blob
if exists:
    print("Updating dataset...", flush=True)
    subprocess.run(["kaggle", "datasets", "version", "-p", str(stage),
                    "-m", "Q1 BTC sharded cache from builder kernel", "--dir-mode", "zip"], check=True)
else:
    print("Creating dataset...", flush=True)
    subprocess.run(["kaggle", "datasets", "create", "-p", str(stage), "--dir-mode", "zip"], check=True)
print("PUBLISH_OK", flush=True)
'''
    (d / "publish_kernel.py").write_text(code, encoding="utf-8")
    meta = {
        "id": f"{user}/{SLUG}",
        "title": SLUG,
        "code_file": "publish_kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [f"{user}/{BUILDER}"],
    }
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    ref = f"{user}/{SLUG}"
    print(f"Pushing {ref} (kernel_sources={user}/{BUILDER})")
    run(["kaggle", "kernels", "push", "-p", str(d)])
    if a.no_wait:
        print(f"Pushed. kaggle kernels status {ref}")
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
        if state in {"COMPLETE", "ERROR", "CANCEL"} and (seen or elapsed > 60):
            break
        if elapsed > MAX_WAIT_MIN * 60:
            sys.exit("Timed out")
        time.sleep(POLL_S)

    out_dir = ROOT / "experiments" / "_cache_publish" / "kaggle_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    run(["kaggle", "kernels", "output", ref, "-p", str(out_dir)], check=False)
    if state != "COMPLETE":
        sys.exit(f"Finished as {state}")
    print("Done.")


if __name__ == "__main__":
    main()
