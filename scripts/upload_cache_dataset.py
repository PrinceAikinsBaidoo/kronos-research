#!/usr/bin/env python3
"""Create/update private Kaggle dataset kronos-cmaa-cache from a local cache tree.

Expects:
  <cache-dir>/BTC/*.npy + meta.json
  optional <cache-dir>/XAU/...

Usage:
  python scripts/upload_cache_dataset.py --cache-dir experiments/_cache_build/kaggle_out_q1/cache
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / ".kaggle_cache_dataset"
SLUG = "kronos-cmaa-cache"
REQUIRED = ("meta.json", "s1_ids.npy", "s2_ids.npy", "y.npy")


def run(cmd):
    return subprocess.run(cmd, cwd=ROOT, check=True, text=True)


def kaggle_user() -> str:
    r = subprocess.run(
        ["kaggle", "kernels", "list", "--mine", "-p", "1"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    for line in (r.stdout or "").splitlines():
        if "/" in line and not line.lower().startswith("ref"):
            return line.split()[0].split("/")[0].strip()
    sys.exit("Cannot resolve Kaggle username")


def validate_asset(d: Path) -> None:
    missing = [f for f in REQUIRED if not (d / f).exists()]
    if missing:
        sys.exit(f"{d}: missing {missing}")
    shard_meta = d / "text_states_shards.json"
    if shard_meta.exists():
        sm = json.loads(shard_meta.read_text(encoding="utf-8"))
        shards = sm.get("shards") or []
        if not shards:
            sys.exit(f"{d}: text_states_shards.json has no shards")
        for name in shards:
            p = d / name
            if not p.exists() or p.stat().st_size < 1024:
                sys.exit(f"{d}: bad shard {name}")
        print(f"  {d.name}: {len(shards)} text_states shards OK")
    else:
        ts = d / "text_states.npy"
        if not ts.exists() or ts.stat().st_size < 1024:
            sys.exit(f"{d}: text_states.npy missing/empty — rebuild cache first")
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    print(f"  {d.name}: N={meta.get('N')} U={meta.get('U')} has_text={meta.get('has_text_rate')}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--message", default="Q1 BTC cache freeze")
    a = ap.parse_args()

    cache = a.cache_dir
    if not cache.is_dir():
        sys.exit(f"Not a directory: {cache}")
    assets = [p for p in cache.iterdir() if p.is_dir() and (p / "meta.json").exists()]
    if not assets:
        sys.exit(f"No asset folders with meta.json under {cache}")
    for d in assets:
        validate_asset(d)

    user = kaggle_user()
    shutil.rmtree(STAGE, ignore_errors=True)
    STAGE.mkdir(parents=True)
    for d in assets:
        dest = STAGE / d.name
        shutil.copytree(d, dest)
    meta = {"title": SLUG, "id": f"{user}/{SLUG}", "licenses": [{"name": "CC0-1.0"}]}
    (STAGE / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    r = subprocess.run(
        ["kaggle", "datasets", "status", f"{user}/{SLUG}"],
        cwd=ROOT, capture_output=True, text=True,
    )
    exists = r.returncode == 0 and "404" not in (r.stdout + r.stderr) and "403" not in (r.stdout + r.stderr)
    if exists:
        print(f"Updating dataset {user}/{SLUG} ...")
        run(["kaggle", "datasets", "version", "-p", str(STAGE), "-m", a.message, "--dir-mode", "zip"])
    else:
        print(f"Creating dataset {user}/{SLUG} ...")
        run(["kaggle", "datasets", "create", "-p", str(STAGE), "--dir-mode", "zip"])
    print("Done.")


if __name__ == "__main__":
    main()
