#!/usr/bin/env python3
"""Create/update the private Kaggle dataset kronos-cmaa-raw from data/raw/*.csv."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
STAGE = ROOT / ".kaggle_raw_dataset"
SLUG = "kronos-cmaa-raw"
FILES = ["btc_1h.csv", "btc_news.csv", "xau_1h.csv", "xau_news.csv"]


def run(cmd):
    return subprocess.run(cmd, cwd=ROOT, check=True, text=True)


def kaggle_user() -> str:
    r = subprocess.run(["kaggle", "kernels", "list", "--mine", "-p", "1"],
                       cwd=ROOT, capture_output=True, text=True, check=False)
    for line in (r.stdout or "").splitlines():
        if "/" in line and not line.lower().startswith("ref"):
            return line.split()[0].split("/")[0].strip()
    sys.exit("Cannot resolve Kaggle username")


def main():
    missing = [f for f in FILES if not (RAW / f).exists()]
    if missing:
        sys.exit(f"Missing raw files: {missing}")
    user = kaggle_user()
    shutil.rmtree(STAGE, ignore_errors=True)
    STAGE.mkdir(parents=True)
    for f in FILES:
        shutil.copy2(RAW / f, STAGE / f)
    meta = {
        "title": "kronos-cmaa-raw",
        "id": f"{user}/{SLUG}",
        "licenses": [{"name": "CC0-1.0"}],
    }
    (STAGE / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # create or version
    r = subprocess.run(["kaggle", "datasets", "status", f"{user}/{SLUG}"],
                       cwd=ROOT, capture_output=True, text=True)
    exists = r.returncode == 0 and "404" not in (r.stdout + r.stderr)
    if exists:
        print(f"Updating dataset {user}/{SLUG} ...")
        run(["kaggle", "datasets", "version", "-p", str(STAGE), "-m", "refresh raw CSVs", "--dir-mode", "zip"])
    else:
        print(f"Creating dataset {user}/{SLUG} ...")
        run(["kaggle", "datasets", "create", "-p", str(STAGE), "--dir-mode", "zip"])
    print("Done.")


if __name__ == "__main__":
    main()
