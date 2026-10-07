#!/usr/bin/env python3
"""Create/update the private Kaggle dataset kronos-cmaa-raw from data/raw/*.csv.

Q1 freeze: pass --use-q1-news to stage data/processed/btc_news_q1_slim.csv as btc_news.csv
(cache builder expects that filename).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--use-q1-news",
        action="store_true",
        help="Stage processed Q1 slim BTC news as btc_news.csv",
    )
    a = ap.parse_args()

    missing = [f for f in FILES if f != "btc_news.csv" and not (RAW / f).exists()]
    if missing:
        sys.exit(f"Missing raw files: {missing}")
    if a.use_q1_news:
        q1 = PROCESSED / "btc_news_q1_slim.csv"
        if not q1.exists():
            q1 = PROCESSED / "btc_news_q1.csv"
        if not q1.exists():
            sys.exit("Missing data/processed/btc_news_q1_slim.csv — run prepare_q1_btc_news.py first")
    elif not (RAW / "btc_news.csv").exists():
        sys.exit("Missing data/raw/btc_news.csv")

    user = kaggle_user()
    shutil.rmtree(STAGE, ignore_errors=True)
    STAGE.mkdir(parents=True)
    for f in FILES:
        if f == "btc_news.csv" and a.use_q1_news:
            src = PROCESSED / "btc_news_q1_slim.csv"
            if not src.exists():
                # strip source_family if only full q1 exists
                full = PROCESSED / "btc_news_q1.csv"
                import pandas as pd
                df = pd.read_csv(full)
                df[["published_at", "text"]].to_csv(STAGE / f, index=False)
                print(f"  staged {f} from {full} (slimmed)")
                continue
            shutil.copy2(src, STAGE / f)
            print(f"  staged {f} from {src}")
        else:
            shutil.copy2(RAW / f, STAGE / f)
    meta = {
        "title": "kronos-cmaa-raw",
        "id": f"{user}/{SLUG}",
        "licenses": [{"name": "CC0-1.0"}],
    }
    (STAGE / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    r = subprocess.run(["kaggle", "datasets", "status", f"{user}/{SLUG}"],
                       cwd=ROOT, capture_output=True, text=True)
    exists = r.returncode == 0 and "404" not in (r.stdout + r.stderr)
    msg = "Q1 processed BTC news freeze" if a.use_q1_news else "refresh raw CSVs"
    if exists:
        print(f"Updating dataset {user}/{SLUG} ...")
        run(["kaggle", "datasets", "version", "-p", str(STAGE), "-m", msg, "--dir-mode", "zip"])
    else:
        print(f"Creating dataset {user}/{SLUG} ...")
        run(["kaggle", "datasets", "create", "-p", str(STAGE), "--dir-mode", "zip"])
    print("Done.")


if __name__ == "__main__":
    main()

