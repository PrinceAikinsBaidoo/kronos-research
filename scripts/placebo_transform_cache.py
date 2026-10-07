#!/usr/bin/env python3
"""Create placebo news CSVs before rebuilding a separate cache.

Does NOT modify production data/raw or data/processed/btc_news_q1.csv.

  python scripts/placebo_transform_cache.py --mode shuffle_text \\
      --in data/processed/btc_news_q1_slim.csv --out data/processed/placebo_shuffle_text.csv

  python scripts/placebo_transform_cache.py --mode random_date \\
      --in data/processed/btc_news_q1_slim.csv --out data/processed/placebo_random_date.csv --seed 42
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["shuffle_text", "random_date"], required=True)
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    df = pd.read_csv(a.inp)
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, format="ISO8601", errors="coerce")
    df = df.dropna(subset=["published_at", "text"]).reset_index(drop=True)
    rng = np.random.default_rng(a.seed)

    if a.mode == "shuffle_text":
        perm = rng.permutation(len(df))
        df["text"] = df["text"].to_numpy()[perm]
    else:
        # Keep texts; permute timestamps so content no longer matches calendar time
        perm = rng.permutation(len(df))
        df["published_at"] = df["published_at"].to_numpy()[perm]

    df = df.sort_values("published_at").reset_index(drop=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df[["published_at", "text"]].to_csv(a.out, index=False)
    print(f"Wrote {a.out} mode={a.mode} rows={len(df)}")
    print("Rebuild a SEPARATE cache from this CSV; do not overwrite the main kronos-cmaa-cache.")


if __name__ == "__main__":
    main()
