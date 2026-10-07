#!/usr/bin/env python3
"""Draw a stratified sample for human BTC-relevance labeling.

  python scripts/sample_relevance_audit.py
  python scripts/sample_relevance_audit.py --n 400 --seed 42
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BTC_RE = re.compile(
    r"bitcoin|\bbtc\b|crypto|blockchain|ethereum|\beth\b|coinbase|microstrategy|"
    r"\bmstr\b|stablecoin|etf|binance|mining|hashrate|satoshi|web3|defi",
    re.I,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", type=Path, default=None)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "processed" / "relevance_audit_sample.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--preview-chars", type=int, default=400)
    a = ap.parse_args()

    candidates = [
        a.inp,
        ROOT / "data" / "processed" / "btc_news_q1.csv",
        ROOT / "data" / "raw" / "btc_news.csv",
    ]
    inp = next((p for p in candidates if p is not None and p.exists()), None)
    if inp is None:
        raise SystemExit("No input news CSV found")

    df = pd.read_csv(inp)
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, format="ISO8601", errors="coerce")
    df = df.dropna(subset=["published_at", "text"])
    df["text"] = df["text"].astype(str)
    df["auto_relevant"] = df["text"].str.contains(BTC_RE, na=False)
    df["ym"] = df["published_at"].dt.to_period("M")

    # Stratify by month
    months = df["ym"].unique()
    per = max(1, a.n // max(1, len(months)))
    parts = []
    rng = a.seed
    for i, m in enumerate(sorted(months, key=str)):
        g = df[df["ym"] == m]
        k = min(len(g), per)
        parts.append(g.sample(n=k, random_state=rng + i))
    sample = pd.concat(parts, ignore_index=True)
    if len(sample) < a.n:
        extra = df.drop(sample.index, errors="ignore")
        need = min(a.n - len(sample), len(extra))
        if need > 0:
            sample = pd.concat([sample, extra.sample(n=need, random_state=rng)], ignore_index=True)
    sample = sample.head(a.n).reset_index(drop=True)

    out = pd.DataFrame({
        "audit_id": [f"A{i:04d}" for i in range(len(sample))],
        "published_at": sample["published_at"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "text": sample["text"].str[: a.preview_chars],
        "auto_relevant": sample["auto_relevant"].astype(bool),
        "human_label": "",
        "notes": "",
    })
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    print(f"Wrote {a.out} n={len(out)} from {inp}")
    print("auto_relevant rate", float(out["auto_relevant"].mean()))


if __name__ == "__main__":
    main()
