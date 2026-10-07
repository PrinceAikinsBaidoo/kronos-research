#!/usr/bin/env python3
"""Prepare publication-grade BTC news for Q1 freeze.

Keeps data/raw/btc_news.csv untouched. Writes processed CSVs under --out-dir.

  python scripts/prepare_q1_btc_news.py
  python scripts/prepare_q1_btc_news.py --in tests/fixtures/q1_news_sample.csv --out-dir data/processed/_dry
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

URL_RE = re.compile(r"https?://\S+")
HANDLE_RE = re.compile(r"@\w+")
WS_RE = re.compile(r"\s+")
BTC_RE = re.compile(
    r"bitcoin|\bbtc\b|crypto|blockchain|ethereum|\beth\b|coinbase|microstrategy|"
    r"\bmstr\b|stablecoin|etf|binance|mining|hashrate|satoshi|web3|defi",
    re.I,
)


def light_clean_series(s: pd.Series, max_pre: int) -> pd.Series:
    # Truncate first to avoid MemoryError on multi-MB aggregate rows
    out = s.fillna("").astype(str).map(lambda x: x[:max_pre] if len(x) > max_pre else x)
    out = out.str.replace(URL_RE, "", regex=True)
    out = out.str.replace(HANDLE_RE, "", regex=True)
    out = out.str.replace(WS_RE, " ", regex=True).str.strip()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", type=Path, default=ROOT / "data" / "raw" / "btc_news.csv")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data" / "processed")
    ap.add_argument("--max-chars", type=int, default=4000)
    ap.add_argument("--min-chars", type=int, default=10)
    ap.add_argument("--relevance-filter", action=argparse.BooleanOptionalAction, default=True)
    a = ap.parse_args()

    if not a.inp.exists():
        raise SystemExit(f"Missing input: {a.inp}")

    df = pd.read_csv(a.inp)
    if "published_at" not in df.columns or "text" not in df.columns:
        raise SystemExit(f"Need published_at,text columns; got {list(df.columns)}")

    n_in = len(df)
    text = light_clean_series(df["text"], max_pre=a.max_chars * 2)
    pub = df["published_at"]
    fam = df["source_family"].astype(str) if "source_family" in df.columns else pd.Series(["unknown"] * n_in)

    empty = text.str.len() == 0
    too_short = (~empty) & (text.str.len() < a.min_chars)
    if a.relevance_filter:
        relevant = text.str.contains(BTC_RE, na=False)
    else:
        relevant = pd.Series([True] * n_in)
    not_rel = (~empty) & (~too_short) & (~relevant)

    drop_parts = []
    if empty.any():
        drop_parts.append(pd.DataFrame({
            "published_at": pub[empty].values,
            "text_preview": text[empty].str[:200].values,
            "reason": "empty",
        }))
    if too_short.any():
        drop_parts.append(pd.DataFrame({
            "published_at": pub[too_short].values,
            "text_preview": text[too_short].str[:200].values,
            "reason": "too_short",
        }))
    if not_rel.any():
        drop_parts.append(pd.DataFrame({
            "published_at": pub[not_rel].values,
            "text_preview": text[not_rel].str[:200].values,
            "reason": "not_relevant",
        }))
    drop_df = pd.concat(drop_parts, ignore_index=True) if drop_parts else pd.DataFrame(
        columns=["published_at", "text_preview", "reason"]
    )

    keep = (~empty) & (~too_short) & relevant
    kept_text = text[keep].map(lambda t: (t[: a.max_chars].rstrip() + "…") if len(t) > a.max_chars else t)
    out = pd.DataFrame({
        "published_at": pd.to_datetime(pub[keep], utc=True, format="ISO8601", errors="coerce"),
        "text": kept_text.values,
        "source_family": fam[keep].values,
    })
    out = out.dropna(subset=["published_at"])
    out["published_at"] = out["published_at"].dt.floor("s")
    out = out.sort_values("published_at").drop_duplicates("text").reset_index(drop=True)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    kept_path = a.out_dir / "btc_news_q1.csv"
    drop_path = a.out_dir / "btc_news_dropped.csv"
    # Cache/build expects published_at,text only — write dual: full + slim
    out.to_csv(kept_path, index=False)
    out[["published_at", "text"]].to_csv(a.out_dir / "btc_news_q1_slim.csv", index=False)
    drop_df.to_csv(drop_path, index=False)
    print(f"input={n_in} kept={len(out)} dropped={len(drop_df)}")
    if len(drop_df):
        print(drop_df["reason"].value_counts().to_string())
    print(f"Wrote {kept_path}")
    print(f"Wrote {a.out_dir / 'btc_news_q1_slim.csv'}")
    print(f"Wrote {drop_path}")


if __name__ == "__main__":
    main()
