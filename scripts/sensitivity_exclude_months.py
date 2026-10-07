#!/usr/bin/env python3
"""Document / apply month exclusions for Q1 sensitivity (e.g. 2025-06, 2026-04).

Writes a filtered slim news CSV and a short JSON report of coverage impact.
Does not modify raw files.

  python scripts/sensitivity_exclude_months.py --months 2025-06 2026-04
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--news", type=Path, default=ROOT / "data" / "processed" / "btc_news_q1_slim.csv")
    ap.add_argument("--months", nargs="+", default=["2025-06", "2026-04"])
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "processed" / "btc_news_q1_sensitivity.csv")
    ap.add_argument("--report", type=Path, default=ROOT / "data" / "processed" / "sensitivity_months.json")
    a = ap.parse_args()

    src = a.news
    if not src.exists():
        src = ROOT / "data" / "raw" / "btc_news.csv"
    df = pd.read_csv(src)
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, format="ISO8601", errors="coerce")
    df = df.dropna(subset=["published_at", "text"])
    before = len(df)
    mask = pd.Series(False, index=df.index)
    for m in a.months:
        y, mo = map(int, m.split("-"))
        mask |= (df["published_at"].dt.year == y) & (df["published_at"].dt.month == mo)
    kept = df.loc[~mask].copy()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    kept[["published_at", "text"]].to_csv(a.out, index=False)
    report = {
        "excluded_months": a.months,
        "rows_before": before,
        "rows_after": len(kept),
        "rows_removed": int(mask.sum()),
        "out": str(a.out).replace("\\", "/"),
    }
    a.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
