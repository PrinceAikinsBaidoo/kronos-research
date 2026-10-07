#!/usr/bin/env python3
"""Score human labels vs auto BTC-relevance filter.

  python scripts/score_relevance_audit.py --labeled data/processed/relevance_audit_labeled.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labeled", type=Path, required=True)
    a = ap.parse_args()
    if not a.labeled.exists():
        raise SystemExit(f"Missing {a.labeled}")

    df = pd.read_csv(a.labeled)
    if "human_label" not in df.columns or "auto_relevant" not in df.columns:
        raise SystemExit("Need human_label and auto_relevant columns")

    df["human_label"] = df["human_label"].astype(str).str.strip().str.lower()
    df["auto_relevant"] = df["auto_relevant"].astype(str).str.strip().str.lower().isin(
        {"1", "true", "yes", "t"}
    ) | (df["auto_relevant"] == True)  # noqa: E712

    labeled = df[df["human_label"].isin(["relevant", "weak", "irrelevant"])].copy()
    print(f"labeled rows={len(labeled)} / {len(df)}")
    print("human_label counts:")
    print(labeled["human_label"].value_counts().to_string())

    # Binary: relevant vs irrelevant (exclude weak)
    binary = labeled[labeled["human_label"].isin(["relevant", "irrelevant"])].copy()
    y_true = binary["human_label"] == "relevant"
    y_pred = binary["auto_relevant"]
    tp = int(((y_true) & (y_pred)).sum())
    fp = int(((~y_true) & (y_pred)).sum())
    fn = int(((y_true) & (~y_pred)).sum())
    tn = int(((~y_true) & (~y_pred)).sum())
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    print("\nBinary (relevant vs irrelevant; weak excluded):")
    print(f"  TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"  precision={prec:.3f}  recall={rec:.3f}")

    # Treat weak as negative
    weak_as_neg = labeled.copy()
    y_true2 = weak_as_neg["human_label"] == "relevant"
    y_pred2 = weak_as_neg["auto_relevant"]
    tp2 = int(((y_true2) & (y_pred2)).sum())
    fp2 = int(((~y_true2) & (y_pred2)).sum())
    fn2 = int(((y_true2) & (~y_pred2)).sum())
    prec2 = tp2 / (tp2 + fp2) if (tp2 + fp2) else float("nan")
    rec2 = tp2 / (tp2 + fn2) if (tp2 + fn2) else float("nan")
    print("\nTreating weak as negative:")
    print(f"  precision={prec2:.3f}  recall={rec2:.3f}")


if __name__ == "__main__":
    main()
