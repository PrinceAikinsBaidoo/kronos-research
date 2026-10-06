#!/usr/bin/env python3
"""QA for data/raw CSVs before build_cache (P1 gate)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PRICE_NEED = {"timestamp", "open", "high", "low", "close", "volume"}
NEWS_NEED = {"published_at", "text"}


def _load_prices(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    tcol = next((c for c in ("timestamp", "open_time", "datetime", "date", "time")
                 if c in df.columns), None)
    if tcol is None:
        raise ValueError(f"{path.name}: missing timestamp column")
    ts = df[tcol]
    if pd.api.types.is_numeric_dtype(ts):
        unit = "us" if ts.iloc[0] > 10**15 else ("ms" if ts.iloc[0] > 10**11 else "s")
        df["timestamp"] = pd.to_datetime(ts, unit=unit, utc=True)
    else:
        df["timestamp"] = pd.to_datetime(ts, utc=True)
    for c in ("open", "high", "low", "close", "volume"):
        if c not in df.columns:
            raise ValueError(f"{path.name}: missing {c}")
    if "amount" not in df.columns:
        df["amount"] = 0.0
    return df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)


def _load_news(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "published_at" not in df.columns or "text" not in df.columns:
        raise ValueError(f"{path.name}: need published_at, text; got {list(df.columns)}")
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, format="ISO8601")
    df["text"] = df["text"].astype(str)
    return df.sort_values("published_at").drop_duplicates("text").reset_index(drop=True)


def report_prices(name: str, path: Path) -> dict:
    df = _load_prices(path)
    dt = df["timestamp"].diff().dropna()
    med = dt.median()
    gaps = int((dt > med * 3).sum()) if len(dt) else 0
    info = {
        "file": path.name,
        "bars": len(df),
        "start": str(df["timestamp"].iloc[0]),
        "end": str(df["timestamp"].iloc[-1]),
        "median_step": str(med),
        "large_gaps": gaps,
        "nan_ohlc": int(df[["open", "high", "low", "close"]].isna().any(axis=1).sum()),
    }
    print(f"[{name} prices] {info}")
    return {"df": df, **info}


def report_news(name: str, path: Path) -> dict:
    df = _load_news(path)
    info = {
        "file": path.name,
        "rows": len(df),
        "start": str(df["published_at"].iloc[0]) if len(df) else None,
        "end": str(df["published_at"].iloc[-1]) if len(df) else None,
        "empty_text": int((df["text"].str.len() < 10).sum()),
    }
    print(f"[{name} news] {info}")
    return {"df": df, **info}


def coverage(asset: str, prices: pd.DataFrame, news: pd.DataFrame | None):
    if news is None or news.empty:
        print(f"[{asset} coverage] no news file yet")
        return
    # Bar close ≈ open + median step (fallback 1h)
    step = prices["timestamp"].diff().median()
    if pd.isna(step):
        step = pd.Timedelta(hours=1)
    close = prices["timestamp"] + step
    # Fraction of bars with ≥1 headline in (close-24h, close]
    window = pd.Timedelta(hours=24)
    # Approximate with searchsorted
    pubs = pd.to_datetime(news["published_at"], utc=True, format="ISO8601").dt.tz_localize(None).to_numpy()
    close_ns = pd.to_datetime(close, utc=True).dt.tz_localize(None).to_numpy()
    win = np.timedelta64(int(window.total_seconds()), "s")
    lo = np.searchsorted(pubs, close_ns - win, side="right")
    hi = np.searchsorted(pubs, close_ns, side="right")
    has = float((hi > lo).mean()) if len(close_ns) else 0.0
    t0 = max(prices["timestamp"].iloc[0], news["published_at"].iloc[0])
    t1 = min(prices["timestamp"].iloc[-1], news["published_at"].iloc[-1])
    print(f"[{asset} coverage] bars_with_text_24h={has:.3f}  overlap={t0} -> {t1}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", type=Path, default=RAW)
    a = ap.parse_args()
    ok = True
    for asset, pfile, nfile in (
        ("BTC", "btc_1h.csv", "btc_news.csv"),
        ("XAU", "xau_1h.csv", "xau_news.csv"),
    ):
        pp = a.raw_dir / pfile
        npth = a.raw_dir / nfile
        prices = news = None
        if pp.exists():
            try:
                prices = report_prices(asset, pp)["df"]
            except Exception as e:
                ok = False
                print(f"[{asset} prices] FAIL {e}")
        else:
            print(f"[{asset} prices] missing {pp}")
            ok = False
        if npth.exists():
            try:
                news = report_news(asset, npth)["df"]
            except Exception as e:
                ok = False
                print(f"[{asset} news] FAIL {e}")
        else:
            print(f"[{asset} news] missing {npth} (ok until news fetch)")
        if prices is not None:
            coverage(asset, prices, news)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
