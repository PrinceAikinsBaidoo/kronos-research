#!/usr/bin/env python3
"""Fetch raw price CSVs for Kronos-CMAA (P1).

Writes build_cache-compatible files under data/raw/:
  btc_1h.csv   — Binance Vision spot BTCUSDT 1h (free, no key)
  xau_1h.csv   — Dukascopy XAUUSD H1 via dukascopy-node (free; needs Node/npx)

Optional later (news): use scripts/fetch_news.py once sources are chosen.

Examples
  python scripts/fetch_raw_data.py --start 2021-01-01
  python scripts/fetch_raw_data.py --asset BTC
  python scripts/fetch_raw_data.py --asset XAU --start 2021-01-01 --end 2026-10-01
"""
from __future__ import annotations

import argparse
import io
import subprocess
import sys
import zipfile
from datetime import date, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
BINANCE_BASE = "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1h"
BINANCE_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "n_trades", "taker_base", "taker_quote", "ignore",
]


def _get(url: str) -> bytes:
    req = Request(url, headers={"User-Agent": "kronos-cmaa-fetch/1.0"})
    with urlopen(req, timeout=120) as r:
        return r.read()


def month_range(start: date, end: date):
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield y, m
        m += 1
        if m > 12:
            y, m = y + 1, 1


def _open_times_to_utc(ot: pd.Series) -> pd.Series:
    """Binance Vision mixes ms and us epochs across months — detect per value."""
    ot = ot.astype("int64")
    # us if >= 1e14 (year ~1973 in ms would be smaller; 2020-ms ~1.6e12, 2020-us ~1.6e15)
    unit = np.where(ot >= 10**14, "us", "ms")
    # pandas needs a single unit; split and concat
    out = pd.Series(pd.NaT, index=ot.index, dtype="datetime64[ns, UTC]")
    for u in ("ms", "us"):
        mask = (unit == u) if u == "us" else (ot < 10**14)
        if mask.any():
            out.loc[mask] = pd.to_datetime(ot.loc[mask], unit=u, utc=True)
    return out


def fetch_btc(start: date, end: date, out: Path) -> Path:
    frames = []
    missing = []
    for y, m in month_range(start, end):
        name = f"BTCUSDT-1h-{y}-{m:02d}.zip"
        url = f"{BINANCE_BASE}/{name}"
        try:
            raw = _get(url)
        except HTTPError as e:
            if e.code == 404:
                missing.append(name)
                continue
            raise
        except URLError as e:
            raise SystemExit(f"Binance download failed: {e}") from e
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            csv_name = zf.namelist()[0]
            df = pd.read_csv(zf.open(csv_name), header=None, names=BINANCE_COLS)
        df["timestamp"] = _open_times_to_utc(df["open_time"])
        frames.append(df)
        print(f"  + {name}  rows={len(df)}  {df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]}")

    if not frames:
        raise SystemExit("No Binance months downloaded; check --start/--end")
    if missing:
        print(f"  (skipped {len(missing)} missing month file(s), e.g. {missing[0]})")

    df = pd.concat(frames, ignore_index=True)
    out_df = pd.DataFrame({
        "timestamp": df["timestamp"],
        "open": df["open"].astype(float),
        "high": df["high"].astype(float),
        "low": df["low"].astype(float),
        "close": df["close"].astype(float),
        "volume": df["volume"].astype(float),
        "amount": df["quote_volume"].astype(float),
    })
    out_df = out_df.sort_values("timestamp").drop_duplicates("timestamp")
    # Clip to [start, end] inclusive by calendar day in UTC
    t0 = pd.Timestamp(start, tz="UTC")
    t1 = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    out_df = out_df[(out_df["timestamp"] >= t0) & (out_df["timestamp"] < t1)]
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)
    print(f"Wrote {out}  bars={len(out_df)}  "
          f"{out_df['timestamp'].iloc[0]} -> {out_df['timestamp'].iloc[-1]}")
    return out


def _npx_cmd() -> list[str]:
    """Resolve npx on Windows (npx.cmd) so subprocess finds it."""
    for name in ("npx.cmd", "npx"):
        p = Path(r"C:\Program Files\nodejs") / name
        if p.exists():
            return [str(p)]
    return ["npx"]


def fetch_xau(start: date, end: date, out: Path) -> Path:
    """Download XAUUSD H1 via dukascopy-node (npx)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = out.parent / "_dukascopy_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    cmd = _npx_cmd() + [
        "--yes", "dukascopy-node",
        "-i", "xauusd",
        "-from", start.isoformat(),
        "-to", end.isoformat(),
        "-t", "h1",
        "-f", "csv",
        "-v",
        "-fl",
        "-dir", str(tmp_dir),
    ]
    print("  running:", " ".join(cmd))
    try:
        r = subprocess.run(cmd, cwd=ROOT, check=False, text=True, capture_output=True, shell=False)
    except FileNotFoundError as e:
        raise SystemExit(
            "npx/Node.js not found. Install Node LTS, then re-run --asset XAU.\n"
            "Or download XAUUSD H1 manually from Dukascopy Historical Data Export "
            "and save as data/raw/xau_1h.csv with columns timestamp,open,high,low,close,volume."
        ) from e
    if r.returncode != 0:
        sys.stderr.write(r.stdout + "\n" + r.stderr)
        raise SystemExit(f"dukascopy-node failed with code {r.returncode}")

    csvs = sorted(tmp_dir.rglob("*.csv"))
    if not csvs:
        raise SystemExit(f"dukascopy-node produced no CSV under {tmp_dir}")
    # Prefer the newest / largest file
    src = max(csvs, key=lambda p: p.stat().st_size)
    df = pd.read_csv(src)
    df.columns = [c.strip().lower() for c in df.columns]
    # dukascopy-node typically: timestamp, open, high, low, close, volume
    tcol = next((c for c in ("timestamp", "time", "date", "datetime") if c in df.columns), None)
    if tcol is None:
        raise SystemExit(f"Unexpected Dukascopy columns: {list(df.columns)}")
    ts = df[tcol]
    if pd.api.types.is_numeric_dtype(ts):
        # ms epoch
        ts = pd.to_datetime(ts, unit="ms", utc=True)
    else:
        ts = pd.to_datetime(ts, utc=True)
    out_df = pd.DataFrame({
        "timestamp": ts,
        "open": df["open"].astype(float),
        "high": df["high"].astype(float),
        "low": df["low"].astype(float),
        "close": df["close"].astype(float),
        "volume": df["volume"].astype(float) if "volume" in df.columns else 0.0,
        "amount": 0.0,
    })
    out_df = out_df.sort_values("timestamp").drop_duplicates("timestamp")
    out_df.to_csv(out, index=False)
    print(f"Wrote {out}  bars={len(out_df)}  "
          f"{out_df['timestamp'].iloc[0]} -> {out_df['timestamp'].iloc[-1]}  (from {src.name})")
    return out


def parse_day(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--asset", choices=["BTC", "XAU", "ALL"], default="ALL")
    ap.add_argument("--start", default="2021-01-01", help="UTC start date YYYY-MM-DD")
    ap.add_argument("--end", default=date.today().isoformat(), help="UTC end date YYYY-MM-DD")
    ap.add_argument("--out-dir", type=Path, default=RAW)
    a = ap.parse_args()
    start, end = parse_day(a.start), parse_day(a.end)
    if end < start:
        raise SystemExit("--end must be >= --start")
    a.out_dir.mkdir(parents=True, exist_ok=True)

    if a.asset in ("BTC", "ALL"):
        print(f"[BTC] Binance Vision {start} -> {end}")
        fetch_btc(start, end, a.out_dir / "btc_1h.csv")
    if a.asset in ("XAU", "ALL"):
        print(f"[XAU] Dukascopy H1 {start} -> {end}")
        fetch_xau(start, end, a.out_dir / "xau_1h.csv")
    print("Done. Next: fetch news CSVs, then python scripts/qa_raw_data.py")


if __name__ == "__main__":
    main()
