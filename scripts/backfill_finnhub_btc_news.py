#!/usr/bin/env python3
"""Backfill BTC-related news from Finnhub company-news into data/raw/btc_news.csv.

Uses FINNHUB_API_KEY from .env. Free tier typically retains ~1y of company news;
crypto market /news is latest-only (not used for history).

  python scripts/backfill_finnhub_btc_news.py
  python scripts/backfill_finnhub_btc_news.py --from 2025-10-01 --to 2026-03-31
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.fetch_news import clean_text, write_news  # noqa: E402
from scripts.load_env import load_env  # noqa: E402

RAW = ROOT / "data" / "raw"
BTC_RE = re.compile(
    r"bitcoin|\bbtc\b|crypto|blockchain|ethereum|\beth\b|coinbase|microstrategy|"
    r"\bmstr\b|stablecoin|etf|binance|mining|hashrate|satoshi|web3|defi",
    re.I,
)
# BTC-adjacent US equities (company-news is NA stocks on free tier)
DEFAULT_SYMBOLS = [
    "MSTR", "COIN", "MARA", "RIOT", "CLSK", "HUT", "BITF", "CIFR", "BTBT", "HIVE",
]


def finnhub_get(path: str, params: dict, token: str, retries: int = 5):
    params = dict(params)
    params["token"] = token
    url = "https://finnhub.io/api/v1" + path + "?" + urlencode(params)
    last: Exception | None = None
    for i in range(retries):
        try:
            req = Request(url, headers={"User-Agent": "kronos-cmaa/1.0"})
            with urlopen(req, timeout=90) as r:
                return json.loads(r.read().decode())
        except HTTPError as e:
            last = e
            if e.code == 429:
                time.sleep(2.0 * (i + 1))
                continue
            if e.code in (401, 403):
                raise SystemExit(f"Finnhub auth failed ({e.code}): check FINNHUB_API_KEY") from e
            time.sleep(1.2 * (i + 1))
        except (URLError, TimeoutError) as e:
            last = e
            time.sleep(1.5 * (i + 1))
    raise SystemExit(f"Finnhub request failed after retries: {last}")


def fetch_range(token: str, symbols: list[str], start: str, end: str, sleep_s: float) -> pd.DataFrame:
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    rows = []
    for sym in symbols:
        cur = start_ts
        while cur <= end_ts:
            nxt = min(cur + pd.Timedelta(days=6), end_ts)
            fr, to = cur.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d")
            data = finnhub_get("/company-news", {"symbol": sym, "from": fr, "to": to}, token)
            if not isinstance(data, list):
                print(f"  {sym} {fr}: unexpected {type(data)}")
            else:
                kept = 0
                for it in data:
                    headline = str(it.get("headline") or "")
                    summary = str(it.get("summary") or "")
                    text = f"{headline}. {summary}".strip(". ")
                    if not BTC_RE.search(text):
                        continue
                    ts = it.get("datetime")
                    if ts is None:
                        continue
                    rows.append({
                        "published_at": pd.to_datetime(int(ts), unit="s", utc=True),
                        "text": clean_text(text),
                        "symbol": sym,
                        "source": "finnhub",
                    })
                    kept += 1
                print(f"  {sym} {fr}->{to}: api={len(data)} kept={kept}")
            cur = nxt + pd.Timedelta(days=1)
            time.sleep(sleep_s)
    if not rows:
        return pd.DataFrame(columns=["published_at", "text"])
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from", dest="date_from", default="2025-10-01")
    ap.add_argument("--to", dest="date_to", default="2026-03-31")
    ap.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    ap.add_argument("--sleep", type=float, default=1.05, help="seconds between API calls")
    ap.add_argument("--out", type=Path, default=RAW / "btc_news.csv")
    a = ap.parse_args()

    load_env()
    token = os.environ.get("FINNHUB_API_KEY", "").strip()
    if not token:
        raise SystemExit("FINNHUB_API_KEY missing in .env")

    # sanity
    q = finnhub_get("/quote", {"symbol": "AAPL"}, token)
    if not isinstance(q, dict) or not q.get("c"):
        raise SystemExit(f"Finnhub quote sanity failed: {q}")
    print(f"Finnhub OK (AAPL last={q.get('c')})")

    symbols = [s.strip().upper() for s in a.symbols.split(",") if s.strip()]
    print(f"Fetching {symbols}  {a.date_from} -> {a.date_to}")
    new = fetch_range(token, symbols, a.date_from, a.date_to, a.sleep)
    print(f"new rows (pre-dedupe): {len(new)}")
    if len(new) == 0:
        print("Nothing to merge.")
        return

    if a.out.exists():
        old = pd.read_csv(a.out)
        old["published_at"] = pd.to_datetime(
            old["published_at"], utc=True, format="ISO8601", errors="coerce"
        )
        # Avoid re-running clean_text on the full corpus (some rows are multi-MB aggregates).
        old = old.dropna(subset=["published_at", "text"])
        old["text"] = old["text"].astype(str)
        new2 = new[["published_at", "text"]].copy()
        merged = pd.concat([old[["published_at", "text"]], new2], ignore_index=True)
        merged = merged.dropna(subset=["published_at", "text"])
        merged = merged[merged["text"].str.len() >= 10]
        merged = merged.sort_values("published_at").drop_duplicates("text").reset_index(drop=True)
        merged["published_at"] = pd.to_datetime(
            merged["published_at"], utc=True, errors="coerce"
        ).dt.floor("s")
        a.out.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(a.out, index=False)
        print(
            f"Wrote {a.out}  rows={len(merged)}  "
            f"{merged['published_at'].iloc[0]} -> {merged['published_at'].iloc[-1]}"
        )
    else:
        write_news(new[["published_at", "text"]], a.out)

    w0 = pd.Timestamp(a.date_from, tz="UTC")
    w1 = pd.Timestamp(a.date_to, tz="UTC") + pd.Timedelta(days=1)
    out = pd.read_csv(a.out)
    out["published_at"] = pd.to_datetime(out["published_at"], utc=True, format="ISO8601")
    in_w = (out["published_at"] >= w0) & (out["published_at"] < w1)
    print("Monthly counts in fetch window (after merge):")
    print(out.loc[in_w].set_index("published_at").resample("MS").size().to_string())


if __name__ == "__main__":
    main()
