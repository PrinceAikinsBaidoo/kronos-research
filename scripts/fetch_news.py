#!/usr/bin/env python3
"""Build news CSVs for Kronos-CMAA (published_at, text).

BTC: Kaggle dataset imadallal/sentiment-analysis-of-bitcoin-news-2021-2024
     (download first) or a local CSV path via --btc-csv
XAU: Parse.bot ForexFactory get_news_latest + get_calendar (USD / Gold related)
     — latest/calendar oriented; pair with confluence sources later if sparse.

Examples
  kaggle datasets download -d imadallal/sentiment-analysis-of-bitcoin-news-2021-2024 -p data/raw/_news_tmp --unzip
  python scripts/fetch_news.py --btc-from-kaggle-dir data/raw/_news_tmp
  python scripts/fetch_news.py --xau-parsebot
"""
from __future__ import annotations

import argparse
import io
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
from scripts.load_env import load_env  # noqa: E402

RAW = ROOT / "data" / "raw"
# Canonical ForexFactory scraper on Parse.bot marketplace
PARSE_SCRAPER = "0d3aa2e2-80b6-42dc-986a-d7f0845f4deb"
URL_RE = re.compile(r"https?://\S+")
HANDLE_RE = re.compile(r"@\w+")


def clean_text(s: str) -> str:
    s = HANDLE_RE.sub("", URL_RE.sub("", str(s)))
    return re.sub(r"\s+", " ", s).strip()


def write_news(df: pd.DataFrame, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    df = df.dropna(subset=["published_at", "text"])
    df["text"] = df["text"].map(clean_text)
    df = df[df["text"].str.len() >= 10]
    df = df.sort_values("published_at").drop_duplicates("text").reset_index(drop=True)
    df.to_csv(out, index=False)
    print(f"Wrote {out}  rows={len(df)}  "
          f"{df['published_at'].iloc[0] if len(df) else None} -> "
          f"{df['published_at'].iloc[-1] if len(df) else None}")
    return out


def btc_from_kaggle_dir(d: Path, out: Path) -> Path:
    csvs = list(d.rglob("*.csv"))
    if not csvs:
        raise SystemExit(f"No CSV under {d}")
    # Prefer the main sentiments file if present
    src = next((p for p in csvs if "sentiment" in p.name.lower() or "bitcoin" in p.name.lower()), csvs[0])
    print(f"  reading {src}")
    df = pd.read_csv(src)
    cols = {c.lower().strip(): c for c in df.columns}
    # Heuristic column map
    date_c = next((cols[k] for k in ("date", "published_at", "published", "datetime", "time", "pub_date")
                   if k in cols), None)
    text_c = next((cols[k] for k in (
        "text", "title", "headline", "article", "description", "summary", "content",
        "short description", "short_description",
    ) if k in cols), None)
    if date_c is None or text_c is None:
        # try common forked schema
        raise SystemExit(f"Cannot map columns in {src.name}: {list(df.columns)}")
    # Prefer title+description if both exist
    if "title" in cols and "description" in cols:
        text = (df[cols["title"]].astype(str) + ". " + df[cols["description"]].astype(str))
    else:
        text = df[text_c].astype(str)
    out_df = pd.DataFrame({
        "published_at": pd.to_datetime(df[date_c], utc=True, errors="coerce"),
        "text": text,
    })
    return write_news(out_df, out)


def btc_from_csv(path: Path, out: Path) -> Path:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "published_at" not in df.columns:
        for alt in ("date", "datetime", "published", "time"):
            if alt in df.columns:
                df = df.rename(columns={alt: "published_at"})
                break
    if "text" not in df.columns:
        for alt in ("title", "headline", "content", "description"):
            if alt in df.columns:
                df = df.rename(columns={alt: "text"})
                break
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    return write_news(df[["published_at", "text"]], out)


def _parse_get(endpoint: str, key: str, params: dict | None = None) -> dict | list:
    q = f"?{urlencode(params)}" if params else ""
    url = f"https://api.parse.bot/scraper/{PARSE_SCRAPER}/{endpoint}{q}"
    req = Request(url, headers={"X-API-Key": key, "User-Agent": "kronos-cmaa/1.0"})
    with urlopen(req, timeout=90) as r:
        return json.loads(r.read().decode("utf-8"))


def _unwrap(payload):
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    data = payload.get("data", payload)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for k in ("stories", "events", "news", "articles", "items", "results"):
            if isinstance(data.get(k), list):
                return data[k]
    for k in ("stories", "events", "news", "articles"):
        if isinstance(payload.get(k), list):
            return payload[k]
    return []


def xau_from_parsebot(out: Path, calendar_weeks: int = 12) -> Path:
    load_env()
    key = os.environ.get("PARSEBOT", "").strip()
    if not key:
        raise SystemExit("PARSEBOT missing in .env")
    rows = []

    try:
        news = _parse_get("get_news_latest", key)
        items = _unwrap(news)
        for it in items:
            if not isinstance(it, dict):
                continue
            headline = it.get("headline") or it.get("title") or ""
            preview = it.get("preview") or it.get("summary") or ""
            text = f"{headline}. {preview}".strip(". ")
            ts = it.get("date") or it.get("published_at") or it.get("time") or pd.Timestamp.utcnow()
            rows.append({"published_at": ts, "text": text})
        print(f"  get_news_latest -> {len(items)} items")
    except (HTTPError, URLError, json.JSONDecodeError) as e:
        print(f"  get_news_latest failed: {e}")

    # Walk recent weeks: week=jun09.2026 style (Free tier credits are limited).
    today = pd.Timestamp.utcnow().normalize()
    for i in range(calendar_weeks):
        day = today - pd.Timedelta(days=7 * i)
        # ForexFactory week keys are typically the Sunday of that week
        sunday = day - pd.Timedelta(days=(day.dayofweek + 1) % 7)
        week = sunday.strftime("%b%d.%Y").lower()  # oct05.2026
        try:
            cal = _parse_get("get_calendar", key, {"week": week})
            events = _unwrap(cal)
            kept = 0
            for ev in events:
                if not isinstance(ev, dict):
                    continue
                cur = str(ev.get("currency") or ev.get("currency_code") or "").upper()
                title = str(ev.get("name") or ev.get("title") or ev.get("event") or "")
                impact = str(ev.get("impact") or "").lower()
                if cur not in ("USD", "XAU", "GOLD", "ALL") and "gold" not in title.lower():
                    continue
                if impact in ("low", "1", "grey", "gray"):
                    continue
                text = (f"{cur} {title}. impact={impact}. "
                        f"actual={ev.get('actual')} forecast={ev.get('forecast')} "
                        f"previous={ev.get('previous')}")
                date_s = str(ev.get("date") or "")
                # "Sun Oct 4" -> attach year from week sunday
                ts = pd.to_datetime(f"{date_s} {sunday.year}", errors="coerce", utc=True)
                if pd.isna(ts):
                    ts = sunday
                rows.append({"published_at": ts, "text": text})
                kept += 1
            print(f"  get_calendar week={week} events={len(events)} kept={kept}")
        except (HTTPError, URLError, json.JSONDecodeError) as e:
            print(f"  get_calendar week={week} failed: {e}")

    if not rows:
        raise SystemExit("Parse.bot returned no usable XAU/USD news or calendar rows")
    df = pd.DataFrame(rows)
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    return write_news(df[["published_at", "text"]], out)


def xau_from_gdelt(out: Path, start: str = "2021-01-01", end: str | None = None) -> Path:
    """Free GDELT DOC API — gold headlines (CSV mode; JSON often breaks on bad escapes)."""
    end = end or pd.Timestamp.now("UTC").strftime("%Y-%m-%d")
    query = '(gold OR "xauusd" OR "spot gold" OR "gold price") sourcelang:english'
    # Pull month-sized windows to improve coverage within GDELT DOC limits
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    frames = []
    cur = start_ts
    while cur < end_ts:
        nxt = min(cur + pd.DateOffset(months=2), end_ts)
        url = ("https://api.gdeltproject.org/api/v2/doc/doc?"
               + urlencode({
                   "query": query,
                   "mode": "ArtList",
                   "maxrecords": "250",
                   "format": "csv",
                   "sort": "DateDesc",
                   "startdatetime": cur.strftime("%Y%m%d%H%M%S"),
                   "enddatetime": nxt.strftime("%Y%m%d%H%M%S"),
               }))
        req = Request(url, headers={"User-Agent": "kronos-cmaa/1.0"})
        try:
            raw = urlopen(req, timeout=120).read().decode("utf-8", errors="replace")
        except (HTTPError, URLError) as e:
            print(f"  GDELT window {cur.date()}->{nxt.date()} failed: {e}")
            cur = nxt
            continue
        if not raw.strip() or raw.lstrip().startswith("<"):
            print(f"  GDELT window {cur.date()}->{nxt.date()}: empty/html")
            cur = nxt
            continue
        try:
            chunk = pd.read_csv(io.StringIO(raw))
        except Exception as e:
            print(f"  GDELT window {cur.date()}->{nxt.date()} parse fail: {e}")
            cur = nxt
            continue
        frames.append(chunk)
        print(f"  GDELT {cur.date()}->{nxt.date()} rows={len(chunk)}")
        cur = nxt
        time.sleep(8)  # GDELT rate-limits aggressive DOC queries
    if not frames:
        raise SystemExit("GDELT returned no articles")
    arts = pd.concat(frames, ignore_index=True)
    cols = {c.lower(): c for c in arts.columns}
    title_c = cols.get("title") or cols.get("seentitle")
    date_c = cols.get("seendate") or cols.get("date")
    if not title_c or not date_c:
        raise SystemExit(f"Unexpected GDELT columns: {list(arts.columns)}")
    df = pd.DataFrame({
        "published_at": arts[date_c],
        "text": arts[title_c].astype(str),
    })
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    if out.exists():
        old = pd.read_csv(out)
        old["published_at"] = pd.to_datetime(old["published_at"], utc=True, errors="coerce")
        df = pd.concat([old, df[["published_at", "text"]]], ignore_index=True)
    return write_news(df[["published_at", "text"]], out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--btc-from-kaggle-dir", type=Path)
    ap.add_argument("--btc-csv", type=Path)
    ap.add_argument("--xau-parsebot", action="store_true")
    ap.add_argument("--xau-gdelt", action="store_true")
    ap.add_argument("--calendar-weeks", type=int, default=12)
    ap.add_argument("--out-dir", type=Path, default=RAW)
    a = ap.parse_args()
    if not any([a.btc_from_kaggle_dir, a.btc_csv, a.xau_parsebot, a.xau_gdelt]):
        ap.error("Specify a news source flag")
    if a.btc_from_kaggle_dir:
        btc_from_kaggle_dir(a.btc_from_kaggle_dir, a.out_dir / "btc_news.csv")
    elif a.btc_csv:
        btc_from_csv(a.btc_csv, a.out_dir / "btc_news.csv")
    if a.xau_parsebot:
        xau_from_parsebot(a.out_dir / "xau_news.csv", calendar_weeks=a.calendar_weeks)
    if a.xau_gdelt:
        xau_from_gdelt(a.out_dir / "xau_news.csv")


if __name__ == "__main__":
    main()
