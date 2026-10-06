#!/usr/bin/env python3
"""Backfill BTC news for mid-2025 → Oct 2026 from free local/API sources.

Merges into data/raw/btc_news.csv (published_at, text). Does not overwrite
pre-window rows. Sources under data/tmp_kaggle_news/ are optional if present.
"""
from __future__ import annotations

import argparse
import io
import json
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

RAW = ROOT / "data" / "raw"
TMP = ROOT / "data" / "tmp_kaggle_news"
URL_RE = re.compile(r"https?://\S+")
BTC_RE = re.compile(
    r"bitcoin|\bbtc\b|crypto|blockchain|ethereum|\beth\b|coinbase|binance|stablecoin",
    re.I,
)


def _ts(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.replace(r"Z$", "", regex=True)
    return pd.to_datetime(s, utc=True, errors="coerce")


def _frame(published_at, text, source: str) -> pd.DataFrame:
    df = pd.DataFrame({"published_at": published_at, "text": text, "source": source})
    df["published_at"] = _ts(df["published_at"])
    df["text"] = df["text"].map(lambda x: clean_text(x) if pd.notna(x) else "")
    df = df.dropna(subset=["published_at"])
    df = df[df["text"].str.len() >= 10]
    return df


def from_argus(p: Path) -> pd.DataFrame:
    df = pd.read_parquet(p)
    return _frame(df["published_at"], df["title"], "argus")


def from_mouadja(p: Path) -> pd.DataFrame:
    df = pd.read_parquet(p)
    text = (df["HEADLINE"].fillna("") + ". " + df["SUMMARY"].fillna("")).str.strip(". ")
    keep = text.str.contains(BTC_RE)
    df = df.loc[keep]
    return _frame(df["DATETIME"], text.loc[keep], "mouadja")


def from_cryptopulse(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    text = (df["title"].fillna("") + ". " + df["description"].fillna("")).str.strip(". ")
    return _frame(df["createdAt"], text, "cryptopulse")


def from_itsananya(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    text = (df["title"].fillna("") + ". " + df["description"].fillna("")).str.strip(". ")
    return _frame(df["createdAt"], text, "itsananya")


def from_newsapi_kaggle(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    text = (df["title"].fillna("") + ". " + df["description"].fillna("")).str.strip(". ")
    return _frame(df["publishedAt"], text, "newsapi_kaggle")


def from_finnews(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    text = (df["title"].fillna("") + ". " + df["summary"].fillna("")).str.strip(". ")
    keep = text.str.contains(BTC_RE) | df["main_ticker"].fillna("").astype(str).str.contains(
        r"BTC|BITCOIN", case=False, regex=True
    )
    df = df.loc[keep]
    return _frame(df["time_published"], text.loc[keep], "finnews2026")


def from_sample_yevent(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    # One headline per news_hash (multi-ticker duplicates)
    if "news_hash" in df.columns:
        df = df.drop_duplicates("news_hash")
    return _frame(df["published_at"], df["headline"], "yevent_sample")


def from_cryptocurrency_cv() -> pd.DataFrame:
    rows = []
    for path, params in (
        ("/api/bitcoin", {"limit": 100}),
        ("/api/search", {"q": "bitcoin", "limit": 100}),
    ):
        url = "https://cryptocurrency.cv" + path + "?" + urlencode(params)
        try:
            req = Request(url, headers={"User-Agent": "kronos-cmaa/1.0"})
            with urlopen(req, timeout=60) as r:
                data = json.loads(r.read().decode())
        except (HTTPError, URLError, json.JSONDecodeError) as e:
            print(f"  cryptocurrency.cv {path} failed: {e}")
            continue
        for a in data.get("articles") or []:
            title = a.get("title") or ""
            desc = a.get("description") or ""
            rows.append({
                "published_at": a.get("pubDate") or a.get("published_at"),
                "text": f"{title}. {desc}".strip(". "),
            })
    if not rows:
        return pd.DataFrame(columns=["published_at", "text", "source"])
    df = pd.DataFrame(rows)
    return _frame(df["published_at"], df["text"], "cryptocurrency_cv")


def from_gdelt(start: str, end: str, sleep_s: float = 10.0) -> pd.DataFrame:
    query = "(bitcoin OR BTC) sourcelang:english"
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    frames = []
    cur = start_ts
    while cur < end_ts:
        nxt = min(cur + pd.DateOffset(months=1), end_ts)
        url = (
            "https://api.gdeltproject.org/api/v2/doc/doc?"
            + urlencode({
                "query": query,
                "mode": "ArtList",
                "maxrecords": "250",
                "format": "csv",
                "sort": "DateAsc",
                "startdatetime": cur.strftime("%Y%m%d%H%M%S"),
                "enddatetime": nxt.strftime("%Y%m%d%H%M%S"),
            })
        )
        try:
            raw = urlopen(Request(url, headers={"User-Agent": "kronos-cmaa/1.0"}), timeout=120)
            text = raw.read().decode("utf-8", errors="replace")
        except (HTTPError, URLError) as e:
            print(f"  GDELT {cur.date()}->{nxt.date()} failed: {e}")
            cur = nxt
            time.sleep(sleep_s)
            continue
        if not text.strip() or text.lstrip().startswith("<"):
            print(f"  GDELT {cur.date()}->{nxt.date()}: empty/html")
            cur = nxt
            time.sleep(sleep_s)
            continue
        try:
            chunk = pd.read_csv(io.StringIO(text))
        except Exception as e:
            print(f"  GDELT {cur.date()}->{nxt.date()} parse fail: {e}")
            cur = nxt
            time.sleep(sleep_s)
            continue
        frames.append(chunk)
        print(f"  GDELT {cur.date()}->{nxt.date()} rows={len(chunk)}")
        cur = nxt
        time.sleep(sleep_s)
    if not frames:
        return pd.DataFrame(columns=["published_at", "text", "source"])
    arts = pd.concat(frames, ignore_index=True)
    cols = {c.lower(): c for c in arts.columns}
    title_c = cols.get("title") or cols.get("seentitle")
    date_c = cols.get("seendate") or cols.get("date")
    if not title_c or not date_c:
        print(f"  GDELT unexpected cols: {list(arts.columns)}")
        return pd.DataFrame(columns=["published_at", "text", "source"])
    return _frame(arts[date_c], arts[title_c], "gdelt")


def load_optional(label: str, path: Path, loader) -> pd.DataFrame:
    if not path.exists():
        print(f"  skip {label}: missing {path}")
        return pd.DataFrame(columns=["published_at", "text", "source"])
    df = loader(path)
    print(f"  {label}: {len(df)} rows  "
          f"{df['published_at'].min() if len(df) else None} -> "
          f"{df['published_at'].max() if len(df) else None}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--window-start", default="2025-06-05")
    ap.add_argument("--window-end", default="2026-10-06")
    ap.add_argument("--gdelt", action="store_true", help="Also pull GDELT for sparse months")
    ap.add_argument("--gdelt-start", default="2025-06-01")
    ap.add_argument("--gdelt-end", default="2026-04-01")
    ap.add_argument("--out", type=Path, default=RAW / "btc_news.csv")
    a = ap.parse_args()

    w0 = pd.Timestamp(a.window_start, tz="UTC")
    w1 = pd.Timestamp(a.window_end, tz="UTC") + pd.Timedelta(days=1)

    chunks = [
        load_optional("argus", TMP / "argus" / "raw_news.parquet", from_argus),
        load_optional("mouadja", TMP / "mouadja" / "bitcoin-news.parquet", from_mouadja),
        load_optional("cryptopulse", TMP / "crypto_news_dataset.csv", from_cryptopulse),
        load_optional("itsananya", TMP / "itsananya" / "crypto_news.csv", from_itsananya),
        load_optional("newsapi_kaggle", TMP / "bitcoin.csv", from_newsapi_kaggle),
        load_optional(
            "finnews2026",
            TMP / "finnews" / "financial_news_final_2026.csv",
            from_finnews,
        ),
        load_optional("yevent_sample", TMP / "sample_20260528.csv", from_sample_yevent),
    ]
    print("  cryptocurrency.cv live…")
    cv = from_cryptocurrency_cv()
    print(f"  cryptocurrency_cv: {len(cv)} rows")
    chunks.append(cv)

    if a.gdelt:
        print("  GDELT DOC (slow; rate-limited)…")
        chunks.append(from_gdelt(a.gdelt_start, a.gdelt_end))

    new = pd.concat([c for c in chunks if len(c)], ignore_index=True)
    new = new[(new["published_at"] >= w0) & (new["published_at"] < w1)]
    print(f"  new in window (pre-dedupe): {len(new)}")

    if a.out.exists():
        old = pd.read_csv(a.out)
        old["published_at"] = _ts(old["published_at"])
        old["text"] = old["text"].map(clean_text)
        old["source"] = "existing"
        merged = pd.concat([old, new[["published_at", "text", "source"]]], ignore_index=True)
    else:
        merged = new

    # Deduplicate on cleaned text; keep earliest timestamp
    merged = merged.sort_values("published_at")
    merged = merged.drop_duplicates("text", keep="first")
    out_df = merged[["published_at", "text"]].reset_index(drop=True)
    write_news(out_df, a.out)

    # Coverage report in window
    ts = out_df["published_at"]
    in_w = (ts >= w0) & (ts < w1)
    print("Monthly counts in backfill window:")
    print(out_df.loc[in_w].set_index("published_at").resample("MS").size().to_string())


if __name__ == "__main__":
    main()
