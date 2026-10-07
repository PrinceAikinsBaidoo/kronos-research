#!/usr/bin/env python3
"""Push a CPU Kaggle kernel that backfills BTC news via GDELT for Jul 2025–Mar 2026.

Uses week-sized windows (GDELT DOC caps at 250/query) + long sleeps/retries.
Output: /kaggle/working/btc_news_gdelt_sparse.csv  (published_at, text)

  python scripts/push_gdelt_sparse_kaggle.py
  python scripts/push_gdelt_sparse_kaggle.py --no-wait
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SLUG = "kronos-cmaa-gdelt-sparse"
POLL_S = 30
MAX_WAIT_MIN = 120

KERNEL = r'''
import io, time, json
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
import pandas as pd

OUT = Path("/kaggle/working/btc_news_gdelt_sparse.csv")
QUERY = "(bitcoin OR BTC) sourcelang:english"
START = pd.Timestamp("2025-07-01", tz="UTC")
END = pd.Timestamp("2026-04-01", tz="UTC")
SLEEP = 45.0
RETRIES = 6
UA = {"User-Agent": "kronos-cmaa-gdelt-sparse/1.0"}

def fetch_window(t0, t1):
    url = "https://api.gdeltproject.org/api/v2/doc/doc?" + urlencode({
        "query": QUERY, "mode": "ArtList", "maxrecords": "250",
        "format": "csv", "sort": "DateAsc",
        "startdatetime": t0.strftime("%Y%m%d%H%M%S"),
        "enddatetime": t1.strftime("%Y%m%d%H%M%S"),
    })
    last = None
    for attempt in range(RETRIES):
        try:
            raw = urlopen(Request(url, headers=UA), timeout=120).read().decode("utf-8", "replace")
            if not raw.strip() or raw.lstrip().startswith("<"):
                print(f"  empty/html {t0.date()} attempt {attempt}", flush=True)
                time.sleep(SLEEP * (attempt + 1))
                continue
            return pd.read_csv(io.StringIO(raw))
        except Exception as e:
            last = e
            wait = SLEEP * (attempt + 1)
            print(f"  fail {t0.date()}->{t1.date()} {e} sleep {wait:.0f}s", flush=True)
            time.sleep(wait)
    print(f"  GIVE UP {t0.date()} last={last}", flush=True)
    return None

frames = []
cur = START
while cur < END:
    nxt = min(cur + pd.Timedelta(days=7), END)
    chunk = fetch_window(cur, nxt)
    if chunk is not None and len(chunk):
        frames.append(chunk)
        print(f"OK {cur.date()}->{nxt.date()} rows={len(chunk)}", flush=True)
    else:
        print(f"MISS {cur.date()}->{nxt.date()}", flush=True)
    cur = nxt
    time.sleep(SLEEP)

if not frames:
    raise SystemExit("No GDELT rows fetched")

arts = pd.concat(frames, ignore_index=True)
cols = {c.lower(): c for c in arts.columns}
title_c = cols.get("title") or cols.get("seentitle")
date_c = cols.get("date") or cols.get("seendate")
df = pd.DataFrame({
    "published_at": pd.to_datetime(arts[date_c], utc=True, format="ISO8601", errors="coerce"),
    "text": arts[title_c].astype(str).str.replace(r"https?://\S+", "", regex=True).str.replace(r"\s+", " ", regex=True).str.strip(),
})
df = df.dropna(subset=["published_at"])
df = df[df["text"].str.len() >= 10]
df = df.sort_values("published_at").drop_duplicates("text").reset_index(drop=True)
df["published_at"] = df["published_at"].dt.floor("s")
df.to_csv(OUT, index=False)
print(f"Wrote {OUT} rows={len(df)} {df.published_at.iloc[0]} -> {df.published_at.iloc[-1]}", flush=True)
print("Monthly:")
print(df.set_index("published_at").resample("MS").size().to_string(), flush=True)
'''


def run(cmd, check=True, capture=False):
    return subprocess.run(cmd, cwd=ROOT, check=check, text=True, capture_output=capture)


def kaggle_user() -> str:
    r = run(["kaggle", "kernels", "list", "--mine", "-p", "1"], check=False, capture=True)
    for line in (r.stdout or "").splitlines():
        if "/" in line and not line.lower().startswith("ref"):
            return line.split()[0].split("/")[0].strip()
    sys.exit("Cannot resolve Kaggle username")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-wait", action="store_true")
    a = ap.parse_args()

    user = kaggle_user()
    d = ROOT / ".kaggle_push_gdelt"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir()
    (d / "gdelt_sparse_kernel.py").write_text(KERNEL.lstrip("\n"), encoding="utf-8")
    meta = {
        "id": f"{user}/{SLUG}",
        "title": SLUG,
        "code_file": "gdelt_sparse_kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
    }
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    ref = f"{user}/{SLUG}"
    print(f"Pushing {ref} (CPU + internet, ~30–60 min)…")
    run(["kaggle", "kernels", "push", "-p", str(d)])
    if a.no_wait:
        print(f"Pushed. Status: kaggle kernels status {ref}")
        return

    t0 = time.time()
    seen = False
    state = "UNKNOWN"
    while True:
        r = run(["kaggle", "kernels", "status", ref], check=False, capture=True)
        out = (r.stdout + r.stderr).upper()
        state = next((k for k in ("COMPLETE", "ERROR", "CANCEL", "RUNNING", "QUEUED") if k in out), "UNKNOWN")
        if state in {"RUNNING", "QUEUED"}:
            seen = True
        elapsed = time.time() - t0
        print(f"  [{int(elapsed // 60):02d}:{int(elapsed % 60):02d}] {state}")
        if state in {"COMPLETE", "ERROR", "CANCEL"} and (seen or elapsed > 90):
            break
        if elapsed > MAX_WAIT_MIN * 60:
            sys.exit("Timed out waiting for GDELT kernel")
        time.sleep(POLL_S)

    out_dir = ROOT / "data" / "tmp_kaggle_news" / "gdelt_sparse"
    out_dir.mkdir(parents=True, exist_ok=True)
    run(["kaggle", "kernels", "output", ref, "-p", str(out_dir)], check=False)
    print(f"Downloaded to {out_dir}")
    if state != "COMPLETE":
        sys.exit(f"Kernel finished as {state}")


if __name__ == "__main__":
    main()
