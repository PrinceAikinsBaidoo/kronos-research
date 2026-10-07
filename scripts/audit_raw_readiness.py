#!/usr/bin/env python3
"""Readiness audit for Kronos-CMAA raw BTC/XAU (no cache build)."""
from __future__ import annotations

import numpy as np
import pandas as pd

RAW = "data/raw"


def load_btc():
    news = pd.read_csv(f"{RAW}/btc_news.csv")
    news["published_at"] = pd.to_datetime(news["published_at"], utc=True, format="ISO8601")
    px = pd.read_csv(f"{RAW}/btc_1h.csv")
    px["timestamp"] = pd.to_datetime(px["timestamp"], utc=True)
    return px, news


def bar_has_text(px: pd.DataFrame, news: pd.DataFrame, hours: int = 24) -> pd.Series:
    pubs = news["published_at"].sort_values().to_numpy().astype("datetime64[ns]")
    close = (px["timestamp"] + pd.Timedelta(hours=1)).to_numpy().astype("datetime64[ns]")
    win = np.timedelta64(hours, "h")
    lo = np.searchsorted(pubs, close - win, side="right")
    hi = np.searchsorted(pubs, close, side="right")
    return pd.Series(hi > lo, index=px.index)


def main():
    px, news = load_btc()
    has = bar_has_text(px, news)
    px = px.assign(has=has)

    print("=== BTC spans ===")
    print(f"prices {len(px)}  {px.timestamp.min()} -> {px.timestamp.max()}")
    print(f"news   {len(news)}  {news.published_at.min()} -> {news.published_at.max()}")
    t0 = max(px.timestamp.min(), news.published_at.min())
    t1 = min(px.timestamp.max(), news.published_at.max())
    print(f"intersection {t0} -> {t1}")
    print(f"bars_with_text_24h overall {float(has.mean()):.3f}")

    print("\n=== Regime coverage ===")
    for label, a, b in [
        ("2021-2024", "2021-01-01", "2025-01-01"),
        ("2025 H1", "2025-01-01", "2025-07-01"),
        ("Jul25-Mar26 (was sparse)", "2025-07-01", "2026-04-01"),
        ("Apr26+", "2026-04-01", "2027-01-01"),
    ]:
        m = (px.timestamp >= a) & (px.timestamp < b)
        print(f"  {label}: {float(px.loc[m, 'has'].mean()):.3f}  bars={int(m.sum())}")

    print("\n=== Monthly (2025-06+) news count + % bars w/ text ===")
    sub = px[px.timestamp >= "2025-06-01"].copy()
    sub["ym"] = sub.timestamp.dt.to_period("M")
    cov = sub.groupby("ym")["has"].agg(pct="mean", bars="count")
    nc = news[news.published_at >= "2025-06-01"].set_index("published_at").resample("MS").size()
    nc.index = nc.index.to_period("M")
    rep = cov.join(nc.rename("n_news"), how="left").fillna(0)
    rep["pct"] = (rep["pct"] * 100).round(1)
    print(rep.to_string())

    lens = news["text"].str.len()
    print("\n=== BTC text quality ===")
    print(f"rows={len(news)} empty={int((lens < 10).sum())}")
    print(f"len p10/p50/p90 = {int(lens.quantile(0.1))}/{int(lens.median())}/{int(lens.quantile(0.9))}")
    print(f"short<40={float((lens < 40).mean()):.3f}  <100={float((lens < 100).mean()):.3f}")
    print(f">5k chars={int((lens > 5000).sum())}  >50k={int((lens > 50000).sum())}")
    # GDELT-era sample
    sw = news[(news.published_at >= "2025-07-01") & (news.published_at < "2026-04-01")]
    print(f"Jul25-Mar26 news rows={len(sw)}")
    print("samples:")
    for t in sw.sort_values("published_at")["text"].head(4):
        print(" -", str(t)[:110])

    d = px.timestamp.diff()
    print("\n=== BTC price integrity ===")
    print(f"median step={d.median()}  gaps>2h={int((d > pd.Timedelta('2h')).sum())}  max={d.max()}")
    ohlc = px[["open", "high", "low", "close"]]
    print(f"nan_ohlc={int(ohlc.isna().any(axis=1).sum())}  high<low={int((px.high < px.low).sum())}")

    # chronological proxy
    L = 512
    idx = np.arange(L - 1, len(px))
    N = len(idx)
    tr, va = int(0.7 * N), int(0.85 * N)

    def rate(sl):
        return float(px.has.iloc[idx[sl]].mean())

    print("\n=== Approx chrono has_text (L=512 bars) ===")
    print(f"train={rate(slice(0, tr)):.3f}  val={rate(slice(tr, va)):.3f}  test={rate(slice(va, None)):.3f}")

    # XAU
    xau_n = pd.read_csv(f"{RAW}/xau_news.csv")
    xau_n["published_at"] = pd.to_datetime(xau_n["published_at"], utc=True, format="ISO8601", errors="coerce")
    xau_p = pd.read_csv(f"{RAW}/xau_1h.csv")
    xau_p["timestamp"] = pd.to_datetime(xau_p["timestamp"], utc=True)
    xhas = bar_has_text(xau_p, xau_n.dropna(subset=["published_at"]))
    print("\n=== XAU ===")
    print(f"prices {len(xau_p)}  {xau_p.timestamp.min()} -> {xau_p.timestamp.max()}")
    print(f"news   {len(xau_n)}  {xau_n.published_at.min()} -> {xau_n.published_at.max()}")
    print(f"bars_with_text_24h={float(xhas.mean()):.3f}")

    print("\n=== Readiness verdict (BTC primary) ===")
    sparse = float(px.loc[(px.timestamp >= "2025-07-01") & (px.timestamp < "2026-04-01"), "has"].mean())
    overall = float(has.mean())
    ok_price = int((d > pd.Timedelta("2h")).sum()) <= 10 and int(ohlc.isna().any(axis=1).sum()) == 0
    ok_cov = overall >= 0.85 and sparse >= 0.70
    print(f"price_ok={ok_price}  coverage_ok={ok_cov} (overall={overall:.3f}, Jul25-Mar26={sparse:.3f})")
    if ok_price and ok_cov:
        print("BTC: READY for multimodal experiment (with caveats: mixed headline/fulltext sources).")
    elif ok_price:
        print("BTC: prices ready; news coverage still weak in places — densify or clip before final cache.")
    else:
        print("BTC: not ready — fix prices first.")
    print("XAU: NOT ready for same claim (news ~1% coverage).")


if __name__ == "__main__":
    main()
