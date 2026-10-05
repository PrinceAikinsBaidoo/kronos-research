#!/usr/bin/env python3
"""Build the per-asset cache for Kronos-CMAA. Run ONCE per asset on a GPU notebook.

Why a cache: Kronos and FinBERT are frozen, so tokens and text states never change between
experiments. Computing them once makes each agent experiment a few minutes of adapter training.

Inputs (CSV, all timestamps UTC)
  --prices  columns (case-insensitive): timestamp | open_time | datetime, open, high, low, close,
            volume, [amount]. Timestamps are bar OPEN times. Missing volume/amount become zeros,
            which matches how the Kronos predictor treats them.
  --news    columns: published_at, text. One headline or post per row.

Output folder <out>/<ASSET>/
  meta.json           settings, split bounds, label threshold, z-score stats, class balance
  s1_ids.npy, s2_ids.npy   (N, L) int16  Kronos BSQ tokens of the L-bar window ending at each sample
  y.npy               (N,) int8   spike label (1 = high forward volatility)
  mom.npy, sent.npy   (N,) float32 z-scored past momentum and text sentiment (train stats only)
  text_idx.npy        (N,) int32  row in the text arrays below, -1 = no news in the window
  text_states.npy     (U, T, 768) float16  FinBERT last-layer TOKEN states, one row per unique text
  text_mask.npy       (U, T) bool
  text_sent_raw.npy   (U,) float32  P(positive) - P(negative) from FinBERT
  bar_close_ns.npy, text_last_pub_ns.npy  (N,) int64  audit: newest headline time <= bar close time
Samples are in chronological order. Split with locked.eval.chronological_split(N); meta.json lists
the bounds. `embargo` in meta.json = number of samples to drop at the end of the train and val
slices (labels look `horizon` bars ahead).

Not yet run end to end on real data. Check the two assumptions marked VERIFY against the Kronos repo.
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from locked.eval import chronological_split, make_spike_labels  # noqa: E402

PRICE_COLS = ["open", "high", "low", "close", "volume", "amount"]
D_TEXT = 768
KRONOS_REPO = "https://github.com/shiyu-coder/Kronos.git"
KRONOS_COMMIT = "67b630e67f6a18c9e9be918d9b4337c960db1e9a"  # from the proposal; VERIFY it exists


# ----------------------------------------------------------------------------- loading
def load_prices(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    tcol = next((c for c in ("timestamp", "timestamps", "open_time", "datetime", "date", "time")
                 if c in df.columns), None)
    if tcol is None:
        raise SystemExit("prices CSV needs a timestamp column")
    ts = df[tcol]
    if pd.api.types.is_numeric_dtype(ts):
        ts = pd.to_datetime(ts, unit="ms" if ts.iloc[0] > 1e11 else "s", utc=True)
    else:
        ts = pd.to_datetime(ts, utc=True)
    df["ts"] = ts
    for c in ("open", "high", "low", "close"):
        if c not in df.columns:
            raise SystemExit(f"prices CSV is missing '{c}'")
    if "volume" not in df.columns:
        df["volume"] = 0.0
    if "amount" not in df.columns:
        alt = next((c for c in ("quote_volume", "quote_asset_volume") if c in df.columns), None)
        df["amount"] = df[alt] if alt else 0.0
    df = df.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
    return df[["ts"] + PRICE_COLS]


URL_RE = re.compile(r"https?://\S+")
HANDLE_RE = re.compile(r"@\w+")


def clean_text(s):
    s = HANDLE_RE.sub("", URL_RE.sub("", s))  # strip links and usernames (anonymisation)
    return re.sub(r"\s+", " ", s).strip().lower()


def load_news(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df[~df["text"].astype(str).str.match(r"^\s*rt\b", case=False)]  # drop retweets
    out = pd.DataFrame({"pub": pd.to_datetime(df["published_at"], utc=True),
                        "text": df["text"].astype(str).map(clean_text)})
    out = out[out["text"].str.len() >= 10].sort_values("pub")
    return out.drop_duplicates("text").reset_index(drop=True)  # keeps the earliest copy


# ----------------------------------------------------------------------------- pure helpers
def to_ns(series):
    return (series.dt.tz_convert("UTC").dt.tz_localize(None).values
            .astype("datetime64[ns]").astype("int64"))


def normalize_windows(w):
    """Per-window z-score then clip to [-5, 5]. w: (B, L, 6).
    VERIFY against KronosPredictor's preprocessing (population std, eps 1e-5, clip 5)."""
    mu = w.mean(axis=1, keepdims=True)
    sd = w.std(axis=1, keepdims=True)
    return np.clip((w - mu) / (sd + 1e-5), -5, 5).astype(np.float32)


def text_windows(close_ns, pub_ns, window_ns, k):
    """For each bar-close time, the span [start, hi) of the newest <= k headlines published
    in (close - window, close]. Returns (unique spans (U, 2), text_idx (N,) with -1 = none)."""
    hi = np.searchsorted(pub_ns, close_ns, side="right")
    lo = np.searchsorted(pub_ns, close_ns - window_ns, side="right")
    start = np.maximum(lo, hi - k)
    has = hi > start
    text_idx = np.full(len(close_ns), -1, dtype=np.int32)
    if not has.any():
        return np.zeros((0, 2), dtype=np.int64), text_idx, hi
    spans = np.stack([start, hi], axis=1)[has]
    uniq, inv = np.unique(spans, axis=0, return_inverse=True)
    text_idx[has] = inv.reshape(-1)
    assert (pub_ns[hi[has] - 1] <= close_ns[has]).all(), "look-ahead: headline after bar close"
    return uniq, text_idx, hi


# ----------------------------------------------------------------------------- model wrappers
def ensure_kronos(path):
    p = Path(path)
    if not (p / "model").exists():
        subprocess.run(["git", "clone", "--quiet", KRONOS_REPO, str(p)], check=True)
        r = subprocess.run(["git", "checkout", "--quiet", KRONOS_COMMIT], cwd=p)
        if r.returncode:
            print("WARNING: could not check out the pinned Kronos commit; using the default branch")
    return str(p)


def make_kronos_encoder(kronos_dir, tokenizer_id, device):
    import torch
    sys.path.insert(0, ensure_kronos(kronos_dir))
    from model import KronosTokenizer  # from the Kronos repo
    tok = KronosTokenizer.from_pretrained(tokenizer_id).to(device).eval()

    @torch.no_grad()
    def encode(x):
        z = tok.encode(torch.from_numpy(x).to(device), half=True)  # (s1_ids, s2_ids), each (B, L)
        return z[0].cpu().numpy().astype(np.int16), z[1].cpu().numpy().astype(np.int16)
    return encode


def make_finbert_embedder(model_id, device, max_tokens):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    m = AutoModelForSequenceClassification.from_pretrained(model_id).to(device).eval()
    if device == "cuda":
        m.half()
    labels = {int(k): str(v).lower() for k, v in m.config.id2label.items()}
    pos = next(k for k, v in labels.items() if v == "positive")
    neg = next(k for k, v in labels.items() if v == "negative")

    @torch.no_grad()
    def embed(texts):
        enc = tok(texts, padding="max_length", truncation=True, max_length=max_tokens,
                  return_tensors="pt").to(device)
        out = m(**enc, output_hidden_states=True)
        p = torch.softmax(out.logits.float(), dim=-1)
        return (out.hidden_states[-1].cpu().numpy().astype(np.float16),
                enc["attention_mask"].cpu().numpy().astype(bool),
                (p[:, pos] - p[:, neg]).cpu().numpy().astype(np.float32))
    return embed


# ----------------------------------------------------------------------------- main build
def build(asset, prices, news, a, encode_fn, embed_fn, out_root):
    L, H = a.lookback, a.horizon
    assert L > a.mom_bars, "lookback must exceed mom_bars"
    ohl = prices[PRICE_COLS].to_numpy(dtype=np.float64)
    ts_ns = to_ns(prices["ts"])
    bar_ns = int(np.median(np.diff(ts_ns)))
    bad = np.zeros(len(prices), dtype=bool)

    if a.regular_grid:  # proposal rule: forward-fill up to 3 bars, otherwise exclude
        full = pd.date_range(prices["ts"].iloc[0], prices["ts"].iloc[-1], freq=pd.Timedelta(bar_ns, unit="ns"))
        g = prices.set_index("ts").reindex(full)
        g[["open", "high", "low", "close"]] = g[["open", "high", "low", "close"]].ffill(limit=3)
        g[["volume", "amount"]] = g[["volume", "amount"]].fillna(0.0)
        bad = g["close"].isna().to_numpy()
        ohl = g[PRICE_COLS].ffill().bfill().to_numpy(dtype=np.float64)  # filled for computation only
        ts_ns = to_ns(pd.Series(full))
    n = len(ohl)

    csum = np.concatenate([[0], np.cumsum(bad)])
    t = np.arange(L - 1, n - H)
    ok = ((csum[t + 1] - csum[t - L + 1]) == 0) & ((csum[t + H + 1] - csum[t + 1]) == 0)
    t = t[ok][:: a.stride]
    N = len(t)
    if N < 2000:
        print(f"WARNING: only {N} samples; training will be unstable")

    tr, va, te = chronological_split(N)
    train_last_bar = int(t[tr.stop - 1])
    fit_end = train_last_bar - H + 1  # label threshold may only see bars up to the end of train
    y_all, thr = make_spike_labels(ohl[:, 3], H, a.spike_quantile, fit_end=fit_end)
    y = y_all[t]
    assert (y >= 0).all()
    close = ohl[:, 3]
    mom_raw = np.log(close[t] / close[t - a.mom_bars]).astype(np.float32)

    # ---- text windows (look-ahead safe: headline time <= bar CLOSE time)
    close_ns = ts_ns[t] + bar_ns
    pub_ns = to_ns(news["pub"]) if news is not None and len(news) else np.zeros(0, dtype=np.int64)
    uniq, text_idx, hi = text_windows(close_ns, pub_ns, int(a.text_window_hours * 3600 * 10**9),
                                      a.max_headlines)
    U = len(uniq)
    has = text_idx >= 0
    last_pub = np.where(has, pub_ns[np.maximum(hi - 1, 0)] if len(pub_ns) else 0, 0).astype(np.int64)
    est_gb = U * a.max_text_tokens * D_TEXT * 2 / 1e9

    def frac(arr, sl):
        return float(np.mean(arr[sl]))
    print(f"[{asset}] samples N={N}  unique texts U={U}  estimated text cache {est_gb:.2f} GB")
    print(f"  spike rate  train {frac(y, tr):.3f}  val {frac(y, va):.3f}  test {frac(y, te):.3f}")
    print(f"  has text    train {frac(has, tr):.3f}  val {frac(has, va):.3f}  test {frac(has, te):.3f}")
    if a.dry_run:
        return
    if est_gb > a.max_gb:
        sys.exit(f"Text cache would be {est_gb:.1f} GB (> --max-gb {a.max_gb}). "
                 "Lower --max-text-tokens, --max-headlines, or raise --stride.")

    out = Path(out_root) / asset
    out.mkdir(parents=True, exist_ok=True)

    # ---- price tokens
    view = np.lib.stride_tricks.sliding_window_view(ohl, L, axis=0)  # (n-L+1, 6, L)
    s1 = np.zeros((N, L), dtype=np.int16)
    s2 = np.zeros((N, L), dtype=np.int16)
    for i in range(0, N, a.batch):
        idx = t[i:i + a.batch] - (L - 1)
        w = np.transpose(view[idx], (0, 2, 1))
        s1[i:i + len(idx)], s2[i:i + len(idx)] = encode_fn(normalize_windows(w))

    # ---- FinBERT token states, one row per unique text, streamed to disk
    T = a.max_text_tokens
    if U:
        states = np.lib.format.open_memmap(out / "text_states.npy", mode="w+", dtype=np.float16,
                                           shape=(U, T, D_TEXT))
    else:
        states = None
    mask = np.zeros((U, T), dtype=bool)
    sent_u = np.zeros(U, dtype=np.float32)
    texts_all = news["text"].tolist() if U else []
    for i in range(0, U, a.text_batch):
        chunk = [" [SEP] ".join(texts_all[s:h][::-1]) for s, h in uniq[i:i + a.text_batch]]  # newest first
        hid, m, s = embed_fn(chunk)
        states[i:i + len(chunk)] = hid
        mask[i:i + len(chunk)] = m
        sent_u[i:i + len(chunk)] = s
    if states is not None:
        states.flush()
    else:
        np.save(out / "text_states.npy", np.zeros((0, T, D_TEXT), dtype=np.float16))

    # ---- z-score with TRAIN statistics only
    sent_raw = np.zeros(N, dtype=np.float32)
    sent_raw[has] = sent_u[text_idx[has]]
    trh = has[:tr.stop]
    s_mu, s_sd = (float(sent_raw[:tr.stop][trh].mean()), float(sent_raw[:tr.stop][trh].std() + 1e-8)) if trh.any() else (0.0, 1.0)
    m_mu, m_sd = float(mom_raw[:tr.stop].mean()), float(mom_raw[:tr.stop].std() + 1e-8)
    sent = np.where(has, (sent_raw - s_mu) / s_sd, 0.0).astype(np.float32)
    mom = ((mom_raw - m_mu) / m_sd).astype(np.float32)

    for name, arr in {"s1_ids": s1, "s2_ids": s2, "y": y.astype(np.int8), "mom": mom, "sent": sent,
                      "text_idx": text_idx, "text_mask": mask, "text_sent_raw": sent_u,
                      "bar_close_ns": close_ns.astype(np.int64), "text_last_pub_ns": last_pub}.items():
        np.save(out / f"{name}.npy", arr)

    meta = {"asset": asset, "N": N, "U": U, "lookback": L, "horizon": H, "embargo": H,
            "spike_quantile": a.spike_quantile, "spike_threshold": thr, "mom_bars": a.mom_bars,
            "text_window_hours": a.text_window_hours, "max_headlines": a.max_headlines,
            "max_text_tokens": T, "d_text": D_TEXT, "stride": a.stride, "regular_grid": a.regular_grid,
            "bar_seconds": bar_ns / 1e9,
            "splits": {"train": [tr.start, tr.stop], "val": [va.start, va.stop], "test": [te.start, te.stop]},
            "spike_rate": {"train": frac(y, tr), "val": frac(y, va), "test": frac(y, te)},
            "has_text_rate": {"train": frac(has, tr), "val": frac(has, va), "test": frac(has, te)},
            "mom_stats": [m_mu, m_sd], "sent_stats": [s_mu, s_sd],
            "tokenizer_id": a.tokenizer_id, "finbert_id": a.finbert_id, "kronos_commit": KRONOS_COMMIT,
            "first_bar_utc": str(pd.Timestamp(int(ts_ns[t[0]]), unit="ns")),
            "last_bar_utc": str(pd.Timestamp(int(ts_ns[t[-1]]), unit="ns"))}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote cache to {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--asset", required=True, choices=["BTC", "XAU"])
    ap.add_argument("--prices", required=True)
    ap.add_argument("--news", required=True)
    ap.add_argument("--out", default="cache")
    ap.add_argument("--kronos-dir", default="Kronos")
    ap.add_argument("--tokenizer-id", default="NeoQuasar/Kronos-Tokenizer-base")
    ap.add_argument("--finbert-id", default="ProsusAI/finbert")
    ap.add_argument("--lookback", type=int, default=256, help="bars per window (Kronos max context 512)")
    ap.add_argument("--horizon", type=int, default=24, help="spike label horizon in bars")
    ap.add_argument("--spike-quantile", type=float, default=0.90)
    ap.add_argument("--mom-bars", type=int, default=24)
    ap.add_argument("--text-window-hours", type=float, default=24)
    ap.add_argument("--max-headlines", type=int, default=8)
    ap.add_argument("--max-text-tokens", type=int, default=128)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--text-batch", type=int, default=128)
    ap.add_argument("--regular-grid", action="store_true",
                    help="ffill up to 3 bars and drop windows touching longer gaps (crypto). "
                         "Leave off for XAUUSD, whose weekend closures would remove nearly every window.")
    ap.add_argument("--max-gb", type=float, default=8.0)
    ap.add_argument("--dry-run", action="store_true", help="report sizes and rates, load no models")
    a = ap.parse_args()

    prices, news = load_prices(a.prices), load_news(a.news)
    print(f"prices: {len(prices)} bars, news: {len(news)} unique items")
    enc = emb = None
    if not a.dry_run:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        if dev == "cpu":
            print("WARNING: no GPU found; this will be very slow")
        enc = make_kronos_encoder(a.kronos_dir, a.tokenizer_id, dev)
        emb = make_finbert_embedder(a.finbert_id, dev, a.max_text_tokens)
    build(a.asset, prices, news, a, enc, emb, a.out)


if __name__ == "__main__":
    main()
