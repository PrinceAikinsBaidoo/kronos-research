# Kronos-CMAA research timeline

Living checklist. Update **Status** as gates pass. Visual twin:
open the Cursor canvas `kronos-cmaa-research-timeline.canvas.tsx` beside chat.

Primary metric: **validation AUC-PR**. Explore GPU budget: **18 h / rolling 7 days**.

---

## Progress snapshot

| Phase | Name | Status | Gate (must be true before leaving) |
| --- | --- | --- | --- |
| **P0** | Scaffold & tooling | **Done** | Repo public, train stubs, smoke folder, clean-commit, `.env` |
| **P1** | Raw data + confluence | **In progress** ← you are here | Four CSVs, aligned schema, overlapping UTC window |
| **P2** | Build Kronos-CMAA cache | Pending | Private Kaggle dataset `kronos-cmaa-cache` (`BTC/`, `XAU/`) |
| **P3** | Lock price-only baselines | Pending | `locked/baselines/BTC.json` and `XAU.json` committed |
| **P4** | Smoke on Kaggle | Pending | `000_smoke` → `metrics_BTC_smoke.json` with `status=ok` |
| **P5** | Agent experiment loop | Pending | H000 noise measured; H001–H008 under budgets |
| **P6** | Final human eval | Pending | Frozen config; expanding-window folds + test once (`--final`) |
| **P7** | Methods & disclosure | Pending | Methods section + GOAL open decisions resolved |

---

## P0 — Scaffold & tooling (done)

- [x] GitHub `PrinceAikinsBaidoo/kronos-research`
- [x] `load_cache` / `build_model` in `src/train.py`
- [x] `experiments/000_smoke/`
- [x] Kaggle CLI authenticated
- [x] Clean-commit skill (no Cursor co-author)
- [x] Local `.env` (gitignored); `PARSEBOT` set

---

## P1 — Raw data + confluence (current)

### Data needs

| Need | Primary source | Confluence / cross-check | Cost | Status |
| --- | --- | --- | --- | --- |
| BTC prices 1h | [Binance Vision](https://data.binance.vision/) spot `BTCUSDT/1h` (+ daily for current month) | Optional CryptoCompare bars | Free | **Done** (`btc_1h.csv`, 50 473 bars, **2021-01-01 → 2026-10-05**) |
| XAU prices 1h | Dukascopy H1 (`dukascopy-node`) | Optional second broker/OpenDataBay spot-check | Free | **Done** (`data/raw/xau_1h.csv`, 49 405 bars) |
| BTC news | monstaws 3h + Argus + mouadja + CryptoPulse + GDELT + others (`scripts/backfill_btc_news_window.py`) | CryptoCompare key still empty; GDELT rate-limits leave **H2 2025 / early 2026 thin** | Free | **Backfilled to 2026-10-05** (61 338 rows; dry-run has_text train 1.00 / val 0.57 / test 0.72) |
| XAU news | Parse.bot ForexFactory + GDELT gold | More GDELT windows (rate-limited) | Free credits + DIY | **Partial** (`xau_news.csv`, 2 072 rows; ~1.2% bars w/ text — densify later) |

### Alignment alterations (lock these)

| Field | BTC | XAU | Shared rule |
| --- | --- | --- | --- |
| Asset | BTC/USDT spot | XAUUSD spot | Train separately; no joint batches |
| Bar clock | Binance 1h UTC **open** | Dukascopy H1 UTC (weekend gaps) | All times → UTC; column = bar open |
| OHLCV | + quote volume → `amount` | `amount=0` OK | Match `build_cache` `PRICE_COLS` |
| Gaps | `--regular-grid` (ffill ≤3) | **No** `--regular-grid` | Do not apply crypto fill to gold |
| News | `published_at`, `text` | same | Drop retweets/dupes; clean URLs/@ |
| Align | headline ≤ bar **close** (open+1h) | same | `text_windows()` in `build_cache.py` |
| Coverage | price ∩ news UTC range | same | Trim both modalities to intersection |
| Labels | fwd 24h vol > train q0.90 | same, train-only fit | Only `locked/eval.make_spike_labels` |

### P1 exit checklist

- [x] `data/raw/btc_1h.csv`, `data/raw/xau_1h.csv`
- [x] `data/raw/btc_news.csv` (solid); `data/raw/xau_news.csv` (sparse — OK to proceed on BTC first)
- [x] Schema QA via `python scripts/qa_raw_data.py`
- [x] Coverage report recorded (BTC bars_with_text_24h ≈ 0.89 over full overlap; dry-run has_text train/val/test ≈ 1.00 / 0.57 / 0.72 — val dips on sparse mid-2025→early-2026; densest from 2026-04)
- [ ] Optional: densify XAU news (`python scripts/fetch_news.py --xau-gdelt` with sleeps)
- [ ] Optional: fill thin BTC months (2025-07/08, 2025-11→2026-03) via CryptoCompare key or calmer GDELT retries

**Next concrete step:** re-upload `kronos-cmaa-raw` and rebuild BTC cache on Kaggle (P2).

---

## P2 — Cache build

- [ ] Kaggle/Colab GPU notebook; clone this repo
- [ ] `build_cache.py --asset BTC ... --regular-grid --dry-run` then full
- [ ] `build_cache.py --asset XAU ...` (no `--regular-grid`) dry-run then full
- [ ] Inspect `meta.json` spike/text rates; VERIFY Kronos preprocess notes
- [ ] Upload `BTC/` + `XAU/` as private dataset **`kronos-cmaa-cache`**

---

## P3 — Locked baselines (evaluation reference)

**Purpose:** every CMAA finding is reported as a delta vs this baseline.

**Primary baseline (required):** price-only Kronos — frozen backbone, **no text,
no CMAA**, spike head only. Same cache, splits, and spike labels as CMAA.

```text
python scripts/run_price_only_baseline.py --asset BTC --cache-dir <cache> --out locked/baselines/BTC.json
python scripts/run_price_only_baseline.py --asset XAU --cache-dir <cache> --out locked/baselines/XAU.json
```

See `locked/baselines/README.md` for the JSON schema and “beating the baseline” rule
(must exceed H000 seed noise).

**Secondary baselines** (M2VN / FININ / RiskLabs vs early-fusion / FinBERT-score):
blocked on supervisor confirmation (GOAL open decision #5). Until then, only
price-only Kronos is locked.

- [ ] `locked/baselines/BTC.json`
- [ ] `locked/baselines/XAU.json`
- [ ] Commit baselines; never edit `locked/eval.py`
- [ ] (Optional later) secondary baseline list confirmed with supervisor

---

## P4 — Smoke

```text
python scripts/run_on_kaggle.py --exp 000_smoke --asset BTC --smoke
```

- [ ] `status: ok` in smoke metrics
- [ ] Fix wiring before spending explore budget

---

## P5 — Experiment loop (`queue.json`)

Order: **H000** (noise) → **H001** (reference) → H002–H008 (one change each).

- [ ] H000 seeds 42/43/44
- [ ] H001 BTC + XAU
- [ ] Remaining open hypotheses
- [ ] Update `NOTES.md` / `queue.json` every run
- [ ] Stop on `STOP`, `needs_review.md`, or budget

Driver: `MAX_ITERS=10 bash scripts/loop.sh` (when ready).

---

## P6 — Final eval (human only)

- [ ] Freeze best config from val (steering only until now)
- [ ] Expanding-window folds (`locked/eval.py`, default 6) + embargo
- [ ] Single test evaluation with `--final`
- [ ] Paired report vs locked baseline
- [ ] Confirm significance design with supervisor (GOAL open decision #1)

---

## P7 — Write-up

- [ ] Disclose AI agent for experiment iteration
- [ ] Resolve GOAL.md open decisions 1–7
- [ ] CHRPE before any trader survey (decision #7)

---

## Budgets & honesty rules (always)

- Train ≤ 25 min/run (hard kill 40)
- Explore ≤ 18 GPU-hours / 7 days; `--final` is human-reserved
- Agent never sees test; validation is for steering only
- One change per experiment; log failures too

---

## How we update this file

After each gate: flip checkboxes, set phase **Status** in the snapshot table,
and keep the canvas in sync if phase status changes.
