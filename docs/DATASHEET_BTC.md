# Datasheet: Kronos-CMAA BTC multimodal corpus

Gebru et al.–style datasheet for the **BTC** study used in Kronos-CMAA.
XAU is **out of scope** for the primary paper claim until news coverage matches BTC.

## 1. Motivation

- **For what purpose was the dataset created?**  
  Support a methods-first CS/fintech study: news-conditioned volatility-spike prediction with frozen Kronos (price) + FinBERT (text) and a trainable Cross-Modal Attention Adapter (CMAA). Primary metric: validation AUC-PR vs a locked price-only Kronos baseline.

- **Who created it?**  
  Prince Aikins Baidoo / Kronos-CMAA research repo (`kronos-research`).

- **Funding?**  
  None disclosed beyond free public APIs and Kaggle compute.

## 2. Composition

- **Prices:** Binance Vision spot `BTCUSDT` 1h OHLCV (+ quote volume as `amount`), UTC bar **open**. Typical span after confluence: 2021-01-01 → 2026-10-05. See `data/raw/btc_1h.csv`.

- **News (raw merge):** English headlines / short text with `published_at`, `text`, assembled from:
  - monstaws-style 3h aggregated crypto news dumps
  - Argus crypto news titles
  - mouadja bitcoin-news parquet (Finnhub/AlphaVantage/Guardian-tagged)
  - GDELT DOC API headlines (sparse + densified Jul–Sep 2025 + June 2025 densify)
  - Finnhub `company-news` for BTC-adjacent US equities (MSTR, COIN, miners, …) with keyword filter (history ≈ from 2025-10 on free tier)

- **Processed (Q1 freeze):** `data/processed/btc_news_q1.csv` via `scripts/prepare_q1_btc_news.py` (truncate, light clean, optional relevance regex). Slim twin `btc_news_q1_slim.csv` has only `published_at,text` for `build_cache.py`.

- **Soft spots:**  
  - **2026-04** ~82% of bars have ≥1 headline in prior 24h — disclosed; run sensitivity excluding that month.  
  - **2025-06** densified via GDELT (target ≥~90% bar text); residual documented in audit if any.

- **Does the dataset contain confidential data?** No personal identifiers intended; public market/news text only.

## 3. Collection process

- Prices: HTTPS download from `data.binance.vision` monthly/daily zip klines (`scripts/fetch_raw_data.py`).
- News: free APIs + Kaggle/HF/GitHub dumps + GDELT on Kaggle CPU kernels; Finnhub via `FINNHUB_API_KEY` in local `.env` (never committed).
- Alignment rule (look-ahead safe): a headline is eligible for bar open `t` only if `published_at ≤ t + 1h` (bar close). Implemented in `scripts/build_cache.py` `text_windows()`.

## 4. Preprocessing

1. Merge + dedupe on cleaned text (`scripts/fetch_news.write_news` / backfill scripts).  
2. Q1 hygiene: `scripts/prepare_q1_btc_news.py` — strip URLs/@, min length, truncate to 4000 chars, BTC relevance regex.  
3. Coverage audit: `scripts/audit_raw_readiness.py`.  
4. Freeze hashes: `scripts/hash_freeze.py` → `data/processed/MANIFEST.json`.

## 5. Uses

- **Recommended:** BTC multimodal spike classification research; ablations (price-only, placebos, source families).  
- **Not recommended as:** a gold-standard “complete market news” archive; dual-asset (XAU) claims; finance-journal identification without licensed news.

## 6. Distribution & maintenance

- Processed artifacts + `MANIFEST.json` hashes for reproducibility.  
- Raw CSVs may remain local/gitignored; release package should ship processed news + price CSV + datasheet.  
- Contact: repo maintainer on GitHub `PrinceAikinsBaidoo/kronos-research`.

## 7. Splits & labels

- Chronological **70% / 15% / 15%** train/val/test.  
- Spike label: forward 24h realized vol above train-fitted q0.90 (`locked/eval.make_spike_labels`).  
- Test window untouched by the agent loop until final human eval.
