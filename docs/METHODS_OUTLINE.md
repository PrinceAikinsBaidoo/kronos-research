# Methods outline (CS / Fintech Q1 draft)

Skeleton for the primary **BTC** paper. Fill results after baselines + ablations.

## 1. Problem
Predict binary volatility-spike labels on 1h BTC bars using price history and recent public news text. Report **validation AUC-PR** as primary metric; secondary AUC-ROC, macro-F1, OOD perplexity.

## 2. Data
See [DATASHEET_BTC.md](DATASHEET_BTC.md). Prices: Binance Vision. News: constructed public corpus (GDELT, Finnhub-filtered equities, aggregators), Q1-hygiened and hashed.

**Alignment:** headlines with `published_at ≤ bar_close`. Chronological 70/15/15. Spike labels from train-only quantile.

**Limitations:** heterogeneous sources; soft month 2026-04; XAU deferred.

## 3. Models
- **Backbone:** frozen Kronos (`NeoQuasar/Kronos-base`) on quantized/normalized OHLCV windows.  
- **Text:** frozen FinBERT last-layer token states (not CLS-only pooling for CMAA K/V).  
- **Trainable:** Cross-Modal Attention Adapter (CMAA) inserted after selected Kronos blocks; spike head; contrastive projection.  
- **Baseline:** price-only Kronos (same splits/labels, no text/CMAA) — locked in `locked/baselines/BTC.json`.

## 4. Training
Focal loss for spikes + soft contrastive alignment (sentiment vs momentum). Seed 42 for reported runs; seeds 43–44 only for H000 noise. Budgets per GOAL.md.

## 5. Evaluation protocol
- Primary: Δ val AUC-PR vs locked price-only baseline; improvements only after H000 noise floor.  
- Expanding-window folds for final human eval (`locked/eval.py`); do not use future-leaking stratified CV.  
- **Ablations:** full text; source-family filters when available; no mega-aggregate blobs.  
- **Placebos:** shuffle text across time; random-date reassignment — expect Δ≈0 if CMAA uses genuine alignment.

## 6. Reproducibility
`data/processed/MANIFEST.json` SHA256; scripts to rebuild cache; Kaggle runners. No secrets in git.

## 7. Ethics / disclosure
Public news only; no trading advice; constructed corpus ≠ licensed news vendor.
