# GOAL: Kronos-CMAA experiment loop

## What we are building
Kronos-CMAA: a trainable Cross-Modal Attention Adapter (CMAA) inserted between frozen Kronos
transformer blocks. Kronos price tokens are Queries; frozen FinBERT last-layer TOKEN-level hidden
states (not pooled CLS) are Keys/Values. Training uses a combined loss: focal-loss spike
classification + soft CLIP-style contrastive alignment between narrative sentiment and price momentum.

Assets: BTC/USDT and XAUUSD, trained and evaluated INDEPENDENTLY (no joint training).

## Objective of the loop
For each asset, find the CMAA configuration that maximises validation AUC-PR for volatility-spike
classification, and measure its gain over the locked price-only Kronos baseline.
Secondary: validation OOD perplexity (lower is better), AUC-ROC, macro F1.

## Hard constraints (never violate)
1. Kronos backbone and the text encoder stay FROZEN. Only CMAA and projection/heads train.
2. Splits are chronological: 70% train / 15% val / 15% test per asset. The TEST window is never
   loaded, read, or evaluated by the loop. Only the human touches it, once, at the end.
3. Everything in `locked/` is read-only: evaluation code, split logic, spike-label definition,
   baseline metrics. Do not edit it, work around it, or re-implement it elsewhere.
4. Metric definitions do not change. Primary metric is validation AUC-PR.
5. Seed 42 for all reported runs. Seeds 43 and 44 are allowed only for noise estimation (H000).
6. One change per experiment. Every run is logged, including failures and bad results.

## Preconditions (human completes these before the loop starts)
- `locked/baselines/BTC.json` and `locked/baselines/XAU.json` exist (locked price-only Kronos metrics).
- Cache dataset `kronos-cmaa-cache` uploaded to Kaggle (Kronos tokens + FinBERT token states + labels).
- `load_cache` and `build_model` in `src/train.py` implemented; smoke test passes.

## Budgets
- Per run: 25 minutes of training (config `train.max_minutes`), hard kill at 40 minutes.
- Exploration GPU budget: 18 hours per rolling 7 days (enforced by scripts/run_on_kaggle.py).
- Reserved for the human: about 10 hours per week for final 5-fold and test runs (`--final`).
- Per loop invocation: at most 20 iterations (see scripts/loop.sh).

## Stop conditions
- Create `needs_review.md` and stop if: a locked file was touched, a result looks too good
  (for example val AUC-PR above 0.95 or a jump larger than 0.10 in one change), the budget is
  exhausted, three consecutive runs fail, or you are unsure what the right action is.
- Stop (write a final summary in NOTES.md) when the open queue is empty or 8 consecutive
  experiments fail to beat the current best by more than the noise level from H000.

## Open decisions (HUMAN ONLY, the agent must not resolve these)
1. Significance test: a paired Wilcoxon test over 5 folds can never reach p < 0.05, because the
   smallest possible two-sided p is 2/2^5 = 0.0625. Options: 6+ folds (min p 0.031), multiple seeds,
   or a block bootstrap. Confirm with supervisor.
2. CV scheme: the proposal says "5-fold stratified" and "strictly chronological". Stratified folds
   leak the future. `locked/eval.py` provides expanding-window folds (default 6, with an embargo).
3. Bar counts: 2021 to 2026 at 1-hour bars is about 45,000 bars per asset, not 260,000.
4. "4 cross-attention layers inserted after blocks 6 and 9": treated as 2 layers per insertion
   point. Confirm.
5. Baseline lists differ between Section 4A (M2VN, FININ, RiskLabs) and 4E (early fusion,
   FinBERT-score feature). Settle one list.
6. Soft contrastive targets: `src/cmaa.py` uses Gaussian affinity between z-scored momentum and
   z-scored sentiment. This is one reading of the proposal. Confirm it is the intended one.
7. Ethics: Section 5 says no human participants, but Section 4C runs a survey of traders.
   Resolve with CHRPE before any data collection.
