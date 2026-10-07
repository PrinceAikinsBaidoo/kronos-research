# Ablations & placebos (BTC)

All claims vs locked price-only baseline: `locked/baselines/BTC.json`.

Primary metric: **validation AUC-PR**. Seed **42** for reported runs. Seeds 43/44 only for **H000** noise.

## Suite

| Name | Config | Intent |
| --- | --- | --- |
| price_only | use `scripts/run_price_only_baseline.py` | Locked baseline (no text / no CMAA) |
| full_text | `configs/full_text.yaml` | Full CMAA + news cache |
| placebo_shuffle_text | `configs/placebo_shuffle_text.yaml` | Break alignment; expect Δ≈0 |
| placebo_random_date | `configs/placebo_random_date.yaml` | Randomize news dates before cache; expect Δ≈0 |

Source-family ablations (GDELT-only / Finnhub-only / no-aggregate) need filtered processed news + separate caches — see `scripts/placebo_transform_cache.py` notes.

## Run (after BTC cache exists)

```bash
# Baseline (once, then commit JSON)
python scripts/run_price_only_baseline.py --asset BTC --cache-dir <cache>/BTC --out locked/baselines/BTC.json

# Ablation wrapper (records delta vs baseline)
python scripts/run_ablation.py --name full_text --config experiments/ablations/configs/full_text.yaml \
  --cache-dir <cache> --asset BTC

# Or push via Kaggle
python scripts/run_on_kaggle.py --exp ablations_full --config experiments/ablations/configs/full_text.yaml --asset BTC
```

Results land in `experiments/ablations/results/<name>_metrics.json`.
