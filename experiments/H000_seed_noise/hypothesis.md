# H000 — seed noise floor (BTC)

**Change:** identical full_text CMAA config; seeds **42, 43, 44** only.

**Predict:** val AUC-PR std across seeds defines noise; later CMAA “wins” must exceed this Δ.

**Does not show:** that CMAA beats price-only (compare means only after this floor exists).

## Run

```bash
python scripts/run_ablation.py --name H000_seed42 --config experiments/ablations/configs/full_text.yaml --cache-dir <cache> --seed 42
# repeat with configs that set seed 43 / 44 or override when train.py supports --seed
```

Record the three val AUC-PR values and report noise = max−min (or std) in NOTES.md.
