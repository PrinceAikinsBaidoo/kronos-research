# Reproducibility (BTC Q1 freeze)

1. Verify hashes: open `data/processed/MANIFEST.json` and re-run  
   `python scripts/hash_freeze.py` — digests must match.
2. Datasheet: [DATASHEET_BTC.md](DATASHEET_BTC.md).
3. Rebuild cache on Kaggle from `kronos-cmaa-raw` (BTC news = Q1 slim):  
   `python scripts/build_cache_on_kaggle.py --asset BTC`
4. Lock baseline:  
   `python scripts/run_price_only_baseline.py --asset BTC --cache-dir <cache> --out locked/baselines/BTC.json`
5. Ablations: [experiments/ablations/README.md](../experiments/ablations/README.md).
6. H000 seed noise before claiming CMAA gains: [experiments/H000_seed_noise/](../experiments/H000_seed_noise/).
