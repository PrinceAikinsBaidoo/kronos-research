# Evaluation baselines (locked)

This folder holds **frozen reference metrics** the research is judged against.
The agent loop must not edit these files after they are committed.

## Primary baseline (required before P5)

**Price-only Kronos** — same frozen Kronos backbone, same chronological splits,
same spike labels (`locked/eval.make_spike_labels`), **no text / no CMAA**.

| Asset | File | How to produce |
| --- | --- | --- |
| BTC | `BTC.json` | `python scripts/run_price_only_baseline.py --asset BTC --cache-dir <cache>` |
| XAU | `XAU.json` | `python scripts/run_price_only_baseline.py --asset XAU --cache-dir <cache>` |

After a successful run, copy/commit the written JSON here and never regenerate
with a different label rule or split.

### What “beating the baseline” means

- **Primary:** validation **AUC-PR** (CMAA) − AUC-PR (price-only)  
- Until **H000** finishes, do not call any delta an improvement (seed noise).  
- After H000: improvement only if Δ AUC-PR **>** measured seed noise.  
- Secondary (report, do not optimize for): AUC-ROC, macro F1, OOD perplexity.

### JSON schema (locked)

```json
{
  "asset": "BTC",
  "model": "price_only_kronos",
  "kronos_model_id": "NeoQuasar/Kronos-base",
  "seed": 42,
  "split": "chronological_70_15_15",
  "metrics_window": "validation",
  "val": {
    "auc_pr": 0.0,
    "auc_roc": 0.0,
    "macro_f1": 0.0,
    "threshold": 0.0,
    "ood_perplexity": 0.0
  },
  "notes": "No CMAA; spike head on pooled Kronos context only."
}
```

Test-set numbers are **not** stored here. The human runs `--final` once at P6.

## Secondary baselines (supervisor decision — GOAL #5)

The proposal lists two different comparison sets. **Do not lock these until the
human confirms one list with the supervisor:**

| Option A (proposal §4A) | Option B (proposal §4E) |
| --- | --- |
| M2VN | Early fusion |
| FININ | FinBERT-score feature |
| RiskLabs | |

Until settled, CMAA is only compared to **price-only Kronos**. Optional
secondary baselines can be added later as `BTC_m2vn.json` etc. without
changing metric definitions in `locked/eval.py`.

## Rules

1. Same cache, same `y`, same train/val indices as CMAA runs.  
2. Never retune the spike quantile or horizon to lift the baseline.  
3. Seed 42 for the locked file; seeds 43/44 only for H000 noise, not for rewriting these JSON files.
