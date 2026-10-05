# kronos-cmaa: experiment loop scaffold

Autonomous (but guard-railed) experiment loop for the Kronos-CMAA research project. An agent
(Claude Code) iterates on the adapter; training runs on Kaggle's free GPU. All state lives in files.

```
GOAL.md              objective, constraints, budgets, stop conditions, open decisions
CLAUDE.md            instructions the agent reads every iteration
NOTES.md             journal and current best results (agent-maintained)
queue.json           ranked hypotheses
configs/default.yaml starting configuration
src/cmaa.py          adapter + soft contrastive + focal loss        (agent-editable)
src/train.py         training entry point                           (agent-editable; 2 TODOs for you)
locked/              evaluation contract, labels, splits, baselines (READ-ONLY for the agent)
kaggle/              script that runs on Kaggle (filled in by the push script)
scripts/build_cache.py     one-off: Kronos tokens + FinBERT token states + labels, per asset
scripts/run_on_kaggle.py   push, poll, fetch metrics, track GPU hours
scripts/loop.sh      headless driver with guardrails
experiments/NNN_slug/      one folder per experiment (hypothesis, config, metrics, result)
```

## One-time setup
1. `pip install kaggle pyyaml`. In Kaggle: Settings, API, Create New Token. Put `kaggle.json` in
   `~/.kaggle/` or set `KAGGLE_USERNAME` and `KAGGLE_KEY`. Verify your phone number so notebooks
   can use the internet.
2. Build the cache once per asset, on a Kaggle or Colab GPU notebook with internet on:
   ```
   !git clone https://github.com/PrinceAikinsBaidoo/kronos-research.git
   %cd kronos-research
   !pip install -q -r requirements.txt
   !python scripts/build_cache.py --asset BTC --prices btc_1h.csv --news btc_news.csv --out /kaggle/working/cache --dry-run
   ```
   Check the printed sample count, text-cache size and spike/text rates, then rerun without
   `--dry-run`. Do XAUUSD separately (without `--regular-grid`, because of weekend closures).
   Put both assets' folders (`BTC/`, `XAU/`) in one private Kaggle Dataset named `kronos-cmaa-cache`.
   Input CSVs: prices need a UTC open-time column plus open, high, low, close, volume, [amount];
   news needs `published_at` (UTC) and `text`. Two details in the script are marked VERIFY.
3. Keep this GitHub repo public while developing (the Kaggle script clones it). If it must be
   private, create a read-only token and attach it as a Kaggle secret named `GITHUB_TOKEN`.
4. Run and lock the price-only Kronos baseline yourself. Save its metrics to
   `locked/baselines/BTC.json` and `locked/baselines/XAU.json`, and commit.
5. Implement `load_cache()` and `build_model()` in `src/train.py` (see docstrings).
6. Create `experiments/000_smoke/config.yaml` (copy `configs/default.yaml`), commit, push, then:
   `python scripts/run_on_kaggle.py --exp 000_smoke --asset BTC --smoke`

## Running the loop
`MAX_ITERS=10 bash scripts/loop.sh`
Stop it any time with `touch STOP`. If `needs_review.md` appears, read it before continuing.

## Rules that keep the results honest
- The agent never sees the test window. You run the final 5/6-fold and test evaluation yourself
  with a frozen config, using `--final`.
- Hundreds of agent runs overfit the validation set, so validation numbers are for steering only.
- Disclose the use of AI agents for experiment iteration to your supervisor and in the methods section.
