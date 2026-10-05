# Agent instructions (read every iteration)

You are running one iteration of an experiment loop for the Kronos-CMAA research project.
Do exactly ONE iteration, then stop. An outside script calls you again.

## Files
- GOAL.md: objective, constraints, budgets, stop conditions. Re-read it every time.
- NOTES.md: running journal and current best results.
- queue.json: ranked hypotheses.
- src/cmaa.py, src/train.py, configs/: you may edit these.
- locked/: DO NOT EDIT. Evaluation contract, labels, splits, baseline metrics.
- scripts/run_on_kaggle.py: the only way to run training (GPU is on Kaggle, not here).

## Iteration procedure
1. If a file named STOP or needs_review.md exists in the repo root, stop immediately.
2. Read GOAL.md, NOTES.md, queue.json. Pick the highest-priority hypothesis with status "open"
   whose depends_on items are all "done".
3. Create experiments/NNN_slug/ with hypothesis.md (what you change, why, what you predict) and
   config.yaml (copy of the current best config with exactly one change).
4. If you changed code under src/, first run a smoke test:
   python scripts/run_on_kaggle.py --exp NNN_slug --asset BTC --smoke
   Fix errors before spending real GPU time.
5. Commit and push BEFORE each run (the Kaggle kernel clones the pushed commit):
   git add -A && git commit -m "exp NNN: <summary>" && git push
6. Run each asset listed in the hypothesis:
   python scripts/run_on_kaggle.py --exp NNN_slug --asset BTC
   Never use --final. That flag is for the human.
7. Read experiments/NNN_slug/metrics_BTC.json (and metrics_XAU.json if run). Write experiments/NNN_slug/result.md: the numbers,
   the delta versus the current best and versus the locked baseline, a one-sentence interpretation,
   and one sentence on what the result does NOT show.
8. Update the table in NOTES.md and the status in queue.json (done, failed, or rejected).
   You may add at most 2 new hypotheses, each with a stated reason.
9. Commit and push. Stop.

## Rules
- Never read, load, or evaluate the test window. Never edit anything in locked/.
- Never change a metric, label, split, threshold rule, or seed to make a result look better.
- A result counts as an improvement only if it beats the current best by more than the seed noise
  measured in H000. Until H000 is done, call nothing an improvement.
- Report negative and failed results exactly as you report positive ones.
- Do not claim causes you did not test. Say "consistent with" rather than "because".
- If a result looks too good, assume a bug or leakage first and investigate before logging it as a win.
- If something is unclear or blocked, write needs_review.md explaining why, and stop.
- Keep each experiment small enough to finish inside the per-run time cap.
