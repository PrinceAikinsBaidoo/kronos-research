#!/usr/bin/env bash
# Headless driver: calls Claude Code once per experiment iteration, with guardrails.
# Run from the repo root on any light machine (laptop, Codespaces). GPUs are on Kaggle.
# Check `claude --help` for the current flag names; adjust --allowedTools to taste.
set -uo pipefail
MAX_ITERS="${MAX_ITERS:-20}"

for i in $(seq 1 "$MAX_ITERS"); do
  if [ -f STOP ] || [ -f needs_review.md ]; then
    echo "STOP or needs_review.md present, ending loop."; break
  fi
  before="$(git rev-parse HEAD)"
  echo "=== iteration $i of $MAX_ITERS ==="
  claude -p "Follow CLAUDE.md. Do exactly one experiment iteration, then stop." \
    --max-turns 60 \
    --allowedTools "Read" "Edit" "Write" "Bash(git:*)" "Bash(python scripts/run_on_kaggle.py:*)" \
                   "Bash(ls:*)" "Bash(cat:*)" "Bash(mkdir:*)" "Bash(cp:*)" \
    || echo "claude exited non-zero"

  # Guardrail: the agent must never modify locked/.
  if git diff --name-only "$before" HEAD | grep -q '^locked/'; then
    echo "locked/ was modified. Stopping for human review."
    echo "Agent modified locked/ in iteration $i. Inspect with: git diff $before HEAD -- locked/" > needs_review.md
    break
  fi
  sleep 5
done
