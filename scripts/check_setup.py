#!/usr/bin/env python3
"""Local readiness check for the Kronos-CMAA experiment loop. No GPU / no training."""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OK, WARN, FAIL = "[ok]", "[!!]", "[--]"


def check(label, ok, detail=""):
    mark = OK if ok else FAIL
    print(f"  {mark} {label}" + (f" — {detail}" if detail else ""))
    return bool(ok)


def main():
    print(f"Repo: {ROOT}\n")
    n_fail = 0

    print("Local code")
    train_py = ROOT / "src" / "train.py"
    n_fail += not check("src/train.py exists", train_py.exists())
    if train_py.exists():
        text = train_py.read_text(encoding="utf-8")
        n_fail += not check("load_cache implemented",
                            "def load_cache" in text and "raise NotImplementedError(\"implement load_cache\")" not in text)
        n_fail += not check("build_model implemented",
                            "def build_model" in text and "raise NotImplementedError(\"implement build_model\")" not in text)

    smoke = ROOT / "experiments" / "000_smoke" / "config.yaml"
    n_fail += not check("experiments/000_smoke/config.yaml", smoke.exists())

    print("\nGit")
    git = ROOT / ".git"
    n_fail += not check(".git present", git.exists(), "run: git init")
    try:
        import subprocess
        r = subprocess.run(["git", "remote", "get-url", "origin"], cwd=ROOT,
                           capture_output=True, text=True)
        n_fail += not check("origin remote", r.returncode == 0, (r.stdout or r.stderr).strip()[:80])
    except Exception as e:
        n_fail += not check("git available", False, str(e)[:80])

    print("\nKaggle API (needed to push kernels)")
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    env_ok = bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
    n_fail += not check("credentials", kaggle_json.exists() or env_ok,
                        "put kaggle.json in ~/.kaggle/ or set KAGGLE_USERNAME + KAGGLE_KEY")
    try:
        import importlib.util
        check("kaggle package", importlib.util.find_spec("kaggle") is not None)
    except Exception:
        n_fail += not check("kaggle package", False, "pip install kaggle")

    print("\nLocked baselines (needed before the real loop, not for smoke)")
    for asset in ("BTC", "XAU"):
        p = ROOT / "locked" / "baselines" / f"{asset}.json"
        check(f"locked/baselines/{asset}.json", p.exists(), "run price-only Kronos and save metrics")

    print("\nCache dataset (built on Kaggle/Colab, then uploaded as private dataset)")
    print(f"  {WARN} Build with scripts/build_cache.py on a GPU notebook, then upload")
    print(f"        BTC/ and XAU/ together as private Kaggle dataset: kronos-cmaa-cache")
    print(f"  {WARN} Needs price CSVs (OHLCV + UTC open time) and news CSVs (published_at, text)")

    print("\nGitHub (Kaggle clones this repo at a pushed commit)")
    print(f"  {WARN} Repo must exist and be public (or attach GITHUB_TOKEN Kaggle secret):")
    print(f"        https://github.com/PrinceAikinsBaidoo/kronos-research")

    print("\n" + ("READY for next manual steps above." if n_fail == 0 else
                  f"{n_fail} blocking item(s) — fix the [!!] lines first."))
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
