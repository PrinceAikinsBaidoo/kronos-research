#!/usr/bin/env python3
"""Record ablation metrics and Δ AUC-PR vs locked BTC baseline.

If --cache-dir is set and train can run locally, invokes src/train.py.
Otherwise writes a structured placeholder and prints Kaggle instructions.

  python scripts/run_ablation.py --name full_text \\
      --config experiments/ablations/configs/full_text.yaml --cache-dir <cache> --asset BTC
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_baseline(path: Path) -> float | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    val = data.get("val") or {}
    return val.get("auc_pr")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--asset", default="BTC")
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--baseline", type=Path, default=ROOT / "locked" / "baselines" / "BTC.json")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "experiments" / "ablations" / "results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    if not a.config.exists():
        raise SystemExit(f"Missing config {a.config}")

    a.out_dir.mkdir(parents=True, exist_ok=True)
    out_metrics = a.out_dir / f"{a.name}_metrics.json"
    base_auc = load_baseline(a.baseline)

    result = {
        "status": "not_run",
        "ablation": a.name,
        "seed": a.seed,
        "asset": a.asset,
        "config": str(a.config).replace("\\", "/"),
        "val": {"auc_pr": None},
        "delta_vs_baseline_auc_pr": None,
        "baseline_path": str(a.baseline).replace("\\", "/") if a.baseline else None,
        "baseline_auc_pr": base_auc,
        "instructions": (
            "Run on Kaggle: python scripts/run_on_kaggle.py "
            f"--config {a.config} --asset {a.asset} "
            "after kronos-cmaa-cache exists; then copy metrics into this file."
        ),
    }

    if a.cache_dir and a.cache_dir.exists():
        run_out = a.out_dir / f"_run_{a.name}"
        run_out.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, str(ROOT / "src" / "train.py"),
            "--config", str(a.config),
            "--asset", a.asset,
            "--out", str(run_out),
            "--cache-dir", str(a.cache_dir),
            "--commit", "local",
        ]
        if a.smoke:
            cmd.append("--smoke")
        try:
            subprocess.run(cmd, cwd=ROOT, check=True)
            mpath = run_out / "metrics.json"
            if mpath.exists():
                m = json.loads(mpath.read_text(encoding="utf-8"))
                result["status"] = m.get("status", "ok")
                # train.py metric layout may vary
                val = m.get("val") or m.get("metrics", {}).get("val") or {}
                auc = val.get("auc_pr") if isinstance(val, dict) else None
                if auc is None:
                    auc = m.get("auc_pr")
                result["val"]["auc_pr"] = auc
                if auc is not None and base_auc is not None:
                    result["delta_vs_baseline_auc_pr"] = float(auc) - float(base_auc)
                result["raw_metrics"] = m
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
    else:
        print("No --cache-dir; writing placeholder metrics.")
        print(result["instructions"])

    out_metrics.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_metrics}")


if __name__ == "__main__":
    main()
