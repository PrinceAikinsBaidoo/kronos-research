#!/usr/bin/env python3
"""Push one experiment to Kaggle, wait for it to finish, fetch metrics.

Usage:
  python scripts/run_on_kaggle.py --exp 003_tau_sweep --asset BTC [--smoke] [--final] [--allow-busy]

Needs: `pip install kaggle`, KAGGLE_USERNAME (or ~/.kaggle/kaggle.json), a clean git tree.
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SLUG = "kronos-cmaa-runner"
CACHE_SLUG = "kronos-cmaa-cache"
BUILDER_SLUG = "kronos-cmaa-cache-builder"
REPO_URL = "https://github.com/PrinceAikinsBaidoo/kronos-research.git"
EXPLORE_BUDGET_H = 18.0   # rolling 7 days; --final runs are exempt (human use only)
POLL_S = 30
MAX_WAIT_MIN = 60
LEDGER = ROOT / "experiments" / "gpu_ledger.csv"
TERMINAL = {"COMPLETE", "ERROR", "CANCEL"}


def run(cmd, check=True, capture=False):
    return subprocess.run(cmd, cwd=ROOT, check=check, text=True, capture_output=capture)


def kaggle_user():
    u = os.environ.get("KAGGLE_USERNAME")
    if u:
        return u
    p = Path.home() / ".kaggle" / "kaggle.json"
    if p.exists():
        return json.loads(p.read_text())["username"]
    sys.exit("Set KAGGLE_USERNAME or create ~/.kaggle/kaggle.json")


def head_commit():
    if run(["git", "status", "--porcelain"], capture=True).stdout.strip():
        sys.exit("Working tree is not clean. Commit your changes first (Kaggle runs the pushed commit).")
    sha = run(["git", "rev-parse", "HEAD"], capture=True).stdout.strip()
    if not run(["git", "branch", "-r", "--contains", sha], capture=True).stdout.strip():
        print("HEAD is not on the remote yet, pushing...")
        run(["git", "push"])
    return sha


def used_hours_last_7d():
    if not LEDGER.exists():
        return 0.0
    cutoff = datetime.now(timezone.utc) - timedelta(days=7)
    total = 0.0
    with LEDGER.open() as f:
        for row in csv.DictReader(f):
            if row["kind"] != "final" and datetime.fromisoformat(row["utc"]) >= cutoff:
                total += float(row["seconds"])
    return total / 3600


def log_ledger(exp, kind, seconds):
    new = not LEDGER.exists()
    LEDGER.parent.mkdir(exist_ok=True)
    with LEDGER.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["utc", "exp", "kind", "seconds"])
        w.writerow([datetime.now(timezone.utc).isoformat(), exp, kind, round(seconds)])


def status(ref):
    r = run(["kaggle", "kernels", "status", ref], check=False, capture=True)
    out = (r.stdout + r.stderr).upper()
    for key in ("COMPLETE", "CANCEL", "RUNNING", "QUEUED", "ERROR"):
        if key in out:
            return key
    return "UNKNOWN"


def render_push_dir(user, sha, exp, asset, smoke, config):
    d = ROOT / ".kaggle_push"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir()
    code = (ROOT / "kaggle" / "run_experiment.py").read_text()
    subs = {"__REPO__": REPO_URL, "__COMMIT__": sha, "__EXP_ID__": exp, "__ASSET__": asset,
            "__SMOKE__": "1" if smoke else "0", "__CONFIG__": config, "__CACHE_SLUG__": CACHE_SLUG}
    for k, v in subs.items():
        code = code.replace(k, v)
    (d / "run_experiment.py").write_text(code)
    # Use cache-builder kernel output until private dataset kronos-cmaa-cache exists
    meta = {"id": f"{user}/{SLUG}", "title": SLUG, "code_file": "run_experiment.py",
            "language": "python", "kernel_type": "script", "is_private": True,
            "enable_gpu": True, "enable_internet": True,
            "dataset_sources": [], "competition_sources": [],
            "kernel_sources": [f"{user}/{BUILDER_SLUG}"]}
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="experiment folder name under experiments/")
    ap.add_argument("--asset", required=True, choices=["BTC", "XAU"])
    ap.add_argument("--config", help="default: experiments/<exp>/config.yaml")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--final", action="store_true", help="human use only: exempt from the budget")
    ap.add_argument("--allow-busy", action="store_true")
    a = ap.parse_args()

    exp_dir = ROOT / "experiments" / a.exp
    config = a.config or f"experiments/{a.exp}/config.yaml"
    if not (ROOT / config).exists():
        sys.exit(f"Config not found: {config}")
    kind = "final" if a.final else ("smoke" if a.smoke else "explore")

    used = used_hours_last_7d()
    if kind != "final" and used >= EXPLORE_BUDGET_H:
        sys.exit(f"Exploration budget used up ({used:.1f} h of {EXPLORE_BUDGET_H} h in the last 7 days).")

    user = kaggle_user()
    ref = f"{user}/{SLUG}"
    if not a.allow_busy and status(ref) in {"RUNNING", "QUEUED"}:
        sys.exit(f"{ref} is already running or queued. Wait, or pass --allow-busy.")

    sha = head_commit()
    push_dir = render_push_dir(user, sha, a.exp, a.asset, a.smoke, config)
    print(f"Pushing {a.exp} ({a.asset}, {kind}) at commit {sha[:8]}")
    t0 = time.time()
    run(["kaggle", "kernels", "push", "-p", str(push_dir)])

    # The status call can briefly report the PREVIOUS run, so require a non-terminal status first.
    time.sleep(15)
    seen_active = False
    final_state = "UNKNOWN"
    while True:
        s = status(ref)
        if s in {"RUNNING", "QUEUED"}:
            seen_active = True
        elapsed = time.time() - t0
        print(f"  [{int(elapsed // 60):02d}:{int(elapsed % 60):02d}] {s}")
        if s in TERMINAL and (seen_active or elapsed > 90):
            final_state = s
            break
        if elapsed > MAX_WAIT_MIN * 60:
            log_ledger(a.exp, kind, elapsed)
            sys.exit(f"Timed out waiting. Check: kaggle kernels status {ref}")
        time.sleep(POLL_S)

    seconds = time.time() - t0
    log_ledger(a.exp, kind, seconds)

    out_dir = exp_dir / "kaggle_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    run(["kaggle", "kernels", "output", ref, "-p", str(out_dir)], check=False)
    found = sorted(out_dir.rglob("metrics.json"))
    if not found:
        print(f"Kernel finished as {final_state} but no metrics.json was produced.")
        for log in sorted(out_dir.rglob("*.log")):
            print(f"--- tail of {log.name} ---")
            print("\n".join(log.read_text(errors="replace").splitlines()[-40:]))
        sys.exit(1)

    metrics_path = exp_dir / f"metrics_{a.asset}{'_smoke' if a.smoke else ''}.json"
    shutil.copy(found[-1], metrics_path)
    m = json.loads(metrics_path.read_text())
    print(f"\nSaved {metrics_path.relative_to(ROOT)}  (wall time {seconds / 60:.1f} min)")
    if m.get("status") != "ok":
        print("Run FAILED inside the kernel:\n" + str(m.get("error", ""))[-1500:])
        sys.exit(1)
    print(json.dumps(m["val"], indent=2))


if __name__ == "__main__":
    main()
