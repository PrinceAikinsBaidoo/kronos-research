"""Runs ON KAGGLE. scripts/run_on_kaggle.py fills in the __PLACEHOLDERS__ and pushes this file.

It clones the repo at an exact commit, runs src/train.py, and leaves only metrics.json (and logs)
in /kaggle/working, so the output download stays small.
"""
import json
import os
import shutil
import subprocess
import sys

REPO = "__REPO__"
COMMIT = "__COMMIT__"
EXP_ID = "__EXP_ID__"
ASSET = "__ASSET__"
SMOKE = "__SMOKE__"
CONFIG = "__CONFIG__"
CACHE_SLUG = "__CACHE_SLUG__"

WORK = "/kaggle/working"
OUT = f"{WORK}/out/{EXP_ID}"
REPO_DIR = f"{WORK}/repo"
HARD_TIMEOUT_S = 40 * 60
os.makedirs(OUT, exist_ok=True)


def fail(msg):
    with open(f"{OUT}/metrics.json", "w") as f:
        json.dump({"status": "error", "error": msg, "exp": EXP_ID}, f, indent=2)
    print(msg)
    shutil.rmtree(REPO_DIR, ignore_errors=True)
    sys.exit(1)


def find_cache():
    # Dataset layout: .../kronos-cmaa-cache[/BTC]
    for root, dirs, _ in os.walk("/kaggle/input"):
        if CACHE_SLUG in os.path.basename(root):
            return root
        if root.count(os.sep) > 6:
            dirs[:] = []
    # Builder-kernel layout: .../kronos-cmaa-cache-builder/cache/BTC/meta.json
    for dirpath, _, files in os.walk("/kaggle/input"):
        if "meta.json" in files and os.path.basename(dirpath) == ASSET:
            return os.path.dirname(dirpath) if os.path.basename(os.path.dirname(dirpath)) == "cache" else dirpath
    return None


token = ""
try:
    from kaggle_secrets import UserSecretsClient
    token = UserSecretsClient().get_secret("GITHUB_TOKEN")
except Exception:
    pass
url = REPO.replace("https://", f"https://{token}@") if token else REPO

# Output of git is not printed, because the URL may contain a token.
r = subprocess.run(["git", "clone", "--quiet", url, REPO_DIR], capture_output=True, text=True)
if r.returncode:
    fail("git clone failed (is the repo public, or is the GITHUB_TOKEN secret attached?)")
r = subprocess.run(["git", "checkout", "--quiet", COMMIT], cwd=REPO_DIR, capture_output=True, text=True)
if r.returncode:
    fail(f"git checkout {COMMIT} failed (was the commit pushed?)")

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"],
               cwd=REPO_DIR, check=False)

cache = find_cache()
if not cache:
    fail(f"cache dataset '{CACHE_SLUG}' not found under /kaggle/input")

cmd = [sys.executable, "src/train.py", "--config", CONFIG, "--asset", ASSET, "--out", OUT,
       "--commit", COMMIT, "--cache-dir", cache]
if SMOKE == "1":
    cmd.append("--smoke")

try:
    subprocess.run(cmd, cwd=REPO_DIR, timeout=HARD_TIMEOUT_S, check=True)
except subprocess.TimeoutExpired:
    fail(f"hard timeout after {HARD_TIMEOUT_S} seconds")
except subprocess.CalledProcessError:
    if not os.path.exists(f"{OUT}/metrics.json"):
        fail("train.py crashed without writing metrics.json")
finally:
    shutil.rmtree(REPO_DIR, ignore_errors=True)
