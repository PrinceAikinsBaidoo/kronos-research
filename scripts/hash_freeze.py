#!/usr/bin/env python3
"""Write SHA256 manifest for frozen Q1 artifacts.

  python scripts/hash_freeze.py
  python scripts/hash_freeze.py --files data/raw/btc_1h.csv data/processed/btc_news_q1.csv \\
      --out data/processed/MANIFEST.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    n = 0
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
            n += len(chunk)
    return h.hexdigest(), n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--files",
        nargs="+",
        default=[
            str(ROOT / "data" / "raw" / "btc_1h.csv"),
            str(ROOT / "data" / "processed" / "btc_news_q1.csv"),
        ],
    )
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "processed" / "MANIFEST.json")
    a = ap.parse_args()

    entries = []
    for raw in a.files:
        p = Path(raw)
        if not p.is_absolute():
            p = ROOT / p
        if not p.exists():
            print(f"WARN missing: {p}")
            continue
        digest, nbytes = sha256_file(p)
        try:
            rel = str(p.relative_to(ROOT)).replace("\\", "/")
        except ValueError:
            rel = str(p)
        entries.append({"path": rel, "sha256": digest, "bytes": nbytes})
        print(f"{digest}  {rel}  ({nbytes} bytes)")

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "files": entries,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {a.out}")


if __name__ == "__main__":
    main()
