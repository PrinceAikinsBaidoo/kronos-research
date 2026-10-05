#!/usr/bin/env python3
"""Run the locked price-only Kronos baseline (P3) and write metrics JSON.

Uses the same cache / chronological train+val slices as CMAA training, but:
  - no FinBERT / text inputs
  - no CMAA adapters
  - trainable spike head (+ optional tiny proj) on frozen Kronos only

Example (on Kaggle GPU after cache exists):
  python scripts/run_price_only_baseline.py --asset BTC --cache-dir /kaggle/input/.../kronos-cmaa-cache \\
      --out locked/baselines/BTC.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from locked.eval import spike_metrics, perplexity_from_nll  # noqa: E402
from src.train import (  # noqa: E402
    KRONOS_MODEL_ID,
    _ensure_kronos_on_path,
    load_cache,
    set_seed,
    to_device,
)


class PriceOnlyKronos(nn.Module):
    """Frozen Kronos + spike head; NLL from teacher-forced next-token CE."""

    def __init__(self, kronos):
        super().__init__()
        self.kronos = kronos
        d = kronos.d_model
        self.spike_head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Linear(d // 2, 1),
        )

    def forward(self, batch):
        import torch.nn.functional as F

        s1 = batch["s1_ids"].long()
        s2 = batch["s2_ids"].long()
        x = self.kronos.embedding([s1, s2])
        x = self.kronos.token_drop(x)
        for layer in self.kronos.transformer:
            x = layer(x, key_padding_mask=None)
        context = self.kronos.norm(x)
        s1_logits = self.kronos.head(context)
        s2_logits = self.kronos.decode_s2(context, s1, padding_mask=None)
        nll_s1 = F.cross_entropy(
            s1_logits[:, :-1].reshape(-1, s1_logits.size(-1)),
            s1[:, 1:].reshape(-1), reduction="none",
        ).view(s1.size(0), -1).mean(dim=1)
        nll_s2 = F.cross_entropy(
            s2_logits[:, :-1].reshape(-1, s2_logits.size(-1)),
            s2[:, 1:].reshape(-1), reduction="none",
        ).view(s2.size(0), -1).mean(dim=1)
        h = context.mean(dim=1)
        return {
            "logit": self.spike_head(h).squeeze(-1),
            "nll": 0.5 * (nll_s1 + nll_s2),
        }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ss, nlls = [], [], []
    for b in loader:
        b = to_device(b, device)
        o = model(b)
        ys.append(b["y"].cpu().numpy())
        ss.append(torch.sigmoid(o["logit"]).cpu().numpy())
        nlls.append(o["nll"].cpu().numpy())
    m = spike_metrics(np.concatenate(ys), np.concatenate(ss))
    m["ood_perplexity"] = perplexity_from_nll(np.concatenate(nlls))
    return m


def train_head(model, loader, device, lr, epochs, smoke):
    import torch.nn.functional as F

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    model.kronos.eval()
    for epoch in range(1 if smoke else epochs):
        model.train()
        model.kronos.eval()
        for step, b in enumerate(loader):
            if smoke and step >= 20:
                break
            b = to_device(b, device)
            o = model(b)
            loss = F.binary_cross_entropy_with_logits(o["logit"], b["y"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--asset", required=True, choices=["BTC", "XAU"])
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--out", help="default: locked/baselines/<ASSET>.json")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=5)
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_cache(cfg, a.asset, a.cache_dir, a.smoke)
    _ensure_kronos_on_path()
    from model import Kronos

    kronos = Kronos.from_pretrained(cfg.get("kronos_model_id", KRONOS_MODEL_ID))
    for p in kronos.parameters():
        p.requires_grad = False
    kronos.eval()
    model = PriceOnlyKronos(kronos).to(device)
    train_head(model, data["train"], device, cfg["train"]["lr"], a.epochs, a.smoke)
    val = evaluate(model, data["val"], device)

    payload = {
        "asset": a.asset,
        "model": "price_only_kronos",
        "kronos_model_id": cfg.get("kronos_model_id", KRONOS_MODEL_ID),
        "seed": cfg["seed"],
        "split": "chronological_70_15_15",
        "metrics_window": "validation",
        "val": val,
        "notes": "No CMAA; spike head on pooled Kronos context only. Same cache/labels as CMAA.",
        "smoke": a.smoke,
    }
    out = Path(a.out) if a.out else ROOT / "locked" / "baselines" / f"{a.asset}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
