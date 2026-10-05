"""Training entry point (runs on Kaggle). Agent-editable, except the metrics.json contract.

load_cache() and build_model() depend on the Kronos reference code and the cache layout from
scripts/build_cache.py.
"""
import argparse
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from locked.eval import chronological_split, perplexity_from_nll, spike_metrics  # noqa: E402
from src.cmaa import CMAAAdapter, focal_loss, soft_contrastive_loss  # noqa: E402

KRONOS_REPO = "https://github.com/shiyu-coder/Kronos.git"
KRONOS_COMMIT = "67b630e67f6a18c9e9be918d9b4337c960db1e9a"
KRONOS_MODEL_ID = "NeoQuasar/Kronos-base"
D_TEXT = 768


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def _ensure_kronos_on_path(kronos_dir="Kronos"):
    """Clone the pinned Kronos commit if needed and put it on sys.path."""
    import subprocess

    root = Path(__file__).resolve().parents[1]
    p = Path(kronos_dir)
    if not p.is_absolute():
        p = root / p
    if not (p / "model").exists():
        subprocess.run(["git", "clone", "--quiet", KRONOS_REPO, str(p)], check=True)
        r = subprocess.run(["git", "checkout", "--quiet", KRONOS_COMMIT], cwd=p)
        if r.returncode:
            print("WARNING: could not check out the pinned Kronos commit; using default branch")
    path = str(p.resolve())
    if path not in sys.path:
        sys.path.insert(0, path)
    return path


def find_asset_dir(cache_dir, asset):
    """Locate <cache>/BTC or nested Kaggle layout .../kronos-cmaa-cache/BTC."""
    root = Path(cache_dir)
    direct = root / asset
    if (direct / "meta.json").exists():
        return direct
    for p in root.rglob("meta.json"):
        if p.parent.name.upper() == asset.upper():
            return p.parent
    raise FileNotFoundError(
        f"No cache for asset {asset} under {cache_dir}. "
        "Expected meta.json inside a BTC/ or XAU/ folder."
    )


class CacheDataset(Dataset):
    """Memory-maps cache arrays; resolves text rows via text_idx (-1 = no news)."""

    def __init__(self, arrays, indices):
        self.s1 = arrays["s1_ids"]
        self.s2 = arrays["s2_ids"]
        self.y = arrays["y"]
        self.mom = arrays["mom"]
        self.sent = arrays["sent"]
        self.text_idx = arrays["text_idx"]
        self.text_states = arrays["text_states"]
        self.text_mask = arrays["text_mask"]
        self.indices = np.asarray(indices, dtype=np.int64)
        self.T = int(self.text_states.shape[1]) if len(self.text_states) else 128
        self.d_text = int(self.text_states.shape[2]) if len(self.text_states) else D_TEXT

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        j = int(self.indices[i])
        ti = int(self.text_idx[j])
        if ti >= 0 and len(self.text_states):
            text = np.asarray(self.text_states[ti], dtype=np.float32)
            mask = np.asarray(self.text_mask[ti], dtype=bool)
        else:
            text = np.zeros((self.T, self.d_text), dtype=np.float32)
            mask = np.zeros((self.T,), dtype=bool)
        return {
            "s1_ids": torch.from_numpy(np.asarray(self.s1[j], dtype=np.int64)),
            "s2_ids": torch.from_numpy(np.asarray(self.s2[j], dtype=np.int64)),
            "text_states": torch.from_numpy(text),
            "text_mask": torch.from_numpy(mask),
            "y": torch.tensor(int(self.y[j]), dtype=torch.float32),
            "mom": torch.tensor(float(self.mom[j]), dtype=torch.float32),
            "sent": torch.tensor(float(self.sent[j]), dtype=torch.float32),
        }


def load_cache(cfg, asset, cache_dir, smoke):
    """Return {"train": DataLoader, "val": DataLoader}. Never loads the test window."""
    asset_dir = find_asset_dir(cache_dir, asset)
    meta = json.loads((asset_dir / "meta.json").read_text())
    N = int(meta["N"])
    embargo = int(meta.get("embargo", cfg["data"]["spike_horizon_bars"]))

    arrays = {
        "s1_ids": np.load(asset_dir / "s1_ids.npy", mmap_mode="r"),
        "s2_ids": np.load(asset_dir / "s2_ids.npy", mmap_mode="r"),
        "y": np.load(asset_dir / "y.npy", mmap_mode="r"),
        "mom": np.load(asset_dir / "mom.npy", mmap_mode="r"),
        "sent": np.load(asset_dir / "sent.npy", mmap_mode="r"),
        "text_idx": np.load(asset_dir / "text_idx.npy", mmap_mode="r"),
        "text_states": np.load(asset_dir / "text_states.npy", mmap_mode="r"),
        "text_mask": np.load(asset_dir / "text_mask.npy", mmap_mode="r"),
    }
    assert arrays["s1_ids"].shape[0] == N

    if "splits" in meta:
        tr0, tr1 = meta["splits"]["train"]
        va0, va1 = meta["splits"]["val"]
        train_sl = slice(tr0, tr1)
        val_sl = slice(va0, va1)
    else:
        train_sl, val_sl, _ = chronological_split(N)

    # Drop embargo samples at the end of train/val so forward labels cannot cross the cut.
    train_idx = np.arange(train_sl.start, max(train_sl.start, train_sl.stop - embargo))
    val_idx = np.arange(val_sl.start, max(val_sl.start, val_sl.stop - embargo))

    if smoke:
        train_idx = train_idx[: min(256, len(train_idx))]
        val_idx = val_idx[: min(64, len(val_idx))]

    bs = int(cfg["train"]["batch_size"])
    if smoke:
        bs = min(bs, 16)
    kw = dict(batch_size=bs, num_workers=0, pin_memory=torch.cuda.is_available())
    return {
        "train": DataLoader(CacheDataset(arrays, train_idx), shuffle=True, drop_last=not smoke, **kw),
        "val": DataLoader(CacheDataset(arrays, val_idx), shuffle=False, drop_last=False, **kw),
    }


class KronosWithCMAA(nn.Module):
    """Frozen Kronos-base with CMAA adapters after selected transformer blocks + spike head."""

    def __init__(self, kronos, adapters, d_model, proj_dim=256):
        super().__init__()
        self.kronos = kronos
        self.adapters = nn.ModuleDict({str(i): a for i, a in adapters.items()})
        self.insert_after = set(adapters.keys())
        self.spike_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        self.price_proj = nn.Linear(d_model, proj_dim)
        self.text_proj = nn.Linear(D_TEXT, proj_dim)

    def forward(self, batch):
        s1 = batch["s1_ids"].long()
        s2 = batch["s2_ids"].long()
        text = batch["text_states"].float()
        text_mask = batch["text_mask"].bool()

        x = self.kronos.embedding([s1, s2])
        x = self.kronos.token_drop(x)

        for i, layer in enumerate(self.kronos.transformer):
            x = layer(x, key_padding_mask=None)
            if i in self.insert_after:
                x, _ = self.adapters[str(i)](x, text, text_mask)

        context = self.kronos.norm(x)
        s1_logits = self.kronos.head(context)
        s2_logits = self.kronos.decode_s2(context, s1, padding_mask=None)

        # Teacher-forced next-token NLL (nats), mean over sequence → (B,)
        nll_s1 = F.cross_entropy(
            s1_logits[:, :-1].reshape(-1, s1_logits.size(-1)),
            s1[:, 1:].reshape(-1),
            reduction="none",
        ).view(s1.size(0), -1).mean(dim=1)
        nll_s2 = F.cross_entropy(
            s2_logits[:, :-1].reshape(-1, s2_logits.size(-1)),
            s2[:, 1:].reshape(-1),
            reduction="none",
        ).view(s2.size(0), -1).mean(dim=1)
        nll = 0.5 * (nll_s1 + nll_s2)

        price_h = context.mean(dim=1)
        denom = text_mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
        text_h = (text * text_mask.unsqueeze(-1).float()).sum(dim=1) / denom

        return {
            "logit": self.spike_head(price_h).squeeze(-1),
            "price_emb": self.price_proj(price_h),
            "text_emb": self.text_proj(text_h),
            "nll": nll,
        }


def build_model(cfg):
    """Frozen Kronos-base with CMAAAdapter modules after cfg['adapter']['insert_after']."""
    _ensure_kronos_on_path()
    from model import Kronos  # from the Kronos repo

    ac = cfg["adapter"]
    kronos = Kronos.from_pretrained(cfg.get("kronos_model_id", KRONOS_MODEL_ID))
    for p in kronos.parameters():
        p.requires_grad = False
    kronos.eval()

    d_model = int(ac.get("d_model", kronos.d_model))
    if d_model != kronos.d_model:
        print(f"WARNING: adapter.d_model={d_model} != Kronos d_model={kronos.d_model}; using Kronos")
        d_model = kronos.d_model

    adapters = {}
    for idx in ac["insert_after"]:
        adapters[int(idx)] = CMAAAdapter(
            d_model=d_model,
            d_text=D_TEXT,
            n_layers=int(ac["layers_per_point"]),
            n_heads=int(ac["n_heads"]),
            dropout=float(ac["dropout"]),
            gate=bool(ac["gate"]),
        )
    return KronosWithCMAA(kronos, adapters, d_model=d_model)


def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


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


def train(cfg, data, model, device, smoke):
    tc, cc, lc = cfg["train"], cfg["contrastive"], cfg["loss"]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=tc["lr"])
    deadline = time.time() + tc["max_minutes"] * 60
    best, best_epoch, bad, history, timed_out = None, -1, 0, [], False
    epochs = 1 if smoke else tc["epochs"]
    for epoch in range(epochs):
        model.train()
        # Kronos stays in eval (dropout off); only adapters/heads train.
        model.kronos.eval()
        for step, b in enumerate(data["train"]):
            if smoke and step >= 20:
                break
            b = to_device(b, device)
            o = model(b)
            loss = focal_loss(o["logit"], b["y"], gamma=lc["focal_gamma"])
            if cc["weight"] > 0:
                loss = loss + cc["weight"] * soft_contrastive_loss(
                    o["price_emb"], o["text_emb"], b["mom"], b["sent"], cc["tau"], cc["sigma"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if time.time() > deadline:
                timed_out = True
                break
        m = evaluate(model, data["val"], device)
        history.append({"epoch": epoch, **m})
        if best is None or m["auc_pr"] > best["auc_pr"]:
            best, best_epoch, bad = m, epoch, 0
        else:
            bad += 1
        if timed_out or bad >= tc["patience"]:
            break
    return best, best_epoch, history, timed_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--asset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--commit", default="unknown")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cfg = yaml.safe_load(open(args.config))
    t0 = time.time()
    result = {"status": "error", "asset": args.asset, "commit": args.commit, "smoke": args.smoke,
              "seed": cfg["seed"], "config": cfg}
    try:
        set_seed(cfg["seed"])
        device = "cuda" if torch.cuda.is_available() else "cpu"
        data = load_cache(cfg, args.asset, args.cache_dir, args.smoke)
        model = build_model(cfg).to(device)
        best, best_epoch, history, timed_out = train(cfg, data, model, device, args.smoke)
        result.update(status="ok", val=best, best_epoch=best_epoch, history=history,
                      stopped_for_time=timed_out)
    except Exception:
        result["error"] = traceback.format_exc()
    result["train_seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: result.get(k) for k in ("status", "val", "train_seconds")}, indent=2))
    sys.exit(0 if result["status"] == "ok" else 1)


if __name__ == "__main__":
    main()
