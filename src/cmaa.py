"""Cross-Modal Attention Adapter (CMAA) and losses. Agent-editable.

Shapes
  h        : (B, T, d_model)  Kronos hidden states at an insertion point (price tokens, Queries)
  text     : (B, L, d_text)   frozen FinBERT last-layer token states (Keys/Values)
  text_mask: (B, L) bool, True where a text token is real, False where it is padding
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CMAALayer(nn.Module):
    def __init__(self, d_model, d_text, n_heads=8, dropout=0.1, gate=True):
        super().__init__()
        self.q_norm = nn.LayerNorm(d_model)
        self.kv_proj = nn.Linear(d_text, d_model)
        # Learned "no news" key/value so a sample with no text never produces all-masked attention.
        self.null_kv = nn.Parameter(torch.zeros(1, 1, d_model))
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 4 * d_model), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(4 * d_model, d_model))
        # Zero-init gate: the adapter starts as an identity map, so the frozen backbone's
        # behaviour is preserved at step 0.
        self.gate = nn.Parameter(torch.zeros(1)) if gate else None

    def forward(self, h, text, text_mask=None):
        B = h.size(0)
        kv = torch.cat([self.null_kv.expand(B, -1, -1), self.kv_proj(text)], dim=1)
        if text_mask is None:
            text_mask = torch.ones(text.shape[:2], dtype=torch.bool, device=text.device)
        valid = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=h.device), text_mask], dim=1)
        out, weights = self.attn(self.q_norm(h), kv, kv, key_padding_mask=~valid, need_weights=True)
        out = out + self.ffn(out)
        g = torch.tanh(self.gate) if self.gate is not None else 1.0
        return h + g * out, weights  # weights: (B, T, 1 + L), for attention heatmaps


class CMAAAdapter(nn.Module):
    """A stack of CMAA layers placed at ONE insertion point of the frozen backbone."""

    def __init__(self, d_model, d_text, n_layers=2, n_heads=8, dropout=0.1, gate=True):
        super().__init__()
        self.layers = nn.ModuleList(
            [CMAALayer(d_model, d_text, n_heads, dropout, gate) for _ in range(n_layers)])

    def forward(self, h, text, text_mask=None):
        weights = None
        for layer in self.layers:
            h, weights = layer(h, text, text_mask)
        return h, weights


def soft_contrastive_loss(price_emb, text_emb, mom, sent, tau=0.07, sigma=1.0):
    """Soft CLIP-style loss. Targets are graded, not one-hot (after SoftCLT).

    price_emb, text_emb: (B, D). mom, sent: (B,) z-scored momentum and sentiment.
    Affinity A_ij = exp(-(mom_i - sent_j)^2 / (2 sigma^2)): a bullish headline is pulled toward any
    positive-momentum window, in proportion to how close the two scores are.
    """
    p = F.normalize(price_emb, dim=-1)
    t = F.normalize(text_emb, dim=-1)
    logits = p @ t.t() / tau
    aff = torch.exp(-((mom[:, None] - sent[None, :]) ** 2) / (2 * sigma ** 2))
    tgt_p2t = aff / aff.sum(1, keepdim=True)
    tgt_t2p = aff.t() / aff.t().sum(1, keepdim=True)
    l_p2t = -(tgt_p2t * F.log_softmax(logits, dim=1)).sum(1).mean()
    l_t2p = -(tgt_t2p * F.log_softmax(logits.t(), dim=1)).sum(1).mean()
    return 0.5 * (l_p2t + l_t2p)


def focal_loss(logits, y, gamma=2.0, pos_weight=1.0):
    """Class-weighted focal loss for rare spike events. y in {0, 1}."""
    y = y.float()
    bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    p_t = torch.exp(-bce)
    w = 1.0 + (pos_weight - 1.0) * y
    return (w * (1 - p_t) ** gamma * bce).mean()
