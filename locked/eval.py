"""LOCKED. Do not edit. Evaluation contract for Kronos-CMAA.

Defines chronological splits, expanding-window folds, spike labels, metrics and the paired test.
"""
import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

TRAIN_FRAC, VAL_FRAC = 0.70, 0.15


def chronological_split(n):
    """Return (train, val, test) slices. Strictly chronological."""
    a = int(n * TRAIN_FRAC)
    b = int(n * (TRAIN_FRAC + VAL_FRAC))
    return slice(0, a), slice(a, b), slice(b, n)


def expanding_window_folds(n, n_folds=6, min_train_frac=0.4, embargo=24):
    """Yield (train_idx, val_idx) over range(n), where n is the size of the train+val window.

    Train always precedes validation. `embargo` bars are dropped between them so forward-looking
    spike labels cannot leak across the boundary.
    """
    start = int(n * min_train_frac)
    size = (n - start) // n_folds
    for k in range(n_folds):
        v0 = start + k * size
        v1 = v0 + size if k < n_folds - 1 else n
        yield np.arange(0, max(v0 - embargo, 0)), np.arange(v0, v1)


def make_spike_labels(close, horizon=24, q=0.90, fit_end=None):
    """Label bar t as a spike if realised volatility over the NEXT `horizon` bars exceeds the
    q-quantile of that quantity measured on bars [0, fit_end) only (train window).

    Returns (labels, threshold). labels is -1 where the forward window is incomplete.
    """
    close = np.asarray(close, dtype=float)
    n = len(close)
    r = np.diff(np.log(close))
    vol = np.full(n, np.nan)
    if len(r) >= horizon:
        win = np.lib.stride_tricks.sliding_window_view(r, horizon)
        fwd = win.std(axis=1)
        vol[: len(fwd)] = fwd
    fit_end = n if fit_end is None else fit_end
    thr = np.nanquantile(vol[:fit_end], q)
    y = np.where(np.isnan(vol), -1, (vol > thr).astype(int))
    return y, float(thr)


def best_f1_threshold(y, s):
    """Pick the score threshold that maximises macro F1 on the data given (calibration set)."""
    y, s = np.asarray(y), np.asarray(s)
    grid = np.unique(np.quantile(s, np.linspace(0.5, 0.999, 200)))
    scores = [f1_score(y, s >= t, average="macro", zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def spike_metrics(y, s, threshold=None):
    """AUC-PR (primary), AUC-ROC, macro F1. If threshold is None it is fitted on (y, s) itself,
    which is acceptable for validation reporting only. For test, fit it on validation."""
    y, s = np.asarray(y), np.asarray(s)
    if y.min() == y.max():
        return {"auc_pr": float("nan"), "auc_roc": float("nan"), "macro_f1": float("nan"),
                "threshold": float("nan")}
    if threshold is None:
        threshold = best_f1_threshold(y, s)
    return {
        "auc_pr": float(average_precision_score(y, s)),
        "auc_roc": float(roc_auc_score(y, s)),
        "macro_f1": float(f1_score(y, s >= threshold, average="macro", zero_division=0)),
        "threshold": float(threshold),
    }


def perplexity_from_nll(nll):
    """Perplexity from per-token (or per-sample) negative log-likelihoods in nats."""
    return float(np.exp(np.mean(nll)))


def min_possible_p(n_pairs):
    """Smallest two-sided p an exact Wilcoxon signed-rank test can return with n pairs."""
    return 2.0 / (2 ** n_pairs)


def paired_report(base, new):
    """Paired Wilcoxon signed-rank over per-fold (or per-seed) scores."""
    base, new = np.asarray(base, float), np.asarray(new, float)
    d = new - base
    out = {"n": int(len(d)), "mean_delta": float(d.mean()), "min_possible_p": min_possible_p(len(d))}
    out["p"] = 1.0 if np.all(d == 0) else float(wilcoxon(new, base).pvalue)
    return out
