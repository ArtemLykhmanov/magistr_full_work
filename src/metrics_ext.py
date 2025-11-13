from __future__ import annotations
import numpy as np
from typing import Tuple
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score, brier_score_loss


def pr_auc(y_true, y_score) -> float:
    return float(average_precision_score(y_true, y_score))


def roc_auc(y_true, y_score) -> float:
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float("nan")


def f1_at_threshold(y_true, y_score, thr: float) -> float:
    y_pred = (y_score >= thr).astype(int)
    return float(f1_score(y_true, y_pred, zero_division=0))


def best_f1_on_set(y_true, y_score) -> Tuple[float, float]:
    # diagnostic only (not for deployment)
    order = np.argsort(y_score)
    thr_candidates = np.unique(y_score[order])
    best = (0.0, 0.5)
    for t in thr_candidates:
        f1 = f1_at_threshold(y_true, y_score, t)
        if f1 > best[0]:
            best = (f1, float(t))
    return best


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        if not np.any(mask):
            continue
        acc = y_true[mask].mean() if mask.sum() > 0 else 0.0
        conf = y_prob[mask].mean() if mask.sum() > 0 else 0.0
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def brier(y_true, y_prob) -> float:
    return float(brier_score_loss(y_true, y_prob))