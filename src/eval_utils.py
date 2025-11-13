from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.stats import binomtest


def threshold_for_target_fpr(y_true, y_score, target_fpr: float) -> float:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    neg_scores = y_score[y_true == 0]
    if len(neg_scores) == 0:
        return 1.0  # if no negatives, make threshold impossible
    # We need fraction of negatives misclassified as positive to be <= target_fpr
    # Sort descending, choose cutoff where FP/N_neg <= target_fpr
    order = np.argsort(-neg_scores)
    neg_sorted = neg_scores[order]
    N = len(neg_sorted)
    k = int(np.floor(target_fpr * N))
    if k <= 0:
        # choose threshold above max negative score
        return float(np.nextafter(neg_sorted.max(), 1.0))
    thr = float(neg_sorted[k - 1])
    return thr


def mcnemar_test(y_true, y_pred_a, y_pred_b) -> Dict[str, Any]:
    y_true = np.asarray(y_true)
    a = np.asarray(y_pred_a)
    b = np.asarray(y_pred_b)
    # b = errors where A wrong and B right; c = A right and B wrong
    a_err = (a != y_true)
    b_err = (b != y_true)
    b_count = int(np.sum(a_err & ~b_err))
    c_count = int(np.sum(~a_err & b_err))
    n = b_count + c_count
    if n == 0:
        p = 1.0
    else:
        p = binomtest(min(b_count, c_count), n, 0.5, alternative='two-sided').pvalue
    return {"b": b_count, "c": c_count, "n": n, "p_value": float(p)}


def save_manifest(outdir: Path, params: Dict[str, Any]):
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "params": params,
    }
    (outdir / "logs").mkdir(exist_ok=True, parents=True)
    with open(outdir / "logs" / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        
def compute_shap_feature_importance(model, X, outpath):
    import shap
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X[:100]) # subset for speed
    shap.plots.bar(shap_values, max_display=10, show=False)
    import matplotlib.pyplot as plt
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()