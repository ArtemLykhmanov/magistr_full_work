from __future__ import annotations
from typing import List, Dict
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc

plt.rcParams["figure.dpi"] = 150


def bar_pr_auc_by_pair(pairs: List[str], models: List[str], pr_values: Dict[str, Dict[str, float]], outpath: str):
    idx = np.arange(len(pairs))
    width = 0.8 / max(1, len(models))
    fig, ax = plt.subplots(figsize=(max(8, len(pairs) * 0.4), 4))
    for j, m in enumerate(models):
        vals = [pr_values.get(p, {}).get(m, np.nan) for p in pairs]
        ax.bar(idx + j * width, vals, width=width, label=m)
    ax.set_xticks(idx + width * (len(models) - 1) / 2)
    ax.set_xticklabels(pairs, rotation=45, ha='right')
    ax.set_ylabel('PR-AUC')
    ax.set_title('PR-AUC by Train->Test Pair')
    ax.legend(ncols=min(4, len(models)))
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def reliability_diagram(y_true, y_prob, bins: int, outpath: str, strategy: str = 'uniform'):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if strategy == 'quantile':
        qs = np.linspace(0, 1, bins + 1)
        edges = np.unique(np.quantile(y_prob, qs))
        if len(edges) < 2:
            edges = np.array([0.0, 1.0])
    else:
        edges = np.linspace(0, 1, bins + 1)
    conf = []
    acc = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        m = (y_prob >= lo) & (y_prob < hi if i < len(edges) - 2 else y_prob <= hi)
        if m.sum() == 0:
            conf.append(np.nan)
            acc.append(np.nan)
        else:
            conf.append(y_prob[m].mean())
            acc.append(y_true[m].mean())
    conf = np.asarray(conf)
    acc = np.asarray(acc)
    mask = ~np.isnan(conf) & ~np.isnan(acc)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot([0, 1], [0, 1], linestyle='--')
    if mask.any():
        ax.plot(conf[mask], acc[mask], marker='o')
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Accuracy')
    ax.set_title('Reliability Diagram')
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def f1_vs_threshold_curve(y_true, y_score, outpath: str):
    """Plot F1 vs threshold and save to outpath."""
    from sklearn.metrics import f1_score

    thresholds = np.linspace(0.0, 1.0, 100)
    f1s = [f1_score(y_true, y_score >= t, zero_division=0) for t in thresholds]
    plt.figure(figsize=(5, 4))
    plt.plot(thresholds, f1s, label='F1')
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.title('F1 vs. Threshold')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def pr_curve_overlay(y_true, probs_by_model: Dict[str, np.ndarray], outpath: str):
    """Plot precision-recall curves for multiple models on the same axes.

    Robust to NaNs in prediction arrays: filters non-finite values per model.
    """
    y_true = np.asarray(y_true)
    plt.figure(figsize=(6, 4))
    for name, probs in probs_by_model.items():
        try:
            p = np.asarray(probs, dtype=float)
            m = np.isfinite(p)
            if not np.any(m):
                continue
            precision, recall, _ = precision_recall_curve(y_true[m], p[m])
            pr_auc_val = auc(recall, precision)
            plt.plot(recall, precision, label=f"{name} (AUPR={pr_auc_val:.3f})")
        except Exception:
            continue
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def roc_curve_overlay(y_true, probs_by_model: Dict[str, np.ndarray], outpath: str):
    """Plot ROC curves for multiple models on the same axes.

    Robust to NaNs in prediction arrays: filters non-finite values per model.
    """
    y_true = np.asarray(y_true)
    plt.figure(figsize=(6, 4))
    for name, probs in probs_by_model.items():
        try:
            p = np.asarray(probs, dtype=float)
            m = np.isfinite(p)
            if not np.any(m):
                continue
            fpr, tpr, _ = roc_curve(y_true[m], p[m])
            roc_auc_val = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_val:.3f})")
        except Exception:
            continue
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def bar_metrics(models: List[str], values: Dict[str, float], ylabel: str, outpath: str):
    """Simple bar chart for a single metric across models."""
    vals = [values.get(m, np.nan) for m in models]
    idx = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(max(6, len(models) * 1.2), 4))
    ax.bar(idx, vals, width=0.6)
    ax.set_xticks(idx)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel + ' by model')
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def feature_importance_heatmap(df_importance: pd.DataFrame, outpath: str, top_n: int = 30, title: str | None = None):
    """Draw a heatmap of feature importance across models.

    df_importance: columns ['model','feature','importance'] (importance already normalized per model)
    """
    try:
        pivot = df_importance.pivot_table(index="feature", columns="model", values="importance", aggfunc="sum", fill_value=0.0)
        totals = pivot.sum(axis=1)
        top_features = totals.sort_values(ascending=False).head(top_n).index
        P = pivot.loc[top_features]
        data = P.values
        features = list(P.index)
        models = list(P.columns)

        h = max(4, 0.3 * len(features) + 1)
        w = max(6, 0.6 * len(models) + 1)
        fig, ax = plt.subplots(figsize=(w, h))
        im = ax.imshow(data, aspect='auto', cmap='magma')
        ax.set_xticks(np.arange(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_yticks(np.arange(len(features)))
        ax.set_yticklabels(features)
        ax.set_xlabel('Model')
        ax.set_ylabel('Feature')
        ax.set_title(title or 'Feature Importance Across Models')
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Normalized Importance')
        fig.tight_layout()
        fig.savefig(outpath)
        plt.close(fig)
    except Exception:
        pass


def feature_importance_heatmap_overall(df_importance: pd.DataFrame, outpath: str, top_n: int = 20, title: str | None = None):
    """Heatmap of aggregated importance across all models (single column).

    Shows top_n features by total importance summed over models.
    """
    try:
        s = df_importance.groupby('feature', as_index=True)['importance'].sum().sort_values(ascending=False)
        if top_n is not None and top_n > 0:
            s = s.head(top_n)
        data = s.values.reshape(-1, 1)
        features = list(s.index)
        models = ['overall']
        h = max(4, 0.3 * len(features) + 1)
        fig, ax = plt.subplots(figsize=(4, h))
        im = ax.imshow(data, aspect='auto', cmap='magma')
        ax.set_xticks([0])
        ax.set_xticklabels(models)
        ax.set_yticks(np.arange(len(features)))
        ax.set_yticklabels(features)
        ax.set_xlabel('')
        ax.set_ylabel('Feature')
        ax.set_title(title or f'Top {len(features)} Features (Aggregated)')
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Aggregated Importance')
        fig.tight_layout()
        fig.savefig(outpath)
        plt.close(fig)
    except Exception:
        pass
