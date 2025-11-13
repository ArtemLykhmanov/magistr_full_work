from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

# Make project src importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Paths
from src.data_io import load_csv, unsw_files
from src.preprocess import build_preprocessor


def rf_model(seed: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=300, max_depth=None, random_state=seed, n_jobs=-1)


def shap_importance(model, X_sample: np.ndarray) -> Tuple[np.ndarray, str]:
    """Return (importances, method_label). Falls back to RF feature_importances_ if SHAP not available."""
    try:
        import shap  # type: ignore
        try:
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X_sample)
            if isinstance(sv, list):
                if hasattr(model, 'classes_') and 1 in getattr(model, 'classes_', [1]):
                    idx = int(np.where(model.classes_ == 1)[0][0])
                else:
                    idx = min(1, len(sv) - 1)
                S = sv[idx]
            else:
                S = sv
        except Exception:
            explainer = shap.Explainer(model, X_sample[: min(256, len(X_sample))])
            S = explainer(X_sample).values
        S = np.asarray(S)
        if S.ndim == 3:
            S = S[..., 0]
        imp = np.mean(np.abs(S), axis=0)
        return imp, "shap"
    except Exception:
        # Fallback: tree-based impurity gain importances
        imp = getattr(model, 'feature_importances_', None)
        if imp is None:
            raise
        return np.asarray(imp), "rf_feature_importance"


def ranks_from_importance(imp: np.ndarray) -> np.ndarray:
    # Higher importance => higher rank; convert to ranks where larger better
    # Spearman needs ranks; use dense ranking with average ties
    import pandas as pd
    s = pd.Series(imp)
    r = s.rank(method='average', ascending=False).to_numpy()
    return r


def spearman_from_ranks(r1: np.ndarray, r2: np.ndarray) -> float:
    # Spearman = Pearson of ranks
    r1 = np.asarray(r1, dtype=float)
    r2 = np.asarray(r2, dtype=float)
    if r1.size != r2.size or r1.size == 0:
        return float('nan')
    c = np.corrcoef(r1, r2)[0, 1]
    return float(c)


def jaccard_topk(idx1: np.ndarray, idx2: np.ndarray, k: int) -> float:
    A = set(idx1[:k].tolist())
    B = set(idx2[:k].tolist())
    if not A and not B:
        return 1.0
    return float(len(A & B) / len(A | B))


def kuncheva_topk(idx1: np.ndarray, idx2: np.ndarray, k: int, d: int) -> float:
    # Kuncheva stability index: (|A∩B| - k^2/d) / (k - k^2/d)
    A = set(idx1[:k].tolist())
    B = set(idx2[:k].tolist())
    r = len(A & B)
    denom = (k - (k * k) / max(d, 1))
    if denom <= 0:
        return float('nan')
    return float((r - (k * k) / max(d, 1)) / denom)


def main():
    parser = argparse.ArgumentParser(description="SHAP stability analysis on UNSW with RandomForest")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--sample-size", type=int, default=800, help="Rows from test set for SHAP")
    parser.add_argument("--out-dir", type=Path, default=Path("out/unsw_shap"))
    parser.add_argument("--topk", nargs="*", type=int, default=[10, 20])
    args = parser.parse_args()

    paths = Paths.from_root('.')
    files = unsw_files(paths.data)
    Xtr_raw, ytr = load_csv(files['train'])
    Xte_raw, yte = load_csv(files['test'])

    pre = build_preprocessor(Xtr_raw)
    Xtr = pre.fit_transform(Xtr_raw, ytr)
    Xte = pre.transform(Xte_raw)

    d = Xtr.shape[1]
    names = []
    try:
        names = list(pre.get_feature_names_out())
    except Exception:
        names = [f"f{i}" for i in range(d)]

    # SHAP on a fixed subset from test for comparability
    n = min(args.sample_size, Xte.shape[0])
    rng = np.random.default_rng(42)
    idx_sample = rng.choice(Xte.shape[0], size=n, replace=False)
    Xsh = Xte[idx_sample]

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
    imps: List[np.ndarray] = []
    orders: List[np.ndarray] = []
    ranks: List[np.ndarray] = []
    method_used = None

    run = 0
    for tr_idx, _ in skf.split(Xtr, ytr):
        run += 1
        model = rf_model(seed=42 + run)
        model.fit(Xtr[tr_idx], np.asarray(ytr)[tr_idx])
        imp, method = shap_importance(model, Xsh)
        if method_used is None:
            method_used = method
        imps.append(imp)
        ord_idx = np.argsort(imp)[::-1]
        orders.append(ord_idx)
        ranks.append(ranks_from_importance(imp))

    # Pairwise stability metrics
    pairs_rows = []
    m = len(imps)
    for i in range(m):
        for j in range(i + 1, m):
            sp = spearman_from_ranks(ranks[i], ranks[j])
            row = {"run_i": i, "run_j": j, "spearman": sp}
            for k in args.topk:
                jac = jaccard_topk(orders[i], orders[j], k)
                kun = kuncheva_topk(orders[i], orders[j], k, d)
                row[f"jaccard@{k}"] = jac
                row[f"kuncheva@{k}"] = kun
            pairs_rows.append(row)

    pairs_df = pd.DataFrame(pairs_rows)

    # Summary (mean/std over pairs)
    agg = pairs_df.agg(['mean', 'std']).reset_index().rename(columns={'index': 'stat'})

    # Output directories
    outdir = args.out_dir
    tdir = outdir / 'tables'
    fdir = outdir / 'figures'
    tdir.mkdir(parents=True, exist_ok=True)
    fdir.mkdir(parents=True, exist_ok=True)

    pairs_csv = tdir / 'table_unsw_shap_stability_pairs.csv'
    pairs_df.to_csv(pairs_csv, index=False)

    summary_csv = tdir / 'table_unsw_shap_stability_summary.csv'
    agg.to_csv(summary_csv, index=False)

    # Heatmap of Spearman
    try:
        mat = np.ones((m, m), dtype=float)
        for i in range(m):
            for j in range(i + 1, m):
                sp = pairs_df[(pairs_df['run_i'] == i) & (pairs_df['run_j'] == j)]['spearman'].values
                mat[i, j] = mat[j, i] = float(sp[0]) if len(sp) else np.nan
        fig, ax = plt.subplots(figsize=(4 + 0.5 * m, 4 + 0.5 * m))
        im = ax.imshow(mat, vmin=0, vmax=1, cmap='viridis')
        title = 'Spearman rank corr. (SHAP importances)'
        if method_used != 'shap':
            title = 'Spearman rank corr. (RF feature_importances_)'
        ax.set_title(title)
        ax.set_xticks(range(m))
        ax.set_yticks(range(m))
        ax.set_xticklabels([f"r{i}" for i in range(m)])
        ax.set_yticklabels([f"r{i}" for i in range(m)])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(fdir / 'heatmap_spearman.png', dpi=200)
        plt.close(fig)
    except Exception:
        pass

    # Bars for Jaccard@k (mean over pairs)
    try:
        means = []
        labels = []
        for k in args.topk:
            labels.append(f"Jaccard@{k}")
            means.append(float(pairs_df[f"jaccard@{k}"].mean()))
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(labels, means, color=['#6699cc', '#66cc99'])
        ax.set_ylim(0, 1.05)
        if method_used == 'shap':
            ax.set_title('Mean Jaccard overlap of top-k SHAP features (UNSW)')
        else:
            ax.set_title('Mean Jaccard overlap of top-k RF importances (UNSW)')
        for i, v in enumerate(means):
            ax.text(i, v + 0.02, f"{v:.2f}", ha='center')
        fig.tight_layout()
        fig.savefig(fdir / 'bar_jaccard.png', dpi=200)
        plt.close(fig)
    except Exception:
        pass

    print("SHAP stability (UNSW, RF)")
    print(pairs_df.to_string(index=False))
    print("\nSummary:")
    print(agg.to_string(index=False))
    print(f"\nSaved pairs: {pairs_csv}")
    print(f"Saved summary: {summary_csv}")
    print(f"Saved figures to: {fdir}")


if __name__ == "__main__":
    main()
