from __future__ import annotations

import argparse
from pathlib import Path
import sys
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Make project src importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Paths
from src.data_io import load_csv, unsw_files
from src.preprocess import build_preprocessor
from src.metrics_ext import pr_auc, roc_auc, f1_at_threshold
from src.plots import f1_vs_threshold_curve


@dataclass
class EvalResult:
    tag: str
    n_features: str
    pr_auc: float
    roc_auc: float
    f1_0_5: float


def rf_model(random_state: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=300, max_depth=None, random_state=random_state, n_jobs=-1)


def get_feature_names(preprocessor, X_sample: pd.DataFrame) -> List[str]:
    try:
        names = preprocessor.get_feature_names_out()
        return [str(n) for n in names]
    except Exception:
        # Fallback: try to transform one row and infer length only
        Xt = preprocessor.fit_transform(X_sample.head(5))
        return [f"f{i}" for i in range(Xt.shape[1])]


def get_base_feature_names(preprocessor) -> List[str]:
    """Map each transformed column to its originating raw feature name.

    Uses ColumnTransformer.transformers_ to expand categorical one-hot widths.
    """
    base_names: List[str] = []
    try:
        for name, trans, cols in preprocessor.transformers_:
            # Pipeline expected for our preprocess; handle common cases
            if hasattr(trans, 'named_steps'):
                if name == 'num':
                    # one column per numeric
                    for c in cols:
                        base_names.append(str(c))
                elif name == 'cat_low':
                    ohe = trans.named_steps.get('ohe')
                    if hasattr(ohe, 'categories_'):
                        for c, cats in zip(cols, ohe.categories_):
                            width = len(cats)
                            base_names.extend([str(c)] * width)
                    else:
                        # Fallback: assume small expansion of unknown width
                        base_names.extend([str(c) for c in cols])
                elif name == 'cat_high':
                    # OrdinalEncoder -> one column per categorical feature
                    for c in cols:
                        base_names.append(str(c))
                else:
                    # Unknown transformer: best-effort one-to-one
                    for c in cols:
                        base_names.append(str(c))
            else:
                # passthrough/array: try to expand cols as one-to-one
                for c in cols:
                    base_names.append(str(c))
        return base_names
    except Exception:
        # Fallback to feature_names length if available
        try:
            n = len(preprocessor.get_feature_names_out())
        except Exception:
            n = 0
        return [f"base_{i}" for i in range(n)]


def evaluate_on_set(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float, float]:
    return pr_auc(y_true, y_score), roc_auc(y_true, y_score), f1_at_threshold(y_true, y_score, 0.5)


def run_ablation(top_ns: List[int], out_tables: Path, out_figures: Path, seed: int = 42) -> Tuple[pd.DataFrame, List[str]]:
    paths = Paths.from_root('.')
    files = unsw_files(paths.data)
    X_train, y_train = load_csv(files["train"])  # binary labels
    X_test, y_test = load_csv(files["test"])    # binary labels

    # Build and fit preprocessing on train for consistency
    pre = build_preprocessor(X_train)
    # Fit on train, transform both sets
    Xtr = pre.fit_transform(X_train, y_train)
    Xte = pre.transform(X_test)
    feature_names = get_feature_names(pre, X_train)

    # Baseline model on all features
    rf_all = rf_model(seed)
    rf_all.fit(Xtr, y_train)
    proba_all = rf_all.predict_proba(Xte)[:, 1]
    pr_all, roc_all, f1_all = evaluate_on_set(np.asarray(y_test), proba_all)

    rows: List[EvalResult] = []
    rows.append(EvalResult(tag="rf_all", n_features=str(Xtr.shape[1]), pr_auc=pr_all, roc_auc=roc_all, f1_0_5=f1_all))

    # Importance ranking on all features
    importances = rf_all.feature_importances_
    order = np.argsort(importances)[::-1]

    # Determine best-N by F1 on a validation split from train
    try:
        tr_idx, val_idx = train_test_split(
            np.arange(Xtr.shape[0]), test_size=0.2, random_state=seed, stratify=y_train
        )
        best = (None, -1.0)  # (n, f1)
        unique_candidates = sorted(set([min(n, Xtr.shape[1]) for n in top_ns]))
        for n in unique_candidates:
            keep_idx = order[:n]
            rf_n = rf_model(seed)
            rf_n.fit(Xtr[tr_idx][:, keep_idx], np.asarray(y_train)[tr_idx])
            proba_val = rf_n.predict_proba(Xtr[val_idx][:, keep_idx])[:, 1]
            f1_val = f1_at_threshold(np.asarray(y_train)[val_idx], proba_val, 0.5)
            if f1_val > best[1] or (f1_val == best[1] and (best[0] is None or n < best[0])):
                best = (n, f1_val)
        best_n = best[0]
    except Exception:
        best_n = None

    # For plotting convenience, collect points
    plot_points = [(Xtr.shape[1], f1_all, pr_all)]

    for n in top_ns:
        n_eff = min(n, Xtr.shape[1])
        keep_idx = order[:n_eff]
        # Slice transformed matrices
        Xtr_n = Xtr[:, keep_idx]
        Xte_n = Xte[:, keep_idx]

        rf_n = rf_model(seed)
        rf_n.fit(Xtr_n, y_train)
        proba_n = rf_n.predict_proba(Xte_n)[:, 1]
        pr_n, roc_n, f1_n = evaluate_on_set(np.asarray(y_test), proba_n)

        rows.append(EvalResult(tag=f"rf_top{n_eff}", n_features=str(n_eff), pr_auc=pr_n, roc_auc=roc_n, f1_0_5=f1_n))
        plot_points.append((n_eff, f1_n, pr_n))

    # Add best-N row if available
    if best_n is not None and best_n > 0:
        keep_idx = order[:best_n]
        rf_b = rf_model(seed)
        rf_b.fit(Xtr[:, keep_idx], y_train)
        proba_b = rf_b.predict_proba(Xte[:, keep_idx])[:, 1]
        pr_b, roc_b, f1_b = evaluate_on_set(np.asarray(y_test), proba_b)
        rows.append(EvalResult(tag=f"rf_top{best_n}_bestF1", n_features=str(best_n), pr_auc=pr_b, roc_auc=roc_b, f1_0_5=f1_b))
        plot_points.append((best_n, f1_b, pr_b))

    # Prepare output table
    df = pd.DataFrame([r.__dict__ for r in rows])
    out_tables.mkdir(parents=True, exist_ok=True)
    out_figures.mkdir(parents=True, exist_ok=True)
    csv_path = out_tables / "table_3_feature_ablation.csv"
    df.to_csv(csv_path, index=False)

    # Plot F1 and PR-AUC vs number of features
    try:
        import matplotlib.pyplot as plt
        pts = sorted(plot_points, key=lambda t: t[0], reverse=True)
        xs = [p[0] for p in pts]
        f1s = [p[1] for p in pts]
        prs = [p[2] for p in pts]
        fig, ax1 = plt.subplots(figsize=(6, 4))
        ax1.plot(xs, f1s, marker='o', label='F1@0.5')
        ax1.plot(xs, prs, marker='s', label='PR-AUC')
        ax1.set_xlabel('Number of features (top-N or All)')
        ax1.set_ylabel('Score')
        ax1.set_title('Feature Ablation on UNSW (RF)')
        ax1.set_ylim(0, 1.05)
        ax1.grid(True)
        ax1.legend()
        fig.tight_layout()
        fig_path = out_figures / "fig_3_feature_ablation.png"
        fig.savefig(fig_path, dpi=200)
        plt.close(fig)
    except Exception:
        pass

    # Return dataframe and a readable feature report for the top model
    base_names = get_base_feature_names(pre)
    top_k = max(top_ns) if top_ns else 0
    top_features = []
    for i in order[:top_k]:
        fname = feature_names[i] if i < len(feature_names) else f"f{i}"
        bname = base_names[i] if i < len(base_names) else fname
        top_features.append(f"{fname} -> {bname}")
    return df, top_features


def main():
    parser = argparse.ArgumentParser(description="Feature selection ablation on UNSW using RandomForest")
    parser.add_argument("--top-n", nargs="*", type=int, default=[20, 10], help="Top-N feature counts to evaluate")
    args = parser.parse_args()

    out_tables = Path("out/tables")
    out_figures = Path("out/figures")
    df, top_feats = run_ablation(args.top_n, out_tables, out_figures)

    print("Feature Ablation Results (UNSW, RF):")
    print(df.to_string(index=False))
    print("\nTop features by importance (first 20 shown):")
    print("\n".join(top_feats[:20]))
    print(f"\nSaved table: {out_tables / 'table_3_feature_ablation.csv'}")
    print(f"Saved figure: {out_figures / 'fig_3_feature_ablation.png'}")


if __name__ == "__main__":
    main()
