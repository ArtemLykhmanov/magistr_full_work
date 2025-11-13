from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


# Make project src importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.dummy import DummyClassifier

from src.config import Paths
from src.data_io import load_csv, unsw_files, cicids_day_files
from src.preprocess import build_preprocessor
from src.models import make_models


def _load_unsw(paths: Paths):
    files = unsw_files(paths.data)
    Xtr, ytr = load_csv(files["train"])  # binary labels
    Xte, yte = load_csv(files["test"])   # binary labels
    return Xtr, ytr, Xte, yte


def _load_cicids(paths: Paths, train_days: list[str], test_day: str):
    files = cicids_day_files(paths.data)
    # concatenate multiple train days if needed
    Xtr_list = []
    ytr_list = []
    for d in train_days:
        Xd, yd = load_csv(files[d])
        Xtr_list.append(Xd)
        ytr_list.append(yd)
    Xtr = pd.concat(Xtr_list, axis=0).reset_index(drop=True)
    ytr = pd.concat(ytr_list, axis=0).reset_index(drop=True)
    Xte, yte = load_csv(files[test_day])
    return Xtr, ytr, Xte, yte


def compute_f1_curve(y_true: np.ndarray, y_score: np.ndarray, n_points: int = 101):
    thresholds = np.linspace(0.0, 1.0, n_points)
    f1s = [f1_score(y_true, (y_score >= t).astype(int), zero_division=0) for t in thresholds]
    return thresholds, np.asarray(f1s)


def plot_overlay(th1, f1_1, th2, f1_2, label1: str, label2: str, out: Path, title: str):
    plt.figure(figsize=(6, 4))
    plt.plot(th1, f1_1, label=label1, lw=2)
    plt.plot(th2, f1_2, label=label2, lw=2)
    plt.xlabel("Threshold")
    plt.ylabel("F1")
    plt.title(title)
    plt.grid(True)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot F1 vs threshold for Dummy vs a main model")
    parser.add_argument("--dataset", choices=["unsw", "cicids"], default="unsw")
    parser.add_argument("--cicids-train-days", nargs="+", default=["tuesday", "wednesday"],
                        help="Days to concatenate for CICIDS training")
    parser.add_argument("--cicids-test-day", default="thursday_web",
                        help="CICIDS test day key (e.g., thursday_web, friday_ddos)")
    parser.add_argument("--model", choices=["logreg", "rf", "hgb", "mlp", "xgb"], default="logreg")
    parser.add_argument("--dummy-strategy", choices=["most_frequent", "stratified", "uniform"], default="most_frequent")
    parser.add_argument("--out", type=Path, default=Path("out/fig_3_6_f1_threshold_dummy_vs_model.png"))
    args = parser.parse_args()

    paths = Paths.from_root('.')

    if args.dataset == "unsw":
        Xtr, ytr, Xte, yte = _load_unsw(paths)
        dataset_title = "UNSW → UNSW"
    else:
        Xtr, ytr, Xte, yte = _load_cicids(paths, args.cicids_train_days, args.cicids_test_day)
        ds_tr = "+".join(args.cicids_train_days)
        dataset_title = f"CIC({ds_tr}) → {args.cicids_test_day.upper()}"

    # Build preprocessor and select model
    pre = build_preprocessor(Xtr)
    models = make_models(include_calibrated_rf=False)
    if args.model not in models:
        # fallback to logreg if not available
        model_name = "logreg"
    else:
        model_name = args.model
    clf = models[model_name]

    pipe = Pipeline([
        ("prep", pre),
        ("clf", clf),
    ])
    pipe.fit(Xtr, ytr)
    # main model positive class probability
    proba = pipe.predict_proba(Xte)
    # find column index for class 1
    classes = getattr(pipe.named_steps["clf"], "classes_", np.array([0, 1]))
    try:
        pos_idx = int(np.where(classes == 1)[0][0])
    except Exception:
        pos_idx = 1 if proba.shape[1] > 1 else 0
    y_score_model = proba[:, pos_idx]

    # Dummy baseline probabilities (constant features; Dummy ignores features)
    dummy = DummyClassifier(strategy=args.dummy_strategy, random_state=42)
    dummy.fit(np.zeros((len(ytr), 1)), ytr)
    y_score_dummy = dummy.predict_proba(np.zeros((len(yte), 1)))[:, 1 if 1 in getattr(dummy, 'classes_', [0,1]) else 0]

    th_m, f1_m = compute_f1_curve(np.asarray(yte), y_score_model)
    th_d, f1_d = compute_f1_curve(np.asarray(yte), y_score_dummy)

    title = f"F1 vs Threshold: Dummy({args.dummy_strategy}) vs {model_name} — {dataset_title}"
    plot_overlay(th_d, f1_d, th_m, f1_m, f"Dummy ({args.dummy_strategy})", model_name, args.out, title)
    print(f"Saved figure: {args.out}")


if __name__ == "__main__":
    main()

