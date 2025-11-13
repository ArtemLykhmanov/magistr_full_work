from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from typing import Tuple

import joblib

# Make project src importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Paths
from src.data_io import load_csv, unsw_files


def load_models(rf_path: Path, xgb_path: Path):
    if not rf_path.exists():
        raise FileNotFoundError(f"Missing RF model file: {rf_path}")
    if not xgb_path.exists():
        raise FileNotFoundError(f"Missing XGB model file: {xgb_path}")
    rf = joblib.load(rf_path)
    xgb = joblib.load(xgb_path)
    return rf, xgb


def to_labels(model, X) -> np.ndarray:
    # Prefer predict() if available
    if hasattr(model, "predict"):
        y_pred = model.predict(X)
        # Some models may return probabilities from predict; guard
        y_pred = np.asarray(y_pred)
        if y_pred.ndim == 2 and y_pred.shape[1] > 1:
            # choose class 1
            return (y_pred[:, 1] >= 0.5).astype(int)
        if y_pred.ndim == 1 and (y_pred.dtype.kind == 'f'):
            # float scores: threshold at 0.5
            return (y_pred >= 0.5).astype(int)
        return y_pred.astype(int)
    # Otherwise use predict_proba if possible
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        proba = np.asarray(proba)
        pos_idx = 1 if proba.shape[1] > 1 else 0
        return (proba[:, pos_idx] >= 0.5).astype(int)
    raise AttributeError("Model has neither predict nor predict_proba")


def mcnemar_stats(y_true: np.ndarray, y_a: np.ndarray, y_b: np.ndarray) -> Tuple[int, int, float]:
    # b: A wrong and B correct; c: B wrong and A correct
    y_true = np.asarray(y_true)
    a = np.asarray(y_a)
    b = np.asarray(y_b)
    a_err = (a != y_true)
    b_err = (b != y_true)
    b_count = int(np.sum(a_err & ~b_err))
    c_count = int(np.sum(~a_err & b_err))
    table = [[0, b_count], [c_count, 0]]
    p_value = None
    try:
        from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar
        res = sm_mcnemar(table, exact=True)
        p_value = float(res.pvalue)
    except Exception:
        # Fallback to built-in implementation (binomial test) via eval_utils
        try:
            from src.eval_utils import mcnemar_test as local_mcnemar
            res = local_mcnemar(y_true, a, b)
            p_value = float(res.get("p_value", float("nan")))
        except Exception:
            p_value = float("nan")
    return b_count, c_count, p_value


def main():
    parser = argparse.ArgumentParser(description="McNemar test: RF vs XGB on UNSW test set")
    parser.add_argument("--rf", type=Path, default=Path("out/models/rf.joblib"))
    parser.add_argument("--xgb", type=Path, default=Path("out/models/xgb.joblib"))
    args = parser.parse_args()

    paths = Paths.from_root('.')
    files = unsw_files(paths.data)
    X_test, y_test = load_csv(files["test"])  # binary 0/1 labels

    rf, xgb = load_models(args.rf, args.xgb)

    # Generate predictions (assumes models can accept raw features — ideally pipelines)
    try:
        y_rf = to_labels(rf, X_test)
    except Exception as e:
        raise RuntimeError(f"RF predict failed. Ensure the saved RF is a pipeline that accepts raw features. Error: {e}")
    try:
        y_xgb = to_labels(xgb, X_test)
    except Exception as e:
        raise RuntimeError(f"XGB predict failed. Ensure the saved XGB is a pipeline that accepts raw features. Error: {e}")

    b, c, p = mcnemar_stats(y_test, y_rf, y_xgb)

    print("McNemar test: RF vs XGB on UNSW test")
    print(f"b (RF wrong, XGB correct): {b}")
    print(f"c (XGB wrong, RF correct): {c}")
    print(f"p-value: {p}")
    if p == p and p < 0.05:  # p==p guards nan
        print("Result: difference is statistically significant (p < 0.05)")
    else:
        print("Result: no statistically significant difference (p >= 0.05 or p is NaN)")


if __name__ == "__main__":
    main()

