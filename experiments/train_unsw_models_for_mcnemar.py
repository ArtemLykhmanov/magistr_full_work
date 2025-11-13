from __future__ import annotations

import argparse
from pathlib import Path
import sys
import joblib
import numpy as np

# Make project src importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.pipeline import Pipeline

from src.config import Paths
from src.data_io import load_csv, unsw_files
from src.preprocess import build_preprocessor
from src.models import make_models
from src.metrics_ext import pr_auc, roc_auc, f1_at_threshold


def main():
    parser = argparse.ArgumentParser(description="Train RF/XGB pipelines on UNSW and save for McNemar eval")
    parser.add_argument("--out-dir", type=Path, default=Path("out/models"))
    args = parser.parse_args()

    paths = Paths.from_root('.')
    files = unsw_files(paths.data)
    Xtr_raw, ytr = load_csv(files["train"])  # binary labels
    Xte_raw, yte = load_csv(files["test"])   # binary labels

    pre = build_preprocessor(Xtr_raw)

    models = make_models(include_calibrated_rf=False)
    to_train = {}
    if "rf" in models:
        to_train["rf"] = models["rf"]
    if "xgb" in models and models["xgb"] is not None:
        to_train["xgb"] = models["xgb"]

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for name, est in to_train.items():
        print(f"Training {name} on UNSW train...")
        pipe = Pipeline([
            ("prep", pre),
            ("est", est),
        ])
        pipe.fit(Xtr_raw, ytr)
        # quick eval on test
        try:
            import numpy as np
            proba = pipe.predict_proba(Xte_raw)
            pos_idx = 1 if proba.shape[1] > 1 else 0
            y_score = proba[:, pos_idx]
            print(f"{name} PR-AUC={pr_auc(yte, y_score):.4f} ROC-AUC={roc_auc(yte, y_score):.4f} F1@0.5={f1_at_threshold(yte, y_score, 0.5):.4f}")
        except Exception:
            pass
        out_path = args.out_dir / f"{name}.joblib"
        joblib.dump(pipe, out_path)
        print(f"Saved {name} pipeline to: {out_path}")

    missing = []
    if (args.out_dir / "rf.joblib").exists() is False:
        missing.append("rf.joblib")
    if (args.out_dir / "xgb.joblib").exists() is False:
        missing.append("xgb.joblib")
    if missing:
        print("Warning: missing outputs:", ", ".join(missing))


if __name__ == "__main__":
    main()

