"""
Study stability of a fixed threshold (picked on validation) across CICIDS days.
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.data_io import cicids_day_files, load_csv
from src.eval_utils import threshold_for_target_fpr
from sklearn.metrics import recall_score, confusion_matrix
from joblib import load

SEED = 42
TARGET_FPR = 0.001
MODEL_PATH = "out/models/rf_cal.joblib"
VAL_DAY = "monday"


if __name__ == "__main__":
    out_dir = "out/stability"
    os.makedirs(out_dir, exist_ok=True)

    files = cicids_day_files(Path("data"))
    day_order = ["monday", "tuesday", "wednesday", "thursday_web", "thursday_inf", "friday_port", "friday_ddos"]

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")
    model = load(MODEL_PATH)

    # Pick threshold on validation day
    X_val, y_val = load_csv(files[VAL_DAY])
    proba_val = model.predict_proba(X_val)[:, 1]
    tau = threshold_for_target_fpr(y_val, proba_val, TARGET_FPR)

    results = []
    for day in day_order:
        X_day, y_day = load_csv(files[day])
        proba = model.predict_proba(X_day)[:, 1]
        y_pred = (proba >= tau).astype(int)
        recall = recall_score(y_day, y_pred, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_day, y_pred).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        results.append({"day": day, "recall": recall, "fpr": fpr})

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(out_dir, "threshold_stability.csv"), index=False)

    # Plot
    plt.figure(figsize=(6, 4))
    plt.plot(df["day"], df["recall"], marker='o', label='Recall')
    plt.plot(df["day"], df["fpr"], marker='s', label='FPR')
    plt.xticks(rotation=45)
    plt.ylabel("Score")
    plt.title(f"Threshold stability at tau={tau:.3f} (val={VAL_DAY})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_threshold_stability.png"))
    plt.close()

    print("Done: saved CSV and figure in out/stability")

