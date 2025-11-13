"""
Plot F1, Recall and FPR vs. decision threshold from a CSV with y_true,y_score.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, recall_score, confusion_matrix


if __name__ == "__main__":
    input_path = "out/preds/test_rf.csv"  # CSV with y_true, y_score
    out_fig = "out/figures/f1_recall_fpr_vs_threshold.png"

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Missing prediction file: {input_path}")

    df = pd.read_csv(input_path)
    y_true = df['y_true'].values
    y_score = df['y_score'].values

    thresholds = np.linspace(0.0, 1.0, 100)
    f1s, recalls, fprs = [], [], []

    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        f1s.append(f1_score(y_true, y_pred, zero_division=0))
        recalls.append(recall_score(y_true, y_pred, zero_division=0))
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fprs.append(fpr)

    plt.figure(figsize=(6, 4))
    plt.plot(thresholds, f1s, label="F1-score")
    plt.plot(thresholds, recalls, label="Recall")
    plt.plot(thresholds, fprs, label="FPR")
    plt.xlabel("Threshold")
    plt.ylabel("Metric")
    plt.title("F1, Recall and FPR vs. Threshold")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_fig)
    plt.close()

    print(f"Saved plot -> {out_fig}")

