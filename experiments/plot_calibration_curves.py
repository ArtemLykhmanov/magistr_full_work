"""
Plot a reliability diagram from a CSV with columns y_true,y_score.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve


def plot_reliability(y_true, y_prob, n_bins=10, title="", outpath="reliability.png"):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
    plt.figure(figsize=(4.5, 4.5))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect calibration')
    plt.plot(prob_pred, prob_true, marker='o', label='Model')
    plt.xlabel('Predicted probability')
    plt.ylabel('True frequency')
    plt.title(title or 'Reliability Diagram')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


if __name__ == "__main__":
    # Example: expects CSV with columns y_true, y_score
    input_path = "out/preds/val_rf_cal.csv"
    out_fig_path = "out/figures/reliability_rf_cal.png"

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Missing prediction file: {input_path}")

    df = pd.read_csv(input_path)
    y_true = df['y_true'].values
    y_score = df['y_score'].values

    plot_reliability(y_true, y_score, n_bins=10, title="RF Calibrated (CICIDS Val)", outpath=out_fig_path)
    print(f"Saved reliability diagram -> {out_fig_path}")

