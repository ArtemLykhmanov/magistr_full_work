from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def load_baseline(baseline_csv: Path):
    df = pd.read_csv(baseline_csv)
    # Ensure presence of expected columns
    expected = {"Strategy", "TrainData", "TestData", "Dummy_F1"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Baseline CSV missing columns: {missing}")
    return df


def panel_unsw(ax, baseline_df: pd.DataFrame, unsw_csv: Path):
    # Baseline for UNSW->UNSW
    base_unsw = (
        baseline_df[(baseline_df["TrainData"] == "UNSW") & (baseline_df["TestData"] == "UNSW")]
        .copy()
    )
    base_unsw["label"] = base_unsw["Strategy"].map({
        "most_frequent": "Dummy (most_frequent)",
        "stratified": "Dummy (stratified)",
        "uniform": "Dummy (uniform)",
    }).fillna(base_unsw["Strategy"])
    base_unsw = base_unsw[["label", "Dummy_F1"]].rename(columns={"Dummy_F1": "F1"})

    # Main models (UNSW test table)
    unsw = pd.read_csv(unsw_csv)
    models = unsw[["model", "f1@0.5"]].rename(columns={"model": "label", "f1@0.5": "F1"})

    plot_df = pd.concat([base_unsw, models], ignore_index=True)
    x = plot_df["label"].tolist()
    y = plot_df["F1"].tolist()
    ax.bar(range(len(y)), y, color='#6699cc')
    ax.set_xticks(range(len(x)))
    ax.set_xticklabels(x)
    ax.set_title("UNSW → UNSW (F1@0.5 for models, Dummy labels)")
    ax.set_xlabel("")
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis='x', rotation=30)
    for c in ax.containers:
        ax.bar_label(c, fmt='%.3f', padding=2, fontsize=8)


def panel_cross(ax, baseline_df: pd.DataFrame, cross_csv: Path):
    # Baseline for CICIDS->UNSW
    base = (
        baseline_df[(baseline_df["TrainData"] == "CICIDS") & (baseline_df["TestData"] == "UNSW")]
        .copy()
    )
    base["label"] = base["Strategy"].map({
        "most_frequent": "Dummy (most_frequent)",
        "stratified": "Dummy (stratified)",
        "uniform": "Dummy (uniform)",
    }).fillna(base["Strategy"])
    base = base[["label", "Dummy_F1"]].rename(columns={"Dummy_F1": "F1"})

    # Main models: select CIC(Tue+Wed)->UNSW(test)
    cross = pd.read_csv(cross_csv)
    sub = cross[cross["train->test"].astype(str).str.contains("CIC(Tue+Wed)->UNSW(test)")]
    models = sub[["model", "f1@0.5"]].rename(columns={"model": "label", "f1@0.5": "F1"})

    plot_df = pd.concat([base, models], ignore_index=True)
    x = plot_df["label"].tolist()
    y = plot_df["F1"].tolist()
    ax.bar(range(len(y)), y, color='#66cc99')
    ax.set_xticks(range(len(x)))
    ax.set_xticklabels(x)
    ax.set_title("CIC(Tue+Wed) → UNSW (F1@0.5 for models, Dummy labels)")
    ax.set_xlabel("")
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis='x', rotation=30)
    for c in ax.containers:
        ax.bar_label(c, fmt='%.3f', padding=2, fontsize=8)


def main():
    parser = argparse.ArgumentParser(description="Plot Dummy baseline vs. main models comparisons")
    parser.add_argument("--baseline-csv", type=Path, default=Path("results/table_3_3_baseline_metrics.csv"))
    parser.add_argument("--unsw-test-csv", type=Path, default=Path("out/unsw/tables/table_unsw_test.csv"))
    parser.add_argument("--cross-csv", type=Path, default=Path("out/xdom_harm/tables/table_cross_dataset.csv"))
    parser.add_argument("--out", type=Path, default=Path("out/fig_3_5_baseline_comparison.png"))
    args = parser.parse_args()

    baseline = load_baseline(args.baseline_csv)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    panel_unsw(axes[0], baseline, args.unsw_test_csv)
    panel_cross(axes[1], baseline, args.cross_csv)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(f"Saved figure: {args.out}")


if __name__ == "__main__":
    main()
