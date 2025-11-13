from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Callable, Dict, Tuple, List

import numpy as np
import pandas as pd

# Make project src importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Paths
from src.data_io import load_csv, unsw_files


def nearly_equal(a: float, b: float, tol: float) -> bool:
    if np.isnan(a) or np.isnan(b):
        return False
    return abs(a - b) <= tol


def check_unsw_single(model: str, tol: float = 1e-4) -> Tuple[bool, str]:
    """Re-run UNSW test for one model and compare with saved table_unsw_test.csv.

    We run via eval_cross_many.run_unsw_cv_and_test into a separate outdir to avoid overwriting.
    Compares pr_auc, roc_auc, f1@0.5.
    """
    # Load expected
    exp_path = Path("out/unsw/tables/table_unsw_test.csv")
    if not exp_path.exists():
        return False, f"Missing expected table: {exp_path}"
    exp_df = pd.read_csv(exp_path)
    if model not in exp_df["model"].astype(str).values:
        return False, f"Model '{model}' not found in {exp_path}"
    row = exp_df[exp_df["model"].astype(str) == model].iloc[0]
    exp_metrics = {
        "pr_auc": float(row["pr_auc"]),
        "roc_auc": float(row["roc_auc"]),
        "f1@0.5": float(row["f1@0.5"]),
    }

    # Re-run into a separate outdir and read actuals
    try:
        import experiments.eval_cross_many as xeval
        paths = Paths.from_root(".")
        outdir = Path("out/unsw_reprocheck")
        models_sel = [model]
        xeval.run_unsw_cv_and_test(paths, outdir, models_sel, target_fpr=0.01)
    except Exception as e:
        return False, f"Re-run failed: {e}"

    act_path = outdir / "tables" / "table_unsw_test.csv"
    if not act_path.exists():
        return False, f"Missing reproduced table: {act_path}"
    act_df = pd.read_csv(act_path)
    if model not in act_df["model"].astype(str).values:
        return False, f"Reproduced table missing model '{model}'"
    row2 = act_df[act_df["model"].astype(str) == model].iloc[0]
    act_metrics = {
        "pr_auc": float(row2["pr_auc"]),
        "roc_auc": float(row2["roc_auc"]),
        "f1@0.5": float(row2["f1@0.5"]),
    }

    all_ok = all(nearly_equal(exp_metrics[k], act_metrics[k], tol) for k in exp_metrics)
    note = (
        f"Model={model} | pr_auc exp={exp_metrics['pr_auc']:.6f} act={act_metrics['pr_auc']:.6f}; "
        f"roc_auc exp={exp_metrics['roc_auc']:.6f} act={act_metrics['roc_auc']:.6f}; "
        f"f1@0.5 exp={exp_metrics['f1@0.5']:.6f} act={act_metrics['f1@0.5']:.6f} (tol={tol})"
    )
    return all_ok, note


def check_dummy_baseline(tol: float = 1e-9) -> Tuple[bool, str]:
    """Verify baseline CSV presence and sanity pattern."""
    p = Path("results/table_3_3_baseline_metrics.csv")
    if not p.exists():
        return False, f"Missing baseline file: {p}"
    df = pd.read_csv(p)
    mdl = df[df.get("Strategy", "most_frequent").astype(str) == "most_frequent"]
    a = mdl[(mdl["TrainData"] == "UNSW") & (mdl["TestData"] == "UNSW")]
    b = mdl[(mdl["TrainData"] == "CICIDS") & (mdl["TestData"] == "CICIDS")]
    if a.empty or b.empty:
        return False, "Baseline rows missing (UNSW->UNSW or CICIDS->CICIDS)"
    f1_a = float(a.iloc[0]["Dummy_F1"])
    f1_b = float(b.iloc[0]["Dummy_F1"])
    ok = (f1_a >= 0.6) and (f1_b <= 0.1 + tol)
    return ok, f"UNSW->UNSW F1={f1_a:.3f}; CICIDS->CICIDS F1={f1_b:.3f}"


def check_feature_ablation(tol: float = 0.03) -> Tuple[bool, str]:
    p = Path("out/tables/table_3_feature_ablation.csv")
    if not p.exists():
        return False, f"Missing ablation table: {p}"
    df = pd.read_csv(p)
    base = float(df.loc[df["tag"] == "rf_all", "f1_0_5"].iloc[0])
    t10 = float(df[df["tag"].astype(str).str.startswith("rf_top10")]["f1_0_5"].iloc[0])
    ok = (base >= 0.85) and (abs(base - t10) <= tol)
    return ok, f"rf_all F1={base:.3f}, rf_top10 F1={t10:.3f}, tol={tol}"


def check_shap_stability_unsw(tol: float = 0.02) -> Tuple[bool, str]:
    p = Path("out/unsw_shap/tables/table_unsw_shap_stability_summary.csv")
    if not p.exists():
        return False, f"Missing SHAP stability summary: {p}"
    df = pd.read_csv(p)
    mean_row = df[df["stat"] == "mean"].iloc[0]
    spearman = float(mean_row["spearman"])
    ok = (spearman >= 0.95 - tol)
    return ok, f"Spearman mean={spearman:.3f}"


def check_mcnemar_rf_vs_xgb(tol: float = 0.05) -> Tuple[bool, str]:
    # Minimal check: ensure p-value < 0.05 (significance)
    from experiments.eval_mcnemar_rf_vs_xgb import load_models, to_labels
    from src.eval_utils import mcnemar_test

    rf_path = Path("out/models/rf.joblib")
    xgb_path = Path("out/models/xgb.joblib")
    if not (rf_path.exists() and xgb_path.exists()):
        return False, "Missing models; run experiments.train_unsw_models_for_mcnemar first"

    paths = Paths.from_root('.')
    files = unsw_files(paths.data)
    Xte, yte = load_csv(files["test"])  # raw features
    rf, xgb = load_models(rf_path, xgb_path)
    y_rf = to_labels(rf, Xte)
    y_xgb = to_labels(xgb, Xte)
    res = mcnemar_test(yte, y_rf, y_xgb)
    p = float(res.get("p_value", float("nan")))
    ok = (p == p) and (p < tol)
    return ok, f"p-value={p:.6f} (expect < {tol})"


def main():
    parser = argparse.ArgumentParser(description="Reproducibility checks for main experiments")
    parser.add_argument("--experiment", required=True,
                        choices=[
                            "unsw_single",
                            "baseline_dummy",
                            "feature_ablation",
                            "shap_stability_unsw",
                            "mcnemar_rf_vs_xgb",
                            "all_list",
                        ])
    parser.add_argument("--model", default="rf", help="Model for unsw_single (rf|logreg|mlp|hgb|xgb|rf_cal)")
    parser.add_argument("--tol", type=float, default=1e-4, help="Tolerance for numeric comparisons (unsw_single)")
    args = parser.parse_args()

    if args.experiment == "all_list":
        print("Available experiments to check:")
        print("- unsw_single (args: --model, --tol)")
        print("- baseline_dummy")
        print("- feature_ablation")
        print("- shap_stability_unsw")
        print("- mcnemar_rf_vs_xgb")
        return

    if args.experiment == "unsw_single":
        ok, note = check_unsw_single(args.model, tol=args.tol)
    elif args.experiment == "baseline_dummy":
        ok, note = check_dummy_baseline()
    elif args.experiment == "feature_ablation":
        ok, note = check_feature_ablation()
    elif args.experiment == "shap_stability_unsw":
        ok, note = check_shap_stability_unsw()
    elif args.experiment == "mcnemar_rf_vs_xgb":
        ok, note = check_mcnemar_rf_vs_xgb()
    else:
        raise SystemExit("Unknown experiment")

    print(("PASS" if ok else "FAIL") + f": {args.experiment} — {note}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

