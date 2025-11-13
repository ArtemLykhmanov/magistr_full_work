import argparse
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# Use project utilities for consistent loading/labels if available
try:
    from src.config import Paths
    from src.data_io import load_csv as project_load_csv, unsw_files, cicids_day_files
except Exception:
    Paths = None
    project_load_csv = None
    unsw_files = cicids_day_files = None


@dataclass
class Scenario:
    train_name: str
    X_train: pd.DataFrame
    y_train: pd.Series
    test_name: str
    X_test: pd.DataFrame
    y_test: pd.Series


def load_dataset_from_csv(csv_path: str, label_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    # Prefer project loader for robust label detection/cleaning
    if project_load_csv is not None:
        return project_load_csv(Path(csv_path))
    # Fallback: simple loader with explicit label column
    df = pd.read_csv(csv_path)
    if label_col not in df.columns:
        raise ValueError(
            f"Label column '{label_col}' not found in {csv_path}. Columns: {list(df.columns)}"
        )
    X = df.drop(columns=[label_col])
    y = df[label_col]
    return X, y


def maybe_internal_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    random_state: int,
    shuffle: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if test_size and test_size > 0:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle,
            stratify=y if shuffle else None,
        )
        return X_tr, X_te, y_tr, y_te
    # If no split requested, return as-is
    return X, X, y, y


def build_scenarios(args: argparse.Namespace) -> List[Scenario]:
    scenarios: List[Scenario] = []

    # If no explicit CSVs provided, try project defaults
    if (
        args.unsw_train_csv is None
        and args.unsw_test_csv is None
        and args.unsw_csv is None
        and args.cicids_train_csv is None
        and args.cicids_test_csv is None
        and args.cicids_csv is None
        and Paths is not None
        and unsw_files is not None
        and cicids_day_files is not None
    ):
        paths = Paths.from_root('.')
        unsw = unsw_files(paths.data)
        cicids = cicids_day_files(paths.data)
        # Prefer UNSW official train/test; for CICIDS pick Tuesday as attack-heavy test day by default
        if (unsw['train'].exists() and unsw['test'].exists()):
            args.unsw_train_csv = str(unsw['train'])
            args.unsw_test_csv = str(unsw['test'])
        # Choose two CICIDS days if available: use Tuesday as train and Wednesday as test by default
        cicids_candidates = ['tuesday', 'wednesday']
        if all(k in cicids for k in cicids_candidates):
            if cicids['tuesday'].exists() and cicids['wednesday'].exists():
                args.cicids_train_csv = str(cicids['tuesday'])
                args.cicids_test_csv = str(cicids['wednesday'])

    # UNSW
    X_unsw_train = y_unsw_train = X_unsw_test = y_unsw_test = None
    if args.unsw_train_csv and args.unsw_test_csv:
        X_unsw_train, y_unsw_train = load_dataset_from_csv(args.unsw_train_csv, args.label_col)
        X_unsw_test, y_unsw_test = load_dataset_from_csv(args.unsw_test_csv, args.label_col)
    elif args.unsw_csv:
        X_unsw, y_unsw = load_dataset_from_csv(args.unsw_csv, args.label_col)
        X_unsw_train, X_unsw_test, y_unsw_train, y_unsw_test = maybe_internal_split(
            X_unsw, y_unsw, args.internal_split, args.random_state, args.shuffle
        )

    # CICIDS
    X_cicids_train = y_cicids_train = X_cicids_test = y_cicids_test = None
    if args.cicids_train_csv and args.cicids_test_csv:
        X_cicids_train, y_cicids_train = load_dataset_from_csv(args.cicids_train_csv, args.label_col)
        X_cicids_test, y_cicids_test = load_dataset_from_csv(args.cicids_test_csv, args.label_col)
    elif args.cicids_csv:
        X_cicids, y_cicids = load_dataset_from_csv(args.cicids_csv, args.label_col)
        X_cicids_train, X_cicids_test, y_cicids_train, y_cicids_test = maybe_internal_split(
            X_cicids, y_cicids, args.internal_split, args.random_state, args.shuffle
        )

    # Within-dataset scenarios if train/test available
    if X_unsw_train is not None and X_unsw_test is not None:
        scenarios.append(
            Scenario(
                train_name="UNSW",
                X_train=X_unsw_train,
                y_train=y_unsw_train,
                test_name="UNSW",
                X_test=X_unsw_test,
                y_test=y_unsw_test,
            )
        )
    if X_cicids_train is not None and X_cicids_test is not None:
        scenarios.append(
            Scenario(
                train_name="CICIDS",
                X_train=X_cicids_train,
                y_train=y_cicids_train,
                test_name="CICIDS",
                X_test=X_cicids_test,
                y_test=y_cicids_test,
            )
        )

    # Cross-dataset scenarios if both datasets present
    if (X_unsw_train is not None and X_cicids_test is not None):
        scenarios.append(
            Scenario(
                train_name="UNSW",
                X_train=X_unsw_train,
                y_train=y_unsw_train,
                test_name="CICIDS",
                X_test=X_cicids_test,
                y_test=y_cicids_test,
            )
        )
    if (X_cicids_train is not None and X_unsw_test is not None):
        scenarios.append(
            Scenario(
                train_name="CICIDS",
                X_train=X_cicids_train,
                y_train=y_cicids_train,
                test_name="UNSW",
                X_test=X_unsw_test,
                y_test=y_unsw_test,
            )
        )

    return scenarios


def coerce_pos_label(y: pd.Series, pos_label: Optional[str]):
    if pos_label is None:
        return None
    # Try to coerce type to match y dtype
    if y.dtype.kind in {"i", "u"}:  # int/uint
        try:
            return int(pos_label)
        except Exception:
            return pos_label
    if y.dtype.kind == "f":  # float
        try:
            return float(pos_label)
        except Exception:
            return pos_label
    return pos_label  # keep as string or category


def compute_metrics(y_true, y_pred, average: str, pos_label: Optional[str]):
    kwargs = {"zero_division": 0}
    if average == "binary":
        if pos_label is None:
            raise ValueError("pos_label must be provided when average='binary'")
        prec = precision_score(y_true, y_pred, pos_label=pos_label, average="binary", **kwargs)
        rec = recall_score(y_true, y_pred, pos_label=pos_label, average="binary", **kwargs)
        f1 = f1_score(y_true, y_pred, pos_label=pos_label, average="binary", **kwargs)
        return prec, rec, f1
    else:
        prec = precision_score(y_true, y_pred, average=average, **kwargs)
        rec = recall_score(y_true, y_pred, average=average, **kwargs)
        f1 = f1_score(y_true, y_pred, average=average, **kwargs)
        return prec, rec, f1


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate DummyClassifier baseline (most_frequent) across within- and cross-dataset scenarios."
        )
    )

    # Dataset inputs
    parser.add_argument("--unsw-train-csv", dest="unsw_train_csv", type=str, default=None)
    parser.add_argument("--unsw-test-csv", dest="unsw_test_csv", type=str, default=None)
    parser.add_argument("--unsw-csv", dest="unsw_csv", type=str, default=None, help="Single UNSW CSV to split")

    parser.add_argument("--cicids-train-csv", dest="cicids_train_csv", type=str, default=None)
    parser.add_argument("--cicids-test-csv", dest="cicids_test_csv", type=str, default=None)
    parser.add_argument("--cicids-csv", dest="cicids_csv", type=str, default=None, help="Single CICIDS CSV to split")

    parser.add_argument("--label-col", type=str, default="label")

    # Splitting behavior for single-CSV inputs
    parser.add_argument("--internal-split", type=float, default=0.2, help="Test size for internal split (0..1)")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle before internal split (stratified if possible)")
    parser.add_argument("--random-state", type=int, default=42)

    # Metrics configuration
    parser.add_argument(
        "--average",
        type=str,
        default="binary",
        choices=["binary", "weighted", "macro", "micro"],
        help="Averaging strategy for metrics",
    )
    parser.add_argument(
        "--pos-label",
        dest="pos_label",
        type=str,
        default="1",
        help="Positive label for binary metrics (when --average=binary)",
    )

    # Dummy strategy configuration
    parser.add_argument(
        "--dummy-strategy",
        dest="dummy_strategy",
        nargs="+",
        default=["most_frequent"],
        choices=["most_frequent", "stratified", "uniform"],
        help="DummyClassifier strategy (one or more)",
    )

    # Output path
    parser.add_argument(
        "--output-csv",
        type=str,
        default=os.path.join("results", "table_3_3_baseline_metrics.csv"),
        help="Path to write the metrics table",
    )

    args = parser.parse_args()

    scenarios = build_scenarios(args)
    if not scenarios:
        raise SystemExit(
            "No scenarios constructed. Provide CSVs via --unsw-*/--cicids-* or single CSVs with --*-csv and --internal-split."
        )

    rows = []
    for strategy in args.dummy_strategy:
        dummy = DummyClassifier(strategy=strategy, random_state=args.random_state)
        for sc in scenarios:
            print(f"Training DummyClassifier[{strategy}] on {sc.train_name}, testing on {sc.test_name}...")

            # DummyClassifier ignores features; use shape-agnostic constant features to avoid mismatch issues
            Xtr = np.zeros((len(sc.y_train), 1), dtype=float)
            ytr = np.asarray(sc.y_train)
            dummy.fit(Xtr, ytr)

            Xte = np.zeros((len(sc.y_test), 1), dtype=float)
            y_pred = dummy.predict(Xte)

            pos_label = coerce_pos_label(sc.y_test, args.pos_label) if args.average == "binary" else None
            prec, rec, f1 = compute_metrics(sc.y_test, y_pred, average=args.average, pos_label=pos_label)

            rows.append(
                {
                    "Strategy": strategy,
                    "TrainData": sc.train_name,
                    "TestData": sc.test_name,
                    "Average": args.average,
                    "PosLabel": pos_label if args.average == "binary" else "-",
                    "Dummy_Precision": prec,
                    "Dummy_Recall": rec,
                    "Dummy_F1": f1,
                }
            )

    df = pd.DataFrame(rows)
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print("\nBaseline Dummy Performance by Scenario:")
    print(df.to_string(index=False))
    print(f"\nSaved CSV: {out_path}")


if __name__ == "__main__":
    main()
