from __future__ import annotations
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


NUM_INFIXES = {"len", "bytes", "pkts", "avg", "std", "min", "max", "rate", "dur", "time", "ttl", "window"}


def infer_feature_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    cat_cols = [c for c in X.columns if X[c].dtype == "O" or X[c].dtype.name.startswith("category")]
    num_cols = [c for c in X.columns if c not in cat_cols]
    # heuristic fix: some numeric-looking strings
    for c in list(cat_cols):
        try:
            parsed = pd.to_numeric(X[c].dropna().head(100), errors="raise")
            # if convertible and column name hints numeric → treat as numeric
            if any(inf in c.lower() for inf in NUM_INFIXES):
                num_cols.append(c)
                cat_cols.remove(c)
        except Exception:
            pass
    return sorted(num_cols), sorted(cat_cols)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols, cat_cols = infer_feature_types(X)
    num_tf = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ]
    # Split categorical columns by cardinality to avoid exploding one-hot matrices
    low_card_cols = []
    high_card_cols = []
    for c in cat_cols:
        try:
            nuniq = int(X[c].nunique(dropna=True))
        except Exception:
            nuniq = 0
        if nuniq <= 10:
            low_card_cols.append(c)
        else:
            high_card_cols.append(c)

    transformers = [("num", Pipeline(num_tf), num_cols)]

    # Low-cardinality: safe to one-hot (small expansion)
    if low_card_cols:
        low_tf = [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop='if_binary')),
        ]
        transformers.append(("cat_low", Pipeline(low_tf), low_card_cols))

    # High-cardinality: use OrdinalEncoder to keep a single numeric column per categorical feature
    if high_card_cols:
        high_tf = [
            ("imputer", SimpleImputer(strategy="constant", fill_value="-1")),
            ("ord", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
        ]
        transformers.append(("cat_high", Pipeline(high_tf), high_card_cols))

    ct = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        # allow dense output (we use small one-hots + ordinals) and avoid huge sparse->dense conversions
        sparse_threshold=0.3,
        verbose_feature_names_out=False,
        n_jobs=None,
    )
    return ct

class PipelineCompat:
    """A minimal pipeline-like wrapper to avoid importing sklearn.pipeline if desired."""
    def __init__(self, steps):
        self.steps = steps
        self.named_steps = {}
        prev = None
        for name, step in steps:
            self.named_steps[name] = step
        self.steps_tuple = steps

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        params = {'steps': self.steps}
        if deep:
            for name, step in self.steps:
                if hasattr(step, 'get_params'):
                    for key, value in step.get_params(deep=True).items():
                        params[f'{name}__{key}'] = value
        return params

    def set_params(self, **params):
        """Set parameters for this estimator."""
        items = (params.items())
        names = []
        if not items:
            return self
        for name, param in items:
            key = name.split('__', 1)
            names.append(key[0])
            if len(key) > 1:
                sub_key = key[1]
                step = self.named_steps[key[0]]
                if hasattr(step, 'set_params'):
                    step.set_params(**{sub_key: param})
            else:
                self.steps = param
                self.steps_tuple = param
                self.named_steps = dict(param)
        return self

    def fit(self, X, y=None):
        Xt = X
        for name, step in self.steps_tuple:
            if hasattr(step, "fit_transform"):
                Xt = step.fit_transform(Xt, y)
            else:
                step.fit(Xt, y)
                Xt = step.transform(Xt)
        self._Xt_shape_ = getattr(Xt, "shape", None)
        return self

    def transform(self, X):
        Xt = X
        for name, step in self.steps_tuple:
            Xt = step.transform(Xt)
        return Xt

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)