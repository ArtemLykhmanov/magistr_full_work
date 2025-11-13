from __future__ import annotations
from typing import Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
try:
    from xgboost import XGBClassifier  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    XGBClassifier = None  # type: ignore


def make_models(include_calibrated_rf: bool = True) -> Dict[str, Any]:
    models = {
        "logreg": LogisticRegression(max_iter=5000, n_jobs=None, solver="lbfgs"),
        "rf": RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1),
        "hgb": HistGradientBoostingClassifier(random_state=42),
        "mlp": MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", batch_size=512, max_iter=200, early_stopping=True, random_state=42),
    }
    # Optionally add XGBoost if available
    if XGBClassifier is not None:
        models["xgb"] = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
            eval_metric="logloss",
        )
    if include_calibrated_rf:
        base = models["rf"]
        models["rf_cal"] = (base, "calibrate")  # handled in trainer
    return models
