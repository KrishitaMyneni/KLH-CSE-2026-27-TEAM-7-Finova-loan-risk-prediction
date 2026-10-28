"""
Per-applicant explanations for whichever model is champion.

Returns additive contributions in the model's raw (log-odds) space:

  XGBoost / LightGBM / CatBoost  native TreeSHAP (exact, fast)
  Random Forest                  shap.TreeExplainer
  Logistic Regression            coefficient x (value - training mean), exact

Contributions are computed on the uncalibrated model. Isotonic calibration is
a monotone map applied afterwards, so the direction and ranking of the drivers
are unchanged; only the final probability is rescaled.

One-hot columns are summed back into their source feature, so a reason code
reads "Loan purpose" rather than "purpose_small_business".
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.models import unwrap_pipeline


class Explainer:
    def __init__(self, calibrated_model, numeric: list, categorical: list,
                 background: pd.DataFrame | None = None):
        pipe = unwrap_pipeline(calibrated_model)
        self.prep = pipe.named_steps["prep"]
        self.model = pipe.named_steps["model"]
        self.kind = type(self.model).__name__
        self.out_names = list(self.prep.get_feature_names_out())
        self.parents = _parent_map(self.out_names, numeric, categorical)
        self.features = list(dict.fromkeys(self.parents))  # unique, ordered

        self._background_mean = None
        if background is not None and len(background):
            self._background_mean = np.asarray(self.prep.transform(background)).mean(axis=0)

        self._tree_explainer = None
        if self.kind == "RandomForestClassifier":
            import shap

            self._tree_explainer = shap.TreeExplainer(self.model)

    @property
    def supported(self) -> bool:
        return self.kind in {
            "XGBClassifier", "LGBMClassifier", "CatBoostClassifier",
            "RandomForestClassifier", "LogisticRegression",
        }

    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Rows = applicants, columns = source features, values = log-odds push."""
        Xp = np.asarray(self.prep.transform(X), dtype="float64")
        raw = self._raw_contribs(Xp)
        df = pd.DataFrame(raw, columns=self.out_names, index=X.index)
        return df.T.groupby(self.parents, sort=False).sum().T[self.features]

    def _raw_contribs(self, Xp: np.ndarray) -> np.ndarray:
        if self.kind == "XGBClassifier":
            import xgboost as xgb

            booster = self.model.get_booster()
            return booster.predict(xgb.DMatrix(Xp), pred_contribs=True)[:, :-1]
        if self.kind == "LGBMClassifier":
            return self.model.predict(Xp, pred_contrib=True)[:, :-1]
        if self.kind == "CatBoostClassifier":
            from catboost import Pool

            return self.model.get_feature_importance(Pool(Xp), type="ShapValues")[:, :-1]
        if self.kind == "RandomForestClassifier":
            vals = self._tree_explainer.shap_values(Xp, check_additivity=False)
            vals = np.asarray(vals)
            if vals.ndim == 3:  # (n, features, classes) or (classes, n, features)
                vals = vals[..., 1] if vals.shape[-1] == 2 else vals[1]
            return vals
        if self.kind == "LogisticRegression":
            centre = self._background_mean if self._background_mean is not None else 0.0
            return (Xp - centre) * self.model.coef_.ravel()
        return np.zeros_like(Xp)


def _parent_map(out_names: list, numeric: list, categorical: list) -> list:
    """Map every transformed column back to the raw feature it came from."""
    num = set(numeric)
    cats = sorted(categorical, key=len, reverse=True)  # longest prefix wins
    parents = []
    for name in out_names:
        if name in num:
            parents.append(name)
            continue
        parents.append(next((c for c in cats if name.startswith(c + "_")), name))
    return parents


def global_importance(explainer: Explainer, X: pd.DataFrame) -> pd.DataFrame:
    """Mean |contribution| per source feature: comparable across model types."""
    contrib = explainer.contributions(X)
    imp = contrib.abs().mean().sort_values(ascending=False)
    return imp.rename("importance").rename_axis("feature").reset_index()
