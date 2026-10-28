"""
Show what post-origination leakage does: the same XGBoost, same split, trained
once on application-time features only and once with three columns that are
only known after the loan has been repaying.

    python scripts/leakage_demo.py            # writes artifacts/leakage_demo.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from src import cleaning, features, models
from src.data_loader import load_raw

LEAKS = ["total_pymnt", "recoveries", "last_pymnt_amnt"]


def fit_auc(train, test, num, cat):
    pos_weight = float((train[config.TARGET] == 0).sum() / train[config.TARGET].sum())
    pipe = Pipeline([("prep", features.build_preprocessor(num, cat)),
                     ("model", models._xgb(pos_weight))])
    pipe.fit(train[num + cat], train[config.TARGET])
    prob = pipe.predict_proba(test[num + cat])[:, 1]
    return round(float(roc_auc_score(test[config.TARGET], prob)), 4), \
        round(models.ks_statistic(test[config.TARGET], prob), 4)


def main():
    raw = load_raw(usecols=config.USECOLS + LEAKS)
    leak_cols = raw[["id"] + LEAKS].drop_duplicates("id")
    clean_df, _ = cleaning.clean(raw.drop(columns=LEAKS), verbose=False)
    clean_df = clean_df.merge(leak_cols, on="id", how="left")
    feat = features.build_features(clean_df, verbose=False)
    train, _, test = features.time_split(feat, return_cal=True)
    num, cat = features.select_feature_columns(feat)

    honest = fit_auc(train, test, num, cat)
    leaked = fit_auc(train, test, num + LEAKS, cat)
    result = {
        "model": "XGBoost, same hyperparameters and split",
        "test_window": f"{test['issue_d'].min():%Y-%m} to {test['issue_d'].max():%Y-%m}",
        "n_test": int(len(test)),
        "application_features_only": {"roc_auc": honest[0], "ks": honest[1]},
        "plus_leaked_columns": {"columns": LEAKS, "roc_auc": leaked[0], "ks": leaked[1]},
    }
    print(json.dumps(result, indent=2))
    out = config.ARTIFACT_DIR / "leakage_demo.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    pd.set_option("display.width", 120)
    main()
