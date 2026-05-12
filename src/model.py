"""
Demand model.

Why RandomForestRegressor:
- Robust to mixed-scale features (price vs. encoded ids) with zero preprocessing.
- Captures non-linear interactions (discount × seasonality × category) that the
  business cares about.
- Trains in seconds on the synthetic dataset — keeps the demo snappy.
- Exposes feature_importances_ which we use for the Data Quality page.

Tree models also degrade gracefully on the sparse Kaggle dataset (no price
features), where a linear model would need careful preprocessing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .feature_engineering import FEATURE_COLUMNS, TARGET, build_features, time_split
from .metrics import regression_metrics


@dataclass
class TrainedModel:
    estimator: RandomForestRegressor
    features: list[str]
    test_metrics: dict[str, float]
    feature_importance: pd.DataFrame
    test_predictions: pd.DataFrame  # for diagnostic plots


def train_demand_model(df: pd.DataFrame, holdout_weeks: int = 12,
                       n_estimators: int = 200, max_depth: int = 12) -> TrainedModel:
    """Train + evaluate the demand model with a time-based holdout."""
    feat_df, _ = build_features(df)
    train, test = time_split(feat_df, holdout_weeks=holdout_weeks)

    X_train, y_train = train[FEATURE_COLUMNS], train[TARGET]
    X_test, y_test = test[FEATURE_COLUMNS], test[TARGET]

    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=-1,
        random_state=42,
        min_samples_leaf=5,
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    metrics = regression_metrics(y_test.values, y_pred)

    fi = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)

    test_preds = test[["date", "store_id", "item_id", "category", "promo_flag"]].copy()
    test_preds["actual"] = y_test.values
    test_preds["predicted"] = y_pred

    return TrainedModel(
        estimator=rf,
        features=FEATURE_COLUMNS,
        test_metrics=metrics,
        feature_importance=fi,
        test_predictions=test_preds,
    )


def predict_units(model: TrainedModel, df: pd.DataFrame) -> np.ndarray:
    """Run the trained model on a feature frame and return predicted units."""
    feat_df, _ = build_features(df)
    return np.maximum(0, model.estimator.predict(feat_df[FEATURE_COLUMNS]))
