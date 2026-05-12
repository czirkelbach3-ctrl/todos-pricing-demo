"""
Feature engineering for the demand forecasting model.

Justification for each feature group:
- Calendar features (DOW, week-of-year, month) capture seasonality.
- Lag features (1w, 4w) capture short-term momentum.
- Rolling means (4w, 8w) capture medium-term level.
- Promo features (flag, discount depth) capture the controllable lever the
  merchant manipulates in the workbench.
- Encoded IDs let the tree-based model learn item/store/category-level effects
  without needing one-hot blow-up.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


FEATURE_COLUMNS = [
    "day_of_week", "week_of_year", "month",
    "item_id_enc", "store_id_enc", "category_enc",
    "promo_flag", "discount_pct",
    "regular_price", "promo_price",
    "lag_1_week_sales", "lag_4_week_sales",
    "rolling_4_week_avg", "rolling_8_week_avg",
    "seasonality_index",
]
TARGET = "units"


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """Compute training-ready features. Returns (X_df, encoders)."""
    df = df.sort_values(["store_id", "item_id", "week"]).copy()

    # Calendar
    df["day_of_week"] = pd.to_datetime(df["date"]).dt.dayofweek
    df["week_of_year"] = pd.to_datetime(df["date"]).dt.isocalendar().week.astype(int)
    df["month"] = pd.to_datetime(df["date"]).dt.month

    # Lags & rollings — grouped by (store, item) so we don't leak across series.
    # Using groupby.transform keeps the original index aligned correctly.
    grp_keys = ["store_id", "item_id"]
    grp = df.groupby(grp_keys)["units"]
    df["lag_1_week_sales"] = grp.shift(1).fillna(0)
    df["lag_4_week_sales"] = grp.shift(4).fillna(0)
    df["rolling_4_week_avg"] = (
        df.groupby(grp_keys)["units"]
          .transform(lambda s: s.shift(1).rolling(4, min_periods=1).mean())
          .fillna(0)
    )
    df["rolling_8_week_avg"] = (
        df.groupby(grp_keys)["units"]
          .transform(lambda s: s.shift(1).rolling(8, min_periods=1).mean())
          .fillna(0)
    )

    # Seasonality index = ratio of units to item-level mean
    item_mean = df.groupby("item_id")["units"].transform("mean").replace(0, np.nan)
    df["seasonality_index"] = (df["units"] / item_mean).fillna(1.0)

    # Encoders
    encoders = {}
    for col in ["item_id", "store_id", "category"]:
        enc = LabelEncoder()
        df[f"{col}_enc"] = enc.fit_transform(df[col].astype(str))
        encoders[col] = enc

    return df, encoders


def time_split(df: pd.DataFrame, holdout_weeks: int = 12) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Time-based train/test split (never random for time series)."""
    max_week = df["week"].max()
    cutoff = max_week - holdout_weeks
    train = df[df["week"] <= cutoff].copy()
    test = df[df["week"] > cutoff].copy()
    return train, test
