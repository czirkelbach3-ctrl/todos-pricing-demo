"""
Data loading dispatch.

Why this is a separate module: Streamlit pages should never `import synthetic_data`
directly — they call `load_dataset()` which picks the source (Kaggle vs synthetic),
caches the result, and returns a uniform schema. This keeps every page agnostic
to where the data came from.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from . import synthetic_data as syn


DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Kaggle artifacts we know how to consume (graceful fallback if missing).
KAGGLE_DEMAND_TRAIN = DATA_DIR / "train.csv"          # store-item-demand-forecasting
KAGGLE_FAVORITA_TRAIN = DATA_DIR / "favorita_train.csv"
KAGGLE_FAVORITA_STORES = DATA_DIR / "stores.csv"
KAGGLE_FAVORITA_OIL = DATA_DIR / "oil.csv"
KAGGLE_FAVORITA_HOLIDAYS = DATA_DIR / "holidays_events.csv"


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def load_dataset(mode: str = "auto") -> Tuple[pd.DataFrame, str]:
    """Load the working dataset.

    Args:
        mode: "auto" (try Kaggle, else synthetic), "synthetic", or "kaggle".

    Returns:
        (df, source_label). source_label is one of:
        "Synthetic Demo", "Kaggle: Store Item Demand", "Kaggle: Favorita".
    """
    if mode == "synthetic":
        return syn.generate_dataset(), "Synthetic Demo"

    if mode == "kaggle":
        df = _try_kaggle()
        if df is None:
            raise FileNotFoundError(
                "Kaggle files not found in /data. See data/README.md for download steps."
            )
        return df

    # auto
    df = _try_kaggle()
    if df is not None:
        return df
    return syn.generate_dataset(), "Synthetic Demo"


def kaggle_available() -> bool:
    return KAGGLE_DEMAND_TRAIN.exists() or KAGGLE_FAVORITA_TRAIN.exists()


# --------------------------------------------------------------------------- #
# Kaggle ingestion
# --------------------------------------------------------------------------- #

def _try_kaggle() -> Optional[Tuple[pd.DataFrame, str]]:
    if KAGGLE_FAVORITA_TRAIN.exists():
        return _load_favorita(), "Kaggle: Favorita"
    if KAGGLE_DEMAND_TRAIN.exists():
        return _load_store_item_demand(), "Kaggle: Store Item Demand"
    return None


def _load_store_item_demand() -> pd.DataFrame:
    """Kaggle: 'Store Item Demand Forecasting Challenge'.

    Provides only (date, store, item, sales) — we enrich it with synthetic
    metadata (prices, categories, promo) so every UI screen still works.
    """
    raw = pd.read_csv(KAGGLE_DEMAND_TRAIN, parse_dates=["date"])
    raw = raw.rename(columns={"sales": "units", "store": "store_id", "item": "item_id"})
    return _enrich_with_synth_metadata(raw)


def _load_favorita() -> pd.DataFrame:
    """Kaggle: 'Store Sales — Favorita Time Series Forecasting'.

    Same idea: take the real demand signal and graft synthetic merchandising
    attributes on top so the merchant/pricing UI still has every field it needs.
    """
    raw = pd.read_csv(KAGGLE_FAVORITA_TRAIN, parse_dates=["date"])
    # Favorita uses 'sales' as revenue, not units. Approximate units = sales / 5
    # (the platform uses a placeholder price scale; demo only).
    raw = raw.rename(columns={"store_nbr": "store_id"})
    raw["units"] = np.maximum(1, np.round(raw["sales"].fillna(0) / 5)).astype(int)
    return _enrich_with_synth_metadata(raw)


def _enrich_with_synth_metadata(real: pd.DataFrame) -> pd.DataFrame:
    """Combine real demand with synthetic merchandising attributes.

    Strategy: we generate the synthetic panel anyway (it's cheap) and join the
    item/store metadata onto the real demand panel. Where the real data is
    sparser than ours, we pad with synthetic; where it's denser, we keep real.
    """
    syn_df = syn.generate_dataset()
    items = (syn_df[["item_id", "item_description", "category", "department", "vendor",
                     "regular_price", "cost", "vendor_funding_pct", "elasticity"]]
             .drop_duplicates("item_id"))
    stores = syn_df[["store_id", "store_name", "region"]].drop_duplicates("store_id")

    # Map real ids onto the synthetic universe by modulo so demo never breaks.
    real["item_id"] = real["item_id"].astype(int)
    real["store_id"] = real["store_id"].astype(int)
    real["item_id"] = items["item_id"].iloc[real["item_id"] % len(items)].reset_index(drop=True).values
    real["store_id"] = stores["store_id"].iloc[real["store_id"] % len(stores)].reset_index(drop=True).values

    real = real.merge(items, on="item_id", how="left").merge(stores, on="store_id", how="left")
    real["promo_flag"] = 0
    real["promo_tactic"] = "No Promo"
    real["discount_pct"] = 0.0
    real["promo_price"] = real["regular_price"]
    real["sales"] = real["units"] * real["promo_price"]
    real["gross_margin"] = real["units"] * (real["promo_price"] - real["cost"])
    real["inventory_on_hand"] = (real["units"] * 4).astype(int)
    real["sell_through_pct"] = 0.6
    real["ly_units"] = 0
    real["ly_sales"] = 0.0
    real["baseline_units"] = real["units"]
    real["forecast_units"] = real["units"]
    real["forecast_sales"] = real["sales"]
    real["forecast_margin"] = real["gross_margin"]
    real["confidence_score"] = 0.75
    real["year_offset"] = 0
    real["fiscal_week"] = real["date"].dt.isocalendar().week.astype(int)
    real["week"] = ((real["date"] - real["date"].min()).dt.days // 7) + 1

    return real
