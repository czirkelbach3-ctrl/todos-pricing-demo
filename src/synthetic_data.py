"""
Synthetic retail dataset generator.

Design choices (and why):
- We simulate ~104 fiscal weeks (LY + TY) so the app can show LY hindsight AND a
  TY pre-season forecast view without a separate dataset.
- Demand is a *multiplicative* model: baseline * seasonality * promo_lift * noise.
  This mirrors how merchants intuitively think (lifts compound) and keeps the
  elasticity math interpretable.
- Promo cadence is event-driven (Memorial Day, July 4th, Back-to-School, Black
  Friday, Holiday, Super Bowl, Spring Reset) so LY effectiveness analysis has
  recognizable promotional windows.
- Vendor funding is a *vendor* attribute, not item — matches how vendor co-op
  deals are negotiated in real merchandising orgs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List


# --------------------------------------------------------------------------- #
# Reference taxonomies
# --------------------------------------------------------------------------- #

DEPARTMENTS: Dict[str, List[str]] = {
    "Consumables": ["Beverages", "Snacks", "Household"],
    "Hardlines":   ["Seasonal", "Electronics"],
    "Softlines":   ["Apparel"],
}

# Realistic item name fragments per category
ITEM_TEMPLATES: Dict[str, List[str]] = {
    "Beverages":   ["Cola 12pk", "Sparkling Water 24pk", "Energy Drink 4pk", "Bottled Water 35pk",
                    "Sports Drink Variety", "Cold Brew Coffee 4pk", "Premium Juice 6pk",
                    "Hard Seltzer 12pk", "Iced Tea Gallon", "Plant Milk Carton"],
    "Snacks":      ["Tortilla Chips Family", "Trail Mix Club", "Protein Bar 24ct", "Pretzel Variety",
                    "Mixed Nuts 32oz", "Popcorn Multi", "Cookie Mega Pack", "Cheese Crackers 40ct",
                    "Jerky Variety 18oz", "Granola Pouches 30ct"],
    "Household":   ["Bath Tissue 30 Roll", "Paper Towels 12 Roll", "Laundry Pods 96ct",
                    "Dish Detergent 2pk", "Trash Bags 200ct", "Disinfecting Wipes 5pk",
                    "Dishwasher Pods 96ct", "Foil Roll 200ft", "Storage Bags 580ct",
                    "Hand Soap 4pk"],
    "Seasonal":    ["Patio Heater", "Inflatable Pool", "Holiday Tree 7ft", "Outdoor String Lights",
                    "Beach Tent", "Backyard Grill", "Costume Variety", "Halloween Candy Tub",
                    "Snow Shovel", "Pumpkin Spice Decor"],
    "Electronics": ["55in 4K TV", "Wireless Earbuds", "Smart Doorbell", "Tablet 64GB",
                    "Streaming Stick", "Bluetooth Speaker", "Robot Vacuum", "Action Camera",
                    "Smartwatch Series 9", "Mesh Wifi 3pk"],
    "Apparel":     ["Mens Tee 3pk", "Womens Leggings", "Kids Joggers", "Athletic Socks 12pk",
                    "Fleece Pullover", "Rain Jacket", "Denim 5pkt", "Down Vest",
                    "Performance Polo", "Sleepwear Set"],
}

VENDORS = ["Vendor A", "Vendor B", "Vendor C", "Vendor D"]

PROMO_TACTICS = ["No Promo", "Percent Off", "Dollar Off", "BOGO", "Clipless", "Digital"]

# Promo windows (fiscal week of year) — drives LY promo events
PROMO_WINDOWS: Dict[str, List[int]] = {
    "Super Bowl":      [5, 6],
    "Spring Reset":    [14, 15],
    "Memorial Day":    [21, 22],
    "July 4th":        [26, 27],
    "Back to School":  [32, 33, 34],
    "Halloween":       [42, 43],
    "Black Friday":    [47, 48],
    "Holiday":         [50, 51, 52],
}

# Category elasticity priors (negative = price up reduces units).
# These are merchandising-team heuristics that the data scientist can later
# replace with an estimated elasticity per item-category.
CATEGORY_ELASTICITY = {
    "Beverages":   -1.8,
    "Snacks":      -1.5,
    "Household":   -1.1,
    "Seasonal":    -2.3,
    "Electronics": -2.0,
    "Apparel":     -1.7,
}


@dataclass
class GenConfig:
    n_stores: int = 8
    items_per_category: int = 5     # 5 categories worth of items active in each dept
    n_weeks: int = 104              # LY + TY
    start_date: str = "2024-01-07"  # Sunday
    seed: int = 42


# --------------------------------------------------------------------------- #
# Item & store catalogs
# --------------------------------------------------------------------------- #

def _build_item_catalog(cfg: GenConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    rows = []
    item_counter = 100000
    for dept, cats in DEPARTMENTS.items():
        for cat in cats:
            templates = ITEM_TEMPLATES[cat]
            for i in range(cfg.items_per_category):
                # Reasonable price range per category — keeps the demo believable
                base_price = {
                    "Beverages":   rng.uniform(6, 22),
                    "Snacks":      rng.uniform(8, 28),
                    "Household":   rng.uniform(10, 40),
                    "Seasonal":    rng.uniform(20, 250),
                    "Electronics": rng.uniform(35, 600),
                    "Apparel":     rng.uniform(12, 70),
                }[cat]
                cost = base_price * rng.uniform(0.55, 0.78)  # 22%-45% gross margin
                rows.append({
                    "item_id": item_counter,
                    "item_description": f"{templates[i % len(templates)]} ({cat[:3].upper()}-{item_counter})",
                    "category": cat,
                    "department": dept,
                    "vendor": rng.choice(VENDORS),
                    "regular_price": round(base_price, 2),
                    "cost": round(cost, 2),
                    "vendor_funding_pct": round(rng.choice([0.0, 0.02, 0.04, 0.06, 0.08]), 3),
                    "elasticity": float(CATEGORY_ELASTICITY[cat] + rng.normal(0, 0.15)),
                    # Larger items move fewer units; tunes the demand baseline
                    "baseline_velocity": rng.gamma(shape=2.0, scale={
                        "Beverages": 90, "Snacks": 70, "Household": 35,
                        "Seasonal": 6,   "Electronics": 4, "Apparel": 18,
                    }[cat]),
                })
                item_counter += 1
    return pd.DataFrame(rows)


def _build_store_catalog(cfg: GenConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed + 1)
    regions = ["Northeast", "Southeast", "Midwest", "South Central", "Mountain", "West"]
    rows = []
    for s in range(cfg.n_stores):
        rows.append({
            "store_id": 100 + s,
            "store_name": f"Club #{100 + s}",
            "region": rng.choice(regions),
            "size_index": float(rng.uniform(0.85, 1.20)),  # multiplier on demand
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Main panel builder
# --------------------------------------------------------------------------- #

def _seasonality(fw: int, category: str) -> float:
    """Annual seasonality curve per category, indexed at 1.0."""
    # week-of-year sinusoid + category-specific bumps
    base = 1.0 + 0.18 * np.sin(2 * np.pi * (fw - 1) / 52.0 - np.pi / 3)
    bumps = {
        "Beverages":   0.10 if fw in range(22, 36) else 0.0,
        "Snacks":      0.08 if fw in (5, 6, 47, 48, 50, 51, 52) else 0.0,
        "Household":   0.05 if fw in (14, 15) else 0.0,
        "Seasonal":    0.45 if fw in (47, 48, 50, 51, 52) else (0.30 if fw in range(20, 28) else -0.10),
        "Electronics": 0.50 if fw in (47, 48) else (0.20 if fw in range(32, 36) else 0.0),
        "Apparel":     0.20 if fw in range(32, 36) else (0.15 if fw in (47, 48) else 0.0),
    }
    return float(base + bumps.get(category, 0.0))


def _promo_assignment(fw: int, item_id: int, rng) -> tuple[float, str, float]:
    """Decide whether an item is on promo this fiscal week.

    Returns (discount_pct, promo_tactic, promo_flag).
    """
    # Items are eligible for ~2-3 promo windows per year, randomly selected.
    elig_windows = []
    rng_local = np.random.default_rng(int(item_id) % (2**32))
    chosen = rng_local.choice(list(PROMO_WINDOWS.keys()), size=3, replace=False)
    for w in chosen:
        elig_windows.extend(PROMO_WINDOWS[w])
    if fw in elig_windows:
        tactic = rng.choice(PROMO_TACTICS[1:], p=[0.35, 0.20, 0.10, 0.20, 0.15])
        depth = {
            "Percent Off": rng.choice([0.10, 0.15, 0.20, 0.25, 0.30]),
            "Dollar Off":  rng.choice([0.08, 0.12, 0.18]),
            "BOGO":        0.40,         # effective discount
            "Clipless":    rng.choice([0.10, 0.15]),
            "Digital":     rng.choice([0.08, 0.12, 0.18]),
        }[tactic]
        return float(depth), str(tactic), 1.0
    return 0.0, "No Promo", 0.0


def generate_dataset(cfg: GenConfig | None = None) -> pd.DataFrame:
    """Generate the full item × store × week synthetic panel.

    Returns a long-format DataFrame with every column required by the spec.
    """
    cfg = cfg or GenConfig()
    rng = np.random.default_rng(cfg.seed)

    items = _build_item_catalog(cfg)
    stores = _build_store_catalog(cfg)

    weeks = pd.date_range(start=cfg.start_date, periods=cfg.n_weeks, freq="7D")
    week_df = pd.DataFrame({
        "date": weeks,
        "week": np.arange(1, cfg.n_weeks + 1),
        "fiscal_week": ((np.arange(cfg.n_weeks)) % 52) + 1,
        "year_offset": np.arange(cfg.n_weeks) // 52,   # 0 = LY, 1 = TY
    })

    # Cross product. Smaller stores × items × weeks footprint to stay snappy.
    panel = (items.assign(_k=1)
                  .merge(stores.assign(_k=1), on="_k")
                  .merge(week_df.assign(_k=1), on="_k")
                  .drop(columns="_k"))

    # Seasonality
    panel["seasonality"] = [
        _seasonality(int(fw), cat) for fw, cat in zip(panel["fiscal_week"], panel["category"])
    ]

    # Promo assignment per row (vectorised approximation: do per item-week, broadcast across stores)
    iw = panel[["item_id", "fiscal_week"]].drop_duplicates()
    promo_rows = []
    for r in iw.itertuples(index=False):
        d, t, f = _promo_assignment(int(r.fiscal_week), int(r.item_id), rng)
        promo_rows.append((int(r.item_id), int(r.fiscal_week), d, t, f))
    promo_df = pd.DataFrame(promo_rows, columns=["item_id", "fiscal_week",
                                                  "discount_pct", "promo_tactic", "promo_flag"])
    panel = panel.merge(promo_df, on=["item_id", "fiscal_week"], how="left")

    # Pricing
    panel["promo_price"] = np.round(panel["regular_price"] * (1 - panel["discount_pct"]), 2)

    # Demand: baseline * seasonality * elasticity-driven promo lift * store size * noise
    # promo_lift = (promo_price / regular_price) ** elasticity, capped for realism
    price_ratio = (panel["promo_price"] / panel["regular_price"]).clip(lower=0.4)
    promo_lift = np.power(price_ratio, panel["elasticity"]).clip(upper=4.5)
    promo_lift = np.where(panel["promo_flag"] == 1, promo_lift, 1.0)

    noise = rng.lognormal(mean=0.0, sigma=0.18, size=len(panel))
    demand = (panel["baseline_velocity"]
              * panel["seasonality"]
              * promo_lift
              * panel["size_index"]
              * noise)
    panel["units"] = np.maximum(0, np.round(demand)).astype(int)
    panel["sales"] = np.round(panel["units"] * panel["promo_price"], 2)
    panel["gross_margin"] = np.round(panel["units"] * (panel["promo_price"] - panel["cost"]), 2)

    # Inventory & sell-through (simplified: planned receipt = expected baseline * 6)
    expected_baseline = panel["baseline_velocity"] * panel["seasonality"] * panel["size_index"]
    planned_inventory = (expected_baseline * 6).clip(lower=4).round()
    # Real-world inventory varies; add noise so risk distribution is non-trivial
    inv_noise = rng.uniform(0.65, 1.35, size=len(panel))
    panel["inventory_on_hand"] = np.maximum(0, np.round(planned_inventory * inv_noise)).astype(int)
    panel["sell_through_pct"] = np.where(
        panel["inventory_on_hand"] + panel["units"] > 0,
        panel["units"] / (panel["inventory_on_hand"] + panel["units"]),
        0.0,
    ).round(3)

    # LY columns: shift by 52 weeks within (store_id, item_id)
    panel = panel.sort_values(["store_id", "item_id", "week"]).reset_index(drop=True)
    panel["ly_units"] = panel.groupby(["store_id", "item_id"])["units"].shift(52).fillna(0).astype(int)
    panel["ly_sales"] = panel.groupby(["store_id", "item_id"])["sales"].shift(52).fillna(0.0)

    # Baseline forecast = rolling mean of non-promo weeks (proxy for "what would happen w/o promo")
    non_promo_units = panel["units"].where(panel["promo_flag"] == 0)
    panel["baseline_units"] = (
        non_promo_units.groupby([panel["store_id"], panel["item_id"]])
        .transform(lambda s: s.rolling(8, min_periods=2).mean())
        .fillna(panel["units"].mean())
        .round()
        .astype(int)
    )

    # Forecast columns will be populated by the real model later; for now seed
    # them with promo-adjusted baseline so screens always render even pre-train.
    panel["forecast_units"] = (panel["baseline_units"] * np.where(panel["promo_flag"] == 1, 1.2, 1.0)).round().astype(int)
    panel["forecast_sales"] = np.round(panel["forecast_units"] * panel["promo_price"], 2)
    panel["forecast_margin"] = np.round(panel["forecast_units"] * (panel["promo_price"] - panel["cost"]), 2)

    # Confidence score: high when promo depth is close to historically observed depths
    obs_depth_mean = panel.groupby("item_id")["discount_pct"].transform("mean")
    obs_depth_std = panel.groupby("item_id")["discount_pct"].transform("std").fillna(0.05)
    z = (panel["discount_pct"] - obs_depth_mean).abs() / (obs_depth_std + 0.01)
    panel["confidence_score"] = (1.0 / (1.0 + 0.25 * z)).clip(0.35, 0.97).round(3)

    # Tidy & order
    keep = [
        "date", "week", "fiscal_week", "store_id", "store_name", "region",
        "item_id", "item_description", "category", "department", "vendor",
        "regular_price", "promo_price", "discount_pct", "promo_flag", "promo_tactic",
        "units", "sales", "cost", "gross_margin",
        "inventory_on_hand", "sell_through_pct", "vendor_funding_pct",
        "ly_units", "ly_sales",
        "baseline_units", "forecast_units", "forecast_sales", "forecast_margin",
        "elasticity", "confidence_score", "year_offset",
    ]
    return panel[keep].reset_index(drop=True)


def split_ly_ty(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (last_year, this_year) views — TY is the planning horizon."""
    ly = df[df["year_offset"] == 0].copy()
    ty = df[df["year_offset"] == 1].copy()
    return ly, ty
