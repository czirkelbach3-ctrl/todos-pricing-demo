"""
Scenario engine — generates and scores the canonical five scenarios used by
the Scenario Sandbox page.

Each scenario is a *policy* applied to a base panel (item × store × week).
The engine recomputes price, discount, units, sales, and margin for every
row under the scenario policy, then aggregates and scores.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List

import numpy as np
import pandas as pd

from .optimization import demand_at_price, find_optimal_price, Guardrails


# --------------------------------------------------------------------------- #
# Scenario policies
# --------------------------------------------------------------------------- #

@dataclass
class ScenarioPolicy:
    name: str
    description: str
    apply: Callable[[pd.DataFrame], pd.DataFrame]


def _apply_economics(panel: pd.DataFrame) -> pd.DataFrame:
    """Recompute units/sales/margin from price changes using elasticity."""
    out = panel.copy()
    out["discount_pct"] = (1 - out["promo_price"] / out["regular_price"]).clip(lower=0.0)
    # Elasticity-based demand response, anchored on baseline_units
    ratio = (out["promo_price"] / out["regular_price"]).clip(lower=0.4)
    lift = np.power(ratio, out["elasticity"]).clip(upper=4.5)
    out["forecast_units"] = np.maximum(0, np.round(out["baseline_units"] * lift)).astype(int)
    out["forecast_sales"] = (out["forecast_units"] * out["promo_price"]).round(2)
    out["forecast_margin"] = (out["forecast_units"] * (out["promo_price"] - out["cost"])).round(2)
    out["vendor_funding_dollars"] = (out["forecast_units"] * out["regular_price"] * out["vendor_funding_pct"]).round(2)
    out["adjusted_margin"] = out["forecast_margin"] + out["vendor_funding_dollars"]
    out["promo_flag"] = (out["discount_pct"] > 0.0).astype(int)
    # Sell-through estimate: forecast units / planned inventory
    out["projected_sell_through"] = (out["forecast_units"] /
                                     (out["inventory_on_hand"] + out["forecast_units"]).clip(lower=1)).clip(0, 1)
    return out


def _baseline_no_promo(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["promo_price"] = out["regular_price"]
    out["promo_tactic"] = "No Promo"
    return _apply_economics(out)


def _current_plan(df: pd.DataFrame) -> pd.DataFrame:
    # "Current plan" = whatever was on the panel as-is (the merchant's TY plan).
    return _apply_economics(df.copy())


def _deeper_discount(df: pd.DataFrame, extra: float = 0.10) -> pd.DataFrame:
    out = df.copy()
    on_promo = out["promo_flag"] == 1
    out.loc[on_promo, "promo_price"] = (out.loc[on_promo, "promo_price"] *
                                        (1 - extra)).round(2)
    return _apply_economics(out)


def _optimized_price(df: pd.DataFrame, guardrails: Guardrails) -> pd.DataFrame:
    """For each item, search for the adjusted-margin-maximizing price.
    This is slow per-row, so we solve once per item and broadcast.
    """
    out = df.copy()
    by_item = (out.groupby("item_id")
                 .agg(regular_price=("regular_price", "first"),
                      cost=("cost", "first"),
                      elasticity=("elasticity", "first"),
                      vendor_funding_pct=("vendor_funding_pct", "first"),
                      baseline_units=("baseline_units", "mean"))
                 .reset_index())
    rec_prices = []
    for r in by_item.itertuples(index=False):
        rec = find_optimal_price(r.baseline_units, r.regular_price, r.cost,
                                 r.elasticity, r.vendor_funding_pct, guardrails)
        rec_prices.append({"item_id": r.item_id, "rec_price": rec["price"]})
    rec_df = pd.DataFrame(rec_prices)
    out = out.merge(rec_df, on="item_id", how="left")
    out["promo_price"] = out["rec_price"].fillna(out["regular_price"]).round(2)
    out = out.drop(columns=["rec_price"])
    out["promo_tactic"] = np.where(out["promo_price"] < out["regular_price"],
                                   "Percent Off", "No Promo")
    return _apply_economics(out)


def _vendor_funded_aggressive(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Aggressive: drop to 30% off, with vendor funding offsetting margin pain.
    out["promo_price"] = (out["regular_price"] * 0.70).round(2)
    out["promo_tactic"] = "Percent Off"
    # Simulate vendor agreeing to fund extra 5pts on top of current funding %
    out["vendor_funding_pct"] = (out["vendor_funding_pct"] + 0.05).clip(upper=0.20)
    return _apply_economics(out)


def build_scenarios(panel: pd.DataFrame,
                    guardrails: Guardrails) -> Dict[str, pd.DataFrame]:
    return {
        "Baseline / No Promo": _baseline_no_promo(panel),
        "Current Plan":        _current_plan(panel),
        "Deeper Discount":     _deeper_discount(panel, extra=0.10),
        "Optimized Price":     _optimized_price(panel, guardrails),
        "Vendor-Funded Aggressive": _vendor_funded_aggressive(panel),
    }


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

@dataclass
class ScenarioWeights:
    margin_impact: float = 0.35
    sales_impact: float = 0.25
    sell_through: float = 0.20
    confidence:   float = 0.15
    vendor_funding: float = 0.05

    def normalized(self) -> "ScenarioWeights":
        total = (self.margin_impact + self.sales_impact + self.sell_through
                 + self.confidence + self.vendor_funding)
        if total == 0:
            return self
        return ScenarioWeights(
            margin_impact=self.margin_impact / total,
            sales_impact=self.sales_impact / total,
            sell_through=self.sell_through / total,
            confidence=self.confidence / total,
            vendor_funding=self.vendor_funding / total,
        )


def summarize_scenario(name: str, df: pd.DataFrame, baseline: pd.DataFrame) -> dict:
    sales = df["forecast_sales"].sum()
    units = df["forecast_units"].sum()
    margin = df["forecast_margin"].sum()
    adj_margin = df["adjusted_margin"].sum()
    vendor = df["vendor_funding_dollars"].sum()
    base_sales = baseline["forecast_sales"].sum()
    base_margin = baseline["forecast_margin"].sum()
    incremental_sales = sales - base_sales
    incremental_margin = adj_margin - base_margin
    sell_through = df["projected_sell_through"].mean()
    confidence = df["confidence_score"].mean() if "confidence_score" in df.columns else 0.75
    inventory_risk = float((df["projected_sell_through"] < 0.40).mean()) + \
                     float((df["projected_sell_through"] > 0.95).mean())
    return {
        "scenario": name,
        "forecast_sales": float(sales),
        "forecast_units": float(units),
        "forecast_margin": float(margin),
        "adjusted_margin": float(adj_margin),
        "margin_rate": float(margin / sales) if sales > 0 else 0.0,
        "vendor_funding": float(vendor),
        "incremental_sales": float(incremental_sales),
        "incremental_margin": float(incremental_margin),
        "avg_sell_through": float(sell_through),
        "avg_confidence": float(confidence),
        "inventory_risk_share": float(inventory_risk),
    }


def score_scenarios(summaries: List[dict], weights: ScenarioWeights) -> pd.DataFrame:
    df = pd.DataFrame(summaries)
    w = weights.normalized()

    def _norm(col: str, higher_is_better: bool = True) -> pd.Series:
        x = df[col].astype(float)
        if x.max() == x.min():
            return pd.Series([0.5] * len(df), index=df.index)
        s = (x - x.min()) / (x.max() - x.min())
        return s if higher_is_better else 1 - s

    df["score"] = (
        w.margin_impact   * _norm("incremental_margin") +
        w.sales_impact    * _norm("incremental_sales") +
        w.sell_through    * _norm("avg_sell_through") +
        w.confidence      * _norm("avg_confidence") +
        w.vendor_funding  * _norm("vendor_funding")
    )
    df["rank"] = df["score"].rank(ascending=False, method="dense").astype(int)
    return df.sort_values("rank")
