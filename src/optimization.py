"""
Price optimization engine.

Uses a constant-elasticity demand model:
    demand(p) = baseline_units * (p / regular_price) ** elasticity

This is the standard log-log retail elasticity form and is mechanically what a
pricing analyst would defend in a review meeting. Real production tools layer
hierarchical Bayes / structural models on top, but this captures the core
shape and gives realistic curves for a demo.

Objective: maximize *adjusted_margin* = unit_margin*units + vendor_funding_$.

Guardrails are explicit (and toggleable) because in real merchandising the
buyer needs to override them with justification.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class Guardrails:
    allow_above_regular: bool = False
    allow_below_cost: bool = False
    allow_negative_margin: bool = False
    max_discount_pct: float = 0.40
    min_discount_pct: float = 0.0
    # Confidence haircut grows as the planned discount deviates from observed
    # discount history. Captures "we've never run -40% on this item before".
    observed_discount_mean: float = 0.10
    observed_discount_std: float = 0.08


def demand_at_price(baseline_units: float, regular_price: float,
                    new_price: float, elasticity: float) -> float:
    if regular_price <= 0:
        return 0.0
    ratio = max(new_price / regular_price, 0.01)
    return float(max(0.0, baseline_units * (ratio ** elasticity)))


def price_economics(baseline_units: float, regular_price: float, cost: float,
                    new_price: float, elasticity: float,
                    vendor_funding_pct: float = 0.0) -> dict[str, float]:
    units = demand_at_price(baseline_units, regular_price, new_price, elasticity)
    revenue = units * new_price
    gross_margin = units * (new_price - cost)
    vendor_funding_dollars = units * regular_price * vendor_funding_pct
    adjusted_margin = gross_margin + vendor_funding_dollars
    return {
        "units": units,
        "revenue": revenue,
        "gross_margin": gross_margin,
        "vendor_funding": vendor_funding_dollars,
        "adjusted_margin": adjusted_margin,
        "margin_rate": gross_margin / revenue if revenue > 0 else 0.0,
    }


def confidence_for_discount(discount_pct: float, g: Guardrails) -> float:
    """Confidence decays as discount moves outside historical observed range."""
    z = abs(discount_pct - g.observed_discount_mean) / max(1e-3, g.observed_discount_std)
    return float(np.clip(1.0 / (1.0 + 0.25 * z), 0.35, 0.97))


def price_response_curve(baseline_units: float, regular_price: float, cost: float,
                         elasticity: float, vendor_funding_pct: float = 0.0,
                         n: int = 41,
                         guardrails: Optional[Guardrails] = None) -> pd.DataFrame:
    """Sweep candidate prices and return the full response curve."""
    g = guardrails or Guardrails()

    low = regular_price * (1 - g.max_discount_pct)
    high = regular_price * (1.10 if g.allow_above_regular else 1.0)
    prices = np.linspace(low, high, n)

    rows = []
    for p in prices:
        econ = price_economics(baseline_units, regular_price, cost, p, elasticity, vendor_funding_pct)
        discount = max(0.0, 1 - p / regular_price)
        econ.update({
            "price": p,
            "discount_pct": discount,
            "confidence_score": confidence_for_discount(discount, g),
        })
        rows.append(econ)
    return pd.DataFrame(rows)


def find_optimal_price(baseline_units: float, regular_price: float, cost: float,
                       elasticity: float, vendor_funding_pct: float = 0.0,
                       guardrails: Optional[Guardrails] = None) -> dict:
    """Search the response curve for the adjusted-margin-maximizing price.

    Returns recommendation incl. price, discount, expected units/sales/margin,
    confidence and any triggered risk flags.
    """
    g = guardrails or Guardrails()
    curve = price_response_curve(baseline_units, regular_price, cost, elasticity,
                                 vendor_funding_pct, n=81, guardrails=g)

    valid = curve.copy()
    if not g.allow_below_cost:
        valid = valid[valid["price"] >= cost]
    if not g.allow_negative_margin:
        # adjusted margin must be ≥ 0 even after vendor funding
        valid = valid[valid["adjusted_margin"] >= 0]
    valid = valid[(valid["discount_pct"] >= g.min_discount_pct) &
                  (valid["discount_pct"] <= g.max_discount_pct)]

    if valid.empty:
        # Fall back to regular price if every candidate violated guardrails.
        valid = curve[curve["price"] == regular_price]

    best = valid.sort_values("adjusted_margin", ascending=False).iloc[0]

    risks = []
    if best["price"] < cost:
        risks.append("Below cost")
    if best["discount_pct"] > 0.30:
        risks.append("Deep discount (>30%)")
    if best["confidence_score"] < 0.55:
        risks.append("Low historical confidence")
    if best["margin_rate"] < 0.05:
        risks.append("Thin margin rate")

    return {
        "price": float(best["price"]),
        "discount_pct": float(best["discount_pct"]),
        "expected_units": float(best["units"]),
        "expected_revenue": float(best["revenue"]),
        "expected_margin": float(best["gross_margin"]),
        "adjusted_margin": float(best["adjusted_margin"]),
        "vendor_funding_dollars": float(best["vendor_funding"]),
        "confidence_score": float(best["confidence_score"]),
        "risks": risks,
        "curve": curve,
    }
