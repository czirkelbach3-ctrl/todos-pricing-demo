"""
AI-Powered Retail Pricing Optimization — Prototype
---------------------------------------------------
A Streamlit demo that recommends dynamic prices for retail SKUs (dairy &
snacks) using a constant-elasticity demand model plus adjustments for
demand signals, competitor prices, and time-of-day factors.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Retail Pricing Optimizer",
    page_icon="🛒",
    layout="wide",
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "sample_skus.csv")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_skus(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the sample SKU master file.

    Schema is modeled after public Kaggle retail datasets such as
    'Retail Price Optimization' (Sandeep Singh) and 'Grocery Sales' —
    cost, price, demand units, and an estimated own-price elasticity.
    """
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Pricing engine
# ---------------------------------------------------------------------------
@dataclass
class PricingInputs:
    demand_signal: float        # multiplier on base demand (e.g., 1.2 = +20%)
    competitor_weight: float    # 0–1: how much to anchor to competitor
    time_of_day_factor: float   # multiplier (e.g., evening rush = 1.15)
    elasticity_override: float  # if provided, overrides SKU elasticity
    min_margin_pct: float       # floor margin (e.g., 0.15 = 15%)
    max_price_change_pct: float # cap on price movement vs static (e.g., 0.25)


def demand_at_price(base_demand: float, base_price: float, price: float,
                    elasticity: float) -> float:
    """Constant-elasticity demand: Q = Q0 * (P/P0)^elasticity.

    Elasticity is expected to be negative. This is the workhorse model in
    most retail pricing textbooks (Phillips, 'Pricing and Revenue
    Optimization') and is the simplest form that still produces sensible
    profit-maximizing solutions.
    """
    if base_price <= 0:
        return 0.0
    return base_demand * (price / base_price) ** elasticity


def optimal_price_elasticity(unit_cost: float, elasticity: float) -> float:
    """Closed-form profit-maximizing price under constant elasticity.

    From the standard monopoly markup formula:
        P* = c * e / (e + 1)   where e = elasticity (negative)
    Only valid for |e| > 1 (elastic demand); we clip otherwise.
    """
    if elasticity >= -1:  # inelastic — formula blows up, fall back to markup
        return unit_cost * 2.0
    return unit_cost * elasticity / (elasticity + 1)


def ai_recommended_price(row: pd.Series, inp: PricingInputs) -> dict:
    """Combine elasticity-optimal price with demand, competitor, and
    time-of-day signals, then enforce margin and movement guardrails.
    """
    elasticity = inp.elasticity_override if inp.elasticity_override else row["price_elasticity"]

    # 1. Start with elasticity-optimal price
    p_opt = optimal_price_elasticity(row["unit_cost"], elasticity)

    # 2. Anchor partially to competitor
    p_anchored = (1 - inp.competitor_weight) * p_opt + inp.competitor_weight * row["competitor_price"]

    # 3. Apply demand signal & time-of-day as a small multiplicative tilt
    # Stronger demand or peak hours -> higher price; capped at +/-15% influence
    tilt = 1 + 0.5 * (inp.demand_signal - 1) + 0.5 * (inp.time_of_day_factor - 1)
    tilt = float(np.clip(tilt, 0.85, 1.15))
    p_tilted = p_anchored * tilt

    # 4. Guardrails — margin floor and max movement vs static price
    min_price = row["unit_cost"] / (1 - inp.min_margin_pct) if inp.min_margin_pct < 1 else row["unit_cost"]
    static = row["static_price"]
    lo = static * (1 - inp.max_price_change_pct)
    hi = static * (1 + inp.max_price_change_pct)
    p_final = float(np.clip(p_tilted, max(min_price, lo), hi))

    # 5. Forecast demand & margin under both strategies
    base_demand = row["base_demand_units"] * inp.demand_signal * inp.time_of_day_factor
    q_static = demand_at_price(base_demand, static, static, elasticity)
    q_ai = demand_at_price(base_demand, static, p_final, elasticity)

    margin_static = (static - row["unit_cost"]) * q_static
    margin_ai = (p_final - row["unit_cost"]) * q_ai
    revenue_static = static * q_static
    revenue_ai = p_final * q_ai

    return {
        "sku_id": row["sku_id"],
        "sku_name": row["sku_name"],
        "category": row["category"],
        "unit_cost": row["unit_cost"],
        "static_price": static,
        "competitor_price": row["competitor_price"],
        "elasticity_used": elasticity,
        "ai_price": round(p_final, 2),
        "demand_static": round(q_static, 1),
        "demand_ai": round(q_ai, 1),
        "revenue_static": round(revenue_static, 2),
        "revenue_ai": round(revenue_ai, 2),
        "margin_static": round(margin_static, 2),
        "margin_ai": round(margin_ai, 2),
        "margin_lift_pct": round((margin_ai - margin_static) / margin_static * 100, 1)
            if margin_static else 0.0,
        "price_change_pct": round((p_final - static) / static * 100, 1),
    }


def run_optimizer(df: pd.DataFrame, inp: PricingInputs) -> pd.DataFrame:
    return pd.DataFrame([ai_recommended_price(r, inp) for _, r in df.iterrows()])


# ---------------------------------------------------------------------------
# UI — Sidebar inputs
# ---------------------------------------------------------------------------
st.sidebar.header("Pricing Signals")

demand_signal = st.sidebar.slider(
    "Demand signal (1.0 = normal)",
    min_value=0.6, max_value=1.6, value=1.0, step=0.05,
    help="Lift or drop in expected demand vs baseline (e.g., promo lift, weather).",
)

competitor_weight = st.sidebar.slider(
    "Competitor anchoring weight",
    min_value=0.0, max_value=1.0, value=0.30, step=0.05,
    help="0 = ignore competitor, 1 = match competitor exactly.",
)

time_of_day = st.sidebar.selectbox(
    "Time-of-day factor",
    options=[
        ("Early morning (0.95)", 0.95),
        ("Midday (1.00)", 1.00),
        ("Afternoon (1.05)", 1.05),
        ("Evening rush (1.15)", 1.15),
        ("Late night (0.90)", 0.90),
    ],
    index=1,
    format_func=lambda x: x[0],
)
time_of_day_factor = time_of_day[1]

elasticity_override = st.sidebar.number_input(
    "Elasticity override (0 = use SKU value)",
    min_value=-5.0, max_value=0.0, value=0.0, step=0.1,
    help="Negative number. Leave at 0 to use each SKU's estimated elasticity.",
)

min_margin = st.sidebar.slider(
    "Minimum margin floor",
    min_value=0.0, max_value=0.5, value=0.15, step=0.01,
    format="%.0f%%",
)

max_change = st.sidebar.slider(
    "Max price move vs static",
    min_value=0.05, max_value=0.50, value=0.25, step=0.05,
    format="%.0f%%",
)

inputs = PricingInputs(
    demand_signal=demand_signal,
    competitor_weight=competitor_weight,
    time_of_day_factor=time_of_day_factor,
    elasticity_override=elasticity_override,
    min_margin_pct=min_margin,
    max_price_change_pct=max_change,
)

# ---------------------------------------------------------------------------
# Header & data
# ---------------------------------------------------------------------------
st.title("🛒 AI-Powered Retail Pricing Optimizer")
st.caption(
    "Prototype for dynamic pricing on dairy & snacks SKUs. "
    "Adjust signals in the sidebar to see how AI prices, demand, and margins shift "
    "vs static (everyday) pricing."
)

df = load_skus()
category_filter = st.multiselect(
    "Filter categories", options=sorted(df["category"].unique()),
    default=sorted(df["category"].unique()),
)
df_view = df[df["category"].isin(category_filter)].copy()

results = run_optimizer(df_view, inputs)

# ---------------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
total_margin_static = results["margin_static"].sum()
total_margin_ai = results["margin_ai"].sum()
margin_lift = (total_margin_ai - total_margin_static) / total_margin_static * 100 \
    if total_margin_static else 0.0
rev_static = results["revenue_static"].sum()
rev_ai = results["revenue_ai"].sum()

col1.metric("Static margin ($)", f"${total_margin_static:,.0f}")
col2.metric("AI margin ($)", f"${total_margin_ai:,.0f}", f"{margin_lift:+.1f}%")
col3.metric("Static revenue ($)", f"${rev_static:,.0f}")
col4.metric("AI revenue ($)", f"${rev_ai:,.0f}",
            f"{(rev_ai - rev_static) / rev_static * 100:+.1f}%" if rev_static else None)

st.divider()

# ---------------------------------------------------------------------------
# Recommendations table
# ---------------------------------------------------------------------------
st.subheader("Per-SKU price recommendations")
display_cols = [
    "sku_id", "sku_name", "category", "unit_cost", "static_price",
    "competitor_price", "ai_price", "price_change_pct",
    "demand_ai", "margin_static", "margin_ai", "margin_lift_pct",
]
st.dataframe(
    results[display_cols].style.format({
        "unit_cost": "${:.2f}",
        "static_price": "${:.2f}",
        "competitor_price": "${:.2f}",
        "ai_price": "${:.2f}",
        "price_change_pct": "{:+.1f}%",
        "demand_ai": "{:.1f}",
        "margin_static": "${:.2f}",
        "margin_ai": "${:.2f}",
        "margin_lift_pct": "{:+.1f}%",
    }),
    use_container_width=True,
)

# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------
st.subheader("Static vs AI pricing — side-by-side")

price_long = results.melt(
    id_vars=["sku_name", "category"],
    value_vars=["static_price", "ai_price", "competitor_price"],
    var_name="Strategy", value_name="Price",
)
fig_price = px.bar(
    price_long, x="sku_name", y="Price", color="Strategy", barmode="group",
    title="Recommended price vs static & competitor",
    labels={"sku_name": "SKU"},
)
fig_price.update_layout(xaxis_tickangle=-30, height=420)
st.plotly_chart(fig_price, use_container_width=True)

st.subheader("Margin lift by SKU")
fig_margin = px.bar(
    results.sort_values("margin_lift_pct"),
    x="margin_lift_pct", y="sku_name", color="category", orientation="h",
    title="AI margin uplift vs static (%)",
    labels={"margin_lift_pct": "Margin lift (%)", "sku_name": "SKU"},
)
fig_margin.update_layout(height=420)
st.plotly_chart(fig_margin, use_container_width=True)

# ---------------------------------------------------------------------------
# Drill-down — price/profit curve for one SKU
# ---------------------------------------------------------------------------
st.subheader("Drill-down: profit curve at chosen signals")
sku_pick = st.selectbox("Pick an SKU", options=results["sku_name"].tolist())
row = df_view[df_view["sku_name"] == sku_pick].iloc[0]
rec = results[results["sku_name"] == sku_pick].iloc[0]
elasticity = rec["elasticity_used"]

p_range = np.linspace(row["unit_cost"] * 1.05, row["static_price"] * 1.6, 60)
base_demand = row["base_demand_units"] * demand_signal * time_of_day_factor
q_range = demand_at_price(base_demand, row["static_price"], p_range, elasticity)
profit = (p_range - row["unit_cost"]) * q_range

fig_curve = go.Figure()
fig_curve.add_trace(go.Scatter(x=p_range, y=profit, mode="lines", name="Profit ($)"))
fig_curve.add_vline(
    x=row["static_price"], line_dash="dash", line_color="gray",
    annotation_text="Static", annotation_position="top",
)
fig_curve.add_vline(
    x=rec["ai_price"], line_dash="dot", line_color="green",
    annotation_text="AI", annotation_position="top",
)
fig_curve.add_vline(
    x=row["competitor_price"], line_dash="dashdot", line_color="orange",
    annotation_text="Competitor", annotation_position="bottom",
)
fig_curve.update_layout(
    title=f"{sku_pick} — profit vs price (elasticity {elasticity:.2f})",
    xaxis_title="Price ($)", yaxis_title="Profit ($)", height=420,
)
st.plotly_chart(fig_curve, use_container_width=True)

# ---------------------------------------------------------------------------
# Methodology
# ---------------------------------------------------------------------------
with st.expander("How the AI pricing engine works"):
    st.markdown(
        """
        **Model.** Constant-elasticity demand: $Q = Q_0 (P/P_0)^{\\varepsilon}$
        with $\\varepsilon < 0$.

        **Step 1 — Elasticity-optimal price.** Closed-form monopoly markup:
        $P^* = c \\cdot \\varepsilon / (\\varepsilon + 1)$.

        **Step 2 — Competitor anchoring.** Blend $P^*$ with the observed
        competitor price using the sidebar weight.

        **Step 3 — Demand & time-of-day tilt.** Apply a capped multiplicative
        tilt so prices rise modestly when demand is strong or it's peak hour.

        **Step 4 — Guardrails.** Enforce a minimum margin floor and cap how
        far the AI can move from the everyday (static) price.

        **Static baseline.** Each SKU's `static_price` field — what the
        retailer would charge without dynamic optimization.
        """
    )

with st.expander("Data sources & how to swap in Kaggle data"):
    st.markdown(
        """
        The bundled `data/sample_skus.csv` is a synthetic file modeled after
        public Kaggle retail datasets:

        - *Retail Price Optimization* (Sandeep Singh)
        - *Grocery Store Dataset* (Heeral Dedhia)
        - *Instacart Market Basket Analysis*

        To use real Kaggle data, place a CSV at `data/sample_skus.csv` with
        columns: `sku_id, sku_name, category, unit_cost, static_price,
        base_demand_units, price_elasticity, competitor_price, shelf_life_days`.
        Elasticities can be estimated by regressing log-quantity on log-price
        with promo/seasonality controls.
        """
    )
