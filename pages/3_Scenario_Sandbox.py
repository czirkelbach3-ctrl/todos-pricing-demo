"""Scenario Sandbox — five canonical scenarios with tunable scoring weights."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_loader import load_dataset
from src.ui_components import (
    page_setup, kpi_row, fmt_money, fmt_int, fmt_pct, badge,
    sidebar_filters, apply_filters,
)
from src.scenario_engine import (
    build_scenarios, summarize_scenario, score_scenarios, ScenarioWeights,
)
from src.optimization import Guardrails


page_setup("Scenario Sandbox", icon=":crystal_ball:")

if "df" not in st.session_state:
    df, source = load_dataset(mode=st.session_state.get("data_mode", "auto"))
    st.session_state["df"] = df
    st.session_state["source"] = source
df = st.session_state["df"]

filters = sidebar_filters(df, show_item=False)
ty = apply_filters(df[df["year_offset"] == 1], filters)

st.markdown(
    "Compare five canonical scenarios side-by-side. Adjust the **scoring weights** "
    "to reflect leadership priorities — the recommended badge updates live."
)

# --------------------------------------------------------------------------- #
# Guardrails + weights
# --------------------------------------------------------------------------- #
with st.sidebar.expander("Guardrails", expanded=False):
    allow_above = st.checkbox("Allow price > regular", value=False)
    allow_below = st.checkbox("Allow price < cost", value=False)
    allow_neg = st.checkbox("Allow negative margin (vendor-offset)", value=False)
    max_disc = st.slider("Max discount %", 0.0, 0.60, 0.40, 0.05)
guardrails = Guardrails(allow_above_regular=allow_above, allow_below_cost=allow_below,
                        allow_negative_margin=allow_neg, max_discount_pct=max_disc)

st.sidebar.markdown("### Scoring weights")
w_margin = st.sidebar.slider("Margin impact",   0.0, 1.0, 0.35, 0.05)
w_sales  = st.sidebar.slider("Sales impact",    0.0, 1.0, 0.25, 0.05)
w_sell   = st.sidebar.slider("Sell-through",    0.0, 1.0, 0.20, 0.05)
w_conf   = st.sidebar.slider("Confidence",      0.0, 1.0, 0.15, 0.05)
w_vend   = st.sidebar.slider("Vendor funding",  0.0, 1.0, 0.05, 0.05)
weights = ScenarioWeights(w_margin, w_sales, w_sell, w_conf, w_vend)

# --------------------------------------------------------------------------- #
# Build scenarios
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Building scenarios...", ttl=60 * 30)
def _scenarios_cached(panel: pd.DataFrame, _g_key: tuple):
    return build_scenarios(panel, guardrails)

scenarios = _scenarios_cached(ty, (allow_above, allow_below, allow_neg, max_disc))
baseline = scenarios["Baseline / No Promo"]
summaries = [summarize_scenario(name, s, baseline) for name, s in scenarios.items()]
ranked = score_scenarios(summaries, weights)

# --------------------------------------------------------------------------- #
# Comparison table
# --------------------------------------------------------------------------- #
st.subheader("Side-by-side comparison")
disp = ranked.copy()
disp["forecast_sales"] = disp["forecast_sales"].map(fmt_money)
disp["forecast_units"] = disp["forecast_units"].map(fmt_int)
disp["forecast_margin"] = disp["forecast_margin"].map(fmt_money)
disp["adjusted_margin"] = disp["adjusted_margin"].map(fmt_money)
disp["margin_rate"] = disp["margin_rate"].map(fmt_pct)
disp["vendor_funding"] = disp["vendor_funding"].map(fmt_money)
disp["incremental_sales"] = disp["incremental_sales"].map(fmt_money)
disp["incremental_margin"] = disp["incremental_margin"].map(fmt_money)
disp["avg_sell_through"] = disp["avg_sell_through"].map(fmt_pct)
disp["avg_confidence"] = disp["avg_confidence"].map(fmt_pct)
disp["score"] = disp["score"].map(lambda v: f"{v:.3f}")

st.dataframe(disp[[
    "rank", "scenario", "forecast_sales", "forecast_units", "forecast_margin",
    "adjusted_margin", "margin_rate", "vendor_funding",
    "incremental_sales", "incremental_margin",
    "avg_sell_through", "avg_confidence", "score",
]], use_container_width=True, hide_index=True)

best = ranked.iloc[0]
st.success(
    f"**Recommended scenario:** {best['scenario']}   ·   "
    f"Adjusted margin {fmt_money(best['adjusted_margin'])}   ·   "
    f"Score {best['score']:.3f}"
)

# --------------------------------------------------------------------------- #
# Bar charts
# --------------------------------------------------------------------------- #
c1, c2 = st.columns(2)
with c1:
    fig = px.bar(ranked, x="scenario", y="adjusted_margin",
                 title="Adjusted margin by scenario",
                 text=ranked["adjusted_margin"].map(fmt_money))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
with c2:
    fig = px.bar(ranked, x="scenario", y="incremental_sales",
                 title="Incremental sales vs baseline",
                 text=ranked["incremental_sales"].map(fmt_money))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------- #
# Per-scenario details
# --------------------------------------------------------------------------- #
st.markdown("### Scenario detail")
which = st.selectbox("Pick a scenario to inspect", list(scenarios.keys()))
panel = scenarios[which]
st.dataframe(
    panel.groupby(["item_id", "item_description", "category"], as_index=False)
         .agg(planned_price=("promo_price", "mean"),
              discount_pct=("discount_pct", "mean"),
              forecast_units=("forecast_units", "sum"),
              forecast_sales=("forecast_sales", "sum"),
              forecast_margin=("forecast_margin", "sum"),
              vendor_funding=("vendor_funding_dollars", "sum"),
              projected_sell_through=("projected_sell_through", "mean"))
         .sort_values("forecast_sales", ascending=False)
         .head(50),
    use_container_width=True, hide_index=True,
)
