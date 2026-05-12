"""Pre-Season Forecasting Workbench — merchant & planner workspace."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.data_loader import load_dataset
from src.ui_components import (
    page_setup, kpi_row, fmt_money, fmt_int, fmt_pct,
    sidebar_filters, apply_filters, line_by_week, PALETTE,
)
from src.optimization import demand_at_price, confidence_for_discount, Guardrails


page_setup("Pre-Season Forecasting Workbench", icon=":wrench:")

if "df" not in st.session_state:
    df, source = load_dataset(mode=st.session_state.get("data_mode", "auto"))
    st.session_state["df"] = df
    st.session_state["source"] = source
df = st.session_state["df"]

filters = sidebar_filters(df)
view = apply_filters(df[df["year_offset"] == 1], filters)

st.markdown(
    "Build, edit, and pressure-test the **TY pre-season plan**. "
    "Adjust discount depth, tactic, duration, and vendor funding — the forecast updates live."
)

# --------------------------------------------------------------------------- #
# Item-week forecast table
# --------------------------------------------------------------------------- #

st.subheader("Item × week forecast")
forecast_cols = [
    "item_id", "item_description", "category", "fiscal_week",
    "baseline_units", "forecast_units", "forecast_sales", "forecast_margin",
    "regular_price", "promo_price", "discount_pct", "elasticity", "confidence_score",
]
g = (view.groupby(["item_id", "item_description", "category", "fiscal_week"], as_index=False)
        .agg(baseline_units=("baseline_units", "sum"),
             forecast_units=("forecast_units", "sum"),
             forecast_sales=("forecast_sales", "sum"),
             forecast_margin=("forecast_margin", "sum"),
             regular_price=("regular_price", "mean"),
             promo_price=("promo_price", "mean"),
             discount_pct=("discount_pct", "mean"),
             elasticity=("elasticity", "mean"),
             confidence_score=("confidence_score", "mean")))
g = g.rename(columns={"promo_price": "planned_price"})
st.dataframe(g.head(500), use_container_width=True, hide_index=True)
st.caption(f"Showing first 500 of {len(g):,} item-week rows.")

st.markdown("---")

# --------------------------------------------------------------------------- #
# Editable promo controls
# --------------------------------------------------------------------------- #

st.subheader("Edit a promo plan")
items_in_view = view[["item_id", "item_description"]].drop_duplicates().sort_values("item_description")
if items_in_view.empty:
    st.warning("No items in the current filter set.")
    st.stop()
item_label = st.selectbox(
    "Pick an item to plan",
    options=items_in_view["item_id"].tolist(),
    format_func=lambda i: items_in_view.set_index("item_id")["item_description"].get(i, str(i)),
)
item_panel = view[view["item_id"] == item_label]
item_row = item_panel.iloc[0]

c1, c2, c3 = st.columns(3)
with c1:
    new_discount = st.slider("Discount depth", 0.0, 0.50,
                             float(round(item_panel["discount_pct"].max(), 2)), 0.01)
    tactic = st.selectbox("Promo tactic",
                          ["No Promo", "Percent Off", "Dollar Off", "BOGO", "Clipless", "Digital"],
                          index=1 if new_discount > 0 else 0)
with c2:
    weeks_present = sorted(item_panel["fiscal_week"].unique().tolist())
    start_wk, end_wk = st.select_slider(
        "Promo window (fiscal week)",
        options=weeks_present,
        value=(weeks_present[0], weeks_present[min(3, len(weeks_present)-1)]),
    )
    duration = (end_wk - start_wk + 1)
    st.caption(f"Duration: {duration} week(s)")
with c3:
    vendor_funding = st.slider("Vendor funding %",
                               0.0, 0.20, float(item_row["vendor_funding_pct"]), 0.01)
    cannibalization = st.slider("Cannibalization assumption (placeholder)",
                                0.0, 0.30, 0.05, 0.01,
                                help="Share of incremental units assumed to cannibalize substitutes.")

# Recompute forecast for the planned window
plan_panel = item_panel.copy()
in_window = (plan_panel["fiscal_week"] >= start_wk) & (plan_panel["fiscal_week"] <= end_wk)
plan_panel.loc[in_window, "promo_price"] = (plan_panel.loc[in_window, "regular_price"]
                                             * (1 - new_discount)).round(2)
plan_panel.loc[in_window, "promo_tactic"] = tactic
plan_panel["discount_pct"] = (1 - plan_panel["promo_price"] / plan_panel["regular_price"]).clip(lower=0)
plan_panel["vendor_funding_pct"] = vendor_funding

# Elasticity-based demand
ratio = (plan_panel["promo_price"] / plan_panel["regular_price"]).clip(lower=0.4)
lift = np.power(ratio, plan_panel["elasticity"]).clip(upper=4.5)
plan_panel["forecast_units"] = np.maximum(0, np.round(plan_panel["baseline_units"] * lift)).astype(int)
# Apply cannibalization haircut to *incremental* units
incr_units = (plan_panel["forecast_units"] - plan_panel["baseline_units"]).clip(lower=0)
plan_panel["forecast_units"] = (plan_panel["baseline_units"] + incr_units * (1 - cannibalization)).round().astype(int)

plan_panel["forecast_sales"] = (plan_panel["forecast_units"] * plan_panel["promo_price"]).round(2)
plan_panel["forecast_margin"] = (plan_panel["forecast_units"] * (plan_panel["promo_price"] - plan_panel["cost"])).round(2)
plan_panel["vendor_funding_dollars"] = (plan_panel["forecast_units"] * plan_panel["regular_price"] * vendor_funding).round(2)
plan_panel["adjusted_margin"] = plan_panel["forecast_margin"] + plan_panel["vendor_funding_dollars"]
plan_panel["confidence_score"] = [
    confidence_for_discount(d, Guardrails(observed_discount_mean=item_panel["discount_pct"].mean(),
                                          observed_discount_std=max(0.04, item_panel["discount_pct"].std() or 0.05)))
    for d in plan_panel["discount_pct"]
]

# --------------------------------------------------------------------------- #
# Live impact KPIs
# --------------------------------------------------------------------------- #

before = item_panel
after = plan_panel
kpi_row([
    {"label": "Demand lift vs baseline",
     "value": fmt_pct(after["forecast_units"].sum() / max(1, before["baseline_units"].sum()) - 1),
     "delta_positive": after["forecast_units"].sum() >= before["baseline_units"].sum()},
    {"label": "Revenue impact",
     "value": fmt_money(after["forecast_sales"].sum() - before["forecast_sales"].sum()),
     "delta_positive": after["forecast_sales"].sum() >= before["forecast_sales"].sum()},
    {"label": "Margin impact",
     "value": fmt_money(after["forecast_margin"].sum() - before["forecast_margin"].sum()),
     "delta_positive": after["forecast_margin"].sum() >= before["forecast_margin"].sum()},
    {"label": "Vendor-funded margin",
     "value": fmt_money(after["vendor_funding_dollars"].sum())},
    {"label": "Confidence", "value": fmt_pct(after["confidence_score"].mean())},
])

c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(line_by_week(after, "forecast_units", "Planned forecast units by week"),
                    use_container_width=True)
with c2:
    st.plotly_chart(line_by_week(after, "forecast_margin", "Planned forecast margin by week",
                                  color=PALETTE["accent"]),
                    use_container_width=True)

st.markdown("**Planned weeks (item-store detail)**")
st.dataframe(plan_panel[in_window][[
    "fiscal_week", "store_id", "regular_price", "promo_price", "discount_pct",
    "forecast_units", "forecast_sales", "forecast_margin",
    "vendor_funding_dollars", "adjusted_margin", "confidence_score",
]].sort_values(["fiscal_week", "store_id"]),
    use_container_width=True, hide_index=True)
