"""Inventory & Sell-Through Risk — markdown vs OOS classification."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_loader import load_dataset
from src.ui_components import (
    page_setup, kpi_row, fmt_money, fmt_int, fmt_pct,
    sidebar_filters, apply_filters, PALETTE,
)


page_setup("Inventory & Sell-Through Risk", icon=":package:")

if "df" not in st.session_state:
    df, source = load_dataset(mode=st.session_state.get("data_mode", "auto"))
    st.session_state["df"] = df
    st.session_state["source"] = source
df = st.session_state["df"]

filters = sidebar_filters(df, show_item=False)
ty = apply_filters(df[df["year_offset"] == 1], filters)

target_sell_through = st.sidebar.slider("Target sell-through %", 0.40, 0.95, 0.75, 0.05)

# --------------------------------------------------------------------------- #
# Risk classification per item
# --------------------------------------------------------------------------- #

inv = (ty.groupby(["item_id", "item_description", "category", "department"], as_index=False)
         .agg(inventory_on_hand=("inventory_on_hand", "sum"),
              forecast_units=("forecast_units", "sum"),
              forecast_sales=("forecast_sales", "sum"),
              forecast_margin=("forecast_margin", "sum"),
              regular_price=("regular_price", "mean"),
              cost=("cost", "mean")))
inv["projected_sell_through"] = (inv["forecast_units"] /
                                 (inv["inventory_on_hand"] + inv["forecast_units"]).clip(lower=1))
weeks_in_view = max(1, ty["fiscal_week"].nunique())
inv["weekly_velocity"] = inv["forecast_units"] / weeks_in_view
inv["weeks_of_supply"] = inv["inventory_on_hand"] / inv["weekly_velocity"].replace(0, np.nan)


def _classify(row) -> str:
    if row["projected_sell_through"] >= min(0.95, target_sell_through + 0.15):
        return "Underbought / OOS risk"
    if row["projected_sell_through"] < target_sell_through - 0.20:
        return "Overbought / markdown risk"
    return "Healthy"


def _recommend(row) -> str:
    if row["risk"] == "Underbought / OOS risk":
        return "Increase buy / shift allocation"
    if row["risk"] == "Overbought / markdown risk":
        return "Add promo / plan markdown"
    return "Hold price"


inv["risk"] = inv.apply(_classify, axis=1)
inv["recommendation"] = inv.apply(_recommend, axis=1)
inv["markdown_risk_$"] = np.where(
    inv["risk"] == "Overbought / markdown risk",
    (inv["inventory_on_hand"] - inv["forecast_units"]).clip(lower=0) * inv["regular_price"] * 0.15,
    0.0,
)
inv["missed_sales_risk_$"] = np.where(
    inv["risk"] == "Underbought / OOS risk",
    (inv["forecast_units"] * 0.10) * inv["regular_price"],
    0.0,
)

# --------------------------------------------------------------------------- #
# KPI strip
# --------------------------------------------------------------------------- #

kpi_row([
    {"label": "Items at OOS risk",
     "value": fmt_int((inv["risk"] == "Underbought / OOS risk").sum())},
    {"label": "Items at markdown risk",
     "value": fmt_int((inv["risk"] == "Overbought / markdown risk").sum())},
    {"label": "Estimated markdown exposure",
     "value": fmt_money(inv["markdown_risk_$"].sum())},
    {"label": "Estimated missed-sales exposure",
     "value": fmt_money(inv["missed_sales_risk_$"].sum())},
])

# --------------------------------------------------------------------------- #
# Projected sell-through by week (across selected scope)
# --------------------------------------------------------------------------- #
wk = (ty.groupby("fiscal_week", as_index=False)
        .agg(forecast_units=("forecast_units", "sum"),
             inventory_on_hand=("inventory_on_hand", "sum")))
wk["projected_sell_through"] = wk["forecast_units"] / (
    wk["forecast_units"] + wk["inventory_on_hand"]).clip(lower=1)
fig = px.line(wk, x="fiscal_week", y="projected_sell_through",
              title="Projected sell-through by week", markers=True)
fig.add_hline(y=target_sell_through, line_dash="dash",
              annotation_text=f"Target {target_sell_through:.0%}")
fig.update_yaxes(tickformat=".0%")
fig.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------- #
# Risk tables
# --------------------------------------------------------------------------- #
c1, c2 = st.columns(2)
with c1:
    st.subheader("Highest markdown risk")
    md = inv.sort_values("markdown_risk_$", ascending=False).head(12)
    md["projected_sell_through"] = md["projected_sell_through"].map(fmt_pct)
    md["weeks_of_supply"] = md["weeks_of_supply"].map(lambda v: f"{v:.1f}" if pd.notna(v) else "—")
    md["markdown_risk_$"] = md["markdown_risk_$"].map(fmt_money)
    st.dataframe(md[["item_description", "category", "projected_sell_through",
                     "weeks_of_supply", "markdown_risk_$", "recommendation"]],
                 use_container_width=True, hide_index=True)
with c2:
    st.subheader("Highest OOS / missed-sales risk")
    oos = inv.sort_values("missed_sales_risk_$", ascending=False).head(12)
    oos["projected_sell_through"] = oos["projected_sell_through"].map(fmt_pct)
    oos["weeks_of_supply"] = oos["weeks_of_supply"].map(lambda v: f"{v:.1f}" if pd.notna(v) else "—")
    oos["missed_sales_risk_$"] = oos["missed_sales_risk_$"].map(fmt_money)
    st.dataframe(oos[["item_description", "category", "projected_sell_through",
                      "weeks_of_supply", "missed_sales_risk_$", "recommendation"]],
                 use_container_width=True, hide_index=True)

st.subheader("Full risk classification")
view_df = inv.copy()
view_df["projected_sell_through"] = view_df["projected_sell_through"].map(fmt_pct)
view_df["weeks_of_supply"] = view_df["weeks_of_supply"].map(lambda v: f"{v:.1f}" if pd.notna(v) else "—")
view_df["forecast_sales"] = view_df["forecast_sales"].map(fmt_money)
view_df["markdown_risk_$"] = view_df["markdown_risk_$"].map(fmt_money)
view_df["missed_sales_risk_$"] = view_df["missed_sales_risk_$"].map(fmt_money)
st.dataframe(view_df[["item_description", "category", "department",
                      "inventory_on_hand", "forecast_units", "projected_sell_through",
                      "weeks_of_supply", "forecast_sales",
                      "risk", "markdown_risk_$", "missed_sales_risk_$", "recommendation"]]
             .sort_values("risk"),
             use_container_width=True, hide_index=True)
