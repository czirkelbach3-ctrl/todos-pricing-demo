"""Promo Effectiveness / LY Hindsight — which LY promos worked?"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_loader import load_dataset
from src.ui_components import (
    page_setup, fmt_money, fmt_int, fmt_pct, sidebar_filters, apply_filters, PALETTE,
)


page_setup("Promo Effectiveness — LY Hindsight", icon=":mag:")

if "df" not in st.session_state:
    df, source = load_dataset(mode=st.session_state.get("data_mode", "auto"))
    st.session_state["df"] = df
    st.session_state["source"] = source
df = st.session_state["df"]

filters = sidebar_filters(df, show_item=False, show_week=False)
ly = apply_filters(df[df["year_offset"] == 0], filters)
ly_promo = ly[ly["promo_flag"] == 1].copy()

if ly_promo.empty:
    st.info("No promotional weeks in the selected scope.")
    st.stop()

# --------------------------------------------------------------------------- #
# Build LY-level effectiveness table
# --------------------------------------------------------------------------- #

agg = (ly_promo
       .groupby(["item_id", "item_description", "category", "promo_tactic"], as_index=False)
       .agg(discount_pct=("discount_pct", "mean"),
            ly_units=("units", "sum"),
            ly_sales=("sales", "sum"),
            ly_margin=("gross_margin", "sum"),
            estimated_baseline=("baseline_units", "sum"),
            cost=("cost", "mean"),
            regular_price=("regular_price", "mean")))
agg["incremental_units"] = agg["ly_units"] - agg["estimated_baseline"]
# Incremental margin = incremental units * (avg promo price - cost). Approximate
# promo price as regular * (1 - discount).
agg["incremental_sales"] = agg["incremental_units"] * agg["regular_price"] * (1 - agg["discount_pct"])
agg["incremental_margin"] = agg["incremental_units"] * (agg["regular_price"] * (1 - agg["discount_pct"]) - agg["cost"])
agg["roi"] = agg["incremental_margin"] / (
    (agg["estimated_baseline"] * agg["regular_price"] * agg["discount_pct"]).replace(0, np.nan)
)
# Effectiveness score: weighted blend of ROI, incremental units share, margin yield
incr_share = agg["incremental_units"] / agg["ly_units"].replace(0, np.nan)
agg["effectiveness_score"] = (
    0.50 * agg["roi"].clip(lower=-2, upper=4).fillna(0) +
    0.30 * incr_share.fillna(0) +
    0.20 * (agg["incremental_margin"] / agg["incremental_margin"].abs().max()).fillna(0)
)


def _recommend(row) -> str:
    if row["effectiveness_score"] >= 0.6 and row["roi"] >= 1.0:
        return "Keep"
    if 0.2 <= row["effectiveness_score"] < 0.6:
        return "Refine"
    if row["incremental_margin"] < 0:
        return "Drop"
    return "Test Alternative"


def _why(row) -> str:
    bits = []
    if row["roi"] >= 1.0:
        bits.append(f"ROI {row['roi']:.2f}× — funded itself")
    elif row["roi"] < 0:
        bits.append("Negative ROI — promo cost > incremental margin")
    if row["incremental_units"] < 0:
        bits.append("No demand lift detected")
    if row["discount_pct"] > 0.30:
        bits.append("Deep discount — consider shallower depth")
    if not bits:
        bits.append("Mid-tier performance — test alternative tactic")
    return "; ".join(bits)


agg["recommendation"] = agg.apply(_recommend, axis=1)
agg["reason"] = agg.apply(_why, axis=1)

# --------------------------------------------------------------------------- #
# Top-line callouts
# --------------------------------------------------------------------------- #

c1, c2, c3, c4 = st.columns(4)
c1.metric("LY promo events", f"{len(agg):,}")
c2.metric("Incremental sales", fmt_money(agg["incremental_sales"].sum()))
c3.metric("Incremental margin", fmt_money(agg["incremental_margin"].sum()))
c4.metric("Avg promo ROI", f"{agg['roi'].replace([np.inf, -np.inf], np.nan).dropna().mean():.2f}×")

# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #

c1, c2 = st.columns(2)
with c1:
    tac = (agg.groupby("promo_tactic", as_index=False)
              .agg(incremental_margin=("incremental_margin", "sum"),
                   incremental_units=("incremental_units", "sum"),
                   roi=("roi", "mean")))
    fig = px.bar(tac, x="promo_tactic", y="incremental_margin",
                 color="roi", color_continuous_scale="RdYlGn",
                 title="Promo tactic effectiveness — incremental margin (color = avg ROI)")
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)
with c2:
    fig = px.scatter(agg, x="discount_pct", y="incremental_units",
                     color="category", size="ly_sales", hover_data=["item_description"],
                     title="Discount depth vs incremental units (size = LY sales)")
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

c1, c2 = st.columns(2)
with c1:
    fund = (agg.assign(vendor_funded=(agg["ly_sales"] * 0.05))  # rough proxy
              .groupby("promo_tactic", as_index=False)
              .agg(vendor_funded=("vendor_funded", "sum"),
                   incremental_margin=("incremental_margin", "sum")))
    fund["funding_roi"] = fund["incremental_margin"] / fund["vendor_funded"].replace(0, np.nan)
    fig = px.bar(fund, x="promo_tactic", y="funding_roi",
                 title="Vendor funding ROI by tactic")
    fig.update_traces(marker_color=PALETTE["primary"])
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
with c2:
    top = agg.sort_values("effectiveness_score", ascending=False).head(8)
    bot = agg.sort_values("effectiveness_score").head(8)
    combined = pd.concat([top.assign(grp="Top"), bot.assign(grp="Bottom")])
    fig = px.bar(combined, x="effectiveness_score", y="item_description",
                 color="grp", orientation="h",
                 title="Top vs Bottom performing promos")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------- #
# Recommendation table
# --------------------------------------------------------------------------- #
st.subheader("LY promo recommendations")
disp = agg.copy()
disp["discount_pct"] = disp["discount_pct"].map(fmt_pct)
disp["ly_sales"] = disp["ly_sales"].map(fmt_money)
disp["ly_margin"] = disp["ly_margin"].map(fmt_money)
disp["incremental_sales"] = disp["incremental_sales"].map(fmt_money)
disp["incremental_margin"] = disp["incremental_margin"].map(fmt_money)
disp["roi"] = disp["roi"].map(lambda v: f"{v:.2f}×" if pd.notna(v) else "—")
disp["effectiveness_score"] = disp["effectiveness_score"].map(lambda v: f"{v:.2f}")

st.dataframe(disp[[
    "item_description", "category", "promo_tactic", "discount_pct",
    "ly_units", "ly_sales", "ly_margin",
    "incremental_units", "incremental_sales", "incremental_margin",
    "roi", "effectiveness_score", "recommendation", "reason",
]].sort_values("recommendation"),
    use_container_width=True, hide_index=True)
