"""Pricing Optimization Engine — elasticity-driven price/margin curves."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import load_dataset
from src.ui_components import (
    page_setup, kpi_row, fmt_money, fmt_int, fmt_pct, badge, PALETTE,
)
from src.optimization import (
    Guardrails, price_response_curve, find_optimal_price,
)


page_setup("Pricing Optimization Engine", icon=":dollar:")

if "df" not in st.session_state:
    df, source = load_dataset(mode=st.session_state.get("data_mode", "auto"))
    st.session_state["df"] = df
    st.session_state["source"] = source
df = st.session_state["df"]

ty = df[df["year_offset"] == 1]

# Choose item
st.markdown(
    "Sweep candidate prices for a single item, see the demand/revenue/margin "
    "response curve, and the **adjusted-margin-optimal price** under the chosen guardrails."
)

items = (ty[["item_id", "item_description", "category", "regular_price", "cost",
             "elasticity", "vendor_funding_pct", "baseline_units"]]
         .groupby(["item_id", "item_description", "category"], as_index=False)
         .agg(regular_price=("regular_price", "mean"),
              cost=("cost", "mean"),
              elasticity=("elasticity", "mean"),
              vendor_funding_pct=("vendor_funding_pct", "mean"),
              baseline_units=("baseline_units", "mean")))
items = items.sort_values("item_description")

cat_filter = st.sidebar.multiselect("Category", sorted(items["category"].unique()),
                                    default=sorted(items["category"].unique()))
items = items[items["category"].isin(cat_filter)]
item_id = st.sidebar.selectbox("Item",
                               options=items["item_id"].tolist(),
                               format_func=lambda i: items.set_index("item_id")["item_description"].get(i, str(i)))

# Guardrails
st.sidebar.markdown("### Guardrails")
allow_above = st.sidebar.checkbox("Allow price > regular", value=False)
allow_below = st.sidebar.checkbox("Allow price < cost", value=False)
allow_neg = st.sidebar.checkbox("Allow negative margin (vendor-offset)", value=False)
max_disc = st.sidebar.slider("Max discount %", 0.0, 0.60, 0.40, 0.05)

# Item-level historical observed discounts (for confidence)
hist = df[(df["item_id"] == item_id) & (df["promo_flag"] == 1)]["discount_pct"]
obs_mean = float(hist.mean()) if len(hist) else 0.10
obs_std = float(hist.std()) if len(hist) > 1 else 0.08

guardrails = Guardrails(
    allow_above_regular=allow_above, allow_below_cost=allow_below,
    allow_negative_margin=allow_neg, max_discount_pct=max_disc,
    observed_discount_mean=obs_mean, observed_discount_std=max(0.04, obs_std),
)

# Item summary
row = items[items["item_id"] == item_id].iloc[0]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Regular price", f"${row['regular_price']:.2f}")
c2.metric("Cost", f"${row['cost']:.2f}")
c3.metric("Elasticity", f"{row['elasticity']:.2f}")
c4.metric("Vendor funding", fmt_pct(row["vendor_funding_pct"]))

# Compute curve + recommendation
curve = price_response_curve(row["baseline_units"], row["regular_price"], row["cost"],
                             row["elasticity"], row["vendor_funding_pct"], n=61,
                             guardrails=guardrails)
rec = find_optimal_price(row["baseline_units"], row["regular_price"], row["cost"],
                         row["elasticity"], row["vendor_funding_pct"], guardrails)

# Curves
fig = go.Figure()
fig.add_trace(go.Scatter(x=curve["price"], y=curve["units"], name="Units (demand)",
                         line=dict(color=PALETTE["primary"]), yaxis="y1"))
fig.add_trace(go.Scatter(x=curve["price"], y=curve["revenue"], name="Revenue",
                         line=dict(color=PALETTE["accent"], dash="dot"), yaxis="y2"))
fig.add_trace(go.Scatter(x=curve["price"], y=curve["adjusted_margin"], name="Adjusted margin",
                         line=dict(color=PALETTE["good"]), yaxis="y2"))
fig.add_vline(x=rec["price"], line_dash="dash", line_color="#374151",
              annotation_text=f"Optimal ${rec['price']:.2f}",
              annotation_position="top")
fig.add_vline(x=row["regular_price"], line_dash="dot", line_color=PALETTE["muted"],
              annotation_text="Regular", annotation_position="bottom")
fig.update_layout(
    title="Price response — demand, revenue, adjusted margin",
    height=460, margin=dict(l=10, r=10, t=40, b=10),
    yaxis=dict(title="Units", side="left"),
    yaxis2=dict(title="Dollars", overlaying="y", side="right"),
    legend=dict(orientation="h", yanchor="bottom", y=1.05, x=0),
)
st.plotly_chart(fig, use_container_width=True)

# Recommendation cards
st.subheader("Recommendation")
kpi_row([
    {"label": "Recommended price", "value": f"${rec['price']:.2f}"},
    {"label": "Recommended discount", "value": fmt_pct(rec["discount_pct"])},
    {"label": "Expected units", "value": fmt_int(rec["expected_units"])},
    {"label": "Expected sales", "value": fmt_money(rec["expected_revenue"])},
    {"label": "Expected margin", "value": fmt_money(rec["expected_margin"])},
    {"label": "Adjusted margin", "value": fmt_money(rec["adjusted_margin"])},
    {"label": "Confidence", "value": fmt_pct(rec["confidence_score"])},
])
if rec["risks"]:
    st.warning("**Risk flags:** " + ", ".join(rec["risks"]))
else:
    st.success("No risk flags triggered under current guardrails.")

with st.expander("How this is computed"):
    st.markdown(
        f"""
        - **Demand model:** constant-elasticity form
          $units(p) = baseline \\cdot (p / regular)^{{\\varepsilon}}$  with $\\varepsilon = {row['elasticity']:.2f}$.
        - **Objective:** maximize *adjusted margin* = gross margin + vendor funding $.
        - **Guardrails enforced:** max discount {max_disc:.0%}, below-cost {'allowed' if allow_below else 'blocked'},
          negative margin {'allowed' if allow_neg else 'blocked'}.
        - **Confidence:** decays as the planned discount diverges from observed history
          (mean {obs_mean:.0%}, std {max(0.04, obs_std):.0%}).
        """
    )
