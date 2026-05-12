"""Executive Overview — leadership-ready view of TY plan vs LY."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_loader import load_dataset
from src.ui_components import (
    page_setup, kpi_row, fmt_money, fmt_int, fmt_pct, badge,
    sidebar_filters, apply_filters, line_by_week, waterfall, PALETTE,
)
from src.scenario_engine import build_scenarios, summarize_scenario
from src.optimization import Guardrails


page_setup("Executive Overview", icon=":crown:")

# Bootstrap data if a user lands here directly
if "df" not in st.session_state:
    df, source = load_dataset(mode=st.session_state.get("data_mode", "auto"))
    st.session_state["df"] = df
    st.session_state["source"] = source
df = st.session_state["df"]

filters = sidebar_filters(df, show_store=True, show_item=False)
view = apply_filters(df, filters)
ly = view[view["year_offset"] == 0]
ty = view[view["year_offset"] == 1]

# Compose scope label (kept out of the f-string to stay 3.10-compatible)
_store_ids = filters.get("store_id")
if _store_ids is None or len(_store_ids) == df["store_id"].nunique():
    _store_scope = "chainwide"
else:
    _store_scope = f"{len(_store_ids)} stores"
_n_dept = len(filters["department"]) or "all"
_n_cat = len(filters["category"]) or "all"
_src = st.session_state.get("source", "—")
st.caption(
    f"Scope: **{_n_dept} departments**, **{_n_cat} categories**, **{_store_scope}**. "
    f"Data source: **{_src}**."
)

# --------------------------------------------------------------------------- #
# KPI strip
# --------------------------------------------------------------------------- #

forecast_sales = ty["forecast_sales"].sum()
forecast_units = ty["forecast_units"].sum()
forecast_margin = ty["forecast_margin"].sum()
ly_sales = ly["sales"].sum()
ly_margin = ly["gross_margin"].sum()
incr_sales = forecast_sales - ly_sales
incr_margin = forecast_margin - ly_margin
confidence = ty["confidence_score"].mean() if len(ty) else 0.0
n_promos = int((ty["promo_flag"] == 1).sum())

kpi_row([
    {"label": "TY Forecast Sales", "value": fmt_money(forecast_sales),
     "delta": f"{fmt_pct(incr_sales / max(ly_sales,1))} vs LY",
     "delta_positive": incr_sales >= 0},
    {"label": "TY Forecast Units", "value": fmt_int(forecast_units)},
    {"label": "TY Forecast Margin", "value": fmt_money(forecast_margin),
     "delta": f"{fmt_pct(incr_margin / max(ly_margin,1))} vs LY",
     "delta_positive": incr_margin >= 0},
    {"label": "Margin Rate",
     "value": fmt_pct(forecast_margin / max(forecast_sales, 1))},
])
kpi_row([
    {"label": "Incremental Sales vs LY", "value": fmt_money(incr_sales),
     "delta_positive": incr_sales >= 0},
    {"label": "Incremental Margin vs LY", "value": fmt_money(incr_margin),
     "delta_positive": incr_margin >= 0},
    {"label": "Forecast Confidence", "value": fmt_pct(confidence)},
    {"label": "# Planned Promos", "value": fmt_int(n_promos)},
])

st.markdown("---")

# --------------------------------------------------------------------------- #
# Category roll-up
# --------------------------------------------------------------------------- #

st.subheader("Category roll-up")
roll = (ty.groupby(["department", "category"], as_index=False)
          .agg(forecast_sales=("forecast_sales", "sum"),
               forecast_units=("forecast_units", "sum"),
               forecast_margin=("forecast_margin", "sum"),
               confidence=("confidence_score", "mean"),
               planned_promos=("promo_flag", "sum")))
roll["margin_rate"] = roll["forecast_margin"] / roll["forecast_sales"].replace(0, np.nan)
ly_roll = (ly.groupby(["department", "category"], as_index=False)
             .agg(ly_sales=("sales", "sum"), ly_margin=("gross_margin", "sum")))
roll = roll.merge(ly_roll, on=["department", "category"], how="left").fillna(0)
roll["sales_vs_ly_pct"] = (roll["forecast_sales"] - roll["ly_sales"]) / roll["ly_sales"].replace(0, np.nan)
roll["margin_vs_ly_pct"] = (roll["forecast_margin"] - roll["ly_margin"]) / roll["ly_margin"].replace(0, np.nan)

st.dataframe(
    roll.assign(
        forecast_sales=lambda d: d["forecast_sales"].map(lambda v: fmt_money(v)),
        forecast_margin=lambda d: d["forecast_margin"].map(lambda v: fmt_money(v)),
        forecast_units=lambda d: d["forecast_units"].map(lambda v: fmt_int(v)),
        margin_rate=lambda d: d["margin_rate"].map(lambda v: fmt_pct(v)),
        sales_vs_ly_pct=lambda d: d["sales_vs_ly_pct"].map(lambda v: fmt_pct(v)),
        margin_vs_ly_pct=lambda d: d["margin_vs_ly_pct"].map(lambda v: fmt_pct(v)),
        confidence=lambda d: d["confidence"].map(lambda v: fmt_pct(v)),
    )[[
        "department", "category", "forecast_sales", "forecast_units",
        "forecast_margin", "margin_rate", "sales_vs_ly_pct", "margin_vs_ly_pct",
        "confidence", "planned_promos",
    ]],
    use_container_width=True, hide_index=True,
)

st.markdown("---")

# --------------------------------------------------------------------------- #
# Waterfall: LY actual → baseline → current plan → optimized
# --------------------------------------------------------------------------- #

st.subheader("Sales & margin waterfall")

@st.cache_data(show_spinner="Computing scenarios...", ttl=60 * 30)
def _scenarios_cached(panel: pd.DataFrame):
    return build_scenarios(panel, Guardrails())

scenarios = _scenarios_cached(ty)
summaries = [summarize_scenario(name, s, scenarios["Baseline / No Promo"])
             for name, s in scenarios.items()]
summary_df = pd.DataFrame(summaries)

c1, c2 = st.columns(2)
with c1:
    ly_actual = ly["sales"].sum()
    base = summary_df.loc[summary_df["scenario"] == "Baseline / No Promo", "forecast_sales"].iloc[0]
    plan = summary_df.loc[summary_df["scenario"] == "Current Plan", "forecast_sales"].iloc[0]
    opt = summary_df.loc[summary_df["scenario"] == "Optimized Price", "forecast_sales"].iloc[0]
    fig = waterfall([
        ("LY Actual", ly_actual),
        ("Δ to Baseline TY", base - ly_actual),
        ("Δ from Promo Plan", plan - base),
        ("Δ from Optimization", opt - plan),
        ("Optimized Plan", 0),
    ], title="Sales waterfall ($)")
    st.plotly_chart(fig, use_container_width=True)
with c2:
    ly_margin_a = ly["gross_margin"].sum()
    base_m = summary_df.loc[summary_df["scenario"] == "Baseline / No Promo", "forecast_margin"].iloc[0]
    plan_m = summary_df.loc[summary_df["scenario"] == "Current Plan", "forecast_margin"].iloc[0]
    opt_m = summary_df.loc[summary_df["scenario"] == "Optimized Price", "adjusted_margin"].iloc[0]
    fig = waterfall([
        ("LY Actual", ly_margin_a),
        ("Δ to Baseline TY", base_m - ly_margin_a),
        ("Δ from Promo Plan", plan_m - base_m),
        ("Δ from Optimization", opt_m - plan_m),
        ("Optimized Plan", 0),
    ], title="Margin waterfall ($, incl. vendor funding)")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# --------------------------------------------------------------------------- #
# Time-series + category scatter
# --------------------------------------------------------------------------- #

c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(line_by_week(ty, "forecast_sales", "Forecast sales by week",
                                  color=PALETTE["primary"]),
                    use_container_width=True)
with c2:
    st.plotly_chart(line_by_week(ty, "forecast_margin", "Forecast margin by week",
                                  color=PALETTE["accent"]),
                    use_container_width=True)

scat = (ty.groupby("category", as_index=False)
          .agg(forecast_sales=("forecast_sales", "sum"),
               forecast_margin=("forecast_margin", "sum"),
               forecast_units=("forecast_units", "sum")))
scat["margin_rate"] = scat["forecast_margin"] / scat["forecast_sales"].replace(0, np.nan)
fig = px.scatter(scat, x="forecast_sales", y="margin_rate", size="forecast_units",
                 color="category", text="category",
                 title="Category performance — sales vs margin rate (bubble = units)")
fig.update_traces(textposition="top center")
fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# --------------------------------------------------------------------------- #
# Leadership approval panel
# --------------------------------------------------------------------------- #

st.subheader("Leadership approval panel")

recommended = summary_df.sort_values("adjusted_margin", ascending=False).iloc[0]
risks = []
if confidence < 0.70:
    risks.append(f"{badge('Confidence', 'warn')} Plan-wide confidence is {fmt_pct(confidence)} — review low-history items.")
items_low_conf = (ty.groupby(["item_id", "item_description"], as_index=False)
                    ["confidence_score"].mean()
                    .query("confidence_score < 0.55")
                    .sort_values("confidence_score"))
if len(items_low_conf):
    risks.append(f"{badge(str(len(items_low_conf)) + ' items < 55% conf', 'bad')} need merchant review.")
vendor_exposure = ty["vendor_funding_pct"].mean() * ty["forecast_sales"].sum()
if vendor_exposure > 0:
    risks.append(f"{badge('Vendor funding', 'good')} Estimated vendor co-op exposure: {fmt_money(vendor_exposure)}.")

c1, c2 = st.columns([1, 1])
with c1:
    st.markdown(f"**Recommended plan:** {recommended['scenario']}")
    st.write(f"• Forecast sales: **{fmt_money(recommended['forecast_sales'])}**")
    st.write(f"• Adjusted margin: **{fmt_money(recommended['adjusted_margin'])}**")
    st.write(f"• Margin rate: **{fmt_pct(recommended['margin_rate'])}**")
    st.write(f"• Forecast confidence: **{fmt_pct(recommended['avg_confidence'])}**")
with c2:
    st.markdown("**Key risks & callouts**")
    if not risks:
        st.success("No material risks flagged at the plan level.")
    for r in risks:
        st.markdown(r, unsafe_allow_html=True)
    if len(items_low_conf):
        st.dataframe(items_low_conf.head(10), use_container_width=True, hide_index=True)
