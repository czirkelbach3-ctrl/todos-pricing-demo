"""
Retail Pricing Optimization Studio
==================================
Streamlit entrypoint. Launch with:

    streamlit run app.py

This file is the "home" page. The eight functional pages live under /pages
and are auto-discovered by Streamlit's multi-page convention. Sidebar
navigation appears automatically; we just style it.
"""
from __future__ import annotations

import streamlit as st

from src.data_loader import load_dataset, kaggle_available
from src.model import train_demand_model
from src.ui_components import page_setup, kpi_row, fmt_money, fmt_int, fmt_pct, PALETTE


# --------------------------------------------------------------------------- #
# Cached data + model
# --------------------------------------------------------------------------- #

@st.cache_data(show_spinner="Generating dataset...", ttl=60 * 60)
def get_data(mode: str):
    df, source = load_dataset(mode=mode)
    return df, source


@st.cache_resource(show_spinner="Training demand model...")
def get_model(_df):
    return train_demand_model(_df)


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #

page_setup("Retail Pricing Optimization Studio", icon=":shopping_bags:")

# Mode selector (Demo vs Kaggle) — sticky across pages via session_state
if "data_mode" not in st.session_state:
    st.session_state["data_mode"] = "auto"

mode_label = st.sidebar.radio(
    "Data source",
    options=["Auto (Kaggle if present, else Demo)", "Demo (synthetic)", "Kaggle only"],
    index=0,
    help="Demo Mode generates a realistic synthetic dataset; Kaggle Mode uses files in /data."
)
mode_map = {
    "Auto (Kaggle if present, else Demo)": "auto",
    "Demo (synthetic)": "synthetic",
    "Kaggle only": "kaggle",
}
st.session_state["data_mode"] = mode_map[mode_label]

if st.sidebar.button("Reset cache", use_container_width=True):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("Use the page navigator above to explore each module.")

try:
    df, source = get_data(st.session_state["data_mode"])
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

st.session_state["df"] = df
st.session_state["source"] = source

# --------------------------------------------------------------------------- #
# Hero
# --------------------------------------------------------------------------- #

c1, c2 = st.columns([3, 1])
with c1:
    st.markdown(
        f"""
        ### A merchandising decisioning workbench for pre-season pricing & promo

        Bring **forecast science**, **elasticity-aware scenario planning**,
        **category roll-ups**, and **vendor funding economics** into a single
        shareable surface — so merchants, planners, pricing, and leadership work
        from the same numbers.

        <span class="small-muted">Active data source:</span> **{source}**
        """,
        unsafe_allow_html=True,
    )
with c2:
    st.success("Tool is in **Demo** posture." if "Synthetic" in source
               else f"Connected: **{source}**")
    if not kaggle_available():
        st.caption("Add Kaggle files in `/data` to enable Kaggle Mode. See `data/README.md`.")

# --------------------------------------------------------------------------- #
# At-a-glance KPIs (TY only)
# --------------------------------------------------------------------------- #

ty = df[df.get("year_offset", 0) == 1] if "year_offset" in df.columns else df

cards = [
    {"label": "TY Forecast Sales", "value": fmt_money(ty["forecast_sales"].sum())},
    {"label": "TY Forecast Units", "value": fmt_int(ty["forecast_units"].sum())},
    {"label": "TY Forecast Margin", "value": fmt_money(ty["forecast_margin"].sum())},
    {"label": "Margin Rate",
     "value": fmt_pct(ty["forecast_margin"].sum() /
                      max(1.0, ty["forecast_sales"].sum()))},
    {"label": "Avg Confidence", "value": fmt_pct(ty["confidence_score"].mean())},
    {"label": "Planned Promos",
     "value": fmt_int((ty["promo_flag"] == 1).sum())},
]
kpi_row(cards)

st.markdown("---")

# --------------------------------------------------------------------------- #
# Page map
# --------------------------------------------------------------------------- #

st.subheader("Where to go next")
nav_cols = st.columns(2)
nav = [
    ("Executive Overview", "Leadership-ready KPIs, category roll-up, waterfall, and approval panel."),
    ("Pre-Season Forecasting Workbench", "Merchant workspace: filter, edit promo plan, see live impact."),
    ("Scenario Sandbox", "Five canonical scenarios, side-by-side, with tunable scoring weights."),
    ("Promo Effectiveness / LY Hindsight", "Which promos worked LY — keep, refine, drop, test."),
    ("Pricing Optimization Engine", "Elasticity-driven price/margin curve with guardrails."),
    ("Inventory & Sell-Through Risk", "Markdown vs. OOS risk classification by item."),
    ("Data Quality & Model Trust", "Forecast accuracy, confidence, feature importance, missing data."),
    ("Export Center", "Download scenario, leadership, item-week, and comparison artifacts."),
]
for i, (name, desc) in enumerate(nav):
    with nav_cols[i % 2]:
        st.markdown(f"**{i+1}. {name}**  \n{desc}")

st.markdown("---")
st.caption(
    "This is a prototype. Forecasts are model-driven but rely on synthetic merchandising "
    "attributes when running on public Kaggle data. Not a production decisioning system."
)
