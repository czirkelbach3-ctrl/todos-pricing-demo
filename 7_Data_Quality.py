"""Data Quality & Model Trust — model performance, confidence, explainability."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import load_dataset
from src.model import train_demand_model
from src.ui_components import (
    page_setup, kpi_row, fmt_money, fmt_int, fmt_pct, PALETTE,
)


page_setup("Data Quality & Model Trust", icon=":microscope:")

if "df" not in st.session_state:
    df, source = load_dataset(mode=st.session_state.get("data_mode", "auto"))
    st.session_state["df"] = df
    st.session_state["source"] = source
df = st.session_state["df"]


@st.cache_resource(show_spinner="Training demand model...")
def _model_cached(_df):
    return train_demand_model(_df)


model = _model_cached(df)

# --------------------------------------------------------------------------- #
# Top model metrics
# --------------------------------------------------------------------------- #
m = model.test_metrics
kpi_row([
    {"label": "MAE",  "value": f"{m['MAE']:.2f} units"},
    {"label": "RMSE", "value": f"{m['RMSE']:.2f} units"},
    {"label": "MAPE", "value": fmt_pct(m['MAPE'])},
    {"label": "R²",   "value": f"{m['R2']:.3f}"},
])
st.caption("Holdout = last 12 weeks (time-based split). Tree-based regressor, no leakage from same-row promo data.")

# --------------------------------------------------------------------------- #
# Confidence distribution
# --------------------------------------------------------------------------- #
c1, c2 = st.columns(2)
with c1:
    fig = px.histogram(df, x="confidence_score", nbins=30,
                       title="Confidence score distribution",
                       color_discrete_sequence=[PALETTE["primary"]])
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)
with c2:
    fi = model.feature_importance
    fig = px.bar(fi, x="importance", y="feature", orientation="h",
                 title="Feature importance",
                 color_discrete_sequence=[PALETTE["accent"]])
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=10),
                      yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------- #
# Actual vs predicted (holdout)
# --------------------------------------------------------------------------- #
pred = model.test_predictions
agg = pred.groupby("date", as_index=False).agg(actual=("actual", "sum"),
                                                predicted=("predicted", "sum"))
fig = go.Figure()
fig.add_trace(go.Scatter(x=agg["date"], y=agg["actual"], name="Actual",
                         line=dict(color=PALETTE["primary"])))
fig.add_trace(go.Scatter(x=agg["date"], y=agg["predicted"], name="Predicted",
                         line=dict(color=PALETTE["accent"], dash="dash")))
fig.update_layout(title="Actual vs predicted (holdout, summed by week)",
                  height=340, margin=dict(l=10, r=10, t=40, b=10))
st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------- #
# Missing data + outliers
# --------------------------------------------------------------------------- #
st.subheader("Missing & outlier diagnostics")
c1, c2 = st.columns(2)
with c1:
    miss = df.isna().sum().to_frame("missing").reset_index().rename(columns={"index": "column"})
    miss["missing_pct"] = miss["missing"] / len(df)
    miss = miss[miss["missing"] > 0].sort_values("missing", ascending=False)
    if miss.empty:
        st.success("No missing values in working dataset.")
    else:
        st.dataframe(miss.assign(missing_pct=miss["missing_pct"].map(fmt_pct)),
                     use_container_width=True, hide_index=True)
with c2:
    # Simple outlier detection: |z| > 4 on units within category
    grp = df.groupby("category")["units"]
    z = (df["units"] - grp.transform("mean")) / grp.transform("std").replace(0, np.nan)
    outliers = df[(z.abs() > 4)][["item_description", "category", "fiscal_week", "units", "promo_flag"]]
    st.caption(f"{len(outliers):,} outlier rows (|z| > 4 on units within category).")
    st.dataframe(outliers.head(15), use_container_width=True, hide_index=True)

# --------------------------------------------------------------------------- #
# Explainability panel
# --------------------------------------------------------------------------- #
st.subheader("Explainability")
top_feats = model.feature_importance.head(5)["feature"].tolist()
low_conf_items = (df.groupby(["item_id", "item_description"], as_index=False)
                    ["confidence_score"].mean()
                    .sort_values("confidence_score").head(10))

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Top drivers of forecast (RF importance)**")
    st.write(", ".join(top_feats))
    st.markdown(
        "**Elasticity source:** category prior, perturbed per item. Override in "
        "`src/synthetic_data.py::CATEGORY_ELASTICITY` or wire to a learned elasticity model."
    )
    st.markdown(
        "**Similar-item fallback:** items with < 12 weeks of history reuse the "
        "category-level mean baseline (built into baseline rolling logic)."
    )
with c2:
    st.markdown("**Items with lowest forecast confidence**")
    low_conf_items["confidence_score"] = low_conf_items["confidence_score"].map(fmt_pct)
    st.dataframe(low_conf_items, use_container_width=True, hide_index=True)
    st.warning("Low-confidence forecasts should not auto-execute. Flag for merchant review.")
