"""
Reusable Streamlit UI helpers.

Goal: every page should feel like the same product. Centralizing KPI cards,
filters, callouts, and the color palette here keeps the demo cohesive.
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px


# Brand palette — calm, professional, no neon
PALETTE = {
    "primary": "#1F4E79",
    "accent":  "#E97132",
    "good":    "#2E7D32",
    "warn":    "#ED6C02",
    "bad":     "#C62828",
    "muted":   "#6B7280",
    "bg":      "#F5F7FA",
}


# --------------------------------------------------------------------------- #
# Page setup
# --------------------------------------------------------------------------- #

def page_setup(title: str, icon: str = ":chart_with_upwards_trend:") -> None:
    st.set_page_config(page_title=f"{title} | Retail Pricing Optimization Studio",
                       page_icon=icon, layout="wide", initial_sidebar_state="expanded")
    st.markdown(
        f"""
        <style>
        .kpi-card {{
            border: 1px solid #E5E7EB;
            border-radius: 10px;
            padding: 14px 16px;
            background: #FFFFFF;
            box-shadow: 0 1px 2px rgba(16,24,40,0.04);
        }}
        .kpi-label {{ color: {PALETTE['muted']}; font-size: 12px; text-transform: uppercase;
                     letter-spacing: .06em; margin-bottom: 4px; }}
        .kpi-value {{ font-size: 24px; font-weight: 600; color: {PALETTE['primary']}; }}
        .kpi-delta-pos {{ color: {PALETTE['good']}; font-size: 12px; }}
        .kpi-delta-neg {{ color: {PALETTE['bad']};  font-size: 12px; }}
        .badge {{
            display:inline-block; padding:2px 10px; border-radius:12px;
            font-size: 11px; font-weight: 600; letter-spacing: .03em;
        }}
        .badge-good {{ background:#E8F5E9; color:#2E7D32; }}
        .badge-warn {{ background:#FFF3E0; color:#E65100; }}
        .badge-bad  {{ background:#FFEBEE; color:#C62828; }}
        .small-muted {{ color:{PALETTE['muted']}; font-size: 12px; }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title(title)


# --------------------------------------------------------------------------- #
# KPI cards
# --------------------------------------------------------------------------- #

def kpi_card(label: str, value: str, delta: Optional[str] = None,
             delta_positive: bool = True, help_text: Optional[str] = None) -> None:
    delta_html = ""
    if delta is not None:
        cls = "kpi-delta-pos" if delta_positive else "kpi-delta-neg"
        delta_html = f'<div class="{cls}">{delta}</div>'
    tooltip = f' title="{help_text}"' if help_text else ""
    st.markdown(
        f"""
        <div class="kpi-card"{tooltip}>
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_row(cards: list[dict]) -> None:
    cols = st.columns(len(cards))
    for col, card in zip(cols, cards):
        with col:
            kpi_card(**card)


def fmt_money(x: float, abbreviate: bool = True) -> str:
    if x is None or pd.isna(x):
        return "—"
    if abbreviate:
        if abs(x) >= 1e9: return f"${x/1e9:,.2f}B"
        if abs(x) >= 1e6: return f"${x/1e6:,.2f}M"
        if abs(x) >= 1e3: return f"${x/1e3:,.1f}K"
    return f"${x:,.0f}"


def fmt_int(x: float) -> str:
    if x is None or pd.isna(x):
        return "—"
    if abs(x) >= 1e9: return f"{x/1e9:,.2f}B"
    if abs(x) >= 1e6: return f"{x/1e6:,.2f}M"
    if abs(x) >= 1e3: return f"{x/1e3:,.1f}K"
    return f"{x:,.0f}"


def fmt_pct(x: float, digits: int = 1) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{x*100:.{digits}f}%"


def badge(label: str, kind: str = "good") -> str:
    return f'<span class="badge badge-{kind}">{label}</span>'


# --------------------------------------------------------------------------- #
# Sidebar filters
# --------------------------------------------------------------------------- #

def sidebar_filters(df: pd.DataFrame, *,
                    show_store: bool = True,
                    show_item: bool = True,
                    show_week: bool = True) -> dict:
    """Render the standard set of filters and return the selections."""
    st.sidebar.markdown("### Filters")

    departments = sorted(df["department"].dropna().unique())
    dept = st.sidebar.multiselect("Department", departments, default=departments)

    cats_pool = df[df["department"].isin(dept)] if dept else df
    categories = sorted(cats_pool["category"].dropna().unique())
    cat = st.sidebar.multiselect("Category", categories, default=categories)

    vendors_pool = cats_pool[cats_pool["category"].isin(cat)] if cat else cats_pool
    vendors = sorted(vendors_pool["vendor"].dropna().unique())
    vendor = st.sidebar.multiselect("Vendor", vendors, default=vendors)

    store_sel: Iterable[int] | None = None
    if show_store:
        store_opts = sorted(df["store_id"].unique().tolist())
        scope = st.sidebar.radio("Store scope", ["Chainwide", "Store(s)"], horizontal=True)
        if scope == "Store(s)":
            store_sel = st.sidebar.multiselect("Stores", store_opts, default=store_opts[:3])
        else:
            store_sel = store_opts

    item_sel = None
    if show_item:
        items_pool = vendors_pool.copy()
        if vendor:
            items_pool = items_pool[items_pool["vendor"].isin(vendor)]
        item_opts = (items_pool[["item_id", "item_description"]]
                     .drop_duplicates().sort_values("item_description"))
        item_sel = st.sidebar.multiselect(
            "Item(s) (optional)",
            options=item_opts["item_id"].tolist(),
            format_func=lambda i: item_opts.set_index("item_id")["item_description"].get(i, str(i)),
        )

    week_range = None
    if show_week:
        wks = sorted(df["fiscal_week"].unique().tolist())
        if wks:
            week_range = st.sidebar.slider("Fiscal week range", min_value=min(wks),
                                           max_value=max(wks), value=(min(wks), max(wks)))

    return {
        "department": dept, "category": cat, "vendor": vendor,
        "store_id": list(store_sel) if store_sel is not None else None,
        "item_id": item_sel, "week_range": week_range,
    }


def apply_filters(df: pd.DataFrame, f: dict) -> pd.DataFrame:
    out = df
    if f.get("department"):
        out = out[out["department"].isin(f["department"])]
    if f.get("category"):
        out = out[out["category"].isin(f["category"])]
    if f.get("vendor"):
        out = out[out["vendor"].isin(f["vendor"])]
    if f.get("store_id"):
        out = out[out["store_id"].isin(f["store_id"])]
    if f.get("item_id"):
        out = out[out["item_id"].isin(f["item_id"])]
    if f.get("week_range"):
        lo, hi = f["week_range"]
        out = out[(out["fiscal_week"] >= lo) & (out["fiscal_week"] <= hi)]
    return out


# --------------------------------------------------------------------------- #
# Chart helpers
# --------------------------------------------------------------------------- #

def line_by_week(df: pd.DataFrame, value: str, title: str, color: str = PALETTE["primary"]) -> go.Figure:
    g = df.groupby("fiscal_week", as_index=False)[value].sum()
    fig = px.line(g, x="fiscal_week", y=value, title=title, markers=True)
    fig.update_traces(line_color=color, line_width=2.5)
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=320,
                      hovermode="x unified", showlegend=False)
    return fig


def bar_by_dim(df: pd.DataFrame, dim: str, value: str, title: str,
               color: str = PALETTE["primary"], horizontal: bool = False) -> go.Figure:
    g = df.groupby(dim, as_index=False)[value].sum().sort_values(value, ascending=horizontal)
    if horizontal:
        fig = px.bar(g, x=value, y=dim, orientation="h", title=title)
    else:
        fig = px.bar(g, x=dim, y=value, title=title)
    fig.update_traces(marker_color=color)
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=360, showlegend=False)
    return fig


def waterfall(steps: list[tuple[str, float]], title: str) -> go.Figure:
    """Simple Plotly waterfall."""
    labels = [s[0] for s in steps]
    values = [s[1] for s in steps]
    measures = ["absolute"] + ["relative"] * (len(steps) - 2) + ["total"]
    fig = go.Figure(go.Waterfall(
        x=labels, y=values, measure=measures,
        connector={"line": {"color": "#9CA3AF"}},
        increasing={"marker": {"color": PALETTE["good"]}},
        decreasing={"marker": {"color": PALETTE["bad"]}},
        totals={"marker": {"color": PALETTE["primary"]}},
    ))
    fig.update_layout(title=title, margin=dict(l=10, r=10, t=40, b=10), height=360)
    return fig


def callout_methodology(label: str, body: str) -> None:
    with st.expander(f"Methodology — {label}"):
        st.markdown(body)
