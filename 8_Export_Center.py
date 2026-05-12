"""Export Center — download artifacts for downstream operational use."""
from __future__ import annotations

import io
from datetime import datetime

import pandas as pd
import streamlit as st

from src.data_loader import load_dataset
from src.scenario_engine import build_scenarios, summarize_scenario, score_scenarios, ScenarioWeights
from src.optimization import Guardrails
from src.ui_components import page_setup, fmt_money, fmt_pct, fmt_int


page_setup("Export Center", icon=":outbox_tray:")

if "df" not in st.session_state:
    df, source = load_dataset(mode=st.session_state.get("data_mode", "auto"))
    st.session_state["df"] = df
    st.session_state["source"] = source
df = st.session_state["df"]
ty = df[df["year_offset"] == 1]


def _to_csv(d: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    d.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


# --------------------------------------------------------------------------- #
# Build scenarios for export
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Preparing scenarios...", ttl=60 * 30)
def _scenarios_cached(panel: pd.DataFrame):
    return build_scenarios(panel, Guardrails())


scenarios = _scenarios_cached(ty)
summaries = [summarize_scenario(name, s, scenarios["Baseline / No Promo"])
             for name, s in scenarios.items()]
ranked = score_scenarios(summaries, ScenarioWeights())

st.markdown("Download operational and leadership-ready artifacts.")

# --------------------------------------------------------------------------- #
# 1. Selected scenario
# --------------------------------------------------------------------------- #
st.subheader("1. Selected scenario CSV")
which = st.selectbox("Scenario", list(scenarios.keys()))
sel = scenarios[which]
st.download_button(
    f"Download {which} (item-store-week)",
    data=_to_csv(sel),
    file_name=f"scenario_{which.lower().replace(' / ','_').replace(' ','_')}.csv",
    mime="text/csv",
    use_container_width=True,
)
st.dataframe(sel.head(10), use_container_width=True, hide_index=True)

st.markdown("---")

# --------------------------------------------------------------------------- #
# 2. Leadership summary CSV
# --------------------------------------------------------------------------- #
st.subheader("2. Leadership summary CSV")
leadership = ranked[[
    "scenario", "rank", "forecast_sales", "forecast_units", "forecast_margin",
    "adjusted_margin", "margin_rate", "incremental_sales", "incremental_margin",
    "avg_sell_through", "avg_confidence", "vendor_funding", "score",
]]
st.dataframe(leadership, use_container_width=True, hide_index=True)
st.download_button("Download leadership summary CSV",
                   data=_to_csv(leadership),
                   file_name="leadership_summary.csv",
                   mime="text/csv", use_container_width=True)

st.markdown("---")

# --------------------------------------------------------------------------- #
# 3. Item-week forecast
# --------------------------------------------------------------------------- #
st.subheader("3. Item × week forecast")
iw = (ty.groupby(["item_id", "item_description", "category", "department",
                  "vendor", "fiscal_week"], as_index=False)
        .agg(regular_price=("regular_price", "mean"),
             promo_price=("promo_price", "mean"),
             discount_pct=("discount_pct", "mean"),
             promo_tactic=("promo_tactic", lambda s: s.mode().iloc[0] if not s.mode().empty else "No Promo"),
             baseline_units=("baseline_units", "sum"),
             forecast_units=("forecast_units", "sum"),
             forecast_sales=("forecast_sales", "sum"),
             forecast_margin=("forecast_margin", "sum"),
             confidence_score=("confidence_score", "mean")))
st.dataframe(iw.head(15), use_container_width=True, hide_index=True)
st.download_button("Download item-week forecast CSV", data=_to_csv(iw),
                   file_name="item_week_forecast.csv", mime="text/csv",
                   use_container_width=True)

st.markdown("---")

# --------------------------------------------------------------------------- #
# 4. Scenario comparison
# --------------------------------------------------------------------------- #
st.subheader("4. Scenario comparison CSV")
comp = ranked.copy()
st.download_button("Download scenario comparison CSV", data=_to_csv(comp),
                   file_name="scenario_comparison.csv", mime="text/csv",
                   use_container_width=True)

st.markdown("---")

# --------------------------------------------------------------------------- #
# 5. Leadership summary (HTML / markdown)
# --------------------------------------------------------------------------- #
st.subheader("5. Leadership summary (HTML — paste into deck/email)")
best = ranked.iloc[0]
generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
html = f"""
<h2>TY Pre-Season Pricing & Promo Plan — Leadership Brief</h2>
<p><em>Generated {generated_at}. Source: {st.session_state.get('source', '—')}</em></p>
<h3>Recommended scenario: {best['scenario']}</h3>
<ul>
  <li><b>Forecast sales:</b> {fmt_money(best['forecast_sales'])}</li>
  <li><b>Forecast units:</b> {fmt_int(best['forecast_units'])}</li>
  <li><b>Forecast margin:</b> {fmt_money(best['forecast_margin'])}</li>
  <li><b>Adjusted margin (incl. vendor funding):</b> {fmt_money(best['adjusted_margin'])}</li>
  <li><b>Margin rate:</b> {fmt_pct(best['margin_rate'])}</li>
  <li><b>Incremental sales vs baseline:</b> {fmt_money(best['incremental_sales'])}</li>
  <li><b>Incremental margin vs baseline:</b> {fmt_money(best['incremental_margin'])}</li>
  <li><b>Average projected sell-through:</b> {fmt_pct(best['avg_sell_through'])}</li>
  <li><b>Forecast confidence:</b> {fmt_pct(best['avg_confidence'])}</li>
</ul>
<h3>Scenario ranking</h3>
<table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse;">
  <tr><th>Rank</th><th>Scenario</th><th>Adj. Margin</th><th>Incr. Sales</th><th>Confidence</th></tr>
  {''.join(
    f"<tr><td>{r['rank']}</td><td>{r['scenario']}</td><td>{fmt_money(r['adjusted_margin'])}</td>"
    f"<td>{fmt_money(r['incremental_sales'])}</td><td>{fmt_pct(r['avg_confidence'])}</td></tr>"
    for _, r in ranked.iterrows()
  )}
</table>
<p><i>Prototype — not for production decisioning.</i></p>
"""
st.code(html, language="html")
st.download_button("Download leadership_brief.html", data=html.encode("utf-8"),
                   file_name="leadership_brief.html", mime="text/html",
                   use_container_width=True)
