"""Standalone Macro & Sector Analysis page."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd

from dotenv import load_dotenv
load_dotenv()

from macro_analysis import MacroAnalyzer
from sector_analysis import SectorAnalyzer

# Re-use chart functions from app module (avoids duplication)
from app import (
    create_cycle_diagram,
    create_macro_dashboard,
    create_yield_curve_chart,
    create_sector_heatmap,
    create_sector_rotation_chart,
    run_macro_analysis,
    run_sector_analysis,
)

st.set_page_config(
    page_title="Macro Analysis — Financial Analyst",
    page_icon="🌍",
    layout="wide",
)

st.markdown("# 🌍 Macroeconomic & Sector Analysis")
st.caption("Global macro conditions and sector rotation — updated every 30 minutes.")

with st.spinner("Loading macro data…"):
    macro_result = run_macro_analysis()

if not macro_result or not macro_result.cycle:
    st.warning("Macro data unavailable — Yahoo Finance may be rate-limiting. Try again shortly.")
    st.stop()

cycle = macro_result.cycle

# ---------------------------------------------------------------------------
# Business cycle
# ---------------------------------------------------------------------------
st.markdown(create_cycle_diagram(cycle.phase), unsafe_allow_html=True)

phase_colors = {
    "Expansion": "#00C853",
    "Peak": "#FFA726",
    "Contraction": "#FF1744",
    "Recovery": "#42A5F5",
}
pc = phase_colors.get(cycle.phase, "#888")
st.markdown(
    f"""
    <div style="background:linear-gradient(135deg, {pc}22, {pc}11);
                border:1px solid {pc}66;border-radius:12px;padding:1.2rem;margin:0.5rem 0 1rem 0">
        <h3 style="margin:0;color:{pc}">{cycle.phase} Phase
            <span style="font-size:0.85rem;color:#aaa;margin-left:0.5rem">
            (Confidence: {cycle.confidence}) — {cycle.risk_posture}</span></h3>
        <p style="color:#ccc;margin:0.5rem 0 0 0">{cycle.description}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

fc1, fc2 = st.columns(2)
with fc1:
    st.markdown("**Favored Sectors**")
    for s in cycle.favored_sectors:
        st.markdown(f"<span style='color:#69F0AE'>● {s}</span>", unsafe_allow_html=True)
with fc2:
    st.markdown("**Sectors to Underweight**")
    for s in cycle.avoid_sectors:
        st.markdown(f"<span style='color:#FF8A65'>● {s}</span>", unsafe_allow_html=True)

st.divider()

# ---------------------------------------------------------------------------
# Macro indicators
# ---------------------------------------------------------------------------
st.markdown("### Key Macro Indicators")
st.plotly_chart(create_macro_dashboard(macro_result.indicators), use_container_width=True)

yc_fig = create_yield_curve_chart(macro_result.yield_curve)
if yc_fig:
    st.plotly_chart(yc_fig, use_container_width=True)

st.markdown("#### Indicator Details")
for ind in macro_result.indicators:
    sig_color = {"bullish": "#00C853", "neutral": "#FFD54F", "bearish": "#FF1744"}[ind.signal]
    chg_1m_str = f"{ind.change_1m:+.1%}" if ind.change_1m is not None else "—"
    chg_3m_str = f"{ind.change_3m:+.1%}" if ind.change_3m is not None else "—"
    val_str = f"{ind.value:,.2f}" if ind.value is not None else "N/A"
    st.markdown(
        f"<div style='padding:0.4rem 0.8rem;background:rgba(28,31,48,0.6);"
        f"border-left:3px solid {sig_color};border-radius:0 8px 8px 0;margin-bottom:0.3rem'>"
        f"<span style='color:{sig_color};font-weight:700'>{ind.name}</span> "
        f"<span style='color:white'>{val_str}</span> "
        f"<span style='color:#888'>(1M: {chg_1m_str}, 3M: {chg_3m_str})</span> — "
        f"<span style='color:#aaa'>{ind.interpretation}</span></div>",
        unsafe_allow_html=True,
    )

st.divider()

# ---------------------------------------------------------------------------
# Sector rotation
# ---------------------------------------------------------------------------
st.markdown("### Sector Rotation Analysis")

rates_rising = any(
    ind.name == "10-Year Treasury Yield" and (ind.change_1m or 0) > 0.02
    for ind in macro_result.indicators
)

with st.spinner("Loading sector data…"):
    sector_result = run_sector_analysis(cycle.phase, rates_rising)

if sector_result and sector_result.sectors:
    col_heat, col_rot = st.columns(2)
    heatmap_fig = create_sector_heatmap(sector_result.sectors)
    if heatmap_fig:
        col_heat.plotly_chart(heatmap_fig, use_container_width=True)
    rotation_fig = create_sector_rotation_chart(sector_result.sectors, cycle.phase)
    if rotation_fig:
        col_rot.plotly_chart(rotation_fig, use_container_width=True)

    st.markdown("#### Rotation Recommendation")
    st.markdown(sector_result.rotation_recommendation)

    st.markdown("#### All Sectors Ranked")
    sect_rows = [
        {
            "Rank": i,
            "Sector": s.name,
            "ETF": s.etf,
            "1W": f"{(s.change_1w or 0) * 100:+.1f}%",
            "1M": f"{(s.change_1m or 0) * 100:+.1f}%",
            "3M": f"{(s.change_3m or 0) * 100:+.1f}%",
            "6M": f"{(s.change_6m or 0) * 100:+.1f}%",
            "Rel Strength": f"{s.relative_strength:+.1f}%",
            "Momentum": f"{s.momentum_score:+.3f}",
            "Cycle Fit": f"{s.cycle_alignment:.2f}",
            "Rate Sens.": s.rate_sensitivity.title(),
        }
        for i, s in enumerate(sector_result.sectors, 1)
    ]
    st.dataframe(pd.DataFrame(sect_rows), use_container_width=True, hide_index=True)
else:
    st.info("Sector data unavailable — may be rate limited. Try again shortly.")
