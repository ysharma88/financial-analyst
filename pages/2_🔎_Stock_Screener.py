"""Standalone Stock Screener page."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from dotenv import load_dotenv
load_dotenv()

from sector_analysis import SECTOR_ETFS
from app import run_stock_screener, score_to_color

st.set_page_config(
    page_title="Stock Screener — Financial Analyst",
    page_icon="🔎",
    layout="wide",
)

st.markdown("# 🔎 Stock Screener")
st.caption("Find quality stocks across any sector using fundamental filters.")

# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------
sectors = sorted(SECTOR_ETFS.keys())

col1, col2, col3, col4, col5, col6 = st.columns(6)
sector = col1.selectbox("Sector", sectors, index=sectors.index("Technology") if "Technology" in sectors else 0)
max_stocks = col2.number_input("Max Stocks", 10, 50, 20, 5)
max_pe = col3.number_input("Max P/E", 0.0, 200.0, 50.0, 5.0, help="0 = no filter")
max_de = col4.number_input("Max Debt/Equity", 0.0, 500.0, 200.0, 25.0, help="0 = no filter")
min_roe = col5.number_input("Min ROE %", 0.0, 100.0, 0.0, 5.0, help="0 = no filter")
min_rg = col6.number_input("Min Rev Growth %", -50.0, 200.0, 0.0, 5.0, help="0 = no filter")

run_btn = st.button("🔎 Run Screener", type="primary")

if not run_btn:
    st.info("Select filters above and click **Run Screener** to find top stocks in a sector.")
    st.stop()

with st.spinner(f"Screening {sector} stocks… this may take up to a minute."):
    scr_result = run_stock_screener(
        sector,
        int(max_stocks),
        0,
        max_pe if max_pe > 0 else None,
        max_de if max_de > 0 else None,
        min_roe if min_roe > 0 else None,
        min_rg if min_rg != 0 else None,
    )

if not scr_result or not scr_result.stocks:
    st.warning("No stocks matched the filters. Try relaxing the criteria.")
    st.stop()

st.success(f"Found **{len(scr_result.stocks)}** stocks (from {scr_result.total_found} in sector)")

# ---------------------------------------------------------------------------
# Titans highlight
# ---------------------------------------------------------------------------
if scr_result.titans:
    st.markdown("### Top 10 Titans")
    for i, s in enumerate(scr_result.titans, 1):
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(i, f"**#{i}**")
        sc_color = (
            "#00C853" if s.composite_score >= 70
            else "#69F0AE" if s.composite_score >= 60
            else "#FFD54F" if s.composite_score >= 50
            else "#FF8A65"
        )
        roe_str = f"{s.roe * 100:.1f}%" if s.roe else "N/A"
        pe_str = f"{s.pe_trailing:.1f}" if s.pe_trailing else "N/A"
        de_str = f"{s.debt_to_equity:.0f}" if s.debt_to_equity else "N/A"
        roic_str = f"{s.roic * 100:.1f}%" if s.roic else "N/A"
        rg_str = f"{s.revenue_growth * 100:.1f}%" if s.revenue_growth else "N/A"
        st.markdown(
            f"<div style='padding:0.6rem 0.8rem;background:rgba(28,31,48,0.6);"
            f"border:1px solid rgba(255,255,255,0.08);border-radius:10px;margin-bottom:0.4rem;"
            f"display:flex;align-items:center;gap:0.8rem'>"
            f"<span style='font-size:1.3rem'>{medal}</span>"
            f"<div style='flex:1'>"
            f"<span style='color:white;font-weight:600'>{s.ticker}</span> "
            f"<span style='color:#aaa'>— {s.name}</span><br>"
            f"<span style='color:#888;font-size:0.82rem'>"
            f"P/E: {pe_str} · ROE: {roe_str} · ROIC: {roic_str} · "
            f"D/E: {de_str} · Rev Growth: {rg_str} · Rating: {s.analyst_rating}</span>"
            f"</div>"
            f"<div style='text-align:center'>"
            f"<div style='color:{sc_color};font-size:1.4rem;font-weight:700'>{s.composite_score:.0f}</div>"
            f"<div style='color:#888;font-size:0.7rem'>SCORE</div></div>"
            f"</div>",
            unsafe_allow_html=True,
        )

st.divider()

# ---------------------------------------------------------------------------
# Full results table
# ---------------------------------------------------------------------------
st.markdown("### Full Screener Results")
rows = []
for s in scr_result.stocks:
    rows.append({
        "Ticker": s.ticker,
        "Company": s.name,
        "Score": s.composite_score,
        "Rating": s.analyst_rating,
        "Price": f"${s.price:.2f}" if s.price else "N/A",
        "P/E": f"{s.pe_trailing:.1f}" if s.pe_trailing else "N/A",
        "Fwd P/E": f"{s.pe_forward:.1f}" if s.pe_forward else "N/A",
        "PEG": f"{s.peg_ratio:.2f}" if s.peg_ratio else "N/A",
        "ROE": f"{s.roe * 100:.1f}%" if s.roe else "N/A",
        "ROIC": f"{s.roic * 100:.1f}%" if s.roic else "N/A",
        "Net Margin": f"{s.net_margin * 100:.1f}%" if s.net_margin else "N/A",
        "Rev Growth": f"{s.revenue_growth * 100:.1f}%" if s.revenue_growth else "N/A",
        "D/E": f"{s.debt_to_equity:.0f}" if s.debt_to_equity else "N/A",
        "Div Yield": f"{s.dividend_yield * 100:.2f}%" if s.dividend_yield else "—",
    })

st.dataframe(
    pd.DataFrame(rows),
    use_container_width=True,
    hide_index=True,
    height=min(600, 40 + len(rows) * 38),
)

# ---------------------------------------------------------------------------
# Score distribution chart
# ---------------------------------------------------------------------------
st.markdown("### Score Distribution")
sorted_stocks = sorted(scr_result.stocks, key=lambda x: x.composite_score, reverse=True)
score_fig = go.Figure(go.Bar(
    x=[s.ticker for s in sorted_stocks],
    y=[s.composite_score for s in sorted_stocks],
    marker_color=[
        "#00C853" if s.composite_score >= 70
        else "#69F0AE" if s.composite_score >= 60
        else "#FFD54F" if s.composite_score >= 50
        else "#FF8A65"
        for s in sorted_stocks
    ],
    text=[f"{s.composite_score:.0f}" for s in sorted_stocks],
    textposition="auto",
))
score_fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis_title="Ticker",
    yaxis_title="Composite Score",
    height=350,
)
st.plotly_chart(score_fig, use_container_width=True)
