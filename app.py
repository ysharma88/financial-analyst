"""Streamlit dashboard for the Financial Analyst stock analysis tool."""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import json
import os
import sys
import urllib.parse
import urllib.request
from datetime import date

from dotenv import load_dotenv
load_dotenv()

# On Streamlit Cloud, secrets live in st.secrets — push them into os.environ
# so all downstream modules that call os.getenv() pick them up automatically.
try:
    for _k in ("ANTHROPIC_API_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID",
               "DISCORD_BOT_TOKEN", "DISCORD_CHANNEL_ID"):
        if _k in st.secrets and not os.environ.get(_k):
            os.environ[_k] = st.secrets[_k]
except Exception:
    pass

from data_fetcher import StockDataFetcher
from fundamental_analysis import FundamentalAnalyzer
from technical_analysis import TechnicalAnalyzer
from recommendation_engine import HolisticRecommendation
from risk_management import RiskManager
from macro_analysis import MacroAnalyzer
from sector_analysis import SectorAnalyzer, SECTOR_ETFS
from stock_screener import StockScreener
from news_sentiment import NewsSentimentAnalyzer
from governance_redflags import GovernanceAnalyzer
from reasoning_engine import ReasoningEngine
from quality_scores import QualityScorer
from dcf_model import DCFValuator
from peer_comps import PeerComparator
from options_analysis import OptionsAnalyzer
import institutional_risk as _inst_risk
import alternative_data as _alt_data
import forensic_nlp as _forensic

from earnings_quality import EarningsQualityAnalyzer
from capital_efficiency import CapitalEfficiencyAnalyzer
from technical_enhanced import TechnicalEnhancedAnalyzer
from credit_conditions import CreditConditionsAnalyzer
from portfolio_risk import PortfolioRiskAnalyzer
from sector_metrics import SectorMetricsAnalyzer


def send_telegram_test_message(bot_token: str, chat_id: str, message: str) -> tuple[bool, str]:
    try:
        payload = urllib.parse.urlencode({
            "chat_id": chat_id,
            "text": message,
        }).encode("utf-8")
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            data=payload,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status == 200:
                return True, "Test message sent successfully."
            return False, f"Telegram API returned status {resp.status}"
    except Exception as exc:
        return False, f"Failed to send message: {exc}"


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Financial Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* ── Research Report Typography ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Sidebar clean */
    section[data-testid="stSidebar"] { background: #0d0f1a; border-right: 1px solid rgba(255,255,255,0.06); }
    section[data-testid="stSidebar"] * { font-size: 0.85rem; }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: rgba(15, 17, 30, 0.8);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 8px;
        padding: 10px 14px;
    }
    div[data-testid="stMetric"] label { font-size: 0.72rem !important; color: #666 !important; text-transform: uppercase; letter-spacing: 0.06em; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { font-size: 1.15rem !important; font-weight: 600; }

    /* Report section headers */
    .rpt-section {
        font-size: 0.68rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #555;
        border-bottom: 1px solid rgba(255,255,255,0.06);
        padding-bottom: 0.3rem;
        margin: 1.4rem 0 0.8rem 0;
    }

    /* Verdict banner */
    .verdict-banner {
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }

    /* Ticker header */
    .ticker-name { font-size: 1.6rem; font-weight: 700; color: #fff; line-height: 1.2; }
    .ticker-sub  { font-size: 0.82rem; color: #555; margin-top: 0.1rem; letter-spacing: 0.02em; }
    .ticker-price { font-size: 2rem; font-weight: 700; }

    /* Divider */
    hr { border-color: rgba(255,255,255,0.06) !important; }

    /* Tab strip */
    button[data-baseweb="tab"] { font-size: 0.8rem !important; padding: 0.4rem 0.9rem !important; }

    /* Progress bar */
    div[data-testid="stProgress"] > div { background: rgba(102,126,234,0.3); }

    /* Watchlist chip */
    .wl-chip {
        display:inline-block; padding:0.25rem 0.65rem;
        background:rgba(102,126,234,0.12); border:1px solid rgba(102,126,234,0.25);
        border-radius:20px; color:#9fa8da; font-size:0.78rem;
        margin:0.15rem; cursor:pointer;
    }

    /* Score bar */
    .score-row { display:flex; align-items:center; gap:0.5rem; margin-bottom:0.2rem; font-size:0.82rem; }
    .score-label { color:#aaa; width:100px; flex-shrink:0; }
    .score-track { flex:1; height:5px; background:rgba(255,255,255,0.07); border-radius:3px; }
    .score-fill  { height:5px; border-radius:3px; }
    .score-val   { color:#ccc; width:44px; text-align:right; flex-shrink:0; }

    /* Rank card */
    .rank-card {
        background: rgba(15,17,30,0.7);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.5rem;
        transition: border-color 0.15s;
    }
    .rank-card:hover { border-color: rgba(102,126,234,0.35); }

    /* SR level pill */
    .sr-pill {
        display:inline-flex; align-items:center; gap:0.4rem;
        padding:0.25rem 0.65rem; border-radius:6px;
        font-size:0.8rem; font-weight:600; margin:0.15rem;
    }

    /* Info badge */
    .badge {
        display:inline-block; padding:0.15rem 0.5rem; border-radius:4px;
        font-size:0.72rem; font-weight:600; letter-spacing:0.04em;
    }

    /* Hide Streamlit footer only — keep header for sidebar toggle */
    #MainMenu {visibility:hidden;}
    footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    # ── Logo ──
    st.markdown(
        "<div style='padding:0.8rem 0 0.4rem 0'>"
        "<span style='font-size:1.3rem;font-weight:700;color:#fff'>📊 Financial Analyst</span><br>"
        "<span style='font-size:0.72rem;color:#444;letter-spacing:0.06em;text-transform:uppercase'>"
        "Equity Research Platform</span></div>",
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Config paths ──
    config_dir = os.path.join(os.path.dirname(__file__), "config")
    tracker_file_path    = os.path.join(config_dir, "ui_tracked_stocks.json")
    alert_config_path    = os.path.join(config_dir, "alert_settings.json")
    telegram_config_path = os.path.join(config_dir, "telegram_credentials.json")

    # ── Watchlists ──
    WATCHLISTS = {
        "Mag 7":      "AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA",
        "FAANG":      "META, AAPL, AMZN, NFLX, GOOGL",
        "Index ETFs": "SPY, QQQ, IWM, DIA, GLD",
        "China Tech": "BABA, JD, BIDU, PDD, NTES",
        "Financials": "JPM, BAC, GS, MS, BRK-B",
    }
    with st.expander("⚡ Quick Watchlists"):
        for wl_name, wl_tickers in WATCHLISTS.items():
            if st.button(wl_name, key=f"wl_{wl_name}", use_container_width=True):
                st.session_state["_ticker_prefill"] = wl_tickers

    # ── Ticker input ──
    st.markdown("<div class='rpt-section'>SECURITIES</div>", unsafe_allow_html=True)

    if "ui_tracked_symbols" not in st.session_state:
        loaded_symbols = []
        if os.path.exists(tracker_file_path):
            try:
                with open(tracker_file_path, "r", encoding="utf-8") as fh:
                    payload = json.load(fh)
                raw = payload.get("symbols", []) if isinstance(payload, dict) else (payload if isinstance(payload, list) else [])
                loaded_symbols = [str(s).upper() for s in raw if str(s).strip()]
            except Exception:
                loaded_symbols = []
        st.session_state["ui_tracked_symbols"] = loaded_symbols

    _prefill = st.session_state.pop("_ticker_prefill", None)
    if _prefill:
        st.session_state["ticker_input"] = _prefill
    elif "ticker_input" not in st.session_state:
        st.session_state["ticker_input"] = (
            ",".join(st.session_state["ui_tracked_symbols"])
            if st.session_state["ui_tracked_symbols"] else "AAPL"
        )

    ticker_input = st.text_area(
        "Tickers",
        key="ticker_input",
        height=68,
        placeholder="AAPL, MSFT, GOOGL ...",
        help="Comma-separated. Max 5 at a time to avoid rate limits.",
        label_visibility="collapsed",
    )

    # Watchlist tracker (compact)
    with st.expander("📌 Saved Watchlist"):
        tracker_symbol = st.text_input("Add ticker", value="", placeholder="TSLA", key="tracker_symbol_input", label_visibility="collapsed")
        tc1, tc2, tc3 = st.columns(3)
        with tc1:
            if st.button("Add", use_container_width=True):
                sym = tracker_symbol.strip().upper()
                if sym and sym not in st.session_state["ui_tracked_symbols"]:
                    st.session_state["ui_tracked_symbols"].append(sym)
        with tc2:
            if st.button("Load", use_container_width=True):
                st.session_state["_ticker_prefill"] = ",".join(st.session_state["ui_tracked_symbols"])
                st.rerun()
        with tc3:
            if st.button("Clear", use_container_width=True):
                st.session_state["ui_tracked_symbols"] = []
        if st.session_state["ui_tracked_symbols"]:
            st.caption(", ".join(st.session_state["ui_tracked_symbols"]))
            if st.button("💾 Save", use_container_width=True):
                os.makedirs(os.path.dirname(tracker_file_path), exist_ok=True)
                with open(tracker_file_path, "w", encoding="utf-8") as fh:
                    json.dump({"symbols": st.session_state["ui_tracked_symbols"]}, fh, indent=2)
                st.success("Saved.")

    # ── Analysis parameters ──
    st.markdown("<div class='rpt-section'>PARAMETERS</div>", unsafe_allow_html=True)

    period = st.selectbox(
        "Period",
        ["6mo", "1y", "2y", "5y"],
        index=1,
        help="Historical period for technical analysis",
    )
    fundamental_weight = st.slider(
        "Fundamental Weight %",
        min_value=0, max_value=100, value=50, step=5,
        help="Balance between fundamental and technical scoring",
    )
    technical_weight = 100 - fundamental_weight
    st.caption(f"Technical: {technical_weight}%  |  Fundamental: {fundamental_weight}%")

    # Execution strategy
    strategy_options = ["VWAP", "TWAP", "POV", "IS"]
    selected_strategy = st.selectbox(
        "Execution",
        strategy_options,
        index=0,
        key="execution_strategy_ui",
        help="Order execution algorithm",
    )
    strategy_blurb = {
        "VWAP": "Volume-weighted — best for liquid names",
        "TWAP": "Time-sliced — stealth in illiquid names",
        "POV":  "Participate-at-volume — adapts to flow",
        "IS":   "Implementation Shortfall — minimise slippage",
    }
    st.caption(strategy_blurb[selected_strategy])

    # ── Analyze button ──
    st.markdown("<br>", unsafe_allow_html=True)
    analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)

    # ── Cache status ──
    try:
        import cache as _cache_mod
        _cache_db = os.path.join(os.path.dirname(__file__), "cache", "yfinance_cache.db")
        if os.path.exists(_cache_db):
            _mtime = os.path.getmtime(_cache_db)
            from datetime import datetime as _dt
            _last = _dt.fromtimestamp(_mtime).strftime("%H:%M")
            st.caption(f"Cache last updated: {_last}")
        else:
            st.caption("Cache: empty")
    except Exception:
        pass

    # ── Integrations (collapsed) ──
    st.markdown("<div class='rpt-section'>INTEGRATIONS</div>", unsafe_allow_html=True)

    with st.expander("🔔 Alert Settings"):
        if "alert_settings" not in st.session_state:
            default_alerts = {
                "poll_seconds": 300,
                "price_jump_threshold_pct": 1.2,
                "atr_spike_threshold_pct": 20.0,
                "momentum_spike_abs_pct": 2.5,
                "fundamental": {
                    "debt_to_equity_gte": 2.0, "roe_lt": 0.15,
                    "dividend_payout_gt": 0.75, "peg_lt": 1.0,
                    "price_to_sales_gt": 50.0, "wacc_assumed": 0.10,
                },
                "technical": {"rsi_overbought": 70.0, "rsi_oversold": 30.0, "atr_spike_vs_median_pct": 25.0},
                "macro": {"vix_crisis_gt": 30.0, "vix_calm_lt": 20.0, "buffett_indicator_gt": 120.0, "inflation_surprise_abs_gt": 0.2},
                "institutional": {"ownership_change_gt_pct": 5.0, "sentiment_abs_gt": 0.225, "earnings_revision_bottom_pct": 40.0},
                "macro_inputs": {"buffett_indicator_pct": None, "us10y": None, "us2y": None, "inflation_surprise_pct": None},
                "institutional_inputs": {"ownership_change_pct": None, "sentiment_score": None, "earnings_revision_percentile": None},
            }
            if os.path.exists(alert_config_path):
                try:
                    with open(alert_config_path, "r", encoding="utf-8") as fh:
                        payload = json.load(fh)
                    if isinstance(payload, dict):
                        default_alerts.update(payload)
                except Exception:
                    pass
            st.session_state["alert_settings"] = default_alerts

        a1, a2 = st.columns(2)
        poll_seconds_ui = a1.number_input("Poll (s)", min_value=10, max_value=3600,
            value=int(st.session_state["alert_settings"].get("poll_seconds", 300)), step=10, key="alert_poll_seconds_ui")
        price_jump_ui = a2.number_input("Price jump %", min_value=0.1, max_value=20.0,
            value=float(st.session_state["alert_settings"].get("price_jump_threshold_pct", 1.2)), step=0.1, key="alert_price_jump_ui")
        a3, a4 = st.columns(2)
        atr_spike_ui = a3.number_input("ATR spike %", min_value=1.0, max_value=200.0,
            value=float(st.session_state["alert_settings"].get("atr_spike_threshold_pct", 20.0)), step=1.0, key="alert_atr_spike_ui")
        momentum_shift_ui = a4.number_input("Momentum pts", min_value=0.1, max_value=50.0,
            value=float(st.session_state["alert_settings"].get("momentum_spike_abs_pct", 2.5)), step=0.1, key="alert_momentum_shift_ui")

        with st.expander("Advanced Fundamental Thresholds"):
            f1, f2 = st.columns(2)
            de_ui = f1.number_input("D/E >=", min_value=0.0, max_value=20.0, value=float(st.session_state["alert_settings"].get("fundamental", {}).get("debt_to_equity_gte", 2.0)), step=0.1)
            roe_ui = f2.number_input("ROE <", min_value=0.0, max_value=1.0, value=float(st.session_state["alert_settings"].get("fundamental", {}).get("roe_lt", 0.15)), step=0.01)
            f3, f4 = st.columns(2)
            payout_ui = f3.number_input("Payout >", min_value=0.0, max_value=2.0, value=float(st.session_state["alert_settings"].get("fundamental", {}).get("dividend_payout_gt", 0.75)), step=0.01)
            peg_ui = f4.number_input("PEG <", min_value=0.0, max_value=10.0, value=float(st.session_state["alert_settings"].get("fundamental", {}).get("peg_lt", 1.0)), step=0.1)
            f5, f6 = st.columns(2)
            ps_ui = f5.number_input("P/S >", min_value=0.0, max_value=500.0, value=float(st.session_state["alert_settings"].get("fundamental", {}).get("price_to_sales_gt", 50.0)), step=1.0)
            wacc_ui = f6.number_input("WACC", min_value=0.0, max_value=1.0, value=float(st.session_state["alert_settings"].get("fundamental", {}).get("wacc_assumed", 0.10)), step=0.01)

        with st.expander("Advanced Technical Thresholds"):
            t1, t2 = st.columns(2)
            rsi_over_ui = t1.number_input("RSI overbought", min_value=50.0, max_value=100.0, value=float(st.session_state["alert_settings"].get("technical", {}).get("rsi_overbought", 70.0)), step=1.0)
            rsi_under_ui = t2.number_input("RSI oversold", min_value=0.0, max_value=50.0, value=float(st.session_state["alert_settings"].get("technical", {}).get("rsi_oversold", 30.0)), step=1.0)
            atr_median_ui = st.number_input("ATR vs median %", min_value=1.0, max_value=200.0, value=float(st.session_state["alert_settings"].get("technical", {}).get("atr_spike_vs_median_pct", 25.0)), step=1.0)

        with st.expander("Macro Thresholds + Inputs"):
            m1, m2 = st.columns(2)
            vix_crisis_ui = m1.number_input("VIX crisis >", min_value=5.0, max_value=100.0, value=float(st.session_state["alert_settings"].get("macro", {}).get("vix_crisis_gt", 30.0)), step=1.0)
            vix_calm_ui = m2.number_input("VIX calm <", min_value=5.0, max_value=50.0, value=float(st.session_state["alert_settings"].get("macro", {}).get("vix_calm_lt", 20.0)), step=1.0)
            buffett_th_ui = st.number_input("Buffett ind. %", min_value=50.0, max_value=500.0, value=float(st.session_state["alert_settings"].get("macro", {}).get("buffett_indicator_gt", 120.0)), step=5.0)
            inflation_abs_ui = st.number_input("Inflation surprise", min_value=0.0, max_value=5.0, value=float(st.session_state["alert_settings"].get("macro", {}).get("inflation_surprise_abs_gt", 0.2)), step=0.1)
            macro_input_default = st.session_state["alert_settings"].get("macro_inputs", {})
            buffett_input_ui = st.text_input("Buffett current", value="" if macro_input_default.get("buffett_indicator_pct") is None else str(macro_input_default.get("buffett_indicator_pct")))
            us10_input_ui  = st.text_input("US 10Y yield", value="" if macro_input_default.get("us10y") is None else str(macro_input_default.get("us10y")))
            us2_input_ui   = st.text_input("US 2Y yield", value="" if macro_input_default.get("us2y") is None else str(macro_input_default.get("us2y")))
            inflation_input_ui = st.text_input("Inflation surprise", value="" if macro_input_default.get("inflation_surprise_pct") is None else str(macro_input_default.get("inflation_surprise_pct")))

        with st.expander("Institutional Thresholds + Inputs"):
            i1, i2 = st.columns(2)
            own_th_ui = i1.number_input("Ownership chg %", min_value=0.0, max_value=100.0, value=float(st.session_state["alert_settings"].get("institutional", {}).get("ownership_change_gt_pct", 5.0)), step=0.5)
            sent_th_ui = i2.number_input("Sentiment abs", min_value=0.0, max_value=1.0, value=float(st.session_state["alert_settings"].get("institutional", {}).get("sentiment_abs_gt", 0.225)), step=0.01)
            rev_th_ui = st.number_input("Earnings rev. %ile", min_value=1.0, max_value=100.0, value=float(st.session_state["alert_settings"].get("institutional", {}).get("earnings_revision_bottom_pct", 40.0)), step=1.0)
            inst_input_default = st.session_state["alert_settings"].get("institutional_inputs", {})
            own_input_ui  = st.text_input("Ownership chg now", value="" if inst_input_default.get("ownership_change_pct") is None else str(inst_input_default.get("ownership_change_pct")))
            sent_input_ui = st.text_input("Sentiment now", value="" if inst_input_default.get("sentiment_score") is None else str(inst_input_default.get("sentiment_score")))
            rev_input_ui  = st.text_input("Earnings rev. now", value="" if inst_input_default.get("earnings_revision_percentile") is None else str(inst_input_default.get("earnings_revision_percentile")))

        if st.button("Save Alert Settings", use_container_width=True):
            def _to_float_or_none(v):
                try:
                    s = str(v).strip()
                    return None if s == "" else float(s)
                except Exception:
                    return None
            st.session_state["alert_settings"] = {
                "poll_seconds": int(poll_seconds_ui),
                "price_jump_threshold_pct": float(price_jump_ui),
                "atr_spike_threshold_pct": float(atr_spike_ui),
                "momentum_spike_abs_pct": float(momentum_shift_ui),
                "fundamental": {"debt_to_equity_gte": float(de_ui), "roe_lt": float(roe_ui), "dividend_payout_gt": float(payout_ui), "peg_lt": float(peg_ui), "price_to_sales_gt": float(ps_ui), "wacc_assumed": float(wacc_ui)},
                "technical": {"rsi_overbought": float(rsi_over_ui), "rsi_oversold": float(rsi_under_ui), "atr_spike_vs_median_pct": float(atr_median_ui)},
                "macro": {"vix_crisis_gt": float(vix_crisis_ui), "vix_calm_lt": float(vix_calm_ui), "buffett_indicator_gt": float(buffett_th_ui), "inflation_surprise_abs_gt": float(inflation_abs_ui)},
                "institutional": {"ownership_change_gt_pct": float(own_th_ui), "sentiment_abs_gt": float(sent_th_ui), "earnings_revision_bottom_pct": float(rev_th_ui)},
                "macro_inputs": {"buffett_indicator_pct": _to_float_or_none(buffett_input_ui), "us10y": _to_float_or_none(us10_input_ui), "us2y": _to_float_or_none(us2_input_ui), "inflation_surprise_pct": _to_float_or_none(inflation_input_ui)},
                "institutional_inputs": {"ownership_change_pct": _to_float_or_none(own_input_ui), "sentiment_score": _to_float_or_none(sent_input_ui), "earnings_revision_percentile": _to_float_or_none(rev_input_ui)},
            }
            os.makedirs(config_dir, exist_ok=True)
            with open(alert_config_path, "w", encoding="utf-8") as fh:
                json.dump(st.session_state["alert_settings"], fh, indent=2)
            st.success("Saved.")

    with st.expander("✈️ Telegram Alerts"):
        if "telegram_credentials" not in st.session_state:
            creds = {"bot_token": os.getenv("TELEGRAM_BOT_TOKEN", ""), "chat_id": os.getenv("TELEGRAM_CHAT_ID", "")}
            if not creds["bot_token"] and os.path.exists(telegram_config_path):
                try:
                    with open(telegram_config_path, "r", encoding="utf-8") as fh:
                        payload = json.load(fh)
                    if isinstance(payload, dict):
                        creds["bot_token"] = str(payload.get("bot_token", "")).strip()
                        creds["chat_id"] = str(payload.get("chat_id", "")).strip()
                except Exception:
                    pass
            st.session_state["telegram_credentials"] = creds

        bot_token_ui = st.text_input("Bot Token", value=st.session_state["telegram_credentials"].get("bot_token", ""),
            type="password", key="telegram_bot_token_ui", help="Token from @BotFather")
        chat_id_ui = st.text_input("Chat ID", value=st.session_state["telegram_credentials"].get("chat_id", ""),
            key="telegram_chat_id_ui")
        if st.button("Save Telegram", use_container_width=True):
            os.makedirs(config_dir, exist_ok=True)
            creds = {"bot_token": bot_token_ui.strip(), "chat_id": chat_id_ui.strip()}
            with open(telegram_config_path, "w", encoding="utf-8") as fh:
                json.dump(creds, fh, indent=2)
            try: os.chmod(telegram_config_path, 0o600)
            except Exception: pass
            st.session_state["telegram_credentials"] = creds
            st.success("Saved.")
        test_message_ui = st.text_input("Test message", value="Test alert", key="telegram_test_message_ui")
        if st.button("Send Test", use_container_width=True):
            token_for_test = bot_token_ui.strip() or st.session_state["telegram_credentials"].get("bot_token", "")
            chat_for_test  = chat_id_ui.strip()   or st.session_state["telegram_credentials"].get("chat_id", "")
            if not token_for_test or not chat_for_test:
                st.error("Set Bot Token and Chat ID first.")
            else:
                ok, msg = send_telegram_test_message(token_for_test, chat_for_test, test_message_ui.strip() or "Test alert")
                st.success(msg) if ok else st.error(msg)

    with st.expander("💬 Discord Bot"):
        import subprocess
        discord_config_path = os.path.join(config_dir, "discord_credentials.json")
        if "discord_credentials" not in st.session_state:
            dc = {"bot_token": os.getenv("DISCORD_BOT_TOKEN", ""), "channel_id": os.getenv("DISCORD_CHANNEL_ID", "")}
            if not dc["bot_token"] and os.path.exists(discord_config_path):
                try:
                    with open(discord_config_path, "r", encoding="utf-8") as _fh:
                        _payload = json.load(_fh)
                    dc["bot_token"] = str(_payload.get("bot_token", "")).strip()
                    dc["channel_id"] = str(_payload.get("channel_id", "")).strip()
                except Exception:
                    pass
            st.session_state["discord_credentials"] = dc
        if "discord_proc" not in st.session_state:
            st.session_state["discord_proc"] = None

        discord_token_ui = st.text_input("Bot Token", value=st.session_state["discord_credentials"].get("bot_token", ""),
            type="password", key="discord_bot_token_ui")
        discord_channel_ui = st.text_input("Channel ID (optional)", value=st.session_state["discord_credentials"].get("channel_id", ""),
            key="discord_channel_id_ui")
        if st.button("Save Discord", use_container_width=True):
            os.makedirs(config_dir, exist_ok=True)
            _dc = {"bot_token": discord_token_ui.strip(), "channel_id": discord_channel_ui.strip()}
            with open(discord_config_path, "w", encoding="utf-8") as _fh:
                json.dump(_dc, _fh, indent=2)
            try: os.chmod(discord_config_path, 0o600)
            except Exception: pass
            st.session_state["discord_credentials"] = _dc
            st.success("Saved.")

        _proc = st.session_state["discord_proc"]
        _bot_running = _proc is not None and _proc.poll() is None
        if _bot_running:
            st.success("🟢 Bot running")
            if st.button("🛑 Stop", use_container_width=True):
                try: _proc.terminate(); _proc.wait(timeout=5)
                except Exception: pass
                st.session_state["discord_proc"] = None
                st.rerun()
        else:
            if _proc is not None:
                st.warning("⚠️ Bot stopped unexpectedly.")
                st.session_state["discord_proc"] = None
            if st.button("▶️ Start Bot", use_container_width=True):
                _token = discord_token_ui.strip() or st.session_state["discord_credentials"].get("bot_token", "")
                _ch_id = discord_channel_ui.strip() or st.session_state["discord_credentials"].get("channel_id", "")
                if not _token:
                    st.error("Set Discord Bot Token first.")
                else:
                    _env = os.environ.copy()
                    _env["DISCORD_BOT_TOKEN"] = _token
                    if _ch_id: _env["DISCORD_CHANNEL_ID"] = _ch_id
                    _bot_script = os.path.join(os.path.dirname(__file__), "discord_bot.py")
                    _proc = subprocess.Popen([sys.executable, _bot_script], env=_env,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                    st.session_state["discord_proc"] = _proc
                    st.success("Bot started!")
                    st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("Data: Yahoo Finance · Not financial advice")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_tickers(raw: str) -> list:
    """Parse comma/space/newline separated tickers into a clean list."""
    import re
    tokens = re.split(r"[,\s\n]+", raw.strip())
    seen = set()
    tickers = []
    for t in tokens:
        t = t.strip().upper()
        if t and t not in seen:
            seen.add(t)
            tickers.append(t)
    return tickers


def format_large_number(n) -> str:
    if n is None:
        return "N/A"
    if abs(n) >= 1e12:
        return f"${n / 1e12:.2f}T"
    if abs(n) >= 1e9:
        return f"${n / 1e9:.2f}B"
    if abs(n) >= 1e6:
        return f"${n / 1e6:.1f}M"
    return f"${n:,.0f}"


def format_pct(v, mult100=False) -> str:
    if v is None:
        return "N/A"
    val = v * 100 if mult100 else v
    return f"{val:.2f}%"


def score_to_color(score: float) -> str:
    if score >= 0.3:
        return "#00C853"
    if score >= 0.1:
        return "#69F0AE"
    if score >= -0.1:
        return "#FFD54F"
    if score >= -0.3:
        return "#FF8A65"
    return "#FF1744"


def score_to_label(score: float) -> str:
    if score >= 0.5:
        return "STRONG BUY"
    if score >= 0.2:
        return "BUY"
    if score >= -0.2:
        return "HOLD"
    if score >= -0.5:
        return "SELL"
    return "STRONG SELL"


def score_bar_html(score: float, label: str = "") -> str:
    pct = (score + 1) / 2 * 100
    color = score_to_color(score)
    return f"""
    <div style="margin-bottom:4px">
        <span style="font-size:0.8rem;color:#aaa">{label}</span>
        <span style="float:right;font-size:0.8rem;color:{color};font-weight:600">{score:+.2f}</span>
    </div>
    <div style="background:#333;border-radius:6px;height:8px;margin-bottom:10px">
        <div style="background:{color};width:{pct:.0f}%;height:8px;border-radius:6px"></div>
    </div>
    """


TICKER_COLORS = [
    "#667eea", "#42A5F5", "#26a69a", "#FFA726", "#AB47BC",
    "#ef5350", "#66BB6A", "#FF7043", "#5C6BC0", "#26C6DA",
    "#FFCA28", "#8D6E63", "#78909C", "#EC407A", "#9CCC65",
]


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def create_price_chart(df: pd.DataFrame, ticker: str, sr_levels=None) -> go.Figure:
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=("Price & Moving Averages", "RSI (14)", "MACD"),
    )

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="Price",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    ), row=1, col=1)

    for col, color, name in [
        ("SMA_20", "#42A5F5", "SMA 20"),
        ("SMA_50", "#FFA726", "SMA 50"),
        ("SMA_200", "#AB47BC", "SMA 200"),
    ]:
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col], name=name,
                line=dict(color=color, width=1.2),
            ), row=1, col=1)

    if "BB_Upper" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_Upper"], name="BB Upper",
            line=dict(color="rgba(128,128,128,0.3)", width=1, dash="dot"),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_Lower"], name="BB Lower",
            line=dict(color="rgba(128,128,128,0.3)", width=1, dash="dot"),
            fill="tonexty", fillcolor="rgba(128,128,128,0.05)",
        ), row=1, col=1)

    if "RSI" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["RSI"], name="RSI",
            line=dict(color="#42A5F5", width=1.5),
        ), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="rgba(255,82,82,0.5)", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="rgba(76,175,80,0.5)", row=2, col=1)
        fig.add_hrect(y0=30, y1=70, fillcolor="rgba(128,128,128,0.05)", line_width=0, row=2, col=1)

    if "MACD" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["MACD"], name="MACD",
            line=dict(color="#42A5F5", width=1.5),
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["MACD_Signal"], name="Signal",
            line=dict(color="#FFA726", width=1.2),
        ), row=3, col=1)
        colors = ["#26a69a" if v >= 0 else "#ef5350" for v in df["MACD_Hist"].fillna(0)]
        fig.add_trace(go.Bar(
            x=df.index, y=df["MACD_Hist"], name="Histogram",
            marker_color=colors, opacity=0.6,
        ), row=3, col=1)

    # Support & Resistance levels
    if sr_levels:
        for level in sr_levels:
            is_support = level.kind == "support"
            color = "rgba(76,175,80,0.6)" if is_support else "rgba(255,82,82,0.6)"
            dash = "dash" if level.strength == "strong" else "dot"
            width = 1.8 if level.strength == "strong" else 1.2 if level.strength == "moderate" else 0.8
            fig.add_hline(
                y=level.price, line_dash=dash, line_color=color, line_width=width,
                annotation_text=f"{'S' if is_support else 'R'} ${level.price:.2f} ({level.strength})",
                annotation_position="right",
                annotation_font_size=9,
                annotation_font_color=color,
                row=1, col=1,
            )

    fig.update_layout(
        template="plotly_dark",
        height=700,
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_rangeslider_visible=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def create_volume_chart(df: pd.DataFrame) -> go.Figure:
    colors = []
    for i in range(len(df)):
        if i == 0:
            colors.append("#42A5F5")
        elif df["Close"].iloc[i] >= df["Close"].iloc[i - 1]:
            colors.append("#26a69a")
        else:
            colors.append("#ef5350")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"], name="Volume",
        marker_color=colors, opacity=0.7,
    ))
    if "Volume_SMA_20" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["Volume_SMA_20"], name="Vol SMA 20",
            line=dict(color="#FFA726", width=1.5),
        ))

    fig.update_layout(
        template="plotly_dark",
        height=250,
        margin=dict(l=10, r=10, t=30, b=10),
        title="Volume Analysis",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def create_gauge(score: float, title: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number=dict(suffix="", font=dict(size=28)),
        gauge=dict(
            axis=dict(range=[-1, 1], tickvals=[-1, -0.5, 0, 0.5, 1],
                      ticktext=["Strong Sell", "Sell", "Hold", "Buy", "Strong Buy"]),
            bar=dict(color=score_to_color(score), thickness=0.3),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            steps=[
                dict(range=[-1, -0.5], color="rgba(255,23,68,0.15)"),
                dict(range=[-0.5, -0.2], color="rgba(255,138,101,0.15)"),
                dict(range=[-0.2, 0.2], color="rgba(255,213,79,0.15)"),
                dict(range=[0.2, 0.5], color="rgba(105,240,174,0.15)"),
                dict(range=[0.5, 1], color="rgba(0,200,83,0.15)"),
            ],
            threshold=dict(line=dict(color="white", width=2), thickness=0.75, value=score),
        ),
        title=dict(text=title, font=dict(size=14)),
    ))
    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
    )
    return fig


def create_category_radar(fundamental_cats: dict, technical_cats: dict) -> go.Figure:
    all_cats = {}
    for k, v in fundamental_cats.items():
        all_cats[f"F: {k}"] = v
    for k, v in technical_cats.items():
        all_cats[f"T: {k}"] = v

    categories = list(all_cats.keys())
    values = [(v + 1) / 2 * 100 for v in all_cats.values()]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill="toself",
        fillcolor="rgba(102,126,234,0.2)",
        line=dict(color="#667eea", width=2),
        name="Score",
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], showticklabels=False, gridcolor="rgba(255,255,255,0.1)"),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=False,
        height=350,
        margin=dict(l=60, r=60, t=30, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", size=11),
    )
    return fig


# ---------------------------------------------------------------------------
# Multi-stock comparison charts
# ---------------------------------------------------------------------------
def create_comparison_bar(stock_results: dict) -> go.Figure:
    """Horizontal bar chart comparing Overall / Fundamental / Technical scores."""
    tickers = list(stock_results.keys())
    overall = [stock_results[t]["recommendation"].overall_score for t in tickers]
    fund = [stock_results[t]["fund_result"].overall_score for t in tickers]
    tech = [stock_results[t]["tech_result"].overall_score for t in tickers]

    sorted_pairs = sorted(zip(tickers, overall, fund, tech), key=lambda x: x[1], reverse=True)
    tickers = [p[0] for p in sorted_pairs]
    overall = [p[1] for p in sorted_pairs]
    fund = [p[2] for p in sorted_pairs]
    tech = [p[3] for p in sorted_pairs]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=tickers, x=overall, name="Overall",
        orientation="h",
        marker_color=[score_to_color(s) for s in overall],
        text=[f"{s:+.2f}" for s in overall],
        textposition="auto",
    ))
    fig.add_trace(go.Bar(
        y=tickers, x=fund, name="Fundamental",
        orientation="h",
        marker_color="rgba(102,126,234,0.7)",
        text=[f"{s:+.2f}" for s in fund],
        textposition="auto",
    ))
    fig.add_trace(go.Bar(
        y=tickers, x=tech, name="Technical",
        orientation="h",
        marker_color="rgba(171,71,188,0.7)",
        text=[f"{s:+.2f}" for s in tech],
        textposition="auto",
    ))

    fig.update_layout(
        template="plotly_dark",
        barmode="group",
        height=max(250, len(tickers) * 70 + 80),
        margin=dict(l=10, r=10, t=40, b=10),
        title="Score Comparison",
        xaxis=dict(range=[-1, 1], title="Score"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.add_vline(x=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
    return fig


def create_normalized_price_chart(stock_results: dict, period: str) -> go.Figure:
    """Overlay normalized (% change from start) price for all stocks."""
    fig = go.Figure()
    for i, (ticker, data) in enumerate(stock_results.items()):
        df = data["chart_df"]
        if df is None or len(df) == 0:
            continue
        base = df["Close"].iloc[0]
        if base == 0:
            continue
        normalized = (df["Close"] / base - 1) * 100
        color = TICKER_COLORS[i % len(TICKER_COLORS)]
        fig.add_trace(go.Scatter(
            x=df.index, y=normalized, name=ticker,
            line=dict(color=color, width=2),
        ))

    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=10, r=10, t=40, b=10),
        title=f"Price Performance Comparison ({period})",
        yaxis=dict(title="% Change", ticksuffix="%"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.2)")
    return fig


def create_multi_radar(stock_results: dict) -> go.Figure:
    """Overlay radar charts for all stocks."""
    fig = go.Figure()

    all_categories = None
    for i, (ticker, data) in enumerate(stock_results.items()):
        f_cats = data["fund_result"].category_scores
        t_cats = data["tech_result"].category_scores
        combined = {}
        for k, v in f_cats.items():
            combined[f"F: {k}"] = v
        for k, v in t_cats.items():
            combined[f"T: {k}"] = v

        if all_categories is None:
            all_categories = list(combined.keys())

        values = [(combined.get(c, 0) + 1) / 2 * 100 for c in all_categories]
        color = TICKER_COLORS[i % len(TICKER_COLORS)]

        # Convert hex color to rgba for the fill
        def hex_to_rgba(hex_color, alpha=0.08):
            h = hex_color.lstrip("#")
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            return f"rgba({r},{g},{b},{alpha})"

        fill = hex_to_rgba(color) if color.startswith("#") else color.replace(")", f",0.08)").replace("rgb(", "rgba(")

        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=all_categories + [all_categories[0]],
            fill="toself",
            fillcolor=fill,
            line=dict(color=color, width=2),
            name=ticker,
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], showticklabels=False, gridcolor="rgba(255,255,255,0.1)"),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
            bgcolor="rgba(0,0,0,0)",
        ),
        height=450,
        margin=dict(l=80, r=80, t=40, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", size=11),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
    )
    return fig


def create_metric_comparison_chart(stock_results: dict, metric_name: str, metric_key: str, fmt: str = ".2f") -> go.Figure:
    """Bar chart comparing a single fundamental metric across stocks."""
    tickers = []
    values = []
    for ticker, data in stock_results.items():
        v = data["fund_data"].get(metric_key)
        if v is not None:
            tickers.append(ticker)
            values.append(v)

    if not tickers:
        return None

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=tickers, y=values,
        marker_color=[TICKER_COLORS[i % len(TICKER_COLORS)] for i in range(len(tickers))],
        text=[f"{v:{fmt}}" for v in values],
        textposition="auto",
    ))
    fig.update_layout(
        template="plotly_dark",
        height=300,
        margin=dict(l=10, r=10, t=40, b=10),
        title=metric_name,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ---------------------------------------------------------------------------
# Main analysis (cached per ticker)
# ---------------------------------------------------------------------------
@st.cache_resource(ttl=900, show_spinner=False)
def run_single_analysis(ticker: str, period: str, f_weight: int):
    fetcher = StockDataFetcher(ticker)
    if not fetcher.validate():
        return None

    import time
    time.sleep(1)

    fund_data = fetcher.get_fundamental_data()
    history = fetcher.get_history(period=period)

    fund_analyzer = FundamentalAnalyzer()
    tech_analyzer = TechnicalAnalyzer()

    fund_result = fund_analyzer.analyze(fund_data, fetcher.get_sector())
    tech_result = tech_analyzer.analyze(history)

    chart_df = tech_analyzer.get_chart_data(history)

    recommendation = HolisticRecommendation(
        fundamental=fund_result,
        technical=tech_result,
        fundamental_weight=f_weight / 100,
        technical_weight=(100 - f_weight) / 100,
    )

    company_info = {
        "name": fetcher.get_company_name(),
        "sector": fetcher.get_sector(),
        "industry": fetcher.get_industry(),
        "price": fetcher.get_current_price(),
        "market_cap": fetcher.get_market_cap(),
        "52w_high": fund_data.get("fifty_two_week_high"),
        "52w_low": fund_data.get("fifty_two_week_low"),
        "beta": fund_data.get("beta"),
        "avg_volume": fund_data.get("avg_volume"),
    }

    # Analyst & Investor data
    analyst_targets = fetcher.get_analyst_price_targets()
    upgrades_downgrades = fetcher.get_upgrades_downgrades(limit=20)
    recommendations_summary = fetcher.get_recommendations_summary()
    institutional_holders = fetcher.get_institutional_holders()
    mutualfund_holders = fetcher.get_mutualfund_holders()
    major_holders = fetcher.get_major_holders()

    # Risk management
    risk_mgr = RiskManager()
    target_list = []
    mean_target = analyst_targets.get("mean")
    if mean_target and mean_target > 0:
        target_list.append(mean_target)
    high_target = analyst_targets.get("high")
    if high_target and high_target > 0 and high_target != mean_target:
        target_list.append(high_target)

    risk_profile = risk_mgr.analyze(
        df=history,
        entry_price=company_info["price"] or 0,
        account_size=100_000,
        risk_pct=1.0,
        target_prices=target_list or None,
    )

    # Governance & red flags
    gov_analyzer = GovernanceAnalyzer()
    earnings_history = fetcher.get_earnings_history()
    insider_purchases = fetcher.get_insider_purchases()
    governance_scores = fetcher.get_governance_scores()
    company_officers = fetcher.get_company_officers()

    governance_result = gov_analyzer.analyze(
        info=fetcher.info,
        company_officers=company_officers,
        earnings_history=earnings_history,
        governance_scores=governance_scores,
        insider_purchases=insider_purchases,
    )

    # Quality scores (Piotroski, Altman, Beneish)
    quality_scorer = QualityScorer()
    income_stmt_full = fetcher.get_income_statement()
    balance_sheet_full = fetcher.get_balance_sheet()
    cashflow_stmt = fetcher.get_cashflow_statement()
    quality_scores = quality_scorer.compute(
        info=fetcher.info,
        income_stmt=income_stmt_full,
        balance_sheet=balance_sheet_full,
        cashflow=cashflow_stmt,
    )

    # DCF intrinsic value
    dcf_valuator = DCFValuator()
    dcf_result = dcf_valuator.compute(
        info=fetcher.info,
        income_stmt=income_stmt_full,
        balance_sheet=balance_sheet_full,
        cashflow=cashflow_stmt,
    )

    # Financial trends
    financial_trends = fetcher.get_financial_trends()

    # Event calendar
    event_calendar = fetcher.get_calendar()

    # Peer comparable analysis
    peer_comparator = PeerComparator()
    peer_comps_result = peer_comparator.compare(
        subject_ticker=ticker,
        subject_info=fetcher.info,
        sector=fetcher.get_sector(),
        max_peers=8,
    )

    # Options market signals
    options_analyzer = OptionsAnalyzer()
    options_result = options_analyzer.analyze(ticker, history)

    # Institutional risk (GEX, smart money, credit proxy)
    inst_risk_result = _inst_risk.analyze(
        ticker=ticker,
        price=company_info.get("price", 0),
        info=fetcher.info,
        income_stmt=fetcher.get_income_statement(),
    )

    alt_data_bundle = _alt_data.cached_analyze(
        ticker=ticker,
        price=company_info.get("price", 0),
    )

    # Forensic NLP — EDGAR MD&A + Loughran-McDonald + Claude deception analysis
    _qs_dict = {}
    if quality_scores:
        _qs_dict = {
            "beneish_m": getattr(quality_scores, "beneish_m", None),
            "piotroski_f": getattr(quality_scores, "piotroski_f", None),
            "altman_z": getattr(quality_scores, "altman_z", None),
            "tata": getattr(quality_scores, "tata", None),
        }
    forensic_result = _forensic.cached_analyze(ticker=ticker, quality_scores=_qs_dict)

    # New institutional-grade modules
    import traceback as _tb

    eq_analyzer = EarningsQualityAnalyzer()
    earnings_quality_result = None
    _eq_error = None
    try:
        earnings_quality_result = eq_analyzer.analyze(
            info=fetcher.info,
            income_stmt=income_stmt_full,
            balance_sheet=balance_sheet_full,
            cashflow=cashflow_stmt,
            earnings_history=earnings_history,
        )
    except Exception as e:
        _eq_error = f"EarningsQuality error: {e}\n{_tb.format_exc()}"

    ce_analyzer = CapitalEfficiencyAnalyzer()
    capital_efficiency_result = None
    _ce_error = None
    try:
        capital_efficiency_result = ce_analyzer.analyze(
            info=fetcher.info,
            income_stmt=income_stmt_full,
            balance_sheet=balance_sheet_full,
            cashflow=cashflow_stmt,
        )
    except Exception as e:
        _ce_error = f"CapitalEfficiency error: {e}\n{_tb.format_exc()}"

    te_analyzer = TechnicalEnhancedAnalyzer()
    tech_enhanced_result = None
    _te_error = None
    try:
        tech_enhanced_result = te_analyzer.analyze(
            ticker=ticker,
            history=history,
            info=fetcher.info,
        )
    except Exception as e:
        _te_error = f"TechnicalEnhanced error: {e}\n{_tb.format_exc()}"

    cc_analyzer = CreditConditionsAnalyzer()
    credit_conditions_result = None
    _cc_error = None
    try:
        credit_conditions_result = cc_analyzer.analyze()
    except Exception as e:
        _cc_error = f"CreditConditions error: {e}\n{_tb.format_exc()}"

    sm_analyzer = SectorMetricsAnalyzer()
    sector_metrics_result = None
    _sm_error = None
    try:
        sector_metrics_result = sm_analyzer.analyze(
            ticker=ticker,
            info=fetcher.info,
            income_stmt=income_stmt_full,
            balance_sheet=balance_sheet_full,
            cashflow=cashflow_stmt,
        )
    except Exception as e:
        _sm_error = f"SectorMetrics error: {e}\n{_tb.format_exc()}"

    _module_errors = {k: v for k, v in {
        "earnings_quality": _eq_error,
        "capital_efficiency": _ce_error,
        "tech_enhanced": _te_error,
        "credit_conditions": _cc_error,
        "sector_metrics": _sm_error,
    }.items() if v}

    return {
        "company_info": company_info,
        "fund_data": fund_data,
        "fund_result": fund_result,
        "tech_result": tech_result,
        "chart_df": chart_df,
        "recommendation": recommendation,
        "analyst_targets": analyst_targets,
        "upgrades_downgrades": upgrades_downgrades,
        "recommendations_summary": recommendations_summary,
        "institutional_holders": institutional_holders,
        "mutualfund_holders": mutualfund_holders,
        "major_holders": major_holders,
        "risk_profile": risk_profile,
        "governance": governance_result,
        "quality_scores": quality_scores,
        "dcf_result": dcf_result,
        "financial_trends": financial_trends,
        "event_calendar": event_calendar,
        "peer_comps_result": peer_comps_result,
        "options_result": options_result,
        "inst_risk": inst_risk_result,
        "alt_data": alt_data_bundle,
        "forensic": forensic_result,
        "earnings_quality": earnings_quality_result,
        "capital_efficiency": capital_efficiency_result,
        "tech_enhanced": tech_enhanced_result,
        "credit_conditions": credit_conditions_result,
        "sector_metrics": sector_metrics_result,
        "_module_errors": _module_errors,
    }


# ---------------------------------------------------------------------------
# Analyst & Investor charts
# ---------------------------------------------------------------------------
def create_price_target_chart(current_price: float, targets: dict) -> go.Figure:
    """Bullet-style chart showing current price vs analyst target range."""
    low = targets.get("low", 0)
    mean = targets.get("mean", 0)
    median = targets.get("median", 0)
    high = targets.get("high", 0)

    if not any([low, mean, high]):
        return None

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=[high - low], y=["Target"], base=[low],
        orientation="h", name="Target Range",
        marker_color="rgba(102,126,234,0.15)",
        hoverinfo="skip", showlegend=False,
        width=0.5,
    ))

    fig.add_trace(go.Scatter(
        x=[low], y=["Target"], mode="markers+text",
        marker=dict(color="#FF8A65", size=14, symbol="triangle-left"),
        text=[f"Low ${low:.0f}"], textposition="bottom center",
        name="Low", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=[high], y=["Target"], mode="markers+text",
        marker=dict(color="#69F0AE", size=14, symbol="triangle-right"),
        text=[f"High ${high:.0f}"], textposition="bottom center",
        name="High", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=[mean], y=["Target"], mode="markers+text",
        marker=dict(color="#42A5F5", size=16, symbol="diamond"),
        text=[f"Mean ${mean:.2f}"], textposition="top center",
        name="Mean Target",
    ))
    if median and median != mean:
        fig.add_trace(go.Scatter(
            x=[median], y=["Target"], mode="markers+text",
            marker=dict(color="#AB47BC", size=14, symbol="square"),
            text=[f"Median ${median:.0f}"], textposition="top center",
            name="Median Target",
        ))
    fig.add_trace(go.Scatter(
        x=[current_price], y=["Target"], mode="markers+text",
        marker=dict(color="#FFCA28", size=18, symbol="star"),
        text=[f"Current ${current_price:.2f}"], textposition="bottom center",
        name="Current Price",
    ))

    fig.update_layout(
        template="plotly_dark",
        height=180,
        margin=dict(l=10, r=10, t=30, b=40),
        title="Analyst Price Target Range",
        xaxis=dict(title="Price ($)"),
        yaxis=dict(visible=False),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.1, xanchor="center", x=0.5),
    )
    return fig


def create_recommendations_chart(rec_df: pd.DataFrame) -> go.Figure:
    """Stacked bar chart of analyst recommendation counts over time."""
    if rec_df is None or len(rec_df) == 0:
        return None

    labels = rec_df.get("period", rec_df.index)
    categories = [
        ("strongBuy", "Strong Buy", "#00C853"),
        ("buy", "Buy", "#69F0AE"),
        ("hold", "Hold", "#FFD54F"),
        ("sell", "Sell", "#FF8A65"),
        ("strongSell", "Strong Sell", "#FF1744"),
    ]

    fig = go.Figure()
    for col, name, color in categories:
        if col in rec_df.columns:
            fig.add_trace(go.Bar(
                x=labels, y=rec_df[col], name=name,
                marker_color=color, opacity=0.85,
            ))

    fig.update_layout(
        template="plotly_dark",
        barmode="stack",
        height=300,
        margin=dict(l=10, r=10, t=40, b=10),
        title="Analyst Recommendations Over Time",
        xaxis=dict(title="Period"),
        yaxis=dict(title="# of Analysts"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    return fig


def create_upgrades_downgrades_chart(ud_df: pd.DataFrame) -> go.Figure:
    """Scatter chart of analyst price targets by firm."""
    if ud_df is None or len(ud_df) == 0 or "currentPriceTarget" not in ud_df.columns:
        return None

    valid = ud_df[ud_df["currentPriceTarget"] > 0].copy()
    if len(valid) == 0:
        return None

    action_colors = {
        "up": "#00C853",
        "main": "#42A5F5",
        "reit": "#AB47BC",
        "down": "#FF1744",
        "init": "#FFA726",
    }

    colors = [action_colors.get(str(a).lower(), "#888") for a in valid.get("Action", ["main"] * len(valid))]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=valid.index if valid.index.name == "GradeDate" else list(range(len(valid))),
        y=valid["currentPriceTarget"],
        mode="markers+text",
        marker=dict(color=colors, size=12, line=dict(width=1, color="rgba(255,255,255,0.3)")),
        text=valid["Firm"],
        textposition="top center",
        textfont=dict(size=9, color="#ccc"),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Target: $%{y:.2f}<br>"
            "Date: %{x}<br>"
            "<extra></extra>"
        ),
    ))

    if "priorPriceTarget" in valid.columns:
        prior_valid = valid[valid["priorPriceTarget"] > 0]
        if len(prior_valid) > 0:
            fig.add_trace(go.Scatter(
                x=prior_valid.index if prior_valid.index.name == "GradeDate" else list(range(len(prior_valid))),
                y=prior_valid["priorPriceTarget"],
                mode="markers",
                marker=dict(color="rgba(255,255,255,0.3)", size=8, symbol="x"),
                name="Prior Target",
            ))

    fig.update_layout(
        template="plotly_dark",
        height=350,
        margin=dict(l=10, r=10, t=40, b=10),
        title="Recent Analyst Price Targets by Firm",
        yaxis=dict(title="Price Target ($)"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig


def create_holders_pie(major_df: pd.DataFrame) -> go.Figure:
    """Pie chart of ownership breakdown."""
    if major_df is None or len(major_df) == 0:
        return None

    labels_map = {
        "insidersPercentHeld": "Insiders",
        "institutionsPercentHeld": "Institutions",
    }
    labels = []
    values = []

    try:
        if "Breakdown" in major_df.columns and "Value" in major_df.columns:
            for _, row in major_df.iterrows():
                key = row["Breakdown"]
                if key in labels_map:
                    labels.append(labels_map[key])
                    values.append(float(row["Value"]) * 100)
        else:
            for key, label in labels_map.items():
                if key in major_df.index:
                    val = major_df.loc[key]
                    if hasattr(val, "iloc"):
                        val = val.iloc[0]
                    labels.append(label)
                    values.append(float(val) * 100)
    except Exception:
        return None

    if not values:
        return None

    retail = 100 - sum(values)
    if retail > 0:
        labels.append("Retail / Other")
        values.append(retail)

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        marker=dict(colors=["#667eea", "#AB47BC", "#42A5F5", "#FFA726"]),
        textinfo="label+percent",
        hole=0.45,
    ))
    fig.update_layout(
        template="plotly_dark",
        height=300,
        margin=dict(l=10, r=10, t=40, b=10),
        title="Ownership Breakdown",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig


def create_institutional_bar(inst_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of top institutional holders."""
    if inst_df is None or len(inst_df) == 0:
        return None

    df = inst_df.head(10).copy()
    holders = df["Holder"].tolist()[::-1]
    pcts = (df["pctHeld"].fillna(0) * 100).tolist()[::-1]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=holders, x=pcts,
        orientation="h",
        marker_color=[TICKER_COLORS[i % len(TICKER_COLORS)] for i in range(len(holders))],
        text=[f"{p:.2f}%" for p in pcts],
        textposition="auto",
    ))
    fig.update_layout(
        template="plotly_dark",
        height=max(280, len(holders) * 35 + 60),
        margin=dict(l=10, r=10, t=40, b=10),
        title="Top Institutional Holders (% of Shares)",
        xaxis=dict(title="% Held", ticksuffix="%"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ---------------------------------------------------------------------------
# Macro & Sector cached analysis
# ---------------------------------------------------------------------------
@st.cache_resource(ttl=1800, show_spinner=False)
def run_macro_analysis():
    analyzer = MacroAnalyzer()
    return analyzer.analyze()


@st.cache_resource(ttl=1800, show_spinner=False)
def run_sector_analysis(cycle_phase: str, rates_rising: bool):
    analyzer = SectorAnalyzer()
    return analyzer.analyze(cycle_phase=cycle_phase, rates_rising=rates_rising)


@st.cache_resource(ttl=1800, show_spinner=False)
def run_stock_screener(sector: str, max_stocks: int, min_mc: float, max_pe, max_de, min_roe, min_rg):
    screener = StockScreener()
    return screener.screen_sector(
        sector=sector, max_stocks=max_stocks,
        min_market_cap=min_mc, max_pe=max_pe,
        max_debt_equity=max_de, min_roe=min_roe,
        min_revenue_growth=min_rg,
    )


@st.cache_resource(ttl=900, show_spinner=False)
def run_news_analysis(ticker: str, company_name: str):
    analyzer = NewsSentimentAnalyzer()
    return analyzer.analyze(ticker, company_name)


# ---------------------------------------------------------------------------
# Macro & Sector charts
# ---------------------------------------------------------------------------
def create_yield_curve_chart(yield_curve: dict) -> go.Figure:
    points = yield_curve.get("points", {})
    if not points:
        return None
    labels = list(points.keys())
    values = list(points.values())

    status = yield_curve.get("status", "normal")
    color = "#FF1744" if status == "inverted" else "#FFD54F" if status == "flat" else "#00C853"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=labels, y=values, mode="lines+markers+text",
        line=dict(color=color, width=3),
        marker=dict(size=12, color=color),
        text=[f"{v:.2f}%" for v in values],
        textposition="top center",
        textfont=dict(color="white"),
    ))
    fig.update_layout(
        template="plotly_dark", height=300,
        margin=dict(l=10, r=10, t=40, b=10),
        title=f"US Treasury Yield Curve ({status.upper()})",
        yaxis=dict(title="Yield %", ticksuffix="%"),
        xaxis=dict(title="Maturity"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def create_macro_dashboard(indicators) -> go.Figure:
    categories = {}
    for ind in indicators:
        categories.setdefault(ind.category, []).append(ind)

    signal_colors = {"bullish": "#00C853", "neutral": "#FFD54F", "bearish": "#FF1744"}

    fig = go.Figure()
    y_labels = []
    x_vals = []
    colors = []

    for ind in indicators:
        y_labels.append(f"{ind.name}")
        score = {"bullish": 1, "neutral": 0, "bearish": -1}[ind.signal]
        x_vals.append(score)
        colors.append(signal_colors[ind.signal])

    fig.add_trace(go.Bar(
        y=y_labels[::-1], x=x_vals[::-1], orientation="h",
        marker_color=colors[::-1],
        text=[{1: "Bullish", 0: "Neutral", -1: "Bearish"}[v] for v in x_vals[::-1]],
        textposition="auto",
    ))

    fig.update_layout(
        template="plotly_dark",
        height=max(300, len(y_labels) * 40 + 60),
        margin=dict(l=10, r=10, t=40, b=10),
        title="Macro Indicator Signals",
        xaxis=dict(range=[-1.2, 1.2], tickvals=[-1, 0, 1], ticktext=["Bearish", "Neutral", "Bullish"]),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig


def create_sector_heatmap(sectors) -> go.Figure:
    if not sectors:
        return None

    names = [s.name for s in sectors]
    periods = ["1W", "1M", "3M", "6M"]
    data = []
    for s in sectors:
        row = [
            (s.change_1w or 0) * 100,
            (s.change_1m or 0) * 100,
            (s.change_3m or 0) * 100,
            (s.change_6m or 0) * 100,
        ]
        data.append(row)

    z = np.array(data)

    fig = go.Figure(go.Heatmap(
        z=z, x=periods, y=names,
        colorscale=[[0, "#FF1744"], [0.5, "#333"], [1, "#00C853"]],
        zmid=0,
        text=[[f"{v:.1f}%" for v in row] for row in z],
        texttemplate="%{text}",
        textfont=dict(size=11),
        colorbar=dict(title="% Change"),
    ))
    fig.update_layout(
        template="plotly_dark", height=max(350, len(names) * 35 + 80),
        margin=dict(l=10, r=10, t=40, b=10),
        title="Sector Performance Heatmap",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def create_sector_rotation_chart(sectors, cycle_phase: str) -> go.Figure:
    if not sectors:
        return None

    names = [s.name for s in sectors]
    momentum = [s.momentum_score for s in sectors]
    alignment = [s.cycle_alignment for s in sectors]
    rel_strength = [s.relative_strength for s in sectors]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=alignment, y=momentum,
        mode="markers+text",
        text=names,
        textposition="top center",
        textfont=dict(size=10, color="#ccc"),
        marker=dict(
            size=[max(8, abs(r) * 2 + 10) for r in rel_strength],
            color=momentum,
            colorscale=[[0, "#FF1744"], [0.5, "#FFD54F"], [1, "#00C853"]],
            showscale=True,
            colorbar=dict(title="Momentum"),
            line=dict(width=1, color="rgba(255,255,255,0.3)"),
        ),
        hovertemplate="<b>%{text}</b><br>Cycle Alignment: %{x:.2f}<br>Momentum: %{y:.3f}<extra></extra>",
    ))

    fig.add_vline(x=1.0, line_dash="dash", line_color="rgba(255,255,255,0.2)")
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.2)")

    fig.update_layout(
        template="plotly_dark", height=450,
        margin=dict(l=10, r=10, t=40, b=10),
        title=f"Sector Rotation Map — {cycle_phase} Phase",
        xaxis=dict(title="Cycle Alignment (higher = more favored)"),
        yaxis=dict(title="Momentum Score"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )

    fig.add_annotation(x=1.3, y=0.3, text="FAVORED", showarrow=False,
                       font=dict(color="rgba(0,200,83,0.3)", size=20))
    fig.add_annotation(x=0.7, y=-0.3, text="AVOID", showarrow=False,
                       font=dict(color="rgba(255,23,68,0.3)", size=20))
    return fig


def create_cycle_diagram(phase: str) -> str:
    """Return HTML for a business cycle visualization."""
    phases = ["Recovery", "Expansion", "Peak", "Contraction"]
    colors = {"Recovery": "#42A5F5", "Expansion": "#00C853", "Peak": "#FFA726", "Contraction": "#FF1744"}
    active_color = colors.get(phase, "#888")

    html = '<div style="display:flex;justify-content:center;gap:0;margin:1rem 0">'
    for p in phases:
        is_active = p == phase
        bg = colors[p] if is_active else "rgba(255,255,255,0.05)"
        border = f"2px solid {colors[p]}" if is_active else "1px solid rgba(255,255,255,0.1)"
        text_color = "white" if is_active else "#888"
        font_weight = "700" if is_active else "400"
        scale = "1.08" if is_active else "1"
        html += (
            f'<div style="flex:1;text-align:center;padding:1rem 0.5rem;background:{bg}33;'
            f'border:{border};font-weight:{font_weight};color:{text_color};'
            f'transform:scale({scale});transition:all 0.3s;'
            f'{"border-radius:12px 0 0 12px" if p == phases[0] else "border-radius:0 12px 12px 0" if p == phases[-1] else ""}">'
            f'<div style="font-size:1.1rem">{p}</div>'
            f'{"<div style=font-size:0.7rem;margin-top:4px>◆ WE ARE HERE</div>" if is_active else ""}'
            f'</div>'
        )
    html += '</div>'
    return html


# ---------------------------------------------------------------------------
# Risk management charts
# ---------------------------------------------------------------------------
def create_stop_loss_chart(entry_price: float, stop_losses, current_price: float = None) -> go.Figure:
    """Horizontal bullet chart of stop loss levels relative to entry."""
    if not stop_losses:
        return None

    methods = [s.method for s in stop_losses]
    prices = [s.stop_price for s in stop_losses]
    distances = [s.distance_pct for s in stop_losses]

    colors = [score_to_color(-d / 10) for d in distances]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=methods, x=prices, orientation="h",
        marker_color=colors, opacity=0.85,
        text=[f"${p:.2f} ({d:.1f}%)" for p, d in zip(prices, distances)],
        textposition="auto",
        name="Stop Level",
    ))

    fig.add_vline(x=entry_price, line_dash="solid", line_color="#FFCA28", line_width=2,
                  annotation_text=f"Entry ${entry_price:.2f}", annotation_position="top")

    x_min = min(prices) * 0.97
    x_max = entry_price * 1.03

    fig.update_layout(
        template="plotly_dark",
        height=max(250, len(methods) * 35 + 80),
        margin=dict(l=10, r=10, t=40, b=10),
        title="Stop Loss Levels",
        xaxis=dict(title="Price ($)", range=[x_min, x_max]),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig


def create_risk_reward_chart(entry_price: float, scenarios, primary_stop_price: float) -> go.Figure:
    """Visual risk/reward diagram."""
    if not scenarios:
        return None

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=[s.label for s in scenarios],
        y=[s.ratio for s in scenarios],
        marker_color=[
            "#00C853" if s.ratio >= 3 else "#69F0AE" if s.ratio >= 2 else "#FFD54F" if s.ratio >= 1 else "#FF8A65"
            for s in scenarios
        ],
        text=[f"{s.ratio:.1f}:1" for s in scenarios],
        textposition="auto",
        name="R:R Ratio",
    ))

    fig.add_hline(y=2, line_dash="dash", line_color="rgba(105,240,174,0.5)",
                  annotation_text="2:1 (Good)", annotation_position="right")
    fig.add_hline(y=3, line_dash="dash", line_color="rgba(0,200,83,0.5)",
                  annotation_text="3:1 (Excellent)", annotation_position="right")

    fig.update_layout(
        template="plotly_dark",
        height=320,
        margin=dict(l=10, r=10, t=40, b=10),
        title="Risk/Reward Ratios",
        yaxis=dict(title="Reward : Risk"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def create_position_size_chart(sizes) -> go.Figure:
    """Bar chart comparing position sizing methods."""
    if not sizes:
        return None

    methods = [s.method for s in sizes]
    shares = [s.shares for s in sizes]
    pcts = [s.pct_of_portfolio for s in sizes]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=methods, y=shares, name="Shares",
        marker_color=[TICKER_COLORS[i % len(TICKER_COLORS)] for i in range(len(methods))],
        text=[f"{s:,}" for s in shares],
        textposition="auto",
        opacity=0.85,
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=methods, y=pcts, name="% of Portfolio",
        mode="markers+lines+text",
        marker=dict(color="#FFCA28", size=10),
        line=dict(color="#FFCA28", width=2, dash="dot"),
        text=[f"{p:.1f}%" for p in pcts],
        textposition="top center",
        textfont=dict(color="#FFCA28"),
    ), secondary_y=True)

    fig.update_layout(
        template="plotly_dark",
        height=350,
        margin=dict(l=10, r=10, t=40, b=10),
        title="Position Sizing Comparison",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    fig.update_yaxes(title_text="Number of Shares", secondary_y=False)
    fig.update_yaxes(title_text="% of Portfolio", secondary_y=True)
    return fig


def create_stop_on_price_chart(df: pd.DataFrame, entry: float, stop_losses, trailing_stops: dict) -> go.Figure:
    """Price chart with stop loss levels drawn as horizontal lines."""
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Price", increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    ))

    fig.add_hline(y=entry, line_dash="solid", line_color="#FFCA28", line_width=2,
                  annotation_text=f"Entry ${entry:.2f}", annotation_position="right",
                  annotation_font_color="#FFCA28")

    stop_colors = ["#FF1744", "#FF5252", "#FF8A65", "#FFA726", "#FFD54F", "#AB47BC", "#42A5F5", "#26C6DA"]
    for i, sl in enumerate(stop_losses[:6]):
        color = stop_colors[i % len(stop_colors)]
        fig.add_hline(y=sl.stop_price, line_dash="dash", line_color=color, line_width=1,
                      annotation_text=f"{sl.method}: ${sl.stop_price:.2f}",
                      annotation_position="left",
                      annotation_font_color=color,
                      annotation_font_size=10)

    chandelier = trailing_stops.get("chandelier")
    if chandelier:
        fig.add_hline(y=chandelier["stop_price"], line_dash="dot", line_color="#66BB6A", line_width=1.5,
                      annotation_text=f"Chandelier: ${chandelier['stop_price']:.2f}",
                      annotation_position="right",
                      annotation_font_color="#66BB6A",
                      annotation_font_size=10)

    fig.update_layout(
        template="plotly_dark",
        height=450,
        margin=dict(l=10, r=10, t=40, b=10),
        title="Price Chart with Stop Loss Levels",
        xaxis_rangeslider_visible=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
def _enrich_sector_for_stock(sector_result, company_info):
    """Score the stock's sector in the rotation ranking."""
    if sector_result and sector_result.sectors:
        stock_sector = company_info.get("sector", "")
        return SectorAnalyzer().score_stock_sector(sector_result, stock_sector)
    return sector_result


# Render single-stock deep dive
# ---------------------------------------------------------------------------
def render_stock_detail(ticker: str, data: dict):
    company_info = data["company_info"]
    fund_data = data["fund_data"]
    fund_result = data["fund_result"]
    tech_result = data["tech_result"]
    chart_df = data["chart_df"]
    recommendation = data["recommendation"]
    governance = data.get("governance")

    # Prefetch macro / sector / news (cached)
    macro_result = None
    sector_result = None
    news_result = None
    try:
        macro_result = run_macro_analysis()
    except Exception:
        pass
    try:
        if macro_result and macro_result.cycle:
            yc = macro_result.yield_curve
            rates_rising = (yc.get("10y_change_3m") or 0) > 0
            sector_result = run_sector_analysis(macro_result.cycle.phase, rates_rising)
            sector_result = _enrich_sector_for_stock(sector_result, company_info)
    except Exception:
        pass
    try:
        news_result = run_news_analysis(ticker, company_info.get("name", ticker))
    except Exception:
        pass

    # Reasoning engine
    engine = ReasoningEngine()
    ci_for_engine = {**company_info, "ticker": ticker}
    verdict = engine.synthesize(
        company_info=ci_for_engine,
        fund_data=fund_data,
        fund_result=fund_result,
        tech_result=tech_result,
        recommendation=recommendation,
        governance=governance,
        macro_result=macro_result,
        sector_result=sector_result,
        news_result=news_result,
        risk_profile=data.get("risk_profile"),
        sr_levels=getattr(tech_result, 'support_resistance', []),
    )

    # Header metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    price = company_info.get("price")
    col1.metric("Price", f"${price:.2f}" if price else "N/A")
    col2.metric("Market Cap", format_large_number(company_info.get("market_cap")))
    w52h = company_info.get("52w_high")
    w52l = company_info.get("52w_low")
    col3.metric("52W Range", f"${w52l:.0f}–${w52h:.0f}" if w52l and w52h else "N/A")
    col4.metric("Beta", f"{company_info['beta']:.2f}" if company_info.get("beta") else "N/A")
    col5.metric("Avg Volume", f"{company_info['avg_volume']:,.0f}" if company_info.get("avg_volume") else "N/A")
    st.divider()

    # ── 4-Gate badges ──
    qs = data.get("quality_scores")
    dcf = data.get("dcf_result")
    te = data.get("tech_enhanced")

    # Gate 1: Business Quality
    pf = getattr(qs, "piotroski_f", None) if qs else None
    gov_risk = getattr(governance, "overall_risk_level", None) if governance else None
    biz_pass = (
        (pf is None or pf >= 5) and
        fund_result.overall_score >= 0 and
        gov_risk not in ("CRITICAL",)
    )
    biz_label = "PASS" if biz_pass else "FAIL"
    biz_color = "#00C853" if biz_pass else "#FF1744"

    # Gate 2: Price (DCF)
    mos = getattr(dcf, "margin_of_safety", None) if dcf and not getattr(dcf, "error", None) else None
    if mos is not None and mos > 0.20:
        price_label, price_color = "ATTRACTIVE", "#00C853"
    elif mos is not None and mos > -0.10:
        price_label, price_color = "FAIR", "#FFD54F"
    else:
        price_label, price_color = "STRETCHED", "#FF1744"

    # Gate 3: Timing (technical)
    mom_signal = None
    if te:
        try:
            mom_signal = te.momentum.momentum_signal
        except Exception:
            pass
    timing_ok = (
        tech_result.overall_score >= 0.05 and
        mom_signal in (None, "BULLISH", "NEUTRAL")
    )
    timing_label = "FAVORABLE" if timing_ok else "WAIT"
    timing_color = "#00C853" if timing_ok else "#FFA726"

    # Gate 4: Trade setup (risk profile has stop and target)
    rp = data.get("risk_profile")
    has_stop = rp and bool(getattr(rp, "stop_losses", []))
    has_target = rp and bool(getattr(rp, "risk_reward", []))
    trade_label = "READY" if (has_stop or has_target) else "SETUP NEEDED"
    trade_color = "#00C853" if (has_stop or has_target) else "#FFA726"

    gate_cols = st.columns(4)
    for col, gate_name, label, color, icon in [
        (gate_cols[0], "BUSINESS QUALITY", biz_label, biz_color, "🏢"),
        (gate_cols[1], "FAIR PRICE", price_label, price_color, "💰"),
        (gate_cols[2], "GOOD TIMING", timing_label, timing_color, "⏱️"),
        (gate_cols[3], "TRADE SETUP", trade_label, trade_color, "🛡️"),
    ]:
        col.markdown(
            f"<div style='text-align:center;padding:0.7rem;background:{color}11;"
            f"border:1px solid {color}44;border-radius:10px'>"
            f"<div style='font-size:1.2rem'>{icon}</div>"
            f"<div style='color:#888;font-size:0.65rem;text-transform:uppercase;letter-spacing:0.08em'>{gate_name}</div>"
            f"<div style='color:{color};font-size:0.95rem;font-weight:700;margin-top:0.2rem'>{label}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    st.markdown("<br>", unsafe_allow_html=True)

    # 5 Tabs
    tab_verdict, tab_business, tab_valuation, tab_timing, tab_trade = st.tabs([
        "🎯 Verdict", "🏢 Business Quality", "💰 Valuation", "⏱️ Timing", "🛡️ Trade Plan"
    ])

    # ── TAB 1: VERDICT ──
    with tab_verdict:
        vc = verdict.score_color
        action_emoji = {"STRONG BUY": "🟢", "BUY": "🟩", "HOLD": "🟡", "SELL": "🟧", "STRONG SELL": "🔴"}.get(verdict.action, "⚪")
        st.markdown(
            f"<div style='background:linear-gradient(135deg,{vc}22,{vc}08);"
            f"border:2px solid {vc};border-radius:14px;padding:1.5rem 1.8rem;margin-bottom:1rem'>"
            f"<div style='display:flex;align-items:center;gap:1.2rem;flex-wrap:wrap'>"
            f"<div style='font-size:3rem'>{action_emoji}</div>"
            f"<div style='flex:1;min-width:200px'>"
            f"<div style='color:{vc};font-size:2rem;font-weight:800;letter-spacing:1px'>{verdict.action}</div>"
            f"<div style='color:#aaa;font-size:0.9rem;margin-top:0.2rem'>Confidence: <b>{verdict.confidence}</b> · "
            f"Composite Score: <b>{verdict.composite_score:+.3f}</b></div>"
            f"</div>"
            f"<div style='text-align:right;min-width:120px'>"
            f"<div style='color:{vc};font-size:2.4rem;font-weight:700'>{verdict.composite_score:+.2f}</div>"
            f"<div style='color:#666;font-size:0.75rem'>MULTI-FACTOR SCORE</div>"
            f"</div></div>"
            f"<div style='color:#ccc;font-size:0.92rem;margin-top:1rem;line-height:1.6'>{verdict.thesis}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown("##### Reasoning Chain — 7-Pillar Analysis")
        signal_colors = {"BULLISH": "#00C853", "NEUTRAL": "#FFD54F", "CAUTION": "#FFA726", "BEARISH": "#FF1744"}
        signal_icons = {"BULLISH": "🟢", "NEUTRAL": "🟡", "CAUTION": "🟠", "BEARISH": "🔴"}
        pillar_icons = {"Governance": "🏛️", "Macro": "🌍", "Sector": "📊", "Fundamental": "💰", "Technical": "📈", "Sentiment": "📰", "Risk": "🛡️"}
        pillar_accent = {"Governance": "#AB47BC", "Macro": "#26A69A", "Sector": "#42A5F5", "Fundamental": "#FFA726", "Technical": "#667eea", "Sentiment": "#EF5350", "Risk": "#FFCA28"}
        pillar_bg = {
            "Governance": "linear-gradient(135deg, rgba(171,71,188,0.10), rgba(171,71,188,0.03))",
            "Macro": "linear-gradient(135deg, rgba(38,166,154,0.10), rgba(38,166,154,0.03))",
            "Sector": "linear-gradient(135deg, rgba(66,165,245,0.10), rgba(66,165,245,0.03))",
            "Fundamental": "linear-gradient(135deg, rgba(255,167,38,0.10), rgba(255,167,38,0.03))",
            "Technical": "linear-gradient(135deg, rgba(102,126,234,0.10), rgba(102,126,234,0.03))",
            "Sentiment": "linear-gradient(135deg, rgba(239,83,80,0.10), rgba(239,83,80,0.03))",
            "Risk": "linear-gradient(135deg, rgba(255,202,40,0.10), rgba(255,202,40,0.03))",
        }
        for step in verdict.reasoning_chain:
            sc = signal_colors.get(step.signal, "#888")
            si = signal_icons.get(step.signal, "⚪")
            pi = pillar_icons.get(step.pillar, "•")
            bg = pillar_bg.get(step.pillar, "rgba(28,31,48,0.5)")
            accent = pillar_accent.get(step.pillar, "#888")
            bar_pct = (step.score + 1) / 2 * 100
            bar_color = "#00C853" if step.score > 0.15 else "#FF1744" if step.score < -0.15 else "#FFD54F"
            st.markdown(
                f"<div style='padding:0.7rem 0.9rem;background:{bg};"
                f"border-left:4px solid {accent};border-radius:0 12px 12px 0;margin-bottom:0.45rem;"
                f"border:1px solid {accent}18;border-left:4px solid {accent}'>"
                f"<div style='display:flex;align-items:center;gap:0.6rem'>"
                f"<span style='font-size:1.15rem'>{pi}</span>"
                f"<span style='color:{accent};font-weight:700;flex:1;font-size:0.95rem'>{step.pillar}</span>"
                f"<span style='font-size:0.85rem'>{si} <span style='color:{sc};font-weight:600'>{step.signal}</span></span>"
                f"<span style='color:{sc};font-weight:700;width:50px;text-align:right'>{step.score:+.2f}</span>"
                f"</div>"
                f"<div style='color:#e0e0e0;font-size:0.85rem;margin:0.35rem 0 0.3rem 1.8rem;font-weight:600'>{step.headline}</div>"
                f"<div style='color:#aaa;font-size:0.8rem;margin:0 0 0.4rem 1.8rem;line-height:1.5'>{step.detail}</div>"
                f"<div style='background:rgba(255,255,255,0.06);height:5px;border-radius:3px;margin:0 0 0 1.8rem'>"
                f"<div style='background:{bar_color};height:5px;border-radius:3px;width:{bar_pct:.0f}%;transition:width 0.3s'></div></div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        bb_col, ca_col = st.columns(2)
        with bb_col:
            st.markdown("##### Bull Case")
            st.markdown(f"<div style='padding:0.7rem;background:rgba(0,200,83,0.06);border:1px solid rgba(0,200,83,0.2);border-radius:10px;color:#a5d6a7;font-size:0.88rem;line-height:1.6'>{verdict.bull_case}</div>", unsafe_allow_html=True)
            st.markdown("##### Bear Case")
            st.markdown(f"<div style='padding:0.7rem;background:rgba(255,23,68,0.06);border:1px solid rgba(255,23,68,0.2);border-radius:10px;color:#ef9a9a;font-size:0.88rem;line-height:1.6'>{verdict.bear_case}</div>", unsafe_allow_html=True)
        with ca_col:
            if verdict.catalysts:
                st.markdown("##### Catalysts to Watch")
                for cat in verdict.catalysts:
                    st.markdown(f"<div style='padding:0.35rem 0.6rem;background:rgba(102,126,234,0.08);border-left:3px solid #667eea;border-radius:0 6px 6px 0;margin-bottom:0.3rem;color:#b0bec5;font-size:0.85rem'>⚡ {cat}</div>", unsafe_allow_html=True)
            if verdict.key_risks:
                st.markdown("##### Key Risks")
                for risk in verdict.key_risks:
                    st.markdown(f"<div style='padding:0.35rem 0.6rem;background:rgba(255,87,34,0.08);border-left:3px solid #FF5722;border-radius:0 6px 6px 0;margin-bottom:0.3rem;color:#ffab91;font-size:0.85rem'>⚠️ {risk}</div>", unsafe_allow_html=True)
        if verdict.action_plan:
            st.markdown("##### Action Plan")
            st.markdown(f"<div style='padding:0.8rem 1rem;background:linear-gradient(135deg,rgba(102,126,234,0.08),rgba(118,75,162,0.08));border:1px solid rgba(102,126,234,0.25);border-radius:10px;color:#ccc;font-size:0.88rem;line-height:1.7'>{verdict.action_plan}</div>", unsafe_allow_html=True)

        st.divider()
        g1, g2, g3 = st.columns(3)
        with g1:
            st.plotly_chart(create_gauge(fund_result.overall_score, "Fundamental"), use_container_width=True)
        with g2:
            st.plotly_chart(create_gauge(tech_result.overall_score, "Technical"), use_container_width=True)
        with g3:
            st.plotly_chart(create_gauge(verdict.composite_score, "Multi-Factor"), use_container_width=True)

        # Show any module errors so they're not silently hidden
        module_errors = data.get("_module_errors", {})
        if module_errors:
            with st.expander(f"⚠️ {len(module_errors)} module error(s) — click to expand", expanded=False):
                for mod, err in module_errors.items():
                    st.error(f"**{mod}**")
                    st.code(err, language="text")

    # ── TAB 2: BUSINESS QUALITY ──
    with tab_business:
        st.markdown("##### Valuation Metrics")
        vc4 = st.columns(4)
        vc4[0].metric("P/E (TTM)", f"{fund_data['pe_trailing']:.1f}" if fund_data.get('pe_trailing') else "N/A",
                      help="Price-to-Earnings (trailing 12 months). Lower = cheaper relative to current earnings. S&P 500 average ~20-25x.")
        vc4[1].metric("Forward P/E", f"{fund_data['pe_forward']:.1f}" if fund_data.get('pe_forward') else "N/A",
                      help="Price relative to next year's estimated earnings. Forward P/E < TTM P/E suggests earnings growth expected.")
        vc4[2].metric("PEG Ratio", f"{fund_data['peg_ratio']:.2f}" if fund_data.get('peg_ratio') else "N/A",
                      help="P/E divided by growth rate. <1 = potentially undervalued relative to growth. >2 = expensive growth.")
        vc4[3].metric("EV/EBITDA", f"{fund_data['ev_ebitda']:.1f}" if fund_data.get('ev_ebitda') else "N/A",
                      help="Enterprise Value / EBITDA. The most reliable acquisition metric. <10x = cheap, >20x = expensive.")

        vc4b = st.columns(4)
        vc4b[0].metric("P/B Ratio", f"{fund_data['pb_ratio']:.2f}" if fund_data.get('pb_ratio') else "N/A",
                       help="Price-to-Book. <1 = trading below asset value. Useful for banks/industrials, less so for asset-light tech.")
        vc4b[1].metric("P/S Ratio", f"{fund_data['ps_ratio']:.2f}" if fund_data.get('ps_ratio') else "N/A",
                       help="Price-to-Sales. Pre-profit companies often valued on revenue. <2x = cheap for tech, varies by sector.")
        vc4b[2].metric("EPS (TTM)", f"${fund_data['eps_trailing']:.2f}" if fund_data.get('eps_trailing') else "N/A")
        vc4b[3].metric("Book Value", f"${fund_data['book_value']:.2f}" if fund_data.get('book_value') else "N/A")

        st.divider()
        st.markdown("##### Profitability")
        pc5 = st.columns(5)
        pc5[0].metric("ROE", format_pct(fund_data.get('roe'), mult100=True),
                      help="Return on Equity. How much profit per dollar of shareholder equity. >15% = strong. Buffett target: >20%.")
        pc5[1].metric("ROA", format_pct(fund_data.get('roa'), mult100=True),
                      help="Return on Assets. Profit relative to total assets. >5% is good. Reflects asset efficiency.")
        pc5[2].metric("Gross Margin", format_pct(fund_data.get('gross_margin'), mult100=True),
                      help="Revenue minus cost of goods sold. Higher = more pricing power. Software: 60-80%, retail: 20-30%.")
        pc5[3].metric("Op. Margin", format_pct(fund_data.get('operating_margin'), mult100=True),
                      help="Operating income / revenue. Shows profitability before interest and taxes. Higher = more efficient.")
        pc5[4].metric("Net Margin", format_pct(fund_data.get('net_margin'), mult100=True),
                      help="Bottom-line profit as % of revenue. Wide margins = durable competitive advantage.")

        st.divider()
        st.markdown("##### Growth & Financial Health")
        gh = st.columns(4)
        gh[0].metric("Revenue Growth", format_pct(fund_data.get('revenue_growth'), mult100=True),
                     help="YoY revenue increase. >20% = high growth, 10-20% = healthy, <5% = mature/slow.")
        gh[1].metric("Earnings Growth", format_pct(fund_data.get('earnings_growth'), mult100=True))
        de = fund_data.get('debt_to_equity')
        gh[2].metric("Debt/Equity", f"{de:.1f}" if de else "N/A",
                     help="Total debt / shareholder equity. <1 = conservative, >2 = leveraged. Context matters (utilities can be higher).")
        gh[3].metric("Free Cash Flow", format_large_number(fund_data.get('free_cashflow')),
                     help="Free Cash Flow = Operating cash minus capex. The 'real' earnings — harder to manipulate than net income.")

        trends = data.get("financial_trends", {})
        if trends:
            st.divider()
            st.markdown("##### Financial Trends (YoY)")
            tc4 = st.columns(4)
            def trend_metric(col, label, val):
                if val is not None:
                    delta_str = f"{val:+.1%}"
                    col.metric(label, delta_str, delta=delta_str)
                else:
                    col.metric(label, "N/A")
            trend_metric(tc4[0], "Revenue YoY", trends.get("revenue_yoy"))
            trend_metric(tc4[1], "Net Income YoY", trends.get("net_income_yoy"))
            trend_metric(tc4[2], "FCF YoY", trends.get("fcf_yoy"))
            trend_metric(tc4[3], "Op. Income YoY", trends.get("operating_income_yoy"))

            rev_series = trends.get("revenue_series", {})
            ni_series = trends.get("net_income_series", {})
            if rev_series:
                dates = sorted(rev_series.keys())
                trend_fig = go.Figure()
                trend_fig.add_trace(go.Bar(x=dates, y=[rev_series[d]/1e9 for d in dates], name="Revenue ($B)", marker_color="#42A5F5"))
                if ni_series:
                    ni_dates = sorted(ni_series.keys())
                    trend_fig.add_trace(go.Bar(x=ni_dates, y=[ni_series[d]/1e9 for d in ni_dates], name="Net Income ($B)", marker_color="#66BB6A"))
                trend_fig.update_layout(template="plotly_dark", height=280, barmode="group", margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", yaxis_title="$B")
                st.plotly_chart(trend_fig, use_container_width=True)

        st.divider()
        st.markdown("##### Governance & Red Flags")
        if governance:
            gov_risk_level = getattr(governance, "overall_risk_level", None)
            gov_score = getattr(governance, "risk_score", None)
            gov_color = "#FF1744" if gov_risk_level == "CRITICAL" else "#FFA726" if gov_risk_level == "HIGH" else "#FFD54F" if gov_risk_level == "MEDIUM" else "#00C853"
            g_c1, g_c2 = st.columns(2)
            g_c1.markdown(f"<div style='text-align:center;padding:1rem;background:{gov_color}18;border:1px solid {gov_color}66;border-radius:10px'>"
                          f"<div style='color:{gov_color};font-size:1.8rem;font-weight:800'>{gov_risk_level or 'N/A'}</div>"
                          f"<div style='color:#ccc;font-weight:600'>Governance Risk</div>"
                          f"<div style='color:#888;font-size:0.78rem'>Score: {gov_score or 'N/A'}</div>"
                          f"</div>", unsafe_allow_html=True)
            red_flags = getattr(governance, "red_flags", []) or []
            with g_c2:
                if red_flags:
                    st.markdown("**Red Flags**")
                    for flag in red_flags[:6]:
                        flag_text = str(flag)
                        st.markdown(f"<div style='padding:0.3rem 0.6rem;background:rgba(255,23,68,0.08);border-left:3px solid #FF1744;border-radius:0 6px 6px 0;margin-bottom:0.25rem;color:#ffab91;font-size:0.83rem'>🚩 {flag_text}</div>", unsafe_allow_html=True)
                else:
                    st.success("No major governance red flags detected.")
        else:
            st.info("Governance data unavailable.")

        st.divider()
        st.markdown("##### Earnings Quality")
        eq = data.get("earnings_quality")
        if eq:
            eq_color = "#00C853" if eq.overall_score > 0.1 else "#FFA726" if eq.overall_score > -0.1 else "#FF1744"
            eq_cols = st.columns(4)
            eq_cols[0].metric("FCF Conversion", f"{eq.fcf_conversion_rate:.0%}" if eq.fcf_conversion_rate is not None else "N/A")
            eq_cols[1].metric("Sloan Accruals", f"{eq.sloan_accruals_ratio:.1%}" if eq.sloan_accruals_ratio is not None else "N/A")
            eq_cols[2].metric("Avg EPS Surprise", f"{eq.avg_surprise_pct:+.1f}%" if eq.avg_surprise_pct is not None else "N/A")
            eq_cols[3].metric("SBC % FCF", f"{eq.sbc_pct_fcf:.1%}" if eq.sbc_pct_fcf is not None else "N/A")
            st.markdown(f"<div style='padding:0.6rem 1rem;background:{eq_color}18;border:1px solid {eq_color}55;border-radius:8px;margin:0.5rem 0'>"
                        f"<span style='color:{eq_color};font-weight:700'>{eq.signal}</span>  ·  Score: {eq.overall_score:+.3f}  ·  {eq.summary}</div>", unsafe_allow_html=True)
            if eq.quality_flags:
                for flag in eq.quality_flags:
                    st.caption(f"⚠️ {flag}")
        else:
            st.info("Earnings quality data unavailable.")

        st.divider()
        st.markdown("##### Capital Efficiency")
        ce = data.get("capital_efficiency")
        if ce:
            ce_color = "#00C853" if ce.overall_score > 0.1 else "#FFA726" if ce.overall_score > -0.1 else "#FF1744"
            wc = getattr(ce, "working_capital", None)
            cp = getattr(ce, "capex", None)
            cr = getattr(ce, "capital_returns", None)
            ce_cols = st.columns(4)
            if wc:
                ce_cols[0].metric("Cash Conv. Cycle", f"{wc.ccc:.0f}d" if wc.ccc is not None else "N/A")
                ce_cols[1].metric("DSO", f"{wc.dso:.0f}d" if wc.dso is not None else "N/A")
            if cp:
                ce_cols[2].metric("Capex/Revenue", f"{cp.capex_to_revenue:.1%}" if cp.capex_to_revenue is not None else "N/A")
            if cr:
                ce_cols[3].metric("Buyback Yield", f"{cr.buyback_yield:.1%}" if cr.buyback_yield is not None else "N/A")
            st.markdown(f"<div style='padding:0.6rem 1rem;background:{ce_color}18;border:1px solid {ce_color}55;border-radius:8px;margin:0.5rem 0'>"
                        f"<span style='color:{ce_color};font-weight:700'>{ce.signal}</span>  ·  Score: {ce.overall_score:+.3f}  ·  {ce.summary}</div>", unsafe_allow_html=True)
        else:
            st.info("Capital efficiency data unavailable.")

    # ── TAB 3: VALUATION ──
    with tab_valuation:
        dcf = data.get("dcf_result")
        qs = data.get("quality_scores")

        st.markdown("##### DCF Intrinsic Value (3-Stage Model)")
        if dcf and dcf.intrinsic_value and not getattr(dcf, "error", None):
            price_now = dcf.current_price or company_info.get("price", 0)
            upside = dcf.upside_pct or 0
            mos_val = dcf.margin_of_safety or 0
            iv_color = "#00C853" if mos_val > 0.10 else "#FFA726" if mos_val > -0.10 else "#FF1744"
            label_color = "#00C853" if "Under" in str(dcf.valuation_label) else "#FF1744" if "Over" in str(dcf.valuation_label) else "#FFA726"
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Intrinsic Value", f"${dcf.intrinsic_value:.2f}",
                      help="DCF estimate of fair value per share based on projected free cash flows discounted at WACC.")
            d2.metric("Current Price", f"${price_now:.2f}" if price_now else "N/A")
            d3.metric("Upside / Downside", f"{upside:+.1%}" if dcf.upside_pct is not None else "N/A",
                      delta=f"{upside:+.1%}" if dcf.upside_pct is not None else None,
                      help="(Intrinsic Value - Current Price) / Current Price. Positive = potentially undervalued.")
            d4.metric("Margin of Safety", f"{mos_val:.1%}" if dcf.margin_of_safety is not None else "N/A",
                      help="Buffer between price and intrinsic value. Buffett's rule: buy only at >30% margin of safety.")
            st.markdown(
                f"<div style='padding:0.6rem 1rem;background:{iv_color}18;border:1px solid {iv_color}66;border-radius:8px;margin:0.5rem 0'>"
                f"<span style='color:{label_color};font-weight:700'>{dcf.valuation_label}</span>"
                f"  —  FCF Base: {format_large_number(dcf.fcf_base)}  ·  WACC: {dcf.wacc*100:.1f}%  ·  "
                f"Stage 1 Growth: {dcf.growth_stage1*100:.1f}%  ·  Stage 2 Growth: {dcf.growth_stage2*100:.1f}%  ·  "
                f"Terminal Growth: {dcf.terminal_growth*100:.1f}%</div>",
                unsafe_allow_html=True,
            )
            if dcf.sensitivity:
                st.markdown("**Sensitivity Analysis** — Intrinsic Value by WACC and Growth Rate")
                wacc_ds = sorted(set(k[0] for k in dcf.sensitivity))
                grow_ds = sorted(set(k[1] for k in dcf.sensitivity))
                z_vals = [[dcf.sensitivity.get((wd, gd)) for gd in grow_ds] for wd in wacc_ds]
                z_clean = [[v if v is not None else 0 for v in row] for row in z_vals]
                sen_fig = go.Figure(go.Heatmap(z=z_clean, x=[f"Growth {g:+.1f}%" for g in grow_ds], y=[f"WACC {w:+.1f}%" for w in wacc_ds], colorscale="RdYlGn", text=[[f"${v:.0f}" if v else "N/A" for v in row] for row in z_clean], texttemplate="%{text}", showscale=True))
                sen_fig.update_layout(template="plotly_dark", height=280, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)", title=f"Current Price: ${price_now:.2f}" if price_now else "")
                st.plotly_chart(sen_fig, use_container_width=True)
        elif dcf and getattr(dcf, "error", None):
            st.info(f"DCF not computed: {dcf.error}")
        else:
            st.info("DCF valuation data unavailable.")

        st.divider()
        st.markdown("##### Quality & Distress Scores")
        if qs:
            q1, q2, q3 = st.columns(3)
            pf = qs.piotroski_f
            pf_color = "#00C853" if pf is not None and pf >= 7 else "#FFA726" if pf is not None and pf >= 4 else "#FF1744"
            q1.markdown(f"<div style='text-align:center;padding:1rem;background:{pf_color}18;border:1px solid {pf_color}66;border-radius:10px'>"
                        f"<div style='color:{pf_color};font-size:2.5rem;font-weight:800'>{pf if pf is not None else 'N/A'}<span style='font-size:1rem'>/9</span></div>"
                        f"<div style='color:#ccc;font-weight:600'>Piotroski F-Score</div>"
                        f"<div style='color:#888;font-size:0.78rem'>{qs.piotroski_label}</div></div>", unsafe_allow_html=True)
            az = qs.altman_z
            az_color = "#00C853" if az is not None and az > 2.99 else "#FFA726" if az is not None and az > 1.81 else "#FF1744"
            q2.markdown(f"<div style='text-align:center;padding:1rem;background:{az_color}18;border:1px solid {az_color}66;border-radius:10px'>"
                        f"<div style='color:{az_color};font-size:2.5rem;font-weight:800'>{f'{az:.2f}' if az is not None else 'N/A'}</div>"
                        f"<div style='color:#ccc;font-weight:600'>Altman Z-Score</div>"
                        f"<div style='color:#888;font-size:0.78rem'>{qs.altman_zone or 'N/A'}</div></div>", unsafe_allow_html=True)
            bm = qs.beneish_m
            bm_color = "#FF1744" if qs.beneish_flag else "#00C853"
            bm_label = "⚠️ Manipulation Risk" if qs.beneish_flag else "✅ Low Manipulation Risk"
            q3.markdown(f"<div style='text-align:center;padding:1rem;background:{bm_color}18;border:1px solid {bm_color}66;border-radius:10px'>"
                        f"<div style='color:{bm_color};font-size:2.5rem;font-weight:800'>{f'{bm:.2f}' if bm is not None else 'N/A'}</div>"
                        f"<div style='color:#ccc;font-weight:600'>Beneish M-Score</div>"
                        f"<div style='color:#888;font-size:0.78rem'>{bm_label}</div></div>", unsafe_allow_html=True)

        st.divider()
        st.markdown("##### Sector-Specific Multiples")
        sm = data.get("sector_metrics")
        if sm:
            sm_color = "#00C853" if sm.overall_score > 0.1 else "#FFA726" if sm.overall_score > -0.1 else "#FF1744"
            u = getattr(sm, "universal", None)
            if u:
                sm_cols = st.columns(3)
                sm_cols[0].metric("EV/Revenue", f"{u.ev_revenue:.1f}x" if u.ev_revenue is not None else "N/A")
                sm_cols[1].metric("EV/EBIT", f"{u.ev_ebit:.1f}x" if u.ev_ebit is not None else "N/A")
                sm_cols[2].metric("Total Sh. Yield", f"{u.total_shareholder_yield:.1%}" if u.total_shareholder_yield is not None else "N/A")
            st.markdown(f"<div style='padding:0.6rem 1rem;background:{sm_color}18;border:1px solid {sm_color}55;border-radius:8px;margin:0.5rem 0'>"
                        f"<span style='color:{sm_color};font-weight:700'>{sm.signal}</span>  ·  Module: {sm.applicable_module}  ·  {sm.summary}</div>", unsafe_allow_html=True)
            if sm.quality_flags:
                for flag in sm.quality_flags:
                    st.caption(f"⚠️ {flag}")
        else:
            st.info("Sector multiples data unavailable.")

        st.divider()
        peer_comps_result = data.get("peer_comps_result")
        st.markdown("##### Peer Comparable Analysis")
        if peer_comps_result and peer_comps_result.peers:
            peer_rows = []
            for p in peer_comps_result.peers:
                peer_rows.append({
                    "Ticker": p.ticker,
                    "Company": p.name,
                    "Price": f"${p.price:.2f}" if p.price else "N/A",
                    "P/E": f"{p.pe_trailing:.1f}" if p.pe_trailing else "N/A",
                    "P/S": f"{p.ps_ratio:.2f}" if p.ps_ratio else "N/A",
                    "P/B": f"{p.pb_ratio:.2f}" if p.pb_ratio else "N/A",
                    "EV/EBITDA": f"{p.ev_ebitda:.1f}" if p.ev_ebitda else "N/A",
                    "Rev. Growth": f"{p.revenue_growth*100:.1f}%" if p.revenue_growth else "N/A",
                    "Net Margin": f"{p.net_margin*100:.1f}%" if p.net_margin else "N/A",
                    "Market Cap": format_large_number(p.market_cap),
                })
            peer_df = pd.DataFrame(peer_rows)
            st.dataframe(peer_df, use_container_width=True, hide_index=True)
            if peer_comps_result.error:
                st.caption(f"⚠️ {peer_comps_result.error}")

            with st.expander("🔬 Full Peer Analysis — Percentile Ranks, Medians & Chart", expanded=False):
                # Percentile rank cards
                if peer_comps_result.percentile_ranks:
                    st.markdown("**Percentile Ranks vs Peers** — *higher is better (valuation multiples adjusted: lower P/E = better rank)*")
                    from peer_comps import MULTIPLES as _MULTIPLES
                    rank_label_map = {k: lbl for k, lbl in _MULTIPLES}
                    rank_cols = st.columns(min(len(peer_comps_result.percentile_ranks), 5))
                    for ci2, (field_k, pct_val) in enumerate(peer_comps_result.percentile_ranks.items()):
                        card_color = "#00C853" if pct_val >= 60 else "#FFA726" if pct_val >= 40 else "#FF1744"
                        rank_cols[ci2 % len(rank_cols)].markdown(
                            f"<div style='text-align:center;padding:0.7rem 0.4rem;background:{card_color}18;"
                            f"border:1px solid {card_color}55;border-radius:8px;margin-bottom:0.3rem'>"
                            f"<div style='color:{card_color};font-size:1.5rem;font-weight:800'>{pct_val:.0f}</div>"
                            f"<div style='color:#888;font-size:0.7rem;text-transform:uppercase'>{rank_label_map.get(field_k, field_k)}</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                # Full multiples table with subject row highlighted
                st.markdown("**Full Multiples Table**")
                def _fmt_opt(v, fmt, mult=1, prefix="", suffix=""):
                    if v is None:
                        return "N/A"
                    try:
                        return f"{prefix}{v*mult:{fmt}}{suffix}"
                    except Exception:
                        return "N/A"

                full_rows = []
                for p in peer_comps_result.peers:
                    full_rows.append({
                        "Ticker": ("★ " if p.is_subject else "") + p.ticker,
                        "Company": p.name,
                        "Price": _fmt_opt(p.price, ".2f", prefix="$"),
                        "P/E": _fmt_opt(p.pe_trailing, ".1f"),
                        "Fwd P/E": _fmt_opt(p.pe_forward, ".1f"),
                        "EV/EBITDA": _fmt_opt(p.ev_ebitda, ".1f"),
                        "P/S": _fmt_opt(p.ps_ratio, ".2f"),
                        "P/B": _fmt_opt(p.pb_ratio, ".2f"),
                        "PEG": _fmt_opt(p.peg_ratio, ".2f"),
                        "ROE": _fmt_opt(p.roe, ".1f", mult=100, suffix="%"),
                        "Net Margin": _fmt_opt(p.net_margin, ".1f", mult=100, suffix="%"),
                        "Rev Growth": _fmt_opt(p.revenue_growth, ".1f", mult=100, suffix="%"),
                        "D/E": _fmt_opt(p.debt_to_equity, ".1f"),
                        "_is_subject": p.is_subject,
                    })
                full_df = pd.DataFrame(full_rows)

                def _highlight_subject(row):
                    if row.get("_is_subject", False):
                        return ["background-color: rgba(66,165,245,0.15); color: #90CAF9; font-weight:600"] * (len(row) - 1) + [""]
                    return [""] * len(row)

                display_full = full_df.drop(columns=["_is_subject"])
                st.dataframe(
                    display_full.style.apply(_highlight_subject, axis=1, subset=display_full.columns),
                    use_container_width=True,
                    hide_index=True,
                )

                # Sector medians vs subject
                if peer_comps_result.medians:
                    st.markdown("**Sector Medians vs Subject**")
                    subj_peer = next((p for p in peer_comps_result.peers if p.is_subject), None)
                    med_cols = st.columns(min(len(peer_comps_result.medians), 5))
                    for mi, (fk, med_val) in enumerate(peer_comps_result.medians.items()):
                        lbl = rank_label_map.get(fk, fk)
                        subj_v = getattr(subj_peer, fk, None) if subj_peer else None
                        delta_str = None
                        if subj_v is not None:
                            # percentage-based fields
                            if fk in ("roe", "net_margin", "gross_margin", "revenue_growth"):
                                delta_str = f"{(subj_v - med_val)*100:+.1f}pp vs median"
                            else:
                                delta_str = f"{subj_v - med_val:+.2f} vs median"
                        med_cols[mi % len(med_cols)].metric(
                            f"Median {lbl}",
                            _fmt_opt(med_val, ".2f"),
                            delta=delta_str,
                        )

                # Interactive bar chart
                from peer_comps import MULTIPLES as _PEER_MULTIPLES
                st.markdown("**Interactive Multiple Bar Chart**")
                multiples_choices = [(k, lbl) for k, lbl in _PEER_MULTIPLES]
                sel_multiple_lbl = st.selectbox(
                    "Visualize multiple",
                    [lbl for _, lbl in multiples_choices],
                    key=f"peer_bar_multiple_{ticker}",
                )
                sel_multiple_key = next((k for k, lbl in multiples_choices if lbl == sel_multiple_lbl), None)
                if sel_multiple_key:
                    bar_tickers = []
                    bar_vals = []
                    bar_colors = []
                    subject_val = None
                    for p in peer_comps_result.peers:
                        v = getattr(p, sel_multiple_key, None)
                        if v is not None:
                            bar_tickers.append(p.ticker)
                            bar_vals.append(v)
                            bar_colors.append("#42A5F5" if p.is_subject else "#667eea")
                            if p.is_subject:
                                subject_val = v
                    if bar_vals:
                        bar_fig = go.Figure()
                        bar_fig.add_trace(go.Bar(
                            x=bar_tickers, y=bar_vals,
                            marker_color=bar_colors,
                            text=[f"{v:.2f}" for v in bar_vals],
                            textposition="auto",
                        ))
                        med_val_bar = peer_comps_result.medians.get(sel_multiple_key)
                        if med_val_bar is not None:
                            bar_fig.add_hline(
                                y=med_val_bar, line_dash="dash", line_color="#FFA726",
                                annotation_text=f"Sector Median: {med_val_bar:.2f}",
                                annotation_position="top right",
                            )
                        bar_fig.update_layout(
                            template="plotly_dark", height=320,
                            margin=dict(l=10, r=10, t=40, b=10),
                            title=f"{sel_multiple_lbl} vs Peers",
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            showlegend=False,
                        )
                        st.plotly_chart(bar_fig, use_container_width=True)
        else:
            st.info("Peer comparison data unavailable.")

        analyst_targets = data.get("analyst_targets", {})
        current_price = company_info.get("price")
        if analyst_targets and current_price:
            st.divider()
            fig_pt = create_price_target_chart(current_price, analyst_targets)
            if fig_pt:
                st.plotly_chart(fig_pt, use_container_width=True)
            at_cols = st.columns(4)
            at_cols[0].metric("Analyst Low", f"${analyst_targets.get('low',0):.2f}" if analyst_targets.get('low') else "N/A")
            at_cols[1].metric("Analyst Mean", f"${analyst_targets.get('mean',0):.2f}" if analyst_targets.get('mean') else "N/A")
            at_cols[2].metric("Analyst Median", f"${analyst_targets.get('median',0):.2f}" if analyst_targets.get('median') else "N/A")
            at_cols[3].metric("Analyst High", f"${analyst_targets.get('high',0):.2f}" if analyst_targets.get('high') else "N/A")

    # ── TAB 4: TIMING ──
    with tab_timing:
        sr_levels = getattr(tech_result, 'support_resistance', [])
        st.plotly_chart(create_price_chart(chart_df, ticker, sr_levels=sr_levels), use_container_width=True)
        st.plotly_chart(create_volume_chart(chart_df), use_container_width=True)

        if sr_levels:
            st.markdown("##### Support & Resistance Levels")
            current_price = company_info.get("price", 0)
            supports = [l for l in sr_levels if l.kind == "support"]
            resistances = [l for l in sr_levels if l.kind == "resistance"]

            sr_c1, sr_c2 = st.columns(2)
            with sr_c1:
                st.markdown("**Support Levels**")
                for lvl in sorted(supports, key=lambda l: l.price, reverse=True):
                    dist_pct = (current_price - lvl.price) / current_price * 100 if current_price else 0
                    sc_color = "#00C853" if lvl.strength == "strong" else "#69F0AE" if lvl.strength == "moderate" else "#A5D6A7"
                    st.markdown(f"<div style='padding:0.4rem 0.6rem;background:rgba(76,175,80,0.06);border-left:3px solid {sc_color};border-radius:0 8px 8px 0;margin-bottom:0.3rem'>"
                                f"<div style='display:flex;justify-content:space-between;align-items:center'>"
                                f"<span style='color:white;font-weight:600'>${lvl.price:.2f}</span>"
                                f"<span style='color:{sc_color};font-size:0.8rem;font-weight:600'>{lvl.strength.upper()}</span></div>"
                                f"<div style='color:#888;font-size:0.78rem'>{dist_pct:.1f}% below · {lvl.method}</div></div>", unsafe_allow_html=True)

            with sr_c2:
                st.markdown("**Resistance Levels**")
                for lvl in sorted(resistances, key=lambda l: l.price):
                    dist_pct = (lvl.price - current_price) / current_price * 100 if current_price else 0
                    rc_color = "#FF1744" if lvl.strength == "strong" else "#FF8A65" if lvl.strength == "moderate" else "#FFAB91"
                    st.markdown(f"<div style='padding:0.4rem 0.6rem;background:rgba(255,23,68,0.06);border-left:3px solid {rc_color};border-radius:0 8px 8px 0;margin-bottom:0.3rem'>"
                                f"<div style='display:flex;justify-content:space-between;align-items:center'>"
                                f"<span style='color:white;font-weight:600'>${lvl.price:.2f}</span>"
                                f"<span style='color:{rc_color};font-size:0.8rem;font-weight:600'>{lvl.strength.upper()}</span></div>"
                                f"<div style='color:#888;font-size:0.78rem'>{dist_pct:.1f}% above · {lvl.method}</div></div>", unsafe_allow_html=True)

        st.divider()
        st.markdown("##### Technical Indicators")
        ind = tech_result.indicators
        if ind:
            ind_cols = st.columns(4)
            for i, (k, v) in enumerate(ind.items()):
                ind_cols[i % 4].metric(k.replace("_", " "), f"{v:,.2f}" if isinstance(v, float) else str(v))

        _tech_signals = getattr(tech_result, "signals", None)
        if _tech_signals:
            with st.expander("📊 Full Technical Signal Breakdown", expanded=False):
                for sig in _tech_signals:
                    sig_name = getattr(sig, "name", str(sig))
                    sig_cat = getattr(sig, "category", "")
                    sig_val = getattr(sig, "value", None)
                    sig_interp = getattr(sig, "interpretation", "")
                    sig_score = getattr(sig, "score", 0) or 0
                    sc = "#00C853" if sig_score > 0.1 else "#FF1744" if sig_score < -0.1 else "#FFD54F"
                    bar_pct = (sig_score + 1) / 2 * 100
                    val_str = f"{sig_val:.4f}" if isinstance(sig_val, float) else str(sig_val) if sig_val is not None else "N/A"
                    st.markdown(
                        f"<div style='padding:0.5rem 0.8rem;background:rgba(15,17,30,0.6);"
                        f"border-left:3px solid {sc};border-radius:0 8px 8px 0;margin-bottom:0.35rem'>"
                        f"<div style='display:flex;align-items:center;gap:0.5rem'>"
                        f"<span style='color:{sc};font-weight:700;flex:1'>{sig_name}</span>"
                        f"<span style='color:#666;font-size:0.72rem;text-transform:uppercase'>{sig_cat}</span>"
                        f"<span style='color:#aaa;font-size:0.82rem;margin-left:0.5rem'>{val_str}</span>"
                        f"<span style='color:{sc};font-weight:600;width:48px;text-align:right'>{sig_score:+.2f}</span>"
                        f"</div>"
                        f"<div style='color:#888;font-size:0.78rem;margin:0.2rem 0 0.3rem 0'>{sig_interp}</div>"
                        f"<div style='background:rgba(255,255,255,0.06);height:4px;border-radius:2px'>"
                        f"<div style='background:{sc};height:4px;border-radius:2px;width:{bar_pct:.0f}%'></div></div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

        st.divider()
        st.markdown("##### Enhanced Technical Analysis")
        te = data.get("tech_enhanced")

        def _fmt(val, fmt="+.1f", suffix="%", fallback="N/A"):
            if val is None:
                return fallback
            return f"{val:{fmt}}{suffix}"

        if te:
            te_c1, te_c2, te_c3 = st.columns(3)
            rs = getattr(te, "rs", None)
            if rs:
                rs_composite = getattr(rs, "rs_composite", None)
                rs_color = "#00C853" if rs_composite and rs_composite > 0 else "#FF1744"
                te_c1.markdown(
                    f"<div style='padding:0.8rem;background:{rs_color}11;border:1px solid {rs_color}44;border-radius:8px'>"
                    f"<div style='color:#888;font-size:0.7rem;text-transform:uppercase'>Relative Strength</div>"
                    f"<div style='color:{rs_color};font-size:1.4rem;font-weight:700'>{getattr(rs, 'rs_signal', None) or 'N/A'}</div>"
                    f"<div style='color:#aaa;font-size:0.82rem'>RS Composite: {_fmt(rs_composite)}</div>"
                    f"<div style='color:#666;font-size:0.75rem'>1M: {_fmt(getattr(rs, 'rs_1m', None))}  3M: {_fmt(getattr(rs, 'rs_3m', None))}</div>"
                    f"</div>", unsafe_allow_html=True)
            mom = getattr(te, "momentum", None)
            if mom:
                mom_signal = getattr(mom, "momentum_signal", "") or ""
                mom_color = "#00C853" if mom_signal == "BULLISH" else "#FF1744" if mom_signal == "BEARISH" else "#FFD54F"
                te_c2.markdown(
                    f"<div style='padding:0.8rem;background:{mom_color}11;border:1px solid {mom_color}44;border-radius:8px'>"
                    f"<div style='color:#888;font-size:0.7rem;text-transform:uppercase'>Momentum (ROC)</div>"
                    f"<div style='color:{mom_color};font-size:1.4rem;font-weight:700'>{mom_signal or 'N/A'}</div>"
                    f"<div style='color:#aaa;font-size:0.82rem'>12-1: {_fmt(getattr(mom, 'momentum_12_1', None))}</div>"
                    f"<div style='color:#666;font-size:0.75rem'>1M: {_fmt(getattr(mom, 'roc_1m', None))}  3M: {_fmt(getattr(mom, 'roc_3m', None))}</div>"
                    f"</div>", unsafe_allow_html=True)
            struct = getattr(te, "structure", None)
            if struct:
                struct_signal = getattr(struct, "structure_signal", "") or ""
                struct_color = "#00C853" if struct_signal in ("BULLISH", "NEAR_HIGH") else "#FFA726"
                pct_from_high = getattr(struct, "pct_from_52w_high", None)
                vs_vwap = getattr(struct, "price_vs_vwap_90d", None)
                te_c3.markdown(
                    f"<div style='padding:0.8rem;background:{struct_color}11;border:1px solid {struct_color}44;border-radius:8px'>"
                    f"<div style='color:#888;font-size:0.7rem;text-transform:uppercase'>Price Structure</div>"
                    f"<div style='color:{struct_color};font-size:1.4rem;font-weight:700'>{struct_signal or 'N/A'}</div>"
                    f"<div style='color:#aaa;font-size:0.82rem'>From 52W High: {_fmt(pct_from_high)}</div>"
                    f"<div style='color:#666;font-size:0.75rem'>vs VWAP90: {_fmt(vs_vwap)}</div>"
                    f"</div>", unsafe_allow_html=True)
        else:
            st.info("Enhanced technical data unavailable.")

        st.divider()
        st.markdown("##### Macro Environment & Sector")
        if macro_result:
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Macro Phase", getattr(getattr(macro_result, "cycle", None), "phase", "N/A"))
            mc2.metric("VIX", f"{getattr(macro_result, 'vix', 'N/A')}")
            yc_data = getattr(macro_result, "yield_curve", {}) or {}
            mc3.metric("10Y Yield", f"{yc_data.get('10y', 'N/A')}")
            mc4.metric("2Y Yield", f"{yc_data.get('2y', 'N/A')}")
            if getattr(macro_result, "summary", None):
                st.markdown(f"<div style='padding:0.6rem 1rem;background:rgba(38,166,154,0.08);border-left:3px solid #26A69A;border-radius:0 8px 8px 0;color:#b2dfdb;font-size:0.85rem'>{macro_result.summary}</div>", unsafe_allow_html=True)
        else:
            st.info("Macro data unavailable.")

        if sector_result:
            st.divider()
            st.markdown("##### Sector Analysis")
            s1, s2 = st.columns(2)
            s1.metric("Current Sector", getattr(sector_result, "current_sector", company_info.get("sector", "N/A")))
            s2.metric("Cycle Alignment", getattr(sector_result, "cycle_alignment", "N/A"))
            if getattr(sector_result, "summary", None):
                st.caption(sector_result.summary)

        st.divider()
        st.markdown("##### News & Sentiment")
        if news_result:
            ns_cols = st.columns(3)
            ns_cols[0].metric("Sentiment Score", f"{getattr(news_result, 'sentiment_score', 0):.3f}")
            ns_cols[1].metric("Signal", getattr(news_result, "signal", "N/A"))
            ns_cols[2].metric("Articles Analyzed", str(getattr(news_result, "article_count", "N/A")))
            articles = getattr(news_result, "articles", []) or []
            for art in articles[:5]:
                score = getattr(art, "sentiment_score", 0)
                art_color = "#00C853" if score > 0.1 else "#FF1744" if score < -0.1 else "#888"
                headline = getattr(art, "headline", str(art))
                st.markdown(f"<div style='padding:0.35rem 0.6rem;background:rgba(66,165,245,0.05);border-left:3px solid {art_color};border-radius:0 6px 6px 0;margin-bottom:0.25rem;color:#b0bec5;font-size:0.83rem'>{headline}</div>", unsafe_allow_html=True)
        else:
            st.info("News sentiment data unavailable.")

    # ── TAB 5: TRADE PLAN ──
    with tab_trade:
        rp = data.get("risk_profile")
        st.markdown("##### Risk & Position Sizing")
        if rp:
            rm1, rm2, rm3, rm4 = st.columns(4)
            pos_sizes = getattr(rp, "position_sizes", []) or []
            first_pos = pos_sizes[0] if pos_sizes else None
            rm1.metric("Suggested Shares", str(first_pos.shares) if first_pos else "N/A",
                       help="Position size based on risking 1% of $100,000 account using the ATR stop-loss level.")
            rm2.metric("Risk Amount", format_large_number(first_pos.risk_amount if first_pos else None),
                       help="Dollar amount at risk if stop-loss is hit. Should not exceed 1-2% of total portfolio.")
            sharpe = getattr(rp, "sharpe_approx", None)
            rm3.metric("Sharpe Ratio", f"{sharpe:.2f}" if sharpe is not None else "N/A",
                       help="Return per unit of risk. >1 = good, >2 = very good, <0 = underperforming risk-free rate.")
            max_dd = getattr(rp, "max_drawdown", None)
            rm4.metric("Max Drawdown", f"{max_dd:.1%}" if max_dd is not None else "N/A",
                       help="Largest peak-to-trough decline. Measures downside risk and psychological pain of holding.")

            stop_levels = getattr(rp, "stop_losses", []) or []
            rr_scenarios = getattr(rp, "risk_reward", []) or []
            sl_col, tp_col = st.columns(2)
            with sl_col:
                st.markdown("**Stop-Loss Levels**")
                for sl in stop_levels[:4]:
                    stop_price = sl.stop_price
                    sl_type = sl.method
                    st.markdown(
                        f"<div style='padding:0.4rem 0.6rem;background:rgba(255,23,68,0.06);"
                        f"border-left:3px solid #FF1744;border-radius:0 8px 8px 0;margin-bottom:0.25rem;font-size:0.85rem'>"
                        f"<span style='color:white;font-weight:600'>${stop_price:.2f}</span>"
                        f"  <span style='color:#888'>{sl_type}</span></div>",
                        unsafe_allow_html=True,
                    )
            with tp_col:
                st.markdown("**Risk / Reward Targets**")
                for scenario in rr_scenarios[:4]:
                    tp_val = scenario.target_price
                    rr_ratio = getattr(scenario, "risk_reward_ratio", None)
                    label = getattr(scenario, "label", "Target")
                    rr_str = f"  R:R {rr_ratio:.1f}:1" if rr_ratio else ""
                    st.markdown(
                        f"<div style='padding:0.4rem 0.6rem;background:rgba(0,200,83,0.06);"
                        f"border-left:3px solid #00C853;border-radius:0 8px 8px 0;margin-bottom:0.25rem;font-size:0.85rem'>"
                        f"<span style='color:white;font-weight:600'>${tp_val:.2f}</span>"
                        f"  <span style='color:#888'>{label}{rr_str}</span></div>",
                        unsafe_allow_html=True,
                    )
        else:
            st.info("Risk profile unavailable.")

        st.divider()
        st.markdown("##### Credit & Macro Conditions")
        cc = data.get("credit_conditions")
        if cc:
            cc_color = "#00C853" if cc.credit_score > 0.1 else "#FFA726" if cc.credit_score > -0.1 else "#FF1744"
            hy = getattr(cc, "hy_spread", None)
            ig = getattr(cc, "ig_spread", None)
            rr_cc = getattr(cc, "real_rates", None)
            cc_cols = st.columns(4)
            if hy:
                hy_perf = getattr(hy, "relative_performance_1m", None)
                cc_cols[0].metric("HY vs Duration (1M)", f"{hy_perf:+.1%}" if hy_perf is not None else "N/A")
                cc_cols[1].metric("HY Signal", hy.signal or "N/A")
            if ig:
                ig_perf = getattr(ig, "relative_performance_1m", None)
                cc_cols[2].metric("IG vs Duration (1M)", f"{ig_perf:+.1%}" if ig_perf is not None else "N/A")
            if rr_cc:
                cc_cols[3].metric("Real Rates", rr_cc.signal or "N/A")
            st.markdown(
                f"<div style='padding:0.6rem 1rem;background:{cc_color}18;border:1px solid {cc_color}55;border-radius:8px;margin:0.5rem 0'>"
                f"<span style='color:{cc_color};font-weight:700'>{cc.financial_conditions_signal or 'N/A'}</span>"
                f"  ·  Score: {cc.credit_score:+.3f}  ·  {cc.summary}</div>",
                unsafe_allow_html=True,
            )
            if getattr(cc, "equity_implications", None):
                st.caption(f"Equity implications: {cc.equity_implications}")
        else:
            st.info("Credit conditions data unavailable.")

        st.divider()
        options_result = data.get("options_result")
        opt = options_result
        st.markdown("##### Options Market Signals")
        if opt and not getattr(opt, "error", None):
            # Signal banner
            _opt_signal = getattr(opt, "signal", "Neutral")
            _opt_expiry = getattr(opt, "nearest_expiry", "")
            _opt_summary = getattr(opt, "summary", "")
            _opt_sig_color = "#00C853" if "Bullish" in _opt_signal else "#FF1744" if "Bearish" in _opt_signal else "#FFA726" if "High IV" in _opt_signal else "#42A5F5"
            st.markdown(
                f"<div style='padding:0.7rem 1rem;background:{_opt_sig_color}18;border:1px solid {_opt_sig_color}55;"
                f"border-radius:8px;margin-bottom:0.6rem'>"
                f"<span style='color:{_opt_sig_color};font-weight:700;font-size:1rem'>{_opt_signal}</span>"
                f"{'  ·  Expiry: ' + _opt_expiry if _opt_expiry else ''}"
                f"{'  ·  ' + _opt_summary if _opt_summary else ''}"
                f"</div>",
                unsafe_allow_html=True,
            )
            # Row 1: IV metrics
            _avg_iv = getattr(opt, "avg_iv", None)
            _rv30 = getattr(opt, "realized_vol_30d", None)
            _iv_prem = getattr(opt, "iv_premium", None)
            _iv_rank = getattr(opt, "iv_rank", None)
            _max_pain = getattr(opt, "max_pain_price", None)
            op_r1 = st.columns(5)
            op_r1[0].metric("ATM IV", f"{_avg_iv*100:.1f}%" if _avg_iv is not None else "N/A",
                            help="Implied volatility of near-the-money options. Reflects market's expected price move. Compare to Realized Vol.")
            op_r1[1].metric("30d Realized Vol", f"{_rv30*100:.1f}%" if _rv30 is not None else "N/A")
            op_r1[2].metric("IV Premium", f"{_iv_prem*100:+.1f}pp" if _iv_prem is not None else "N/A",
                            help="IV minus realized vol. Positive = options are expensive relative to actual moves — consider selling premium.")
            op_r1[3].metric("IV Rank", f"{_iv_rank:.0f}" if _iv_rank is not None else "N/A")
            op_r1[4].metric("Max Pain", f"${_max_pain:.2f}" if _max_pain else "N/A",
                            help="Strike price where the most options contracts expire worthless. Price often gravitates here near expiry.")
            # Row 2: Volume/OI metrics
            _pc_oi = getattr(opt, "put_call_oi_ratio", None)
            _pc_vol = getattr(opt, "put_call_vol_ratio", None)
            _call_oi = getattr(opt, "total_call_oi", None)
            _put_oi = getattr(opt, "total_put_oi", None)
            op_r2 = st.columns(4)
            op_r2[0].metric("Put/Call OI Ratio", f"{_pc_oi:.2f}" if _pc_oi is not None else "N/A",
                            help=">1 = more put open interest (bearish hedge or speculation). <0.7 = call-heavy, bullish sentiment.")
            op_r2[1].metric("Put/Call Vol Ratio", f"{_pc_vol:.2f}" if _pc_vol is not None else "N/A")
            op_r2[2].metric("Total Call OI", f"{_call_oi:,}" if _call_oi else "N/A")
            op_r2[3].metric("Total Put OI", f"{_put_oi:,}" if _put_oi else "N/A")
            # P/C contextual message
            if _pc_oi is not None:
                if _pc_oi > 1.3:
                    st.warning(f"Put/Call OI Ratio {_pc_oi:.2f} — elevated put activity suggests bearish hedging or speculation.")
                elif _pc_oi < 0.7:
                    st.success(f"Put/Call OI Ratio {_pc_oi:.2f} — call-heavy positioning suggests bullish sentiment.")
                else:
                    st.info(f"Put/Call OI Ratio {_pc_oi:.2f} — balanced positioning, no strong directional bias.")

            with st.expander("Options Chain & IV Skew", expanded=False):
                chain_summary = getattr(opt, "chain_summary", []) or []
                if chain_summary:
                    chain_df = pd.DataFrame(chain_summary)
                    st.dataframe(chain_df, use_container_width=True, hide_index=True)
                else:
                    st.info("Options chain data unavailable.")
                # IV skew chart: ATM vs wings
                if chain_summary:
                    try:
                        skew_strikes = [row.get("strike") for row in chain_summary if row.get("strike") is not None]
                        skew_ivs = [row.get("avg_iv") or row.get("impliedVolatility") for row in chain_summary]
                        skew_vals = [(s, iv) for s, iv in zip(skew_strikes, skew_ivs) if iv is not None]
                        if skew_vals:
                            skew_x, skew_y = zip(*skew_vals)
                            skew_fig = go.Figure()
                            skew_fig.add_trace(go.Scatter(x=list(skew_x), y=[v * 100 for v in skew_y],
                                                          mode="lines+markers", line=dict(color="#667eea", width=2),
                                                          name="IV Skew"))
                            skew_fig.update_layout(template="plotly_dark", height=260,
                                                   margin=dict(l=10, r=10, t=30, b=10),
                                                   title="IV Skew by Strike",
                                                   xaxis_title="Strike", yaxis_title="IV %",
                                                   paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                            st.plotly_chart(skew_fig, use_container_width=True)
                    except Exception:
                        pass
        else:
            err = getattr(opt, "error", None) if opt else None
            st.info(f"Options data unavailable.{f' ({err})' if err else ''}")

        st.divider()
        inst_risk = data.get("inst_risk")
        st.markdown("##### Institutional Risk Signals")
        if inst_risk:
            ir = inst_risk
            gex = getattr(ir, "gex", None)
            sm_ir = getattr(ir, "smart_money", None)
            credit = getattr(ir, "credit", None)

            # Position summary banner
            pos_summary = getattr(ir, "position_summary", None)
            if pos_summary:
                st.markdown(
                    f"<div style='padding:0.6rem 1rem;background:rgba(66,165,245,0.08);"
                    f"border-left:3px solid #42A5F5;border-radius:0 8px 8px 0;color:#b0bec5;font-size:0.85rem'>"
                    f"{pos_summary}</div>",
                    unsafe_allow_html=True,
                )

            # Two columns: GEX | Smart Money
            ir_gex_col, ir_sm_col = st.columns(2)

            with ir_gex_col:
                st.markdown("**GEX (Dealer Gamma)**")
                if gex:
                    above = gex.above_flip
                    regime_label = "STABLE" if above else "VOLATILE" if above is not None else "N/A"
                    regime_color = "#00C853" if above else "#FF1744" if above is not None else "#888"
                    regime_desc = "Price above gamma flip — dealers hedge by buying dips (stabilizing)." if above else "Price below gamma flip — dealers amplify moves (vol-amplifying)." if above is not None else ""
                    st.markdown(
                        f"<div style='padding:0.5rem 0.8rem;background:{regime_color}18;"
                        f"border:1px solid {regime_color}44;border-radius:8px;margin-bottom:0.5rem'>"
                        f"<span style='color:{regime_color};font-weight:700'>{regime_label}</span>"
                        f"  <span style='color:#888;font-size:0.78rem'>{regime_desc}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    gex_m1, gex_m2, gex_m3 = st.columns(3)
                    gex_m1.metric("Total GEX ($M)", f"${gex.total_gex/1e6:.1f}M" if gex.total_gex else "N/A")
                    gex_m2.metric("Gamma Flip", f"${gex.gamma_flip:.2f}" if gex.gamma_flip else "N/A")
                    curr_px = company_info.get("price")
                    dist_to_flip = None
                    if curr_px and gex.gamma_flip:
                        dist_to_flip = (gex.gamma_flip - curr_px) / curr_px * 100
                    gex_m3.metric("Distance to Flip", f"{dist_to_flip:+.1f}%" if dist_to_flip is not None else "N/A")

                    top_walls = getattr(gex, "top_walls", []) or []
                    top_traps = getattr(gex, "top_trapdoors", []) or []
                    if top_walls or top_traps:
                        wall_names, wall_vals, wall_colors = [], [], []
                        for wl in top_walls[:4]:
                            wall_names.append(f"Wall ${getattr(wl, 'strike', '?')}")
                            wall_vals.append(getattr(wl, "gex_notional", 0) / 1e6)
                            wall_colors.append("#00C853")
                        for tr in top_traps[:4]:
                            wall_names.append(f"Trap ${getattr(tr, 'strike', '?')}")
                            wall_vals.append(getattr(tr, "gex_notional", 0) / 1e6)
                            wall_colors.append("#FF1744")
                        if wall_vals:
                            gex_bar = go.Figure()
                            gex_bar.add_trace(go.Bar(x=wall_names, y=wall_vals, marker_color=wall_colors,
                                                     text=[f"${v:.1f}M" for v in wall_vals], textposition="auto"))
                            gex_bar.update_layout(template="plotly_dark", height=220,
                                                  margin=dict(l=5, r=5, t=30, b=5),
                                                  title="GEX Walls & Trapdoors ($M)",
                                                  paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                                  showlegend=False)
                            st.plotly_chart(gex_bar, use_container_width=True)

            with ir_sm_col:
                st.markdown("**Smart Money (13F)**")
                if sm_ir:
                    sm_m1, sm_m2, sm_m3 = st.columns(3)
                    conc_top5 = getattr(sm_ir, "concentration_top5", None)
                    sm_m1.metric("Top-5 Conc.", f"{conc_top5:.1%}" if conc_top5 is not None else "N/A")
                    net_bp = getattr(sm_ir, "net_buying_pressure", None)
                    sm_m2.metric("Net Buying QoQ", f"{net_bp:+.1%}" if net_bp is not None else "N/A")
                    hc = getattr(sm_ir, "high_conviction_count", None)
                    sm_m3.metric("High Conv. Holders", str(hc) if hc is not None else "N/A")
                    rec_tr = getattr(sm_ir, "rec_trend", "N/A")
                    tr_color = "#00C853" if rec_tr == "Improving" else "#FF1744" if rec_tr == "Deteriorating" else "#FFD54F"
                    st.markdown(
                        f"<div style='padding:0.4rem 0.7rem;background:{tr_color}18;"
                        f"border:1px solid {tr_color}44;border-radius:6px;margin:0.3rem 0'>"
                        f"<span style='color:{tr_color};font-weight:700'>Analyst Trend: {rec_tr}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    holders_list = getattr(sm_ir, "holders", []) or []
                    if holders_list:
                        st.markdown("**Top Institutional Holders**")
                        for h in holders_list[:8]:
                            h_name = getattr(h, "name", "Unknown")
                            h_pct = getattr(h, "pct_held", 0) or 0
                            h_chg = getattr(h, "pct_change", None)
                            chg_str = f"  {h_chg:+.1%} QoQ" if h_chg is not None else ""
                            chg_color = "#00C853" if (h_chg or 0) > 0 else "#FF1744" if (h_chg or 0) < 0 else "#888"
                            st.markdown(
                                f"<div style='display:flex;justify-content:space-between;padding:0.2rem 0.5rem;"
                                f"border-bottom:1px solid rgba(255,255,255,0.04);font-size:0.82rem'>"
                                f"<span style='color:#ccc'>{h_name}</span>"
                                f"<span style='color:#888'>{h_pct:.2%}"
                                f"<span style='color:{chg_color}'>{chg_str}</span></span>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )

            # Credit proxy (full width)
            if credit:
                st.markdown("**Credit Risk Proxy**")
                cds_bps = getattr(credit, "cds_proxy_bps", None)
                cred_rating = getattr(credit, "credit_rating_proxy", "N/A")
                int_cov = getattr(credit, "interest_coverage", None)
                d_ebitda = getattr(credit, "debt_to_ebitda", None)
                cs01 = getattr(credit, "cs01_proxy", None)
                dv01 = getattr(credit, "dv01_proxy", None)
                cr_cols = st.columns(5)
                cr_cols[0].metric("CDS Proxy", f"{cds_bps:.0f}bps" if cds_bps is not None else "N/A",
                                  help="Synthetic credit default swap spread in basis points. Higher = market perceives more default risk.")
                cr_cols[1].metric("Credit Rating Proxy", cred_rating)
                cr_cols[2].metric("Interest Coverage", f"{int_cov:.1f}x" if int_cov is not None else "N/A")
                cr_cols[3].metric("Debt/EBITDA", f"{d_ebitda:.1f}x" if d_ebitda is not None else "N/A")
                cr_cols[4].metric("CS01 / DV01",
                                  f"${cs01/1e3:.0f}K / ${dv01/1e3:.0f}K" if cs01 and dv01 else "N/A")
        else:
            st.info("Institutional risk data unavailable.")

        forensic = data.get("forensic")
        if forensic and not getattr(forensic, "error", None):
            st.divider()
            st.markdown("##### Forensic NLP (EDGAR)")
            fr_verdict = getattr(forensic, "verdict", None)
            if fr_verdict:
                _fv_str = getattr(fr_verdict, "verdict", "N/A")
                _fv_score = getattr(fr_verdict, "score", None)
                _fv_color = {"Clean": "#1a7a4a", "Watch": "#b8860b", "Red Flag": "#c0704a", "Critical": "#c0392b"}.get(_fv_str, "#888")
                f1, f2 = st.columns(2)
                f1.metric("Forensic Verdict", _fv_str)
                f2.metric("Concern Score", f"{_fv_score:.0f}/100" if _fv_score is not None else "N/A")
                _fv_summary = getattr(fr_verdict, "summary", None)
                if _fv_summary:
                    st.caption(_fv_summary)
                _fv_flags = getattr(fr_verdict, "flags", []) or []
                for _ff in _fv_flags[:3]:
                    st.caption(f"⚠️ {_ff}")

            with st.expander("🧬 Forensic NLP — EDGAR MD&A Analysis", expanded=False):
                # Verdict banner
                if fr_verdict:
                    _fv_str = getattr(fr_verdict, "verdict", "N/A")
                    _fv_score = getattr(fr_verdict, "score", None)
                    _fv_color = {"Clean": "#1a7a4a", "Watch": "#b8860b", "Red Flag": "#c0704a", "Critical": "#c0392b"}.get(_fv_str, "#888")
                    st.markdown(
                        f"<div style='padding:0.8rem 1.2rem;background:{_fv_color}22;border:2px solid {_fv_color};"
                        f"border-radius:10px;margin-bottom:0.8rem'>"
                        f"<span style='color:{_fv_color};font-size:1.3rem;font-weight:800'>{_fv_str}</span>"
                        f"{'  ·  Score: ' + str(_fv_score) + '/100' if _fv_score is not None else ''}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                # Filing metadata
                fr_filing = getattr(forensic, "filing", None)
                if fr_filing:
                    fm1, fm2, fm3, fm4 = st.columns(4)
                    fm1.metric("Form Type", getattr(fr_filing, "form_type", "N/A"))
                    fm2.metric("Period", getattr(fr_filing, "period", "N/A"))
                    fm3.metric("Filed Date", getattr(fr_filing, "filed_date", "N/A"))
                    fm4.metric("Word Count", f"{getattr(fr_filing, 'word_count', 0):,}")

                # LM Linguistic Scores chart
                lm = getattr(fr_verdict, "lm_scores", None) if fr_verdict else None
                if lm:
                    st.markdown("**Loughran-McDonald Linguistic Scores**")
                    lm_metrics = [
                        ("Uncertainty", getattr(lm, "uncertainty_ratio", 0) * 100, 2.8),
                        ("Litigious", getattr(lm, "litigious_ratio", 0) * 100, 1.2),
                        ("Negative", getattr(lm, "negative_ratio", 0) * 100, 1.8),
                        ("Positive", getattr(lm, "positive_ratio", 0) * 100, 1.5),
                        ("Hedging", getattr(lm, "hedging_ratio", 0) * 100, 2.5),
                    ]
                    lm_fig = go.Figure()
                    lm_fig.add_trace(go.Bar(
                        x=[m[0] for m in lm_metrics],
                        y=[m[1] for m in lm_metrics],
                        name="Filing",
                        marker_color=["#FF8A65" if m[1] > m[2] else "#42A5F5" for m in lm_metrics],
                        text=[f"{m[1]:.2f}%" for m in lm_metrics],
                        textposition="auto",
                    ))
                    lm_fig.add_trace(go.Scatter(
                        x=[m[0] for m in lm_metrics],
                        y=[m[2] for m in lm_metrics],
                        mode="markers",
                        marker=dict(color="#FFD54F", size=10, symbol="diamond"),
                        name="Benchmark",
                    ))
                    lm_fig.update_layout(
                        template="plotly_dark", height=260,
                        margin=dict(l=10, r=10, t=30, b=10),
                        title="LM Scores vs Benchmarks",
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    )
                    st.plotly_chart(lm_fig, use_container_width=True)

                # Flags list
                if fr_verdict:
                    _all_flags = getattr(fr_verdict, "flags", []) or []
                    if _all_flags:
                        st.markdown("**Detected Flags**")
                        for _fl in _all_flags:
                            st.markdown(f"<div style='padding:0.3rem 0.6rem;background:rgba(255,23,68,0.08);"
                                        f"border-left:3px solid #FF5722;border-radius:0 6px 6px 0;"
                                        f"margin-bottom:0.2rem;color:#ffab91;font-size:0.83rem'>⚠️ {_fl}</div>",
                                        unsafe_allow_html=True)
                    _fv_sum = getattr(fr_verdict, "summary", None)
                    if _fv_sum:
                        st.markdown("**Summary**")
                        st.markdown(f"<div style='padding:0.6rem 1rem;background:rgba(255,255,255,0.04);"
                                    f"border-radius:8px;color:#ccc;font-size:0.88rem;line-height:1.6'>{_fv_sum}</div>",
                                    unsafe_allow_html=True)

                ai_analysis = getattr(forensic, "ai_analysis", None)
                if ai_analysis:
                    with st.expander("AI Narrative Analysis", expanded=False):
                        st.markdown(ai_analysis)

        alt_data = data.get("alt_data")
        if alt_data:
            st.divider()
            st.markdown("##### Alternative Data")
            a1, a2 = st.columns(2)
            a1.metric("Overall Signal", getattr(alt_data, "overall_signal", "N/A"))
            sig_count = getattr(alt_data, "signal_count", None)
            a2.metric("Bullish Layers", str(sig_count) if sig_count is not None else "N/A")
            # Show sentiment sub-signal if available
            _alt_sent = getattr(alt_data, "sentiment", None)
            if _alt_sent:
                st.caption(f"News sentiment: {getattr(_alt_sent, 'signal', 'N/A')}  "
                           f"(score {getattr(_alt_sent, 'composite_score', 0):+.2f})")

            with st.expander("🔭 Alternative Data — 5 Layers", expanded=False):
                # Overall signal banner
                _ad_overall = getattr(alt_data, "overall_signal", "N/A")
                _ad_oc = "#00C853" if "Bull" in _ad_overall else "#FF1744" if "Bear" in _ad_overall else "#FFD54F"
                st.markdown(
                    f"<div style='padding:0.6rem 1rem;background:{_ad_oc}18;border:1px solid {_ad_oc}55;"
                    f"border-radius:8px;margin-bottom:0.7rem'>"
                    f"<span style='color:{_ad_oc};font-weight:700;font-size:1rem'>{_ad_overall}</span>"
                    f"  ·  Bullish layers: {sig_count if sig_count is not None else 'N/A'} / 5"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                # Layer 1: News Sentiment
                st.markdown("**Layer 1 — News Sentiment**")
                sent_l1 = getattr(alt_data, "sentiment", None)
                if sent_l1:
                    _l1c = st.columns(3)
                    _l1c[0].metric("Direction", getattr(sent_l1, "signal", "N/A"))
                    _l1c[1].metric("Composite Score", f"{getattr(sent_l1, 'composite_score', 0):+.3f}")
                    _l1c[2].metric("Articles", str(getattr(sent_l1, "article_count", "N/A")))
                    _headlines = getattr(sent_l1, "top_headlines", []) or []
                    if _headlines:
                        with st.expander("Top Headlines", expanded=False):
                            for h in _headlines:
                                _ht = h.get("title", str(h)) if isinstance(h, dict) else str(h)
                                _hs = h.get("score", 0) if isinstance(h, dict) else 0
                                _hc = "#00C853" if _hs > 0.1 else "#FF1744" if _hs < -0.1 else "#888"
                                st.markdown(
                                    f"<div style='padding:0.3rem 0.6rem;border-left:3px solid {_hc};"
                                    f"border-radius:0 6px 6px 0;margin-bottom:0.2rem;color:#b0bec5;font-size:0.82rem'>"
                                    f"{_ht}</div>",
                                    unsafe_allow_html=True,
                                )
                else:
                    st.info("Sentiment data unavailable.")

                # Layer 2: Insider Flow
                st.markdown("**Layer 2 — Insider Flow**")
                ins_l2 = getattr(alt_data, "insider_flow", None)
                if ins_l2:
                    _l2c = st.columns(4)
                    _l2c[0].metric("Signal", getattr(ins_l2, "signal", "N/A"))
                    _l2c[1].metric("Buy Count", str(getattr(ins_l2, "buy_count", 0)))
                    _l2c[2].metric("Sell Count", str(getattr(ins_l2, "sell_count", 0)))
                    _nv = getattr(ins_l2, "net_value_bought", None)
                    _l2c[3].metric("Net Value Bought", format_large_number(_nv) if _nv is not None else "N/A")
                    _recent_t = getattr(ins_l2, "recent_trades", []) or []
                    if _recent_t:
                        with st.expander("Recent Insider Trades", expanded=False):
                            for t2 in _recent_t[:10]:
                                t2_name = getattr(t2, "name", "?")
                                t2_type = getattr(t2, "transaction_type", "?")
                                t2_shares = getattr(t2, "shares", 0) or 0
                                t2_val = getattr(t2, "value", None)
                                t2_color = "#00C853" if "buy" in str(t2_type).lower() else "#FF1744"
                                st.markdown(
                                    f"<div style='display:flex;justify-content:space-between;padding:0.2rem 0.5rem;"
                                    f"border-bottom:1px solid rgba(255,255,255,0.04);font-size:0.82rem'>"
                                    f"<span style='color:#ccc'>{t2_name}</span>"
                                    f"<span style='color:{t2_color}'>{t2_type}  {t2_shares:,} sh"
                                    f"{'  ' + format_large_number(t2_val) if t2_val else ''}</span>"
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )
                else:
                    st.info("Insider flow data unavailable.")

                # Layer 3: 13F Cluster
                st.markdown("**Layer 3 — 13F Cluster Buying**")
                clust_l3 = getattr(alt_data, "cluster_buying", None)
                if clust_l3:
                    st.markdown(
                        f"<div style='padding:0.5rem 0.8rem;background:rgba(102,126,234,0.08);"
                        f"border-left:3px solid #667eea;border-radius:0 8px 8px 0;margin-bottom:0.4rem'>"
                        f"<span style='color:#9fa8da;font-weight:700'>{getattr(clust_l3, 'consensus_signal', 'N/A')}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    _new_ents = getattr(clust_l3, "new_entrants", []) or []
                    _exits = getattr(clust_l3, "exits", []) or []
                    if _new_ents:
                        with st.expander(f"New Entrants / Accumulators ({len(_new_ents)})", expanded=False):
                            for nh in _new_ents:
                                st.markdown(f"  + **{getattr(nh, 'name', '?')}**  {getattr(nh, 'pct_held', 0):.2%} held  ({getattr(nh, 'pct_change', 0):+.2%} QoQ)")
                    if _exits:
                        with st.expander(f"Exits / Reducers ({len(_exits)})", expanded=False):
                            for eh in _exits:
                                st.markdown(f"  - **{getattr(eh, 'name', '?')}**  {getattr(eh, 'pct_held', 0):.2%} held  ({getattr(eh, 'pct_change', 0):+.2%} QoQ)")
                else:
                    st.info("13F cluster data unavailable.")

                # Layer 4: Vanna/Charm
                st.markdown("**Layer 4 — Vanna/Charm (Options Flow)**")
                vc_l4 = getattr(alt_data, "vanna_charm", None)
                if vc_l4:
                    _l4c = st.columns(4)
                    _l4c[0].metric("Net Vanna ($M)", f"${getattr(vc_l4, 'net_vanna', 0)/1e6:.1f}M")
                    _l4c[1].metric("Net Charm ($M)", f"${getattr(vc_l4, 'net_charm', 0)/1e6:.1f}M")
                    _l4c[2].metric("Vanna Signal", getattr(vc_l4, "vanna_signal", "N/A"))
                    _l4c[3].metric("Charm Signal", getattr(vc_l4, "charm_signal", "N/A"))
                    with st.expander("What are Vanna & Charm?", expanded=False):
                        st.markdown(
                            "**Vanna** = rate of change of delta with respect to implied vol. "
                            "When IV drops (vol crush), dealers with positive net vanna must BUY shares to stay delta-neutral — supporting price.\n\n"
                            "**Charm** = rate of change of delta with respect to time (theta decay). "
                            "As options approach expiry, dealer hedges unwind — the direction depends on whether they are long or short calls/puts."
                        )
                else:
                    st.info("Vanna/Charm data unavailable.")

                # Layer 5: Congressional
                st.markdown("**Layer 5 — Congressional Trades**")
                cong_l5 = getattr(alt_data, "congressional", None)
                if cong_l5:
                    _l5c = st.columns(2)
                    _l5c[0].metric("Congressional Bias", getattr(cong_l5, "net_congressional_bias", "N/A"))
                    _l5c[1].metric("Alpha Signal", getattr(cong_l5, "alpha_signal", "N/A"))
                    _cong_trades = getattr(cong_l5, "trades", []) or []
                    if _cong_trades:
                        with st.expander(f"Congressional Trade Records ({len(_cong_trades)})", expanded=False):
                            cong_rows = []
                            for ct in _cong_trades:
                                cong_rows.append({
                                    "Member": getattr(ct, "member_name", "?"),
                                    "Date": getattr(ct, "transaction_date", "?"),
                                    "Type": getattr(ct, "transaction_type", "?"),
                                    "Amount": getattr(ct, "amount_range", "?"),
                                    "House": getattr(ct, "house", "?"),
                                })
                            if cong_rows:
                                st.dataframe(pd.DataFrame(cong_rows), use_container_width=True, hide_index=True)
                else:
                    st.info("Congressional trades data unavailable.")

        # Analysts & Investors expander
        _analyst_targets = data.get("analyst_targets", {})
        _rec_summary = data.get("recommendations_summary")
        _upgrades = data.get("upgrades_downgrades")
        _inst_holders = data.get("institutional_holders")
        _mf_holders = data.get("mutualfund_holders")
        _current_price_ai = company_info.get("price")

        with st.expander("📋 Analysts & Investors", expanded=False):
            # Price target chart + metrics
            if _analyst_targets and _current_price_ai:
                _fig_pt2 = create_price_target_chart(_current_price_ai, _analyst_targets)
                if _fig_pt2:
                    st.plotly_chart(_fig_pt2, use_container_width=True)
                ai_m = st.columns(5)
                ai_m[0].metric("Current Price", f"${_current_price_ai:.2f}")
                ai_m[1].metric("Analyst Mean", f"${_analyst_targets.get('mean', 0):.2f}" if _analyst_targets.get('mean') else "N/A")
                ai_m[2].metric("Analyst Median", f"${_analyst_targets.get('median', 0):.2f}" if _analyst_targets.get('median') else "N/A")
                ai_m[3].metric("Analyst Low", f"${_analyst_targets.get('low', 0):.2f}" if _analyst_targets.get('low') else "N/A")
                ai_m[4].metric("Analyst High", f"${_analyst_targets.get('high', 0):.2f}" if _analyst_targets.get('high') else "N/A")

            # Recommendation distribution chart + colored boxes
            if _rec_summary is not None and len(_rec_summary) > 0:
                st.markdown("**Analyst Recommendation Distribution**")
                _rec_fig = create_recommendations_chart(_rec_summary)
                if _rec_fig:
                    st.plotly_chart(_rec_fig, use_container_width=True)
                # Latest period summary boxes
                try:
                    _latest_rec = _rec_summary.iloc[0] if hasattr(_rec_summary, "iloc") else None
                    if _latest_rec is not None:
                        rec_categories = [
                            ("Strong Buy", "strongBuy", "#00C853"),
                            ("Buy", "buy", "#69F0AE"),
                            ("Hold", "hold", "#FFD54F"),
                            ("Sell", "sell", "#FF8A65"),
                            ("Strong Sell", "strongSell", "#FF1744"),
                        ]
                        rec_box_cols = st.columns(5)
                        for rbi, (rlbl, rkey, rclr) in enumerate(rec_categories):
                            rval = _latest_rec.get(rkey, 0) if hasattr(_latest_rec, "get") else getattr(_latest_rec, rkey, 0)
                            rec_box_cols[rbi].markdown(
                                f"<div style='text-align:center;padding:0.6rem;background:{rclr}18;"
                                f"border:1px solid {rclr}55;border-radius:8px'>"
                                f"<div style='color:{rclr};font-size:1.4rem;font-weight:800'>{rval or 0}</div>"
                                f"<div style='color:#888;font-size:0.7rem'>{rlbl}</div>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                except Exception:
                    pass

            # Recent analyst actions
            if _upgrades is not None and len(_upgrades) > 0:
                st.markdown("**Recent Analyst Actions**")
                try:
                    ud_display = _upgrades.reset_index(drop=False) if hasattr(_upgrades, "reset_index") else _upgrades
                    st.dataframe(ud_display.head(15), use_container_width=True, hide_index=True)
                except Exception:
                    st.dataframe(_upgrades.head(15) if hasattr(_upgrades, "head") else _upgrades, use_container_width=True, hide_index=True)

            # Institutional holders table
            if _inst_holders is not None and len(_inst_holders) > 0:
                st.markdown("**Top Institutional Holders**")
                try:
                    st.dataframe(_inst_holders.head(15), use_container_width=True, hide_index=True)
                except Exception:
                    st.dataframe(_inst_holders, use_container_width=True, hide_index=True)

            # Mutual fund holders table
            if _mf_holders is not None and len(_mf_holders) > 0:
                st.markdown("**Top Mutual Fund Holders**")
                try:
                    st.dataframe(_mf_holders.head(15), use_container_width=True, hide_index=True)
                except Exception:
                    st.dataframe(_mf_holders, use_container_width=True, hide_index=True)

        cal = data.get("event_calendar", {})
        if cal:
            st.divider()
            st.markdown("##### Event Calendar")
            earnings_date = cal.get("earnings_date") or cal.get("earnings") or cal.get("Earnings Date")
            if earnings_date:
                st.markdown(f"📅 **Next Earnings**: {earnings_date}")
            ex_div = cal.get("ex_dividend_date") or cal.get("Ex-Dividend Date")
            if ex_div:
                st.markdown(f"💰 **Ex-Dividend Date**: {ex_div}")


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
if analyze_btn:
    tickers = parse_tickers(ticker_input)
    if not tickers:
        st.warning("Please enter at least one stock ticker.")
        st.stop()

    # ---- Run analysis for all tickers ----
    stock_results = {}
    failed_tickers = []

    import time as _time

    progress = st.progress(0, text=f"Analyzing {len(tickers)} stock(s)...")
    rate_limited = False
    for idx, ticker in enumerate(tickers):
        progress.progress((idx) / len(tickers), text=f"Analyzing {ticker} ({idx + 1}/{len(tickers)})...")
        try:
            result = run_single_analysis(ticker, period, fundamental_weight)
            if result is not None:
                stock_results[ticker] = result
            else:
                failed_tickers.append(ticker)
        except Exception as e:
            if "RateLimit" in type(e).__name__ or "429" in str(e):
                rate_limited = True
                failed_tickers.append(ticker)
            else:
                failed_tickers.append(ticker)

        if idx < len(tickers) - 1:
            _time.sleep(2)
    progress.progress(1.0, text="Analysis complete!")

    if rate_limited:
        st.warning(
            "**Yahoo Finance rate limit reached.** The cloud server's IP is temporarily "
            "throttled. Try again in a few minutes, or analyze fewer stocks at once. "
            "Results are cached for 15 minutes so repeated runs will be faster."
        )

    if failed_tickers:
        st.warning(f"Could not fetch data for: **{', '.join(failed_tickers)}**. These tickers were skipped.")

    if not stock_results:
        st.error("No valid stock data found. Please try again in a few minutes (Yahoo Finance may be rate-limiting).")
        st.stop()

    # ==================================================================
    # SINGLE STOCK — go straight to detail view (same as before)
    # ==================================================================
    if len(stock_results) == 1:
        ticker = list(stock_results.keys())[0]
        data = stock_results[ticker]
        ci = data["company_info"]
        rec = data["recommendation"]
        fund_data = data.get("fund_data", {})

        # ── Report header bar ──
        vc = rec.score_color if hasattr(rec, "score_color") else "#888"
        verdict_label = rec.recommendation
        verdict_emoji = {"STRONG BUY": "🟢", "BUY": "🟢", "HOLD": "🟡", "SELL": "🔴", "STRONG SELL": "🔴"}.get(verdict_label, "⚪")
        price = ci.get("price", 0)
        hist = data.get("chart_df")
        day_chg_str = ""
        try:
            if hist is not None and len(hist) >= 2:
                prev = float(hist["Close"].iloc[-2])
                curr = float(hist["Close"].iloc[-1])
                pct = (curr - prev) / prev * 100
                arrow = "▲" if pct >= 0 else "▼"
                clr = "#00C853" if pct >= 0 else "#FF1744"
                day_chg_str = f"<span style='color:{clr};font-size:0.9rem'> {arrow} {abs(pct):.2f}%</span>"
        except Exception:
            pass

        analyst_target = fund_data.get("target_mean_price")
        upside_str = ""
        if analyst_target and price:
            upside = (analyst_target - price) / price * 100
            upside_str = f"<span style='color:#aaa;font-size:0.8rem'>  ·  Analyst target ${analyst_target:.0f} ({upside:+.0f}%)</span>"

        st.markdown(
            f"<div style='display:flex;align-items:flex-start;justify-content:space-between;"
            f"border-bottom:1px solid rgba(255,255,255,0.07);padding-bottom:0.9rem;margin-bottom:0.5rem'>"
            f"<div>"
            f"<div class='ticker-name'>{ci.get('name',ticker)} "
            f"<span style='color:#555;font-weight:400;font-size:1rem'>({ticker})</span></div>"
            f"<div class='ticker-sub'>{ci.get('sector','—')}  ·  {ci.get('industry','—')}"
            f"  ·  {format_large_number(ci.get('market_cap'))} mkt cap</div>"
            f"</div>"
            f"<div style='text-align:right'>"
            f"<div class='ticker-price' style='color:#fff'>${price:.2f}{day_chg_str}</div>"
            f"<div style='margin-top:0.15rem'>"
            f"<span style='background:{vc}22;border:1px solid {vc}66;color:{vc};"
            f"padding:0.2rem 0.7rem;border-radius:6px;font-size:0.8rem;font-weight:700'>"
            f"{verdict_emoji} {verdict_label}</span>{upside_str}"
            f"</div></div></div>",
            unsafe_allow_html=True,
        )

        # ── Executive Summary strip (3 column blocks) ──
        dcf = data.get("dcf_result")
        qs  = data.get("quality_scores")
        rp  = data.get("risk_profile")

        ex_c1, ex_c2, ex_c3 = st.columns(3)
        with ex_c1:
            st.markdown("<div class='rpt-section'>VALUATION</div>", unsafe_allow_html=True)
            iv_str = f"${dcf.intrinsic_value:.0f}" if dcf and dcf.intrinsic_value else "—"
            mos_str = f"{dcf.margin_of_safety:+.0f}%" if dcf and dcf.margin_of_safety else "—"
            dcf_label = (dcf.valuation_label or "—") if dcf else "—"
            st.markdown(
                f"<div style='font-size:1.5rem;font-weight:700;color:#fff'>{iv_str}</div>"
                f"<div style='color:#888;font-size:0.78rem'>DCF Intrinsic Value</div>"
                f"<div style='margin-top:0.4rem;font-size:0.82rem;color:#aaa'>"
                f"MoS: <b style='color:#fff'>{mos_str}</b>  ·  {dcf_label}</div>",
                unsafe_allow_html=True,
            )

        with ex_c2:
            st.markdown("<div class='rpt-section'>QUALITY</div>", unsafe_allow_html=True)
            piotr_str = f"{qs.piotroski_f}/9" if qs and qs.piotroski_f is not None else "—"
            altman_str = f"{qs.altman_z:.2f}" if qs and qs.altman_z else "—"
            altman_zone = (qs.altman_zone or "") if qs else ""
            beneish_str = ("🚨 Flag" if qs and qs.beneish_flag else "✅ Clean") if qs else "—"
            st.markdown(
                f"<div style='font-size:1.5rem;font-weight:700;color:#fff'>F {piotr_str}</div>"
                f"<div style='color:#888;font-size:0.78rem'>Piotroski F-Score</div>"
                f"<div style='margin-top:0.4rem;font-size:0.82rem;color:#aaa'>"
                f"Altman Z: <b style='color:#fff'>{altman_str}</b> {altman_zone}  ·  Beneish: {beneish_str}</div>",
                unsafe_allow_html=True,
            )

        with ex_c3:
            st.markdown("<div class='rpt-section'>RISK</div>", unsafe_allow_html=True)
            sharpe_str = f"{rp.sharpe_approx:.2f}" if rp and rp.sharpe_approx else "—"
            maxdd_str = f"{rp.max_drawdown:.1%}" if rp and rp.max_drawdown else "—"
            var_str = f"{rp.var_95:.2%}" if rp and rp.var_95 else "—"
            st.markdown(
                f"<div style='font-size:1.5rem;font-weight:700;color:#fff'>{sharpe_str}</div>"
                f"<div style='color:#888;font-size:0.78rem'>Sharpe Ratio</div>"
                f"<div style='margin-top:0.4rem;font-size:0.82rem;color:#aaa'>"
                f"Max DD: <b style='color:#fff'>{maxdd_str}</b>  ·  VaR 95%: <b style='color:#fff'>{var_str}</b></div>",
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── PDF export ──
        if st.button("📄 Export PDF Report", use_container_width=False):
            try:
                from pdf_export import generate_report
                pdf_bytes = generate_report(
                    company_info=ci, fund_data=fund_data,
                    recommendation=rec, verdict=None,
                    dcf_result=dcf, quality_scores=qs,
                )
                st.download_button(
                    label="⬇ Download PDF",
                    data=pdf_bytes,
                    file_name=f"{ticker}_analysis_{date.today().isoformat()}.pdf",
                    mime="application/pdf",
                )
            except ImportError:
                st.error("Install reportlab: `pip install reportlab`")
            except Exception as e:
                st.error(f"PDF generation failed: {e}")

        render_stock_detail(ticker, data)
        st.stop()

    # ==================================================================
    # MULTI STOCK — Research report portfolio view
    # ==================================================================
    st.markdown(
        f"<div style='border-bottom:1px solid rgba(255,255,255,0.07);padding-bottom:0.6rem;margin-bottom:0.8rem'>"
        f"<span style='font-size:1.4rem;font-weight:700;color:#fff'>Portfolio Analysis</span>  "
        f"<span style='color:#444;font-size:0.78rem'>{len(stock_results)} securities  ·  {period}  ·  "
        f"F/T weights: {fundamental_weight}/{technical_weight}%</span></div>",
        unsafe_allow_html=True,
    )

    ranked = sorted(stock_results.items(), key=lambda x: x[1]["recommendation"].overall_score, reverse=True)
    medals = {0: "🥇", 1: "🥈", 2: "🥉"}

    # Side-by-side cards — up to 4 per row
    cols_per_row = min(len(ranked), 4)
    rows = [ranked[i:i+cols_per_row] for i in range(0, len(ranked), cols_per_row)]

    for row in rows:
        cols = st.columns(len(row))
        for col_idx, (ticker, data) in enumerate(row):
            rec = data["recommendation"]
            ci  = data["company_info"]
            fd  = data.get("fund_data", {})
            vc  = rec.score_color if hasattr(rec, "score_color") else "#888"
            rank_idx = ranked.index((ticker, data))
            medal = medals.get(rank_idx, f"#{rank_idx+1}")

            # Score bar helper
            def _bar(label, score, width_px=90):
                pct = (score + 1) / 2 * 100
                clr = "#00C853" if score > 0.15 else "#FF1744" if score < -0.15 else "#FFD54F"
                return (
                    f"<div class='score-row'>"
                    f"<span class='score-label'>{label}</span>"
                    f"<div class='score-track'><div class='score-fill' style='width:{pct:.0f}%;background:{clr}'></div></div>"
                    f"<span class='score-val'>{score:+.2f}</span></div>"
                )

            pe   = fd.get("pe_trailing")
            roe  = fd.get("roe")
            rev  = fd.get("revenue_growth")
            marg = fd.get("net_margin")
            dcf  = data.get("dcf_result")
            mos_str = f"{dcf.margin_of_safety:+.0f}%" if dcf and dcf.margin_of_safety else "—"

            with cols[col_idx]:
                st.markdown(
                    f"<div class='rank-card'>"
                    f"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem'>"
                    f"<span style='color:#555;font-size:0.78rem'>{medal}</span>"
                    f"<span style='background:{vc}22;border:1px solid {vc}55;color:{vc};"
                    f"padding:0.15rem 0.5rem;border-radius:5px;font-size:0.72rem;font-weight:700'>"
                    f"{rec.recommendation}</span></div>"
                    f"<div style='font-weight:700;color:#fff;font-size:1rem;line-height:1.2'>{ticker}</div>"
                    f"<div style='color:#555;font-size:0.75rem;margin-bottom:0.6rem'>{ci.get('name','')[:28]}</div>"
                    f"<div style='font-size:1.3rem;font-weight:600;color:#fff;margin-bottom:0.1rem'>"
                    f"${ci.get('price',0):.2f}</div>"
                    f"<div style='color:#555;font-size:0.72rem;margin-bottom:0.7rem'>"
                    f"P/E {pe:.1f}" if pe else "P/E —"
                    f"  ·  ROE {roe*100:.1f}%" if roe else "  ·  ROE —"
                    f"  ·  DCF MoS {mos_str}</div>"
                    f"{_bar('Fundamental', rec.fundamental.overall_score)}"
                    f"{_bar('Technical', rec.technical.overall_score)}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    st.divider()

    # ---- Comparison dashboard tabs ----
    tab_rankings, tab_compare, tab_deep = st.tabs([
        "📋 Rankings & Scores",
        "📊 Comparison Charts",
        "🔍 Deep Dive",
    ])

    # ---- Rankings & Scores table ----
    with tab_rankings:
        rows = []
        for ticker, data in stock_results.items():
            ci = data["company_info"]
            fd = data["fund_data"]
            rec = data["recommendation"]
            rows.append({
                "Ticker": ticker,
                "Company": ci["name"],
                "Sector": ci["sector"],
                "Price": ci["price"],
                "Market Cap": ci["market_cap"],
                "Recommendation": rec.recommendation,
                "Overall Score": round(rec.overall_score, 3),
                "Fund. Score": round(rec.fundamental.overall_score, 3),
                "Tech. Score": round(rec.technical.overall_score, 3),
                "Confidence": rec.confidence,
                "P/E": fd.get("pe_trailing"),
                "PEG": fd.get("peg_ratio"),
                "ROE": fd.get("roe"),
                "Revenue Growth": fd.get("revenue_growth"),
                "Net Margin": fd.get("net_margin"),
                "Debt/Equity": fd.get("debt_to_equity"),
                "Dividend Yield": fd.get("dividend_yield"),
                "Beta": fd.get("beta"),
            })

        df_table = pd.DataFrame(rows)
        df_table = df_table.sort_values("Overall Score", ascending=False).reset_index(drop=True)
        df_table.index = df_table.index + 1
        df_table.index.name = "Rank"

        format_dict = {
            "Price": "${:.2f}",
            "Overall Score": "{:+.3f}",
            "Fund. Score": "{:+.3f}",
            "Tech. Score": "{:+.3f}",
        }

        def color_score(val):
            if pd.isna(val):
                return ""
            try:
                v = float(val)
                c = score_to_color(v)
                return f"color: {c}; font-weight: 600"
            except (ValueError, TypeError):
                return ""

        def format_market_cap(val):
            if pd.isna(val) or val is None:
                return "N/A"
            if val >= 1e12:
                return f"${val/1e12:.2f}T"
            if val >= 1e9:
                return f"${val/1e9:.1f}B"
            if val >= 1e6:
                return f"${val/1e6:.0f}M"
            return f"${val:,.0f}"

        display_df = df_table.copy()
        display_df["Market Cap"] = display_df["Market Cap"].apply(format_market_cap)
        display_df["Price"] = display_df["Price"].apply(lambda x: f"${x:.2f}" if x else "N/A")
        for col in ["ROE", "Revenue Growth", "Net Margin"]:
            display_df[col] = display_df[col].apply(lambda x: f"{x*100:.1f}%" if x is not None else "N/A")
        display_df["Dividend Yield"] = display_df["Dividend Yield"].apply(lambda x: f"{x*100:.2f}%" if x is not None else "—")
        display_df["P/E"] = display_df["P/E"].apply(lambda x: f"{x:.1f}" if x is not None else "N/A")
        display_df["PEG"] = display_df["PEG"].apply(lambda x: f"{x:.2f}" if x is not None else "N/A")
        display_df["Debt/Equity"] = display_df["Debt/Equity"].apply(lambda x: f"{x:.1f}" if x is not None else "N/A")
        display_df["Beta"] = display_df["Beta"].apply(lambda x: f"{x:.2f}" if x is not None else "N/A")

        st.dataframe(
            display_df.style.applymap(color_score, subset=["Overall Score", "Fund. Score", "Tech. Score"]),
            use_container_width=True,
            height=min(600, 60 + len(display_df) * 38),
        )

    # ---- Comparison charts ----
    with tab_compare:
        st.markdown("#### Score Comparison")
        st.plotly_chart(create_comparison_bar(stock_results), use_container_width=True)

        st.markdown("#### Price Performance")
        st.plotly_chart(create_normalized_price_chart(stock_results, period), use_container_width=True)

        st.markdown("#### Category Radar Overlay")
        st.plotly_chart(create_multi_radar(stock_results), use_container_width=True)

        st.markdown("#### Key Metrics Comparison")
        metric_charts = [
            ("P/E Ratio (TTM)", "pe_trailing", ".1f"),
            ("Return on Equity", "roe", ".2%"),
            ("Revenue Growth", "revenue_growth", ".1%"),
            ("Net Margin", "net_margin", ".1%"),
            ("Debt to Equity", "debt_to_equity", ".1f"),
            ("PEG Ratio", "peg_ratio", ".2f"),
        ]
        chart_cols = st.columns(2)
        for i, (name, key, fmt) in enumerate(metric_charts):
            fig = create_metric_comparison_chart(stock_results, name, key, fmt)
            if fig:
                chart_cols[i % 2].plotly_chart(fig, use_container_width=True)

    # ---- Deep Dive ----
    with tab_deep:
        st.markdown("#### Select a stock for full analysis")
        selected = st.selectbox(
            "Stock",
            options=list(stock_results.keys()),
            format_func=lambda t: f"{t} — {stock_results[t]['company_info']['name']} "
                                  f"({stock_results[t]['recommendation'].recommendation_emoji} "
                                  f"{stock_results[t]['recommendation'].recommendation})",
        )
        if selected:
            ci = stock_results[selected]["company_info"]
            st.markdown(f"### {ci['name']} ({selected})")
            st.caption(f"{ci['sector']}  •  {ci['industry']}")
            render_stock_detail(selected, stock_results[selected])
else:
    st.markdown(
        "<div style='max-width:640px;margin:3rem auto 0 auto;text-align:center'>"
        "<div style='font-size:2.4rem;font-weight:700;color:#fff;margin-bottom:0.3rem'>"
        "Financial Analyst</div>"
        "<div style='font-size:0.9rem;color:#444;letter-spacing:0.06em;text-transform:uppercase;"
        "margin-bottom:2.5rem'>Equity Research Platform</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    lc, mc, rc = st.columns([1, 2, 1])
    with mc:
        st.markdown(
            "<div style='background:rgba(15,17,30,0.8);border:1px solid rgba(255,255,255,0.07);"
            "border-radius:12px;padding:1.6rem 2rem'>"
            "<div style='font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;"
            "color:#444;margin-bottom:1rem'>WHAT THIS TOOL COVERS</div>"
            "<div style='font-size:0.85rem;color:#888;line-height:1.9'>"
            "📊  Fundamental — valuation, margins, growth, financial health<br>"
            "📈  Technical — SMA/EMA, RSI, MACD, Bollinger, volume, ADX<br>"
            "📐  DCF — 3-stage intrinsic value + sensitivity table<br>"
            "🏆  Quality — Piotroski F, Altman Z, Beneish M scores<br>"
            "🏢  Peer Comps — percentile ranks vs sector<br>"
            "⚙️  Options — IV rank, put/call ratio, max pain<br>"
            "🛡️  Risk — Sharpe, Sortino, Calmar, VaR, drawdown<br>"
            "🤖  AI — Claude-powered thesis, bull/bear case<br>"
            "📄  Export — full PDF research report"
            "</div>"
            "<div style='margin-top:1.2rem;padding-top:1rem;"
            "border-top:1px solid rgba(255,255,255,0.06);"
            "font-size:0.75rem;color:#333;text-align:center'>"
            "Enter a ticker in the sidebar and click Analyze"
            "</div></div>",
            unsafe_allow_html=True,
        )
