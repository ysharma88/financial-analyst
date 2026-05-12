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
@st.cache_data(ttl=900, show_spinner=False)
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
@st.cache_data(ttl=1800, show_spinner=False)
def run_macro_analysis():
    analyzer = MacroAnalyzer()
    return analyzer.analyze()


@st.cache_data(ttl=1800, show_spinner=False)
def run_sector_analysis(cycle_phase: str, rates_rising: bool):
    analyzer = SectorAnalyzer()
    return analyzer.analyze(cycle_phase=cycle_phase, rates_rising=rates_rising)


@st.cache_data(ttl=1800, show_spinner=False)
def run_stock_screener(sector: str, max_stocks: int, min_mc: float, max_pe, max_de, min_roe, min_rg):
    screener = StockScreener()
    return screener.screen_sector(
        sector=sector, max_stocks=max_stocks,
        min_market_cap=min_mc, max_pe=max_pe,
        max_debt_equity=max_de, min_roe=min_roe,
        min_revenue_growth=min_rg,
    )


@st.cache_data(ttl=900, show_spinner=False)
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
    """Render the full analysis for one stock inside an expander."""
    company_info = data["company_info"]
    fund_data = data["fund_data"]
    fund_result = data["fund_result"]
    tech_result = data["tech_result"]
    chart_df = data["chart_df"]
    recommendation = data["recommendation"]
    governance = data.get("governance")

    # Prefetch macro / sector / news (cached, so no extra API cost)
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

    # --- REASONING ENGINE ---
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

    # Top metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Price", f"${company_info['price']:.2f}" if company_info['price'] else "N/A")
    col2.metric("Market Cap", format_large_number(company_info['market_cap']))
    col3.metric("52-Week Range", f"${company_info['52w_low']:.0f} - ${company_info['52w_high']:.0f}"
                if company_info['52w_low'] and company_info['52w_high'] else "N/A")
    col4.metric("Beta", f"{company_info['beta']:.2f}" if company_info['beta'] else "N/A")
    col5.metric("Avg Volume", f"{company_info['avg_volume']:,.0f}" if company_info['avg_volume'] else "N/A")

    st.divider()

    # =====================================================================
    # VERDICT SECTION — Multi-Factor Reasoning Summary
    # =====================================================================
    vc = verdict.score_color
    action_emoji = {"STRONG BUY": "🟢", "BUY": "🟩", "HOLD": "🟡", "SELL": "🟧", "STRONG SELL": "🔴"}.get(verdict.action, "⚪")

    # Main verdict banner
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

    # Pillar-by-pillar reasoning chain
    st.markdown("##### Reasoning Chain — 7-Pillar Analysis")
    signal_colors = {"BULLISH": "#00C853", "NEUTRAL": "#FFD54F", "CAUTION": "#FFA726", "BEARISH": "#FF1744"}
    signal_icons = {"BULLISH": "🟢", "NEUTRAL": "🟡", "CAUTION": "🟠", "BEARISH": "🔴"}
    pillar_icons = {"Governance": "🏛️", "Macro": "🌍", "Sector": "📊", "Fundamental": "💰",
                    "Technical": "📈", "Sentiment": "📰", "Risk": "🛡️"}
    pillar_bg = {
        "Governance":  "linear-gradient(135deg, rgba(171,71,188,0.10), rgba(171,71,188,0.03))",
        "Macro":       "linear-gradient(135deg, rgba(38,166,154,0.10), rgba(38,166,154,0.03))",
        "Sector":      "linear-gradient(135deg, rgba(66,165,245,0.10), rgba(66,165,245,0.03))",
        "Fundamental": "linear-gradient(135deg, rgba(255,167,38,0.10), rgba(255,167,38,0.03))",
        "Technical":   "linear-gradient(135deg, rgba(102,126,234,0.10), rgba(102,126,234,0.03))",
        "Sentiment":   "linear-gradient(135deg, rgba(239,83,80,0.10), rgba(239,83,80,0.03))",
        "Risk":        "linear-gradient(135deg, rgba(255,202,40,0.10), rgba(255,202,40,0.03))",
    }
    pillar_accent = {
        "Governance": "#AB47BC", "Macro": "#26A69A", "Sector": "#42A5F5",
        "Fundamental": "#FFA726", "Technical": "#667eea", "Sentiment": "#EF5350", "Risk": "#FFCA28",
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
            f"<div style='color:#e0e0e0;font-size:0.85rem;margin:0.35rem 0 0.3rem 1.8rem;font-weight:600'>"
            f"{step.headline}</div>"
            f"<div style='color:#aaa;font-size:0.8rem;margin:0 0 0.4rem 1.8rem;line-height:1.5'>"
            f"{step.detail}</div>"
            f"<div style='background:rgba(255,255,255,0.06);height:5px;border-radius:3px;margin:0 0 0 1.8rem'>"
            f"<div style='background:{bar_color};height:5px;border-radius:3px;width:{bar_pct:.0f}%;"
            f"transition:width 0.3s'></div></div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # Bull / Bear / Catalysts / Risks / Action Plan
    bb_col, ca_col = st.columns(2)
    with bb_col:
        st.markdown("##### Bull Case")
        st.markdown(f"<div style='padding:0.7rem;background:rgba(0,200,83,0.06);"
                    f"border:1px solid rgba(0,200,83,0.2);border-radius:10px;"
                    f"color:#a5d6a7;font-size:0.88rem;line-height:1.6'>{verdict.bull_case}</div>",
                    unsafe_allow_html=True)

        st.markdown("##### Bear Case")
        st.markdown(f"<div style='padding:0.7rem;background:rgba(255,23,68,0.06);"
                    f"border:1px solid rgba(255,23,68,0.2);border-radius:10px;"
                    f"color:#ef9a9a;font-size:0.88rem;line-height:1.6'>{verdict.bear_case}</div>",
                    unsafe_allow_html=True)

    with ca_col:
        if verdict.catalysts:
            st.markdown("##### Catalysts to Watch")
            for cat in verdict.catalysts:
                st.markdown(f"<div style='padding:0.35rem 0.6rem;background:rgba(102,126,234,0.08);"
                            f"border-left:3px solid #667eea;border-radius:0 6px 6px 0;margin-bottom:0.3rem;"
                            f"color:#b0bec5;font-size:0.85rem'>⚡ {cat}</div>",
                            unsafe_allow_html=True)

        if verdict.key_risks:
            st.markdown("##### Key Risks")
            for risk in verdict.key_risks:
                st.markdown(f"<div style='padding:0.35rem 0.6rem;background:rgba(255,87,34,0.08);"
                            f"border-left:3px solid #FF5722;border-radius:0 6px 6px 0;margin-bottom:0.3rem;"
                            f"color:#ffab91;font-size:0.85rem'>⚠️ {risk}</div>",
                            unsafe_allow_html=True)

    # Action plan
    if verdict.action_plan:
        st.markdown("##### Action Plan")
        st.markdown(
            f"<div style='padding:0.8rem 1rem;background:linear-gradient(135deg,rgba(102,126,234,0.08),rgba(118,75,162,0.08));"
            f"border:1px solid rgba(102,126,234,0.25);border-radius:10px;"
            f"color:#ccc;font-size:0.88rem;line-height:1.7'>{verdict.action_plan}</div>",
            unsafe_allow_html=True,
        )

    # Gauges (kept for quick visual reference)
    st.divider()
    rec_col1, rec_col2, rec_col3 = st.columns(3)
    with rec_col1:
        st.plotly_chart(create_gauge(fund_result.overall_score, "Fundamental"), use_container_width=True)
    with rec_col2:
        st.plotly_chart(create_gauge(tech_result.overall_score, "Technical"), use_container_width=True)
    with rec_col3:
        st.plotly_chart(create_gauge(verdict.composite_score, "Multi-Factor"), use_container_width=True)

    st.divider()

    # Tabs within each stock
    tab_tech, tab_fund, tab_valuation, tab_peers, tab_options, tab_inst, tab_altdata, tab_forensic, tab_macro, tab_screener, tab_news, tab_gov, tab_risk, tab_analysts, tab_detail = st.tabs([
        "📈 Technical", "📊 Fundamental", "📐 Valuation",
        "🏢 Peer Comps", "⚙️ Options", "🏦 Institutional Risk",
        "🔭 Alternative Data", "🧬 Forensic NLP",
        "🌍 Macro & Sector", "🔎 Screener",
        "📰 News & Sentiment", "🚩 Red Flags", "🛡️ Risk Management", "📋 Analysts & Investors", "🔬 Scores",
    ])

    with tab_tech:
        sr_levels = getattr(tech_result, 'support_resistance', [])
        st.plotly_chart(create_price_chart(chart_df, ticker, sr_levels=sr_levels), use_container_width=True)
        st.plotly_chart(create_volume_chart(chart_df), use_container_width=True)

        # Support & Resistance section
        if sr_levels:
            st.markdown("##### Support & Resistance Levels")
            current_price = company_info.get("price", 0)
            supports = [l for l in sr_levels if l.kind == "support"]
            resistances = [l for l in sr_levels if l.kind == "resistance"]

            sr_col1, sr_col2 = st.columns(2)
            with sr_col1:
                st.markdown("**Support Levels** (price floor)")
                if supports:
                    for lvl in sorted(supports, key=lambda l: l.price, reverse=True):
                        dist_pct = (current_price - lvl.price) / current_price * 100 if current_price else 0
                        strength_color = "#00C853" if lvl.strength == "strong" else "#69F0AE" if lvl.strength == "moderate" else "#A5D6A7"
                        strength_icon = "🟢" if lvl.strength == "strong" else "🟡" if lvl.strength == "moderate" else "⚪"
                        st.markdown(
                            f"<div style='padding:0.4rem 0.6rem;background:rgba(76,175,80,0.06);"
                            f"border-left:3px solid {strength_color};border-radius:0 8px 8px 0;margin-bottom:0.3rem'>"
                            f"<div style='display:flex;justify-content:space-between;align-items:center'>"
                            f"<span style='color:white;font-weight:600'>{strength_icon} ${lvl.price:.2f}</span>"
                            f"<span style='color:{strength_color};font-size:0.8rem;font-weight:600'>{lvl.strength.upper()}</span>"
                            f"</div>"
                            f"<div style='color:#888;font-size:0.78rem'>"
                            f"{dist_pct:.1f}% below · {lvl.method} · {lvl.touches} touch{'es' if lvl.touches != 1 else ''}"
                            f"</div></div>",
                            unsafe_allow_html=True,
                        )
                else:
                    st.caption("No support levels identified in this range.")

            with sr_col2:
                st.markdown("**Resistance Levels** (price ceiling)")
                if resistances:
                    for lvl in sorted(resistances, key=lambda l: l.price):
                        dist_pct = (lvl.price - current_price) / current_price * 100 if current_price else 0
                        strength_color = "#FF1744" if lvl.strength == "strong" else "#FF8A65" if lvl.strength == "moderate" else "#FFAB91"
                        strength_icon = "🔴" if lvl.strength == "strong" else "🟠" if lvl.strength == "moderate" else "⚪"
                        st.markdown(
                            f"<div style='padding:0.4rem 0.6rem;background:rgba(255,23,68,0.06);"
                            f"border-left:3px solid {strength_color};border-radius:0 8px 8px 0;margin-bottom:0.3rem'>"
                            f"<div style='display:flex;justify-content:space-between;align-items:center'>"
                            f"<span style='color:white;font-weight:600'>{strength_icon} ${lvl.price:.2f}</span>"
                            f"<span style='color:{strength_color};font-size:0.8rem;font-weight:600'>{lvl.strength.upper()}</span>"
                            f"</div>"
                            f"<div style='color:#888;font-size:0.78rem'>"
                            f"{dist_pct:.1f}% above · {lvl.method} · {lvl.touches} touch{'es' if lvl.touches != 1 else ''}"
                            f"</div></div>",
                            unsafe_allow_html=True,
                        )
                else:
                    st.caption("No resistance levels identified in this range.")

            # Nearest support/resistance summary
            if supports and resistances:
                nearest_sup = max(supports, key=lambda l: l.price)
                nearest_res = min(resistances, key=lambda l: l.price)
                range_pct = (nearest_res.price - nearest_sup.price) / current_price * 100 if current_price else 0
                st.caption(
                    f"📐 **Trading range**: ${nearest_sup.price:.2f} (support) → "
                    f"${nearest_res.price:.2f} (resistance) = {range_pct:.1f}% width. "
                    f"Current price sits at {((current_price - nearest_sup.price) / (nearest_res.price - nearest_sup.price) * 100):.0f}% of range."
                    if nearest_res.price > nearest_sup.price else ""
                )

            st.divider()

        st.markdown("##### Technical Indicator Values")
        ind = tech_result.indicators
        if ind:
            ind_cols = st.columns(4)
            for i, (k, v) in enumerate(ind.items()):
                ind_cols[i % 4].metric(k.replace("_", " "), f"{v:,.2f}" if isinstance(v, float) else str(v))

        st.markdown("##### Technical Signals")
        for sig in tech_result.signals:
            if sig.value is not None:
                color = score_to_color(sig.score)
                st.markdown(
                    f"<span style='color:{color};font-weight:600'>{sig.name}</span> "
                    f"<span style='color:#888'>({sig.category})</span>: "
                    f"{sig.interpretation} "
                    f"<span style='color:{color}'>[{sig.score:+.2f}]</span>",
                    unsafe_allow_html=True,
                )

    with tab_fund:
        st.markdown("##### Valuation Metrics")
        vc = st.columns(4)
        vc[0].metric("P/E (TTM)", f"{fund_data['pe_trailing']:.1f}" if fund_data.get('pe_trailing') else "N/A")
        vc[1].metric("Forward P/E", f"{fund_data['pe_forward']:.1f}" if fund_data.get('pe_forward') else "N/A")
        vc[2].metric("PEG Ratio", f"{fund_data['peg_ratio']:.2f}" if fund_data.get('peg_ratio') else "N/A")
        vc[3].metric("EV/EBITDA", f"{fund_data['ev_ebitda']:.1f}" if fund_data.get('ev_ebitda') else "N/A")

        vc2 = st.columns(4)
        vc2[0].metric("P/B Ratio", f"{fund_data['pb_ratio']:.2f}" if fund_data.get('pb_ratio') else "N/A")
        vc2[1].metric("P/S Ratio", f"{fund_data['ps_ratio']:.2f}" if fund_data.get('ps_ratio') else "N/A")
        vc2[2].metric("EPS (TTM)", f"${fund_data['eps_trailing']:.2f}" if fund_data.get('eps_trailing') else "N/A")
        vc2[3].metric("Book Value", f"${fund_data['book_value']:.2f}" if fund_data.get('book_value') else "N/A")

        st.divider()
        st.markdown("##### Profitability")
        pc = st.columns(5)
        pc[0].metric("ROE", format_pct(fund_data.get('roe'), mult100=True))
        pc[1].metric("ROA", format_pct(fund_data.get('roa'), mult100=True))
        pc[2].metric("Gross Margin", format_pct(fund_data.get('gross_margin'), mult100=True))
        pc[3].metric("Op. Margin", format_pct(fund_data.get('operating_margin'), mult100=True))
        pc[4].metric("Net Margin", format_pct(fund_data.get('net_margin'), mult100=True))

        st.divider()
        st.markdown("##### Growth")
        gc = st.columns(3)
        gc[0].metric("Revenue Growth", format_pct(fund_data.get('revenue_growth'), mult100=True))
        gc[1].metric("Earnings Growth", format_pct(fund_data.get('earnings_growth'), mult100=True))
        gc[2].metric("Quarterly Earnings Growth", format_pct(fund_data.get('earnings_quarterly_growth'), mult100=True))

        st.divider()
        st.markdown("##### Financial Health")
        hc = st.columns(4)
        de = fund_data.get('debt_to_equity')
        hc[0].metric("Debt/Equity", f"{de:.1f}" if de else "N/A")
        hc[1].metric("Current Ratio", f"{fund_data['current_ratio']:.2f}" if fund_data.get('current_ratio') else "N/A")
        hc[2].metric("Free Cash Flow", format_large_number(fund_data.get('free_cashflow')))
        hc[3].metric("Total Cash", format_large_number(fund_data.get('total_cash')))

        st.divider()
        st.markdown("##### Dividends & Analyst")
        dc = st.columns(4)
        dy = fund_data.get('dividend_yield')
        dc[0].metric("Dividend Yield", f"{dy * 100:.2f}%" if dy else "N/A")
        dc[1].metric("Payout Ratio", f"{fund_data['payout_ratio']:.0%}" if fund_data.get('payout_ratio') else "N/A")
        dc[2].metric("Analyst Target", f"${fund_data['target_mean_price']:.2f}" if fund_data.get('target_mean_price') else "N/A")
        rec_key = fund_data.get('recommendation_key', 'N/A')
        dc[3].metric("Analyst Consensus", rec_key.upper() if rec_key else "N/A")

    with tab_valuation:
        dcf = data.get("dcf_result")
        qs = data.get("quality_scores")
        trends = data.get("financial_trends", {})
        cal = data.get("event_calendar", {})

        # ---- DCF Intrinsic Value ----
        st.markdown("##### DCF Intrinsic Value (3-Stage Model)")
        if dcf and dcf.intrinsic_value and not dcf.error:
            price_now = dcf.current_price or company_info.get("price", 0)
            upside = dcf.upside_pct or 0
            mos    = dcf.margin_of_safety or 0
            iv_color = "#00C853" if mos > 0.10 else "#FFA726" if mos > -0.10 else "#FF1744"
            label_color = "#00C853" if "Under" in dcf.valuation_label else "#FF1744" if "Over" in dcf.valuation_label else "#FFA726"

            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Intrinsic Value", f"${dcf.intrinsic_value:.2f}")
            d2.metric("Current Price", f"${price_now:.2f}" if price_now else "N/A")
            d3.metric("Upside / Downside", f"{upside:+.1%}" if dcf.upside_pct is not None else "N/A",
                      delta=f"{upside:+.1%}" if dcf.upside_pct is not None else None)
            d4.metric("Margin of Safety", f"{mos:.1%}" if dcf.margin_of_safety is not None else "N/A")

            st.markdown(
                f"<div style='padding:0.6rem 1rem;background:{iv_color}18;border:1px solid {iv_color}66;"
                f"border-radius:8px;margin:0.5rem 0'>"
                f"<span style='color:{label_color};font-weight:700'>{dcf.valuation_label}</span>"
                f"  —  FCF Base: {format_large_number(dcf.fcf_base)}  ·  "
                f"WACC: {dcf.wacc*100:.1f}%  ·  "
                f"Stage 1 Growth: {dcf.growth_stage1*100:.1f}%  ·  "
                f"Stage 2 Growth: {dcf.growth_stage2*100:.1f}%  ·  "
                f"Terminal Growth: {dcf.terminal_growth*100:.1f}%"
                f"</div>",
                unsafe_allow_html=True,
            )

            # Sensitivity heatmap
            st.markdown("**Sensitivity Analysis** — Intrinsic Value by WACC and Growth Rate")
            if dcf.sensitivity:
                wacc_ds = sorted(set(k[0] for k in dcf.sensitivity))
                grow_ds = sorted(set(k[1] for k in dcf.sensitivity))
                z_vals = [[dcf.sensitivity.get((wd, gd)) for gd in grow_ds] for wd in wacc_ds]
                # Replace None with 0
                z_clean = [[v if v is not None else 0 for v in row] for row in z_vals]
                sen_fig = go.Figure(go.Heatmap(
                    z=z_clean,
                    x=[f"Growth {g:+.1f}%" for g in grow_ds],
                    y=[f"WACC {w:+.1f}%" for w in wacc_ds],
                    colorscale="RdYlGn",
                    text=[[f"${v:.0f}" if v else "N/A" for v in row] for row in z_clean],
                    texttemplate="%{text}",
                    showscale=True,
                ))
                if price_now:
                    # Draw line where IV = current price
                    sen_fig.add_shape(type="line", x0=-0.5, x1=len(grow_ds)-0.5,
                                      y0=-0.5, y1=-0.5, line=dict(color="white", dash="dot"))
                sen_fig.update_layout(
                    template="plotly_dark", height=280,
                    margin=dict(l=10, r=10, t=30, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    title=f"Current Price: ${price_now:.2f}" if price_now else "",
                )
                st.plotly_chart(sen_fig, use_container_width=True)
        elif dcf and dcf.error:
            st.info(f"DCF not computed: {dcf.error}")
        else:
            st.info("DCF valuation data unavailable.")

        st.divider()

        # ---- Quality Scores ----
        st.markdown("##### Quality & Distress Scores")
        if qs:
            q1, q2, q3 = st.columns(3)

            # Piotroski
            pf = qs.piotroski_f
            pf_color = "#00C853" if pf is not None and pf >= 7 else "#FFA726" if pf is not None and pf >= 4 else "#FF1744"
            q1.markdown(
                f"<div style='text-align:center;padding:1rem;background:{pf_color}18;"
                f"border:1px solid {pf_color}66;border-radius:10px'>"
                f"<div style='color:{pf_color};font-size:2.5rem;font-weight:800'>{pf if pf is not None else 'N/A'}<span style='font-size:1rem'>/9</span></div>"
                f"<div style='color:#ccc;font-weight:600'>Piotroski F-Score</div>"
                f"<div style='color:#888;font-size:0.78rem'>{qs.piotroski_label}</div>"
                f"</div>", unsafe_allow_html=True,
            )

            # Altman Z
            az = qs.altman_z
            az_color = "#00C853" if az is not None and az > 2.99 else "#FFA726" if az is not None and az > 1.81 else "#FF1744"
            q2.markdown(
                f"<div style='text-align:center;padding:1rem;background:{az_color}18;"
                f"border:1px solid {az_color}66;border-radius:10px'>"
                f"<div style='color:{az_color};font-size:2.5rem;font-weight:800'>{f'{az:.2f}' if az is not None else 'N/A'}</div>"
                f"<div style='color:#ccc;font-weight:600'>Altman Z-Score</div>"
                f"<div style='color:#888;font-size:0.78rem'>{qs.altman_zone or 'N/A'}</div>"
                f"</div>", unsafe_allow_html=True,
            )

            # Beneish M
            bm = qs.beneish_m
            bm_color = "#FF1744" if qs.beneish_flag else "#00C853"
            bm_label = "⚠️ Manipulation Risk" if qs.beneish_flag else "✅ Low Manipulation Risk"
            q3.markdown(
                f"<div style='text-align:center;padding:1rem;background:{bm_color}18;"
                f"border:1px solid {bm_color}66;border-radius:10px'>"
                f"<div style='color:{bm_color};font-size:2.5rem;font-weight:800'>{f'{bm:.2f}' if bm is not None else 'N/A'}</div>"
                f"<div style='color:#ccc;font-weight:600'>Beneish M-Score</div>"
                f"<div style='color:#888;font-size:0.78rem'>{bm_label}</div>"
                f"</div>", unsafe_allow_html=True,
            )

            # Piotroski signal breakdown
            if qs.piotroski_signals:
                st.markdown("**Piotroski Signal Breakdown**")
                sig_cols = st.columns(min(len(qs.piotroski_signals), 5))
                for i, (sig_name, sig_val) in enumerate(qs.piotroski_signals.items()):
                    c = sig_cols[i % len(sig_cols)]
                    icon = "✅" if sig_val else "❌"
                    c.markdown(f"{icon} {sig_name}")

        st.divider()

        # ---- Financial Trends ----
        st.markdown("##### Financial Trends (Multi-Period)")
        if trends:
            tc1, tc2, tc3, tc4 = st.columns(4)

            def trend_metric(col, label, val):
                if val is not None:
                    delta_str = f"{val:+.1%}"
                    col.metric(label, delta_str, delta=delta_str)
                else:
                    col.metric(label, "N/A")

            trend_metric(tc1, "Revenue YoY", trends.get("revenue_yoy"))
            trend_metric(tc2, "Net Income YoY", trends.get("net_income_yoy"))
            trend_metric(tc3, "FCF YoY", trends.get("fcf_yoy"))
            trend_metric(tc4, "Op. Income YoY", trends.get("operating_income_yoy"))

            # Revenue trend chart
            rev_series = trends.get("revenue_series", {})
            ni_series = trends.get("net_income_series", {})
            fcf_series = trends.get("fcf_series", {})
            if rev_series:
                dates = sorted(rev_series.keys())
                trend_fig = go.Figure()
                trend_fig.add_trace(go.Bar(
                    x=dates, y=[rev_series[d]/1e9 for d in dates],
                    name="Revenue ($B)", marker_color="#42A5F5",
                ))
                if ni_series:
                    ni_dates = sorted(ni_series.keys())
                    trend_fig.add_trace(go.Bar(
                        x=ni_dates, y=[ni_series[d]/1e9 for d in ni_dates],
                        name="Net Income ($B)", marker_color="#00C853",
                    ))
                if fcf_series:
                    fcf_dates = sorted(fcf_series.keys())
                    trend_fig.add_trace(go.Scatter(
                        x=fcf_dates, y=[fcf_series[d]/1e9 for d in fcf_dates],
                        name="FCF ($B)", line=dict(color="#FFA726", width=2), mode="lines+markers",
                    ))
                trend_fig.update_layout(
                    template="plotly_dark", barmode="group", height=320,
                    margin=dict(l=10, r=10, t=30, b=10),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    yaxis_title="$B",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                st.plotly_chart(trend_fig, use_container_width=True)

        st.divider()

        # ---- Event Calendar ----
        st.markdown("##### Upcoming Events")
        if cal:
            ev1, ev2, ev3 = st.columns(3)
            # Earnings date
            earn_dates = cal.get("Earnings Date") or cal.get("earningsDate") or cal.get("Earnings Date", [])
            if isinstance(earn_dates, list) and earn_dates:
                earn_str = str(earn_dates[0])[:10]
            elif isinstance(earn_dates, str):
                earn_str = earn_dates[:10]
            else:
                earn_str = "N/A"
            ev1.metric("Next Earnings", earn_str)

            # Ex-dividend date
            ex_div = cal.get("Ex-Dividend Date") or cal.get("exDividendDate") or "N/A"
            ev2.metric("Ex-Dividend Date", str(ex_div)[:10] if ex_div and ex_div != "N/A" else "N/A")

            # Earnings estimate
            eps_est = cal.get("EPS Estimate") or cal.get("epsEstimate")
            ev3.metric("EPS Estimate", f"${eps_est:.2f}" if eps_est else "N/A")
        else:
            st.info("Calendar data unavailable for this ticker.")

    # ------------------------------------------------------------------
    with tab_peers:
        st.markdown(f"##### Peer Comparable Analysis — {company_info['name']} vs Sector Peers")
        peer_result = data.get("peer_comps_result")

        if peer_result and len(peer_result.peers) > 1:
            from peer_comps import MULTIPLES

            subject = next((p for p in peer_result.peers if p.is_subject), None)
            peers_only = [p for p in peer_result.peers if not p.is_subject]

            # ---- Percentile rank summary ----
            if peer_result.percentile_ranks:
                st.markdown("##### Where does this stock rank in its peer group?")
                rank_cols = st.columns(min(len(peer_result.percentile_ranks), 6))
                label_map = {f: lbl for f, lbl in MULTIPLES}
                for i, (field_name, pct) in enumerate(peer_result.percentile_ranks.items()):
                    col = rank_cols[i % len(rank_cols)]
                    color = "#00C853" if pct >= 60 else "#FFA726" if pct >= 40 else "#FF5722"
                    col.markdown(
                        f"<div style='text-align:center;padding:0.5rem 0.3rem;"
                        f"background:{color}18;border:1px solid {color}55;border-radius:8px;margin-bottom:0.4rem'>"
                        f"<div style='color:{color};font-size:1.3rem;font-weight:700'>{pct:.0f}<span style='font-size:0.7rem'>th</span></div>"
                        f"<div style='color:#ccc;font-size:0.72rem'>{label_map.get(field_name, field_name)}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                st.caption("Percentile rank vs sector peers — higher is better (valuation multiples adjusted: lower P/E = better rank)")
                st.divider()

            # ---- Comps table ----
            st.markdown("##### Multiples Comparison Table")
            rows = []
            for p in peer_result.peers:
                def fv(v, pct=False):
                    if v is None: return "—"
                    return f"{v*100:.1f}%" if pct else f"{v:.2f}" if v < 100 else f"{v:.0f}"

                rows.append({
                    "Ticker":       f"★ {p.ticker}" if p.is_subject else p.ticker,
                    "Company":      (p.name or "")[:28],
                    "Price":        f"${p.price:.2f}" if p.price else "—",
                    "P/E":          fv(p.pe_trailing),
                    "Fwd P/E":      fv(p.pe_forward),
                    "EV/EBITDA":    fv(p.ev_ebitda),
                    "P/S":          fv(p.ps_ratio),
                    "P/B":          fv(p.pb_ratio),
                    "PEG":          fv(p.peg_ratio),
                    "ROE":          fv(p.roe, pct=True),
                    "Net Margin":   fv(p.net_margin, pct=True),
                    "Rev Growth":   fv(p.revenue_growth, pct=True),
                    "D/E":          fv(p.debt_to_equity),
                })
            comps_df = pd.DataFrame(rows)

            def highlight_subject(row):
                if row["Ticker"].startswith("★"):
                    return ["background-color: rgba(66,165,245,0.15); font-weight: 600"] * len(row)
                return [""] * len(row)

            st.dataframe(
                comps_df.style.apply(highlight_subject, axis=1),
                use_container_width=True,
                hide_index=True,
            )

            # ---- Sector medians ----
            if peer_result.medians:
                st.divider()
                st.markdown("##### Sector Medians")
                med_cols = st.columns(min(len(peer_result.medians), 6))
                for i, (field_name, med) in enumerate(peer_result.medians.items()):
                    lbl = label_map.get(field_name, field_name)
                    subj_val = getattr(subject, field_name, None) if subject else None
                    is_pct = field_name in ("roe", "net_margin", "gross_margin", "revenue_growth")
                    fmt = lambda v: f"{v*100:.1f}%" if is_pct and v else (f"{v:.2f}" if v else "—")
                    delta = None
                    if subj_val and med:
                        delta = f"{((subj_val - med) / abs(med)) * 100:+.1f}% vs median"
                    med_cols[i % len(med_cols)].metric(f"Median {lbl}", fmt(med), delta=delta)

            # ---- Visual: selected multiple bar chart ----
            st.divider()
            chart_multiple = st.selectbox(
                "Visualize multiple",
                options=[f for f, _ in MULTIPLES if any(getattr(p, f) for p in peer_result.peers)],
                format_func=lambda f: dict(MULTIPLES).get(f, f),
                key=f"peer_chart_select_{ticker}",
            )
            if chart_multiple:
                chart_peers = [p for p in peer_result.peers if getattr(p, chart_multiple)]
                chart_vals = [getattr(p, chart_multiple) for p in chart_peers]
                is_pct_chart = chart_multiple in ("roe", "net_margin", "gross_margin", "revenue_growth")
                display_vals = [v * 100 if is_pct_chart else v for v in chart_vals]
                bar_colors = ["#42A5F5" if p.is_subject else "#546E7A" for p in chart_peers]
                peer_fig = go.Figure(go.Bar(
                    x=[p.ticker for p in chart_peers],
                    y=display_vals,
                    marker_color=bar_colors,
                    text=[f"{v:.1f}{'%' if is_pct_chart else 'x'}" for v in display_vals],
                    textposition="auto",
                ))
                if peer_result.medians.get(chart_multiple):
                    med_v = peer_result.medians[chart_multiple]
                    peer_fig.add_hline(
                        y=med_v * 100 if is_pct_chart else med_v,
                        line_dash="dash", line_color="#FFA726",
                        annotation_text=f"Median: {med_v*100:.1f}{'%' if is_pct_chart else 'x'}",
                    )
                peer_fig.update_layout(
                    template="plotly_dark", height=320,
                    margin=dict(l=10, r=10, t=30, b=10),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    yaxis_title=f"{'%' if is_pct_chart else 'x'}",
                )
                st.plotly_chart(peer_fig, use_container_width=True)
        elif peer_result and peer_result.error:
            st.info(f"Peer data unavailable: {peer_result.error}")
        else:
            st.info("Peer comparison data could not be fetched — Yahoo Finance may be rate-limiting.")

    # ------------------------------------------------------------------
    with tab_options:
        st.markdown(f"##### Options Market Signals — {ticker}")
        opt = data.get("options_result")

        if opt and not opt.error:
            # Signal banner
            sig_colors = {
                "High IV (Sell Premium)": "#FF8A65",
                "Low IV (Buy Options)":   "#42A5F5",
                "Bullish (Heavy Call OI)":"#00C853",
                "Bearish (Heavy Put OI)": "#FF1744",
                "Neutral":                "#FFD54F",
            }
            sc = sig_colors.get(opt.signal, "#888")
            st.markdown(
                f"<div style='padding:0.6rem 1rem;background:{sc}18;border:1px solid {sc}66;"
                f"border-radius:8px;margin-bottom:0.8rem'>"
                f"<span style='color:{sc};font-weight:700;font-size:1.1rem'>{opt.signal}</span>"
                f"  <span style='color:#aaa;font-size:0.85rem'>· Expiry: {opt.nearest_expiry} · {opt.summary}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # Key metrics row
            o1, o2, o3, o4, o5 = st.columns(5)
            o1.metric("ATM IV", f"{opt.avg_iv:.1%}" if opt.avg_iv else "N/A",
                      help="Distance-weighted avg implied volatility of near-ATM options")
            o2.metric("30d Realized Vol", f"{opt.realized_vol_30d:.1%}" if opt.realized_vol_30d else "N/A",
                      help="30-day historical volatility (annualised)")
            iv_prem = opt.iv_premium
            o3.metric("IV Premium", f"{iv_prem:+.1%}" if iv_prem is not None else "N/A",
                      delta=f"{'Rich' if iv_prem and iv_prem > 0 else 'Cheap'}",
                      help="IV minus realised vol — positive = options expensive")
            o4.metric("IV Rank", f"{opt.iv_rank:.0f}" if opt.iv_rank is not None else "N/A",
                      help="0–100: where current IV sits relative to recent expiries")
            o5.metric("Max Pain", f"${opt.max_pain_price:.2f}" if opt.max_pain_price else "N/A",
                      help="Strike where total option buyer losses are maximized")

            st.divider()

            # Put/Call ratios
            pc1, pc2, pc3, pc4 = st.columns(4)
            pc1.metric("Put/Call OI Ratio", f"{opt.put_call_oi_ratio:.2f}" if opt.put_call_oi_ratio else "N/A",
                       help=">1 = more put open interest (bearish lean), <1 = call heavy (bullish lean)")
            pc2.metric("Put/Call Vol Ratio", f"{opt.put_call_vol_ratio:.2f}" if opt.put_call_vol_ratio else "N/A")
            pc3.metric("Total Call OI", f"{opt.total_call_oi:,}")
            pc4.metric("Total Put OI", f"{opt.total_put_oi:,}")

            # P/C ratio interpretation
            if opt.put_call_oi_ratio:
                pcr = opt.put_call_oi_ratio
                if pcr > 1.3:
                    st.warning(f"Put/Call OI ratio of **{pcr:.2f}** — heavy put positioning signals bearish sentiment or hedging activity.")
                elif pcr < 0.7:
                    st.success(f"Put/Call OI ratio of **{pcr:.2f}** — call-heavy positioning signals bullish sentiment.")
                else:
                    st.info(f"Put/Call OI ratio of **{pcr:.2f}** — relatively balanced positioning.")

            st.divider()

            # Options chain table (near-ATM strikes)
            if opt.chain_summary:
                st.markdown(f"##### Near-ATM Options Chain (Expiry: {opt.nearest_expiry})")
                current_price = company_info.get("price", 0)
                chain_rows = []
                for row in opt.chain_summary:
                    strike = row["strike"]
                    moneyness = "ATM" if abs(strike - current_price) / current_price < 0.01 else \
                                "ITM" if row["itm"] else "OTM"
                    chain_rows.append({
                        "Strike":    f"${strike:.2f}",
                        "Moneyness": moneyness,
                        "Call IV":   f"{row['call_iv']:.1%}" if row.get("call_iv") else "—",
                        "Put IV":    f"{row['put_iv']:.1%}" if row.get("put_iv") else "—",
                        "Call OI":   f"{row.get('call_oi', 0):,}",
                        "Put OI":    f"{row.get('put_oi', 0):,}",
                        "Call Vol":  f"{row.get('call_vol', 0):,}",
                        "Put Vol":   f"{row.get('put_vol', 0):,}",
                    })
                chain_df = pd.DataFrame(chain_rows)

                def style_moneyness(row):
                    if row["Moneyness"] == "ATM":
                        return ["background-color: rgba(255,213,79,0.15)"] * len(row)
                    return [""] * len(row)

                st.dataframe(
                    chain_df.style.apply(style_moneyness, axis=1),
                    use_container_width=True,
                    hide_index=True,
                )

                # IV skew chart
                st.markdown("##### IV Smile / Skew")
                strikes_chart = [r["strike"] for r in opt.chain_summary if r.get("call_iv") or r.get("put_iv")]
                call_ivs = [r.get("call_iv") for r in opt.chain_summary if r.get("call_iv") or r.get("put_iv")]
                put_ivs  = [r.get("put_iv")  for r in opt.chain_summary if r.get("call_iv") or r.get("put_iv")]

                skew_fig = go.Figure()
                skew_fig.add_trace(go.Scatter(
                    x=strikes_chart, y=[v * 100 if v else None for v in call_ivs],
                    mode="lines+markers", name="Call IV", line=dict(color="#00C853", width=2),
                ))
                skew_fig.add_trace(go.Scatter(
                    x=strikes_chart, y=[v * 100 if v else None for v in put_ivs],
                    mode="lines+markers", name="Put IV", line=dict(color="#FF5722", width=2),
                ))
                if current_price:
                    skew_fig.add_vline(x=current_price, line_dash="dash", line_color="white",
                                       annotation_text=f"${current_price:.2f}")
                skew_fig.update_layout(
                    template="plotly_dark", height=300,
                    margin=dict(l=10, r=10, t=30, b=10),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    xaxis_title="Strike", yaxis_title="Implied Volatility (%)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                st.plotly_chart(skew_fig, use_container_width=True)
        elif opt and opt.error:
            st.info(f"Options data unavailable: {opt.error}")
        else:
            st.info("No options data available for this ticker.")

    # =========================================================================
    with tab_inst:
    # =========================================================================
        ir = data.get("inst_risk")
        price_now = company_info.get("price", 0)

        st.markdown(
            "<div style='font-size:0.72rem;color:#555;text-transform:uppercase;letter-spacing:0.1em;"
            "border-bottom:1px solid rgba(255,255,255,0.06);padding-bottom:0.3rem;margin-bottom:1rem'>"
            "INSTITUTIONAL RISK  ·  Goldman Sachs CRB Methodology</div>",
            unsafe_allow_html=True,
        )

        if ir and ir.position_summary:
            st.markdown(
                f"<div style='background:rgba(102,126,234,0.07);border:1px solid rgba(102,126,234,0.18);"
                f"border-radius:8px;padding:0.9rem 1.1rem;margin-bottom:1.2rem;"
                f"color:#c5cae9;font-size:0.88rem;line-height:1.7'>"
                f"🧠 <b>Position Summary</b><br>{ir.position_summary}</div>",
                unsafe_allow_html=True,
            )

        gi_col, sm_col = st.columns(2)

        # ── GEX Panel ──
        with gi_col:
            st.markdown(
                "<div style='font-size:0.7rem;font-weight:700;text-transform:uppercase;"
                "letter-spacing:0.1em;color:#555;margin-bottom:0.6rem'>GEX — DEALER GAMMA EXPOSURE</div>",
                unsafe_allow_html=True,
            )
            gex = ir.gex if ir else None
            if gex and not gex.error:
                # Regime badge
                if gex.above_flip is not None:
                    regime_color = "#00C853" if gex.above_flip else "#FF1744"
                    regime_label = "STABLE REGIME — Positive GEX" if gex.above_flip else "VOLATILE REGIME — Negative GEX"
                    regime_detail = ("Dealers are net long gamma. They buy dips and sell rips — dampening volatility."
                                     if gex.above_flip else
                                     "Dealers are net short gamma. They must sell on drops and buy on rallies — amplifying moves.")
                    st.markdown(
                        f"<div style='background:{regime_color}15;border:1px solid {regime_color}44;"
                        f"border-radius:8px;padding:0.7rem 1rem;margin-bottom:0.8rem'>"
                        f"<span style='color:{regime_color};font-weight:700;font-size:0.82rem'>{regime_label}</span><br>"
                        f"<span style='color:#aaa;font-size:0.78rem'>{regime_detail}</span></div>",
                        unsafe_allow_html=True,
                    )

                # Key metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("Total GEX", f"${gex.total_gex/1e6:.1f}M")
                m2.metric("Gamma Flip", f"${gex.gamma_flip:.0f}" if gex.gamma_flip else "N/A")
                flip_dist = ((price_now - gex.gamma_flip) / price_now * 100) if gex.gamma_flip and price_now else None
                m3.metric("Dist. to Flip", f"{flip_dist:+.1f}%" if flip_dist is not None else "N/A",
                          help="How far price is from the gamma flip level")

                # GEX walls
                if gex.top_walls:
                    st.markdown("**GEX Support Walls** — dealers buy here (stabilising)")
                    for w in gex.top_walls[:4]:
                        dist = (price_now - w.strike) / price_now * 100 if price_now else 0
                        st.markdown(
                            f"<div style='display:flex;justify-content:space-between;align-items:center;"
                            f"padding:0.3rem 0.6rem;background:rgba(0,200,83,0.06);"
                            f"border-left:3px solid #00C853;border-radius:0 6px 6px 0;margin-bottom:0.25rem'>"
                            f"<span style='color:#fff;font-weight:600'>${w.strike:.0f}</span>"
                            f"<span style='color:#69F0AE;font-size:0.78rem'>${w.net_gex/1e6:.1f}M GEX</span>"
                            f"<span style='color:#888;font-size:0.75rem'>{dist:+.1f}% from price</span></div>",
                            unsafe_allow_html=True,
                        )

                # GEX trapdoors
                if gex.top_trapdoors:
                    st.markdown("**GEX Trapdoors** — dealers forced to sell below these (destabilising)")
                    for t in gex.top_trapdoors[:4]:
                        dist = (price_now - t.strike) / price_now * 100 if price_now else 0
                        st.markdown(
                            f"<div style='display:flex;justify-content:space-between;align-items:center;"
                            f"padding:0.3rem 0.6rem;background:rgba(255,23,68,0.06);"
                            f"border-left:3px solid #FF1744;border-radius:0 6px 6px 0;margin-bottom:0.25rem'>"
                            f"<span style='color:#fff;font-weight:600'>${t.strike:.0f}</span>"
                            f"<span style='color:#FF8A65;font-size:0.78rem'>${abs(t.net_gex)/1e6:.1f}M GEX</span>"
                            f"<span style='color:#888;font-size:0.75rem'>{dist:+.1f}% from price</span></div>",
                            unsafe_allow_html=True,
                        )

                # GEX bar chart
                gex_strikes = [w.strike for w in gex.top_walls[:5]] + [t.strike for t in gex.top_trapdoors[:5]]
                gex_values  = [w.net_gex/1e6 for w in gex.top_walls[:5]] + [t.net_gex/1e6 for t in gex.top_trapdoors[:5]]
                if gex_strikes:
                    gex_fig = go.Figure(go.Bar(
                        x=gex_strikes, y=gex_values,
                        marker_color=["#00C853" if v > 0 else "#FF1744" for v in gex_values],
                        text=[f"${v:.1f}M" for v in gex_values], textposition="auto",
                    ))
                    if gex.gamma_flip:
                        gex_fig.add_vline(x=gex.gamma_flip, line_dash="dash", line_color="#FFCA28",
                                          annotation_text=f"Gamma Flip ${gex.gamma_flip:.0f}",
                                          annotation_font_color="#FFCA28")
                    gex_fig.update_layout(
                        template="plotly_dark", height=260,
                        margin=dict(l=10,r=10,t=30,b=10),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        xaxis_title="Strike", yaxis_title="GEX ($M)",
                        title="Net Dealer GEX by Strike",
                    )
                    st.plotly_chart(gex_fig, use_container_width=True)
            elif gex and gex.error:
                st.info(f"GEX unavailable: {gex.error}")
            else:
                st.info("GEX data not available.")

        # ── Smart Money Panel ──
        with sm_col:
            st.markdown(
                "<div style='font-size:0.7rem;font-weight:700;text-transform:uppercase;"
                "letter-spacing:0.1em;color:#555;margin-bottom:0.6rem'>SMART MONEY — 13F INSTITUTIONAL FLOWS</div>",
                unsafe_allow_html=True,
            )
            sm = ir.smart_money if ir else None
            if sm and not sm.error:
                sm1, sm2, sm3 = st.columns(3)
                sm1.metric("Inst. Ownership", f"{sm.total_institutional_pct:.1%}")
                flow_color = "#00C853" if sm.net_buying_pressure > 0 else "#FF1744"
                sm2.metric("Net Flow (QoQ)", f"{sm.net_buying_pressure:+.2%}",
                           help="Average QoQ position change across top-10 holders")
                sm3.metric("High-Conviction", f"{sm.high_conviction_count}",
                           help="Holders with >2% of float — considered 'high conviction' positions")

                trend_color = {"Improving": "#00C853", "Stable": "#FFD54F", "Deteriorating": "#FF1744"}.get(sm.rec_trend, "#888")
                st.markdown(
                    f"<div style='padding:0.4rem 0.8rem;background:{trend_color}15;"
                    f"border:1px solid {trend_color}44;border-radius:6px;margin-bottom:0.8rem;"
                    f"font-size:0.82rem;color:{trend_color};font-weight:600'>"
                    f"Analyst Consensus Trend: {sm.rec_trend}</div>",
                    unsafe_allow_html=True,
                )

                st.markdown("**Top Institutional Holders**")
                for h in sm.holders[:8]:
                    chg_color = "#00C853" if h.pct_change > 0 else "#FF1744" if h.pct_change < 0 else "#888"
                    hc_badge = " <span style='color:#FFCA28;font-size:0.72rem'>★ HIGH CONVICTION</span>" if h.is_high_conviction else ""
                    chg_str = f"{h.pct_change:+.1%}" if h.pct_change else "—"
                    st.markdown(
                        f"<div style='padding:0.35rem 0.7rem;background:rgba(28,31,48,0.6);"
                        f"border:1px solid rgba(255,255,255,0.06);border-radius:7px;margin-bottom:0.25rem'>"
                        f"<div style='display:flex;justify-content:space-between;align-items:center'>"
                        f"<span style='color:#e0e0e0;font-size:0.82rem'>{h.name[:32]}{hc_badge}</span>"
                        f"<span style='color:#888;font-size:0.78rem'>{h.pct_held:.2%}</span>"
                        f"<span style='color:{chg_color};font-size:0.78rem;font-weight:600'>{chg_str} QoQ</span>"
                        f"</div></div>",
                        unsafe_allow_html=True,
                    )
            elif sm and sm.error:
                st.info(f"Institutional data unavailable: {sm.error}")

        st.divider()

        # ── Credit Proxy Panel (full width) ──
        st.markdown(
            "<div style='font-size:0.7rem;font-weight:700;text-transform:uppercase;"
            "letter-spacing:0.1em;color:#555;margin-bottom:0.6rem'>CREDIT RISK  ·  CDS PROXY / CS01 / DV01</div>",
            unsafe_allow_html=True,
        )
        cr = ir.credit if ir else None
        if cr and not cr.error:
            cc1, cc2, cc3, cc4, cc5 = st.columns(5)
            cc1.metric("CDS Proxy", f"{cr.cds_proxy_bps:.0f} bps" if cr.cds_proxy_bps else "N/A",
                       help="Synthetic credit default swap spread — higher = riskier debt")
            cc2.metric("Credit Profile", cr.credit_rating_proxy,
                       help="Inferred credit quality based on balance sheet ratios")
            cc3.metric("Interest Coverage", f"{cr.interest_coverage:.1f}x" if cr.interest_coverage else "N/A",
                       help="EBIT / Interest Expense — higher is safer. <1.5x is danger zone")
            cc4.metric("Debt / EBITDA", f"{cr.debt_to_ebitda:.1f}x" if cr.debt_to_ebitda else "N/A",
                       help="Turns of leverage. <3x = investment grade territory")
            cc5.metric("CS01 / DV01", f"${abs(cr.cs01_proxy or 0):,.0f}" if cr.cs01_proxy else "N/A",
                       help="Approx $ loss per 1 basis point move in credit spread or interest rate")

            st.markdown(
                "<div style='background:rgba(28,31,48,0.5);border:1px solid rgba(255,255,255,0.06);"
                "border-radius:8px;padding:0.8rem 1rem;margin-top:0.6rem;"
                "color:#888;font-size:0.82rem;line-height:1.7'>"
                "<b style='color:#aaa'>How to read these:</b><br>"
                "• <b>CDS Proxy</b>: The market's implied cost to insure against the company defaulting. "
                "Investment grade typically &lt;100bps, high yield 300–700bps, distress &gt;1000bps.<br>"
                "• <b>CS01</b>: If the CDS spread widens by 1 basis point (0.01%), the firm's debt loses this much in mark-to-market value.<br>"
                "• <b>DV01</b>: Same concept but driven by a 1bp move in the risk-free rate (Treasury yield).<br>"
                "• <b>Interest Coverage &lt;1.5x</b>: Company may struggle to service debt from operating income alone."
                "</div>",
                unsafe_allow_html=True,
            )
        elif cr and cr.error:
            st.info(f"Credit proxy unavailable: {cr.error}")

    with tab_altdata:
        alt_bundle = data.get("alt_data")
        if alt_bundle is None:
            st.info("Alternative data not available.")
            alt = None
            staleness = {}
        else:
            alt = alt_bundle.get("result")
            staleness = alt_bundle.get("staleness", {})
        if alt is None:
            st.info("Alternative data not available.")
        else:
            # ── Staleness header bar ───────────────────────────────────────
            st.markdown(
                "<div style='font-size:0.72rem;color:#888;text-transform:uppercase;"
                "letter-spacing:0.08em;margin-bottom:0.5rem'>Cache Status — Each Layer Refreshes Independently</div>",
                unsafe_allow_html=True,
            )
            badge_cols = st.columns(5)
            layer_order = ["sentiment", "insider", "cluster", "vanna_charm", "congressional"]
            for i, lk in enumerate(layer_order):
                meta = staleness.get(lk, {})
                badge = meta.get("badge", "⚪ No Data")
                label = meta.get("label", lk)
                ttl_h = meta.get("ttl_seconds", 0) // 3600
                ttl_label = f"{ttl_h}h TTL" if ttl_h >= 1 else f"{meta.get('ttl_seconds', 0)//60}m TTL"
                with badge_cols[i]:
                    st.markdown(
                        f"<div style='background:rgba(255,255,255,0.04);border-radius:6px;padding:6px 8px;"
                        f"text-align:center;font-size:.75rem'>"
                        f"<div style='font-weight:600;margin-bottom:2px'>{label}</div>"
                        f"<div>{badge}</div>"
                        f"<div style='color:#666;font-size:.68rem'>{ttl_label}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

            # ── Per-layer refresh buttons ──────────────────────────────────
            with st.expander("🔄 Force Refresh Individual Layers", expanded=False):
                import cache as _cache_mod
                r_cols = st.columns(5)
                force_rerun = False
                for i, lk in enumerate(layer_order):
                    meta = staleness.get(lk, {})
                    label = meta.get("label", lk)
                    with r_cols[i]:
                        if st.button(f"Refresh\n{label}", key=f"refresh_{ticker}_{lk}"):
                            _cache_mod.invalidate_ttl(ticker, f"altdata_{lk}")
                            force_rerun = True
                if st.button("🔄 Refresh ALL Layers", key=f"refresh_all_{ticker}"):
                    for lk in layer_order:
                        _cache_mod.invalidate_ttl(ticker, f"altdata_{lk}")
                    force_rerun = True
                if force_rerun:
                    st.rerun()

            st.divider()

            # ── Overall signal banner ──────────────────────────────────────
            sig_color = {
                "Strong Alternative Buy Signal": "#1a7a4a",
                "Moderate Buy Signal": "#2d9e6b",
                "Mild Bullish Lean": "#4ab87e",
                "Mixed / Conflicting": "#b8860b",
                "Mild Bearish Lean": "#c0704a",
                "Strong Alternative Sell Signal": "#c0392b",
            }.get(alt.overall_signal, "#555")
            st.markdown(
                f"""<div style='background:{sig_color};color:#fff;padding:14px 20px;border-radius:8px;margin-bottom:18px'>
                <span style='font-size:1.15rem;font-weight:700'>🔭 Alternative Data Composite: {alt.overall_signal}</span>
                <span style='float:right;font-size:.9rem'>{alt.signal_count}/5 layers bullish</span>
                </div>""",
                unsafe_allow_html=True,
            )

            col_left, col_right = st.columns(2)

            # ── Layer 1: News Sentiment ────────────────────────────────────
            with col_left:
                sm_meta = staleness.get("sentiment", {})
                st.markdown(f"#### 📰 Layer 1 — News Sentiment <span style='font-size:.7rem;color:#888'>{sm_meta.get('badge','')}</span>", unsafe_allow_html=True)
                sm = alt.sentiment
                if sm and not sm.error:
                    sdir_color = "#1a7a4a" if sm.direction == "Bullish" else "#c0392b" if sm.direction == "Bearish" else "#555"
                    trigger_badge = (
                        "<span style='background:#e74c3c;color:#fff;padding:2px 8px;border-radius:4px;font-size:.8rem'>⚡ TRIGGERED</span>"
                        if sm.triggered else ""
                    )
                    st.markdown(
                        f"<span style='color:{sdir_color};font-weight:700;font-size:1.05rem'>{sm.direction}</span> "
                        f"score {sm.composite_score:+.3f} {trigger_badge}",
                        unsafe_allow_html=True,
                    )
                    st.caption(f"Threshold ±0.225 · {sm.article_count} articles · {sm.bullish_count} bullish · {sm.bearish_count} bearish")
                    if sm.top_headlines:
                        with st.expander("Top Headlines by Sentiment Score", expanded=False):
                            for h in sm.top_headlines:
                                bar_color = "#1a7a4a" if h["score"] > 0 else "#c0392b"
                                st.markdown(
                                    f"<div style='border-left:3px solid {bar_color};padding:4px 10px;margin:4px 0'>"
                                    f"<b>{h['score']:+.3f}</b> · {h['title'][:100]}"
                                    f"<span style='color:#888;font-size:.75rem'> ({h['published']})</span></div>",
                                    unsafe_allow_html=True,
                                )
                elif sm and sm.error:
                    st.caption(f"Sentiment unavailable: {sm.error}")

                st.markdown("---")

                # ── Layer 3: 13F Cluster Buying ────────────────────────────
                cl_meta = staleness.get("cluster", {})
                st.markdown(f"#### 🏛️ Layer 3 — 13F Cluster Buying <span style='font-size:.7rem;color:#888'>{cl_meta.get('badge','')}</span>", unsafe_allow_html=True)
                cb = alt.cluster_buying
                if cb and not cb.error:
                    sig_c = "#1a7a4a" if "Buy" in cb.consensus_signal or "Accum" in cb.consensus_signal else "#c0392b" if "Distrib" in cb.consensus_signal else "#555"
                    st.markdown(f"<span style='color:{sig_c};font-weight:700'>{cb.consensus_signal}</span>", unsafe_allow_html=True)
                    st.caption(f"New entrants: {len(cb.new_entrants)} funds · Exits: {len(cb.exits)} funds · Avg position change: {cb.avg_position_change:+.1%}")
                    if cb.new_entrants:
                        with st.expander(f"New/Growing Positions ({len(cb.new_entrants)} funds)", expanded=False):
                            for h in cb.new_entrants:
                                badge = "🆕 NEW" if h.is_new_position else "📈 ADDING"
                                st.markdown(f"**{badge}** {h.name} — holds {h.pct_held:.1%} of float · QoQ change: **{h.pct_change:+.1%}**")
                    if cb.exits:
                        with st.expander(f"Reducing/Exiting Positions ({len(cb.exits)} funds)", expanded=False):
                            for h in cb.exits:
                                st.markdown(f"🔻 {h.name} — holds {h.pct_held:.1%} of float · QoQ change: **{h.pct_change:+.1%}**")
                elif cb and cb.error:
                    st.caption(f"Cluster data unavailable: {cb.error}")

            with col_right:
                # ── Layer 2: Insider Flow ──────────────────────────────────
                in_meta = staleness.get("insider", {})
                st.markdown(f"#### 👤 Layer 2 — Insider & Regulatory Forensics <span style='font-size:.7rem;color:#888'>{in_meta.get('badge','')}</span>", unsafe_allow_html=True)
                ins = alt.insider_flow
                if ins and not ins.error:
                    sig_c = "#1a7a4a" if ins.signal == "Accumulating" else "#c0392b" if "Distribut" in ins.signal or "Exit" in ins.signal else "#b8860b" if ins.signal == "Mixed" else "#555"
                    cluster_badge = (
                        " <span style='background:#e74c3c;color:#fff;padding:2px 7px;border-radius:4px;font-size:.75rem'>🚨 CLUSTER EXIT</span>"
                        if ins.cluster_exit_flag else ""
                    )
                    st.markdown(
                        f"<span style='color:{sig_c};font-weight:700;font-size:1.05rem'>{ins.signal}</span>{cluster_badge}",
                        unsafe_allow_html=True,
                    )
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Buys (90d)", ins.buy_count)
                    c2.metric("Sales (90d)", ins.sell_count)
                    net_fmt = f"${abs(ins.net_value_bought):,.0f}" if ins.net_value_bought != 0 else "$0"
                    net_dir = "Net Bought" if ins.net_value_bought >= 0 else "Net Sold"
                    c3.metric(net_dir, net_fmt)
                    if ins.recent_trades:
                        with st.expander("Recent Insider Transactions (90 days)", expanded=False):
                            rows = []
                            for tr in ins.recent_trades[:12]:
                                rows.append({
                                    "Date": tr.date,
                                    "Insider": tr.name[:25],
                                    "Title": tr.title[:20],
                                    "Type": tr.transaction,
                                    "Shares": f"{tr.shares:,}",
                                    "Value": f"${tr.value:,.0f}" if tr.value else "—",
                                })
                            if rows:
                                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                    st.caption("⚠️ Cluster Exit = ≥2 insiders selling within 30 days — historical precursor to negative news.")
                elif ins and ins.error:
                    st.caption(f"Insider data unavailable: {ins.error}")

                st.markdown("---")

                # ── Layer 5: Political Alpha ───────────────────────────────
                cg_meta = staleness.get("congressional", {})
                st.markdown(f"#### 🏛️ Layer 5 — Political Alpha (Congressional Trades) <span style='font-size:.7rem;color:#888'>{cg_meta.get('badge','')}</span>", unsafe_allow_html=True)
                cong = alt.congressional
                if cong and not cong.error:
                    sig_c = "#1a7a4a" if "Strong" in cong.alpha_signal and cong.net_congressional_bias == "Buying" else \
                            "#c0392b" if cong.net_congressional_bias == "Selling" else "#555"
                    st.markdown(f"<span style='color:{sig_c};font-weight:700'>{cong.alpha_signal}</span>", unsafe_allow_html=True)
                    if cong.member_count > 0:
                        st.caption(f"{cong.member_count} member(s) traded this ticker · Bias: {cong.net_congressional_bias}")
                        with st.expander(f"Congressional Trades ({len(cong.trades)} filings)", expanded=cong.member_count >= 2):
                            rows = []
                            for tr in cong.trades:
                                delay_flag = "⚠️" if tr.days_to_disclose > 30 else ""
                                rows.append({
                                    "Trade Date": tr.trade_date,
                                    "Disclosed": tr.disclosure_date,
                                    "Member": tr.member[:25],
                                    "Party": tr.party,
                                    "Type": tr.transaction,
                                    "Amount": tr.amount_range,
                                    "Delay": f"{tr.days_to_disclose}d {delay_flag}",
                                })
                            if rows:
                                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                        st.caption("Source: House Stock Watcher (STOCK Act disclosures) · Delay >30 days flagged ⚠️")
                    else:
                        st.caption("No congressional trades found for this ticker in the last 6 months.")
                elif cong and cong.error:
                    st.caption(f"Congressional data unavailable: {cong.error}")

            # ── Layer 4: Vanna & Charm (full width) ───────────────────────
            vc_meta = staleness.get("vanna_charm", {})
            st.markdown("---")
            st.markdown(f"#### ⚙️ Layer 4 — Vanna & Charm (Mechanical Dealer Flow) <span style='font-size:.7rem;color:#888'>{vc_meta.get('badge','')}</span>", unsafe_allow_html=True)
            vc = alt.vanna_charm
            if vc and not vc.error:
                vc_col1, vc_col2, vc_col3, vc_col4 = st.columns(4)
                vanna_color = "#1a7a4a" if "Rally" in vc.vanna_signal else "#c0392b" if "Selloff" in vc.vanna_signal else "#555"
                charm_color = "#1a7a4a" if "Supports" in vc.charm_signal else "#c0392b" if "Pressures" in vc.charm_signal else "#555"
                vc_col1.metric("Net Vanna ($)", f"${vc.net_vanna:,.0f}")
                vc_col2.metric("Net Charm ($)", f"${vc.net_charm:,.0f}")
                vc_col3.markdown(
                    f"<div style='padding:8px'><div style='font-size:.75rem;color:#888'>Vanna Signal</div>"
                    f"<div style='color:{vanna_color};font-weight:700;font-size:.85rem'>{vc.vanna_signal}</div></div>",
                    unsafe_allow_html=True,
                )
                vc_col4.markdown(
                    f"<div style='padding:8px'><div style='font-size:.75rem;color:#888'>Charm Signal</div>"
                    f"<div style='color:{charm_color};font-weight:700;font-size:.85rem'>{vc.charm_signal}</div></div>",
                    unsafe_allow_html=True,
                )
                st.caption(f"Options expiry used: {vc.expiry_used} · Positive Vanna = vol-crush (post-FOMC/CPI) forces dealers to BUY mechanically · Positive Charm = daily theta decay supports price")
                with st.expander("How to read Vanna & Charm", expanded=False):
                    st.markdown("""
**Vanna** measures how dealer delta hedges change when *implied volatility* changes.
- **Positive Net Vanna** = if VIX drops (after FOMC, CPI, earnings), dealers must *buy* equity futures to stay delta-neutral → mechanical rally regardless of the news outcome
- **Negative Net Vanna** = vol crush forces dealers to *sell* → downward pressure after events

**Charm** measures how dealer delta hedges change as *time passes* (theta decay).
- **Positive Net Charm** = each passing day, dealers must buy equity to stay hedged → passive daily support
- **Negative Net Charm** = time decay forces dealers to sell → price drifts lower into expiry

These flows are **price-agnostic** — they move stocks based purely on the math of options hedging, not fundamentals or news.
                    """)
            elif vc and vc.error:
                st.caption(f"Vanna/Charm unavailable: {vc.error}")

            # ── Educational panel ──────────────────────────────────────────
            with st.expander("📚 Why Alternative Data Matters", expanded=False):
                st.markdown("""
**Institutions trade 1–3 layers deeper than the current price reflects.** By the time CNBC reports a story, the edge is gone.

| Layer | Data Source | TTL | Lead Time Before Price Move |
|-------|------------|-----|----------------------------|
| News Sentiment | Public headlines (VADER scored) | 6h | 0–3 days — reactive |
| Insider Flow | Form 4 SEC filings via yfinance | 24h | 2–8 weeks — directional |
| 13F Cluster Buy | 13F institutional filings | 90d | 1–3 quarters — structural |
| Vanna/Charm | Live options chain | 24h | Hours–days — mechanical |
| Political Alpha | STOCK Act disclosures | 7d | 2–6 weeks — event-driven |

**Cluster Exit Rule:** When ≥2 insiders sell within 30 days of each other, the probability of a negative news event in the next 60 days rises significantly based on academic research (Jagolinzer et al., 2011).

**Institutional trigger:** Composite sentiment score crossing ±0.225 is a threshold used in systematic trading models — below/above this level, algo-driven buying/selling is triggered.
                """)

    with tab_forensic:
        fr = data.get("forensic")
        st.markdown(
            "<div style='font-size:0.72rem;color:#888;text-transform:uppercase;"
            "letter-spacing:0.08em;margin-bottom:1rem'>"
            "FORENSIC NLP  ·  SEC EDGAR MD&A Analysis  ·  Loughran-McDonald Word Lists</div>",
            unsafe_allow_html=True,
        )

        if fr is None:
            st.info("Forensic analysis not available.")
        else:
            # ── Verdict banner ─────────────────────────────────────────────
            verdict_color = {
                "Clean":    "#1a7a4a",
                "Watch":    "#b8860b",
                "Red Flag": "#c0704a",
                "Critical": "#c0392b",
                "Unknown":  "#555",
            }.get(getattr(getattr(fr, "verdict", None), "verdict", "Unknown"), "#555")

            v = fr.verdict
            if v:
                cache_note = ""
                import cache as _cache_mod2
                age = _cache_mod2.get_age_seconds(ticker, "forensic_nlp")
                if age is not None:
                    import alternative_data as _alt_data2
                    cache_note = f" · cached {_alt_data2._fmt_age(age)}"

                st.markdown(
                    f"<div style='background:{verdict_color};color:#fff;padding:14px 20px;"
                    f"border-radius:8px;margin-bottom:18px'>"
                    f"<span style='font-size:1.15rem;font-weight:700'>🧬 Forensic Verdict: {v.verdict}</span>"
                    f"<span style='float:right;font-size:.85rem'>Risk Score: {v.score:.0f}/100 · "
                    f"Confidence: {v.confidence}{cache_note}</span></div>",
                    unsafe_allow_html=True,
                )

                # Refresh button
                if st.button("🔄 Refresh Forensic Analysis", key=f"refresh_forensic_{ticker}"):
                    _cache_mod2.invalidate_ttl(ticker, "forensic_nlp")
                    st.rerun()

            # ── Filing metadata ────────────────────────────────────────────
            if fr.filing:
                fi = fr.filing
                st.caption(
                    f"Source: {fi.form_type} · Period: {fi.period} · Filed: {fi.filed_date} · "
                    f"Accession: {fi.accession} · Words analyzed: {fi.word_count:,}"
                )
            elif fr.error:
                st.warning(f"⚠️ {fr.error} — showing scores as unavailable.")

            st.divider()

            left_col, right_col = st.columns([1, 1])

            # ── LM Linguistic Scores ───────────────────────────────────────
            with left_col:
                st.markdown("#### 📊 Loughran-McDonald Linguistic Scores")
                lm = fr.lm
                if lm and lm.total_words > 0:
                    import plotly.graph_objects as go

                    categories = ["Uncertainty", "Litigious", "Negative", "Positive", "Hedging"]
                    values = [
                        lm.uncertainty_ratio * 100,
                        lm.litigious_ratio * 100,
                        lm.negative_ratio * 100,
                        lm.positive_ratio * 100,
                        lm.hedging_ratio * 100,
                    ]
                    # Benchmarks (typical ranges from Loughran-McDonald research)
                    benchmarks = [2.8, 1.2, 1.8, 1.5, 2.5]
                    bar_colors = []
                    for val, bench in zip(values, benchmarks):
                        if val > bench * 1.5:
                            bar_colors.append("#c0392b")
                        elif val > bench * 1.1:
                            bar_colors.append("#e67e22")
                        else:
                            bar_colors.append("#2ecc71")

                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=categories, y=values,
                        marker_color=bar_colors,
                        name="Filing",
                        text=[f"{v:.2f}%" for v in values],
                        textposition="outside",
                    ))
                    fig.add_trace(go.Scatter(
                        x=categories, y=benchmarks,
                        mode="markers+lines",
                        name="Typical Range",
                        line=dict(color="#888", dash="dash"),
                        marker=dict(size=8),
                    ))
                    fig.update_layout(
                        height=280, margin=dict(t=20, b=10, l=10, r=10),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02),
                        font=dict(color="#ccc"),
                        yaxis=dict(title="% of total words", gridcolor="rgba(255,255,255,0.08)"),
                        xaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Sentiment net
                    snet_color = "#1a7a4a" if lm.sentiment_net > 0 else "#c0392b"
                    st.markdown(
                        f"**Net Sentiment:** <span style='color:{snet_color};font-weight:700'>"
                        f"{lm.sentiment_net:+.4f}</span> &nbsp;·&nbsp; "
                        f"**Total words analyzed:** {lm.total_words:,}",
                        unsafe_allow_html=True,
                    )

                    # Deception phrases
                    if lm.deception_phrase_count > 0:
                        with st.expander(f"⚠️ {lm.deception_phrase_count} Evasive Phrases Detected", expanded=lm.deception_phrase_count >= 3):
                            for p in lm.deception_phrases_found:
                                st.markdown(f"- *\"{p}\"*")
                            st.caption("These phrases are associated with management avoiding direct answers in academic earnings call research.")
                    else:
                        st.caption("✅ No evasive/non-answer phrases detected")
                else:
                    st.caption("Linguistic scores unavailable — filing text not retrieved.")

            # ── Flags & Verdict detail ─────────────────────────────────────
            with right_col:
                st.markdown("#### 🚨 Forensic Flags")
                if v and v.flags:
                    for i, flag in enumerate(v.flags):
                        severity = "🔴" if "CRITICAL" in flag or "extreme" in flag.lower() else \
                                   "🟠" if "high" in flag.lower() or "cluster" in flag.lower() else "🟡"
                        st.markdown(
                            f"<div style='border-left:3px solid {'#c0392b' if severity == '🔴' else '#e67e22' if severity == '🟠' else '#f1c40f'};"
                            f"padding:6px 12px;margin:6px 0;font-size:.88rem'>"
                            f"{severity} {flag}</div>",
                            unsafe_allow_html=True,
                        )
                elif v:
                    st.markdown(
                        "<div style='border-left:3px solid #1a7a4a;padding:6px 12px;margin:6px 0;"
                        "font-size:.88rem'>✅ No significant forensic flags detected</div>",
                        unsafe_allow_html=True,
                    )

                st.markdown("---")
                st.markdown("#### 📋 Verdict Summary")
                if v:
                    st.markdown(
                        f"<div style='background:rgba(255,255,255,0.04);border-radius:8px;"
                        f"padding:12px 16px;font-size:.9rem;line-height:1.7'>{v.summary}</div>",
                        unsafe_allow_html=True,
                    )

            # ── Claude AI forensic analysis ────────────────────────────────
            if fr.ai_analysis:
                st.divider()
                st.markdown("#### 🤖 Claude Forensic Analysis")
                st.markdown(
                    f"<div style='background:rgba(102,126,234,0.07);border:1px solid rgba(102,126,234,0.18);"
                    f"border-radius:8px;padding:1rem 1.2rem;font-size:.9rem;line-height:1.8'>"
                    f"{fr.ai_analysis.replace(chr(10), '<br>')}</div>",
                    unsafe_allow_html=True,
                )

            # ── MD&A raw excerpt ───────────────────────────────────────────
            if fr.filing and fr.filing.mda_text:
                with st.expander("📄 MD&A Raw Excerpt (Item 7)", expanded=False):
                    st.text(fr.filing.mda_text[:4000] + ("..." if len(fr.filing.mda_text) > 4000 else ""))
            if fr.filing and fr.filing.risk_text:
                with st.expander("⚠️ Risk Factors Excerpt (Item 1A)", expanded=False):
                    st.text(fr.filing.risk_text[:4000] + ("..." if len(fr.filing.risk_text) > 4000 else ""))

            # ── Educational panel ──────────────────────────────────────────
            with st.expander("📚 How to Read Forensic NLP", expanded=False):
                st.markdown("""
**Loughran-McDonald (LM) Word Lists** are the academic standard for financial text analysis.
Unlike general sentiment tools, LM was built specifically on 10-K filings — words like "risk" or "liability"
are *neutral* in general English but *negative* in financial context.

| Score | Typical Range | Red Flag Level | What It Means |
|-------|--------------|---------------|---------------|
| Uncertainty | ~2.8% | >4.5% | Management avoiding commitments |
| Litigious | ~1.2% | >2.5% | Significant legal exposure |
| Negative | ~1.8% | >3.0% | Pessimistic forward view |
| Hedging | ~2.5% | >4.0% | No firm guidance — evasion |

**TATA (Total Accruals to Total Assets)** = (Net Income − Operating Cash Flow) / Total Assets
- Normal range: -0.05 to +0.05
- Above +0.05: earnings are being manufactured via accruals, not backed by cash
- The Beneish M-Score uses this as one of its 8 inputs

**Deception phrase patterns** come from academic earnings call research (Larcker & Zakolyukina, 2012)
showing that executives who later restated earnings used significantly more hedging and non-answer
phrases during their calls than executives who did not.

**Verdict thresholds:**
- Clean: Score < 20 — language consistent with transparent disclosure
- Watch: Score 20–39 — mild anomalies, monitor next filing
- Red Flag: Score 40–64 — multiple signals, reduce position or add hedge
- Critical: Score ≥ 65 — strong convergence of manipulation signals
                """)

    with tab_macro:
        st.markdown("##### Macroeconomic Environment")

        with st.spinner("Loading macro data..."):
            macro_result = run_macro_analysis()

        if macro_result and macro_result.cycle:
            cycle = macro_result.cycle

            # Business cycle phase diagram
            st.markdown(create_cycle_diagram(cycle.phase), unsafe_allow_html=True)

            # Phase details
            phase_colors = {"Expansion": "#00C853", "Peak": "#FFA726", "Contraction": "#FF1744", "Recovery": "#42A5F5"}
            pc = phase_colors.get(cycle.phase, "#888")
            st.markdown(f"""
            <div style="background:linear-gradient(135deg, {pc}22, {pc}11);
                        border:1px solid {pc}66;border-radius:12px;padding:1.2rem;margin:0.5rem 0 1rem 0">
                <h3 style="margin:0;color:{pc}">{cycle.phase} Phase
                    <span style="font-size:0.85rem;color:#aaa;margin-left:0.5rem">
                    (Confidence: {cycle.confidence}) — {cycle.risk_posture}</span></h3>
                <p style="color:#ccc;margin:0.5rem 0 0 0">{cycle.description}</p>
            </div>
            """, unsafe_allow_html=True)

            fc1, fc2 = st.columns(2)
            with fc1:
                st.markdown("**Favored Sectors in This Phase**")
                for s in cycle.favored_sectors:
                    is_match = s == company_info.get("sector")
                    marker = " ← *your stock*" if is_match else ""
                    color = "#00C853" if is_match else "#69F0AE"
                    st.markdown(f"<span style='color:{color}'>● {s}{marker}</span>", unsafe_allow_html=True)
            with fc2:
                st.markdown("**Sectors to Underweight**")
                for s in cycle.avoid_sectors:
                    is_match = s == company_info.get("sector")
                    marker = " ← *your stock*" if is_match else ""
                    color = "#FF1744" if is_match else "#FF8A65"
                    st.markdown(f"<span style='color:{color}'>● {s}{marker}</span>", unsafe_allow_html=True)

            # Check if stock sector is favored or avoided
            stock_sector = company_info.get("sector", "")
            if stock_sector in cycle.favored_sectors:
                st.success(f"**{ticker}** is in **{stock_sector}** — a favored sector during {cycle.phase}.")
            elif stock_sector in cycle.avoid_sectors:
                st.warning(f"**{ticker}** is in **{stock_sector}** — an underweight sector during {cycle.phase}.")
            else:
                st.info(f"**{ticker}** is in **{stock_sector}** — a neutral sector during {cycle.phase}.")

            st.divider()

            # Macro indicators
            st.markdown("##### Key Macro Indicators")
            macro_fig = create_macro_dashboard(macro_result.indicators)
            st.plotly_chart(macro_fig, use_container_width=True)

            # Yield curve
            yc_fig = create_yield_curve_chart(macro_result.yield_curve)
            if yc_fig:
                st.plotly_chart(yc_fig, use_container_width=True)

            # Indicator details
            st.markdown("##### Indicator Details")
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

            # Sector analysis
            st.markdown("##### Sector Rotation Analysis")

            rates_rising = False
            for ind in macro_result.indicators:
                if ind.name == "10-Year Treasury Yield" and ind.change_1m and ind.change_1m > 0.02:
                    rates_rising = True

            with st.spinner("Loading sector data..."):
                sector_result = run_sector_analysis(cycle.phase, rates_rising)

            if sector_result and sector_result.sectors:
                sector_result = SectorAnalyzer().score_stock_sector(sector_result, stock_sector)

                # Sector heatmap
                heatmap_fig = create_sector_heatmap(sector_result.sectors)
                if heatmap_fig:
                    st.plotly_chart(heatmap_fig, use_container_width=True)

                # Rotation scatter
                rotation_fig = create_sector_rotation_chart(sector_result.sectors, cycle.phase)
                if rotation_fig:
                    st.plotly_chart(rotation_fig, use_container_width=True)

                # Rotation recommendation
                st.markdown("##### Rotation Recommendation")
                st.markdown(sector_result.rotation_recommendation)

                if sector_result.stock_sector_rank:
                    rank = sector_result.stock_sector_rank
                    total = len(sector_result.sectors)
                    rank_color = "#00C853" if rank <= 3 else "#FFD54F" if rank <= 7 else "#FF8A65"
                    st.markdown(
                        f"**{stock_sector}** ranks **#{rank}** out of {total} sectors "
                        f"<span style='color:{rank_color}'>(cycle alignment + momentum)</span>",
                        unsafe_allow_html=True,
                    )

                # Sector table
                st.markdown("##### All Sectors Ranked")
                sect_rows = []
                for i, s in enumerate(sector_result.sectors, 1):
                    sect_rows.append({
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
                    })
                st.dataframe(pd.DataFrame(sect_rows), use_container_width=True, hide_index=True)
            else:
                st.info("Sector data unavailable — may be rate limited. Try again shortly.")
        else:
            st.info("Macro data unavailable — may be rate limited. Try again shortly.")

    with tab_screener:
        stock_sector = company_info.get("sector", "Technology")
        st.markdown(f"##### Stock Screener — Find Titans in **{stock_sector}**")
        st.caption("Screen the top companies in this stock's sector by quality metrics to find the strongest names.")

        # Filters
        fc1, fc2, fc3, fc4, fc5 = st.columns(5)
        scr_max_stocks = fc1.number_input("Max Stocks", 10, 50, 20, 5, key=f"scr_n_{ticker}")
        scr_max_pe = fc2.number_input("Max P/E", 0.0, 200.0, 50.0, 5.0, key=f"scr_pe_{ticker}",
                                       help="0 = no filter")
        scr_max_de = fc3.number_input("Max Debt/Equity", 0.0, 500.0, 200.0, 25.0, key=f"scr_de_{ticker}",
                                       help="0 = no filter")
        scr_min_roe = fc4.number_input("Min ROE %", 0.0, 100.0, 0.0, 5.0, key=f"scr_roe_{ticker}",
                                        help="0 = no filter")
        scr_min_rg = fc5.number_input("Min Rev Growth %", -50.0, 200.0, 0.0, 5.0, key=f"scr_rg_{ticker}",
                                       help="0 = no filter")

        run_screen = st.button("🔎 Run Screener", key=f"scr_btn_{ticker}", type="primary")

        if run_screen:
            with st.spinner(f"Screening {stock_sector} stocks... (this may take a minute)"):
                scr_result = run_stock_screener(
                    stock_sector, scr_max_stocks,
                    0,
                    scr_max_pe if scr_max_pe > 0 else None,
                    scr_max_de if scr_max_de > 0 else None,
                    scr_min_roe if scr_min_roe > 0 else None,
                    scr_min_rg if scr_min_rg != 0 else None,
                )

            if scr_result and scr_result.stocks:
                st.markdown(f"**Found {len(scr_result.stocks)} stocks** (from {scr_result.total_found} in sector)")

                # Titans highlight
                if scr_result.titans:
                    st.markdown("##### Top 10 Titans")
                    for i, s in enumerate(scr_result.titans, 1):
                        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(i, f"**#{i}**")
                        sc_color = "#00C853" if s.composite_score >= 70 else "#69F0AE" if s.composite_score >= 60 else "#FFD54F" if s.composite_score >= 50 else "#FF8A65"
                        is_current = s.ticker == ticker
                        highlight = " ← **YOUR STOCK**" if is_current else ""
                        border = f"border:2px solid {sc_color}" if is_current else f"border:1px solid rgba(255,255,255,0.08)"

                        roe_str = f"{s.roe * 100:.1f}%" if s.roe else "N/A"
                        pe_str = f"{s.pe_trailing:.1f}" if s.pe_trailing else "N/A"
                        de_str = f"{s.debt_to_equity:.0f}" if s.debt_to_equity else "N/A"
                        roic_str = f"{s.roic * 100:.1f}%" if s.roic else "N/A"
                        rg_str = f"{s.revenue_growth * 100:.1f}%" if s.revenue_growth else "N/A"

                        st.markdown(
                            f"<div style='padding:0.6rem 0.8rem;background:rgba(28,31,48,0.6);{border};"
                            f"border-radius:10px;margin-bottom:0.4rem;display:flex;align-items:center;gap:0.8rem'>"
                            f"<span style='font-size:1.3rem'>{medal}</span>"
                            f"<div style='flex:1'>"
                            f"<span style='color:white;font-weight:600'>{s.ticker}</span> "
                            f"<span style='color:#aaa'>— {s.name}</span>{highlight}<br>"
                            f"<span style='color:#888;font-size:0.82rem'>"
                            f"P/E: {pe_str} · ROE: {roe_str} · ROIC: {roic_str} · "
                            f"D/E: {de_str} · Rev Growth: {rg_str} · "
                            f"Rating: {s.analyst_rating}</span>"
                            f"</div>"
                            f"<div style='text-align:center'>"
                            f"<div style='color:{sc_color};font-size:1.4rem;font-weight:700'>{s.composite_score:.0f}</div>"
                            f"<div style='color:#888;font-size:0.7rem'>SCORE</div></div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                st.divider()

                # Full data table
                st.markdown("##### Full Screener Results")
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
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True,
                             height=min(600, 40 + len(rows) * 38))

                # Score distribution chart
                st.markdown("##### Score Distribution")
                score_fig = go.Figure()
                sorted_stocks = sorted(scr_result.stocks, key=lambda x: x.composite_score, reverse=True)
                score_fig.add_trace(go.Bar(
                    x=[s.ticker for s in sorted_stocks],
                    y=[s.composite_score for s in sorted_stocks],
                    marker_color=[
                        "#00C853" if s.composite_score >= 70 else "#69F0AE" if s.composite_score >= 60
                        else "#FFD54F" if s.composite_score >= 50 else "#FF8A65"
                        for s in sorted_stocks
                    ],
                    text=[f"{s.composite_score:.0f}" for s in sorted_stocks],
                    textposition="auto",
                ))
                current_idx = next((i for i, s in enumerate(sorted_stocks) if s.ticker == ticker), None)
                if current_idx is not None:
                    score_fig.add_annotation(
                        x=sorted_stocks[current_idx].ticker,
                        y=sorted_stocks[current_idx].composite_score + 3,
                        text="▼ YOU", showarrow=False,
                        font=dict(color="#FFCA28", size=12, family="Arial Black"),
                    )
                score_fig.update_layout(
                    template="plotly_dark", height=350,
                    margin=dict(l=10, r=10, t=30, b=10),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    yaxis=dict(title="Composite Score", range=[0, 105]),
                )
                st.plotly_chart(score_fig, use_container_width=True)

            else:
                st.warning("No stocks found matching the filters. Try relaxing the criteria.")
        else:
            st.info("Click **Run Screener** to scan the sector. This fetches live data for each stock so it may take a minute.")

    with tab_news:
        st.markdown(f"##### News & Sentiment — {company_info['name']} ({ticker})")

        with st.spinner("Fetching latest news..."):
            news_result = run_news_analysis(ticker, company_info.get("name", ticker))

        if news_result and news_result.articles:
            # Sentiment overview
            sent_color = {"Bullish": "#00C853", "Slightly Bullish": "#69F0AE", "Neutral": "#FFD54F",
                          "Slightly Bearish": "#FF8A65", "Bearish": "#FF1744"}.get(news_result.sentiment_label, "#888")

            ov_c1, ov_c2, ov_c3, ov_c4 = st.columns(4)
            ov_c1.markdown(
                f"<div style='text-align:center;padding:0.8rem;background:rgba(28,31,48,0.6);"
                f"border:1px solid {sent_color}66;border-radius:10px'>"
                f"<div style='color:{sent_color};font-size:1.8rem;font-weight:700'>{news_result.sentiment_label}</div>"
                f"<div style='color:#888;font-size:0.8rem'>Overall Sentiment ({news_result.overall_sentiment:+.3f})</div>"
                f"</div>", unsafe_allow_html=True,
            )
            ov_c2.metric("Bullish", news_result.bullish_count)
            ov_c3.metric("Bearish", news_result.bearish_count)
            ov_c4.metric("Regulatory", news_result.regulatory_count)

            st.caption(news_result.summary)
            st.divider()

            # Sentiment bar chart
            sent_fig = go.Figure()
            for article in news_result.articles:
                color = "#00C853" if article.sentiment_score > 0.15 else "#FF1744" if article.sentiment_score < -0.15 else "#FFD54F"
                sent_fig.add_trace(go.Bar(
                    x=[article.sentiment_score],
                    y=[article.title[:60] + ("..." if len(article.title) > 60 else "")],
                    orientation="h",
                    marker_color=color,
                    showlegend=False,
                    hovertemplate=f"<b>{article.title}</b><br>Score: {article.sentiment_score:+.3f}<br>Publisher: {article.publisher}<extra></extra>",
                ))
            sent_fig.update_layout(
                template="plotly_dark",
                height=max(300, len(news_result.articles) * 35 + 60),
                margin=dict(l=10, r=10, t=40, b=10),
                title="Headline Sentiment Scores",
                xaxis=dict(range=[-1, 1], title="← Bearish | Bullish →"),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False, barmode="stack",
            )
            sent_fig.add_vline(x=0, line_dash="dash", line_color="rgba(255,255,255,0.2)")
            st.plotly_chart(sent_fig, use_container_width=True)

            st.divider()

            # Article cards
            st.markdown("##### Recent Headlines")
            for article in news_result.articles:
                s_color = "#00C853" if article.sentiment_label == "Bullish" else "#FF1744" if article.sentiment_label == "Bearish" else "#FFD54F"
                reg_badge = " <span style='background:#AB47BC33;color:#AB47BC;padding:2px 6px;border-radius:4px;font-size:0.7rem'>REGULATORY</span>" if article.is_regulatory else ""
                topic_badges = ""
                for t in article.key_topics[:3]:
                    topic_badges += f" <span style='background:rgba(255,255,255,0.05);color:#aaa;padding:2px 6px;border-radius:4px;font-size:0.7rem'>{t}</span>"

                st.markdown(
                    f"<div style='padding:0.7rem 0.8rem;background:rgba(28,31,48,0.6);"
                    f"border-left:3px solid {s_color};border-radius:0 10px 10px 0;margin-bottom:0.5rem'>"
                    f"<div style='font-weight:600;color:white;margin-bottom:0.3rem'>"
                    f"<a href='{article.url}' target='_blank' style='color:white;text-decoration:none'>{article.title}</a>"
                    f"</div>"
                    f"<div style='color:#aaa;font-size:0.82rem;margin-bottom:0.3rem'>{article.summary[:200]}{'...' if len(article.summary) > 200 else ''}</div>"
                    f"<div style='display:flex;gap:0.5rem;align-items:center;flex-wrap:wrap'>"
                    f"<span style='color:{s_color};font-weight:600;font-size:0.8rem'>{article.sentiment_label} ({article.sentiment_score:+.2f})</span>"
                    f"<span style='color:#666;font-size:0.75rem'>· {article.publisher} · {article.published_at}</span>"
                    f"{reg_badge}{topic_badges}"
                    f"</div></div>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("No recent news available for this stock.")

    with tab_gov:
        gov = data.get("governance")
        fund_data = data.get("fund_data", {})
        st.markdown(f"##### Governance & Red Flags — {company_info['name']} ({ticker})")

        if gov:
            # Overall risk banner
            risk_colors = {"CRITICAL": "#FF1744", "HIGH": "#FF5722", "MEDIUM": "#FFA726", "LOW": "#FFCA28", "OK": "#00C853", "UNKNOWN": "#888"}
            rc = risk_colors.get(gov.overall_risk_level, "#888")
            st.markdown(
                f"<div style='padding:1rem 1.2rem;background:linear-gradient(135deg,{rc}22,{rc}08);"
                f"border:2px solid {rc};border-radius:12px;margin-bottom:1rem'>"
                f"<div style='display:flex;align-items:center;gap:1rem'>"
                f"<div style='font-size:2.5rem'>{'🚨' if gov.overall_risk_level in ('CRITICAL','HIGH') else '⚠️' if gov.overall_risk_level == 'MEDIUM' else '✅'}</div>"
                f"<div>"
                f"<div style='color:{rc};font-size:1.5rem;font-weight:700'>{gov.overall_risk_level} RISK</div>"
                f"<div style='color:#ccc;font-size:0.9rem'>{gov.summary}</div>"
                f"<div style='color:#888;font-size:0.8rem;margin-top:0.3rem'>Risk Score: {gov.risk_score:.0f}/100</div>"
                f"</div></div></div>",
                unsafe_allow_html=True,
            )

            # ROIC / WACC / FCF Yield headline metrics
            roic_val = fund_data.get("roic")
            wacc_val = fund_data.get("wacc")
            fcf = fund_data.get("free_cashflow")
            mcap = fund_data.get("market_cap")
            fcf_yield = (fcf / mcap) if fcf and mcap and mcap > 0 else None

            st.markdown("##### Capital Efficiency Metrics")
            ce1, ce2, ce3, ce4 = st.columns(4)

            roic_str = f"{roic_val * 100:.1f}%" if roic_val else "N/A"
            wacc_str = f"{wacc_val * 100:.1f}%" if wacc_val else "N/A"
            spread = (roic_val - wacc_val) if roic_val is not None and wacc_val is not None else None
            spread_str = f"{spread * 100:.1f}%" if spread is not None else "N/A"
            fcf_y_str = f"{fcf_yield * 100:.2f}%" if fcf_yield is not None else "N/A"

            roic_color = "#00C853" if roic_val and roic_val > 0.15 else "#FFA726" if roic_val and roic_val > 0.08 else "#FF5722" if roic_val else "#888"
            spread_color = "#00C853" if spread and spread > 0.05 else "#FFA726" if spread and spread > 0 else "#FF5722" if spread else "#888"

            ce1.markdown(
                f"<div style='text-align:center;padding:0.8rem;background:rgba(28,31,48,0.6);"
                f"border:1px solid {roic_color}66;border-radius:10px'>"
                f"<div style='color:{roic_color};font-size:1.6rem;font-weight:700'>{roic_str}</div>"
                f"<div style='color:#888;font-size:0.8rem'>ROIC</div></div>",
                unsafe_allow_html=True,
            )
            ce2.markdown(
                f"<div style='text-align:center;padding:0.8rem;background:rgba(28,31,48,0.6);"
                f"border:1px solid rgba(255,255,255,0.08);border-radius:10px'>"
                f"<div style='color:#aaa;font-size:1.6rem;font-weight:700'>{wacc_str}</div>"
                f"<div style='color:#888;font-size:0.8rem'>WACC</div></div>",
                unsafe_allow_html=True,
            )
            ce3.markdown(
                f"<div style='text-align:center;padding:0.8rem;background:rgba(28,31,48,0.6);"
                f"border:1px solid {spread_color}66;border-radius:10px'>"
                f"<div style='color:{spread_color};font-size:1.6rem;font-weight:700'>{spread_str}</div>"
                f"<div style='color:#888;font-size:0.8rem'>ROIC - WACC</div></div>",
                unsafe_allow_html=True,
            )
            ce4.markdown(
                f"<div style='text-align:center;padding:0.8rem;background:rgba(28,31,48,0.6);"
                f"border:1px solid rgba(255,255,255,0.08);border-radius:10px'>"
                f"<div style='color:#69F0AE;font-size:1.6rem;font-weight:700'>{fcf_y_str}</div>"
                f"<div style='color:#888;font-size:0.8rem'>FCF Yield</div></div>",
                unsafe_allow_html=True,
            )

            if roic_val and wacc_val:
                st.caption(
                    f"**ROIC vs WACC**: A company creates value when ROIC exceeds WACC. "
                    f"{'✅ This company is creating shareholder value.' if spread and spread > 0 else '⚠️ ROIC is below WACC — the company is destroying value.'}"
                )

            st.divider()

            # ISS Governance Scores
            gov_scores = gov.governance_scores
            has_scores = any(v is not None for v in gov_scores.values())
            if has_scores:
                st.markdown("##### ISS Governance Risk Scores")
                st.caption("Scale: 1 (lowest risk) to 10 (highest risk)")

                score_keys = [
                    ("audit_risk", "Audit", "🔍"),
                    ("board_risk", "Board", "👥"),
                    ("compensation_risk", "Compensation", "💰"),
                    ("shareholder_rights_risk", "Shareholder Rights", "⚖️"),
                    ("overall_risk", "Overall", "🏛️"),
                ]
                gc_cols = st.columns(len(score_keys))
                for col, (key, label, icon) in zip(gc_cols, score_keys):
                    val = gov_scores.get(key)
                    if val is None:
                        col.markdown(f"**{icon} {label}**\n\nN/A")
                        continue
                    bar_color = "#00C853" if val <= 3 else "#FFCA28" if val <= 5 else "#FF5722" if val <= 7 else "#FF1744"
                    bar_width = val * 10
                    col.markdown(
                        f"<div style='text-align:center;padding:0.5rem;background:rgba(28,31,48,0.6);"
                        f"border-radius:10px'>"
                        f"<div style='font-size:1.2rem'>{icon}</div>"
                        f"<div style='color:white;font-weight:600;font-size:0.85rem'>{label}</div>"
                        f"<div style='color:{bar_color};font-size:1.8rem;font-weight:700'>{val}</div>"
                        f"<div style='background:rgba(255,255,255,0.08);border-radius:4px;height:6px;margin:0.3rem 0.5rem'>"
                        f"<div style='background:{bar_color};height:6px;border-radius:4px;width:{bar_width}%'></div>"
                        f"</div></div>",
                        unsafe_allow_html=True,
                    )

                st.divider()

            # Red flags detail
            st.markdown("##### Detailed Red Flag Analysis")
            non_ok = [f for f in gov.red_flags if f.severity != "OK"]
            ok_flags = [f for f in gov.red_flags if f.severity == "OK"]

            if non_ok:
                for flag in sorted(non_ok, key=lambda f: {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}.get(f.severity, 4)):
                    fc = risk_colors.get(flag.severity, "#888")
                    icon = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🔵"}.get(flag.severity, "⚪")
                    st.markdown(
                        f"<div style='padding:0.7rem 0.9rem;background:rgba(28,31,48,0.6);"
                        f"border-left:4px solid {fc};border-radius:0 10px 10px 0;margin-bottom:0.5rem'>"
                        f"<div style='display:flex;justify-content:space-between;align-items:center'>"
                        f"<span style='color:white;font-weight:600'>{icon} {flag.title}</span>"
                        f"<span style='background:{fc}33;color:{fc};padding:2px 8px;border-radius:4px;"
                        f"font-size:0.75rem;font-weight:600'>{flag.severity}</span></div>"
                        f"<div style='color:#aaa;font-size:0.82rem;margin-top:0.3rem'>{flag.detail}</div>"
                        f"<div style='color:#666;font-size:0.72rem;margin-top:0.2rem'>"
                        f"Category: {flag.category}"
                        f"{f' · Value: {flag.metric_value:.1f}' if flag.metric_value is not None else ''}"
                        f"{f' · Threshold: {flag.threshold:.0f}' if flag.threshold is not None else ''}"
                        f"</div></div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.success("No material red flags detected — this company has a clean governance profile.")

            if ok_flags:
                with st.expander(f"✅ {len(ok_flags)} areas passed checks"):
                    for flag in ok_flags:
                        st.markdown(f"- **{flag.title}**: {flag.detail}")

            st.divider()

            # Earnings track record
            if gov.earnings_track:
                st.markdown("##### Earnings Track Record — Estimates vs Actuals")
                et_fig = go.Figure()
                quarters = [e.quarter for e in gov.earnings_track]
                actuals = [e.eps_actual for e in gov.earnings_track]
                estimates = [e.eps_estimate for e in gov.earnings_track]

                et_fig.add_trace(go.Bar(
                    x=quarters, y=estimates, name="Estimate",
                    marker_color="rgba(102,126,234,0.5)",
                ))
                et_fig.add_trace(go.Bar(
                    x=quarters, y=actuals, name="Actual",
                    marker_color=["#00C853" if e.beat else "#FF1744" for e in gov.earnings_track],
                ))

                for i, e in enumerate(gov.earnings_track):
                    if e.surprise_pct is not None:
                        color = "#00C853" if e.beat else "#FF1744"
                        symbol = "+" if e.beat else ""
                        et_fig.add_annotation(
                            x=e.quarter, y=max(e.eps_actual or 0, e.eps_estimate or 0) * 1.08,
                            text=f"{symbol}{e.surprise_pct:.1f}%",
                            showarrow=False,
                            font=dict(color=color, size=10),
                        )

                et_fig.update_layout(
                    template="plotly_dark", height=320, barmode="group",
                    margin=dict(l=10, r=10, t=30, b=10),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    legend=dict(orientation="h", y=1.12),
                    yaxis_title="EPS ($)",
                )
                st.plotly_chart(et_fig, use_container_width=True)

                miss_count = sum(1 for e in gov.earnings_track if not e.beat)
                total = len(gov.earnings_track)
                beat_rate = (total - miss_count) / total * 100
                st.caption(f"Beat rate: **{beat_rate:.0f}%** ({total - miss_count}/{total} quarters)")

            st.divider()

            # Executive compensation table
            if gov.officers:
                st.markdown("##### Executive Compensation")
                paid_officers = [o for o in gov.officers if o.total_pay and o.total_pay > 0]
                if paid_officers:
                    if gov.mgmt_pay_pct is not None:
                        pay_color = "#FF1744" if gov.mgmt_pay_pct > 11 else "#FFA726" if gov.mgmt_pay_pct > 5 else "#00C853"
                        st.markdown(
                            f"<div style='padding:0.6rem 0.8rem;background:rgba(28,31,48,0.6);"
                            f"border:1px solid {pay_color}66;border-radius:10px;margin-bottom:0.8rem'>"
                            f"Total management pay: **${gov.total_mgmt_pay:,.0f}** · "
                            f"As % of net income: <span style='color:{pay_color};font-weight:700'>"
                            f"{gov.mgmt_pay_pct:.2f}%</span> "
                            f"(threshold: 11%)</div>",
                            unsafe_allow_html=True,
                        )

                    pay_fig = go.Figure()
                    sorted_officers = sorted(paid_officers, key=lambda o: o.total_pay or 0, reverse=True)
                    pay_fig.add_trace(go.Bar(
                        y=[f"{o.name}\n({o.title[:30]})" for o in sorted_officers],
                        x=[o.total_pay for o in sorted_officers],
                        orientation="h",
                        marker_color="#667eea",
                        text=[f"${o.total_pay:,.0f}" for o in sorted_officers],
                        textposition="auto",
                    ))
                    pay_fig.update_layout(
                        template="plotly_dark",
                        height=max(250, len(sorted_officers) * 50 + 60),
                        margin=dict(l=10, r=10, t=10, b=10),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        xaxis_title="Total Compensation ($)",
                    )
                    st.plotly_chart(pay_fig, use_container_width=True)
                else:
                    st.info("No compensation data available from public filings.")

            # Insider & short interest summary
            if gov.insider_ownership_pct is not None or gov.shares_short_pct is not None:
                st.divider()
                st.markdown("##### Insider & Short Interest")
                ic1, ic2 = st.columns(2)
                if gov.insider_ownership_pct is not None:
                    ic1.metric("Insider Ownership", f"{gov.insider_ownership_pct:.2f}%")
                if gov.shares_short_pct is not None:
                    ic2.metric("Short % of Float", f"{gov.shares_short_pct:.2f}%")

        else:
            st.info("Governance data not available for this stock.")

    with tab_risk:
        risk_profile = data.get("risk_profile")
        if risk_profile is None:
            st.info("Risk analysis unavailable — insufficient price data.")
        else:
            entry = risk_profile.entry_price

            # ---- User inputs for recalculation ----
            st.markdown("##### Portfolio & Risk Parameters")
            rc1, rc2, rc3 = st.columns(3)
            account_size = rc1.number_input(
                "Portfolio Size ($)", min_value=1000, max_value=100_000_000,
                value=100_000, step=5000, key=f"acct_{ticker}",
            )
            risk_pct = rc2.number_input(
                "Risk per Trade (%)", min_value=0.1, max_value=10.0,
                value=1.0, step=0.25, key=f"risk_{ticker}",
                help="Maximum % of portfolio to risk on this single trade",
            )
            custom_entry = rc3.number_input(
                "Entry Price ($)", min_value=0.01,
                value=float(round(entry, 2)), step=0.50, key=f"entry_{ticker}",
                help="Adjust if you plan to enter at a different price",
            )

            # Recalculate if user changed params
            if account_size != 100_000 or risk_pct != 1.0 or abs(custom_entry - entry) > 0.01:
                from risk_management import RiskManager as _RM
                _rm = _RM()
                target_list = []
                at = data.get("analyst_targets", {})
                if at.get("mean"):
                    target_list.append(at["mean"])
                if at.get("high") and at.get("high") != at.get("mean"):
                    target_list.append(at["high"])
                risk_profile = _rm.analyze(
                    df=data["chart_df"],
                    entry_price=custom_entry,
                    account_size=account_size,
                    risk_pct=risk_pct,
                    target_prices=target_list or None,
                )
                entry = custom_entry

            risk_amount = account_size * (risk_pct / 100)
            st.caption(f"Max risk per trade: **${risk_amount:,.0f}** ({risk_pct}% of ${account_size:,.0f})")

            st.divider()

            # ---- Volatility Summary ----
            st.markdown("##### Volatility Profile")
            vm = risk_profile.volatility_metrics
            vc1, vc2, vc3, vc4, vc5 = st.columns(5)
            vc1.metric("ATR (14)", f"${vm.get('atr_14', 0):.2f}")
            vc2.metric("Daily Vol", f"{vm.get('daily_volatility', 0) * 100:.2f}%")
            vc3.metric("Annual Vol", f"{vm.get('annualized_volatility', 0) * 100:.1f}%")
            vc4.metric("Max Drawdown", f"{risk_profile.max_drawdown:.1f}%" if risk_profile.max_drawdown else "N/A")
            vc5.metric("Sharpe Ratio", f"{risk_profile.sharpe_approx:.2f}" if risk_profile.sharpe_approx else "N/A")

            # Advanced risk metrics
            st.markdown("##### Advanced Risk Metrics")
            arm1, arm2, arm3, arm4 = st.columns(4)
            rp = data.get("risk_profile")
            if rp:
                arm1.metric("Sortino Ratio", f"{rp.sortino:.2f}" if rp.sortino else "N/A",
                            help="Like Sharpe but only penalizes downside volatility")
                arm2.metric("Calmar Ratio", f"{rp.calmar:.2f}" if rp.calmar else "N/A",
                            help="Annual return / Max Drawdown — higher is better")
                arm3.metric("VaR (95%)", f"{rp.var_95:.2%}" if rp.var_95 else "N/A",
                            help="Worst expected daily loss 95% of the time")
                arm4.metric("CVaR (95%)", f"{rp.cvar_95:.2%}" if rp.cvar_95 else "N/A",
                            help="Expected loss when VaR is breached (tail risk)")

            st.divider()

            # ---- Stop Loss Strategies ----
            st.markdown("##### Stop Loss Strategies")

            sl_chart_col, sl_price_col = st.columns([1, 1])

            with sl_chart_col:
                sl_fig = create_stop_loss_chart(entry, risk_profile.stop_losses)
                if sl_fig:
                    st.plotly_chart(sl_fig, use_container_width=True)

            with sl_price_col:
                price_fig = create_stop_on_price_chart(
                    data["chart_df"], entry, risk_profile.stop_losses, risk_profile.trailing_stops,
                )
                if price_fig:
                    st.plotly_chart(price_fig, use_container_width=True)

            # Stop loss details table
            if risk_profile.stop_losses:
                st.markdown("**Stop Loss Details**")
                for sl in risk_profile.stop_losses:
                    dist_color = score_to_color(-sl.distance_pct / 10)
                    st.markdown(
                        f"<div style='padding:0.5rem 0.8rem;background:rgba(28,31,48,0.6);"
                        f"border-left:3px solid {dist_color};border-radius:0 8px 8px 0;margin-bottom:0.4rem'>"
                        f"<span style='color:{dist_color};font-weight:700'>{sl.method}</span> — "
                        f"<span style='color:white;font-weight:600'>${sl.stop_price:.2f}</span> "
                        f"<span style='color:#aaa'>({sl.distance_pct:.1f}% below entry, "
                        f"${sl.risk_per_share:.2f}/share risk)</span><br>"
                        f"<span style='color:#888;font-size:0.85rem'>{sl.description}</span></div>",
                        unsafe_allow_html=True,
                    )

            st.divider()

            # ---- Trailing Stops ----
            st.markdown("##### Trailing Stop Strategies")
            ts = risk_profile.trailing_stops
            if ts:
                ts_cols = st.columns(min(3, len(ts)))
                for i, (key, info) in enumerate(ts.items()):
                    col = ts_cols[i % len(ts_cols)]
                    stop_val = info.get("initial_stop") or info.get("stop_price", 0)
                    trail_val = info.get("trail_amount") or info.get("trail_amount_pct", "")
                    if isinstance(trail_val, float):
                        trail_str = f"${trail_val:.2f}"
                    elif isinstance(trail_val, int):
                        trail_str = f"{trail_val}%"
                    else:
                        trail_str = str(trail_val)

                    col.markdown(
                        f"<div style='padding:0.7rem;background:rgba(28,31,48,0.6);"
                        f"border:1px solid rgba(255,255,255,0.08);border-radius:10px'>"
                        f"<div style='font-weight:600;color:#AB47BC;margin-bottom:0.3rem'>{info['method']}</div>"
                        f"<div>Initial Stop: <b>${stop_val:.2f}</b></div>"
                        f"<div>Trail: <b>{trail_str}</b></div>"
                        f"<div style='color:#888;font-size:0.82rem;margin-top:0.3rem'>{info['description']}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No trailing stop data available.")

            st.divider()

            # ---- Position Sizing ----
            st.markdown("##### Position Sizing")

            ps_fig = create_position_size_chart(risk_profile.position_sizes)
            if ps_fig:
                st.plotly_chart(ps_fig, use_container_width=True)

            if risk_profile.position_sizes:
                st.markdown("**Position Sizing Details**")
                for ps in risk_profile.position_sizes:
                    pct_color = "#69F0AE" if ps.pct_of_portfolio < 15 else "#FFD54F" if ps.pct_of_portfolio < 30 else "#FF8A65"
                    st.markdown(
                        f"<div style='padding:0.5rem 0.8rem;background:rgba(28,31,48,0.6);"
                        f"border-left:3px solid {pct_color};border-radius:0 8px 8px 0;margin-bottom:0.4rem'>"
                        f"<span style='color:{pct_color};font-weight:700'>{ps.method}</span> — "
                        f"<span style='color:white;font-weight:600'>{ps.shares:,} shares</span> "
                        f"<span style='color:#aaa'>(${ps.position_value:,.0f}, "
                        f"{ps.pct_of_portfolio:.1f}% of portfolio, "
                        f"risk: ${ps.risk_amount:,.0f})</span><br>"
                        f"<span style='color:#888;font-size:0.85rem'>{ps.description}</span></div>",
                        unsafe_allow_html=True,
                    )

            st.divider()

            # ---- Risk/Reward Analysis ----
            st.markdown("##### Risk/Reward Analysis")

            rr_fig = create_risk_reward_chart(
                entry,
                risk_profile.risk_reward,
                risk_profile.stop_losses[0].stop_price if risk_profile.stop_losses else entry * 0.95,
            )
            if rr_fig:
                st.plotly_chart(rr_fig, use_container_width=True)

            if risk_profile.risk_reward:
                st.markdown("**Scenario Details**")
                for rr in risk_profile.risk_reward:
                    ratio_color = "#00C853" if rr.ratio >= 3 else "#69F0AE" if rr.ratio >= 2 else "#FFD54F" if rr.ratio >= 1 else "#FF8A65"
                    st.markdown(
                        f"<div style='padding:0.5rem 0.8rem;background:rgba(28,31,48,0.6);"
                        f"border-left:3px solid {ratio_color};border-radius:0 8px 8px 0;margin-bottom:0.4rem'>"
                        f"<span style='color:{ratio_color};font-weight:700'>{rr.ratio:.1f}:1</span> — "
                        f"<span style='color:white;font-weight:600'>{rr.label}</span> "
                        f"<span style='color:#aaa'>(Target: ${rr.target_price:.2f}, "
                        f"+{rr.reward_pct:.1f}% reward vs {rr.risk_pct:.1f}% risk)</span></div>",
                        unsafe_allow_html=True,
                    )

            st.divider()

            # ---- Quick reference card ----
            st.markdown("##### Quick Trade Plan")
            best_size = risk_profile.position_sizes[0] if risk_profile.position_sizes else None
            primary_stop = risk_profile.stop_losses[0] if risk_profile.stop_losses else None
            best_rr = risk_profile.risk_reward[0] if risk_profile.risk_reward else None

            if best_size and primary_stop:
                st.markdown(f"""
                <div style="background:linear-gradient(135deg, rgba(102,126,234,0.15), rgba(118,75,162,0.1));
                            border:1px solid rgba(102,126,234,0.3);border-radius:12px;padding:1.5rem;margin:0.5rem 0">
                    <h4 style="margin:0 0 0.8rem 0;color:#667eea">Suggested Trade Setup</h4>
                    <table style="width:100%;color:#ccc;font-size:0.95rem">
                        <tr><td style="padding:4px 0">Entry Price</td>
                            <td style="font-weight:600;color:white">${entry:.2f}</td></tr>
                        <tr><td style="padding:4px 0">Stop Loss ({primary_stop.method})</td>
                            <td style="font-weight:600;color:#FF8A65">${primary_stop.stop_price:.2f}
                            ({primary_stop.distance_pct:.1f}% risk)</td></tr>
                        {"<tr><td style='padding:4px 0'>Target (" + best_rr.label + ")</td><td style='font-weight:600;color:#69F0AE'>$" + f'{best_rr.target_price:.2f}' + " (+" + f'{best_rr.reward_pct:.1f}' + "%, " + f'{best_rr.ratio:.1f}' + ":1 R:R)</td></tr>" if best_rr else ""}
                        <tr><td style="padding:4px 0">Position Size</td>
                            <td style="font-weight:600;color:white">{best_size.shares:,} shares
                            (${best_size.position_value:,.0f},
                            {best_size.pct_of_portfolio:.1f}% of portfolio)</td></tr>
                        <tr><td style="padding:4px 0">Max Loss</td>
                            <td style="font-weight:600;color:#FF8A65">${best_size.risk_amount:,.0f}
                            ({risk_pct:.1f}% of portfolio)</td></tr>
                    </table>
                </div>
                """, unsafe_allow_html=True)

    with tab_analysts:
        analyst_targets = data.get("analyst_targets", {})
        upgrades_downgrades = data.get("upgrades_downgrades", pd.DataFrame())
        recommendations_summary = data.get("recommendations_summary", pd.DataFrame())
        institutional_holders = data.get("institutional_holders", pd.DataFrame())
        mutualfund_holders = data.get("mutualfund_holders", pd.DataFrame())
        major_holders = data.get("major_holders", pd.DataFrame())

        # ---- Analyst Price Targets ----
        st.markdown("##### Analyst Price Targets")
        if analyst_targets:
            current = analyst_targets.get("current", company_info.get("price", 0))
            mean_t = analyst_targets.get("mean", 0)
            median_t = analyst_targets.get("median", 0)
            low_t = analyst_targets.get("low", 0)
            high_t = analyst_targets.get("high", 0)

            tc = st.columns(5)
            tc[0].metric("Current Price", f"${current:.2f}" if current else "N/A")
            tc[1].metric("Mean Target", f"${mean_t:.2f}" if mean_t else "N/A",
                         delta=f"{((mean_t - current) / current * 100):.1f}%" if current and mean_t else None)
            tc[2].metric("Median Target", f"${median_t:.2f}" if median_t else "N/A")
            tc[3].metric("Low Target", f"${low_t:.2f}" if low_t else "N/A")
            tc[4].metric("High Target", f"${high_t:.2f}" if high_t else "N/A")

            pt_fig = create_price_target_chart(current, analyst_targets)
            if pt_fig:
                st.plotly_chart(pt_fig, use_container_width=True)
        else:
            st.info("No analyst price target data available for this stock.")

        st.divider()

        # ---- Recommendation Summary ----
        st.markdown("##### Analyst Recommendation Distribution")
        if len(recommendations_summary) > 0:
            rec_fig = create_recommendations_chart(recommendations_summary)
            if rec_fig:
                st.plotly_chart(rec_fig, use_container_width=True)

            latest = recommendations_summary.iloc[0]
            rc = st.columns(5)
            for i, (col_name, label, color) in enumerate([
                ("strongBuy", "Strong Buy", "#00C853"),
                ("buy", "Buy", "#69F0AE"),
                ("hold", "Hold", "#FFD54F"),
                ("sell", "Sell", "#FF8A65"),
                ("strongSell", "Strong Sell", "#FF1744"),
            ]):
                val = int(latest.get(col_name, 0))
                rc[i].markdown(
                    f"<div style='text-align:center;padding:0.5rem;background:rgba(28,31,48,0.6);"
                    f"border:1px solid {color}44;border-radius:8px'>"
                    f"<div style='color:{color};font-size:1.5rem;font-weight:700'>{val}</div>"
                    f"<div style='color:#aaa;font-size:0.75rem'>{label}</div></div>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("No analyst recommendation data available.")

        st.divider()

        # ---- Upgrades / Downgrades ----
        st.markdown("##### Recent Analyst Actions")
        if len(upgrades_downgrades) > 0:
            ud_fig = create_upgrades_downgrades_chart(upgrades_downgrades)
            if ud_fig:
                st.plotly_chart(ud_fig, use_container_width=True)

            action_colors = {"up": "#00C853", "main": "#42A5F5", "reit": "#AB47BC", "down": "#FF1744", "init": "#FFA726"}

            for idx_val, row in upgrades_downgrades.iterrows():
                firm = row.get("Firm", "Unknown")
                action = str(row.get("Action", ""))
                to_grade = row.get("ToGrade", "")
                from_grade = row.get("FromGrade", "")
                price_action = row.get("priceTargetAction", "")
                cur_target = row.get("currentPriceTarget", 0)
                prior_target = row.get("priorPriceTarget", 0)
                date_str = str(idx_val)[:10] if idx_val is not None else ""

                color = action_colors.get(action.lower(), "#888")
                arrow = {"up": "⬆️", "down": "⬇️", "main": "➡️", "reit": "🔄", "init": "🆕"}.get(action.lower(), "•")

                target_str = ""
                if cur_target and cur_target > 0:
                    target_str = f" — Target: **${cur_target:.0f}**"
                    if prior_target and prior_target > 0 and prior_target != cur_target:
                        change = cur_target - prior_target
                        target_str += f" (was ${prior_target:.0f}, {'+' if change > 0 else ''}{change:.0f})"

                grade_str = f"**{to_grade}**"
                if from_grade and from_grade != to_grade:
                    grade_str = f"{from_grade} → **{to_grade}**"

                st.markdown(
                    f"{arrow} <span style='color:#aaa;font-size:0.8rem'>{date_str}</span> "
                    f"<span style='color:{color};font-weight:600'>{firm}</span> — "
                    f"{grade_str}{target_str}",
                    unsafe_allow_html=True,
                )
        else:
            st.info("No recent analyst upgrades/downgrades available.")

        st.divider()

        # ---- Ownership / Investors ----
        st.markdown("##### Ownership & Investors")
        own_col1, own_col2 = st.columns(2)

        with own_col1:
            pie_fig = create_holders_pie(major_holders)
            if pie_fig:
                st.plotly_chart(pie_fig, use_container_width=True)
            elif len(major_holders) > 0:
                st.markdown("**Major Holders**")
                st.dataframe(major_holders, use_container_width=True)

        with own_col2:
            bar_fig = create_institutional_bar(institutional_holders)
            if bar_fig:
                st.plotly_chart(bar_fig, use_container_width=True)

        # Institutional holders table
        if len(institutional_holders) > 0:
            st.markdown("##### Top Institutional Holders")
            inst_display = institutional_holders.copy()
            if "Value" in inst_display.columns:
                inst_display["Value"] = inst_display["Value"].apply(
                    lambda x: format_large_number(x) if pd.notna(x) else "N/A"
                )
            if "pctHeld" in inst_display.columns:
                inst_display["% Held"] = inst_display["pctHeld"].apply(
                    lambda x: f"{x * 100:.2f}%" if pd.notna(x) else "N/A"
                )
            if "Shares" in inst_display.columns:
                inst_display["Shares"] = inst_display["Shares"].apply(
                    lambda x: f"{int(x):,}" if pd.notna(x) else "N/A"
                )
            if "pctChange" in inst_display.columns:
                inst_display["Change"] = inst_display["pctChange"].apply(
                    lambda x: f"{x * 100:+.2f}%" if pd.notna(x) else "N/A"
                )
            display_cols = [c for c in ["Holder", "% Held", "Shares", "Value", "Change", "Date Reported"] if c in inst_display.columns]
            st.dataframe(inst_display[display_cols], use_container_width=True, hide_index=True)

        # Mutual fund holders table
        if len(mutualfund_holders) > 0:
            st.markdown("##### Top Mutual Fund Holders")
            mf_display = mutualfund_holders.copy()
            if "Value" in mf_display.columns:
                mf_display["Value"] = mf_display["Value"].apply(
                    lambda x: format_large_number(x) if pd.notna(x) else "N/A"
                )
            if "pctHeld" in mf_display.columns:
                mf_display["% Held"] = mf_display["pctHeld"].apply(
                    lambda x: f"{x * 100:.2f}%" if pd.notna(x) else "N/A"
                )
            if "Shares" in mf_display.columns:
                mf_display["Shares"] = mf_display["Shares"].apply(
                    lambda x: f"{int(x):,}" if pd.notna(x) else "N/A"
                )
            if "pctChange" in mf_display.columns:
                mf_display["Change"] = mf_display["pctChange"].apply(
                    lambda x: f"{x * 100:+.2f}%" if pd.notna(x) else "N/A"
                )
            display_cols = [c for c in ["Holder", "% Held", "Shares", "Value", "Change", "Date Reported"] if c in mf_display.columns]
            st.dataframe(mf_display[display_cols], use_container_width=True, hide_index=True)

    with tab_detail:
        radar_col, detail_col = st.columns([1, 1])
        with radar_col:
            st.markdown("##### Category Scores Radar")
            st.plotly_chart(
                create_category_radar(fund_result.category_scores, tech_result.category_scores),
                use_container_width=True,
            )
        with detail_col:
            st.markdown("##### Score Breakdown")
            st.markdown("**Fundamental**")
            for s in fund_result.scores:
                if s.value is not None:
                    val_str = f"{s.value:.2f}" if isinstance(s.value, (int, float)) else str(s.value)
                    st.markdown(score_bar_html(s.score, f"{s.name}: {val_str} — {s.interpretation}"), unsafe_allow_html=True)
            st.markdown("---")
            st.markdown("**Technical**")
            for s in tech_result.signals:
                if s.value is not None:
                    val_str = f"{s.value:.2f}" if isinstance(s.value, (int, float)) else str(s.value)
                    st.markdown(score_bar_html(s.score, f"{s.name}: {val_str} — {s.interpretation}"), unsafe_allow_html=True)


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
            maxdd_str = f"{rp.max_drawdown:.1f}%" if rp and rp.max_drawdown else "—"
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
    tab_overview, tab_compare, tab_portfolio, tab_individual = st.tabs([
        "📋 Overview Table",
        "📊 Comparison Charts",
        "📈 Portfolio Risk",
        "🔍 Individual Deep Dive",
    ])

    # ---- Overview table ----
    with tab_overview:
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

    with tab_portfolio:
        st.markdown("#### Portfolio Correlation & Concentration Risk")
        st.caption("Based on daily returns over the selected analysis period.")

        # Build returns matrix
        import numpy as np

        returns_dict = {}
        for t, d in stock_results.items():
            df = d.get("chart_df")
            if df is not None and "Close" in df.columns and len(df) > 10:
                returns_dict[t] = df["Close"].pct_change().dropna()

        if len(returns_dict) < 2:
            st.info("Add at least 2 stocks to see portfolio correlation analysis.")
        else:
            # Align on common dates
            returns_df = pd.DataFrame(returns_dict).dropna()

            # ---- Correlation matrix heatmap ----
            st.markdown("##### Return Correlation Matrix")
            corr = returns_df.corr()
            tickers_list = list(corr.columns)

            corr_fig = go.Figure(go.Heatmap(
                z=corr.values,
                x=tickers_list,
                y=tickers_list,
                colorscale="RdYlGn",
                zmin=-1, zmax=1,
                text=[[f"{v:.2f}" for v in row] for row in corr.values],
                texttemplate="%{text}",
                showscale=True,
            ))
            corr_fig.update_layout(
                template="plotly_dark",
                height=max(300, len(tickers_list) * 60 + 100),
                margin=dict(l=10, r=10, t=30, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(corr_fig, use_container_width=True)

            # Highly correlated pairs warning
            high_corr_pairs = []
            for i in range(len(tickers_list)):
                for j in range(i + 1, len(tickers_list)):
                    c = corr.iloc[i, j]
                    if abs(c) > 0.75:
                        high_corr_pairs.append((tickers_list[i], tickers_list[j], c))
            if high_corr_pairs:
                st.warning(
                    "**High correlation detected:** " +
                    ", ".join(f"{a}/{b} ({c:.2f})" for a, b, c in high_corr_pairs) +
                    " — these positions may not provide meaningful diversification."
                )

            st.divider()

            # ---- Concentration Risk ----
            st.markdown("##### Concentration Risk (Equal-Weight Assumption)")
            n = len(stock_results)
            weight = 1.0 / n
            weights = np.array([weight] * n)

            # Portfolio volatility
            cov = returns_df.cov() * 252
            port_var = float(weights @ cov.values @ weights)
            port_vol = port_var ** 0.5

            # Individual vols
            ind_vols = {t: float(returns_df[t].std() * (252**0.5)) for t in tickers_list}

            # Diversification ratio
            weighted_avg_vol = sum(weight * ind_vols[t] for t in tickers_list)
            div_ratio = round(weighted_avg_vol / port_vol, 2) if port_vol > 0 else 1.0

            # HHI (Herfindahl-Hirschman Index) — equal weight = 1/n
            hhi = round(sum(w**2 for w in weights) * 10000, 0)

            pc1, pc2, pc3, pc4 = st.columns(4)
            pc1.metric("Portfolio Volatility (Ann.)", f"{port_vol:.1%}")
            pc2.metric("Avg Stock Volatility", f"{weighted_avg_vol:.1%}")
            pc3.metric("Diversification Ratio", f"{div_ratio:.2f}x",
                       help="Portfolio vol / avg stock vol. Higher = better diversification")
            pc4.metric("HHI Concentration", f"{int(hhi)}",
                       help="Equal-weight HHI. 10000/n for equal weight. Lower = more diversified")

            # Individual volatility bar chart
            st.markdown("##### Individual Stock Volatility")
            vol_fig = go.Figure(go.Bar(
                x=list(ind_vols.keys()),
                y=[v * 100 for v in ind_vols.values()],
                marker_color=[
                    "#FF1744" if v > 0.40 else "#FFA726" if v > 0.25 else "#00C853"
                    for v in ind_vols.values()
                ],
                text=[f"{v:.1%}" for v in ind_vols.values()],
                textposition="auto",
            ))
            vol_fig.add_hline(
                y=port_vol * 100,
                line_dash="dash", line_color="white",
                annotation_text=f"Portfolio Vol: {port_vol:.1%}",
            )
            vol_fig.update_layout(
                template="plotly_dark",
                height=300,
                margin=dict(l=10, r=10, t=30, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                yaxis_title="Annualised Volatility (%)",
            )
            st.plotly_chart(vol_fig, use_container_width=True)

            # ---- Rolling correlation (first two stocks) ----
            if len(tickers_list) >= 2:
                st.markdown(f"##### 30-Day Rolling Correlation: {tickers_list[0]} vs {tickers_list[1]}")
                roll_corr = returns_df[tickers_list[0]].rolling(30).corr(returns_df[tickers_list[1]]).dropna()
                if len(roll_corr) > 0:
                    rc_fig = go.Figure(go.Scatter(
                        x=roll_corr.index,
                        y=roll_corr.values,
                        mode="lines",
                        line=dict(color="#42A5F5", width=1.5),
                        fill="tozeroy",
                        fillcolor="rgba(66,165,245,0.1)",
                    ))
                    rc_fig.add_hline(y=0, line_color="rgba(255,255,255,0.2)")
                    rc_fig.add_hline(y=0.75, line_dash="dash", line_color="#FFA726",
                                     annotation_text="High correlation threshold")
                    rc_fig.update_layout(
                        template="plotly_dark", height=250,
                        margin=dict(l=10, r=10, t=30, b=10),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        yaxis=dict(title="Correlation", range=[-1, 1]),
                    )
                    st.plotly_chart(rc_fig, use_container_width=True)

    # ---- Individual deep dives ----
    with tab_individual:
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
