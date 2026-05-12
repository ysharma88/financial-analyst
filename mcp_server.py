"""Financial Analyst MCP Server.

Exposes all analysis engines as MCP tools so any LLM agent framework
(Claude Desktop, LangChain, AutoGen, CrewAI, etc.) can invoke institutional-
grade equity research programmatically.

Usage:
    python mcp_server.py                         # stdio transport (default)
    python mcp_server.py --transport sse          # SSE transport on port 8000
    python mcp_server.py --transport sse --port 9000

Requirements:
    pip install mcp anthropic yfinance pandas numpy plotly ta reportlab

Environment variables (set in .env or shell):
    ANTHROPIC_API_KEY   — required for news_sentiment and reasoning tools
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import math
import os
import sys
from typing import Any

# ── resolve project root so modules can be imported ──────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dotenv import load_dotenv
load_dotenv()

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

# ── project modules ──────────────────────────────────────────────────────────
from data_fetcher import StockDataFetcher
from fundamental_analysis import FundamentalAnalyzer
from technical_analysis import TechnicalAnalyzer
from recommendation_engine import HolisticRecommendation
from risk_management import RiskManager
from macro_analysis import MacroAnalyzer
from sector_analysis import SectorAnalyzer
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
from earnings_quality import EarningsQualityAnalyzer
from capital_efficiency import CapitalEfficiencyAnalyzer
from technical_enhanced import TechnicalEnhancedAnalyzer
from credit_conditions import CreditConditionsAnalyzer
from portfolio_risk import PortfolioRiskAnalyzer
from sector_metrics import SectorMetricsAnalyzer

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("mcp_financial_analyst")

# ── helpers ───────────────────────────────────────────────────────────────────

def _to_json(obj: Any, indent: int = 2) -> str:
    """Serialize dataclasses / pandas objects to JSON string."""
    def _default(o: Any) -> Any:
        if dataclasses.is_dataclass(o) and not isinstance(o, type):
            return dataclasses.asdict(o)
        try:
            import pandas as pd
            if isinstance(o, pd.DataFrame):
                return o.to_dict(orient="records")
            if isinstance(o, pd.Series):
                return o.to_dict()
        except ImportError:
            pass
        if hasattr(o, "__dict__"):
            return o.__dict__
        if isinstance(o, (set, frozenset)):
            return list(o)
        try:
            import numpy as np
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
        except ImportError:
            pass
        return str(o)

    return json.dumps(obj, default=_default, indent=indent)


def _text(data: Any) -> list[types.TextContent]:
    return [types.TextContent(type="text", text=_to_json(data))]


def _err(msg: str) -> list[types.TextContent]:
    return [types.TextContent(type="text", text=json.dumps({"error": msg}))]


# ── MCP server ────────────────────────────────────────────────────────────────
server = Server("financial-analyst")


# ─────────────────────────────────────────────────────────────────────────────
# TOOL DEFINITIONS  (list_tools)
# ─────────────────────────────────────────────────────────────────────────────

@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [

        # ── 1. FULL ANALYSIS ──────────────────────────────────────────────
        types.Tool(
            name="analyze_stock",
            description=(
                "Run a complete institutional-grade equity analysis on a single ticker. "
                "Chains all analysis engines: data fetch → fundamental → technical → "
                "recommendation → risk → macro → sector → news sentiment → governance → "
                "quality scores (Piotroski / Altman / Beneish) → DCF intrinsic value → "
                "peer comparisons → options market signals → institutional risk (GEX, 13F, "
                "CDS proxy) → alternative data (insider flow, cluster buying, political alpha, "
                "vanna/charm) → 7-pillar reasoning synthesis with bull/bear thesis.\n\n"
                "Returns a comprehensive JSON report. This is the primary tool for generating "
                "a full investment research report. Use individual tools for targeted sub-analysis."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol (e.g. 'AAPL', 'MSFT', 'NVDA'). Case-insensitive."
                    },
                    "account_size": {
                        "type": "number",
                        "description": "Portfolio / account size in USD for position sizing calculations. Default 100000.",
                        "default": 100000
                    },
                    "risk_pct": {
                        "type": "number",
                        "description": "Max risk per trade as % of account (e.g. 1.0 = risk 1% of account). Default 1.0.",
                        "default": 1.0
                    }
                },
                "required": ["ticker"]
            }
        ),

        # ── 2. FUNDAMENTAL ANALYSIS ───────────────────────────────────────
        types.Tool(
            name="get_fundamental_analysis",
            description=(
                "Score a stock on fundamental metrics using sector-aware thresholds. "
                "Evaluates: valuation (P/E, P/B, P/S, EV/EBITDA, EV/FCF, PEG), "
                "profitability (gross/operating/net margins, ROE, ROA, ROIC), "
                "capital efficiency (asset turnover, FCF yield, EBITDA margins), "
                "growth (revenue YoY/QoQ, EPS growth, FCF growth), "
                "financial health (current ratio, debt/equity, interest coverage, Altman Z), "
                "dividends (yield, payout ratio), and analyst sentiment (price target upside, "
                "buy/sell/hold consensus).\n\n"
                "Returns overall_score (-1 bearish to +1 bullish), signal (STRONG BUY / BUY / "
                "HOLD / SELL / STRONG SELL), and per-metric breakdowns with interpretations. "
                "Use when you need to evaluate the financial quality of a company."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol (e.g. 'AAPL')."
                    }
                },
                "required": ["ticker"]
            }
        ),

        # ── 3. TECHNICAL ANALYSIS ─────────────────────────────────────────
        types.Tool(
            name="get_technical_analysis",
            description=(
                "Compute technical indicators and price action signals for a stock. "
                "Indicators computed: RSI (14), MACD (12/26/9), Bollinger Bands, "
                "ATR (14), Volume (OBV, MFI), Moving Averages (SMA 20/50/200, EMA 12/26), "
                "Stochastic, ADX/DMI, and trend regime.\n\n"
                "Also identifies support/resistance levels using 5 methods: swing highs/lows, "
                "Fibonacci retracement, volume clusters, psychological round numbers, and MA levels.\n\n"
                "Returns overall_score (-1 to +1), signal (BULLISH/BEARISH/NEUTRAL), "
                "per-indicator readings with interpretations, and a list of S/R levels with "
                "strength (strong/moderate/weak) and touch count. "
                "Use for entry/exit timing and price target setting."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol."
                    },
                    "period": {
                        "type": "string",
                        "description": "Historical data period for analysis. One of: '3mo', '6mo', '1y', '2y', '5y'. Default '1y'.",
                        "default": "1y"
                    }
                },
                "required": ["ticker"]
            }
        ),

        # ── 4. RECOMMENDATION ENGINE ──────────────────────────────────────
        types.Tool(
            name="get_recommendation",
            description=(
                "Combine fundamental and technical scores into a single holistic recommendation. "
                "Weights: 50% fundamental + 50% technical by default.\n\n"
                "Returns: overall_score, recommendation (STRONG BUY / BUY / HOLD / SELL / "
                "STRONG SELL), confidence (HIGH / MEDIUM / LOW), positive/negative key factors, "
                "and a text summary. Confidence is HIGH when both pillars agree and score > 0.3.\n\n"
                "Use this as a quick combined signal when you need a single buy/sell/hold verdict."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol."
                    },
                    "fundamental_weight": {
                        "type": "number",
                        "description": "Weight for fundamental score (0.0–1.0). Default 0.5.",
                        "default": 0.5
                    },
                    "technical_weight": {
                        "type": "number",
                        "description": "Weight for technical score (0.0–1.0). Default 0.5. Must sum to 1.0 with fundamental_weight.",
                        "default": 0.5
                    }
                },
                "required": ["ticker"]
            }
        ),

        # ── 5. RISK MANAGEMENT ────────────────────────────────────────────
        types.Tool(
            name="get_risk_profile",
            description=(
                "Compute a comprehensive risk profile for a potential trade.\n\n"
                "Outputs:\n"
                "- Stop-loss levels: ATR-based (1×/2×/3× ATR), swing low, percent-based (2%/5%/8%)\n"
                "- Trailing stops: ATR trailing, percent trailing\n"
                "- Position sizing: fixed dollar risk, Kelly Criterion, volatility-adjusted, "
                "  max position cap (10% of account)\n"
                "- Risk/reward scenarios: for each target price, shows R:R ratio and expected value\n"
                "- Performance metrics: Sharpe ratio, Sortino ratio, Calmar ratio, "
                "  max drawdown, VaR 95%, CVaR 95%\n"
                "- Volatility: ATR (14/21), daily vol, annualized vol, avg daily range\n\n"
                "Use to size positions correctly and identify stop placement before entering a trade."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol."
                    },
                    "entry_price": {
                        "type": "number",
                        "description": "Intended entry price. Use 0 or omit to use current market price."
                    },
                    "account_size": {
                        "type": "number",
                        "description": "Total account / portfolio size in USD. Default 100000.",
                        "default": 100000
                    },
                    "risk_pct": {
                        "type": "number",
                        "description": "Maximum risk per trade as % of account (e.g. 1.0 = 1%). Default 1.0.",
                        "default": 1.0
                    },
                    "target_prices": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Optional list of price targets for risk/reward calculations (e.g. [155.0, 170.0, 190.0])."
                    }
                },
                "required": ["ticker"]
            }
        ),

        # ── 6. MACRO ANALYSIS ─────────────────────────────────────────────
        types.Tool(
            name="get_macro_analysis",
            description=(
                "Analyze the current macroeconomic environment using market-traded proxies "
                "(no API keys needed — uses yfinance ETF/index data).\n\n"
                "Indicators analyzed: Treasury yield curve (2yr vs 10yr), VIX fear gauge, "
                "US Dollar Index (DXY proxy), gold (risk-off signal), crude oil (inflation/growth), "
                "S&P 500 trend (broad market), Russell 2000 (risk appetite).\n\n"
                "Returns: business cycle phase (Recovery / Expansion / Slowdown / Contraction), "
                "yield curve status (normal / flat / inverted), rate trend (rising/falling), "
                "per-indicator signals with values, and an overall macro score (-1 to +1).\n\n"
                "Use before stock analysis to understand if the macro backdrop is supportive."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),

        # ── 7. SECTOR ANALYSIS ────────────────────────────────────────────
        types.Tool(
            name="get_sector_analysis",
            description=(
                "Score all 11 GICS sectors relative to the current macro cycle and rank them. "
                "Maps each sector's historical performance pattern to the detected business cycle phase.\n\n"
                "Returns for each sector: ETF ticker, score (-1 to +1), cycle_alignment "
                "(how well the sector fits the current macro phase), 1-month performance, "
                "momentum signal, and a recommended action.\n\n"
                "Optionally scores a specific stock's sector against its peers so you can see "
                "if the stock is in a favoured or disfavoured sector.\n\n"
                "Use to identify rotation opportunities and sector tailwinds/headwinds."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "stock_sector": {
                        "type": "string",
                        "description": "Optional. Stock's GICS sector to rank relative to peers "
                                       "(e.g. 'Technology', 'Healthcare', 'Financials'). "
                                       "If omitted, returns sector ranking only."
                    }
                },
                "required": []
            }
        ),

        # ── 8. NEWS SENTIMENT ─────────────────────────────────────────────
        types.Tool(
            name="get_news_sentiment",
            description=(
                "Fetch and score recent news headlines for a ticker. Uses Claude AI for "
                "semantic sentiment scoring when ANTHROPIC_API_KEY is set; falls back to "
                "keyword-based VADER scoring otherwise.\n\n"
                "Returns: composite_score (-1 bearish to +1 bullish), bullish/neutral/bearish "
                "article counts, top headlines with individual scores and topics, recent catalysts "
                "(M&A, earnings beats, product launches, regulatory events), and an overall "
                "sentiment signal (BULLISH / NEUTRAL / BEARISH).\n\n"
                "Use to gauge market narrative and identify recent catalysts before taking a position."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol."
                    }
                },
                "required": ["ticker"]
            }
        ),

        # ── 9. GOVERNANCE ANALYSIS ────────────────────────────────────────
        types.Tool(
            name="get_governance_analysis",
            description=(
                "Detect management quality red flags and governance risks for a stock.\n\n"
                "Checks: insider ownership concentration (too high = entrenchment risk, "
                "too low = misaligned incentives), excessive executive compensation relative "
                "to company size, earnings forecast reliability (accuracy vs analyst consensus), "
                "ISS governance risk scores (if available), short interest level, "
                "recent insider purchase/sale patterns, and fraud risk proxies "
                "(asset turnover decline + accruals spike).\n\n"
                "Returns: governance_score (-1 high risk to +1 low risk), list of red flags "
                "with severity (HIGH/MEDIUM/LOW), executive pay summary, and earnings accuracy "
                "over the last 4–8 quarters.\n\n"
                "Use as a governance gate — severe red flags can override bullish technicals/fundamentals."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol."
                    }
                },
                "required": ["ticker"]
            }
        ),

        # ── 10. QUALITY SCORES ────────────────────────────────────────────
        types.Tool(
            name="get_quality_scores",
            description=(
                "Compute three classic accounting-quality and financial-health scores:\n\n"
                "1. Piotroski F-Score (0–9): Higher = stronger financial position. "
                "   ≥7 = strong, ≤2 = weak. Checks 9 binary signals across profitability, "
                "   leverage, and operating efficiency.\n\n"
                "2. Altman Z-Score: Bankruptcy prediction model. "
                "   >2.99 = safe zone, 1.81–2.99 = grey zone, <1.81 = distress zone. "
                "   Uses 5 financial ratios weighted by Altman's original coefficients.\n\n"
                "3. Beneish M-Score: Earnings manipulation detector. "
                "   >-1.78 = possible manipulation, <-2.22 = likely clean. "
                "   Uses 8 accrual and margin indices.\n\n"
                "Use to screen for accounting red flags and financial distress before investing."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol."
                    }
                },
                "required": ["ticker"]
            }
        ),

        # ── 11. DCF VALUATION ─────────────────────────────────────────────
        types.Tool(
            name="get_dcf_valuation",
            description=(
                "Run a 3-stage Discounted Cash Flow (DCF) model to estimate intrinsic value.\n\n"
                "Model structure:\n"
                "- Stage 1 (years 1–5): explicit FCF forecast using current revenue growth\n"
                "- Stage 2 (years 6–10): fade period where growth gradually decays toward terminal rate\n"
                "- Stage 3: terminal value using Gordon Growth Model\n\n"
                "Calculates WACC from cost of equity (CAPM) and after-tax cost of debt.\n\n"
                "Returns: intrinsic_value_per_share, current_price, margin_of_safety (%), "
                "upside_downside (%), valuation_signal (UNDERVALUED/FAIR/OVERVALUED), "
                "and a sensitivity table showing intrinsic value across WACC ± 1%/2% and "
                "terminal growth rate ± 0.5%/1.0% combinations.\n\n"
                "Use to determine if a stock is cheap or expensive relative to its fundamental value."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol."
                    }
                },
                "required": ["ticker"]
            }
        ),

        # ── 12. PEER COMPARISONS ──────────────────────────────────────────
        types.Tool(
            name="get_peer_comparisons",
            description=(
                "Compare a stock against sector peers on key valuation and quality multiples.\n\n"
                "Multiples compared: P/E ratio, EV/EBITDA, Price/Sales, Price/Book, "
                "ROE, ROIC, gross margin, operating margin, debt/equity, revenue growth.\n\n"
                "Returns: list of peers with their multiples, sector median for each multiple, "
                "subject stock's percentile rank (0 = cheapest/worst, 100 = most expensive/best "
                "— direction depends on metric), and a premium/discount vs. median for each ratio.\n\n"
                "Use to identify if a stock is cheap or expensive relative to its peer group, "
                "and to find the best-in-class name in a sector."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol."
                    },
                    "max_peers": {
                        "type": "integer",
                        "description": "Maximum number of peer stocks to include. Default 8.",
                        "default": 8
                    }
                },
                "required": ["ticker"]
            }
        ),

        # ── 13. OPTIONS ANALYSIS ──────────────────────────────────────────
        types.Tool(
            name="get_options_analysis",
            description=(
                "Analyze options market structure and implied volatility for a stock.\n\n"
                "Computes:\n"
                "- Implied Volatility (IV): current 30-day IV from ATM options\n"
                "- Realized Volatility: 30-day historical vol from price returns\n"
                "- IV Rank: where current IV sits vs its 52-week range (0–100%)\n"
                "- IV Percentile: % of past year IV was below current level\n"
                "- Put/Call OI ratio and Put/Call volume ratio (>1 = bearish sentiment)\n"
                "- Max Pain: price where aggregate option holder loss is maximized\n"
                "  (market makers are motivated to pin price here near expiry)\n"
                "- Nearest expiry chain summary: strikes, OI, and volume\n\n"
                "Returns signal: HIGH_IV_SELL_PREMIUM / LOW_IV_BUY_OPTIONS / NEUTRAL.\n\n"
                "Use to time options strategies and gauge sentiment via put/call ratios."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol."
                    }
                },
                "required": ["ticker"]
            }
        ),

        # ── 14. INSTITUTIONAL RISK ────────────────────────────────────────
        types.Tool(
            name="get_institutional_risk",
            description=(
                "Institutional-grade risk analytics across three dimensions:\n\n"
                "1. Gamma Exposure (GEX): Map dealer hedging obligations by strike. "
                "   Positive GEX = dealers long gamma → acts as support (price gravity). "
                "   Negative GEX = dealers short gamma → amplifies moves (trapdoor). "
                "   Identifies gamma flip level, key walls and trapdoors.\n\n"
                "2. Smart Money Flow (13F): Detect institutional conviction and momentum. "
                "   Counts high-conviction holders (>1% ownership), computes concentration "
                "   (HHI-equivalent), and flags if institutions are adding or reducing.\n\n"
                "3. Credit Proxy (synthetic CDS): Estimate credit risk without bond data. "
                "   Computes synthetic spread from interest coverage and debt/EBITDA, "
                "   CS01 (credit sensitivity per 1bp), DV01 (rate sensitivity per 1bp), "
                "   and implied credit rating bucket.\n\n"
                "Use to understand dealer positioning, institutional conviction, and credit risk."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol."
                    }
                },
                "required": ["ticker"]
            }
        ),

        # ── 15. ALTERNATIVE DATA ──────────────────────────────────────────
        types.Tool(
            name="get_alternative_data",
            description=(
                "Five layers of alternative data signals (all from free public sources):\n\n"
                "Layer 1 — News Sentiment: VADER composite score from recent headlines. "
                "  Institutional trigger fires at ±0.225 (algo momentum threshold).\n\n"
                "Layer 2 — Insider Flow: Net Form 4 buying/selling pressure. "
                "  Detects cluster exits (multiple insiders selling simultaneously), "
                "  10b5-1 plan adoptions, and net insider conviction.\n\n"
                "Layer 3 — 13F Cluster Buying: Flags when 3+ top institutions initiated "
                "  new positions in the same quarter — a high-conviction consensus signal.\n\n"
                "Layer 4 — Vanna & Charm Flow: Second-order options Greeks. "
                "  Positive Vanna → vol-crush forces dealers to BUY mechanically. "
                "  Positive Charm → daily theta decay creates passive dealer buying support.\n\n"
                "Layer 5 — Congressional Trades (Political Alpha): House and Senate member "
                "  stock disclosures. Flags recent trades in the same ticker as an early "
                "  signal for regulatory events or sector tailwinds.\n\n"
                "Returns per-layer scores and signals plus an aggregate alt_data_score."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol."
                    }
                },
                "required": ["ticker"]
            }
        ),

        # ── 16. SUPPORT & RESISTANCE LEVELS ──────────────────────────────
        types.Tool(
            name="get_support_resistance",
            description=(
                "Identify key support and resistance price levels for trade entry/exit planning.\n\n"
                "Detection methods used (5 independent approaches):\n"
                "1. Swing highs / swing lows (local price extremes with configurable lookback)\n"
                "2. Fibonacci retracement levels (23.6%, 38.2%, 50%, 61.8%, 78.6%) from recent swing\n"
                "3. Volume clusters (price levels with high historical traded volume)\n"
                "4. Psychological levels (round numbers: $X0, $X5, $X00)\n"
                "5. Moving average levels (20/50/200-day SMA acting as dynamic S/R)\n\n"
                "Each level is tagged with: price, kind (support/resistance), strength "
                "(strong/moderate/weak), method, and touch count.\n\n"
                "Use to identify buy zones (supports below price) and price targets (resistances above)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol."
                    },
                    "period": {
                        "type": "string",
                        "description": "Historical lookback period for level detection. Default '1y'. Options: '3mo', '6mo', '1y', '2y'.",
                        "default": "1y"
                    }
                },
                "required": ["ticker"]
            }
        ),

        # ── 17. STOCK SCREENER ────────────────────────────────────────────
        types.Tool(
            name="screen_sector",
            description=(
                "Screen and rank the best stocks within a GICS sector using quantitative filters.\n\n"
                "Fetches top companies in the sector from yfinance, then filters and scores them "
                "using: composite score (fundamental + technical + momentum), market cap filter, "
                "valuation filter (max P/E), balance sheet filter (max debt/equity), "
                "profitability filter (min ROE), and growth filter (min revenue growth).\n\n"
                "Returns a ranked list of stocks with: ticker, company name, composite score, "
                "current price, market cap, P/E, ROE, revenue growth, and a brief thesis.\n\n"
                "Supported sectors: Technology, Healthcare, Financials, Consumer Cyclical, "
                "Consumer Defensive, Industrials, Energy, Utilities, Real Estate, "
                "Basic Materials, Communication Services.\n\n"
                "Use to find the best opportunity within a sector for rotation or new positions."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "sector": {
                        "type": "string",
                        "description": "GICS sector name (e.g. 'Technology', 'Healthcare', 'Financials')."
                    },
                    "max_stocks": {
                        "type": "integer",
                        "description": "Maximum number of candidate stocks to evaluate. Default 25.",
                        "default": 25
                    },
                    "min_market_cap": {
                        "type": "number",
                        "description": "Minimum market cap in USD (e.g. 1e9 = $1B). Default 0 (no filter).",
                        "default": 0
                    },
                    "max_pe": {
                        "type": "number",
                        "description": "Maximum P/E ratio filter. Omit for no filter."
                    },
                    "max_debt_equity": {
                        "type": "number",
                        "description": "Maximum debt/equity ratio filter. Omit for no filter."
                    },
                    "min_roe": {
                        "type": "number",
                        "description": "Minimum Return on Equity (0–1 scale, e.g. 0.15 = 15%). Omit for no filter."
                    },
                    "min_revenue_growth": {
                        "type": "number",
                        "description": "Minimum revenue growth rate (0–1 scale, e.g. 0.10 = 10% YoY). Omit for no filter."
                    }
                },
                "required": ["sector"]
            }
        ),

        # ── 18. REASONING / THESIS SYNTHESIS ─────────────────────────────
        types.Tool(
            name="synthesize_investment_thesis",
            description=(
                "Generate a 7-pillar institutional investment thesis with full reasoning chain.\n\n"
                "Decision tree:\n"
                "1. Governance Gate — severe flags veto the thesis regardless of other signals\n"
                "2. Macro — is the macro backdrop supportive (cycle phase, yield curve)?\n"
                "3. Sector — is the stock's sector favoured in the current cycle?\n"
                "4. Fundamental — what do the financial metrics say?\n"
                "5. Technical — what does price action say (trend, momentum, S/R)?\n"
                "6. Sentiment — what is the news/analyst narrative?\n"
                "7. Risk — does the risk/reward justify the trade?\n\n"
                "Returns: final_verdict (STRONG BUY / BUY / HOLD / SELL / STRONG SELL / VETO), "
                "confidence_score, narrative_thesis (2–3 paragraph research note), "
                "bull_case, bear_case, catalysts, risks, action_plan, and per-pillar step scores.\n\n"
                "This tool calls ALL sub-analyses internally — just pass the ticker. "
                "Use when you need a complete written investment memo."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol."
                    },
                    "account_size": {
                        "type": "number",
                        "description": "Account size in USD for position sizing in the action plan. Default 100000.",
                        "default": 100000
                    }
                },
                "required": ["ticker"]
            }
        ),

        # ── 19. STOCK DATA FETCH ──────────────────────────────────────────
        types.Tool(
            name="get_stock_data",
            description=(
                "Fetch raw financial data for a stock from Yahoo Finance (cached). "
                "Returns a structured dict with current price, market cap, P/E ratio, "
                "EPS, revenue, gross/operating/net margins, ROE, ROA, ROIC, debt/equity, "
                "current ratio, dividend yield, beta, 52-week high/low, analyst targets, "
                "and recommendation consensus.\n\n"
                "Use this when you need raw fundamental data without the scoring layer, "
                "or to inspect what data is available before running analysis."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol."
                    }
                },
                "required": ["ticker"]
            }
        ),

        # ── 20. GEX (GAMMA EXPOSURE) ──────────────────────────────────────
        types.Tool(
            name="get_gamma_exposure",
            description=(
                "Compute Dealer Gamma Exposure (GEX) map from live options chains. "
                "Shows where dealers are long vs short gamma — key for understanding "
                "what price levels will attract buying (walls) or amplify selling (trapdoors).\n\n"
                "Positive GEX at a strike → dealers must BUY as price falls toward it (support wall). "
                "Negative GEX at a strike → dealers must SELL as price falls through it (trapdoor).\n\n"
                "Returns: gamma_flip_level (price where net GEX crosses zero), "
                "top walls (support levels with highest positive GEX), "
                "top trapdoors (resistance with highest negative GEX), "
                "net_gex (total dealer exposure), and GEX by strike.\n\n"
                "Use to identify institutional price magnets and avoid stop-hunt zones."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol."
                    }
                },
                "required": ["ticker"]
            }
        ),

        # ── 21. CONGRESSIONAL TRADES ──────────────────────────────────────
        types.Tool(
            name="get_congressional_trades",
            description=(
                "Fetch recent Congressional stock disclosures (House and Senate) for a ticker "
                "via public APIs — no API key required.\n\n"
                "Returns: list of trades with member name, chamber, trade date, "
                "transaction type (Purchase/Sale), amount range, and filing date. "
                "Also returns a signal (BULLISH if net purchases > sales, BEARISH otherwise) "
                "and a political_alpha_score.\n\n"
                "Regulatory note: Congressional trades are disclosed with a 45-day lag. "
                "Recent purchases may signal upcoming regulatory favours or sector legislation. "
                "Use as an early indicator of political tailwinds or headwinds."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol."
                    }
                },
                "required": ["ticker"]
            }
        ),

        # ── 22. INSIDER FLOW ──────────────────────────────────────────────
        types.Tool(
            name="get_insider_flow",
            description=(
                "Analyze insider buying and selling activity from SEC Form 4 filings (via yfinance).\n\n"
                "Detects:\n"
                "- Net insider buy/sell pressure (dollar-weighted)\n"
                "- Cluster exit signal: multiple insiders selling in the same 90-day window\n"
                "- 10b5-1 plan adoptions (pre-planned sales — less informative than discretionary)\n"
                "- Largest single transactions by dollar value\n\n"
                "Returns: net_flow (positive = net buying, negative = net selling), "
                "cluster_exit (bool), plan_adoption (bool), buy_count, sell_count, "
                "total_buy_value, total_sell_value, and signal (BULLISH / BEARISH / NEUTRAL).\n\n"
                "Insider buying is one of the strongest fundamental signals — executives "
                "buy with their own money only when they expect the stock to rise."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol."
                    }
                },
                "required": ["ticker"]
            }
        ),

        # ── 23. EARNINGS QUALITY ──────────────────────────────────────────
        types.Tool(
            name="get_earnings_quality",
            description=(
                "Institutional earnings quality analysis — detects the gap between reported "
                "earnings and real economic cash generation across 6 signal layers:\n\n"
                "1. FCF Conversion Rate (FCF/Net Income): ≥90% = pristine, <50% = accrual-heavy. "
                "   Low conversion means earnings are paper profits, not cash.\n"
                "2. Sloan's Accruals Ratio: Predicts future earnings disappointments. "
                "   High accruals (>+15%) = earnings likely to reverse. "
                "   One of the most cited forensic accounting signals (Sloan 1996).\n"
                "3. AR Growth vs Revenue Growth divergence: AR growing faster than revenue "
                "   means customers are paying slower — classic revenue recognition red flag "
                "   used by every short-selling fund.\n"
                "4. EPS Surprise Magnitude & Trend: Consistent large beats predict continued "
                "   outperformance (Post-Earnings Announcement Drift). Scores both magnitude "
                "   and whether the trend is improving or deteriorating.\n"
                "5. SBC as % of FCF: Stock-based compensation is a real cost that non-GAAP "
                "   earnings hide. SBC/FCF >75% means the company isn't really generating "
                "   free cash for equity holders.\n"
                "6. Share Dilution Rate: Net equity issuance each year dilutes per-share value. "
                "   Negative (buybacks) = accretive, >3% issuance = meaningful headwind.\n\n"
                "Returns: overall_score (-1 to +1), signal (HIGH QUALITY → LOW QUALITY), "
                "per-layer scores, and actionable quality_flags. "
                "Use before any buy decision to verify reported earnings are real."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol."}
                },
                "required": ["ticker"]
            }
        ),

        # ── 24. CAPITAL EFFICIENCY ────────────────────────────────────────
        types.Tool(
            name="get_capital_efficiency",
            description=(
                "Capital efficiency and allocation analysis across three dimensions:\n\n"
                "1. Working Capital (Cash Conversion Cycle):\n"
                "   DSO (Days Sales Outstanding) — how fast customers pay. Rising DSO = concern.\n"
                "   DIO (Days Inventory Outstanding) — inventory efficiency.\n"
                "   DPO (Days Payable Outstanding) — how long the company takes to pay suppliers.\n"
                "   CCC = DSO + DIO - DPO. Lower is better. YoY improvement is bullish.\n\n"
                "2. Capex Intensity:\n"
                "   Capex/Revenue: <3% = asset-light, 3-8% = moderate, >8% = capital-intensive.\n"
                "   Capex/D&A: <0.8 = harvesting assets (underinvesting), >1.2 = growth mode.\n"
                "   Estimates maintenance vs. growth capex split.\n\n"
                "3. Capital Return & Leverage:\n"
                "   Buyback Yield + Dividend Yield = Total Shareholder Yield.\n"
                "   Net Debt/EBITDA (surfaced as standalone metric).\n"
                "   Net Debt trajectory YoY — direction of leverage change.\n\n"
                "Returns: per-dimension metrics, quality_flags, overall_score (-1 to +1). "
                "Use to evaluate capital allocation quality and whether management "
                "is creating or destroying value through investment and financing decisions."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol."}
                },
                "required": ["ticker"]
            }
        ),

        # ── 25. ENHANCED TECHNICAL ────────────────────────────────────────
        types.Tool(
            name="get_technical_enhanced",
            description=(
                "Institutional momentum and market-structure signals beyond standard indicators:\n\n"
                "1. Relative Strength vs SPY (O'Neil RS Rating): "
                "   Stock return / SPY return for 1m/3m/6m/12m. Top-quintile RS stocks "
                "   outperform by 2-5%/year (Jegadeesh & Titman). Composite weighted 10/20/30/40%.\n\n"
                "2. Multi-Period Price ROC (1m/3m/6m/12m): The classic momentum factor. "
                "   momentum_12_1 = 12m minus 1m return = canonical cross-sectional momentum "
                "   used by every quant fund (AQR, Dimensional, Man AHL).\n\n"
                "3. 52-Week High Proximity: Stocks within 3% of 52-week high have strong "
                "   breakout follow-through. Stocks at 52-week lows may indicate structural problems.\n\n"
                "4. VWAP (90-day + YTD): Institutional execution benchmark. "
                "   Price above VWAP = institutional positions in profit → holders less likely to sell. "
                "   Price below VWAP = institutional positions at a loss → overhead supply risk.\n\n"
                "5. Short Interest + Days to Cover: DTC >10 days with improving fundamentals "
                "   = potential short squeeze. Rising short interest with weak fundamentals = bearish.\n\n"
                "6. Options-Implied Expected Move: ATM straddle / price. "
                "   How much the market expects the stock to move — use to size positions around events.\n\n"
                "7. Volatility Skew (Put Skew): OTM put IV minus ATM IV. "
                "   Steep negative skew = institutional fear/hedging demand. More sensitive than VIX.\n\n"
                "Returns: per-signal metrics, overall_score, signal (BULLISH MOMENTUM → BEARISH MOMENTUM)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol."},
                    "period": {
                        "type": "string",
                        "description": "Historical lookback period. Default '1y'.",
                        "default": "1y"
                    }
                },
                "required": ["ticker"]
            }
        ),

        # ── 26. CREDIT CONDITIONS ─────────────────────────────────────────
        types.Tool(
            name="get_credit_conditions",
            description=(
                "Cross-asset credit market signals for equity risk assessment. "
                "Credit markets price in risk BEFORE equities — this is the leading indicator "
                "layer that macro hedge funds (Bridgewater, Tudor, Millennium) monitor daily.\n\n"
                "Signals derived from bond ETF market data (no API keys needed):\n\n"
                "1. HY Credit Spread (HYG vs IEF): High-yield spreads widening = credit stress "
                "   ahead. HY typically leads equity selloffs by 2-4 weeks. Z-score shows "
                "   how extreme current conditions are vs. the past 6 months.\n\n"
                "2. IG Credit Spread (LQD vs IEF): Investment-grade spreads widening = "
                "   systemic risk growing. Tighter = financial conditions easing.\n\n"
                "3. Real Interest Rates (TIP vs IEF): When real rates rise (IEF outperforms TIP), "
                "   equity multiples compress. This was the mechanism behind the 2022 tech selloff. "
                "   The system tracks nominal rates already — this adds the real rate dimension.\n\n"
                "4. HYG vs SPY divergence: When HYG underperforms SPY by >3% over 1 month, "
                "   equities historically catch down to credit within weeks.\n\n"
                "Returns: credit_score (-1 stress to +1 easing), financial_conditions_signal "
                "(EASING/NEUTRAL/TIGHTENING/STRESS), per-spread metrics with z-scores, "
                "and 2-4 actionable equity_implications. "
                "Run this before any large position — if credit says STRESS, reduce size."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),

        # ── 27. PORTFOLIO RISK ────────────────────────────────────────────
        types.Tool(
            name="get_portfolio_risk",
            description=(
                "Portfolio-level risk analytics — what individual stock analysis misses.\n\n"
                "1. Factor Exposures (Barra-lite): Score the stock on 5 institutional factors:\n"
                "   Value (0-100), Momentum (0-100), Quality (0-100), Low-Volatility (0-100), "
                "   Growth (0-100). Identifies which factor the stock is a bet on — critical "
                "   for avoiding unintended factor concentration.\n\n"
                "2. Rolling Alpha & Beta vs SPY (252-day):\n"
                "   Jensen's Alpha (annualized): excess return above what CAPM predicts.\n"
                "   Beta: market sensitivity.\n"
                "   Information Ratio = Alpha / Tracking Error — the standard measure of skill.\n\n"
                "3. Correlation to Portfolio Holdings: Checks correlation of this stock against "
                "   your existing positions. Flags if any correlation >0.75 — adding a highly "
                "   correlated position concentrates risk rather than diversifying it.\n\n"
                "4. Beta-Adjusted Position Sizing: Sizes the position so it contributes 1/10th "
                "   of target portfolio beta — the correct institutional approach vs. fixed-dollar sizing.\n\n"
                "Returns: factor scores, benchmark metrics, correlation warnings, "
                "beta-adjusted share count, and quality_flags. "
                "Use before adding any new position to an existing portfolio."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol."},
                    "account_size": {
                        "type": "number",
                        "description": "Total portfolio size in USD. Default 100000.",
                        "default": 100000
                    },
                    "target_beta": {
                        "type": "number",
                        "description": "Target portfolio beta (e.g. 1.0 = market-neutral). Default 1.0.",
                        "default": 1.0
                    }
                },
                "required": ["ticker"]
            }
        ),

        # ── 28. SECTOR-SPECIFIC METRICS ───────────────────────────────────
        types.Tool(
            name="get_sector_metrics",
            description=(
                "Sector-specific financial metrics that standard equity analysis misses. "
                "Standard P/E and EV/EBITDA are wrong or misleading for certain sectors — "
                "this module auto-detects the sector and applies the correct framework:\n\n"
                "REITs (Real Estate):\n"
                "  FFO (Funds From Operations) = Net Income + D&A - Property Gains\n"
                "  AFFO = FFO - Recurring Capex. P/FFO and P/AFFO are the required multiples.\n"
                "  P/E is meaningless for REITs due to non-cash depreciation charges.\n\n"
                "Banks / Financial Services:\n"
                "  Net Interest Margin (NIM proxy): NII / Avg Assets — the primary profitability driver.\n"
                "  Efficiency Ratio: OpEx / Revenue — <55% = well-run bank.\n\n"
                "SaaS / Software:\n"
                "  Rule of 40: Revenue Growth % + FCF Margin % ≥ 40 = healthy SaaS business.\n"
                "  SBC-adjusted FCF Margin: strips out non-GAAP flattery.\n"
                "  R&D Intensity: R&D / Revenue — is the company investing in moat or milking it?\n\n"
                "Consumer / Retail:\n"
                "  Gross Margin Trend: pricing power vs. cost pressure over time.\n"
                "  Revenue per Employee: operational productivity benchmark.\n\n"
                "All Sectors (Universal):\n"
                "  EV/Revenue: correctly scored (was fetched but never scored in base system).\n"
                "  EV/EBIT: more conservative than EV/EBITDA for asset-heavy businesses.\n"
                "  Total Shareholder Yield: dividend + buyback yield combined.\n\n"
                "Returns: applicable_module (which sector framework was used), "
                "universal metrics, sector-specific metrics, overall_score, and quality_flags."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol."}
                },
                "required": ["ticker"]
            }
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# TOOL HANDLERS  (call_tool)
# ─────────────────────────────────────────────────────────────────────────────

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    try:
        return await asyncio.get_event_loop().run_in_executor(
            None, _dispatch_sync, name, arguments
        )
    except Exception as exc:
        logger.exception("Tool %s raised %s", name, exc)
        return _err(f"{type(exc).__name__}: {exc}")


def _dispatch_sync(name: str, args: dict) -> list[types.TextContent]:
    """Run the tool synchronously (called in thread-pool executor)."""

    # ── helpers ──────────────────────────────────────────────────────────────
    def _fetcher(ticker: str) -> StockDataFetcher:
        return StockDataFetcher(ticker.upper().strip())

    # ─────────────────────────────────────────────────────────────────────────

    if name == "get_stock_data":
        ticker = args["ticker"]
        f = _fetcher(ticker)
        data = {
            "ticker": ticker.upper(),
            "company_name": f.get_company_name(),
            "sector": f.get_sector(),
            "industry": f.get_industry(),
            "current_price": f.get_current_price(),
            "market_cap": f.get_market_cap(),
            "fundamental_data": f.get_fundamental_data(),
            "analyst_price_targets": f.get_analyst_price_targets(),
            "recommendations_summary": f.get_recommendations_summary(),
            "upgrades_downgrades": f.get_upgrades_downgrades(10),
        }
        return _text(data)

    # ─────────────────────────────────────────────────────────────────────────

    elif name == "get_fundamental_analysis":
        ticker = args["ticker"]
        f = _fetcher(ticker)
        fund_data = f.get_fundamental_data()
        sector = f.get_sector() or ""
        result = FundamentalAnalyzer().analyze(fund_data, sector)
        return _text(result)

    # ─────────────────────────────────────────────────────────────────────────

    elif name == "get_technical_analysis":
        ticker = args["ticker"]
        period = args.get("period", "1y")
        f = _fetcher(ticker)
        hist = f.get_history(period=period)
        if hist is None or len(hist) < 20:
            return _err(f"Insufficient price history for {ticker} (period={period}).")
        result = TechnicalAnalyzer().analyze(hist)
        return _text(result)

    # ─────────────────────────────────────────────────────────────────────────

    elif name == "get_recommendation":
        ticker = args["ticker"]
        fw = float(args.get("fundamental_weight", 0.5))
        tw = float(args.get("technical_weight", 0.5))
        f = _fetcher(ticker)
        fund_data = f.get_fundamental_data()
        sector = f.get_sector() or ""
        hist = f.get_history(period="1y")
        if hist is None or len(hist) < 20:
            return _err(f"Insufficient history for {ticker}.")
        fund_result = FundamentalAnalyzer().analyze(fund_data, sector)
        tech_result = TechnicalAnalyzer().analyze(hist)
        rec = HolisticRecommendation(
            fundamental=fund_result,
            technical=tech_result,
            fundamental_weight=fw,
            technical_weight=tw,
        )
        return _text(rec)

    # ─────────────────────────────────────────────────────────────────────────

    elif name == "get_risk_profile":
        ticker = args["ticker"]
        f = _fetcher(ticker)
        hist = f.get_history(period="1y")
        if hist is None or len(hist) < 20:
            return _err(f"Insufficient history for {ticker}.")
        entry = float(args.get("entry_price") or f.get_current_price() or 0)
        if entry == 0:
            return _err("Could not determine entry price — pass entry_price explicitly.")
        targets = [float(x) for x in (args.get("target_prices") or [])]
        profile = RiskManager().analyze(
            df=hist,
            entry_price=entry,
            account_size=float(args.get("account_size", 100_000)),
            risk_pct=float(args.get("risk_pct", 1.0)),
            target_prices=targets or None,
        )
        return _text(profile)

    # ─────────────────────────────────────────────────────────────────────────

    elif name == "get_macro_analysis":
        result = MacroAnalyzer().analyze()
        return _text(result)

    # ─────────────────────────────────────────────────────────────────────────

    elif name == "get_sector_analysis":
        macro = MacroAnalyzer().analyze()
        cycle_phase = macro.cycle.phase if macro.cycle else "Unknown"
        rates_rising = macro.rates_rising if hasattr(macro, "rates_rising") else False
        sa = SectorAnalyzer()
        sector_result = sa.analyze(cycle_phase=cycle_phase, rates_rising=rates_rising)
        out: dict[str, Any] = {"sector_scores": sector_result}
        stock_sector = args.get("stock_sector")
        if stock_sector:
            rank = sa.score_stock_sector(sector_result, stock_sector)
            out["stock_sector_rank"] = rank
        return _text(out)

    # ─────────────────────────────────────────────────────────────────────────

    elif name == "get_news_sentiment":
        ticker = args["ticker"]
        f = _fetcher(ticker)
        company = f.get_company_name() or ticker
        result = NewsSentimentAnalyzer().analyze(ticker=ticker, company_name=company)
        return _text(result)

    # ─────────────────────────────────────────────────────────────────────────

    elif name == "get_governance_analysis":
        ticker = args["ticker"]
        f = _fetcher(ticker)
        info = f.info
        result = GovernanceAnalyzer().analyze(
            info=info,
            company_officers=f.get_company_officers(),
            earnings_history=f.get_earnings_history(),
            governance_scores=f.get_governance_scores(),
            insider_purchases=f.get_insider_purchases(),
        )
        return _text(result)

    # ─────────────────────────────────────────────────────────────────────────

    elif name == "get_quality_scores":
        ticker = args["ticker"]
        f = _fetcher(ticker)
        scores = QualityScorer().compute(
            info=f.info,
            income_stmt=f.get_income_statement(),
            balance_sheet=f.get_balance_sheet(),
            cashflow=f.get_cashflow_statement(),
        )
        return _text(scores)

    # ─────────────────────────────────────────────────────────────────────────

    elif name == "get_dcf_valuation":
        ticker = args["ticker"]
        f = _fetcher(ticker)
        result = DCFValuator().compute(
            info=f.info,
            income_stmt=f.get_income_statement(),
            balance_sheet=f.get_balance_sheet(),
            cashflow=f.get_cashflow_statement(),
        )
        return _text(result)

    # ─────────────────────────────────────────────────────────────────────────

    elif name == "get_peer_comparisons":
        ticker = args["ticker"]
        f = _fetcher(ticker)
        result = PeerComparator().compare(
            subject_ticker=ticker,
            subject_info=f.info,
            sector=f.get_sector() or "",
            max_peers=int(args.get("max_peers", 8)),
        )
        return _text(result)

    # ─────────────────────────────────────────────────────────────────────────

    elif name == "get_options_analysis":
        ticker = args["ticker"]
        f = _fetcher(ticker)
        hist = f.get_history(period="1y")
        if hist is None or len(hist) < 20:
            return _err(f"Insufficient history for {ticker}.")
        result = OptionsAnalyzer().analyze(ticker=ticker, history=hist)
        return _text(result)

    # ─────────────────────────────────────────────────────────────────────────

    elif name == "get_institutional_risk":
        ticker = args["ticker"]
        f = _fetcher(ticker)
        price = f.get_current_price() or 0.0
        result = _inst_risk.analyze(
            ticker=ticker,
            price=price,
            info=f.info,
            income_stmt=f.get_income_statement(),
        )
        return _text(result)

    # ─────────────────────────────────────────────────────────────────────────

    elif name == "get_alternative_data":
        ticker = args["ticker"]
        f = _fetcher(ticker)
        price = f.get_current_price() or 0.0
        result = _alt_data.cached_analyze(ticker=ticker, price=price)
        return _text(result)

    # ─────────────────────────────────────────────────────────────────────────

    elif name == "get_support_resistance":
        ticker = args["ticker"]
        period = args.get("period", "1y")
        f = _fetcher(ticker)
        hist = f.get_history(period=period)
        if hist is None or len(hist) < 20:
            return _err(f"Insufficient history for {ticker}.")
        tech = TechnicalAnalyzer().analyze(hist)
        levels = tech.support_resistance or []
        price = f.get_current_price() or 0.0
        supports = sorted(
            [l for l in levels if l.kind == "support" and l.price < price],
            key=lambda l: l.price, reverse=True
        )
        resistances = sorted(
            [l for l in levels if l.kind == "resistance" and l.price > price],
            key=lambda l: l.price
        )
        return _text({
            "ticker": ticker.upper(),
            "current_price": price,
            "supports": supports,
            "resistances": resistances,
            "all_levels": levels,
        })

    # ─────────────────────────────────────────────────────────────────────────

    elif name == "screen_sector":
        result = StockScreener().screen_sector(
            sector=args["sector"],
            max_stocks=int(args.get("max_stocks", 25)),
            min_market_cap=float(args.get("min_market_cap", 0)),
            max_pe=args.get("max_pe"),
            max_debt_equity=args.get("max_debt_equity"),
            min_roe=args.get("min_roe"),
            min_revenue_growth=args.get("min_revenue_growth"),
        )
        return _text(result)

    # ─────────────────────────────────────────────────────────────────────────

    elif name == "get_gamma_exposure":
        ticker = args["ticker"]
        f = _fetcher(ticker)
        price = f.get_current_price() or 0.0
        result = _inst_risk.compute_gex(ticker=ticker, current_price=price)
        return _text(result)

    # ─────────────────────────────────────────────────────────────────────────

    elif name == "get_congressional_trades":
        ticker = args["ticker"]
        result = _alt_data.compute_congressional_alpha(ticker=ticker)
        return _text(result)

    # ─────────────────────────────────────────────────────────────────────────

    elif name == "get_insider_flow":
        ticker = args["ticker"]
        result = _alt_data.compute_insider_flow(ticker=ticker)
        return _text(result)

    # ─────────────────────────────────────────────────────────────────────────

    elif name == "get_earnings_quality":
        ticker = args["ticker"]
        f = _fetcher(ticker)
        result = EarningsQualityAnalyzer().analyze(
            info=f.info,
            income_stmt=f.get_income_statement(),
            balance_sheet=f.get_balance_sheet(),
            cashflow=f.get_cashflow_statement(),
            earnings_history=f.get_earnings_history(),
        )
        return _text(result)

    # ─────────────────────────────────────────────────────────────────────────

    elif name == "get_capital_efficiency":
        ticker = args["ticker"]
        f = _fetcher(ticker)
        result = CapitalEfficiencyAnalyzer().analyze(
            info=f.info,
            income_stmt=f.get_income_statement(),
            balance_sheet=f.get_balance_sheet(),
            cashflow=f.get_cashflow_statement(),
        )
        return _text(result)

    # ─────────────────────────────────────────────────────────────────────────

    elif name == "get_technical_enhanced":
        ticker = args["ticker"]
        period = args.get("period", "1y")
        f = _fetcher(ticker)
        hist = f.get_history(period=period)
        if hist is None or len(hist) < 20:
            return _err(f"Insufficient history for {ticker}.")
        result = TechnicalEnhancedAnalyzer().analyze(
            ticker=ticker, history=hist, info=f.info
        )
        return _text(result)

    # ─────────────────────────────────────────────────────────────────────────

    elif name == "get_credit_conditions":
        result = CreditConditionsAnalyzer().analyze()
        return _text(result)

    # ─────────────────────────────────────────────────────────────────────────

    elif name == "get_portfolio_risk":
        ticker = args["ticker"]
        f = _fetcher(ticker)
        hist = f.get_history(period="1y")
        if hist is None or len(hist) < 20:
            return _err(f"Insufficient history for {ticker}.")
        result = PortfolioRiskAnalyzer().analyze(
            ticker=ticker,
            history=hist,
            info=f.info,
            account_size=float(args.get("account_size", 100_000)),
            target_beta=float(args.get("target_beta", 1.0)),
        )
        return _text(result)

    # ─────────────────────────────────────────────────────────────────────────

    elif name == "get_sector_metrics":
        ticker = args["ticker"]
        f = _fetcher(ticker)
        result = SectorMetricsAnalyzer().analyze(
            ticker=ticker,
            info=f.info,
            income_stmt=f.get_income_statement(),
            balance_sheet=f.get_balance_sheet(),
            cashflow=f.get_cashflow_statement(),
        )
        return _text(result)

    # ─────────────────────────────────────────────────────────────────────────

    elif name == "synthesize_investment_thesis":
        ticker = args["ticker"]
        account_size = float(args.get("account_size", 100_000))
        f = _fetcher(ticker)

        # Gather all data
        fund_data = f.get_fundamental_data()
        sector = f.get_sector() or ""
        hist = f.get_history(period="1y")
        if hist is None or len(hist) < 20:
            return _err(f"Insufficient history for {ticker}.")
        price = f.get_current_price() or fund_data.get("price") or 0.0

        # Run all engines
        fund_result = FundamentalAnalyzer().analyze(fund_data, sector)
        tech_result = TechnicalAnalyzer().analyze(hist)
        recommendation = HolisticRecommendation(fundamental=fund_result, technical=tech_result)
        governance = GovernanceAnalyzer().analyze(
            info=f.info,
            company_officers=f.get_company_officers(),
            earnings_history=f.get_earnings_history(),
            governance_scores=f.get_governance_scores(),
            insider_purchases=f.get_insider_purchases(),
        )
        macro_result = MacroAnalyzer().analyze()
        cycle_phase = macro_result.cycle.phase if macro_result.cycle else "Unknown"
        rates_rising = getattr(macro_result, "rates_rising", False)
        sector_result = SectorAnalyzer().analyze(cycle_phase=cycle_phase, rates_rising=rates_rising)
        news_result = NewsSentimentAnalyzer().analyze(
            ticker=ticker, company_name=f.get_company_name() or ticker
        )
        risk_profile = RiskManager().analyze(df=hist, entry_price=price, account_size=account_size)
        sr_levels = tech_result.support_resistance or []

        verdict = ReasoningEngine().synthesize(
            company_info=f.info,
            fund_data=fund_data,
            fund_result=fund_result,
            tech_result=tech_result,
            recommendation=recommendation,
            governance=governance,
            macro_result=macro_result,
            sector_result=sector_result,
            news_result=news_result,
            risk_profile=risk_profile,
            sr_levels=sr_levels,
        )
        return _text(verdict)

    # ─────────────────────────────────────────────────────────────────────────

    elif name == "analyze_stock":
        ticker = args["ticker"]
        account_size = float(args.get("account_size", 100_000))
        risk_pct = float(args.get("risk_pct", 1.0))
        f = _fetcher(ticker)

        fund_data = f.get_fundamental_data()
        sector = f.get_sector() or ""
        hist = f.get_history(period="1y")
        if hist is None or len(hist) < 20:
            return _err(f"Insufficient price history for {ticker}.")
        price = f.get_current_price() or fund_data.get("price") or 0.0

        fund_result = FundamentalAnalyzer().analyze(fund_data, sector)
        tech_result = TechnicalAnalyzer().analyze(hist)
        recommendation = HolisticRecommendation(fundamental=fund_result, technical=tech_result)

        governance = GovernanceAnalyzer().analyze(
            info=f.info,
            company_officers=f.get_company_officers(),
            earnings_history=f.get_earnings_history(),
            governance_scores=f.get_governance_scores(),
            insider_purchases=f.get_insider_purchases(),
        )
        macro_result = MacroAnalyzer().analyze()
        cycle_phase = macro_result.cycle.phase if macro_result.cycle else "Unknown"
        rates_rising = getattr(macro_result, "rates_rising", False)
        sector_result = SectorAnalyzer().analyze(cycle_phase=cycle_phase, rates_rising=rates_rising)
        news_result = NewsSentimentAnalyzer().analyze(
            ticker=ticker, company_name=f.get_company_name() or ticker
        )
        risk_profile = RiskManager().analyze(
            df=hist, entry_price=price, account_size=account_size, risk_pct=risk_pct
        )
        quality = QualityScorer().compute(
            info=f.info,
            income_stmt=f.get_income_statement(),
            balance_sheet=f.get_balance_sheet(),
            cashflow=f.get_cashflow_statement(),
        )
        dcf = DCFValuator().compute(
            info=f.info,
            income_stmt=f.get_income_statement(),
            balance_sheet=f.get_balance_sheet(),
            cashflow=f.get_cashflow_statement(),
        )
        peers = PeerComparator().compare(
            subject_ticker=ticker, subject_info=f.info, sector=sector
        )
        options = OptionsAnalyzer().analyze(ticker=ticker, history=hist)
        inst_risk = _inst_risk.analyze(
            ticker=ticker, price=price, info=f.info, income_stmt=f.get_income_statement()
        )
        alt_data = _alt_data.cached_analyze(ticker=ticker, price=price)
        sr_levels = tech_result.support_resistance or []

        verdict = ReasoningEngine().synthesize(
            company_info=f.info,
            fund_data=fund_data,
            fund_result=fund_result,
            tech_result=tech_result,
            recommendation=recommendation,
            governance=governance,
            macro_result=macro_result,
            sector_result=sector_result,
            news_result=news_result,
            risk_profile=risk_profile,
            sr_levels=sr_levels,
        )

        report = {
            "ticker": ticker.upper(),
            "company": f.get_company_name(),
            "sector": sector,
            "current_price": price,
            "recommendation": dataclasses.asdict(recommendation),
            "verdict": dataclasses.asdict(verdict),
            "fundamental": dataclasses.asdict(fund_result),
            "technical": dataclasses.asdict(tech_result),
            "governance": dataclasses.asdict(governance),
            "macro": dataclasses.asdict(macro_result),
            "sector_analysis": dataclasses.asdict(sector_result),
            "news_sentiment": dataclasses.asdict(news_result),
            "risk_profile": dataclasses.asdict(risk_profile),
            "quality_scores": dataclasses.asdict(quality),
            "dcf_valuation": dataclasses.asdict(dcf),
            "peer_comparisons": dataclasses.asdict(peers),
            "options": dataclasses.asdict(options),
            "institutional_risk": dataclasses.asdict(inst_risk),
            "alternative_data": dataclasses.asdict(alt_data),
            "support_resistance": [dataclasses.asdict(l) for l in sr_levels],
        }
        return _text(report)

    # ─────────────────────────────────────────────────────────────────────────

    else:
        return _err(f"Unknown tool: {name}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

async def _run_stdio():
    async with stdio_server() as (r, w):
        await server.run(r, w, server.create_initialization_options())


async def _run_sse(port: int):
    from mcp.server.sse import SseServerTransport
    from starlette.applications import Starlette
    from starlette.routing import Mount, Route
    import uvicorn

    sse = SseServerTransport("/messages/")

    async def handle_sse(request):
        async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
            await server.run(streams[0], streams[1], server.create_initialization_options())

    app = Starlette(routes=[
        Route("/sse", endpoint=handle_sse),
        Mount("/messages/", app=sse.handle_post_message),
    ])

    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="warning")
    s = uvicorn.Server(config)
    print(f"Financial Analyst MCP server running on http://0.0.0.0:{port}/sse")
    await s.serve()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Financial Analyst MCP Server")
    parser.add_argument("--transport", choices=["stdio", "sse"], default="stdio",
                        help="Transport protocol (default: stdio)")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port for SSE transport (default: 8000)")
    a = parser.parse_args()

    if a.transport == "sse":
        asyncio.run(_run_sse(a.port))
    else:
        asyncio.run(_run_stdio())
