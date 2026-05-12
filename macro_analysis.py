"""Macroeconomic analysis — business cycle detection, yield curve, and market regime signals.

Uses market-traded instruments as real-time proxies for macro indicators:
- Treasury yields (2Y/10Y/30Y) → interest rates & yield curve
- TIPS breakevens → inflation expectations
- VIX → market fear / risk sentiment
- Dollar index, gold, oil → macro regime signals
- Sector ETF breadth → expansion vs contraction
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict

import pandas as pd
import numpy as np
import yfinance as yf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Market-traded macro proxies
# ---------------------------------------------------------------------------
MACRO_TICKERS = {
    "us_10y": "^TNX",       # 10-Year Treasury Yield
    "us_2y": "2YY=F",       # 2-Year Treasury Yield
    "us_30y": "^TYX",       # 30-Year Treasury Yield
    "us_3m": "^IRX",        # 13-Week T-Bill
    "vix": "^VIX",          # CBOE Volatility Index
    "dxy": "DX-Y.NYB",      # US Dollar Index
    "gold": "GC=F",         # Gold Futures
    "oil": "CL=F",          # Crude Oil Futures
    "sp500": "^GSPC",       # S&P 500
    "russell2k": "^RUT",    # Russell 2000 (small caps)
}


@dataclass
class MacroIndicator:
    name: str
    value: Optional[float]
    change_1m: Optional[float]  # % change over 1 month
    change_3m: Optional[float]
    signal: str  # "bullish", "bearish", "neutral"
    interpretation: str
    category: str  # "Interest Rates", "Inflation", "Growth", "Sentiment"


@dataclass
class BusinessCycle:
    phase: str  # "Expansion", "Peak", "Contraction", "Recovery"
    confidence: str  # "HIGH", "MEDIUM", "LOW"
    description: str
    favored_sectors: List[str] = field(default_factory=list)
    avoid_sectors: List[str] = field(default_factory=list)
    risk_posture: str = "Neutral"  # "Risk-On", "Risk-Off", "Neutral"


@dataclass
class MacroResult:
    indicators: List[MacroIndicator] = field(default_factory=list)
    cycle: Optional[BusinessCycle] = None
    yield_curve: Dict = field(default_factory=dict)
    overall_signal: str = "NEUTRAL"
    overall_score: float = 0.0  # -1 to +1
    summary: str = ""


# ---------------------------------------------------------------------------
# Fetcher
# ---------------------------------------------------------------------------
def _safe_fetch(ticker: str, period: str = "6mo") -> Optional[pd.DataFrame]:
    """Fetch with rate-limit awareness."""
    try:
        df = yf.Ticker(ticker).history(period=period)
        if df is not None and len(df) > 0:
            return df
    except Exception as e:
        logger.warning("Failed to fetch %s: %s", ticker, e)
    return None


def _pct_change(series: pd.Series, days: int) -> Optional[float]:
    if series is None or len(series) < days + 1:
        return None
    current = float(series.iloc[-1])
    past = float(series.iloc[-days - 1])
    if past == 0:
        return None
    return (current - past) / abs(past)


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------
class MacroAnalyzer:

    def analyze(self) -> MacroResult:
        result = MacroResult()

        data = {}
        for key, ticker in MACRO_TICKERS.items():
            df = _safe_fetch(ticker, period="6mo")
            if df is not None and len(df) > 0:
                data[key] = df
            time.sleep(0.5)

        result.indicators = self._build_indicators(data)
        result.yield_curve = self._analyze_yield_curve(data)
        result.cycle = self._detect_business_cycle(data, result.indicators, result.yield_curve)

        self._compute_overall(result)
        return result

    # ----- Indicators -----
    def _build_indicators(self, data: Dict[str, pd.DataFrame]) -> List[MacroIndicator]:
        indicators = []

        # 10-Year Treasury Yield
        if "us_10y" in data:
            close = data["us_10y"]["Close"]
            val = float(close.iloc[-1])
            chg1m = _pct_change(close, 21)
            chg3m = _pct_change(close, 63)

            if val > 5.0:
                sig, interp = "bearish", "Very high rates — restrictive for growth and valuations"
            elif val > 4.0:
                sig, interp = "bearish", "Elevated rates — headwind for rate-sensitive sectors"
            elif val > 3.0:
                sig, interp = "neutral", "Moderate rates — manageable for most sectors"
            elif val > 2.0:
                sig, interp = "bullish", "Low rates — supportive of growth and borrowing"
            else:
                sig, interp = "bullish", "Very low rates — highly stimulative environment"

            indicators.append(MacroIndicator(
                "10-Year Treasury Yield", val, chg1m, chg3m, sig, interp, "Interest Rates"
            ))

        # 2-Year Yield
        if "us_2y" in data:
            close = data["us_2y"]["Close"]
            val = float(close.iloc[-1])
            chg1m = _pct_change(close, 21)
            sig = "neutral"
            interp = f"2Y yield at {val:.2f}% — reflects near-term rate expectations"
            indicators.append(MacroIndicator(
                "2-Year Treasury Yield", val, chg1m, _pct_change(close, 63), sig, interp, "Interest Rates"
            ))

        # Yield curve spread (10Y - 2Y)
        if "us_10y" in data and "us_2y" in data:
            y10 = float(data["us_10y"]["Close"].iloc[-1])
            y2 = float(data["us_2y"]["Close"].iloc[-1])
            spread = y10 - y2

            if spread < -0.5:
                sig, interp = "bearish", f"Deeply inverted ({spread:+.2f}%) — strong recession signal"
            elif spread < 0:
                sig, interp = "bearish", f"Inverted ({spread:+.2f}%) — recession warning"
            elif spread < 0.5:
                sig, interp = "neutral", f"Flat curve ({spread:+.2f}%) — late cycle or transition"
            elif spread < 1.5:
                sig, interp = "bullish", f"Normal slope ({spread:+.2f}%) — healthy growth expected"
            else:
                sig, interp = "bullish", f"Steep curve ({spread:+.2f}%) — strong recovery/expansion signal"

            indicators.append(MacroIndicator(
                "Yield Curve (10Y-2Y)", spread, None, None, sig, interp, "Interest Rates"
            ))

        # VIX (Fear Index)
        if "vix" in data:
            close = data["vix"]["Close"]
            val = float(close.iloc[-1])
            chg1m = _pct_change(close, 21)

            if val > 30:
                sig, interp = "bearish", "Extreme fear — high market stress, risk-off environment"
            elif val > 20:
                sig, interp = "neutral", "Elevated volatility — caution warranted"
            elif val > 15:
                sig, interp = "bullish", "Normal volatility — healthy risk appetite"
            else:
                sig, interp = "bullish", "Low fear — complacency (strong but watch for reversal)"

            indicators.append(MacroIndicator(
                "VIX (Fear Index)", val, chg1m, _pct_change(close, 63), sig, interp, "Sentiment"
            ))

        # US Dollar Index
        if "dxy" in data:
            close = data["dxy"]["Close"]
            val = float(close.iloc[-1])
            chg1m = _pct_change(close, 21)
            chg3m = _pct_change(close, 63)

            if chg1m is not None and chg1m > 0.02:
                sig, interp = "bearish", "Strengthening dollar — headwind for multinationals and EM"
            elif chg1m is not None and chg1m < -0.02:
                sig, interp = "bullish", "Weakening dollar — tailwind for exporters and commodities"
            else:
                sig, interp = "neutral", "Stable dollar — neutral for most sectors"

            indicators.append(MacroIndicator(
                "US Dollar Index", val, chg1m, chg3m, sig, interp, "Sentiment"
            ))

        # Gold (inflation hedge / safe haven)
        if "gold" in data:
            close = data["gold"]["Close"]
            val = float(close.iloc[-1])
            chg1m = _pct_change(close, 21)
            chg3m = _pct_change(close, 63)

            if chg3m is not None and chg3m > 0.10:
                sig, interp = "bearish", "Gold surging — markets pricing in inflation/uncertainty"
            elif chg1m is not None and chg1m > 0.03:
                sig, interp = "neutral", "Gold rising — mild inflation/safe-haven demand"
            elif chg1m is not None and chg1m < -0.03:
                sig, interp = "bullish", "Gold falling — risk-on, confidence in growth"
            else:
                sig, interp = "neutral", "Gold stable"

            indicators.append(MacroIndicator(
                "Gold", val, chg1m, chg3m, sig, interp, "Inflation"
            ))

        # Crude Oil
        if "oil" in data:
            close = data["oil"]["Close"]
            val = float(close.iloc[-1])
            chg1m = _pct_change(close, 21)
            chg3m = _pct_change(close, 63)

            if val > 90:
                sig, interp = "bearish", "High oil prices — inflationary, squeezes consumers"
            elif val > 70:
                sig, interp = "neutral", "Moderate oil — balanced demand/supply"
            elif val > 50:
                sig, interp = "bullish", "Low oil — eases inflation, supports consumer spending"
            else:
                sig, interp = "bearish", "Very low oil — may signal demand collapse/recession"

            indicators.append(MacroIndicator(
                "Crude Oil (WTI)", val, chg1m, chg3m, sig, interp, "Inflation"
            ))

        # S&P 500 trend (growth proxy)
        if "sp500" in data:
            close = data["sp500"]["Close"]
            val = float(close.iloc[-1])
            chg1m = _pct_change(close, 21)
            chg3m = _pct_change(close, 63)

            sma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else None
            sma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None

            if sma50 and sma200 and val > sma50 > sma200:
                sig, interp = "bullish", "S&P 500 above rising MAs — strong uptrend, expansion mode"
            elif sma50 and val > sma50:
                sig, interp = "bullish", "S&P 500 above 50-day MA — positive short-term momentum"
            elif sma200 and val < sma200:
                sig, interp = "bearish", "S&P 500 below 200-day MA — bearish regime"
            else:
                sig, interp = "neutral", "S&P 500 in mixed trend"

            indicators.append(MacroIndicator(
                "S&P 500", val, chg1m, chg3m, sig, interp, "Growth"
            ))

        # Russell 2000 vs S&P (risk appetite)
        if "russell2k" in data and "sp500" in data:
            rut_chg = _pct_change(data["russell2k"]["Close"], 63)
            sp_chg = _pct_change(data["sp500"]["Close"], 63)
            if rut_chg is not None and sp_chg is not None:
                spread = rut_chg - sp_chg
                if spread > 0.03:
                    sig, interp = "bullish", "Small caps outperforming — risk-on, broad economic confidence"
                elif spread < -0.05:
                    sig, interp = "bearish", "Small caps lagging — risk-off, flight to quality"
                else:
                    sig, interp = "neutral", "Small/large cap spread neutral"

                indicators.append(MacroIndicator(
                    "Small Cap vs Large Cap", round(spread * 100, 2),
                    None, None, sig, interp, "Growth"
                ))

        return indicators

    # ----- Yield Curve -----
    def _analyze_yield_curve(self, data: Dict[str, pd.DataFrame]) -> Dict:
        curve = {}
        for key, label in [("us_3m", "3M"), ("us_2y", "2Y"), ("us_10y", "10Y"), ("us_30y", "30Y")]:
            if key in data:
                curve[label] = float(data[key]["Close"].iloc[-1])

        if not curve:
            return {"status": "unavailable", "points": {}}

        # Determine inversion
        y10 = curve.get("10Y")
        y2 = curve.get("2Y")
        y3m = curve.get("3M")

        status = "normal"
        if y10 and y2 and y10 < y2:
            status = "inverted"
        elif y10 and y3m and y10 < y3m:
            status = "inverted"
        elif y10 and y2 and (y10 - y2) < 0.25:
            status = "flat"

        return {"status": status, "points": curve}

    # ----- Business Cycle Detection -----
    def _detect_business_cycle(
        self,
        data: Dict[str, pd.DataFrame],
        indicators: List[MacroIndicator],
        yield_curve: Dict,
    ) -> BusinessCycle:

        signals = {"expansion": 0, "peak": 0, "contraction": 0, "recovery": 0}

        # Yield curve
        yc_status = yield_curve.get("status", "normal")
        if yc_status == "inverted":
            signals["peak"] += 2
            signals["contraction"] += 1
        elif yc_status == "flat":
            signals["peak"] += 1
        else:
            signals["expansion"] += 1
            signals["recovery"] += 1

        # Market trend
        for ind in indicators:
            if ind.name == "S&P 500":
                if ind.signal == "bullish":
                    signals["expansion"] += 2
                    signals["recovery"] += 1
                elif ind.signal == "bearish":
                    signals["contraction"] += 2

            elif ind.name == "VIX (Fear Index)":
                if ind.value and ind.value > 25:
                    signals["contraction"] += 2
                elif ind.value and ind.value < 15:
                    signals["expansion"] += 1

            elif ind.name == "Small Cap vs Large Cap":
                if ind.signal == "bullish":
                    signals["expansion"] += 1
                    signals["recovery"] += 1
                elif ind.signal == "bearish":
                    signals["contraction"] += 1
                    signals["peak"] += 1

            elif ind.name == "Crude Oil (WTI)":
                if ind.value and ind.value > 85:
                    signals["peak"] += 1
                elif ind.value and ind.value < 50:
                    signals["contraction"] += 1

            elif ind.name == "Gold":
                if ind.signal == "bearish":  # gold surging
                    signals["contraction"] += 1
                elif ind.signal == "bullish":  # gold falling
                    signals["expansion"] += 1

        # S&P 500 momentum
        if "sp500" in data:
            chg3m = _pct_change(data["sp500"]["Close"], 63)
            chg1m = _pct_change(data["sp500"]["Close"], 21)
            if chg3m is not None:
                if chg3m > 0.10:
                    signals["expansion"] += 2
                elif chg3m > 0.03:
                    signals["expansion"] += 1
                elif chg3m < -0.10:
                    signals["contraction"] += 2
                elif chg3m < -0.03:
                    signals["contraction"] += 1

            # Recovery detection: sharp bounce after being down
            if chg1m is not None and chg3m is not None:
                if chg1m > 0.05 and chg3m < 0:
                    signals["recovery"] += 2

        phase = max(signals, key=signals.get)
        total = sum(signals.values())
        top_score = signals[phase]
        confidence = "HIGH" if total > 0 and top_score / total > 0.5 else \
                     "MEDIUM" if total > 0 and top_score / total > 0.35 else "LOW"

        CYCLE_MAP = {
            "expansion": BusinessCycle(
                phase="Expansion",
                confidence=confidence,
                description="Economy is growing — GDP rising, employment strong, corporate earnings expanding. "
                            "Risk appetite is healthy and markets trend upward.",
                favored_sectors=["Technology", "Consumer Discretionary", "Industrials", "Materials", "Financials"],
                avoid_sectors=["Utilities", "Consumer Staples"],
                risk_posture="Risk-On",
            ),
            "peak": BusinessCycle(
                phase="Peak",
                confidence=confidence,
                description="Economy nearing a top — growth is decelerating, inflation may be sticky, "
                            "and the yield curve is flattening. Late-cycle dynamics.",
                favored_sectors=["Energy", "Healthcare", "Materials"],
                avoid_sectors=["Consumer Discretionary", "Technology", "Real Estate"],
                risk_posture="Neutral",
            ),
            "contraction": BusinessCycle(
                phase="Contraction",
                confidence=confidence,
                description="Economy is slowing or in recession — earnings declining, unemployment rising, "
                            "markets under pressure. Defensive positioning is key.",
                favored_sectors=["Healthcare", "Utilities", "Consumer Staples"],
                avoid_sectors=["Consumer Discretionary", "Industrials", "Materials", "Financials"],
                risk_posture="Risk-Off",
            ),
            "recovery": BusinessCycle(
                phase="Recovery",
                confidence=confidence,
                description="Economy is bottoming and beginning to recover — early signs of improving "
                            "data, accommodative policy, and beaten-down valuations rebounding.",
                favored_sectors=["Technology", "Consumer Discretionary", "Financials", "Industrials"],
                avoid_sectors=["Utilities", "Consumer Staples"],
                risk_posture="Risk-On",
            ),
        }

        return CYCLE_MAP[phase]

    # ----- Overall -----
    def _compute_overall(self, result: MacroResult):
        score_map = {"bullish": 1, "neutral": 0, "bearish": -1}
        valid = [i for i in result.indicators if i.signal in score_map]
        if not valid:
            result.overall_score = 0
            result.overall_signal = "NEUTRAL"
            result.summary = "Insufficient macro data."
            return

        result.overall_score = sum(score_map[i.signal] for i in valid) / len(valid)

        if result.overall_score >= 0.4:
            result.overall_signal = "BULLISH"
        elif result.overall_score >= 0.1:
            result.overall_signal = "SLIGHTLY BULLISH"
        elif result.overall_score >= -0.1:
            result.overall_signal = "NEUTRAL"
        elif result.overall_score >= -0.4:
            result.overall_signal = "SLIGHTLY BEARISH"
        else:
            result.overall_signal = "BEARISH"

        phase = result.cycle.phase if result.cycle else "Unknown"
        yc = result.yield_curve.get("status", "unknown")
        result.summary = (
            f"Business cycle: **{phase}** | Yield curve: **{yc}** | "
            f"Macro signal: **{result.overall_signal}** ({result.overall_score:+.2f})"
        )
