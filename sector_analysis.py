"""Sector analysis — performance comparison, rotation signals, and cycle-based scoring.

Tracks all 11 GICS sectors via SPDR Select Sector ETFs and scores them
relative to the current business cycle phase.
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
# S&P 500 sector ETFs
# ---------------------------------------------------------------------------
SECTOR_ETFS = {
    "Technology":           "XLK",
    "Healthcare":           "XLV",
    "Financials":           "XLF",
    "Consumer Discretionary": "XLY",
    "Consumer Staples":     "XLP",
    "Industrials":          "XLI",
    "Energy":               "XLE",
    "Utilities":            "XLU",
    "Materials":            "XLB",
    "Real Estate":          "XLRE",
    "Communication Services": "XLC",
}

# Which cycle phase favors which sectors (weight multiplier)
CYCLE_SECTOR_WEIGHTS = {
    "Expansion": {
        "Technology": 1.5, "Consumer Discretionary": 1.4, "Industrials": 1.3,
        "Materials": 1.2, "Financials": 1.1, "Communication Services": 1.1,
        "Energy": 1.0, "Healthcare": 0.8, "Consumer Staples": 0.7,
        "Utilities": 0.6, "Real Estate": 0.9,
    },
    "Peak": {
        "Energy": 1.4, "Healthcare": 1.3, "Materials": 1.2,
        "Consumer Staples": 1.1, "Utilities": 1.0, "Financials": 1.0,
        "Industrials": 0.9, "Technology": 0.8, "Consumer Discretionary": 0.7,
        "Real Estate": 0.7, "Communication Services": 0.8,
    },
    "Contraction": {
        "Healthcare": 1.5, "Utilities": 1.4, "Consumer Staples": 1.4,
        "Communication Services": 1.0, "Real Estate": 0.9, "Financials": 0.7,
        "Technology": 0.8, "Industrials": 0.6, "Materials": 0.6,
        "Consumer Discretionary": 0.5, "Energy": 0.7,
    },
    "Recovery": {
        "Technology": 1.5, "Consumer Discretionary": 1.4, "Financials": 1.3,
        "Industrials": 1.3, "Materials": 1.2, "Real Estate": 1.1,
        "Communication Services": 1.1, "Energy": 1.0,
        "Healthcare": 0.8, "Consumer Staples": 0.7, "Utilities": 0.6,
    },
}

# Rate sensitivity
RATE_SENSITIVE = {
    "Financials": "positive",       # wider NIM
    "Real Estate": "negative",      # higher financing costs
    "Utilities": "negative",        # higher debt costs, less attractive yield
    "Technology": "negative",       # discounts future earnings more
    "Consumer Discretionary": "negative",
}


@dataclass
class SectorPerformance:
    name: str
    etf: str
    current_price: float
    change_1w: Optional[float]
    change_1m: Optional[float]
    change_3m: Optional[float]
    change_6m: Optional[float]
    relative_strength: float  # vs S&P 500 over 3 months
    momentum_score: float  # -1 to +1
    cycle_alignment: float  # how well this sector fits the current cycle
    rate_sensitivity: str  # "positive", "negative", "neutral"


@dataclass
class SectorResult:
    sectors: List[SectorPerformance] = field(default_factory=list)
    sp500_performance: Dict = field(default_factory=dict)
    rotation_recommendation: str = ""
    stock_sector: Optional[str] = None
    stock_sector_rank: Optional[int] = None
    stock_sector_score: Optional[float] = None
    cycle_phase: str = "Unknown"


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------
class SectorAnalyzer:

    def analyze(self, cycle_phase: str = "Expansion", rates_rising: bool = False) -> SectorResult:
        result = SectorResult(cycle_phase=cycle_phase)

        # Fetch S&P 500 baseline
        sp_df = self._fetch("^GSPC")
        sp_perf = self._calc_performance(sp_df) if sp_df is not None else {}
        result.sp500_performance = sp_perf

        # Fetch each sector
        for sector_name, etf in SECTOR_ETFS.items():
            df = self._fetch(etf)
            if df is None or len(df) < 5:
                continue
            time.sleep(0.3)

            perf = self._calc_performance(df)
            price = float(df["Close"].iloc[-1])

            # Relative strength vs S&P
            rel_3m = (perf.get("3m", 0) or 0) - (sp_perf.get("3m", 0) or 0)

            # Momentum score from multi-timeframe performance
            scores = []
            for key, weight in [("1w", 0.15), ("1m", 0.25), ("3m", 0.35), ("6m", 0.25)]:
                v = perf.get(key)
                if v is not None:
                    if v > 0.10:
                        scores.append(1.0 * weight)
                    elif v > 0.03:
                        scores.append(0.5 * weight)
                    elif v > -0.03:
                        scores.append(0.0)
                    elif v > -0.10:
                        scores.append(-0.5 * weight)
                    else:
                        scores.append(-1.0 * weight)
            momentum = sum(scores) / sum([0.15, 0.25, 0.35, 0.25]) if scores else 0

            # Cycle alignment
            cycle_weights = CYCLE_SECTOR_WEIGHTS.get(cycle_phase, {})
            cycle_align = cycle_weights.get(sector_name, 1.0)

            # Rate sensitivity adjustment
            rate_sens = RATE_SENSITIVE.get(sector_name, "neutral")
            if rates_rising and rate_sens == "negative":
                cycle_align *= 0.85
            elif rates_rising and rate_sens == "positive":
                cycle_align *= 1.15

            result.sectors.append(SectorPerformance(
                name=sector_name,
                etf=etf,
                current_price=price,
                change_1w=perf.get("1w"),
                change_1m=perf.get("1m"),
                change_3m=perf.get("3m"),
                change_6m=perf.get("6m"),
                relative_strength=round(rel_3m * 100, 2),
                momentum_score=round(momentum, 3),
                cycle_alignment=round(cycle_align, 2),
                rate_sensitivity=rate_sens,
            ))

        # Sort by combined score (momentum + cycle alignment)
        for s in result.sectors:
            s._combined = s.momentum_score * 0.5 + (s.cycle_alignment - 1.0) * 0.5
        result.sectors.sort(key=lambda s: s._combined, reverse=True)

        # Rotation recommendation
        if result.sectors:
            top = result.sectors[:3]
            bottom = result.sectors[-3:]
            result.rotation_recommendation = (
                f"**Favor:** {', '.join(s.name for s in top)} — "
                f"aligned with {cycle_phase} cycle and showing positive momentum.\n\n"
                f"**Underweight:** {', '.join(s.name for s in bottom)} — "
                f"misaligned with current cycle or showing weakness."
            )

        return result

    def score_stock_sector(self, result: SectorResult, stock_sector: str) -> SectorResult:
        """Score how well the stock's sector fits the current macro environment."""
        result.stock_sector = stock_sector
        for i, s in enumerate(result.sectors):
            if s.name == stock_sector:
                result.stock_sector_rank = i + 1
                result.stock_sector_score = round(
                    s.momentum_score * 0.4 + (s.cycle_alignment - 1.0) * 0.6, 3
                )
                break
        return result

    def _fetch(self, ticker: str, period: str = "6mo") -> Optional[pd.DataFrame]:
        try:
            df = yf.Ticker(ticker).history(period=period)
            if df is not None and len(df) > 0:
                return df
        except Exception as e:
            logger.warning("Failed to fetch %s: %s", ticker, e)
        return None

    def _calc_performance(self, df: pd.DataFrame) -> Dict[str, Optional[float]]:
        close = df["Close"]
        current = float(close.iloc[-1])
        result = {}
        for label, days in [("1w", 5), ("1m", 21), ("3m", 63), ("6m", 126)]:
            if len(close) > days:
                past = float(close.iloc[-days - 1])
                result[label] = round((current - past) / past, 4) if past != 0 else None
            else:
                result[label] = None
        return result
