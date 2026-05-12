"""Options market signals.

Uses yfinance options chains to compute:
- Implied Volatility (IV) vs 30-day realized volatility
- IV rank (current IV vs 52-week IV range)
- Put/Call open interest ratio
- Put/Call volume ratio
- Max pain price (where total option holder loss is minimized)
- Nearest expiry chain summary
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


@dataclass
class OptionsResult:
    ticker: str
    # IV metrics
    avg_iv: Optional[float] = None           # weighted avg IV of ATM options
    realized_vol_30d: Optional[float] = None # 30-day historical vol
    iv_premium: Optional[float] = None       # avg_iv - realized_vol (IV - RV spread)
    iv_rank: Optional[float] = None          # 0-100: where current IV sits vs 52w range
    iv_percentile: Optional[float] = None    # % of days past year IV was below current
    # Put/Call
    put_call_oi_ratio: Optional[float] = None   # total put OI / total call OI
    put_call_vol_ratio: Optional[float] = None  # total put vol / total call vol
    # Max pain
    max_pain_price: Optional[float] = None
    # Nearest expiry info
    nearest_expiry: str = ""
    total_call_oi: int = 0
    total_put_oi: int = 0
    total_call_volume: int = 0
    total_put_volume: int = 0
    # Chain (top strikes near ATM)
    chain_summary: List[dict] = field(default_factory=list)
    signal: str = "Neutral"   # "Bullish", "Bearish", "Neutral", "High IV", "Low IV"
    summary: str = ""
    error: str = ""


class OptionsAnalyzer:

    def analyze(self, ticker: str, history: pd.DataFrame) -> OptionsResult:
        result = OptionsResult(ticker=ticker.upper())
        try:
            self._run(result, ticker, history)
        except Exception as e:
            result.error = str(e)
            logger.warning("Options analysis failed for %s: %s", ticker, e)
        return result

    def _run(self, result: OptionsResult, ticker: str, history: pd.DataFrame):
        stock = yf.Ticker(ticker)

        # Realized vol from price history
        if history is not None and len(history) >= 30:
            rets = history["Close"].pct_change().dropna()
            result.realized_vol_30d = float(rets.tail(30).std() * math.sqrt(252))

        # Options chain
        expirations = stock.options
        if not expirations:
            result.error = "No options data available"
            return

        result.nearest_expiry = expirations[0]
        chain = stock.option_chain(expirations[0])
        calls = chain.calls
        puts = chain.puts

        if calls.empty and puts.empty:
            result.error = "Empty options chain"
            return

        current_price = None
        if history is not None and len(history) > 0:
            current_price = float(history["Close"].iloc[-1])

        # Put/Call OI and Volume
        result.total_call_oi = int(calls["openInterest"].sum()) if "openInterest" in calls else 0
        result.total_put_oi = int(puts["openInterest"].sum()) if "openInterest" in puts else 0
        result.total_call_volume = int(calls["volume"].fillna(0).sum()) if "volume" in calls else 0
        result.total_put_volume = int(puts["volume"].fillna(0).sum()) if "volume" in puts else 0

        if result.total_call_oi > 0:
            result.put_call_oi_ratio = round(result.total_put_oi / result.total_call_oi, 3)
        if result.total_call_volume > 0:
            result.put_call_vol_ratio = round(result.total_put_volume / result.total_call_volume, 3)

        # ATM IV (weighted average of nearest strikes)
        if current_price:
            result.avg_iv = self._compute_atm_iv(calls, puts, current_price)
            result.max_pain_price = self._compute_max_pain(calls, puts)
            result.chain_summary = self._build_chain_summary(calls, puts, current_price)

        # IV rank: compare to prior expiries (use up to 4 expirations for speed)
        if result.avg_iv:
            iv_history = [result.avg_iv]
            for exp in expirations[1:5]:
                try:
                    ch = stock.option_chain(exp)
                    if current_price:
                        iv = self._compute_atm_iv(ch.calls, ch.puts, current_price)
                        if iv:
                            iv_history.append(iv)
                except Exception:
                    pass
            if len(iv_history) >= 2:
                iv_min, iv_max = min(iv_history), max(iv_history)
                if iv_max > iv_min:
                    result.iv_rank = round((result.avg_iv - iv_min) / (iv_max - iv_min) * 100, 1)
                result.iv_percentile = round(
                    sum(1 for v in iv_history if v <= result.avg_iv) / len(iv_history) * 100, 1
                )

        if result.avg_iv and result.realized_vol_30d:
            result.iv_premium = round(result.avg_iv - result.realized_vol_30d, 4)

        result.signal = self._derive_signal(result)
        result.summary = self._build_summary(result)

    def _compute_atm_iv(self, calls: pd.DataFrame, puts: pd.DataFrame, price: float) -> Optional[float]:
        all_options = pd.concat([calls, puts], ignore_index=True)
        if "impliedVolatility" not in all_options.columns or "strike" not in all_options.columns:
            return None
        all_options = all_options.dropna(subset=["impliedVolatility", "strike"])
        all_options = all_options[all_options["impliedVolatility"] > 0]
        if all_options.empty:
            return None
        all_options["dist"] = abs(all_options["strike"] - price)
        near = all_options.nsmallest(6, "dist")
        weights = 1 / (near["dist"] + 0.01)
        return round(float((near["impliedVolatility"] * weights).sum() / weights.sum()), 4)

    def _compute_max_pain(self, calls: pd.DataFrame, puts: pd.DataFrame) -> Optional[float]:
        """Max pain = strike where total dollar loss for option buyers is maximized."""
        try:
            strikes = sorted(set(calls["strike"].tolist() + puts["strike"].tolist()))
            if not strikes:
                return None
            pain = {}
            for s in strikes:
                call_pain = sum(
                    max(0, s - row["strike"]) * row.get("openInterest", 0)
                    for _, row in calls.iterrows()
                    if row.get("openInterest") and row["openInterest"] > 0
                )
                put_pain = sum(
                    max(0, row["strike"] - s) * row.get("openInterest", 0)
                    for _, row in puts.iterrows()
                    if row.get("openInterest") and row["openInterest"] > 0
                )
                pain[s] = call_pain + put_pain
            return min(pain, key=pain.get)
        except Exception:
            return None

    def _build_chain_summary(self, calls: pd.DataFrame, puts: pd.DataFrame, price: float) -> List[dict]:
        summary = []
        try:
            all_strikes = sorted(set(calls["strike"].tolist() + puts["strike"].tolist()))
            near = sorted(all_strikes, key=lambda s: abs(s - price))[:10]
            near_sorted = sorted(near)
            for strike in near_sorted:
                call_row = calls[calls["strike"] == strike]
                put_row = puts[puts["strike"] == strike]
                summary.append({
                    "strike": strike,
                    "call_iv": round(float(call_row["impliedVolatility"].iloc[0]), 4) if not call_row.empty and "impliedVolatility" in call_row else None,
                    "put_iv": round(float(put_row["impliedVolatility"].iloc[0]), 4) if not put_row.empty and "impliedVolatility" in put_row else None,
                    "call_oi": int(call_row["openInterest"].iloc[0]) if not call_row.empty and "openInterest" in call_row else 0,
                    "put_oi": int(put_row["openInterest"].iloc[0]) if not put_row.empty and "openInterest" in put_row else 0,
                    "call_vol": int(call_row["volume"].fillna(0).iloc[0]) if not call_row.empty and "volume" in call_row else 0,
                    "put_vol": int(put_row["volume"].fillna(0).iloc[0]) if not put_row.empty and "volume" in put_row else 0,
                    "itm": strike < price,
                })
        except Exception as e:
            logger.debug("Chain summary failed: %s", e)
        return summary

    def _derive_signal(self, r: OptionsResult) -> str:
        if r.iv_rank and r.iv_rank > 70:
            return "High IV (Sell Premium)"
        if r.iv_rank and r.iv_rank < 30:
            return "Low IV (Buy Options)"
        if r.put_call_oi_ratio:
            if r.put_call_oi_ratio > 1.3:
                return "Bearish (Heavy Put OI)"
            if r.put_call_oi_ratio < 0.7:
                return "Bullish (Heavy Call OI)"
        return "Neutral"

    def _build_summary(self, r: OptionsResult) -> str:
        parts = []
        if r.avg_iv:
            parts.append(f"ATM IV: {r.avg_iv:.1%}")
        if r.realized_vol_30d:
            parts.append(f"30d RV: {r.realized_vol_30d:.1%}")
        if r.iv_premium is not None:
            sign = "rich" if r.iv_premium > 0 else "cheap"
            parts.append(f"IV {sign} by {abs(r.iv_premium):.1%}")
        if r.put_call_oi_ratio:
            parts.append(f"P/C OI: {r.put_call_oi_ratio:.2f}")
        if r.max_pain_price:
            parts.append(f"Max Pain: ${r.max_pain_price:.2f}")
        return " · ".join(parts)
