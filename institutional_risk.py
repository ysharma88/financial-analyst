"""Institutional-grade risk analytics inspired by Goldman Sachs CRB methodology.

Computes:
  - GEX (Gamma Exposure): dealer hedging obligations by strike, gamma flip level,
    support/resistance walls and trapdoors
  - Smart Money Flow: 13F institutional concentration, high-conviction holders,
    quarterly change signals
  - CDS Proxy: synthetic credit spread estimate from balance sheet ratios
  - CS01 / DV01 proxies: interest rate and credit sensitivity
  - Analyst momentum: upgrades vs downgrades trend
  - Position summary: unified cross-asset view
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("institutional_risk")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GEXLevel:
    strike: float
    net_gex: float        # $ notional — positive = dealer long gamma (support), negative = short gamma (trapdoor)
    open_interest: int
    kind: str             # "wall" (support) or "trapdoor" (resistance pressure)


@dataclass
class GEXResult:
    total_gex: float                    # aggregate dealer gamma in $ notional
    gamma_flip: Optional[float]         # price level where GEX flips sign
    above_flip: Optional[bool]          # True = stable regime, False = vol-amplifying regime
    top_walls: list[GEXLevel]           # positive GEX levels — dealers buy here on drop
    top_trapdoors: list[GEXLevel]       # negative GEX levels — dealers sell here on drop
    call_gex: float = 0.0
    put_gex: float = 0.0
    expiry_used: str = ""
    error: Optional[str] = None


@dataclass
class InstitutionalHolder:
    name: str
    pct_held: float
    shares: int
    value: float
    pct_change: float       # quarter-over-quarter change in position
    is_high_conviction: bool  # owns >2% of float


@dataclass
class SmartMoneyResult:
    total_institutional_pct: float
    holders: list[InstitutionalHolder]
    net_buying_pressure: float      # sum of pct_change across top holders — positive = net buying
    concentration_top5: float       # % of float held by top 5
    high_conviction_count: int      # holders with >2% of float
    analyst_momentum: float         # (upgrades - downgrades) / total over last 30d, range -1 to +1
    rec_trend: str                  # "Improving", "Stable", "Deteriorating"
    error: Optional[str] = None


@dataclass
class CreditProxyResult:
    cds_proxy_bps: Optional[float]    # synthetic credit spread in basis points
    cs01_proxy: Optional[float]       # $ change per 1bp credit spread widening (on total debt)
    dv01_proxy: Optional[float]       # $ change per 1bp interest rate move (on total debt)
    interest_coverage: Optional[float]
    debt_to_ebitda: Optional[float]
    credit_rating_proxy: str          # "Investment Grade", "Speculative", "Distress"
    error: Optional[str] = None


@dataclass
class InstitutionalRiskResult:
    ticker: str
    gex: Optional[GEXResult] = None
    smart_money: Optional[SmartMoneyResult] = None
    credit: Optional[CreditProxyResult] = None
    position_summary: str = ""
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# GEX Calculator
# ---------------------------------------------------------------------------

def _bs_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes gamma — same for calls and puts."""
    try:
        from scipy.stats import norm
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return 0.0
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        return norm.pdf(d1) / (S * sigma * math.sqrt(T))
    except Exception:
        return 0.0


def compute_gex(ticker: str, current_price: float) -> GEXResult:
    """Compute dealer Gamma Exposure across the nearest options expiry."""
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        expiries = t.options
        if not expiries:
            return GEXResult(0, None, None, [], [], error="No options data")

        # Use the nearest expiry with at least 1 day remaining
        today = date.today()
        expiry = None
        for exp in expiries:
            exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
            if (exp_date - today).days >= 1:
                expiry = exp
                break
        if expiry is None:
            expiry = expiries[0]

        chain = t.option_chain(expiry)
        calls = chain.calls
        puts = chain.puts

        exp_date = datetime.strptime(expiry, "%Y-%m-%d").date()
        T = max((exp_date - today).days / 365.0, 1 / 365.0)
        r = 0.045  # risk-free rate proxy

        from collections import defaultdict
        net_gex_by_strike: dict[float, float] = defaultdict(float)
        call_gex_total = 0.0
        put_gex_total = 0.0
        oi_by_strike: dict[float, int] = defaultdict(int)

        for _, row in calls.iterrows():
            iv = row.get("impliedVolatility", 0) or 0
            oi = row.get("openInterest", 0) or 0
            K = float(row["strike"])
            if iv > 0.001 and oi > 0:
                g = _bs_gamma(current_price, K, T, r, iv)
                # Dealer is short calls to clients → negative delta → positive gamma
                # When price rises, dealers buy stock (stabilising)
                gex = g * oi * 100 * current_price
                net_gex_by_strike[K] += gex
                oi_by_strike[K] += int(oi)
                call_gex_total += gex

        for _, row in puts.iterrows():
            iv = row.get("impliedVolatility", 0) or 0
            oi = row.get("openInterest", 0) or 0
            K = float(row["strike"])
            if iv > 0.001 and oi > 0:
                g = _bs_gamma(current_price, K, T, r, iv)
                # Dealer is long puts from clients → dealers short stock to hedge
                # → negative GEX → destabilising (dealers sell on drop)
                gex = -g * oi * 100 * current_price
                net_gex_by_strike[K] += gex
                oi_by_strike[K] += int(oi)
                put_gex_total += gex

        if not net_gex_by_strike:
            return GEXResult(0, None, None, [], [], error="No valid options strikes")

        total_gex = sum(net_gex_by_strike.values())
        sorted_strikes = sorted(net_gex_by_strike.items())  # ascending by strike

        # Gamma flip: the strike where cumulative GEX changes sign
        gamma_flip = None
        prev_gex = None
        for K, g in sorted_strikes:
            if prev_gex is not None and (prev_gex >= 0) != (g >= 0):
                gamma_flip = K
                break
            prev_gex = g

        above_flip = (current_price > gamma_flip) if gamma_flip is not None else None

        # Top walls (positive GEX) — dealer buys here, acts as price support
        pos_levels = [(K, g) for K, g in sorted_strikes if g > 0]
        pos_levels.sort(key=lambda x: x[1], reverse=True)
        top_walls = [
            GEXLevel(K, g, oi_by_strike[K], "wall")
            for K, g in pos_levels[:5]
        ]

        # Top trapdoors (negative GEX) — dealer sells here, amplifies downside
        neg_levels = [(K, g) for K, g in sorted_strikes if g < 0]
        neg_levels.sort(key=lambda x: x[1])
        top_trapdoors = [
            GEXLevel(K, g, oi_by_strike[K], "trapdoor")
            for K, g in neg_levels[:5]
        ]

        return GEXResult(
            total_gex=total_gex,
            gamma_flip=gamma_flip,
            above_flip=above_flip,
            top_walls=top_walls,
            top_trapdoors=top_trapdoors,
            call_gex=call_gex_total,
            put_gex=put_gex_total,
            expiry_used=expiry,
        )
    except Exception as e:
        logger.warning("GEX computation failed for %s: %s", ticker, e)
        return GEXResult(0, None, None, [], [], error=str(e))


# ---------------------------------------------------------------------------
# Smart Money / 13F Flow
# ---------------------------------------------------------------------------

def compute_smart_money(ticker: str) -> SmartMoneyResult:
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        info = t.info or {}

        # Institutional holders
        ih = t.institutional_holders
        holders = []
        net_buying = 0.0
        concentration_top5 = 0.0

        if ih is not None and len(ih) > 0:
            for i, row in ih.head(15).iterrows():
                pct = float(row.get("pctHeld", 0) or 0)
                shares = int(row.get("Shares", 0) or 0)
                value = float(row.get("Value", 0) or 0)
                chg = float(row.get("pctChange", 0) or 0)
                name = str(row.get("Holder", ""))
                holders.append(InstitutionalHolder(
                    name=name,
                    pct_held=pct,
                    shares=shares,
                    value=value,
                    pct_change=chg,
                    is_high_conviction=pct >= 0.02,
                ))

            net_buying = sum(h.pct_change for h in holders[:10])
            concentration_top5 = sum(h.pct_held for h in holders[:5])

        high_conviction = sum(1 for h in holders if h.is_high_conviction)
        total_inst_pct = float(info.get("institutionsPercentHeld", 0) or 0)

        # Analyst momentum from recommendations summary
        analyst_momentum = 0.0
        rec_trend = "Stable"
        try:
            recs = t.recommendations
            if recs is not None and len(recs) >= 2:
                # Current month vs 3 months ago
                cur = recs.iloc[0]
                old = recs.iloc[min(3, len(recs)-1)]
                cur_bull = int(cur.get("strongBuy", 0) or 0) + int(cur.get("buy", 0) or 0)
                cur_total = sum(int(cur.get(c, 0) or 0) for c in ["strongBuy","buy","hold","sell","strongSell"])
                old_bull = int(old.get("strongBuy", 0) or 0) + int(old.get("buy", 0) or 0)
                old_total = sum(int(old.get(c, 0) or 0) for c in ["strongBuy","buy","hold","sell","strongSell"])
                if cur_total > 0 and old_total > 0:
                    cur_ratio = cur_bull / cur_total
                    old_ratio = old_bull / old_total
                    analyst_momentum = cur_ratio - old_ratio
                    if analyst_momentum > 0.05:
                        rec_trend = "Improving"
                    elif analyst_momentum < -0.05:
                        rec_trend = "Deteriorating"
        except Exception:
            pass

        return SmartMoneyResult(
            total_institutional_pct=total_inst_pct,
            holders=holders,
            net_buying_pressure=net_buying,
            concentration_top5=concentration_top5,
            high_conviction_count=high_conviction,
            analyst_momentum=analyst_momentum,
            rec_trend=rec_trend,
        )
    except Exception as e:
        logger.warning("Smart money failed for %s: %s", ticker, e)
        return SmartMoneyResult(0, [], 0, 0, 0, 0, "Stable", error=str(e))


# ---------------------------------------------------------------------------
# Credit Proxy (CDS, CS01, DV01)
# ---------------------------------------------------------------------------

def compute_credit_proxy(ticker: str, info: dict, income_stmt: pd.DataFrame) -> CreditProxyResult:
    try:
        total_debt = float(info.get("totalDebt", 0) or 0)
        ebitda = float(info.get("ebitda", 0) or 0)
        interest_expense = None

        if income_stmt is not None and len(income_stmt) > 0:
            for label in ["Interest Expense", "Interest Expense Non Operating"]:
                if label in income_stmt.index:
                    val = income_stmt.loc[label].iloc[0]
                    if val is not None and not (isinstance(val, float) and math.isnan(val)):
                        interest_expense = abs(float(val))
                        break

        # Interest coverage ratio
        ebit = float(info.get("ebit", 0) or 0)
        if ebit == 0 and income_stmt is not None and len(income_stmt) > 0:
            for label in ["EBIT", "Operating Income"]:
                if label in income_stmt.index:
                    v = income_stmt.loc[label].iloc[0]
                    if v and not (isinstance(v, float) and math.isnan(v)):
                        ebit = float(v)
                        break

        interest_coverage = None
        if interest_expense and interest_expense > 0 and ebit:
            interest_coverage = ebit / interest_expense

        # Debt/EBITDA
        debt_to_ebitda = None
        if ebitda and ebitda > 0 and total_debt:
            debt_to_ebitda = total_debt / ebitda

        # Synthetic CDS proxy (basis points)
        # Based on academic mapping: D/E, interest coverage, margin
        cds_bps = None
        net_margin = float(info.get("profitMargins", 0) or 0)
        de = float(info.get("debtToEquity", 0) or 0)

        if debt_to_ebitda is not None and interest_coverage is not None:
            # Higher D/EBITDA → wider spread; higher coverage → tighter
            base = 50  # bps for investment grade baseline
            debt_penalty = min(debt_to_ebitda * 30, 400)  # each turn of debt = ~30bps
            coverage_discount = max(0, (interest_coverage - 1) * 15)  # coverage relief
            margin_adjustment = max(0, -net_margin * 500)  # negative margin widens spread
            cds_bps = base + debt_penalty - coverage_discount + margin_adjustment
            cds_bps = max(20, min(cds_bps, 2000))

        # CS01: $ sensitivity to 1bp credit spread widening
        # Approximation: CS01 ≈ Duration × 0.0001 × Total Debt
        # Corporate bond duration proxy ~5 years for large cap
        duration_proxy = 5.0
        cs01_proxy = None
        if total_debt > 0:
            cs01_proxy = -duration_proxy * 0.0001 * total_debt  # negative: spread widens → value falls

        # DV01: $ sensitivity to 1bp interest rate move
        # Approximation: DV01 ≈ -Duration × 0.0001 × Total Debt (same formula, different driver)
        dv01_proxy = cs01_proxy  # simplified — same duration assumption

        # Credit rating proxy
        if cds_bps is not None:
            if cds_bps < 100:
                rating = "Investment Grade (IG)"
            elif cds_bps < 400:
                rating = "High Yield / Speculative"
            else:
                rating = "Distressed / CCC equivalent"
        elif debt_to_ebitda is not None:
            rating = "Investment Grade" if debt_to_ebitda < 3 else "High Yield" if debt_to_ebitda < 6 else "Distressed"
        else:
            rating = "Unknown"

        return CreditProxyResult(
            cds_proxy_bps=round(cds_bps, 0) if cds_bps else None,
            cs01_proxy=round(cs01_proxy, 0) if cs01_proxy else None,
            dv01_proxy=round(dv01_proxy, 0) if dv01_proxy else None,
            interest_coverage=round(interest_coverage, 2) if interest_coverage else None,
            debt_to_ebitda=round(debt_to_ebitda, 2) if debt_to_ebitda else None,
            credit_rating_proxy=rating,
        )
    except Exception as e:
        logger.warning("Credit proxy failed for %s: %s", ticker, e)
        return CreditProxyResult(None, None, None, None, None, "Unknown", error=str(e))


# ---------------------------------------------------------------------------
# Position Summary (unified narrative)
# ---------------------------------------------------------------------------

def _build_position_summary(ticker: str, price: float, gex: GEXResult,
                              sm: SmartMoneyResult, cr: CreditProxyResult) -> str:
    lines = []

    # GEX regime
    if gex and not gex.error:
        regime = "stable (positive GEX)" if gex.above_flip else "volatile (negative GEX)"
        flip_str = f"${gex.gamma_flip:.0f}" if gex.gamma_flip else "N/A"
        lines.append(f"Dealer gamma regime is **{regime}** — gamma flip at {flip_str}.")
        if gex.top_walls:
            walls = ", ".join(f"${w.strike:.0f}" for w in gex.top_walls[:3])
            lines.append(f"Key GEX support walls (dealers buy here): {walls}.")
        if gex.top_trapdoors:
            traps = ", ".join(f"${t.strike:.0f}" for t in gex.top_trapdoors[:3])
            lines.append(f"GEX trapdoors (dealers forced to sell below): {traps}.")

    # Institutional flow
    if sm and not sm.error:
        flow = "net buying" if sm.net_buying_pressure > 0 else "net selling"
        lines.append(
            f"Institutional ownership {sm.total_institutional_pct:.1%} of float. "
            f"Top-10 holders show **{flow}** pressure ({sm.net_buying_pressure:+.2%} avg QoQ change). "
            f"Analyst consensus is **{sm.rec_trend}**."
        )

    # Credit
    if cr and not cr.error and cr.cds_proxy_bps:
        lines.append(
            f"Synthetic credit spread ~{cr.cds_proxy_bps:.0f}bps ({cr.credit_rating_proxy}). "
            f"Interest coverage: {cr.interest_coverage:.1f}x. "
            f"CS01/DV01: ${abs(cr.cs01_proxy or 0):,.0f} per bp."
        )

    return " ".join(lines) if lines else "Insufficient data for position summary."


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def analyze(ticker: str, price: float, info: dict, income_stmt=None) -> InstitutionalRiskResult:
    result = InstitutionalRiskResult(ticker=ticker)
    try:
        result.gex = compute_gex(ticker, price)
    except Exception as e:
        result.gex = GEXResult(0, None, None, [], [], error=str(e))
    try:
        result.smart_money = compute_smart_money(ticker)
    except Exception as e:
        result.smart_money = SmartMoneyResult(0, [], 0, 0, 0, 0, "Stable", error=str(e))
    try:
        result.credit = compute_credit_proxy(ticker, info, income_stmt)
    except Exception as e:
        result.credit = CreditProxyResult(None, None, None, None, None, "Unknown", error=str(e))
    try:
        result.position_summary = _build_position_summary(
            ticker, price, result.gex, result.smart_money, result.credit
        )
    except Exception:
        pass
    return result
