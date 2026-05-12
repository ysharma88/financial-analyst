"""Sector-specific financial metrics.

Standard equity metrics fail for certain sectors — this module computes
the metrics that analysts actually use for each sector:

  REITs (Real Estate):
    FFO (Funds From Operations) = Net Income + D&A - Gains on Property Sales
    AFFO (Adjusted FFO) = FFO - Recurring Capex
    P/FFO and P/AFFO multiples

  Banks / Financial Services:
    Net Interest Margin (NIM) proxy = Net Interest Income / Avg Assets
    Efficiency Ratio = Operating Expenses / (NII + Non-Interest Income)
    Loan Loss Reserve Ratio

  SaaS / Software:
    Rule of 40 = Revenue Growth % + FCF Margin %
    SBC-adjusted FCF Margin
    R&D Intensity (R&D / Revenue)

  Consumer / Retail:
    Gross Margin Trend (proxy for pricing power vs. cost pressure)
    Revenue per Employee (operational productivity)
    Working Capital Intensity

  Energy:
    Capex / Operating Cash Flow (reinvestment rate)
    FCF Yield at current commodity prices

  All Sectors:
    EV/Revenue (scored, not just fetched — fills gap in base fundamental analysis)
    EV/EBIT (more conservative than EV/EBITDA)
    Total Shareholder Yield
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sector detection constants
# ---------------------------------------------------------------------------

_REIT_SECTORS = {"Real Estate"}
_BANK_SECTORS = {"Financial Services", "Financials"}
_BANK_INDUSTRY_KEYWORDS = ("Bank", "Insurance", "Brokerage")
_SAAS_SECTORS_PARTIAL = ("Technology", "Communication")
_SAAS_INDUSTRY_KEYWORDS = ("Software", "Internet", "Cloud")
_RETAIL_KEYWORDS = ("Retail", "Restaurant", "Consumer")
_ENERGY_SECTOR_PARTIAL = "Energy"


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _safe(value, default=None):
    """Return numeric value or default when None/NaN/inf."""
    if value is None:
        return default
    try:
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


def _get_stmt_value(stmt: pd.DataFrame, *row_keys) -> Optional[float]:
    """
    Extract a value from a multi-column financial statement DataFrame.
    Tries each key in order; takes the most recent column (index 0).
    Returns None if not found or data is empty.
    """
    if stmt is None or stmt.empty:
        return None
    for key in row_keys:
        # Exact match
        if key in stmt.index:
            row = stmt.loc[key]
            vals = row.dropna()
            if len(vals) > 0:
                return _safe(vals.iloc[0])
        # Case-insensitive partial match
        for idx in stmt.index:
            if isinstance(idx, str) and key.lower() in idx.lower():
                row = stmt.loc[idx]
                vals = row.dropna()
                if len(vals) > 0:
                    return _safe(vals.iloc[0])
    return None


def _get_stmt_two_periods(stmt: pd.DataFrame, *row_keys):
    """Return (current, prior) period values for a given row key."""
    if stmt is None or stmt.empty:
        return None, None
    for key in row_keys:
        if key in stmt.index:
            row = stmt.loc[key].dropna()
            v0 = _safe(row.iloc[0]) if len(row) > 0 else None
            v1 = _safe(row.iloc[1]) if len(row) > 1 else None
            return v0, v1
        for idx in stmt.index:
            if isinstance(idx, str) and key.lower() in idx.lower():
                row = stmt.loc[idx].dropna()
                v0 = _safe(row.iloc[0]) if len(row) > 0 else None
                v1 = _safe(row.iloc[1]) if len(row) > 1 else None
                return v0, v1
    return None, None


# ---------------------------------------------------------------------------
# Sector detection
# ---------------------------------------------------------------------------

def _detect_applicable_module(sector: str, industry: str) -> str:
    """Return the primary sector module name for this stock."""
    sector = sector or ""
    industry = industry or ""

    if sector in _REIT_SECTORS or "REIT" in industry:
        return "REIT"

    if sector in _BANK_SECTORS or any(k in industry for k in _BANK_INDUSTRY_KEYWORDS):
        return "BANK"

    saas_sector_match = any(s in sector for s in _SAAS_SECTORS_PARTIAL)
    saas_industry_match = any(k in industry for k in _SAAS_INDUSTRY_KEYWORDS)
    if saas_sector_match and saas_industry_match:
        return "SAAS"

    if any(k in industry or k in sector for k in _RETAIL_KEYWORDS):
        return "RETAIL"

    if _ENERGY_SECTOR_PARTIAL in sector:
        return "ENERGY"

    return "UNIVERSAL"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class REITMetrics:
    ffo_per_share: Optional[float]      # Funds from Operations per share
    affo_per_share: Optional[float]     # Adjusted FFO per share
    p_ffo: Optional[float]              # Price / FFO
    p_affo: Optional[float]             # Price / AFFO
    ffo_yield: Optional[float]          # FFO / Price (%)
    signal: str                         # qualitative signal


@dataclass
class BankMetrics:
    nim_proxy: Optional[float]          # Net Interest Margin proxy (%)
    efficiency_ratio: Optional[float]   # Operating Expenses / (NII + Non-Interest Income)
    signal: str


@dataclass
class SaaSMetrics:
    rule_of_40: Optional[float]         # Revenue Growth % + FCF Margin %
    sbc_adjusted_fcf_margin: Optional[float]  # FCF - SBC / Revenue (%)
    rd_intensity: Optional[float]       # R&D / Revenue (%)
    signal: str


@dataclass
class RetailMetrics:
    gross_margin_trend: Optional[float]     # YoY change in gross margin (pp)
    revenue_per_employee: Optional[float]   # Revenue / Full-time employees ($)
    signal: str


@dataclass
class UniversalMetrics:
    ev_revenue: Optional[float]         # Enterprise Value / Revenue
    ev_ebit: Optional[float]            # Enterprise Value / EBIT
    total_shareholder_yield: Optional[float]  # (Dividends + Buybacks) / Market Cap (%)
    ev_revenue_signal: str
    ev_ebit_signal: str


@dataclass
class SectorMetricsResult:
    sector: str
    industry: str
    applicable_module: str
    universal: UniversalMetrics
    reit: Optional[REITMetrics]
    bank: Optional[BankMetrics]
    saas: Optional[SaaSMetrics]
    retail: Optional[RetailMetrics]
    quality_flags: list[str]
    overall_score: float      # –1 to +1
    signal: str               # STRONG BUY / BUY / HOLD / SELL / STRONG SELL
    summary: str


# ---------------------------------------------------------------------------
# Universal metrics
# ---------------------------------------------------------------------------

def _compute_universal(
    info: dict,
    income_stmt: pd.DataFrame,
) -> UniversalMetrics:
    """EV/Revenue, EV/EBIT, total shareholder yield."""
    ev = _safe(info.get("enterpriseValue"))
    revenue = _safe(info.get("totalRevenue"))
    market_cap = _safe(info.get("marketCap"))
    shares = _safe(info.get("sharesOutstanding") or info.get("impliedSharesOutstanding"))

    # EBIT from income statement
    ebit = _get_stmt_value(
        income_stmt,
        "EBIT",
        "Operating Income",
        "Total Operating Income As Reported",
        "Ebit",
        "Operating Income Or Loss",
    )

    # EV / Revenue
    ev_revenue: Optional[float] = None
    if ev is not None and revenue is not None and revenue > 0:
        ev_revenue = round(ev / revenue, 2)

    # EV / EBIT
    ev_ebit: Optional[float] = None
    if ev is not None and ebit is not None and ebit > 0:
        ev_ebit = round(ev / ebit, 2)

    # Total Shareholder Yield = (Dividends + Net Buybacks) / Market Cap
    total_shareholder_yield: Optional[float] = None
    if market_cap is not None and market_cap > 0:
        div_yield_raw = _safe(info.get("dividendYield"), default=0.0)
        div_yield = (div_yield_raw or 0.0) * 100  # already as fraction in yfinance

        # Buyback yield proxy: repurchaseOfStock from cashflow (not available here)
        # Use info.get('buybackYield') if available, else 0
        buyback_yield = _safe(info.get("buybackYield"), default=0.0) * 100

        if div_yield > 0 or buyback_yield > 0:
            total_shareholder_yield = round(div_yield + buyback_yield, 2)

    # EV/Revenue signal (sector-agnostic thresholds)
    ev_revenue_signal = "N/A"
    if ev_revenue is not None:
        if ev_revenue < 1.0:
            ev_revenue_signal = "DEEP VALUE — EV/Rev below 1x"
        elif ev_revenue < 3.0:
            ev_revenue_signal = "ATTRACTIVE — EV/Rev below 3x"
        elif ev_revenue < 6.0:
            ev_revenue_signal = "FAIR — EV/Rev 3-6x"
        elif ev_revenue < 12.0:
            ev_revenue_signal = "ELEVATED — EV/Rev 6-12x (needs growth justification)"
        else:
            ev_revenue_signal = f"EXPENSIVE — EV/Rev {ev_revenue:.1f}x"

    # EV/EBIT signal
    ev_ebit_signal = "N/A"
    if ev_ebit is not None:
        if ev_ebit < 10:
            ev_ebit_signal = "CHEAP — EV/EBIT below 10x"
        elif ev_ebit < 15:
            ev_ebit_signal = "FAIR — EV/EBIT 10-15x"
        elif ev_ebit < 25:
            ev_ebit_signal = "MODERATE — EV/EBIT 15-25x"
        elif ev_ebit < 40:
            ev_ebit_signal = "ELEVATED — EV/EBIT 25-40x"
        else:
            ev_ebit_signal = f"STRETCHED — EV/EBIT {ev_ebit:.1f}x"

    return UniversalMetrics(
        ev_revenue=ev_revenue,
        ev_ebit=ev_ebit,
        total_shareholder_yield=total_shareholder_yield,
        ev_revenue_signal=ev_revenue_signal,
        ev_ebit_signal=ev_ebit_signal,
    )


# ---------------------------------------------------------------------------
# REIT metrics
# ---------------------------------------------------------------------------

def _compute_reit(
    info: dict,
    income_stmt: pd.DataFrame,
    balance_sheet: pd.DataFrame,
    cashflow: pd.DataFrame,
) -> REITMetrics:
    """FFO, AFFO, P/FFO, P/AFFO."""
    current_price = _safe(
        info.get("currentPrice")
        or info.get("regularMarketPrice")
        or info.get("previousClose")
    )
    shares = _safe(info.get("sharesOutstanding") or info.get("impliedSharesOutstanding"))

    # Net Income
    net_income = _get_stmt_value(
        income_stmt, "Net Income", "NetIncome", "Net Income Common Stockholders"
    )

    # D&A
    da = _get_stmt_value(
        cashflow,
        "Depreciation And Amortization",
        "Depreciation",
        "DepreciationAndAmortization",
    )
    if da is None:
        da = _get_stmt_value(
            income_stmt,
            "Reconciled Depreciation",
            "Depreciation And Amortization",
        )

    # Capex from cashflow
    capex = _get_stmt_value(
        cashflow,
        "Capital Expenditure",
        "Capex",
        "Purchase Of Plant Property Equipment",
    )
    if capex is not None:
        capex = abs(capex)  # typically reported as negative outflow

    # FFO = Net Income + D&A (gains on property sales approximated as 0)
    ffo = None
    if net_income is not None and da is not None:
        ffo = net_income + da

    # AFFO = FFO - (Capex * 0.5) for maintenance capex assumption
    affo = None
    if ffo is not None and capex is not None:
        affo = ffo - (capex * 0.5)
    elif ffo is not None:
        affo = ffo  # fall back to FFO if no capex data

    ffo_per_share = None
    affo_per_share = None
    p_ffo = None
    p_affo = None
    ffo_yield = None

    if shares is not None and shares > 0:
        if ffo is not None:
            ffo_per_share = round(ffo / shares, 4)
            if current_price is not None and current_price > 0 and ffo_per_share != 0:
                p_ffo = round(current_price / ffo_per_share, 2)
                ffo_yield = round((ffo_per_share / current_price) * 100, 2)
        if affo is not None:
            affo_per_share = round(affo / shares, 4)
            if current_price is not None and current_price > 0 and affo_per_share not in (0, None):
                p_affo = round(current_price / affo_per_share, 2)

    # Signal
    if p_ffo is not None:
        if p_ffo < 12:
            signal = "CHEAP REIT — P/FFO below 12x"
        elif p_ffo < 18:
            signal = "FAIR VALUE — P/FFO 12-18x"
        elif p_ffo < 25:
            signal = "MODERATELY EXPENSIVE — P/FFO 18-25x"
        else:
            signal = f"EXPENSIVE REIT — P/FFO {p_ffo:.1f}x"
    elif ffo_per_share is None:
        signal = "INSUFFICIENT DATA — FFO not computable from available financials"
    else:
        signal = "REIT METRICS PARTIAL — price unavailable"

    return REITMetrics(
        ffo_per_share=ffo_per_share,
        affo_per_share=affo_per_share,
        p_ffo=p_ffo,
        p_affo=p_affo,
        ffo_yield=ffo_yield,
        signal=signal,
    )


# ---------------------------------------------------------------------------
# Bank metrics
# ---------------------------------------------------------------------------

def _compute_bank(
    info: dict,
    income_stmt: pd.DataFrame,
    balance_sheet: pd.DataFrame,
) -> BankMetrics:
    """NIM proxy, efficiency ratio for banks/financials."""
    # Interest Income
    interest_income = _get_stmt_value(
        income_stmt,
        "Interest Income",
        "Net Interest Income",
        "InterestIncome",
        "Total Interest Income",
    )

    # Interest Expense
    interest_expense = _get_stmt_value(
        income_stmt,
        "Interest Expense",
        "InterestExpense",
        "Total Interest Expense",
    )
    if interest_expense is not None:
        interest_expense = abs(interest_expense)

    # Total Assets (current and prior period for average)
    total_assets_curr, total_assets_prior = _get_stmt_two_periods(
        balance_sheet, "Total Assets", "TotalAssets"
    )

    nim_proxy: Optional[float] = None
    if interest_income is not None and total_assets_curr is not None and total_assets_curr > 0:
        nii = interest_income - (interest_expense or 0.0)
        avg_assets = (
            (total_assets_curr + total_assets_prior) / 2
            if total_assets_prior
            else total_assets_curr
        )
        nim_proxy = round((nii / avg_assets) * 100, 3)

    # Efficiency Ratio = Operating Expenses / (NII + Non-Interest Income)
    op_expense = _get_stmt_value(
        income_stmt,
        "Operating Expense",
        "Total Operating Expenses",
        "Noninterest Expense",
        "Operating Expenses",
    )
    non_interest_income = _get_stmt_value(
        income_stmt,
        "Non Interest Income",
        "Noninterest Income",
        "Other Income",
        "Total Other Income Expense Net",
    )

    efficiency_ratio: Optional[float] = None
    if op_expense is not None and interest_income is not None:
        nii = interest_income - (interest_expense or 0.0)
        denominator = nii + (non_interest_income or 0.0)
        if denominator > 0:
            efficiency_ratio = round((abs(op_expense) / denominator) * 100, 2)

    # Signal
    flags: list[str] = []
    if nim_proxy is not None:
        if nim_proxy < 1.5:
            flags.append(f"THIN NIM ({nim_proxy:.2f}%) — margin pressure likely")
        elif nim_proxy < 3.0:
            flags.append(f"ADEQUATE NIM ({nim_proxy:.2f}%)")
        else:
            flags.append(f"STRONG NIM ({nim_proxy:.2f}%)")
    else:
        flags.append("NIM N/A — requires bank-specific disclosures")

    if efficiency_ratio is not None:
        if efficiency_ratio < 45:
            flags.append(f"HIGHLY EFFICIENT bank (ratio={efficiency_ratio:.1f}%)")
        elif efficiency_ratio < 60:
            flags.append(f"AVERAGE EFFICIENCY (ratio={efficiency_ratio:.1f}%)")
        else:
            flags.append(f"POOR EFFICIENCY (ratio={efficiency_ratio:.1f}%) — cost pressure")
    else:
        flags.append("Efficiency ratio N/A")

    overall_good = (
        (nim_proxy is not None and nim_proxy >= 2.5)
        and (efficiency_ratio is not None and efficiency_ratio < 55)
    )
    signal = "STRONG BANK FUNDAMENTALS" if overall_good else " | ".join(flags)

    return BankMetrics(
        nim_proxy=nim_proxy,
        efficiency_ratio=efficiency_ratio,
        signal=signal,
    )


# ---------------------------------------------------------------------------
# SaaS metrics
# ---------------------------------------------------------------------------

def _compute_saas(
    info: dict,
    income_stmt: pd.DataFrame,
    cashflow: pd.DataFrame,
) -> SaaSMetrics:
    """Rule of 40, SBC-adjusted FCF margin, R&D intensity."""
    revenue = _safe(info.get("totalRevenue"))
    revenue_growth = _safe(info.get("revenueGrowth"))
    fcf = _safe(info.get("freeCashflow"))

    # SBC from cashflow
    sbc = _get_stmt_value(
        cashflow,
        "Stock Based Compensation",
        "ShareBasedCompensation",
        "Share Based Compensation",
    )

    # R&D from income statement
    rd = _get_stmt_value(
        income_stmt,
        "Research And Development",
        "ResearchAndDevelopment",
        "R&D Expenses",
        "Research Development",
    )

    rule_of_40: Optional[float] = None
    sbc_adjusted_fcf_margin: Optional[float] = None
    rd_intensity: Optional[float] = None

    if revenue is not None and revenue > 0:
        fcf_margin_pct = (fcf / revenue * 100) if fcf is not None else 0.0
        rev_growth_pct = (revenue_growth or 0.0) * 100

        rule_of_40 = round(rev_growth_pct + fcf_margin_pct, 2)

        # SBC-adjusted FCF = FCF - SBC
        if fcf is not None and sbc is not None:
            sbc_adj_fcf = fcf - abs(sbc)
            sbc_adjusted_fcf_margin = round((sbc_adj_fcf / revenue) * 100, 2)
        elif fcf is not None:
            sbc_adjusted_fcf_margin = round(fcf_margin_pct, 2)

        if rd is not None:
            rd_intensity = round((abs(rd) / revenue) * 100, 2)

    # Signal
    if rule_of_40 is not None:
        if rule_of_40 >= 60:
            signal = f"ELITE SaaS — Rule of 40 = {rule_of_40:.1f} (exceptional)"
        elif rule_of_40 >= 40:
            signal = f"STRONG SaaS — Rule of 40 = {rule_of_40:.1f} (above benchmark)"
        elif rule_of_40 >= 20:
            signal = f"GROWING SaaS — Rule of 40 = {rule_of_40:.1f} (below benchmark)"
        else:
            signal = f"WEAK SaaS PROFILE — Rule of 40 = {rule_of_40:.1f} (needs improvement)"
    else:
        signal = "INSUFFICIENT DATA for SaaS metrics"

    if sbc_adjusted_fcf_margin is not None and rule_of_40 is not None:
        fcf_margin_pct_check = _safe(info.get("freeCashflow"), default=0.0)
        rev_check = revenue or 1
        raw_fcf_margin = (fcf_margin_pct_check / rev_check * 100) if revenue else 0
        diff = raw_fcf_margin - (sbc_adjusted_fcf_margin or 0.0)
        if diff > 10:
            signal += f" | CAUTION: SBC dilution removes {diff:.1f}pp from FCF margin"

    return SaaSMetrics(
        rule_of_40=rule_of_40,
        sbc_adjusted_fcf_margin=sbc_adjusted_fcf_margin,
        rd_intensity=rd_intensity,
        signal=signal,
    )


# ---------------------------------------------------------------------------
# Retail metrics
# ---------------------------------------------------------------------------

def _compute_retail(
    info: dict,
    income_stmt: pd.DataFrame,
) -> RetailMetrics:
    """Gross margin trend, revenue per employee."""
    # Gross profit two periods
    gross_profit_curr, gross_profit_prior = _get_stmt_two_periods(
        income_stmt, "Gross Profit", "GrossProfit"
    )
    revenue_curr, revenue_prior = _get_stmt_two_periods(
        income_stmt, "Total Revenue", "Revenue", "TotalRevenue"
    )

    gross_margin_trend: Optional[float] = None
    if (
        gross_profit_curr is not None
        and gross_profit_prior is not None
        and revenue_curr is not None
        and revenue_prior is not None
        and revenue_curr > 0
        and revenue_prior > 0
    ):
        gm_curr = gross_profit_curr / revenue_curr * 100
        gm_prior = gross_profit_prior / revenue_prior * 100
        gross_margin_trend = round(gm_curr - gm_prior, 2)  # change in pp

    # Revenue per employee
    employees = _safe(info.get("fullTimeEmployees"))
    revenue = _safe(info.get("totalRevenue"))
    revenue_per_employee: Optional[float] = None
    if revenue is not None and employees is not None and employees > 0:
        revenue_per_employee = round(revenue / employees, 2)

    # Signal
    parts: list[str] = []
    if gross_margin_trend is not None:
        if gross_margin_trend > 1.0:
            parts.append(f"Gross margin EXPANDING (+{gross_margin_trend:.1f}pp YoY) — pricing power")
        elif gross_margin_trend > -1.0:
            parts.append(f"Gross margin STABLE ({gross_margin_trend:+.1f}pp YoY)")
        else:
            parts.append(f"Gross margin CONTRACTING ({gross_margin_trend:.1f}pp YoY) — cost pressure")

    if revenue_per_employee is not None:
        rev_k = revenue_per_employee / 1_000
        if rev_k > 500:
            parts.append(f"HIGH productivity: ${rev_k:,.0f}K revenue/employee")
        elif rev_k > 200:
            parts.append(f"AVERAGE productivity: ${rev_k:,.0f}K revenue/employee")
        else:
            parts.append(f"LOW productivity: ${rev_k:,.0f}K revenue/employee")

    signal = " | ".join(parts) if parts else "INSUFFICIENT RETAIL DATA"

    return RetailMetrics(
        gross_margin_trend=gross_margin_trend,
        revenue_per_employee=revenue_per_employee,
        signal=signal,
    )


# ---------------------------------------------------------------------------
# Energy metrics (universal + capex ratio)
# ---------------------------------------------------------------------------

def _compute_energy_flags(info: dict, cashflow: pd.DataFrame) -> list[str]:
    """Return quality flags specific to energy sector."""
    flags: list[str] = []

    capex = _get_stmt_value(
        cashflow, "Capital Expenditure", "Capex", "Purchase Of Plant Property Equipment"
    )
    op_cf = _get_stmt_value(
        cashflow, "Operating Cash Flow", "Cash Flow From Operations", "Total Cash From Operating Activities"
    )

    if capex is not None and op_cf is not None and op_cf > 0:
        reinvestment_rate = abs(capex) / op_cf
        flags.append(
            f"Energy reinvestment rate: {reinvestment_rate:.1%} of OCF spent on capex"
        )
        if reinvestment_rate > 0.8:
            flags.append("HIGH REINVESTMENT: little free cash for dividends/buybacks")
        elif reinvestment_rate < 0.4:
            flags.append("LOW CAPEX: harvesting mode — limited growth investment")

    fcf = _safe(info.get("freeCashflow"))
    market_cap = _safe(info.get("marketCap"))
    if fcf is not None and market_cap is not None and market_cap > 0:
        fcf_yield = (fcf / market_cap) * 100
        flags.append(f"FCF Yield: {fcf_yield:.1f}%")
        if fcf_yield > 10:
            flags.append("EXCEPTIONAL FCF YIELD — potential value or commodity cycle peak")
        elif fcf_yield > 5:
            flags.append("STRONG FCF YIELD for energy sector")

    return flags


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _universal_score(universal: UniversalMetrics) -> float:
    """Score universal metrics, –1 to +1."""
    score = 0.0
    count = 0

    if universal.ev_revenue is not None:
        ev_rev = universal.ev_revenue
        if ev_rev < 1.0:
            score += 1.0
        elif ev_rev < 3.0:
            score += 0.5
        elif ev_rev < 6.0:
            score += 0.0
        elif ev_rev < 12.0:
            score -= 0.5
        else:
            score -= 1.0
        count += 1

    if universal.ev_ebit is not None:
        ev_ebit = universal.ev_ebit
        if ev_ebit < 10:
            score += 1.0
        elif ev_ebit < 15:
            score += 0.5
        elif ev_ebit < 25:
            score += 0.0
        elif ev_ebit < 40:
            score -= 0.5
        else:
            score -= 1.0
        count += 1

    return score / count if count > 0 else 0.0


def _sector_score(
    module: str,
    reit: Optional[REITMetrics],
    bank: Optional[BankMetrics],
    saas: Optional[SaaSMetrics],
    retail: Optional[RetailMetrics],
) -> float:
    """Return sector-specific component of overall_score (–1 to +1)."""
    if module == "REIT" and reit is not None and reit.p_ffo is not None:
        p = reit.p_ffo
        if p < 12:
            return 1.0
        elif p < 18:
            return 0.3
        elif p < 25:
            return -0.3
        else:
            return -1.0

    if module == "BANK" and bank is not None:
        score = 0.0
        count = 0
        if bank.nim_proxy is not None:
            score += 1.0 if bank.nim_proxy >= 3.0 else (0.0 if bank.nim_proxy >= 2.0 else -0.5)
            count += 1
        if bank.efficiency_ratio is not None:
            score += 1.0 if bank.efficiency_ratio < 45 else (0.0 if bank.efficiency_ratio < 60 else -1.0)
            count += 1
        return score / count if count > 0 else 0.0

    if module == "SAAS" and saas is not None and saas.rule_of_40 is not None:
        r40 = saas.rule_of_40
        if r40 >= 60:
            return 1.0
        elif r40 >= 40:
            return 0.5
        elif r40 >= 20:
            return 0.0
        else:
            return -0.5

    if module == "RETAIL" and retail is not None:
        score = 0.0
        count = 0
        if retail.gross_margin_trend is not None:
            score += 0.5 if retail.gross_margin_trend > 1.0 else (-0.5 if retail.gross_margin_trend < -1.0 else 0.0)
            count += 1
        return score / count if count > 0 else 0.0

    return 0.0


def _overall_signal(score: float) -> str:
    if score >= 0.6:
        return "STRONG BUY"
    elif score >= 0.2:
        return "BUY"
    elif score >= -0.2:
        return "HOLD"
    elif score >= -0.6:
        return "SELL"
    else:
        return "STRONG SELL"


# ---------------------------------------------------------------------------
# Main analyser
# ---------------------------------------------------------------------------

class SectorMetricsAnalyzer:
    """Computes sector-appropriate financial metrics for a stock."""

    def analyze(
        self,
        ticker: str,
        info: dict,
        income_stmt: pd.DataFrame,
        balance_sheet: pd.DataFrame,
        cashflow: pd.DataFrame,
    ) -> SectorMetricsResult:
        """
        Parameters
        ----------
        ticker       : stock symbol (e.g. 'O' for Realty Income)
        info         : yfinance Ticker.info dict
        income_stmt  : Ticker.income_stmt (rows = line items, cols = dates)
        balance_sheet: Ticker.balance_sheet
        cashflow     : Ticker.cashflow

        Returns
        -------
        SectorMetricsResult dataclass
        """
        info = info or {}
        sector = info.get("sector") or ""
        industry = info.get("industry") or ""
        applicable_module = _detect_applicable_module(sector, industry)

        # Universal metrics always computed
        universal = _compute_universal(info, income_stmt)

        reit_metrics: Optional[REITMetrics] = None
        bank_metrics: Optional[BankMetrics] = None
        saas_metrics: Optional[SaaSMetrics] = None
        retail_metrics: Optional[RetailMetrics] = None

        quality_flags: list[str] = []

        try:
            if applicable_module == "REIT":
                reit_metrics = _compute_reit(info, income_stmt, balance_sheet, cashflow)
            elif applicable_module == "BANK":
                bank_metrics = _compute_bank(info, income_stmt, balance_sheet)
            elif applicable_module == "SAAS":
                saas_metrics = _compute_saas(info, income_stmt, cashflow)
            elif applicable_module == "RETAIL":
                retail_metrics = _compute_retail(info, income_stmt)
            elif applicable_module == "ENERGY":
                quality_flags.extend(_compute_energy_flags(info, cashflow))
        except Exception as exc:
            logger.warning("Sector module %s failed for %s: %s", applicable_module, ticker, exc)
            quality_flags.append(f"Sector module computation error: {exc}")

        # Universal quality flags
        if universal.ev_revenue is not None and universal.ev_revenue > 20:
            quality_flags.append(
                f"EXTREME VALUATION: EV/Revenue {universal.ev_revenue:.1f}x — "
                f"requires exceptional growth to justify"
            )
        if universal.ev_ebit is not None and universal.ev_ebit < 0:
            quality_flags.append("NEGATIVE EBIT: company is not operating-profitable")
        if universal.total_shareholder_yield is not None and universal.total_shareholder_yield > 10:
            quality_flags.append(
                f"HIGH SHAREHOLDER YIELD {universal.total_shareholder_yield:.1f}% — "
                f"verify sustainability"
            )

        # Scoring
        u_score = _universal_score(universal)
        s_score = _sector_score(applicable_module, reit_metrics, bank_metrics, saas_metrics, retail_metrics)
        # Weight 40% universal, 60% sector-specific
        if applicable_module == "UNIVERSAL":
            overall_score = u_score
        else:
            overall_score = 0.4 * u_score + 0.6 * s_score
        overall_score = round(max(-1.0, min(1.0, overall_score)), 3)

        signal = _overall_signal(overall_score)

        # Summary
        summary_parts = [
            f"{ticker} | Sector: {sector or 'N/A'} | Industry: {industry or 'N/A'} | Module: {applicable_module}.",
        ]

        if universal.ev_revenue is not None:
            summary_parts.append(f"EV/Revenue: {universal.ev_revenue:.1f}x ({universal.ev_revenue_signal}).")
        if universal.ev_ebit is not None:
            summary_parts.append(f"EV/EBIT: {universal.ev_ebit:.1f}x ({universal.ev_ebit_signal}).")
        if universal.total_shareholder_yield is not None:
            summary_parts.append(f"Total Shareholder Yield: {universal.total_shareholder_yield:.1f}%.")

        if reit_metrics is not None:
            if reit_metrics.p_ffo is not None:
                summary_parts.append(
                    f"REIT — P/FFO: {reit_metrics.p_ffo:.1f}x, P/AFFO: {reit_metrics.p_affo or 'N/A'}x, "
                    f"FFO Yield: {reit_metrics.ffo_yield or 'N/A'}%. {reit_metrics.signal}."
                )
            else:
                summary_parts.append(f"REIT — {reit_metrics.signal}.")

        if bank_metrics is not None:
            nim_str = f"{bank_metrics.nim_proxy:.2f}%" if bank_metrics.nim_proxy is not None else "N/A"
            eff_str = f"{bank_metrics.efficiency_ratio:.1f}%" if bank_metrics.efficiency_ratio is not None else "N/A"
            summary_parts.append(
                f"Bank — NIM: {nim_str}, Efficiency Ratio: {eff_str}. {bank_metrics.signal}."
            )

        if saas_metrics is not None:
            r40_str = f"{saas_metrics.rule_of_40:.1f}" if saas_metrics.rule_of_40 is not None else "N/A"
            fcf_str = (
                f"{saas_metrics.sbc_adjusted_fcf_margin:.1f}%"
                if saas_metrics.sbc_adjusted_fcf_margin is not None
                else "N/A"
            )
            rd_str = (
                f"{saas_metrics.rd_intensity:.1f}%"
                if saas_metrics.rd_intensity is not None
                else "N/A"
            )
            summary_parts.append(
                f"SaaS — Rule of 40: {r40_str}, SBC-adj FCF Margin: {fcf_str}, "
                f"R&D Intensity: {rd_str}. {saas_metrics.signal}."
            )

        if retail_metrics is not None:
            gmt_str = (
                f"{retail_metrics.gross_margin_trend:+.1f}pp"
                if retail_metrics.gross_margin_trend is not None
                else "N/A"
            )
            rpe_str = (
                f"${retail_metrics.revenue_per_employee:,.0f}"
                if retail_metrics.revenue_per_employee is not None
                else "N/A"
            )
            summary_parts.append(
                f"Retail — GM Trend: {gmt_str} YoY, Rev/Employee: {rpe_str}. "
                f"{retail_metrics.signal}."
            )

        if quality_flags:
            summary_parts.append(f"Flags: {' | '.join(quality_flags[:4])}.")

        summary_parts.append(
            f"Overall sector score: {overall_score:+.2f} → {signal}."
        )

        return SectorMetricsResult(
            sector=sector,
            industry=industry,
            applicable_module=applicable_module,
            universal=universal,
            reit=reit_metrics,
            bank=bank_metrics,
            saas=saas_metrics,
            retail=retail_metrics,
            quality_flags=quality_flags,
            overall_score=overall_score,
            signal=signal,
            summary=" ".join(summary_parts),
        )
