"""3-stage DCF model + sensitivity analysis for intrinsic value estimation."""
from __future__ import annotations
import math
import logging
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DCFResult:
    intrinsic_value: Optional[float] = None     # per share
    current_price: Optional[float] = None
    margin_of_safety: Optional[float] = None    # (intrinsic - price) / intrinsic
    upside_pct: Optional[float] = None          # (intrinsic - price) / price
    valuation_label: str = ""                   # "Undervalued", "Fairly Valued", "Overvalued"
    # Inputs used
    fcf_base: Optional[float] = None            # FCF used as base
    wacc: Optional[float] = None
    growth_stage1: Optional[float] = None       # yr 1-5
    growth_stage2: Optional[float] = None       # yr 6-10
    terminal_growth: float = 0.03
    shares_outstanding: Optional[float] = None
    net_debt: Optional[float] = None
    # Sensitivity: dict of (wacc_delta, growth_delta) -> intrinsic_value
    sensitivity: dict = field(default_factory=dict)
    error: str = ""


class DCFValuator:

    def compute(
        self,
        info: dict,
        income_stmt: pd.DataFrame,
        balance_sheet: pd.DataFrame,
        cashflow: pd.DataFrame,
    ) -> DCFResult:
        result = DCFResult()
        try:
            self._run(result, info, income_stmt, balance_sheet, cashflow)
        except Exception as e:
            result.error = str(e)
            logger.debug("DCF failed: %s", e)
        return result

    def _run(self, result, info, inc, bs, cf):
        # ---- FCF base: prefer trailing FCF from cash flow stmt, fall back to info ----
        fcf_base = self._get_fcf(cf, inc, info)
        if not fcf_base or fcf_base <= 0:
            result.error = "Negative or unavailable FCF — DCF not meaningful"
            return
        result.fcf_base = fcf_base

        # ---- WACC ----
        wacc = info.get("wacc")
        if not wacc or wacc <= 0:
            # simple fallback
            beta = info.get("beta") or 1.0
            wacc = 0.045 + beta * 0.055
        result.wacc = wacc

        # ---- Growth rates ----
        rev_growth = info.get("revenueGrowth") or 0
        earn_growth = info.get("earningsGrowth") or 0
        analyst_growth = info.get("earningsQuarterlyGrowth") or 0

        # Stage 1 (yr 1-5): blend of recent growth, capped reasonably
        stage1 = max(-0.20, min(0.40, (rev_growth + earn_growth) / 2)) if rev_growth or earn_growth else 0.05
        # Stage 2 (yr 6-10): fade to half of stage1
        stage2 = stage1 * 0.5
        # Ensure stage2 not below terminal
        terminal_g = 0.03
        stage2 = max(stage2, terminal_g)
        result.growth_stage1 = stage1
        result.growth_stage2 = stage2
        result.terminal_growth = terminal_g

        # ---- Shares outstanding ----
        shares = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding")
        if not shares:
            result.error = "Shares outstanding unavailable"
            return
        result.shares_outstanding = shares

        # ---- Net debt ----
        total_debt = info.get("totalDebt") or 0
        total_cash = info.get("totalCash") or 0
        net_debt = total_debt - total_cash
        result.net_debt = net_debt

        # ---- DCF calculation ----
        iv = self._dcf(fcf_base, stage1, stage2, terminal_g, wacc, shares, net_debt)
        result.intrinsic_value = iv

        price = info.get("currentPrice") or info.get("regularMarketPrice")
        result.current_price = price
        if iv and price and price > 0 and iv > 0:
            result.margin_of_safety = (iv - price) / iv
            result.upside_pct = (iv - price) / price
            mos = result.margin_of_safety
            if mos > 0.30:
                result.valuation_label = "Significantly Undervalued"
            elif mos > 0.10:
                result.valuation_label = "Moderately Undervalued"
            elif mos > -0.10:
                result.valuation_label = "Fairly Valued"
            elif mos > -0.30:
                result.valuation_label = "Moderately Overvalued"
            else:
                result.valuation_label = "Significantly Overvalued"

        # ---- Sensitivity table ----
        wacc_deltas = [-0.02, -0.01, 0, +0.01, +0.02]
        growth_deltas = [-0.05, -0.025, 0, +0.025, +0.05]
        for wd in wacc_deltas:
            for gd in growth_deltas:
                w2 = max(0.04, wacc + wd)
                g1_2 = max(-0.15, min(0.50, stage1 + gd))
                g2_2 = max(terminal_g, g1_2 * 0.5)
                sv = self._dcf(fcf_base, g1_2, g2_2, terminal_g, w2, shares, net_debt)
                result.sensitivity[(round(wd*100, 1), round(gd*100, 1))] = sv

    def _dcf(self, fcf, g1, g2, gt, wacc, shares, net_debt):
        if wacc <= gt:
            wacc = gt + 0.02
        pv = 0.0
        cf = fcf
        for yr in range(1, 6):
            cf *= (1 + g1)
            pv += cf / (1 + wacc) ** yr
        for yr in range(6, 11):
            cf *= (1 + g2)
            pv += cf / (1 + wacc) ** yr
        # Terminal value (Gordon Growth)
        tv = cf * (1 + gt) / (wacc - gt)
        pv_tv = tv / (1 + wacc) ** 10
        equity_value = pv + pv_tv - net_debt
        return round(equity_value / shares, 2) if shares > 0 and equity_value > 0 else None

    def _get_fcf(self, cf, inc, info):
        # Try cashflow statement first
        for lbl in ["Free Cash Flow", "Operating Cash Flow"]:
            if cf is not None and not cf.empty and lbl in cf.index:
                v = cf.loc[lbl].iloc[0]
                if v and not (isinstance(v, float) and math.isnan(v)) and float(v) > 0:
                    return float(v)
        # Fall back to info dict
        fcf = info.get("freeCashflow")
        if fcf and fcf > 0:
            return fcf
        ocf = info.get("operatingCashflow")
        capex = None
        if inc is not None and not inc.empty:
            for lbl in ["Capital Expenditure", "Purchases Of Property Plant And Equipment"]:
                if lbl in inc.index:
                    v = inc.loc[lbl].iloc[0]
                    if v is not None and not (isinstance(v, float) and math.isnan(v)):
                        capex = abs(float(v))
                        break
        if ocf and ocf > 0:
            return ocf - (capex or 0)
        return None
