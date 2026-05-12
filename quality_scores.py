"""Quantitative quality and distress scores.

Piotroski F-Score (0-9): financial strength
Altman Z-Score: bankruptcy predictor
Beneish M-Score: earnings manipulation detector
"""
from __future__ import annotations
import math
import logging
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class QualityScores:
    # Piotroski
    piotroski_f: Optional[int] = None          # 0-9
    piotroski_signals: dict = field(default_factory=dict)  # name -> 0/1
    piotroski_label: str = ""

    # Altman
    altman_z: Optional[float] = None
    altman_zone: str = ""   # "Safe", "Grey", "Distress"
    altman_components: dict = field(default_factory=dict)

    # Beneish
    beneish_m: Optional[float] = None
    beneish_flag: bool = False   # True = manipulation risk
    beneish_components: dict = field(default_factory=dict)


class QualityScorer:

    def compute(
        self,
        info: dict,
        income_stmt: pd.DataFrame,    # multi-column, columns = dates descending
        balance_sheet: pd.DataFrame,
        cashflow: pd.DataFrame,
    ) -> QualityScores:
        result = QualityScores()
        try:
            result.piotroski_f, result.piotroski_signals, result.piotroski_label = self._piotroski(info, income_stmt, balance_sheet, cashflow)
        except Exception as e:
            logger.debug("Piotroski failed: %s", e)
        try:
            result.altman_z, result.altman_zone, result.altman_components = self._altman(info, income_stmt, balance_sheet)
        except Exception as e:
            logger.debug("Altman failed: %s", e)
        try:
            result.beneish_m, result.beneish_flag, result.beneish_components = self._beneish(income_stmt, balance_sheet)
        except Exception as e:
            logger.debug("Beneish failed: %s", e)
        return result

    # ---------------------------------------------------------------
    # PIOTROSKI F-SCORE
    # ---------------------------------------------------------------
    def _piotroski(self, info, inc, bs, cf):
        def row(df, *labels):
            for lbl in labels:
                if lbl in df.index:
                    v = df.loc[lbl]
                    return v.iloc[0], (v.iloc[1] if len(v) > 1 else None)
            return None, None

        signals = {}
        score = 0

        # --- Profitability (4 signals) ---
        net_income_curr, net_income_prev = row(inc, "Net Income", "Net Income Common Stockholders")
        roa_curr = None
        total_assets_curr, total_assets_prev = row(bs, "Total Assets")
        if net_income_curr and total_assets_curr and total_assets_curr != 0:
            roa_curr = net_income_curr / total_assets_curr
            signals["ROA > 0"] = int(roa_curr > 0)
            score += signals["ROA > 0"]

        # Operating cash flow
        ocf_curr, _ = row(cf, "Operating Cash Flow", "Cash Flow From Continuing Operating Activities")
        if ocf_curr is not None:
            signals["CFO > 0"] = int(ocf_curr > 0)
            score += signals["CFO > 0"]

        # Delta ROA
        if net_income_curr and net_income_prev and total_assets_curr and total_assets_prev and total_assets_prev != 0:
            roa_prev = net_income_prev / total_assets_prev
            if roa_curr is not None:
                signals["ΔROA > 0"] = int(roa_curr > roa_prev)
                score += signals["ΔROA > 0"]

        # Accruals: CFO / Total Assets > ROA
        if ocf_curr and total_assets_curr and total_assets_curr != 0 and roa_curr is not None:
            accruals = ocf_curr / total_assets_curr
            signals["CFO/Assets > ROA"] = int(accruals > roa_curr)
            score += signals["CFO/Assets > ROA"]

        # --- Leverage / Liquidity (3 signals) ---
        long_debt_curr, long_debt_prev = row(bs, "Long Term Debt", "Long-Term Debt")
        if long_debt_curr is not None and long_debt_prev is not None and total_assets_curr and total_assets_curr != 0:
            if total_assets_prev and total_assets_prev != 0:
                lev_curr = long_debt_curr / total_assets_curr
                lev_prev = long_debt_prev / total_assets_prev
                signals["ΔLeverage < 0"] = int(lev_curr < lev_prev)
                score += signals["ΔLeverage < 0"]

        cur_assets_curr, cur_assets_prev = row(bs, "Current Assets")
        cur_liab_curr, cur_liab_prev = row(bs, "Current Liabilities")
        if cur_assets_curr and cur_liab_curr and cur_liab_curr != 0:
            cr_curr = cur_assets_curr / cur_liab_curr
            if cur_assets_prev and cur_liab_prev and cur_liab_prev != 0:
                cr_prev = cur_assets_prev / cur_liab_prev
                signals["ΔCurrent Ratio > 0"] = int(cr_curr > cr_prev)
                score += signals["ΔCurrent Ratio > 0"]

        shares_curr, shares_prev = row(bs, "Ordinary Shares Number", "Common Stock")
        if shares_curr is not None and shares_prev is not None:
            signals["No Dilution"] = int(shares_curr <= shares_prev)
            score += signals["No Dilution"]

        # --- Operating Efficiency (2 signals) ---
        revenue_curr, revenue_prev = row(inc, "Total Revenue")
        cogs_curr, cogs_prev = row(inc, "Cost Of Revenue", "Cost of Revenue")
        if revenue_curr and cogs_curr and revenue_curr != 0:
            gm_curr = (revenue_curr - cogs_curr) / revenue_curr
            if revenue_prev and cogs_prev and revenue_prev != 0:
                gm_prev = (revenue_prev - cogs_prev) / revenue_prev
                signals["ΔGross Margin > 0"] = int(gm_curr > gm_prev)
                score += signals["ΔGross Margin > 0"]

        if revenue_curr and total_assets_curr and total_assets_curr != 0:
            at_curr = revenue_curr / total_assets_curr
            if revenue_prev and total_assets_prev and total_assets_prev != 0:
                at_prev = revenue_prev / total_assets_prev
                signals["ΔAsset Turnover > 0"] = int(at_curr > at_prev)
                score += signals["ΔAsset Turnover > 0"]

        if score >= 7:
            label = "Strong (≥7) — Financially Healthy"
        elif score >= 4:
            label = f"Average ({score}) — Mixed Signals"
        else:
            label = f"Weak (≤3) — Financial Stress Signals"

        return score, signals, label

    # ---------------------------------------------------------------
    # ALTMAN Z-SCORE (public company model)
    # ---------------------------------------------------------------
    def _altman(self, info, inc, bs):
        def get(df, *labels):
            for lbl in labels:
                if lbl in df.index:
                    v = df.loc[lbl].iloc[0]
                    if v is not None and not (isinstance(v, float) and math.isnan(v)):
                        return float(v)
            return None

        total_assets = get(bs, "Total Assets")
        if not total_assets or total_assets == 0:
            return None, "", {}

        cur_assets = get(bs, "Current Assets") or 0
        cur_liab   = get(bs, "Current Liabilities") or 0
        retained   = get(bs, "Retained Earnings") or 0
        ebit       = get(inc, "EBIT", "Operating Income") or 0
        revenue    = get(inc, "Total Revenue") or 0
        total_liab = get(bs, "Total Liabilities Net Minority Interest", "Total Liabilities") or 0
        market_cap = info.get("marketCap") or 0

        working_capital = cur_assets - cur_liab

        X1 = working_capital / total_assets
        X2 = retained / total_assets
        X3 = ebit / total_assets
        X4 = market_cap / total_liab if total_liab != 0 else 0
        X5 = revenue / total_assets

        z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5

        if z > 2.99:
            zone = "Safe Zone (Z > 2.99)"
        elif z > 1.81:
            zone = "Grey Zone (1.81 < Z ≤ 2.99)"
        else:
            zone = "Distress Zone (Z ≤ 1.81)"

        components = {"X1 (WC/TA)": round(X1, 3), "X2 (RE/TA)": round(X2, 3),
                      "X3 (EBIT/TA)": round(X3, 3), "X4 (Mkt/Liab)": round(X4, 3),
                      "X5 (Rev/TA)": round(X5, 3)}
        return round(z, 2), zone, components

    # ---------------------------------------------------------------
    # BENEISH M-SCORE (8-variable model)
    # ---------------------------------------------------------------
    def _beneish(self, inc, bs):
        """Requires 2 periods of data (columns 0=current, 1=prior)."""
        if inc.shape[1] < 2 or bs.shape[1] < 2:
            return None, False, {}

        def get(df, col_idx, *labels):
            for lbl in labels:
                if lbl in df.index:
                    v = df.loc[lbl].iloc[col_idx]
                    if v is not None and not (isinstance(v, float) and math.isnan(v)):
                        return float(v)
            return None

        # Current (t) and Prior (t-1)
        rev_t   = get(inc, 0, "Total Revenue") or 0
        rev_p   = get(inc, 1, "Total Revenue") or 1
        cogs_t  = get(inc, 0, "Cost Of Revenue", "Cost of Revenue") or 0
        cogs_p  = get(inc, 1, "Cost Of Revenue", "Cost of Revenue") or 1
        ar_t    = get(bs, 0, "Accounts Receivable", "Net Receivables") or 0
        ar_p    = get(bs, 1, "Accounts Receivable", "Net Receivables") or 0
        ta_t    = get(bs, 0, "Total Assets") or 1
        ta_p    = get(bs, 1, "Total Assets") or 1
        ppe_t   = get(bs, 0, "Net PPE", "Properties Plant And Equipment Net") or 0
        ppe_p   = get(bs, 1, "Net PPE", "Properties Plant And Equipment Net") or 1
        dep_t   = abs(get(inc, 0, "Depreciation And Amortization", "Depreciation") or 0)
        dep_p   = abs(get(inc, 1, "Depreciation And Amortization", "Depreciation") or 1)
        sga_t   = get(inc, 0, "Selling General And Administration") or 0
        sga_p   = get(inc, 1, "Selling General And Administration") or 1
        ni_t    = get(inc, 0, "Net Income") or 0
        cfo_t   = None  # fallback if cashflow not available

        # DSRI: Days Sales Receivable Index
        dsri = (ar_t / rev_t) / (ar_p / rev_p) if rev_t and rev_p else 1

        # GMI: Gross Margin Index
        gm_t = (rev_t - cogs_t) / rev_t if rev_t else 0
        gm_p = (rev_p - cogs_p) / rev_p if rev_p else 0
        gmi = gm_p / gm_t if gm_t else 1

        # AQI: Asset Quality Index
        ca_t  = get(bs, 0, "Current Assets") or 0
        ca_p  = get(bs, 1, "Current Assets") or 0
        aqi = ((ta_t - ca_t - ppe_t) / ta_t) / ((ta_p - ca_p - ppe_p) / ta_p) if ta_t and ta_p else 1

        # SGI: Sales Growth Index
        sgi = rev_t / rev_p if rev_p else 1

        # DEPI: Depreciation Index
        depi = (dep_p / (dep_p + ppe_p)) / (dep_t / (dep_t + ppe_t)) if (dep_t + ppe_t) and (dep_p + ppe_p) else 1

        # SGAI: SG&A Index
        sgai = (sga_t / rev_t) / (sga_p / rev_p) if sga_t and sga_p and rev_t and rev_p else 1

        # LVGI: Leverage Index
        ltd_t = get(bs, 0, "Long Term Debt", "Long-Term Debt") or 0
        ltd_p = get(bs, 1, "Long Term Debt", "Long-Term Debt") or 0
        cl_t  = get(bs, 0, "Current Liabilities") or 0
        cl_p  = get(bs, 1, "Current Liabilities") or 0
        lvgi = ((ltd_t + cl_t) / ta_t) / ((ltd_p + cl_p) / ta_p) if ta_t and ta_p else 1

        # TATA: Total Accruals to Total Assets
        tata = (ni_t - (get(bs, 0, "Retained Earnings") or ni_t)) / ta_t if ta_t else 0

        m = (-4.84
             + 0.920 * dsri
             + 0.528 * gmi
             + 0.404 * aqi
             + 0.892 * sgi
             + 0.115 * depi
             - 0.172 * sgai
             + 4.679 * tata
             - 0.327 * lvgi)

        flag = m > -1.78
        components = {
            "DSRI": round(dsri, 3), "GMI": round(gmi, 3), "AQI": round(aqi, 3),
            "SGI": round(sgi, 3), "DEPI": round(depi, 3), "SGAI": round(sgai, 3),
            "LVGI": round(lvgi, 3), "TATA": round(tata, 3),
        }
        return round(m, 2), flag, components
