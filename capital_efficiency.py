"""Capital efficiency and allocation analysis.

Computes:
  Working Capital: DSO, DIO, DPO, Cash Conversion Cycle (CCC) + YoY changes
  Capital Expenditure: intensity (Capex/Revenue), growth vs maintenance (Capex/D&A)
  Capital Return: buyback yield, total shareholder yield (dividend + buyback)
  Leverage: Net Debt/EBITDA (surfaced), Net Debt trajectory (YoY change)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class WorkingCapitalMetrics:
    dso: Optional[float] = None            # Days Sales Outstanding
    dio: Optional[float] = None            # Days Inventory Outstanding
    dpo: Optional[float] = None            # Days Payable Outstanding
    ccc: Optional[float] = None            # Cash Conversion Cycle = DSO + DIO - DPO
    dso_yoy_change: Optional[float] = None # change in days vs prior year
    dio_yoy_change: Optional[float] = None
    ccc_yoy_change: Optional[float] = None
    working_capital_signal: str = "N/A"


@dataclass
class CapexMetrics:
    capex_to_revenue: Optional[float] = None        # Capex / Revenue (0-1 ratio)
    capex_to_depreciation: Optional[float] = None   # Capex / D&A (maintenance benchmark)
    maintenance_capex_est: Optional[float] = None   # Estimated maintenance capex (= D&A)
    growth_capex_est: Optional[float] = None        # Estimated growth capex (= Capex - D&A)
    capex_signal: str = "N/A"


@dataclass
class CapitalReturnMetrics:
    buyback_yield: Optional[float] = None           # Net buybacks / Market Cap
    dividend_yield: Optional[float] = None          # Annual dividends / Market Cap
    total_shareholder_yield: Optional[float] = None # dividend yield + buyback yield
    net_debt_ebitda: Optional[float] = None         # Net Debt / EBITDA
    net_debt_change_yoy: Optional[float] = None     # Change in net debt (absolute $)
    leverage_signal: str = "N/A"


@dataclass
class CapitalEfficiencyResult:
    working_capital: WorkingCapitalMetrics = field(default_factory=WorkingCapitalMetrics)
    capex: CapexMetrics = field(default_factory=CapexMetrics)
    capital_return: CapitalReturnMetrics = field(default_factory=CapitalReturnMetrics)
    quality_flags: list = field(default_factory=list)
    overall_score: float = 0.0
    signal: str = "AVERAGE"
    summary: str = ""


# ---------------------------------------------------------------------------
# Low-level helpers (self-contained, no cross-module imports)
# ---------------------------------------------------------------------------

def _safe_float(val) -> Optional[float]:
    """Convert scalar to float; return None on failure/NaN/Inf."""
    try:
        if val is None:
            return None
        f = float(val)
        return None if (np.isnan(f) or np.isinf(f)) else f
    except (TypeError, ValueError):
        return None


def _get(df: pd.DataFrame, *names: str) -> Optional[pd.Series]:
    """Return the first matching row Series from df, trying case-insensitive fallback."""
    if df is None or df.empty:
        return None
    for name in names:
        if name in df.index:
            return df.loc[name]
        for idx_label in df.index:
            if str(idx_label).lower() == name.lower():
                return df.loc[idx_label]
    return None


def _val(df: pd.DataFrame, *names: str, col: int = 0) -> Optional[float]:
    """Return a single float cell from the first matching row at column index `col`."""
    series = _get(df, *names)
    if series is None:
        return None
    try:
        return _safe_float(series.iloc[col])
    except (IndexError, TypeError):
        return None


def _val_at(series: pd.Series, col: int) -> Optional[float]:
    try:
        return _safe_float(series.iloc[col])
    except (IndexError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _score_ccc_change(ccc_change: Optional[float]) -> float:
    """Negative change (improved) is better."""
    if ccc_change is None:
        return 0.0
    if ccc_change < 0:
        return 0.5          # improved
    if ccc_change <= 15:
        return 0.0          # stable
    if ccc_change <= 30:
        return -0.5         # worsened moderately
    return -1.0             # significantly worsened


def _label_ccc_change(ccc_change: Optional[float]) -> str:
    if ccc_change is None:
        return "N/A"
    if ccc_change < 0:
        return "IMPROVING"
    if ccc_change <= 15:
        return "STABLE"
    if ccc_change <= 30:
        return "WORSENING"
    return "DETERIORATING"


def _score_capex_dna(ratio: Optional[float]) -> tuple[float, str]:
    """Capex / D&A ratio scoring."""
    if ratio is None:
        return 0.0, "N/A"
    if ratio < 0.5:
        return -0.5, "UNDERINVESTING"
    if ratio <= 1.5:
        return 0.5, "HEALTHY"
    if ratio <= 2.5:
        return 0.0, "GROWTH INVESTING"
    return -0.5, "HEAVY EXPANSION"


def _score_capex_revenue(ratio: Optional[float]) -> str:
    if ratio is None:
        return "N/A"
    pct = ratio * 100
    if pct < 3:
        return "ASSET-LIGHT"
    if pct < 8:
        return "MODERATE"
    return "CAPITAL-INTENSIVE"


def _score_tsy(tsy: Optional[float]) -> float:
    """Total shareholder yield scoring."""
    if tsy is None:
        return 0.0
    if tsy > 0.06:
        return 1.0
    if tsy > 0.03:
        return 0.5
    if tsy >= 0:
        return 0.0
    return -0.5             # negative = net dilution outweighs dividends


def _score_nd_ebitda(ratio: Optional[float]) -> tuple[float, str]:
    """Net Debt / EBITDA scoring."""
    if ratio is None:
        return 0.0, "N/A"
    if ratio < 0:
        # Net cash position
        return 1.0, "NET CASH"
    if ratio < 1:
        return 1.0, "VERY LOW"
    if ratio < 2:
        return 0.5, "LOW"
    if ratio < 3:
        return 0.0, "MODERATE"
    if ratio < 4:
        return -0.5, "HIGH"
    return -1.0, "VERY HIGH"


# ---------------------------------------------------------------------------
# Main analyzer
# ---------------------------------------------------------------------------

class CapitalEfficiencyAnalyzer:
    """Scores capital efficiency from financial statement DataFrames."""

    _WC_WEIGHT = 0.30
    _CAPEX_WEIGHT = 0.30
    _RETURN_WEIGHT = 0.40

    def analyze(
        self,
        info: dict,
        income_stmt: pd.DataFrame,
        balance_sheet: pd.DataFrame,
        cashflow: pd.DataFrame,
    ) -> CapitalEfficiencyResult:
        result = CapitalEfficiencyResult()

        try:
            result.working_capital = self._compute_working_capital(income_stmt, balance_sheet)
        except Exception as exc:
            logger.debug("Working capital failed: %s", exc)

        try:
            result.capex = self._compute_capex(income_stmt, cashflow)
        except Exception as exc:
            logger.debug("Capex analysis failed: %s", exc)

        try:
            result.capital_return = self._compute_capital_return(info, income_stmt, balance_sheet, cashflow)
        except Exception as exc:
            logger.debug("Capital return failed: %s", exc)

        self._aggregate(result)
        return result

    # ------------------------------------------------------------------
    # Working Capital
    # ------------------------------------------------------------------

    def _compute_working_capital(
        self,
        income_stmt: pd.DataFrame,
        balance_sheet: pd.DataFrame,
    ) -> WorkingCapitalMetrics:
        wc = WorkingCapitalMetrics()

        # --- Current period values ---
        rev = _val(income_stmt, "Total Revenue", "Revenue", "Net Revenue")
        cogs = _val(
            income_stmt,
            "Cost Of Revenue",
            "Cost of Revenue",
            "Cost Of Goods Sold",
            "Cost Of Goods And Services Sold",
        )
        ar = _val(
            balance_sheet,
            "Accounts Receivable",
            "Net Receivables",
            "Receivables",
            "Gross Accounts Receivable",
        )
        inventory = _val(
            balance_sheet,
            "Inventory",
            "Inventories",
            "Net Inventory",
            "Finished Goods",
        )
        ap = _val(
            balance_sheet,
            "Accounts Payable",
            "Payables",
            "Accounts Payable And Accrued Liabilities",
        )

        # --- Compute current DSO / DIO / DPO / CCC ---
        dso_curr: Optional[float] = None
        dio_curr: Optional[float] = None
        dpo_curr: Optional[float] = None

        if ar is not None and rev and rev != 0:
            dso_curr = (ar / rev) * 365

        if inventory is not None and cogs and cogs != 0:
            dio_curr = (abs(inventory) / abs(cogs)) * 365
        else:
            # Service company — no inventory
            dio_curr = 0.0

        if ap is not None and cogs and cogs != 0:
            dpo_curr = (abs(ap) / abs(cogs)) * 365

        wc.dso = _safe_float(dso_curr)
        wc.dio = _safe_float(dio_curr)
        wc.dpo = _safe_float(dpo_curr)

        if dso_curr is not None and dio_curr is not None and dpo_curr is not None:
            wc.ccc = _safe_float(dso_curr + dio_curr - dpo_curr)

        # --- Prior-year values for YoY change ---
        rev_prev = _val(income_stmt, "Total Revenue", "Revenue", "Net Revenue", col=1)
        cogs_prev = _val(
            income_stmt,
            "Cost Of Revenue", "Cost of Revenue",
            "Cost Of Goods Sold", "Cost Of Goods And Services Sold",
            col=1,
        )
        ar_prev = _val(
            balance_sheet,
            "Accounts Receivable", "Net Receivables", "Receivables",
            col=1,
        )
        inventory_prev = _val(
            balance_sheet,
            "Inventory", "Inventories", "Net Inventory",
            col=1,
        )
        ap_prev = _val(
            balance_sheet,
            "Accounts Payable", "Payables",
            "Accounts Payable And Accrued Liabilities",
            col=1,
        )

        dso_prev: Optional[float] = None
        dio_prev: Optional[float] = None
        dpo_prev: Optional[float] = None

        if ar_prev is not None and rev_prev and rev_prev != 0:
            dso_prev = (ar_prev / rev_prev) * 365
        if inventory_prev is not None and cogs_prev and cogs_prev != 0:
            dio_prev = (abs(inventory_prev) / abs(cogs_prev)) * 365
        else:
            dio_prev = 0.0
        if ap_prev is not None and cogs_prev and cogs_prev != 0:
            dpo_prev = (abs(ap_prev) / abs(cogs_prev)) * 365

        if dso_curr is not None and dso_prev is not None:
            wc.dso_yoy_change = _safe_float(dso_curr - dso_prev)
        if dio_curr is not None and dio_prev is not None:
            wc.dio_yoy_change = _safe_float(dio_curr - dio_prev)

        # CCC YoY
        if (dso_curr is not None and dio_curr is not None and dpo_curr is not None
                and dso_prev is not None and dio_prev is not None and dpo_prev is not None):
            ccc_prev = dso_prev + dio_prev - dpo_prev
            wc.ccc_yoy_change = _safe_float(wc.ccc - ccc_prev)  # type: ignore[operator]

        wc.working_capital_signal = _label_ccc_change(wc.ccc_yoy_change)
        return wc

    # ------------------------------------------------------------------
    # Capex Analysis
    # ------------------------------------------------------------------

    def _compute_capex(
        self,
        income_stmt: pd.DataFrame,
        cashflow: pd.DataFrame,
    ) -> CapexMetrics:
        cx = CapexMetrics()

        revenue = _val(income_stmt, "Total Revenue", "Revenue", "Net Revenue")

        # Capex is usually negative in cashflow; take absolute value
        capex_raw = _val(
            cashflow,
            "Capital Expenditure",
            "Capital Expenditures",
            "Purchase Of Property Plant And Equipment",
            "Purchases Of Property Plant And Equipment",
            "Capital Expenditures Reported",
        )
        capex = abs(capex_raw) if capex_raw is not None else None

        # D&A — try multiple sources
        dna = _val(
            cashflow,
            "Depreciation And Amortization",
            "Depreciation Amortization Depletion",
            "Depreciation",
        )
        if dna is None:
            dna = _val(
                income_stmt,
                "Reconciled Depreciation",
                "Depreciation And Amortization",
                "Depreciation Amortization And Depletion",
            )
        dna = abs(dna) if dna is not None else None

        if capex is not None and revenue and revenue != 0:
            cx.capex_to_revenue = _safe_float(capex / revenue)

        if capex is not None and dna and dna != 0:
            cx.capex_to_depreciation = _safe_float(capex / dna)

        if dna is not None:
            cx.maintenance_capex_est = _safe_float(dna)

        if capex is not None and dna is not None:
            cx.growth_capex_est = _safe_float(max(0.0, capex - dna))

        # Signal uses capex/D&A as primary classifier; fallback to revenue intensity
        if cx.capex_to_depreciation is not None:
            _, label = _score_capex_dna(cx.capex_to_depreciation)
            cx.capex_signal = label
        elif cx.capex_to_revenue is not None:
            cx.capex_signal = _score_capex_revenue(cx.capex_to_revenue)

        return cx

    # ------------------------------------------------------------------
    # Capital Return
    # ------------------------------------------------------------------

    def _compute_capital_return(
        self,
        info: dict,
        income_stmt: pd.DataFrame,
        balance_sheet: pd.DataFrame,
        cashflow: pd.DataFrame,
    ) -> CapitalReturnMetrics:
        cr = CapitalReturnMetrics()

        market_cap = _safe_float(info.get("marketCap"))

        # --- Buyback Yield ---
        buyback_raw = _val(
            cashflow,
            "Repurchase Of Capital Stock",
            "Common Stock Repurchased",
            "Repurchase Of Common Stock",
            "Buyback Of Stock",
            "Purchase Of Business",
            "Common Stock Payments",
        )
        if buyback_raw is not None and market_cap and market_cap > 0:
            cr.buyback_yield = _safe_float(abs(buyback_raw) / market_cap)

        # --- Dividend Yield ---
        div_yield_info = _safe_float(info.get("dividendYield"))
        if div_yield_info is not None:
            # yfinance returns as decimal (e.g. 0.023 = 2.3%)
            cr.dividend_yield = div_yield_info if div_yield_info < 1 else div_yield_info / 100
        else:
            # Try to compute from cashflow
            div_paid = _val(
                cashflow,
                "Payment Of Dividends",
                "Cash Dividends Paid",
                "Common Stock Dividend Paid",
                "Dividends Paid",
            )
            if div_paid is not None and market_cap and market_cap > 0:
                cr.dividend_yield = _safe_float(abs(div_paid) / market_cap)

        # --- Total Shareholder Yield ---
        if cr.buyback_yield is not None or cr.dividend_yield is not None:
            bk = cr.buyback_yield or 0.0
            dv = cr.dividend_yield or 0.0
            cr.total_shareholder_yield = _safe_float(bk + dv)

        # --- Net Debt / EBITDA ---
        # Net Debt
        total_debt = _val(
            balance_sheet,
            "Total Debt",
            "Long Term Debt And Capital Lease Obligation",
            "Long Term Debt",
        )
        short_term_debt = _val(
            balance_sheet,
            "Short Long Term Debt",
            "Current Debt",
            "Short Term Borrowings",
            "Current Portion Of Long Term Debt",
        )
        cash = _val(
            balance_sheet,
            "Cash And Cash Equivalents",
            "Cash Cash Equivalents And Short Term Investments",
            "Cash And Short Term Investments",
        )

        if total_debt is not None:
            total_debt_val = total_debt + (short_term_debt or 0.0)
        elif short_term_debt is not None:
            total_debt_val = short_term_debt
        else:
            total_debt_val = None

        net_debt_curr: Optional[float] = None
        if total_debt_val is not None and cash is not None:
            net_debt_curr = total_debt_val - cash
        elif total_debt_val is not None:
            net_debt_curr = total_debt_val

        # EBITDA — prefer info dict, else construct from statements
        ebitda: Optional[float] = _safe_float(info.get("ebitda"))
        if ebitda is None:
            ebit = _val(income_stmt, "EBIT", "Operating Income", "Operating Income Loss")
            dna_cf = _val(
                cashflow,
                "Depreciation And Amortization",
                "Depreciation Amortization Depletion",
                "Depreciation",
            )
            if ebit is not None and dna_cf is not None:
                ebitda = ebit + abs(dna_cf)

        if net_debt_curr is not None and ebitda and ebitda != 0:
            cr.net_debt_ebitda = _safe_float(net_debt_curr / ebitda)

        # Net debt YoY change
        total_debt_prev = _val(
            balance_sheet,
            "Total Debt",
            "Long Term Debt And Capital Lease Obligation",
            "Long Term Debt",
            col=1,
        )
        cash_prev = _val(
            balance_sheet,
            "Cash And Cash Equivalents",
            "Cash Cash Equivalents And Short Term Investments",
            "Cash And Short Term Investments",
            col=1,
        )
        if total_debt_prev is not None and cash_prev is not None and net_debt_curr is not None:
            net_debt_prev = total_debt_prev - cash_prev
            cr.net_debt_change_yoy = _safe_float(net_debt_curr - net_debt_prev)

        # Leverage signal
        _, lev_label = _score_nd_ebitda(cr.net_debt_ebitda)
        cr.leverage_signal = lev_label

        return cr

    # ------------------------------------------------------------------
    # Aggregate: flags, score, signal, summary
    # ------------------------------------------------------------------

    def _aggregate(self, result: CapitalEfficiencyResult) -> None:
        wc = result.working_capital
        cx = result.capex
        cr = result.capital_return

        # --- Sub-scores ---
        wc_score = _score_ccc_change(wc.ccc_yoy_change)

        capex_score, _ = _score_capex_dna(cx.capex_to_depreciation)
        # If capex/D&A not available, fall back to capex/revenue neutral
        if cx.capex_to_depreciation is None:
            capex_score = 0.0

        tsy_score = _score_tsy(cr.total_shareholder_yield)
        nd_score, _ = _score_nd_ebitda(cr.net_debt_ebitda)

        # Capital return component = blend of TSY and leverage
        capital_return_score = (tsy_score * 0.5) + (nd_score * 0.5)

        # Determine which components have data
        scored_components: list[tuple[float, float]] = []
        if wc.ccc_yoy_change is not None or wc.ccc is not None:
            scored_components.append((wc_score, self._WC_WEIGHT))
        if cx.capex_to_revenue is not None or cx.capex_to_depreciation is not None:
            scored_components.append((capex_score, self._CAPEX_WEIGHT))
        if cr.total_shareholder_yield is not None or cr.net_debt_ebitda is not None:
            scored_components.append((capital_return_score, self._RETURN_WEIGHT))

        if scored_components:
            total_weight = sum(w for _, w in scored_components)
            overall = sum(sc * w for sc, w in scored_components) / total_weight
        else:
            overall = 0.0

        result.overall_score = round(float(overall), 4)

        if overall >= 0.4:
            result.signal = "HIGHLY EFFICIENT"
        elif overall >= 0.1:
            result.signal = "ABOVE AVERAGE"
        elif overall >= -0.1:
            result.signal = "AVERAGE"
        elif overall >= -0.4:
            result.signal = "BELOW AVERAGE"
        else:
            result.signal = "CAPITAL INTENSIVE / INEFFICIENT"

        # --- Quality flags ---
        flags: list[str] = []

        # CCC flag
        if wc.ccc is not None:
            ccc_str = f"{wc.ccc:.0f} days"
            if wc.ccc_yoy_change is not None:
                direction = "improved" if wc.ccc_yoy_change < 0 else "worsened"
                days_abs = abs(wc.ccc_yoy_change)
                if wc.ccc_yoy_change < 0:
                    flags.append(
                        f"✅ CCC {ccc_str} — improved {days_abs:.0f} days YoY; working capital efficiency gaining"
                    )
                elif wc.ccc_yoy_change > 30:
                    flags.append(
                        f"🚨 CCC {ccc_str} — {direction} {days_abs:.0f} days YoY; meaningful working capital deterioration"
                    )
                elif wc.ccc_yoy_change > 15:
                    flags.append(
                        f"⚠️ CCC {ccc_str} — {direction} {days_abs:.0f} days YoY; watch for cash cycle pressure"
                    )
                else:
                    flags.append(f"➡️ CCC {ccc_str} — stable working capital cycle")
            else:
                if wc.ccc < 0:
                    flags.append(f"✅ Negative CCC ({ccc_str}) — business collects cash before paying suppliers")
                elif wc.ccc < 30:
                    flags.append(f"✅ CCC {ccc_str} — tight, efficient cash conversion")
                else:
                    flags.append(f"➡️ CCC {ccc_str} — within typical operating range")

        # DSO / DPO flags
        if wc.dso is not None and wc.dso_yoy_change is not None:
            if wc.dso_yoy_change > 10:
                flags.append(
                    f"⚠️ DSO expanded {wc.dso_yoy_change:.0f} days YoY to {wc.dso:.0f} days — customers paying slower"
                )
        if wc.dpo is not None and wc.dpo > 90:
            flags.append(f"⚠️ DPO {wc.dpo:.0f} days — aggressive supplier payment extension; monitor relationship risk")

        # Capex intensity flag
        if cx.capex_to_revenue is not None:
            pct = cx.capex_to_revenue * 100
            capex_label = _score_capex_revenue(cx.capex_to_revenue)
            if capex_label == "ASSET-LIGHT":
                flags.append(f"✅ Capex/Revenue {pct:.1f}% — asset-light model with high incremental margins")
            elif capex_label == "MODERATE":
                flags.append(f"➡️ Capex/Revenue {pct:.1f}% — moderate capital intensity; balanced reinvestment")
            else:
                flags.append(f"⚠️ Capex/Revenue {pct:.1f}% — capital-intensive; FCF yield depressed by heavy reinvestment")

        # Capex vs D&A flag
        if cx.capex_to_depreciation is not None:
            ratio = cx.capex_to_depreciation
            _, dna_label = _score_capex_dna(ratio)
            if dna_label == "UNDERINVESTING":
                flags.append(
                    f"⚠️ Capex/D&A {ratio:.2f}x — below maintenance threshold; asset base may be deteriorating"
                )
            elif dna_label == "HEALTHY":
                flags.append(f"✅ Capex/D&A {ratio:.2f}x — healthy reinvestment covering depreciation plus growth")
            elif dna_label == "HEAVY EXPANSION":
                flags.append(
                    f"⚠️ Capex/D&A {ratio:.2f}x — aggressive expansion capex; monitor return on invested capital"
                )

        # Shareholder yield flag
        if cr.total_shareholder_yield is not None:
            tsy_pct = cr.total_shareholder_yield * 100
            bk_pct = (cr.buyback_yield or 0.0) * 100
            dv_pct = (cr.dividend_yield or 0.0) * 100
            if tsy_pct > 6:
                flags.append(
                    f"✅ Total shareholder yield {tsy_pct:.1f}% "
                    f"(dividends {dv_pct:.1f}% + buybacks {bk_pct:.1f}%) — excellent capital return"
                )
            elif tsy_pct > 3:
                flags.append(
                    f"✅ Total shareholder yield {tsy_pct:.1f}% — solid capital return to shareholders"
                )
            elif tsy_pct >= 0:
                flags.append(
                    f"➡️ Total shareholder yield {tsy_pct:.1f}% — modest capital return; reinvesting for growth"
                )
            else:
                flags.append(
                    f"⚠️ Negative shareholder yield {tsy_pct:.1f}% — net equity issuance outweighs capital return"
                )

        # Net Debt / EBITDA flag
        if cr.net_debt_ebitda is not None:
            ratio = cr.net_debt_ebitda
            _, nd_label = _score_nd_ebitda(ratio)
            if nd_label in ("NET CASH", "VERY LOW"):
                flags.append(
                    f"✅ Net Debt/EBITDA {ratio:.1f}x — {nd_label.lower()} position provides financial flexibility"
                )
            elif nd_label == "LOW":
                flags.append(f"✅ Net Debt/EBITDA {ratio:.1f}x — conservative leverage; ample debt capacity")
            elif nd_label == "HIGH":
                flags.append(
                    f"⚠️ Net Debt/EBITDA {ratio:.1f}x — elevated leverage; debt service limits financial flexibility"
                )
            elif nd_label == "VERY HIGH":
                flags.append(
                    f"🚨 Net Debt/EBITDA {ratio:.1f}x — excessive leverage; deleveraging required to reduce risk"
                )

        # Net debt trend flag
        if cr.net_debt_change_yoy is not None:
            change_b = cr.net_debt_change_yoy / 1e9
            if cr.net_debt_change_yoy < 0:
                flags.append(f"✅ Net debt fell ${abs(change_b):.1f}B YoY — balance sheet deleveraging in progress")
            elif cr.net_debt_change_yoy > 0 and (cr.net_debt_ebitda or 0) > 3:
                flags.append(f"⚠️ Net debt rose ${change_b:.1f}B YoY despite already elevated leverage")

        result.quality_flags = flags[:5]

        # --- Summary ---
        parts: list[str] = []
        if wc.ccc is not None:
            parts.append(f"CCC of {wc.ccc:.0f} days ({wc.working_capital_signal.lower()})")
        if cx.capex_to_revenue is not None:
            parts.append(f"capex intensity {cx.capex_to_revenue*100:.1f}% of revenue ({cx.capex_signal.lower().replace('_', ' ')})")
        if cr.total_shareholder_yield is not None:
            parts.append(f"total shareholder yield of {cr.total_shareholder_yield*100:.1f}%")
        if cr.net_debt_ebitda is not None:
            parts.append(f"net debt/EBITDA of {cr.net_debt_ebitda:.1f}x")

        detail = ", ".join(parts[:3]) if parts else "limited data available"

        if overall >= 0.1:
            outlook = "Capital allocation appears disciplined, supporting long-run shareholder value creation."
        elif overall >= -0.1:
            outlook = "Capital allocation is adequate, though specific areas warrant monitoring."
        else:
            outlook = "Capital efficiency shows strain; investors should scrutinize reinvestment returns and leverage trajectory."

        result.summary = (
            f"Capital efficiency scores as {result.signal} (overall {result.overall_score:+.2f}), "
            f"with {detail}. {outlook}"
        )
