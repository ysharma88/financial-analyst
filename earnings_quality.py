"""Earnings quality analysis — 6 institutional-grade signal layers.

Catches the gap between reported earnings and economic reality:
  1. FCF Conversion Rate  (FCF / Net Income) — earnings purity
  2. Sloan's Accruals Ratio — predicts future earnings disappointments
  3. AR Growth vs Revenue Growth divergence — revenue recognition quality
  4. EPS Surprise Magnitude & Trend — Post-Earnings Announcement Drift signal
  5. SBC as % of Revenue / FCF — real cost of non-GAAP earnings
  6. Share Dilution Rate — net equity issuance drag on per-share value
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class EarningsQualityResult:
    # FCF Conversion
    fcf_conversion_rate: Optional[float] = None        # FCF / Net Income
    fcf_conversion_signal: str = "N/A"

    # Sloan Accruals
    sloan_accruals_ratio: Optional[float] = None       # (Net Income - CFO - CFI) / Avg Total Assets
    sloan_signal: str = "N/A"

    # AR vs Revenue Growth
    ar_growth_yoy: Optional[float] = None              # % change in accounts receivable
    revenue_growth_yoy: Optional[float] = None         # % change in revenue
    ar_revenue_divergence: Optional[float] = None      # AR growth - Revenue growth (pp)
    ar_divergence_signal: str = "N/A"

    # EPS Surprises
    surprise_magnitudes: list = field(default_factory=list)   # list of % surprises (recent first)
    avg_surprise_pct: Optional[float] = None
    surprise_trend: str = "N/A"                        # "IMPROVING", "STABLE", "DETERIORATING"

    # SBC
    sbc_pct_revenue: Optional[float] = None
    sbc_pct_fcf: Optional[float] = None
    sbc_signal: str = "N/A"

    # Dilution
    dilution_rate_yoy: Optional[float] = None          # (Shares_curr / Shares_prev - 1) * 100
    dilution_signal: str = "N/A"

    # Aggregate
    quality_flags: list = field(default_factory=list)
    overall_score: float = 0.0                         # -1.0 to +1.0
    signal: str = "AVERAGE"
    summary: str = ""


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _safe_float(val) -> Optional[float]:
    """Convert a scalar to float, returning None on failure."""
    try:
        if val is None:
            return None
        f = float(val)
        return None if (np.isnan(f) or np.isinf(f)) else f
    except (TypeError, ValueError):
        return None


def _get(df: pd.DataFrame, *names: str) -> Optional[pd.Series]:
    """Try multiple row-name variants; return the first matching Series or None."""
    if df is None or df.empty:
        return None
    for name in names:
        if name in df.index:
            return df.loc[name]
        # Case-insensitive fallback
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
        raw = series.iloc[col]
        return _safe_float(raw)
    except (IndexError, TypeError):
        return None


def _val_at(series: pd.Series, col: int) -> Optional[float]:
    """Extract a float from a Series at position `col`."""
    try:
        return _safe_float(series.iloc[col])
    except (IndexError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _score_fcf_conversion(rate: Optional[float]) -> tuple[float, str]:
    if rate is None:
        return 0.0, "N/A"
    if rate >= 0.90:
        return 1.0, "STRONG"
    if rate >= 0.70:
        return 0.5, "GOOD"
    if rate >= 0.50:
        return 0.0, "MODERATE"
    if rate >= 0.30:
        return -0.5, "WEAK"
    return -1.0, "POOR"


def _score_sloan(ratio: Optional[float]) -> tuple[float, str]:
    """Sloan accruals ratio as decimal (e.g. 0.05 = 5%)."""
    if ratio is None:
        return 0.0, "N/A"
    pct = ratio * 100  # convert to percentage points for band comparison
    if pct < -5:
        return 1.0, "STRONG"
    if pct <= 5:
        return 0.3, "MODERATE"
    if pct <= 15:
        return -0.5, "WEAK"
    return -1.0, "POOR"


def _score_ar_divergence(divergence_pp: Optional[float]) -> tuple[float, str]:
    """divergence_pp = AR growth % - Revenue growth % (percentage points)."""
    if divergence_pp is None:
        return 0.0, "N/A"
    if divergence_pp < 5:
        return 0.5, "STRONG"
    if divergence_pp < 15:
        return 0.0, "MODERATE"
    if divergence_pp < 30:
        return -0.5, "WEAK"
    return -1.0, "POOR"


def _score_surprise(avg_surprise: Optional[float], surprise_list: list) -> tuple[float, str, str]:
    """Returns (score, signal_label, trend_label)."""
    if avg_surprise is None:
        return 0.0, "N/A", "N/A"

    # Trend: compare first half vs second half (recent = index 0)
    trend = "STABLE"
    if len(surprise_list) >= 4:
        recent_avg = np.mean(surprise_list[:2])
        older_avg = np.mean(surprise_list[2:4])
        if recent_avg > older_avg + 2:
            trend = "IMPROVING"
        elif recent_avg < older_avg - 2:
            trend = "DETERIORATING"

    # Check for consistent misses
    misses = [s for s in surprise_list if s < 0]
    consistent_miss = len(misses) >= 3 and len(misses) / max(len(surprise_list), 1) >= 0.75

    if consistent_miss:
        return -1.0, "CONSISTENT MISS", trend
    if avg_surprise > 10:
        return 1.0, "STRONG"  , trend
    if avg_surprise > 5:
        return 0.5, "GOOD", trend
    if avg_surprise >= 0:
        return 0.0, "MODERATE", trend
    return -0.5, "WEAK", trend


def _score_sbc(sbc_pct_fcf: Optional[float]) -> tuple[float, str]:
    if sbc_pct_fcf is None:
        return 0.0, "N/A"
    if sbc_pct_fcf < 0.15:
        return 1.0, "STRONG"
    if sbc_pct_fcf < 0.40:
        return 0.3, "MODERATE"
    if sbc_pct_fcf < 0.75:
        return -0.3, "ELEVATED"
    return -1.0, "EXCESSIVE"


def _score_dilution(rate_pct: Optional[float]) -> tuple[float, str]:
    """rate_pct: positive = dilution, negative = buybacks."""
    if rate_pct is None:
        return 0.0, "N/A"
    if rate_pct < -1:
        return 1.0, "BUYBACK"
    if rate_pct <= 1:
        return 0.3, "STABLE"
    if rate_pct <= 3:
        return -0.3, "MILD DILUTION"
    return -1.0, "HIGH DILUTION"


# ---------------------------------------------------------------------------
# Main analyzer
# ---------------------------------------------------------------------------

class EarningsQualityAnalyzer:
    """Computes 6-layer earnings quality score from financial statement DataFrames."""

    # Weights for overall score
    _WEIGHTS = {
        "fcf": 0.25,
        "sloan": 0.20,
        "ar": 0.15,
        "surprise": 0.20,
        "sbc": 0.10,
        "dilution": 0.10,
    }

    def analyze(
        self,
        info: dict,
        income_stmt: pd.DataFrame,
        balance_sheet: pd.DataFrame,
        cashflow: pd.DataFrame,
        earnings_history=None,
    ) -> EarningsQualityResult:
        result = EarningsQualityResult()

        try:
            self._compute_fcf_conversion(result, income_stmt, cashflow)
        except Exception as exc:
            logger.debug("FCF conversion failed: %s", exc)

        try:
            self._compute_sloan(result, income_stmt, balance_sheet, cashflow)
        except Exception as exc:
            logger.debug("Sloan accruals failed: %s", exc)

        try:
            self._compute_ar_divergence(result, income_stmt, balance_sheet)
        except Exception as exc:
            logger.debug("AR divergence failed: %s", exc)

        try:
            self._compute_eps_surprises(result, earnings_history)
        except Exception as exc:
            logger.debug("EPS surprises failed: %s", exc)

        try:
            self._compute_sbc(result, income_stmt, cashflow)
        except Exception as exc:
            logger.debug("SBC failed: %s", exc)

        try:
            self._compute_dilution(result, balance_sheet, info)
        except Exception as exc:
            logger.debug("Dilution failed: %s", exc)

        self._aggregate(result)
        return result

    # ------------------------------------------------------------------
    # Layer 1: FCF Conversion Rate
    # ------------------------------------------------------------------

    def _compute_fcf_conversion(
        self,
        result: EarningsQualityResult,
        income_stmt: pd.DataFrame,
        cashflow: pd.DataFrame,
    ) -> None:
        net_income = _val(
            income_stmt,
            "Net Income",
            "NetIncome",
            "Net Income Common Stockholders",
            "Net Income Applicable To Common Shares",
        )
        # FCF = Operating CF - Capex
        ocf = _val(
            cashflow,
            "Operating Cash Flow",
            "Cash Flow From Continuing Operating Activities",
            "Total Cash From Operating Activities",
        )
        capex = _val(
            cashflow,
            "Capital Expenditure",
            "Capital Expenditures",
            "Purchase Of Property Plant And Equipment",
            "Purchases Of Property Plant And Equipment",
        )

        if net_income is None or ocf is None:
            return

        capex_val = capex if capex is not None else 0.0
        # Capex is usually negative in cashflow statements; make it negative for subtraction
        if capex_val > 0:
            capex_val = -capex_val
        fcf = ocf + capex_val  # ocf - |capex|

        if net_income == 0:
            return

        rate = fcf / net_income
        result.fcf_conversion_rate = _safe_float(rate)

        score, label = _score_fcf_conversion(result.fcf_conversion_rate)
        result.fcf_conversion_signal = label

    # ------------------------------------------------------------------
    # Layer 2: Sloan's Accruals Ratio
    # ------------------------------------------------------------------

    def _compute_sloan(
        self,
        result: EarningsQualityResult,
        income_stmt: pd.DataFrame,
        balance_sheet: pd.DataFrame,
        cashflow: pd.DataFrame,
    ) -> None:
        net_income = _val(
            income_stmt,
            "Net Income",
            "NetIncome",
            "Net Income Common Stockholders",
        )
        ocf = _val(
            cashflow,
            "Operating Cash Flow",
            "Cash Flow From Continuing Operating Activities",
            "Total Cash From Operating Activities",
        )
        # Investment CF (usually negative)
        icf = _val(
            cashflow,
            "Investing Cash Flow",
            "Cash Flow From Continuing Investing Activities",
            "Total Cash From Investing Activities",
        )

        ta_series = _get(
            balance_sheet,
            "Total Assets",
        )
        if ta_series is None or len(ta_series) < 2:
            return
        ta_curr = _val_at(ta_series, 0)
        ta_prev = _val_at(ta_series, 1)

        if net_income is None or ocf is None or ta_curr is None or ta_prev is None:
            return

        avg_assets = (ta_curr + ta_prev) / 2.0
        if avg_assets == 0:
            return

        icf_val = icf if icf is not None else 0.0
        # Sloan accruals = Net Income - Operating CF - Investing CF
        accruals = net_income - ocf - icf_val
        ratio = accruals / avg_assets

        result.sloan_accruals_ratio = _safe_float(ratio)
        score, label = _score_sloan(result.sloan_accruals_ratio)
        result.sloan_signal = label

    # ------------------------------------------------------------------
    # Layer 3: AR Growth vs Revenue Growth
    # ------------------------------------------------------------------

    def _compute_ar_divergence(
        self,
        result: EarningsQualityResult,
        income_stmt: pd.DataFrame,
        balance_sheet: pd.DataFrame,
    ) -> None:
        rev_series = _get(income_stmt, "Total Revenue", "Revenue", "Net Revenue")
        ar_series = _get(
            balance_sheet,
            "Accounts Receivable",
            "Net Receivables",
            "Receivables",
            "Gross Accounts Receivable",
        )

        if rev_series is None or ar_series is None:
            return
        if len(rev_series) < 2 or len(ar_series) < 2:
            return

        rev_curr = _val_at(rev_series, 0)
        rev_prev = _val_at(rev_series, 1)
        ar_curr = _val_at(ar_series, 0)
        ar_prev = _val_at(ar_series, 1)

        if None in (rev_curr, rev_prev, ar_curr, ar_prev):
            return
        if rev_prev == 0 or ar_prev == 0:
            return

        rev_growth_pct = (rev_curr / rev_prev - 1) * 100
        ar_growth_pct = (ar_curr / ar_prev - 1) * 100

        result.revenue_growth_yoy = _safe_float(rev_growth_pct)
        result.ar_growth_yoy = _safe_float(ar_growth_pct)
        divergence = ar_growth_pct - rev_growth_pct
        result.ar_revenue_divergence = _safe_float(divergence)

        score, label = _score_ar_divergence(result.ar_revenue_divergence)
        result.ar_divergence_signal = label

    # ------------------------------------------------------------------
    # Layer 4: EPS Surprise Magnitude & Trend
    # ------------------------------------------------------------------

    def _compute_eps_surprises(
        self,
        result: EarningsQualityResult,
        earnings_history,
    ) -> None:
        if earnings_history is None:
            return

        surprises: list[float] = []

        # Handle DataFrame
        if isinstance(earnings_history, pd.DataFrame) and not earnings_history.empty:
            df = earnings_history

            # Try direct 'Surprise(%)' column first
            for col in ("Surprise(%)", "surprisePct", "surprise_pct", "surprise"):
                if col in df.columns:
                    vals = df[col].dropna().tolist()
                    surprises = [_safe_float(v) for v in vals if _safe_float(v) is not None]
                    if surprises:
                        break

            # Compute from actual vs estimate
            if not surprises:
                actual_col = next(
                    (c for c in df.columns if c in ("Reported EPS", "epsActual", "actual", "EPS Actual")),
                    None,
                )
                est_col = next(
                    (c for c in df.columns if c in ("EPS Estimate", "epsEstimate", "estimate", "EPS Estimate")),
                    None,
                )
                if actual_col and est_col:
                    for _, row in df.iterrows():
                        actual = _safe_float(row[actual_col])
                        est = _safe_float(row[est_col])
                        if actual is not None and est is not None and est != 0:
                            surprises.append((actual - est) / abs(est) * 100)

        # Handle dict / list of dicts
        elif isinstance(earnings_history, (list, dict)):
            rows = earnings_history if isinstance(earnings_history, list) else [earnings_history]
            for row in rows:
                if not isinstance(row, dict):
                    continue
                surprise_val = (
                    row.get("Surprise(%)")
                    or row.get("surprisePct")
                    or row.get("surprise_pct")
                )
                if surprise_val is not None:
                    v = _safe_float(surprise_val)
                    if v is not None:
                        surprises.append(v)
                else:
                    actual = _safe_float(row.get("epsActual") or row.get("Reported EPS"))
                    est = _safe_float(row.get("epsEstimate") or row.get("EPS Estimate"))
                    if actual is not None and est is not None and est != 0:
                        surprises.append((actual - est) / abs(est) * 100)

        if not surprises:
            return

        # Most recent first — limit to last 8 quarters
        surprises = surprises[:8]
        result.surprise_magnitudes = surprises
        result.avg_surprise_pct = _safe_float(float(np.mean(surprises)))

        score, signal, trend = _score_surprise(result.avg_surprise_pct, surprises)
        result.surprise_trend = trend

    # ------------------------------------------------------------------
    # Layer 5: SBC as % of Revenue / FCF
    # ------------------------------------------------------------------

    def _compute_sbc(
        self,
        result: EarningsQualityResult,
        income_stmt: pd.DataFrame,
        cashflow: pd.DataFrame,
    ) -> None:
        sbc = _val(
            cashflow,
            "Stock Based Compensation",
            "Share Based Compensation",
            "StockBasedCompensation",
            "Stock-Based Compensation",
        )
        revenue = _val(income_stmt, "Total Revenue", "Revenue", "Net Revenue")
        ocf = _val(
            cashflow,
            "Operating Cash Flow",
            "Cash Flow From Continuing Operating Activities",
            "Total Cash From Operating Activities",
        )
        capex = _val(
            cashflow,
            "Capital Expenditure",
            "Capital Expenditures",
            "Purchase Of Property Plant And Equipment",
        )

        if sbc is None:
            return

        sbc_abs = abs(sbc)  # SBC is sometimes reported as negative add-back

        if revenue and revenue != 0:
            result.sbc_pct_revenue = _safe_float(sbc_abs / revenue)

        # Compute FCF for ratio
        if ocf is not None:
            capex_val = (capex if capex is not None else 0.0)
            if capex_val > 0:
                capex_val = -capex_val
            fcf = ocf + capex_val
            if fcf > 0:
                result.sbc_pct_fcf = _safe_float(sbc_abs / fcf)

        score, label = _score_sbc(result.sbc_pct_fcf)
        result.sbc_signal = label

    # ------------------------------------------------------------------
    # Layer 6: Share Dilution Rate
    # ------------------------------------------------------------------

    def _compute_dilution(
        self,
        result: EarningsQualityResult,
        balance_sheet: pd.DataFrame,
        info: dict,
    ) -> None:
        shares_series = _get(
            balance_sheet,
            "Ordinary Shares Number",
            "Common Stock Shares Outstanding",
            "Share Issued",
            "Common Stock",
        )

        shares_curr: Optional[float] = None
        shares_prev: Optional[float] = None

        if shares_series is not None and len(shares_series) >= 2:
            shares_curr = _val_at(shares_series, 0)
            shares_prev = _val_at(shares_series, 1)

        # Fallback to info dict
        if shares_curr is None:
            shares_curr = _safe_float(info.get("sharesOutstanding") or info.get("impliedSharesOutstanding"))

        if shares_curr is None or shares_prev is None:
            return
        if shares_prev == 0:
            return

        dilution_pct = (shares_curr / shares_prev - 1) * 100
        result.dilution_rate_yoy = _safe_float(dilution_pct)

        score, label = _score_dilution(result.dilution_rate_yoy)
        result.dilution_signal = label

    # ------------------------------------------------------------------
    # Aggregate: flags, score, signal, summary
    # ------------------------------------------------------------------

    def _aggregate(self, result: EarningsQualityResult) -> None:
        W = self._WEIGHTS
        components: dict[str, tuple[float, float]] = {}  # name -> (score, weight)

        # FCF
        fcf_score, _ = _score_fcf_conversion(result.fcf_conversion_rate)
        if result.fcf_conversion_rate is not None:
            components["fcf"] = (fcf_score, W["fcf"])

        # Sloan
        sloan_score, _ = _score_sloan(result.sloan_accruals_ratio)
        if result.sloan_accruals_ratio is not None:
            components["sloan"] = (sloan_score, W["sloan"])

        # AR divergence
        ar_score, _ = _score_ar_divergence(result.ar_revenue_divergence)
        if result.ar_revenue_divergence is not None:
            components["ar"] = (ar_score, W["ar"])

        # Surprise
        if result.avg_surprise_pct is not None:
            surp_score, _, _ = _score_surprise(result.avg_surprise_pct, result.surprise_magnitudes)
            components["surprise"] = (surp_score, W["surprise"])

        # SBC
        sbc_score, _ = _score_sbc(result.sbc_pct_fcf)
        if result.sbc_pct_fcf is not None:
            components["sbc"] = (sbc_score, W["sbc"])

        # Dilution
        dil_score, _ = _score_dilution(result.dilution_rate_yoy)
        if result.dilution_rate_yoy is not None:
            components["dilution"] = (dil_score, W["dilution"])

        # Weighted average (renormalize weights to what's available)
        if components:
            total_weight = sum(w for _, w in components.values())
            overall = sum(sc * w for sc, w in components.values()) / total_weight
        else:
            overall = 0.0

        result.overall_score = round(float(overall), 4)

        # Signal thresholds
        if overall >= 0.4:
            result.signal = "HIGH QUALITY"
        elif overall >= 0.1:
            result.signal = "ABOVE AVERAGE"
        elif overall >= -0.1:
            result.signal = "AVERAGE"
        elif overall >= -0.4:
            result.signal = "BELOW AVERAGE"
        else:
            result.signal = "LOW QUALITY"

        # Build quality flags
        flags: list[str] = []

        # FCF flag
        if result.fcf_conversion_rate is not None:
            pct = result.fcf_conversion_rate * 100
            if result.fcf_conversion_rate >= 0.90:
                flags.append(f"✅ FCF conversion {pct:.0f}% — earnings backed by strong cash generation")
            elif result.fcf_conversion_rate >= 0.70:
                flags.append(f"✅ FCF conversion {pct:.0f}% — reasonable cash backing for reported earnings")
            elif result.fcf_conversion_rate >= 0.50:
                flags.append(f"⚠️ FCF conversion {pct:.0f}% — moderate accrual component in earnings")
            else:
                flags.append(f"⚠️ FCF conversion {pct:.0f}% — accrual-heavy earnings; real cash significantly lower")

        # Sloan flag
        if result.sloan_accruals_ratio is not None:
            pct = result.sloan_accruals_ratio * 100
            if pct < -5:
                flags.append(f"✅ Sloan accruals {pct:.1f}% — cash flow substantially exceeds reported earnings")
            elif pct <= 5:
                flags.append(f"✅ Sloan accruals {pct:.1f}% — balanced accruals; earnings quality intact")
            elif pct <= 15:
                flags.append(f"⚠️ Sloan accruals {pct:.1f}% — elevated accruals predict potential earnings disappointment")
            else:
                flags.append(f"🚨 Sloan accruals {pct:.1f}% — high manipulation risk; future earnings at risk")

        # AR divergence flag
        if result.ar_revenue_divergence is not None and result.ar_growth_yoy is not None:
            div = result.ar_revenue_divergence
            if div < 5:
                flags.append(
                    f"✅ AR growth ({result.ar_growth_yoy:.1f}%) in line with revenue growth "
                    f"({result.revenue_growth_yoy:.1f}%) — clean revenue recognition"
                )
            elif div < 15:
                flags.append(
                    f"⚠️ AR growing {div:.1f}pp faster than revenue — watch for pull-forward recognition"
                )
            else:
                flags.append(
                    f"🚨 AR growing {div:.1f}pp faster than revenue — aggressive revenue recognition signal"
                )

        # EPS surprise flag
        if result.avg_surprise_pct is not None:
            avg = result.avg_surprise_pct
            if avg > 10:
                flags.append(f"✅ Consistent EPS beats (+{avg:.1f}% avg) — strong guidance management")
            elif avg > 5:
                flags.append(f"✅ Positive EPS surprises (+{avg:.1f}% avg) — modest but reliable beats")
            elif avg >= 0:
                flags.append(f"➡️ EPS surprises near zero ({avg:.1f}% avg) — management guides precisely")
            else:
                flags.append(f"⚠️ Negative EPS surprises ({avg:.1f}% avg) — earnings estimates consistently too high")

        # SBC flag
        if result.sbc_pct_revenue is not None:
            r_pct = result.sbc_pct_revenue * 100
            if result.sbc_pct_fcf is not None:
                f_pct = result.sbc_pct_fcf * 100
                if f_pct < 15:
                    flags.append(f"✅ SBC {r_pct:.1f}% of revenue / {f_pct:.0f}% of FCF — manageable compensation cost")
                elif f_pct < 40:
                    flags.append(f"⚠️ SBC {r_pct:.1f}% of revenue / {f_pct:.0f}% of FCF — non-GAAP earnings materially higher than GAAP")
                else:
                    flags.append(f"🚨 SBC {r_pct:.1f}% of revenue / {f_pct:.0f}% of FCF — non-GAAP earnings significantly overstate economic reality")
            else:
                flags.append(f"➡️ SBC {r_pct:.1f}% of revenue — monitor non-GAAP vs GAAP divergence")

        # Dilution flag
        if result.dilution_rate_yoy is not None:
            d = result.dilution_rate_yoy
            if d < -1:
                flags.append(f"✅ Net share count declining ({d:.1f}% YoY) — buybacks accreting per-share value")
            elif d <= 1:
                flags.append(f"➡️ Share count stable ({d:.1f}% YoY) — minimal dilution drag")
            elif d <= 3:
                flags.append(f"⚠️ Mild share dilution ({d:.1f}% YoY) — modest per-share value drag")
            else:
                flags.append(f"🚨 High share dilution ({d:.1f}% YoY) — significant per-share value erosion")

        # Trim to max 6 most informative flags
        result.quality_flags = flags[:6]

        # Build summary
        component_labels = []
        if result.fcf_conversion_rate is not None:
            component_labels.append(
                f"FCF conversion at {result.fcf_conversion_rate*100:.0f}% ({result.fcf_conversion_signal.lower()})"
            )
        if result.sloan_accruals_ratio is not None:
            component_labels.append(
                f"Sloan accruals at {result.sloan_accruals_ratio*100:.1f}%"
            )
        if result.avg_surprise_pct is not None:
            component_labels.append(
                f"avg EPS surprise of {result.avg_surprise_pct:+.1f}%"
            )

        detail = "; ".join(component_labels[:3]) if component_labels else "limited data available"
        result.summary = (
            f"Earnings quality scores as {result.signal} (overall {result.overall_score:+.2f}), "
            f"with {detail}. "
            f"{'Key risk: accrual-heavy earnings may not sustain without cash flow improvement.' if overall < -0.1 else 'Earnings appear broadly supported by underlying cash generation.'}"
        )
