"""Credit market conditions — cross-asset signals for equity risk assessment.

Uses bond ETF market data (all via yfinance, no API keys needed):
  HYG  — iShares iBoxx High Yield Corporate Bond ETF (HY credit proxy)
  LQD  — iShares iBoxx Investment Grade Corporate Bond ETF (IG credit proxy)
  IEF  — iShares 7-10 Year Treasury Bond ETF (duration benchmark)
  TIP  — iShares TIPS Bond ETF (inflation-protected)
  AGG  — iShares Core US Aggregate Bond ETF
  SPY  — SPDR S&P 500 ETF (equity benchmark)

Signals derived:
  HY Credit Spread proxy: HYG performance vs IEF (widening = equity bearish)
  IG Credit Spread proxy: LQD performance vs IEF
  Real Rate signal: TIP vs IEF relative performance (rising real rates = multiple compression)
  Overall Credit Conditions: composite risk-on / risk-off signal
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ETF universe
# ---------------------------------------------------------------------------
CREDIT_TICKERS = ["HYG", "LQD", "IEF", "TIP", "AGG", "SPY"]

# ---------------------------------------------------------------------------
# Data-classes
# ---------------------------------------------------------------------------

@dataclass
class CreditSpread:
    """Relative performance of a credit ETF vs the IEF duration benchmark."""
    etf_ticker: str = ""
    relative_performance_1m: Optional[float] = None   # credit ETF 1m return - IEF 1m return
    relative_performance_3m: Optional[float] = None   # credit ETF 3m return - IEF 3m return
    z_score: Optional[float] = None                   # SDs from 6m mean of the ratio series
    spread_direction: str = "STABLE"                  # TIGHTENING / WIDENING / STABLE
    signal: str = "NEUTRAL"                           # RISK_ON / NEUTRAL / CAUTION / RISK_OFF


@dataclass
class RealRateSignal:
    """Real rate signal derived from TIP vs IEF relative performance."""
    tip_ief_ratio_change_1m: Optional[float] = None   # % change in TIP/IEF ratio over 21 days
    tip_ief_ratio_change_3m: Optional[float] = None   # % change in TIP/IEF ratio over 63 days
    real_rate_trend: str = "STABLE"                   # RISING / FALLING / STABLE
    equity_impact: str = "NEUTRAL"                    # MULTIPLE_COMPRESSION / MULTIPLE_EXPANSION / NEUTRAL
    signal: str = "NEUTRAL"


@dataclass
class CreditConditionsResult:
    """Top-level credit conditions result aggregating all cross-asset signals."""
    hy_spread: CreditSpread = field(default_factory=CreditSpread)
    ig_spread: CreditSpread = field(default_factory=CreditSpread)
    real_rates: RealRateSignal = field(default_factory=RealRateSignal)
    hyg_vs_spy_divergence: Optional[float] = None     # HYG 1m return - SPY 1m return
    financial_conditions_signal: str = "NEUTRAL"      # EASING / NEUTRAL / TIGHTENING / STRESS
    credit_score: float = 0.0                         # -1.0 (very tight/stressed) to +1.0 (easy)
    summary: str = ""
    equity_implications: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_pct_return(prices: pd.Series, n_days: int) -> Optional[float]:
    """(price[-1] - price[-n_days-1]) / price[-n_days-1], or None."""
    if prices is None or len(prices) < n_days + 1:
        return None
    p_now = float(prices.iloc[-1])
    p_past = float(prices.iloc[-n_days - 1])
    if p_past == 0:
        return None
    return (p_now - p_past) / p_past


def _compute_z_score(ratio_series: pd.Series) -> Optional[float]:
    """Z-score of the last value relative to the full series (6m window)."""
    if ratio_series is None or len(ratio_series) < 10:
        return None
    mean = float(ratio_series.mean())
    std = float(ratio_series.std())
    if std == 0:
        return 0.0
    return float((ratio_series.iloc[-1] - mean) / std)


def _fetch_etf_prices(tickers: list[str], period: str = "6mo") -> dict[str, pd.Series]:
    """Download adjusted daily close prices for a list of ETF tickers.

    Returns a dict of {ticker: pd.Series} where each Series is the Close column.
    Missing tickers are omitted rather than raising.
    """
    result: dict[str, pd.Series] = {}
    try:
        # Batch download is more efficient and avoids per-ticker rate limits
        raw = yf.download(
            tickers,
            period=period,
            auto_adjust=True,
            progress=False,
            group_by="ticker",
        )
        if raw is None or raw.empty:
            logger.warning("Batch ETF download returned empty DataFrame.")
            return result

        for ticker in tickers:
            try:
                # yfinance >=0.2 returns MultiIndex columns when multiple tickers
                if isinstance(raw.columns, pd.MultiIndex):
                    if ticker in raw.columns.get_level_values(0):
                        col = raw[ticker]["Close"]
                    elif ticker in raw.columns.get_level_values(1):
                        col = raw.xs(ticker, axis=1, level=1)["Close"]
                    else:
                        logger.debug("Ticker %s not found in MultiIndex columns.", ticker)
                        continue
                else:
                    col = raw["Close"] if len(tickers) == 1 else raw[ticker]["Close"]

                if isinstance(col, pd.DataFrame):
                    col = col.iloc[:, 0]

                col = col.dropna()
                if len(col) >= 5:
                    result[ticker] = col
                else:
                    logger.warning("Ticker %s: insufficient data after dropna (%d rows).", ticker, len(col))
            except Exception as e:
                logger.warning("Could not extract prices for %s: %s", ticker, e)

    except Exception as e:
        logger.error("Batch ETF download failed: %s", e)
        # Fallback: fetch individually
        for ticker in tickers:
            try:
                df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
                if df is not None and not df.empty and "Close" in df.columns:
                    col = df["Close"].dropna()
                    if len(col) >= 5:
                        result[ticker] = col
            except Exception as inner_e:
                logger.warning("Individual fetch failed for %s: %s", ticker, inner_e)

    return result


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class CreditConditionsAnalyzer:
    """Cross-asset credit conditions monitor using bond ETF proxies."""

    def analyze(self) -> CreditConditionsResult:
        result = CreditConditionsResult()
        try:
            prices = _fetch_etf_prices(CREDIT_TICKERS, period="6mo")
            self._build_result(result, prices)
        except Exception as e:
            logger.error("CreditConditionsAnalyzer.analyze failed: %s", e)
            result.summary = f"Credit analysis failed: {e!s:.120s}"
        return result

    # ------------------------------------------------------------------
    # Core builder
    # ------------------------------------------------------------------

    def _build_result(
        self, result: CreditConditionsResult, prices: dict[str, pd.Series]
    ) -> None:
        ief = prices.get("IEF")

        # 1. HY Credit Spread proxy
        hyg = prices.get("HYG")
        result.hy_spread = self._compute_credit_spread("HYG", hyg, ief)

        # 2. IG Credit Spread proxy
        lqd = prices.get("LQD")
        result.ig_spread = self._compute_credit_spread("LQD", lqd, ief)

        # 3. Real rate signal
        tip = prices.get("TIP")
        result.real_rates = self._compute_real_rate_signal(tip, ief)

        # 4. HYG vs SPY divergence (credit leading equity indicator)
        spy = prices.get("SPY")
        if hyg is not None and spy is not None:
            hyg_1m = _safe_pct_return(hyg, 21)
            spy_1m = _safe_pct_return(spy, 21)
            if hyg_1m is not None and spy_1m is not None:
                result.hyg_vs_spy_divergence = hyg_1m - spy_1m

        # 5. Composite credit score
        self._compute_score(result)

        # 6. Financial conditions signal
        self._classify_conditions(result)

        # 7. Summary + equity implications
        result.summary = self._build_summary(result)
        result.equity_implications = self._build_equity_implications(result)

    # ------------------------------------------------------------------
    # Credit Spread
    # ------------------------------------------------------------------

    def _compute_credit_spread(
        self,
        ticker: str,
        etf_prices: Optional[pd.Series],
        ief_prices: Optional[pd.Series],
    ) -> CreditSpread:
        cs = CreditSpread(etf_ticker=ticker)

        if etf_prices is None or len(etf_prices) < 22:
            logger.warning("%s: insufficient data for credit spread.", ticker)
            return cs

        cs.relative_performance_1m = None
        cs.relative_performance_3m = None

        etf_1m = _safe_pct_return(etf_prices, 21)
        etf_3m = _safe_pct_return(etf_prices, 63)

        if ief_prices is not None and len(ief_prices) >= 22:
            ief_1m = _safe_pct_return(ief_prices, 21)
            ief_3m = _safe_pct_return(ief_prices, 63)
            if etf_1m is not None and ief_1m is not None:
                cs.relative_performance_1m = etf_1m - ief_1m
            if etf_3m is not None and ief_3m is not None:
                cs.relative_performance_3m = etf_3m - ief_3m

            # Z-score: use the ratio series ETF/IEF normalised over 6 months
            min_len = min(len(etf_prices), len(ief_prices))
            if min_len >= 10:
                # Align on common dates
                common_idx = etf_prices.index.intersection(ief_prices.index)
                if len(common_idx) >= 10:
                    ratio_series = etf_prices.loc[common_idx] / ief_prices.loc[common_idx]
                    cs.z_score = _compute_z_score(ratio_series)
        else:
            # No IEF benchmark — use standalone 1m return as loose proxy
            cs.relative_performance_1m = etf_1m

        # Spread direction
        rel = cs.relative_performance_1m
        z = cs.z_score
        if rel is not None and (rel > 0.01 or (z is not None and z > 0.5)):
            cs.spread_direction = "TIGHTENING"
        elif rel is not None and (rel < -0.02 or (z is not None and z < -1.0)):
            cs.spread_direction = "WIDENING"
        else:
            cs.spread_direction = "STABLE"

        # Signal
        if cs.spread_direction == "TIGHTENING":
            cs.signal = "RISK_ON"
        elif cs.spread_direction == "STABLE":
            cs.signal = "NEUTRAL"
        elif z is not None and z < -2.0:
            cs.signal = "RISK_OFF"
        else:
            cs.signal = "CAUTION"

        return cs

    # ------------------------------------------------------------------
    # Real Rate Signal
    # ------------------------------------------------------------------

    def _compute_real_rate_signal(
        self,
        tip_prices: Optional[pd.Series],
        ief_prices: Optional[pd.Series],
    ) -> RealRateSignal:
        rr = RealRateSignal()

        if tip_prices is None or ief_prices is None:
            logger.warning("TIP or IEF data unavailable — real rate signal not computed.")
            return rr

        min_len = min(len(tip_prices), len(ief_prices))
        if min_len < 22:
            return rr

        # Align on common index dates
        common_idx = tip_prices.index.intersection(ief_prices.index)
        if len(common_idx) < 22:
            return rr

        tip_aligned = tip_prices.loc[common_idx]
        ief_aligned = ief_prices.loc[common_idx]

        # TIP/IEF ratio change represents real rate trend
        # Rising ratio = TIP outperforming = inflation expectations > real rate fears → falling real rates
        # Falling ratio = IEF outperforming = real rates rising → multiple compression
        ratio = tip_aligned / ief_aligned

        ratio_1m = _safe_pct_return(ratio, 21)
        ratio_3m = _safe_pct_return(ratio, 63)

        rr.tip_ief_ratio_change_1m = ratio_1m
        rr.tip_ief_ratio_change_3m = ratio_3m

        # Real rate trend (inverse of TIP/IEF — when TIP lags, real rates are rising)
        if ratio_1m is not None:
            if ratio_1m < -0.005:
                # IEF outperforming TIP → TIPS losing → real rates rising
                rr.real_rate_trend = "RISING"
                rr.equity_impact = "MULTIPLE_COMPRESSION"
            elif ratio_1m > 0.005:
                # TIP outperforming → real rates falling
                rr.real_rate_trend = "FALLING"
                rr.equity_impact = "MULTIPLE_EXPANSION"
            else:
                rr.real_rate_trend = "STABLE"
                rr.equity_impact = "NEUTRAL"
        else:
            rr.real_rate_trend = "STABLE"
            rr.equity_impact = "NEUTRAL"

        # Signal
        if rr.real_rate_trend == "RISING":
            if ratio_1m is not None and ratio_1m < -0.015:
                rr.signal = "RISING_SHARPLY"
            else:
                rr.signal = "RISING"
        elif rr.real_rate_trend == "FALLING":
            rr.signal = "FALLING"
        else:
            rr.signal = "STABLE"

        return rr

    # ------------------------------------------------------------------
    # Composite scoring
    # ------------------------------------------------------------------

    def _compute_score(self, result: CreditConditionsResult) -> None:
        """
        credit_score = HY(40%) + IG(30%) + RealRate(30%)
        Range: -1.0 (severely stressed/tight) to +1.0 (easy/risk-on)
        """
        hy_s = self._credit_spread_score(result.hy_spread)
        ig_s = self._credit_spread_score(result.ig_spread)
        rr_s = self._real_rate_score(result.real_rates)

        result.credit_score = round(
            hy_s * 0.40 + ig_s * 0.30 + rr_s * 0.30,
            4,
        )

    @staticmethod
    def _credit_spread_score(cs: CreditSpread) -> float:
        """HY/IG spread → score. Tightening = bullish (+1), widening = bearish (-1)."""
        z = cs.z_score
        rel = cs.relative_performance_1m

        if cs.spread_direction == "TIGHTENING":
            if z is not None and z > 1.5:
                return 1.0
            return 0.6
        elif cs.spread_direction == "STABLE":
            return 0.0
        else:  # WIDENING
            if z is not None and z < -2.0:
                return -1.0
            elif z is not None and z < -1.0:
                return -0.5
            elif rel is not None and rel < -0.02:
                return -0.5
            return -0.25

    @staticmethod
    def _real_rate_score(rr: RealRateSignal) -> float:
        """Rising real rates compress multiples → negative equity impact."""
        if rr.real_rate_trend == "RISING":
            # Check magnitude
            chg = rr.tip_ief_ratio_change_1m
            if chg is not None and chg < -0.015:
                return -0.8
            return -0.4
        elif rr.real_rate_trend == "FALLING":
            return 0.5
        return 0.0

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def _classify_conditions(self, result: CreditConditionsResult) -> None:
        s = result.credit_score
        div = result.hyg_vs_spy_divergence

        # Penalise if HYG is significantly lagging SPY (credit leading equity lower)
        adjusted_score = s
        if div is not None and div < -0.03:
            adjusted_score -= 0.2  # credit leading equity lower

        if adjusted_score >= 0.40:
            result.financial_conditions_signal = "EASING"
        elif adjusted_score >= 0.10:
            result.financial_conditions_signal = "NEUTRAL"
        elif adjusted_score >= -0.20:
            result.financial_conditions_signal = "TIGHTENING"
        else:
            result.financial_conditions_signal = "STRESS"

    # ------------------------------------------------------------------
    # Summary + Equity Implications
    # ------------------------------------------------------------------

    @staticmethod
    def _build_summary(result: CreditConditionsResult) -> str:
        parts: list[str] = [
            f"Credit conditions: {result.financial_conditions_signal} (score: {result.credit_score:+.2f})"
        ]

        hy = result.hy_spread
        if hy.spread_direction:
            z_str = f", z={hy.z_score:+.1f}" if hy.z_score is not None else ""
            parts.append(f"HY ({hy.etf_ticker}): {hy.spread_direction}{z_str}")

        ig = result.ig_spread
        if ig.spread_direction:
            z_str = f", z={ig.z_score:+.1f}" if ig.z_score is not None else ""
            parts.append(f"IG ({ig.etf_ticker}): {ig.spread_direction}{z_str}")

        rr = result.real_rates
        if rr.real_rate_trend != "STABLE":
            parts.append(f"Real rates: {rr.real_rate_trend} → {rr.equity_impact}")

        if result.hyg_vs_spy_divergence is not None:
            parts.append(f"HYG vs SPY divergence: {result.hyg_vs_spy_divergence:+.1%}")

        return " | ".join(parts)

    @staticmethod
    def _build_equity_implications(result: CreditConditionsResult) -> list[str]:
        implications: list[str] = []

        # HY spread widening
        hy = result.hy_spread
        if hy.spread_direction == "WIDENING":
            if hy.z_score is not None and hy.z_score < -2.0:
                implications.append(
                    "⚠️ HY spreads widening sharply (z < -2) — historically precedes equity drawdowns 2-4 weeks ahead; consider reducing risk exposure."
                )
            else:
                implications.append(
                    "⚠️ HY spreads widening — typical precursor to equity volatility 2-4 weeks ahead; monitor closely."
                )
        elif hy.spread_direction == "TIGHTENING":
            implications.append(
                "✅ HY spreads tightening — risk-on signal; credit markets supporting equity rally."
            )

        # IG spread widening
        ig = result.ig_spread
        if ig.spread_direction == "WIDENING":
            implications.append(
                "⚠️ IG credit spreads widening — investment-grade stress signal; watch for contagion to equities."
            )
        elif ig.spread_direction == "TIGHTENING":
            implications.append(
                "✅ IG spreads tightening — broad credit demand healthy; positive backdrop for equity valuations."
            )

        # Real rates
        rr = result.real_rates
        if rr.equity_impact == "MULTIPLE_COMPRESSION":
            implications.append(
                "⚠️ Real rates rising — expect valuation multiple compression, especially for long-duration growth stocks."
            )
        elif rr.equity_impact == "MULTIPLE_EXPANSION":
            implications.append(
                "✅ Real rates falling — supportive of P/E multiple expansion; favors growth and rate-sensitive sectors."
            )

        # HYG vs SPY divergence
        div = result.hyg_vs_spy_divergence
        if div is not None and div < -0.03:
            implications.append(
                f"⚠️ Credit-equity divergence: HYG lagging SPY by {abs(div):.1%} — credit is leading equity lower; high-yield bond market is signaling caution."
            )
        elif div is not None and div > 0.02:
            implications.append(
                f"✅ HYG outperforming SPY by {div:.1%} — credit markets leading equities higher; constructive risk-on signal."
            )

        # Overall stress
        if result.financial_conditions_signal == "STRESS":
            implications.append(
                "🚨 Financial conditions in STRESS mode — broad credit deterioration; historically associated with elevated equity drawdown risk. Reduce leverage and cyclical exposure."
            )
        elif result.financial_conditions_signal == "EASING":
            implications.append(
                "✅ Financial conditions EASING — credit tailwind for equities; accommodative environment supports risk assets."
            )

        # De-duplicate and cap at 5
        seen: set[str] = set()
        deduped: list[str] = []
        for item in implications:
            if item not in seen:
                seen.add(item)
                deduped.append(item)

        return deduped[:5]
