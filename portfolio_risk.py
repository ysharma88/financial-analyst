"""Portfolio-level risk analytics.

Computes what individual stock analysis misses — how a stock fits into a portfolio:
  1. Factor Exposures (Barra-lite): Value, Momentum, Quality, Low-Vol, Growth quintile scores
  2. Rolling Alpha & Beta vs SPY (252-day window)
  3. Information Ratio (alpha / tracking error)
  4. Correlation to existing portfolio holdings
  5. Beta-Adjusted Position Sizing (target portfolio beta contribution)
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default holdings file path
# ---------------------------------------------------------------------------
_DEFAULT_HOLDINGS_PATH = (
    "/Users/yogesh.sharma/.claude/skills/financial-analyst-skill/portfolio/holdings.jsonl"
)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class FactorExposures:
    value_score: float        # 0–100, 100 = cheapest
    momentum_score: float     # 0–100, 100 = strongest momentum
    quality_score: float      # 0–100, 100 = highest quality
    low_vol_score: float      # 0–100, 100 = lowest volatility
    growth_score: float       # 0–100, 100 = fastest growth
    dominant_factor: str      # name of highest-scoring factor
    factor_summary: str       # human-readable paragraph


@dataclass
class BenchmarkMetrics:
    beta_vs_spy: float                  # market sensitivity
    alpha_annualized: float             # Jensen's alpha, %
    r_squared: float                    # explanatory power vs SPY
    tracking_error_annualized: float    # annualised std of residuals, %
    information_ratio: float            # alpha / tracking error
    benchmark_signal: str               # e.g. "HIGH ALPHA", "BENCHMARK-LIKE"


@dataclass
class CorrelationRisk:
    correlations: dict                  # ticker -> Pearson correlation
    max_correlation_ticker: str         # ticker with highest correlation
    max_correlation: float              # value of that correlation
    concentration_warning: bool         # True if max_correlation > 0.75
    portfolio_beta_contribution: float  # beta * position weight
    warning_message: str


@dataclass
class PortfolioRiskResult:
    factor_exposures: FactorExposures
    benchmark_metrics: BenchmarkMetrics
    correlation_risk: CorrelationRisk
    beta_adjusted_shares: int
    beta_adjusted_position_value: float
    quality_flags: list[str]
    overall_risk_signal: str
    summary: str


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _safe_float(value, default: float = 0.0) -> float:
    """Return float or default when value is None/NaN."""
    if value is None:
        return default
    try:
        f = float(value)
        return default if math.isnan(f) or math.isinf(f) else f
    except (TypeError, ValueError):
        return default


def _fetch_ticker_returns(ticker: str, period: str = "1y") -> pd.Series:
    """Fetch daily close returns for a ticker, with one retry on failure."""
    for attempt in range(2):
        try:
            hist = yf.Ticker(ticker).history(period=period, interval="1d")
            if hist is not None and len(hist) > 5:
                closes = hist["Close"].dropna()
                if len(closes) > 1:
                    return closes.pct_change().dropna()
        except Exception as exc:
            logger.debug("Fetch failed for %s (attempt %d): %s", ticker, attempt + 1, exc)
            if attempt == 0:
                time.sleep(2)
    return pd.Series(dtype=float)


# ---------------------------------------------------------------------------
# Factor scoring helpers
# ---------------------------------------------------------------------------

def _score_value(info: dict) -> float:
    """Return 0-100 value score from P/E and P/B ratios."""
    pe = _safe_float(info.get("trailingPE") or info.get("forwardPE"), default=None)
    pb = _safe_float(info.get("priceToBook"), default=None)

    scores: list[float] = []

    if pe is not None and pe > 0:
        if pe < 15:
            scores.append(80.0)
        elif pe < 25:
            scores.append(60.0)
        elif pe < 40:
            scores.append(40.0)
        else:
            scores.append(20.0)

    if pb is not None and pb > 0:
        if pb < 2:
            scores.append(80.0)
        elif pb < 5:
            scores.append(55.0)
        elif pb < 10:
            scores.append(35.0)
        else:
            scores.append(20.0)

    return float(np.mean(scores)) if scores else 50.0


def _score_momentum(history: pd.DataFrame) -> float:
    """Return 0-100 momentum score from 12m-1m return."""
    if history is None or history.empty or len(history) < 30:
        return 50.0
    closes = history["Close"].dropna()
    if len(closes) < 30:
        return 50.0

    # 12-month return minus most recent month (standard Jegadeesh-Titman)
    price_now = _safe_float(closes.iloc[-22])   # ~1 month ago
    price_12m = _safe_float(closes.iloc[0])     # start of window

    if price_12m <= 0:
        return 50.0

    ret_12m1m = (price_now - price_12m) / price_12m

    if ret_12m1m > 0.30:
        return 90.0
    elif ret_12m1m > 0.15:
        return 70.0
    elif ret_12m1m >= 0.0:
        return 50.0
    elif ret_12m1m >= -0.15:
        return 30.0
    else:
        return 10.0


def _score_quality(info: dict) -> float:
    """Return 0-100 quality score from ROE and FCF metrics."""
    scores: list[float] = []

    roe = _safe_float(info.get("returnOnEquity"), default=None)
    if roe is not None:
        roe_pct = roe * 100
        if roe_pct > 20:
            scores.append(80.0)
        elif roe_pct > 10:
            scores.append(60.0)
        elif roe_pct > 0:
            scores.append(40.0)
        else:
            scores.append(20.0)

    # FCF proxy: FCF margin = freeCashflow / totalRevenue
    fcf = _safe_float(info.get("freeCashflow"), default=None)
    revenue = _safe_float(info.get("totalRevenue"), default=None)
    if fcf is not None and revenue is not None and revenue > 0:
        fcf_margin = fcf / revenue
        if fcf_margin > 0.20:
            scores.append(85.0)
        elif fcf_margin > 0.10:
            scores.append(65.0)
        elif fcf_margin > 0:
            scores.append(45.0)
        else:
            scores.append(20.0)

    return float(np.mean(scores)) if scores else 50.0


def _score_low_vol(history: pd.DataFrame) -> float:
    """Return 0-100 low-volatility score from annualised daily return std."""
    if history is None or history.empty or len(history) < 20:
        return 50.0
    closes = history["Close"].dropna()
    if len(closes) < 20:
        return 50.0
    daily_returns = closes.pct_change().dropna()
    ann_vol = float(daily_returns.std()) * math.sqrt(252) * 100  # percent

    if ann_vol < 20:
        return 90.0
    elif ann_vol < 30:
        return 70.0
    elif ann_vol < 40:
        return 50.0
    elif ann_vol < 60:
        return 30.0
    else:
        return 10.0


def _score_growth(info: dict) -> float:
    """Return 0-100 growth score from revenue and earnings growth."""
    scores: list[float] = []

    for key in ("revenueGrowth", "earningsGrowth", "earningsQuarterlyGrowth"):
        val = _safe_float(info.get(key), default=None)
        if val is not None:
            pct = val * 100
            if pct > 30:
                scores.append(90.0)
            elif pct > 15:
                scores.append(70.0)
            elif pct > 5:
                scores.append(55.0)
            elif pct >= 0:
                scores.append(40.0)
            else:
                scores.append(20.0)
            break  # use first available

    # Second signal: earnings growth
    for key in ("earningsGrowth", "earningsQuarterlyGrowth"):
        val = _safe_float(info.get(key), default=None)
        if val is not None:
            pct = val * 100
            if pct > 30:
                scores.append(90.0)
            elif pct > 15:
                scores.append(70.0)
            elif pct > 5:
                scores.append(55.0)
            elif pct >= 0:
                scores.append(40.0)
            else:
                scores.append(20.0)
            break

    return float(np.mean(scores)) if scores else 50.0


def _compute_factor_exposures(history: pd.DataFrame, info: dict) -> FactorExposures:
    """Compute all five Barra-lite factor scores."""
    v = _score_value(info)
    m = _score_momentum(history)
    q = _score_quality(info)
    lv = _score_low_vol(history)
    g = _score_growth(info)

    factor_map = {
        "Value": v,
        "Momentum": m,
        "Quality": q,
        "Low-Volatility": lv,
        "Growth": g,
    }
    dominant = max(factor_map, key=factor_map.__getitem__)

    summary = (
        f"Factor scores (0–100): Value={v:.0f}, Momentum={m:.0f}, "
        f"Quality={q:.0f}, Low-Vol={lv:.0f}, Growth={g:.0f}. "
        f"Dominant factor: {dominant} ({factor_map[dominant]:.0f}/100). "
    )
    if v >= 70:
        summary += "Stock screens as attractively valued. "
    if m >= 70:
        summary += "Strong price momentum over the past year. "
    if q >= 70:
        summary += "High-quality business with solid returns and cash generation. "
    if lv >= 70:
        summary += "Low historical volatility — defensive characteristics. "
    if g >= 70:
        summary += "High revenue/earnings growth profile. "

    return FactorExposures(
        value_score=round(v, 1),
        momentum_score=round(m, 1),
        quality_score=round(q, 1),
        low_vol_score=round(lv, 1),
        growth_score=round(g, 1),
        dominant_factor=dominant,
        factor_summary=summary.strip(),
    )


# ---------------------------------------------------------------------------
# Beta / Alpha helpers
# ---------------------------------------------------------------------------

def _compute_benchmark_metrics(
    history: pd.DataFrame,
    ticker: str,
) -> BenchmarkMetrics:
    """Compute rolling 252-day beta, alpha, tracking error, IR vs SPY."""
    _default = BenchmarkMetrics(
        beta_vs_spy=1.0,
        alpha_annualized=0.0,
        r_squared=0.0,
        tracking_error_annualized=0.0,
        information_ratio=0.0,
        benchmark_signal="INSUFFICIENT DATA",
    )

    if history is None or history.empty or len(history) < 60:
        return _default

    closes = history["Close"].dropna()
    stock_returns = closes.pct_change().dropna()

    # Fetch SPY for matching period
    try:
        spy_hist = yf.Ticker("SPY").history(period="2y", interval="1d")
        if spy_hist is None or spy_hist.empty:
            return _default
        spy_returns = spy_hist["Close"].dropna().pct_change().dropna()
    except Exception as exc:
        logger.debug("SPY fetch failed: %s", exc)
        return _default

    # Align on common dates — use last 252 trading days
    aligned = pd.DataFrame({"stock": stock_returns, "spy": spy_returns}).dropna()
    if len(aligned) < 60:
        return _default

    window = aligned.tail(252)
    x = window["spy"].values
    y = window["stock"].values

    # OLS regression via numpy polyfit
    try:
        coeffs = np.polyfit(x, y, 1)
        beta = float(coeffs[0])
        daily_alpha = float(coeffs[1])
    except Exception:
        return _default

    # Annualise alpha (compound)
    alpha_annualized = ((1 + daily_alpha) ** 252 - 1) * 100

    # R-squared
    y_hat = beta * x + daily_alpha
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    r_squared = max(0.0, min(1.0, r_squared))

    # Tracking error = annualised std of residuals
    residuals = y - y_hat
    tracking_error = float(np.std(residuals, ddof=1)) * math.sqrt(252) * 100

    # Information Ratio
    if tracking_error > 0:
        information_ratio = alpha_annualized / tracking_error
    else:
        information_ratio = 0.0

    # Signal
    if alpha_annualized > 5 and information_ratio > 0.5:
        signal = "HIGH ALPHA"
    elif alpha_annualized > 2:
        signal = "POSITIVE ALPHA"
    elif alpha_annualized < -5:
        signal = "NEGATIVE ALPHA — UNDERPERFORMING SPY"
    elif 0.8 <= beta <= 1.2 and r_squared > 0.7:
        signal = "BENCHMARK-LIKE"
    elif beta > 1.5:
        signal = "HIGH BETA — AMPLIFIED MARKET RISK"
    elif beta < 0.5:
        signal = "LOW BETA — DEFENSIVE"
    else:
        signal = "MARKET-CORRELATED"

    return BenchmarkMetrics(
        beta_vs_spy=round(beta, 3),
        alpha_annualized=round(alpha_annualized, 2),
        r_squared=round(r_squared, 3),
        tracking_error_annualized=round(tracking_error, 2),
        information_ratio=round(information_ratio, 3),
        benchmark_signal=signal,
    )


# ---------------------------------------------------------------------------
# Correlation to portfolio holdings
# ---------------------------------------------------------------------------

def _load_holdings(holdings_path: str) -> list[str]:
    """Read tickers from a JSONL holdings file."""
    path = Path(holdings_path)
    if not path.exists():
        return []
    tickers: list[str] = []
    try:
        with path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                t = obj.get("ticker") or obj.get("symbol") or ""
                if t:
                    tickers.append(t.upper())
    except Exception as exc:
        logger.debug("Failed to read holdings file %s: %s", holdings_path, exc)
    return tickers


def _compute_correlation_risk(
    ticker: str,
    history: pd.DataFrame,
    holdings_path: Optional[str],
    beta_vs_spy: float,
    account_size: float,
    current_price: float,
) -> CorrelationRisk:
    """Compute Pearson correlations vs existing portfolio holdings."""
    _default = CorrelationRisk(
        correlations={},
        max_correlation_ticker="N/A",
        max_correlation=0.0,
        concentration_warning=False,
        portfolio_beta_contribution=0.0,
        warning_message="No holdings file found or provided — correlation analysis skipped.",
    )

    if holdings_path is None:
        holdings_path = _DEFAULT_HOLDINGS_PATH

    tickers = _load_holdings(holdings_path)
    if not tickers:
        return _default

    # Remove the candidate ticker itself from comparison set
    tickers = [t for t in tickers if t.upper() != ticker.upper()][:10]

    if history is None or history.empty:
        return _default

    stock_rets = history["Close"].dropna().pct_change().dropna()

    correlations: dict[str, float] = {}
    for hticker in tickers:
        try:
            hrets = _fetch_ticker_returns(hticker, period="6mo")
            if hrets.empty:
                continue
            aligned = pd.DataFrame(
                {"stock": stock_rets, "holding": hrets}
            ).dropna()
            if len(aligned) < 30:
                continue
            corr = float(aligned["stock"].corr(aligned["holding"]))
            if not math.isnan(corr):
                correlations[hticker] = round(corr, 3)
        except Exception as exc:
            logger.debug("Correlation fetch failed for %s: %s", hticker, exc)

    if not correlations:
        _default.warning_message = "Holdings found but correlation data unavailable."
        return _default

    max_ticker = max(correlations, key=correlations.__getitem__)
    max_corr = correlations[max_ticker]
    concentration_warning = max_corr > 0.75

    # Approximate beta contribution: assume equal-weight 10-stock portfolio
    position_value = current_price * max(1, 1)  # placeholder per-share
    position_weight = 1.0 / 10.0  # 10-stock equal-weight assumption
    portfolio_beta_contribution = beta_vs_spy * position_weight

    if concentration_warning:
        warning = (
            f"HIGH CONCENTRATION RISK: {ticker} has {max_corr:.2f} correlation "
            f"with existing holding {max_ticker}. Adding this position provides "
            f"limited diversification benefit."
        )
    elif max_corr > 0.5:
        warning = (
            f"MODERATE CORRELATION: {ticker} correlates {max_corr:.2f} with "
            f"{max_ticker}. Monitor for sector clustering."
        )
    else:
        warning = (
            f"GOOD DIVERSIFICATION: Highest correlation is {max_corr:.2f} with "
            f"{max_ticker} — {ticker} adds meaningful diversification."
        )

    return CorrelationRisk(
        correlations=correlations,
        max_correlation_ticker=max_ticker,
        max_correlation=round(max_corr, 3),
        concentration_warning=concentration_warning,
        portfolio_beta_contribution=round(portfolio_beta_contribution, 4),
        warning_message=warning,
    )


# ---------------------------------------------------------------------------
# Beta-adjusted position sizing
# ---------------------------------------------------------------------------

def _beta_adjusted_sizing(
    beta_vs_spy: float,
    current_price: float,
    account_size: float,
    target_beta: float,
) -> tuple[int, float]:
    """Return (shares, position_value) sized so stock contributes 1/10 of target_beta."""
    if current_price <= 0 or beta_vs_spy <= 0:
        return 0, 0.0

    raw_shares = (account_size * target_beta / beta_vs_spy) / current_price / 10.0
    shares = int(math.floor(raw_shares))

    # Cap at 20% of account
    max_shares = int(math.floor(account_size * 0.20 / current_price))
    shares = min(shares, max_shares)
    shares = max(shares, 0)

    position_value = round(shares * current_price, 2)
    return shares, position_value


# ---------------------------------------------------------------------------
# Main analyser
# ---------------------------------------------------------------------------

class PortfolioRiskAnalyzer:
    """Computes portfolio-context risk metrics for a candidate stock."""

    def analyze(
        self,
        ticker: str,
        history: pd.DataFrame,
        info: dict,
        account_size: float = 100_000,
        target_beta: float = 1.0,
        holdings_path: Optional[str] = None,
    ) -> PortfolioRiskResult:
        """
        Parameters
        ----------
        ticker       : stock symbol (e.g. 'AAPL')
        history      : pd.DataFrame with at least a 'Close' column (1-2 year daily)
        info         : yfinance Ticker.info dict
        account_size : total portfolio value in USD (default $100,000)
        target_beta  : desired net portfolio beta (default 1.0 = market neutral)
        holdings_path: path to JSONL file of existing holdings (optional)

        Returns
        -------
        PortfolioRiskResult dataclass
        """
        ticker = ticker.upper()
        info = info or {}

        current_price = _safe_float(
            info.get("currentPrice")
            or info.get("regularMarketPrice")
            or info.get("previousClose"),
            default=0.0,
        )
        if current_price == 0.0 and history is not None and not history.empty:
            current_price = _safe_float(history["Close"].dropna().iloc[-1], default=0.0)

        # --- Factor exposures ---
        factor_exposures = _compute_factor_exposures(history, info)

        # --- Benchmark metrics ---
        benchmark_metrics = _compute_benchmark_metrics(history, ticker)

        # --- Correlation risk ---
        correlation_risk = _compute_correlation_risk(
            ticker=ticker,
            history=history,
            holdings_path=holdings_path,
            beta_vs_spy=benchmark_metrics.beta_vs_spy,
            account_size=account_size,
            current_price=current_price,
        )

        # --- Beta-adjusted sizing ---
        beta_adj_shares, beta_adj_value = _beta_adjusted_sizing(
            beta_vs_spy=benchmark_metrics.beta_vs_spy,
            current_price=current_price,
            account_size=account_size,
            target_beta=target_beta,
        )

        # --- Quality flags ---
        quality_flags: list[str] = []

        if benchmark_metrics.beta_vs_spy > 2.0:
            quality_flags.append(
                f"EXTREME BETA: beta={benchmark_metrics.beta_vs_spy:.2f} — high market amplification"
            )
        if benchmark_metrics.alpha_annualized < -10:
            quality_flags.append(
                f"PERSISTENT UNDERPERFORMANCE: alpha={benchmark_metrics.alpha_annualized:.1f}% annually"
            )
        if correlation_risk.concentration_warning:
            quality_flags.append(
                f"CONCENTRATION RISK: {correlation_risk.max_correlation:.2f} correlation "
                f"with {correlation_risk.max_correlation_ticker}"
            )
        if factor_exposures.growth_score < 30 and factor_exposures.value_score < 30:
            quality_flags.append("VALUE TRAP RISK: poor growth AND poor value scores")
        if benchmark_metrics.r_squared < 0.1 and benchmark_metrics.beta_vs_spy > 0:
            quality_flags.append("LOW R²: idiosyncratic risk dominates — difficult to hedge")
        if beta_adj_value > account_size * 0.20:
            quality_flags.append("SIZING CAPPED at 20% of account due to portfolio concentration rules")

        # --- Overall risk signal ---
        risk_points = 0.0
        # Alpha contribution
        if benchmark_metrics.alpha_annualized > 5:
            risk_points += 2
        elif benchmark_metrics.alpha_annualized > 2:
            risk_points += 1
        elif benchmark_metrics.alpha_annualized < -5:
            risk_points -= 2
        elif benchmark_metrics.alpha_annualized < 0:
            risk_points -= 1

        # Factor quality
        avg_factor = np.mean([
            factor_exposures.value_score,
            factor_exposures.momentum_score,
            factor_exposures.quality_score,
        ])
        if avg_factor > 70:
            risk_points += 1
        elif avg_factor < 35:
            risk_points -= 1

        # Correlation penalty
        if correlation_risk.concentration_warning:
            risk_points -= 1

        if risk_points >= 2:
            overall_signal = "FAVORABLE PORTFOLIO FIT"
        elif risk_points >= 0:
            overall_signal = "NEUTRAL PORTFOLIO FIT"
        elif risk_points >= -2:
            overall_signal = "CAUTION — REVIEW BEFORE ADDING"
        else:
            overall_signal = "UNFAVORABLE PORTFOLIO FIT"

        # --- Summary ---
        pct_of_account = (beta_adj_value / account_size * 100) if account_size > 0 else 0
        summary = (
            f"{ticker} portfolio risk analysis: "
            f"Beta={benchmark_metrics.beta_vs_spy:.2f}, "
            f"Alpha={benchmark_metrics.alpha_annualized:+.1f}%/yr, "
            f"IR={benchmark_metrics.information_ratio:.2f}. "
            f"Factor profile dominated by {factor_exposures.dominant_factor} "
            f"(score {getattr(factor_exposures, factor_exposures.dominant_factor.lower().replace('-', '_') + '_score', 0):.0f}/100). "
            f"Beta-adjusted sizing: {beta_adj_shares} shares "
            f"(${beta_adj_value:,.0f} = {pct_of_account:.1f}% of account). "
        )
        if correlation_risk.max_correlation_ticker != "N/A":
            summary += (
                f"Max portfolio correlation: {correlation_risk.max_correlation:.2f} "
                f"with {correlation_risk.max_correlation_ticker}. "
            )
        if quality_flags:
            summary += f"Flags: {'; '.join(quality_flags[:3])}."
        else:
            summary += "No major risk flags."

        return PortfolioRiskResult(
            factor_exposures=factor_exposures,
            benchmark_metrics=benchmark_metrics,
            correlation_risk=correlation_risk,
            beta_adjusted_shares=beta_adj_shares,
            beta_adjusted_position_value=beta_adj_value,
            quality_flags=quality_flags,
            overall_risk_signal=overall_signal,
            summary=summary.strip(),
        )
