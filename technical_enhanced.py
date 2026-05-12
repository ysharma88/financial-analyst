"""Enhanced technical analysis — institutional-grade momentum and market-structure signals.

Adds what the base TechnicalAnalyzer misses:
  1. Relative Strength vs SPY (O'Neil RS Rating) — top-quintile RS stocks outperform by 2-5%/yr
  2. Multi-Period Price ROC (1m/3m/6m/12m) — the classic Jegadeesh-Titman momentum factor
  3. 52-Week High Proximity — breakouts to new highs have strong follow-through
  4. VWAP (90-day + YTD) — institutional execution benchmark; above VWAP = smart money in profit
  5. Short Interest + Days to Cover — squeeze potential and directional sentiment
  6. Options-Implied Expected Move — ATM straddle / price; size positions around events
  7. Volatility Skew (Put Skew) — steep negative skew = institutional fear; put/call IV spread
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data-classes
# ---------------------------------------------------------------------------

@dataclass
class RSMetrics:
    """Relative Strength vs SPY — O'Neil-style RS rating."""
    rs_1m: Optional[float] = None       # stock / SPY return ratio over ~21 trading days
    rs_3m: Optional[float] = None       # 63 trading days
    rs_6m: Optional[float] = None       # 126 trading days
    rs_12m: Optional[float] = None      # 252 trading days
    rs_composite: Optional[float] = None  # weighted: 10% 1m + 20% 3m + 30% 6m + 40% 12m
    rs_signal: str = "UNKNOWN"          # STRONG / MODERATE / WEAK / LAGGARD


@dataclass
class MomentumROC:
    """Multi-period price Rate-Of-Change — Jegadeesh-Titman momentum factor."""
    roc_1m: Optional[float] = None      # (P_now - P_21d_ago) / P_21d_ago
    roc_3m: Optional[float] = None      # 63 trading days
    roc_6m: Optional[float] = None      # 126 trading days
    roc_12m: Optional[float] = None     # 252 trading days
    momentum_12_1: Optional[float] = None  # roc_12m - roc_1m (canonical cross-sectional momentum)
    momentum_signal: str = "UNKNOWN"    # STRONG_BULL / BULL / NEUTRAL / BEAR / STRONG_BEAR


@dataclass
class PriceStructure:
    """52-week positioning and VWAP relationships."""
    pct_from_52w_high: Optional[float] = None   # (price - 52w_high) / 52w_high
    pct_from_52w_low: Optional[float] = None    # (price - 52w_low) / 52w_low
    is_near_52w_high: bool = False              # within 3% of 52-week high
    is_at_52w_low: bool = False                 # within 3% of 52-week low
    vwap_90d: Optional[float] = None            # 90-day volume-weighted average price
    vwap_ytd: Optional[float] = None            # year-to-date VWAP
    price_vs_vwap_90d: Optional[float] = None   # % above/below 90d VWAP
    price_vs_vwap_ytd: Optional[float] = None   # % above/below YTD VWAP
    structure_signal: str = "UNKNOWN"           # BREAKOUT / STRONG / MIXED / WEAK / DISTRIBUTION


@dataclass
class ShortInterest:
    """Short interest metrics and squeeze potential."""
    short_pct_float: Optional[float] = None  # % of float sold short
    days_to_cover: Optional[float] = None    # shares_short / avg_daily_volume
    shares_short: Optional[int] = None
    short_signal: str = "UNKNOWN"            # SQUEEZE_RISK / HIGH / MODERATE / LOW / UNKNOWN


@dataclass
class OptionsSignals:
    """Options-derived signals: implied move and volatility skew."""
    implied_move_pct: Optional[float] = None   # ATM straddle / current price
    put_skew: Optional[float] = None           # OTM put IV (90% strike) - ATM IV
    call_skew: Optional[float] = None          # OTM call IV (110% strike) - ATM IV
    skew_ratio: Optional[float] = None         # put IV / call IV
    skew_signal: str = "UNKNOWN"               # FEAR / ELEVATED / NEUTRAL / COMPLACENT
    has_options: bool = False
    nearest_expiry: str = ""
    dte: Optional[int] = None                  # days to expiration used


@dataclass
class TechnicalEnhancedResult:
    """Top-level result aggregating all enhanced technical signals."""
    rs: RSMetrics = field(default_factory=RSMetrics)
    momentum: MomentumROC = field(default_factory=MomentumROC)
    structure: PriceStructure = field(default_factory=PriceStructure)
    short_interest: ShortInterest = field(default_factory=ShortInterest)
    options: OptionsSignals = field(default_factory=OptionsSignals)
    quality_flags: list[str] = field(default_factory=list)   # data-quality warnings
    overall_score: float = 0.0    # -1.0 (very bearish) to +1.0 (very bullish)
    signal: str = "NEUTRAL"       # BULLISH MOMENTUM / MILD BULLISH / NEUTRAL / MILD BEARISH / BEARISH MOMENTUM
    summary: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_return(series: pd.Series, n_days: int) -> Optional[float]:
    """Return (price_now - price_n_days_ago) / price_n_days_ago, or None."""
    if series is None or len(series) < n_days + 1:
        return None
    price_now = float(series.iloc[-1])
    price_past = float(series.iloc[-n_days - 1])
    if price_past == 0:
        return None
    return (price_now - price_past) / price_past


def _compute_vwap(df: pd.DataFrame, window: Optional[int] = None) -> Optional[float]:
    """VWAP = Σ(TP * Volume) / Σ(Volume) over the given window (or entire df if None)."""
    try:
        sub = df.tail(window) if window is not None else df
        if sub.empty or "High" not in sub or "Low" not in sub or "Close" not in sub or "Volume" not in sub:
            return None
        tp = (sub["High"] + sub["Low"] + sub["Close"]) / 3.0
        vol = sub["Volume"].astype(float)
        total_vol = vol.sum()
        if total_vol == 0:
            return None
        return float((tp * vol).sum() / total_vol)
    except Exception as e:
        logger.debug("VWAP computation failed: %s", e)
        return None


def _fetch_spy(n_periods: int = 252) -> Optional[pd.Series]:
    """Download SPY adjusted closes for the past 1 year."""
    try:
        spy_df = yf.download("SPY", period="1y", auto_adjust=True, progress=False)
        if spy_df is None or spy_df.empty:
            return None
        close_col = spy_df["Close"] if "Close" in spy_df.columns else spy_df.iloc[:, 0]
        # Flatten in case of MultiIndex columns (yfinance >=0.2)
        if isinstance(close_col, pd.DataFrame):
            close_col = close_col.iloc[:, 0]
        return close_col.dropna()
    except Exception as e:
        logger.warning("SPY fetch failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class TechnicalEnhancedAnalyzer:
    """Institutional-grade momentum and market-structure signals."""

    def analyze(
        self,
        ticker: str,
        history: pd.DataFrame,
        info: Optional[dict] = None,
    ) -> TechnicalEnhancedResult:
        result = TechnicalEnhancedResult()

        if history is None or history.empty or len(history) < 21:
            result.quality_flags.append("Insufficient price history — need at least 21 days.")
            result.signal = "NEUTRAL"
            result.summary = "Not enough data for enhanced technical analysis."
            return result

        if info is None:
            info = {}

        history = history.copy()
        close = history["Close"]

        # Flatten in case of MultiIndex columns (yfinance >=0.2)
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = close.dropna()

        current_price = float(close.iloc[-1])

        # 1. Relative Strength vs SPY
        result.rs = self._compute_rs(close, result.quality_flags)

        # 2. Price momentum (ROC)
        result.momentum = self._compute_momentum_roc(close, result.quality_flags)

        # 3. Price structure (52w, VWAP)
        result.structure = self._compute_price_structure(history, close, current_price, result.quality_flags)

        # 4. Short interest
        result.short_interest = self._compute_short_interest(info, result.quality_flags)

        # 5. Options signals
        result.options = self._compute_options_signals(ticker, current_price, result.quality_flags)

        # 6. Composite score + signal
        self._compute_overall(result)

        return result

    # ------------------------------------------------------------------
    # 1. Relative Strength
    # ------------------------------------------------------------------

    def _compute_rs(self, close: pd.Series, flags: list[str]) -> RSMetrics:
        rs = RSMetrics()
        spy = _fetch_spy()

        if spy is None or spy.empty:
            flags.append("SPY data unavailable — RS metrics not computed.")
            rs.rs_signal = "UNKNOWN"
            return rs

        def rs_for_period(n_days: int) -> Optional[float]:
            stock_ret = _safe_return(close, n_days)
            spy_ret = _safe_return(spy, n_days)
            if stock_ret is None or spy_ret is None:
                return None
            # RS ratio: (1 + stock_ret) / (1 + spy_ret) — normalized to 0 = market-perform
            if (1 + spy_ret) == 0:
                return None
            return (1 + stock_ret) / (1 + spy_ret) - 1.0

        rs.rs_1m = rs_for_period(21)
        rs.rs_3m = rs_for_period(63)
        rs.rs_6m = rs_for_period(126)
        rs.rs_12m = rs_for_period(252)

        # Weighted composite: 10% 1m + 20% 3m + 30% 6m + 40% 12m
        weights = [(rs.rs_1m, 0.10), (rs.rs_3m, 0.20), (rs.rs_6m, 0.30), (rs.rs_12m, 0.40)]
        valid = [(v, w) for v, w in weights if v is not None]
        if valid:
            total_w = sum(w for _, w in valid)
            rs.rs_composite = sum(v * w for v, w in valid) / total_w
        else:
            flags.append("No RS periods available for composite calculation.")

        # Signal
        c = rs.rs_composite
        if c is None:
            rs.rs_signal = "UNKNOWN"
        elif c >= 0.10:
            rs.rs_signal = "STRONG"
        elif c >= 0.02:
            rs.rs_signal = "MODERATE"
        elif c >= -0.05:
            rs.rs_signal = "WEAK"
        else:
            rs.rs_signal = "LAGGARD"

        return rs

    # ------------------------------------------------------------------
    # 2. Momentum ROC
    # ------------------------------------------------------------------

    def _compute_momentum_roc(self, close: pd.Series, flags: list[str]) -> MomentumROC:
        mom = MomentumROC()

        mom.roc_1m = _safe_return(close, 21)
        mom.roc_3m = _safe_return(close, 63)
        mom.roc_6m = _safe_return(close, 126)
        mom.roc_12m = _safe_return(close, 252)

        if len(close) < 22:
            flags.append("Less than 22 days of data — only partial momentum available.")

        # Canonical cross-sectional momentum: 12m ROC minus 1m ROC
        # (skip the most-recent month to avoid short-term reversal noise)
        if mom.roc_12m is not None and mom.roc_1m is not None:
            mom.momentum_12_1 = mom.roc_12m - mom.roc_1m

        # Signal
        c = mom.momentum_12_1 if mom.momentum_12_1 is not None else mom.roc_3m
        if c is None:
            mom.momentum_signal = "UNKNOWN"
        elif c >= 0.20:
            mom.momentum_signal = "STRONG_BULL"
        elif c >= 0.05:
            mom.momentum_signal = "BULL"
        elif c >= -0.05:
            mom.momentum_signal = "NEUTRAL"
        elif c >= -0.20:
            mom.momentum_signal = "BEAR"
        else:
            mom.momentum_signal = "STRONG_BEAR"

        return mom

    # ------------------------------------------------------------------
    # 3. Price Structure
    # ------------------------------------------------------------------

    def _compute_price_structure(
        self,
        history: pd.DataFrame,
        close: pd.Series,
        current_price: float,
        flags: list[str],
    ) -> PriceStructure:
        ps = PriceStructure()

        # 52-week high / low from full history
        high_52w = float(close.max())
        low_52w = float(close.min())

        if high_52w > 0:
            ps.pct_from_52w_high = (current_price - high_52w) / high_52w
        if low_52w > 0:
            ps.pct_from_52w_low = (current_price - low_52w) / low_52w

        ps.is_near_52w_high = (
            ps.pct_from_52w_high is not None and ps.pct_from_52w_high >= -0.03
        )
        ps.is_at_52w_low = (
            ps.pct_from_52w_low is not None and ps.pct_from_52w_low <= 0.03
        )

        # VWAP — requires OHLCV columns
        required_cols = {"High", "Low", "Close", "Volume"}
        if required_cols.issubset(set(history.columns)):
            ps.vwap_90d = _compute_vwap(history, window=90)
            ps.vwap_ytd = _compute_vwap(history, window=self._ytd_trading_days(history))
        else:
            flags.append("OHLCV columns incomplete — VWAP not computed.")

        if ps.vwap_90d and ps.vwap_90d > 0:
            ps.price_vs_vwap_90d = (current_price - ps.vwap_90d) / ps.vwap_90d
        if ps.vwap_ytd and ps.vwap_ytd > 0:
            ps.price_vs_vwap_ytd = (current_price - ps.vwap_ytd) / ps.vwap_ytd

        # Structure signal
        if ps.is_near_52w_high and (ps.price_vs_vwap_90d or 0) > 0:
            ps.structure_signal = "BREAKOUT"
        elif (ps.pct_from_52w_high or -1) >= -0.10 and (ps.price_vs_vwap_90d or 0) > 0:
            ps.structure_signal = "STRONG"
        elif ps.is_at_52w_low or (ps.price_vs_vwap_90d or 0) < -0.05:
            ps.structure_signal = "DISTRIBUTION"
        elif (ps.pct_from_52w_high or -1) < -0.20:
            ps.structure_signal = "WEAK"
        else:
            ps.structure_signal = "MIXED"

        return ps

    @staticmethod
    def _ytd_trading_days(history: pd.DataFrame) -> int:
        """Estimate the number of trading days since January 1 of the current year."""
        try:
            if history.index.tz is not None:
                today = pd.Timestamp.now(tz=history.index.tz)
            else:
                today = pd.Timestamp.now()
            year_start = pd.Timestamp(today.year, 1, 1)
            if history.index.tz is not None:
                year_start = year_start.tz_localize(history.index.tz)
            ytd_mask = history.index >= year_start
            count = int(ytd_mask.sum())
            return max(count, 1)
        except Exception:
            return 63  # fallback to ~3 months

    # ------------------------------------------------------------------
    # 4. Short Interest
    # ------------------------------------------------------------------

    def _compute_short_interest(self, info: dict, flags: list[str]) -> ShortInterest:
        si = ShortInterest()

        raw_short_pct = info.get("shortPercentOfFloat")
        raw_days = info.get("shortRatio")
        raw_shares = info.get("sharesShort")

        if raw_short_pct is not None:
            try:
                si.short_pct_float = float(raw_short_pct)
            except (TypeError, ValueError):
                pass

        if raw_days is not None:
            try:
                si.days_to_cover = float(raw_days)
            except (TypeError, ValueError):
                pass

        if raw_shares is not None:
            try:
                si.shares_short = int(raw_shares)
            except (TypeError, ValueError):
                pass

        if si.short_pct_float is None and si.days_to_cover is None:
            flags.append("Short interest data unavailable in info dict.")
            si.short_signal = "UNKNOWN"
            return si

        dtc = si.days_to_cover or 0
        spf = si.short_pct_float or 0

        if dtc > 10 or spf > 0.20:
            si.short_signal = "SQUEEZE_RISK"
        elif dtc > 5 or spf > 0.10:
            si.short_signal = "HIGH"
        elif dtc > 2 or spf > 0.05:
            si.short_signal = "MODERATE"
        else:
            si.short_signal = "LOW"

        return si

    # ------------------------------------------------------------------
    # 5. Options Signals
    # ------------------------------------------------------------------

    def _compute_options_signals(
        self,
        ticker: str,
        current_price: float,
        flags: list[str],
    ) -> OptionsSignals:
        opts = OptionsSignals()
        try:
            stock = yf.Ticker(ticker)
            expirations = stock.options
            if not expirations:
                flags.append("No options data available for this ticker.")
                return opts

            # Pick nearest expiry with DTE >= 5
            today = date.today()
            chosen_exp = None
            chosen_dte = None
            for exp_str in expirations:
                try:
                    exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                    dte = (exp_date - today).days
                    if dte >= 5:
                        chosen_exp = exp_str
                        chosen_dte = dte
                        break
                except ValueError:
                    continue

            if chosen_exp is None:
                flags.append("No option expiry with DTE >= 5 found.")
                return opts

            opts.has_options = True
            opts.nearest_expiry = chosen_exp
            opts.dte = chosen_dte

            chain = stock.option_chain(chosen_exp)
            calls = chain.calls
            puts = chain.puts

            if calls.empty or puts.empty:
                flags.append(f"Empty option chain for expiry {chosen_exp}.")
                return opts

            # Find ATM call and put (closest strikes to current price)
            atm_call = self._get_atm_option(calls, current_price)
            atm_put = self._get_atm_option(puts, current_price)

            # Implied expected move: (atm_call + atm_put) / current_price
            if atm_call is not None and atm_put is not None:
                atm_call_price = self._option_mid(atm_call)
                atm_put_price = self._option_mid(atm_put)
                if atm_call_price and atm_put_price and current_price > 0:
                    opts.implied_move_pct = (atm_call_price + atm_put_price) / current_price

            # ATM IV for skew reference
            atm_iv = None
            if atm_call is not None and "impliedVolatility" in atm_call:
                iv_val = atm_call.get("impliedVolatility")
                if iv_val and not (isinstance(iv_val, float) and np.isnan(iv_val)) and iv_val > 0:
                    atm_iv = float(iv_val)

            if atm_iv is None and atm_put is not None and "impliedVolatility" in atm_put:
                iv_val = atm_put.get("impliedVolatility")
                if iv_val and not (isinstance(iv_val, float) and np.isnan(iv_val)) and iv_val > 0:
                    atm_iv = float(iv_val)

            # Skew: OTM put at ~90% strike, OTM call at ~110% strike
            otm_put_strike = current_price * 0.90
            otm_call_strike = current_price * 1.10

            otm_put_iv = self._get_nearest_iv(puts, otm_put_strike)
            otm_call_iv = self._get_nearest_iv(calls, otm_call_strike)

            if atm_iv and otm_put_iv:
                opts.put_skew = otm_put_iv - atm_iv
            if atm_iv and otm_call_iv:
                opts.call_skew = otm_call_iv - atm_iv

            if otm_put_iv and otm_call_iv and otm_call_iv > 0:
                opts.skew_ratio = otm_put_iv / otm_call_iv

            # Skew signal
            if opts.put_skew is not None:
                if opts.put_skew > 0.15:
                    opts.skew_signal = "FEAR"
                elif opts.put_skew > 0.07:
                    opts.skew_signal = "ELEVATED"
                elif opts.put_skew is not None and opts.put_skew < -0.05:
                    opts.skew_signal = "COMPLACENT"
                else:
                    opts.skew_signal = "NEUTRAL"
            else:
                opts.skew_signal = "UNKNOWN"

        except Exception as e:
            logger.warning("Options signals failed for %s: %s", ticker, e)
            flags.append(f"Options analysis error: {e!s:.80s}")

        return opts

    @staticmethod
    def _get_atm_option(chain_df: pd.DataFrame, price: float) -> Optional[dict]:
        """Return the row (as dict) of the strike closest to current price."""
        if chain_df.empty or "strike" not in chain_df.columns:
            return None
        chain_df = chain_df.copy()
        chain_df["_dist"] = (chain_df["strike"] - price).abs()
        row = chain_df.nsmallest(1, "_dist")
        if row.empty:
            return None
        return row.iloc[0].to_dict()

    @staticmethod
    def _option_mid(option_row: dict) -> Optional[float]:
        """Return mid-price (bid+ask)/2 or lastPrice as fallback."""
        bid = option_row.get("bid")
        ask = option_row.get("ask")
        last = option_row.get("lastPrice")
        if bid is not None and ask is not None and bid > 0 and ask > 0:
            return (float(bid) + float(ask)) / 2.0
        if last is not None and last > 0:
            return float(last)
        return None

    @staticmethod
    def _get_nearest_iv(chain_df: pd.DataFrame, target_strike: float) -> Optional[float]:
        """Return impliedVolatility of the strike nearest to target_strike."""
        if chain_df.empty or "strike" not in chain_df.columns or "impliedVolatility" not in chain_df.columns:
            return None
        valid = chain_df.dropna(subset=["impliedVolatility"])
        valid = valid[valid["impliedVolatility"] > 0]
        if valid.empty:
            return None
        valid = valid.copy()
        valid["_dist"] = (valid["strike"] - target_strike).abs()
        row = valid.nsmallest(1, "_dist")
        if row.empty:
            return None
        return float(row.iloc[0]["impliedVolatility"])

    # ------------------------------------------------------------------
    # 6. Overall scoring
    # ------------------------------------------------------------------

    def _compute_overall(self, result: TechnicalEnhancedResult) -> None:
        """
        overall_score = RS(30%) + Momentum(30%) + Structure(20%) + Short(10%) + Options(10%)
        Each component normalised to [-1.0, +1.0].
        """
        rs_score = self._score_rs(result.rs)
        mom_score = self._score_momentum(result.momentum)
        struct_score = self._score_structure(result.structure)
        short_score = self._score_short_interest(result.short_interest)
        options_score = self._score_options(result.options)

        result.overall_score = round(
            rs_score * 0.30
            + mom_score * 0.30
            + struct_score * 0.20
            + short_score * 0.10
            + options_score * 0.10,
            4,
        )

        s = result.overall_score
        if s >= 0.30:
            result.signal = "BULLISH MOMENTUM"
        elif s >= 0.10:
            result.signal = "MILD BULLISH"
        elif s >= -0.10:
            result.signal = "NEUTRAL"
        elif s >= -0.30:
            result.signal = "MILD BEARISH"
        else:
            result.signal = "BEARISH MOMENTUM"

        result.summary = self._build_summary(result)

    # ------------------------------------------------------------------
    # Score helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _score_rs(rs: RSMetrics) -> float:
        """RS composite → normalised score. >0.05 = outperforming → bullish."""
        if rs.rs_composite is None:
            return 0.0
        c = rs.rs_composite
        if c >= 0.20:
            return 1.0
        elif c >= 0.10:
            return 0.7
        elif c >= 0.05:
            return 0.4
        elif c >= 0.0:
            return 0.1
        elif c >= -0.05:
            return -0.2
        elif c >= -0.10:
            return -0.5
        else:
            return -1.0

    @staticmethod
    def _score_momentum(mom: MomentumROC) -> float:
        """12-1 momentum → normalised score."""
        m = mom.momentum_12_1 if mom.momentum_12_1 is not None else mom.roc_3m
        if m is None:
            return 0.0
        if m >= 0.30:
            return 1.0
        elif m >= 0.15:
            return 0.6
        elif m >= 0.05:
            return 0.3
        elif m >= -0.05:
            return 0.0
        elif m >= -0.15:
            return -0.4
        elif m >= -0.25:
            return -0.7
        else:
            return -1.0

    @staticmethod
    def _score_structure(ps: PriceStructure) -> float:
        """52w positioning + VWAP relationship → score."""
        score = 0.0

        # 52-week position
        if ps.is_near_52w_high:
            score += 0.5
        elif ps.pct_from_52w_high is not None:
            d = ps.pct_from_52w_high  # negative number
            if d >= -0.10:
                score += 0.25
            elif d >= -0.25:
                score += 0.0
            elif d >= -0.40:
                score -= 0.25
            else:
                score -= 0.5

        if ps.is_at_52w_low:
            score -= 0.5

        # VWAP above/below
        vwap_score = 0.0
        if ps.price_vs_vwap_90d is not None:
            vwap_score += 0.3 if ps.price_vs_vwap_90d > 0 else -0.3
        if ps.price_vs_vwap_ytd is not None:
            vwap_score += 0.2 if ps.price_vs_vwap_ytd > 0 else -0.2
        score += vwap_score

        return max(-1.0, min(1.0, score))

    @staticmethod
    def _score_short_interest(si: ShortInterest) -> float:
        """Short interest as contrarian signal.
        High days-to-cover with improving technicals → squeeze potential (+).
        Very high short with no apparent catalyst → mild negative.
        """
        if si.short_signal == "UNKNOWN":
            return 0.0
        # Contrarian framing: high short = potential squeeze catalyst
        if si.short_signal == "SQUEEZE_RISK":
            return 0.3     # squeeze risk is a bullish contrarian signal
        elif si.short_signal == "HIGH":
            return 0.15
        elif si.short_signal == "MODERATE":
            return 0.0
        else:  # LOW
            return 0.0

    @staticmethod
    def _score_options(opts: OptionsSignals) -> float:
        """Skew-based sentiment score.
        Neutral skew = 0; extreme fear (steep put skew) = -0.3; complacency = mild positive.
        """
        if not opts.has_options:
            return 0.0
        if opts.skew_signal == "FEAR":
            return -0.3
        elif opts.skew_signal == "ELEVATED":
            return -0.15
        elif opts.skew_signal == "NEUTRAL":
            return 0.0
        elif opts.skew_signal == "COMPLACENT":
            return 0.1
        return 0.0

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    @staticmethod
    def _build_summary(result: TechnicalEnhancedResult) -> str:
        parts: list[str] = []

        if result.rs.rs_composite is not None:
            parts.append(
                f"RS vs SPY: {result.rs.rs_composite:+.1%} ({result.rs.rs_signal})"
            )

        if result.momentum.momentum_12_1 is not None:
            parts.append(
                f"12-1 Momentum: {result.momentum.momentum_12_1:+.1%} ({result.momentum.momentum_signal})"
            )

        if result.structure.pct_from_52w_high is not None:
            parts.append(
                f"52w High proximity: {result.structure.pct_from_52w_high:+.1%} ({result.structure.structure_signal})"
            )

        if result.structure.price_vs_vwap_90d is not None:
            direction = "above" if result.structure.price_vs_vwap_90d > 0 else "below"
            parts.append(
                f"90d VWAP: {abs(result.structure.price_vs_vwap_90d):.1%} {direction}"
            )

        if result.short_interest.short_signal not in ("UNKNOWN",):
            si_detail = ""
            if result.short_interest.days_to_cover is not None:
                si_detail = f" (DTC: {result.short_interest.days_to_cover:.1f}d)"
            parts.append(f"Short interest: {result.short_interest.short_signal}{si_detail}")

        if result.options.has_options:
            if result.options.implied_move_pct is not None:
                parts.append(f"Implied move: ±{result.options.implied_move_pct:.1%}")
            parts.append(f"Skew: {result.options.skew_signal}")

        base = f"[{result.signal} | Score: {result.overall_score:+.2f}]"
        if parts:
            return f"{base} " + " · ".join(parts)
        return base
