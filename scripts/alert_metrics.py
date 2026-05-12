#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from statistics import median

def calculate_rsi(prices: list[float], period: int = 14) -> float | None:
    if len(prices) < period + 1:
        return None
    gains: list[float] = []
    losses: list[float] = []
    for i in range(1, len(prices)):
        d = prices[i] - prices[i - 1]
        gains.append(max(d, 0))
        losses.append(max(-d, 0))
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_atr(prices: list[float], period: int = 14) -> float | None:
    if len(prices) < period + 1:
        return None
    ranges = [abs(prices[i] - prices[i - 1]) for i in range(1, len(prices))]
    return sum(ranges[-period:]) / period


DEFAULT_ALERT_THRESHOLDS = {
    "fundamental": {
        "debt_to_equity_gte": 2.0,
        "roe_lt": 0.15,
        "dividend_payout_gt": 0.75,
        "peg_lt": 1.0,
        "price_to_sales_gt": 50.0,
        "wacc_assumed": 0.10,
        "roic_lt_wacc": True,
        "insider_cluster_selling": True,
    },
    "technical": {
        "rsi_overbought": 70.0,
        "rsi_oversold": 30.0,
        "atr_spike_vs_median_pct": 25.0,
        "support_break_lookback_days": 60,
        "resistance_break_lookback_days": 60,
    },
    "macro": {
        "vix_crisis_gt": 30.0,
        "vix_calm_lt": 20.0,
        "buffett_indicator_gt": 120.0,
        "yield_curve_inversion": True,
        "inflation_surprise_abs_gt": 0.2,
    },
    "institutional": {
        "ownership_change_gt_pct": 5.0,
        "sentiment_abs_gt": 0.225,
        "earnings_revision_bottom_pct": 40.0,
    },
}


@dataclass
class FundamentalSnapshot:
    debt_to_equity: float | None = None
    roe: float | None = None
    dividend_payout_ratio: float | None = None
    peg_ratio: float | None = None
    price_to_sales: float | None = None
    roic: float | None = None
    wacc: float | None = None
    insider_cluster_selling: bool | None = None


@dataclass
class TechnicalSnapshot:
    prices: list[float]
    rsi: float | None = None
    atr: float | None = None
    atr_median_6m: float | None = None
    prev_ma50: float | None = None
    prev_ma200: float | None = None
    ma50: float | None = None
    ma200: float | None = None
    support_level: float | None = None
    resistance_level: float | None = None


@dataclass
class MacroSnapshot:
    vix: float | None = None
    buffett_indicator_pct: float | None = None
    us10y: float | None = None
    us2y: float | None = None
    inflation_surprise_pct: float | None = None


@dataclass
class InstitutionalSnapshot:
    ownership_change_pct: float | None = None
    sentiment_score: float | None = None
    earnings_revision_percentile: float | None = None


def _moving_average(values: list[float], period: int) -> float | None:
    if len(values) < period:
        return None
    return sum(values[-period:]) / period


def _moving_average_prev(values: list[float], period: int) -> float | None:
    if len(values) < period + 1:
        return None
    return sum(values[-period - 1 : -1]) / period


def evaluate_fundamental_alerts(snapshot: FundamentalSnapshot, cfg: dict) -> list[str]:
    alerts: list[str] = []
    if snapshot.debt_to_equity is not None and snapshot.debt_to_equity >= cfg["debt_to_equity_gte"]:
        alerts.append(f"Fundamental: D/E elevated ({snapshot.debt_to_equity:.2f} >= {cfg['debt_to_equity_gte']})")
    if snapshot.roe is not None and snapshot.roe < cfg["roe_lt"]:
        alerts.append(f"Fundamental: ROE weak ({snapshot.roe*100:.1f}% < {cfg['roe_lt']*100:.1f}%)")
    if snapshot.dividend_payout_ratio is not None and snapshot.dividend_payout_ratio > cfg["dividend_payout_gt"]:
        alerts.append(f"Fundamental: Dividend payout fragile ({snapshot.dividend_payout_ratio*100:.1f}% > {cfg['dividend_payout_gt']*100:.1f}%)")
    if snapshot.peg_ratio is not None and snapshot.peg_ratio < cfg["peg_lt"]:
        alerts.append(f"Fundamental: PEG undervaluation signal ({snapshot.peg_ratio:.2f} < {cfg['peg_lt']})")
    if snapshot.price_to_sales is not None and snapshot.price_to_sales > cfg["price_to_sales_gt"]:
        alerts.append(f"Fundamental: P/S bubble risk ({snapshot.price_to_sales:.1f}x > {cfg['price_to_sales_gt']}x)")
    if cfg.get("roic_lt_wacc", True):
        wacc = snapshot.wacc if snapshot.wacc is not None else cfg.get("wacc_assumed", 0.10)
        if snapshot.roic is not None and wacc is not None and snapshot.roic < wacc:
            alerts.append(f"Fundamental: ROIC below WACC ({snapshot.roic*100:.1f}% < {wacc*100:.1f}%)")
    if cfg.get("insider_cluster_selling", True) and snapshot.insider_cluster_selling:
        alerts.append("Fundamental: Insider cluster selling detected")
    return alerts


def evaluate_technical_alerts(snapshot: TechnicalSnapshot, cfg: dict) -> list[str]:
    alerts: list[str] = []
    prices = snapshot.prices
    if not prices:
        return alerts

    rsi = snapshot.rsi if snapshot.rsi is not None else calculate_rsi(prices, 14)
    if rsi is not None:
        if rsi > cfg["rsi_overbought"]:
            alerts.append(f"Technical: RSI overbought ({rsi:.1f} > {cfg['rsi_overbought']})")
        if rsi < cfg["rsi_oversold"]:
            alerts.append(f"Technical: RSI oversold ({rsi:.1f} < {cfg['rsi_oversold']})")

    ma50 = snapshot.ma50 if snapshot.ma50 is not None else _moving_average(prices, 50)
    ma200 = snapshot.ma200 if snapshot.ma200 is not None else _moving_average(prices, 200)
    pma50 = snapshot.prev_ma50 if snapshot.prev_ma50 is not None else _moving_average_prev(prices, 50)
    pma200 = snapshot.prev_ma200 if snapshot.prev_ma200 is not None else _moving_average_prev(prices, 200)
    if ma50 is not None and ma200 is not None and pma50 is not None and pma200 is not None:
        if pma50 <= pma200 and ma50 > ma200:
            alerts.append("Technical: Golden Cross (50D crossed above 200D)")
        if pma50 >= pma200 and ma50 < ma200:
            alerts.append("Technical: Death Cross (50D crossed below 200D)")

    atr = snapshot.atr if snapshot.atr is not None else calculate_atr(prices, 14)
    atr_median = snapshot.atr_median_6m
    if atr is not None and atr_median is None and len(prices) >= 180:
        atr_series = []
        for i in range(20, len(prices)):
            v = calculate_atr(prices[: i + 1], 14)
            if v is not None:
                atr_series.append(v)
        if atr_series:
            atr_median = median(atr_series[-126:]) if len(atr_series) >= 126 else median(atr_series)
    if atr is not None and atr_median and atr_median > 0:
        spike = ((atr - atr_median) / atr_median) * 100
        if spike >= cfg["atr_spike_vs_median_pct"]:
            alerts.append(f"Technical: ATR spike vs 6M median ({spike:.1f}%)")

    lookback_support = int(cfg.get("support_break_lookback_days", 60))
    if len(prices) > lookback_support:
        support = snapshot.support_level if snapshot.support_level is not None else min(prices[-lookback_support:-1])
        if prices[-1] < support:
            alerts.append(f"Technical: Support break ({prices[-1]:.2f} < {support:.2f})")
    lookback_res = int(cfg.get("resistance_break_lookback_days", 60))
    if len(prices) > lookback_res:
        resistance = snapshot.resistance_level if snapshot.resistance_level is not None else max(prices[-lookback_res:-1])
        if prices[-1] > resistance:
            alerts.append(f"Technical: Resistance breakout ({prices[-1]:.2f} > {resistance:.2f})")
    return alerts


def evaluate_macro_alerts(snapshot: MacroSnapshot, cfg: dict) -> list[str]:
    alerts: list[str] = []
    if snapshot.vix is not None:
        if snapshot.vix > cfg["vix_crisis_gt"]:
            alerts.append(f"Macro: VIX crisis regime ({snapshot.vix:.1f} > {cfg['vix_crisis_gt']})")
        elif snapshot.vix < cfg["vix_calm_lt"]:
            alerts.append(f"Macro: VIX calm regime ({snapshot.vix:.1f} < {cfg['vix_calm_lt']})")
    if snapshot.buffett_indicator_pct is not None and snapshot.buffett_indicator_pct > cfg["buffett_indicator_gt"]:
        alerts.append(f"Macro: Buffett indicator overvaluation ({snapshot.buffett_indicator_pct:.1f}% > {cfg['buffett_indicator_gt']}%)")
    if cfg.get("yield_curve_inversion", True) and snapshot.us10y is not None and snapshot.us2y is not None:
        if snapshot.us10y - snapshot.us2y < 0:
            alerts.append(f"Macro: Yield curve inversion ({snapshot.us10y:.2f} - {snapshot.us2y:.2f} < 0)")
    if snapshot.inflation_surprise_pct is not None and abs(snapshot.inflation_surprise_pct) > cfg["inflation_surprise_abs_gt"]:
        alerts.append(f"Macro: Inflation surprise ({snapshot.inflation_surprise_pct:+.2f}% exceeds ±{cfg['inflation_surprise_abs_gt']:.2f}%)")
    return alerts


def evaluate_institutional_alerts(snapshot: InstitutionalSnapshot, cfg: dict) -> list[str]:
    alerts: list[str] = []
    if snapshot.ownership_change_pct is not None and abs(snapshot.ownership_change_pct) > cfg["ownership_change_gt_pct"]:
        alerts.append(f"Institutional: Ownership shift ({snapshot.ownership_change_pct:+.1f}% > {cfg['ownership_change_gt_pct']}%)")
    if snapshot.sentiment_score is not None and abs(snapshot.sentiment_score) > cfg["sentiment_abs_gt"]:
        alerts.append(f"Institutional: Sentiment trigger ({snapshot.sentiment_score:+.3f})")
    if snapshot.earnings_revision_percentile is not None and snapshot.earnings_revision_percentile <= cfg["earnings_revision_bottom_pct"]:
        alerts.append(f"Institutional: Earnings revision weakness (bottom {snapshot.earnings_revision_percentile:.1f}%)")
    return alerts


def evaluate_all_alerts(
    fundamental: FundamentalSnapshot,
    technical: TechnicalSnapshot,
    macro: MacroSnapshot,
    institutional: InstitutionalSnapshot,
    thresholds: dict,
) -> list[str]:
    merged = DEFAULT_ALERT_THRESHOLDS.copy()
    for key in ("fundamental", "technical", "macro", "institutional"):
        base = dict(DEFAULT_ALERT_THRESHOLDS.get(key, {}))
        base.update((thresholds or {}).get(key, {}))
        merged[key] = base
    alerts: list[str] = []
    alerts.extend(evaluate_fundamental_alerts(fundamental, merged["fundamental"]))
    alerts.extend(evaluate_technical_alerts(technical, merged["technical"]))
    alerts.extend(evaluate_macro_alerts(macro, merged["macro"]))
    alerts.extend(evaluate_institutional_alerts(institutional, merged["institutional"]))
    return alerts

