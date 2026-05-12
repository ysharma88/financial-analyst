#!/usr/bin/env python3
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger("fa_strategy")


@dataclass
class StrategyConfig:
    rsi_period: int = 14
    rsi_oversold: float = 28.0
    rsi_overbought: float = 72.0
    momentum_period: int = 14
    momentum_threshold: float = 0.04


def _ma(values: list[float], period: int) -> float | None:
    if len(values) < period:
        return None
    return sum(values[-period:]) / period


def _rsi(prices: list[float], period: int = 14) -> float | None:
    if len(prices) < period + 1:
        return None
    gains, losses = [], []
    for i in range(1, len(prices)):
        d = prices[i] - prices[i - 1]
        gains.append(max(d, 0))
        losses.append(max(-d, 0))
    ag = sum(gains[-period:]) / period
    al = sum(losses[-period:]) / period
    if al == 0:
        return 100.0
    rs = ag / al
    return 100 - (100 / (1 + rs))


def _momentum(prices: list[float], period: int = 14) -> float | None:
    if len(prices) < period + 1:
        return None
    return (prices[-1] - prices[-period - 1]) / prices[-period - 1]


def _bollinger(prices: list[float], period: int = 20) -> tuple[float | None, float | None, float | None, float | None]:
    if len(prices) < period:
        return None, None, None, None
    recent = prices[-period:]
    mid = sum(recent) / period
    var = sum((p - mid) ** 2 for p in recent) / period
    std = var ** 0.5
    up = mid + 2 * std
    low = mid - 2 * std
    pct_b = (prices[-1] - low) / (up - low) if up != low else 0.5
    return up, mid, low, pct_b


def run_strategy(symbol: str, prices: list[float], volumes: list[float], position=None, config: StrategyConfig | None = None, alert_on_signal: bool = True):
    config = config or StrategyConfig()
    if len(prices) < 50:
        return {"symbol": symbol, "timestamp": datetime.now().isoformat(), "signal": "hold", "reason": "Insufficient data", "metrics": {"price": prices[-1] if prices else None}}
    rsi = _rsi(prices, config.rsi_period)
    mom = _momentum(prices, config.momentum_period)
    _, _, _, pct_b = _bollinger(prices, 20)
    ma50 = _ma(prices, 50)
    signal, reason = "hold", "No signal"
    if rsi is not None and rsi < config.rsi_oversold and ma50 and prices[-1] >= ma50:
        signal, reason = "buy", f"Oversold RSI={rsi:.1f}"
    elif mom is not None and mom > config.momentum_threshold and ma50 and prices[-1] >= ma50:
        signal, reason = "buy", f"Momentum +{mom*100:.1f}%"
    elif rsi is not None and rsi > config.rsi_overbought:
        signal, reason = "sell", f"Overbought RSI={rsi:.1f}"
    elif pct_b is not None and pct_b > 1:
        signal, reason = "sell", f"Bollinger overbought %B={pct_b:.2f}"
    if alert_on_signal and signal != "hold":
        logger.warning("ALERT [%s] %s: %s", symbol, signal.upper(), reason)
    trade = None
    if signal == "buy":
        qty = max(1, int(1000 / prices[-1]))
        trade = {"side": "buy", "price": prices[-1], "quantity": qty, "pnl_pct": 0.0}
    return {
        "symbol": symbol,
        "timestamp": datetime.now().isoformat(),
        "signal": signal,
        "reason": reason,
        "trade": trade,
        "metrics": {
            "price": prices[-1],
            "rsi": round(rsi, 2) if rsi is not None else None,
            "momentum_pct": round(mom * 100, 2) if mom is not None else None,
            "bollinger_pct_b": round(pct_b, 2) if pct_b is not None else None,
            "atr": round(abs(prices[-1] - prices[-2]), 2),
        },
    }

