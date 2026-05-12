#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
import urllib.request
from datetime import datetime

logger = logging.getLogger("fa_data_feed")

try:
    import yfinance as yf
    YF_AVAILABLE = True
except Exception:
    YF_AVAILABLE = False


def fetch_yahoo(symbol: str, period: str = "3mo", interval: str = "1d") -> tuple[list[float], list[float], list[datetime]] | None:
    if not YF_AVAILABLE:
        return None
    try:
        df = yf.Ticker(symbol).history(period=period, interval=interval)
        if df.empty:
            return None
        return df["Close"].tolist(), df["Volume"].fillna(0).tolist(), df.index.tolist()
    except Exception as exc:
        logger.error("yahoo fetch failed for %s: %s", symbol, exc)
        return None


def fetch_alpha_vantage(symbol: str, api_key: str) -> tuple[list[float], list[float], list[str]] | None:
    try:
        url = (
            "https://www.alphavantage.co/query"
            f"?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}&outputsize=compact"
        )
        with urllib.request.urlopen(url, timeout=12) as r:
            data = json.loads(r.read().decode())
        ts = data.get("Time Series (Daily)", {})
        if not ts:
            return None
        dates = sorted(ts.keys())[-100:]
        prices = [float(ts[d]["4. close"]) for d in dates]
        volumes = [float(ts[d]["5. volume"]) for d in dates]
        return prices, volumes, dates
    except Exception as exc:
        logger.error("alpha fetch failed for %s: %s", symbol, exc)
        return None


def get_data(symbol: str, source: str = "yahoo", period: str = "3mo", interval: str = "1d", api_key: str | None = None) -> tuple[list[float], list[float]] | None:
    if source == "yahoo":
        out = fetch_yahoo(symbol, period=period, interval=interval)
    elif source == "alphavantage" and api_key:
        out = fetch_alpha_vantage(symbol, api_key)
    else:
        return None
    if out is None:
        return None
    prices, volumes, _ = out
    return prices, volumes

