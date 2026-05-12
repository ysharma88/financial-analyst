"""Technical analysis engine - evaluates price action, momentum, trend, and volume."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import numpy as np
import ta


@dataclass
class TechnicalSignal:
    name: str
    value: Optional[float]
    score: float  # -1.0 (very bearish) to +1.0 (very bullish)
    weight: float
    interpretation: str
    category: str


@dataclass
class SupportResistanceLevel:
    price: float
    kind: str           # "support" or "resistance"
    strength: str       # "strong", "moderate", "weak"
    method: str         # how it was detected
    touches: int = 0    # how many times price tested this level


@dataclass
class TechnicalResult:
    signals: list[TechnicalSignal] = field(default_factory=list)
    overall_score: float = 0.0
    signal: str = "NEUTRAL"
    summary: str = ""
    indicators: dict = field(default_factory=dict)
    support_resistance: list[SupportResistanceLevel] = field(default_factory=list)

    def compute_overall(self):
        valid = [s for s in self.signals if s.value is not None]
        if not valid:
            self.overall_score = 0.0
            self.signal = "NEUTRAL"
            self.summary = "Insufficient technical data."
            return

        total_weight = sum(s.weight for s in valid)
        if total_weight == 0:
            self.overall_score = 0.0
        else:
            self.overall_score = sum(s.score * s.weight for s in valid) / total_weight

        if self.overall_score >= 0.5:
            self.signal = "STRONG BUY"
        elif self.overall_score >= 0.2:
            self.signal = "BUY"
        elif self.overall_score >= -0.2:
            self.signal = "HOLD"
        elif self.overall_score >= -0.5:
            self.signal = "SELL"
        else:
            self.signal = "STRONG SELL"

        bullish = [s.name for s in valid if s.score > 0.3]
        bearish = [s.name for s in valid if s.score < -0.3]
        parts = []
        if bullish:
            parts.append(f"Bullish: {', '.join(bullish[:4])}")
        if bearish:
            parts.append(f"Bearish: {', '.join(bearish[:4])}")
        self.summary = ". ".join(parts) if parts else "Mixed technical signals."

    @property
    def category_scores(self) -> dict[str, float]:
        categories: dict[str, list] = {}
        for s in self.signals:
            if s.value is not None:
                categories.setdefault(s.category, []).append(s.score)
        return {cat: sum(vals) / len(vals) for cat, vals in categories.items()}


class TechnicalAnalyzer:
    """Computes and scores technical indicators on OHLCV data."""

    def analyze(self, df: pd.DataFrame) -> TechnicalResult:
        if df is None or len(df) < 20:
            result = TechnicalResult()
            result.summary = "Not enough historical data for technical analysis."
            return result

        df = df.copy()
        result = TechnicalResult()

        self._compute_indicators(df)
        result.indicators = self._get_indicator_dict(df)

        close = df["Close"].iloc[-1]

        result.signals.extend(self._score_moving_averages(df, close))
        result.signals.extend(self._score_rsi(df))
        result.signals.extend(self._score_macd(df))
        result.signals.extend(self._score_bollinger(df, close))
        result.signals.extend(self._score_stochastic(df))
        result.signals.extend(self._score_volume(df))
        result.signals.extend(self._score_atr(df, close))
        result.signals.extend(self._score_trend_strength(df))

        result.support_resistance = self._compute_support_resistance(df)

        result.compute_overall()
        return result

    def _compute_indicators(self, df: pd.DataFrame):
        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        volume = df["Volume"]

        # Moving Averages
        df["SMA_20"] = ta.trend.sma_indicator(close, window=20)
        df["SMA_50"] = ta.trend.sma_indicator(close, window=50)
        df["SMA_200"] = ta.trend.sma_indicator(close, window=200)
        df["EMA_12"] = ta.trend.ema_indicator(close, window=12)
        df["EMA_26"] = ta.trend.ema_indicator(close, window=26)

        # RSI
        df["RSI"] = ta.momentum.rsi(close, window=14)

        # MACD
        macd = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
        df["MACD"] = macd.macd()
        df["MACD_Signal"] = macd.macd_signal()
        df["MACD_Hist"] = macd.macd_diff()

        # Bollinger Bands
        bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
        df["BB_Upper"] = bb.bollinger_hband()
        df["BB_Middle"] = bb.bollinger_mavg()
        df["BB_Lower"] = bb.bollinger_lband()
        df["BB_Width"] = bb.bollinger_wband()

        # Stochastic
        stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
        df["Stoch_K"] = stoch.stoch()
        df["Stoch_D"] = stoch.stoch_signal()

        # ATR
        df["ATR"] = ta.volatility.average_true_range(high, low, close, window=14)

        # Volume
        df["OBV"] = ta.volume.on_balance_volume(close, volume)
        df["Volume_SMA_20"] = ta.trend.sma_indicator(volume.astype(float), window=20)

        # ADX
        adx = ta.trend.ADXIndicator(high, low, close, window=14)
        df["ADX"] = adx.adx()

    def _get_indicator_dict(self, df: pd.DataFrame) -> dict:
        last = df.iloc[-1]
        indicators = {}
        for col in ["SMA_20", "SMA_50", "SMA_200", "EMA_12", "EMA_26",
                     "RSI", "MACD", "MACD_Signal", "MACD_Hist",
                     "BB_Upper", "BB_Middle", "BB_Lower", "BB_Width",
                     "Stoch_K", "Stoch_D", "ATR", "OBV", "ADX"]:
            val = last.get(col)
            if val is not None and not pd.isna(val):
                indicators[col] = float(val)
        return indicators

    def _score_moving_averages(self, df: pd.DataFrame, close: float) -> list[TechnicalSignal]:
        signals = []
        last = df.iloc[-1]

        sma20 = last.get("SMA_20")
        if sma20 is not None and not pd.isna(sma20):
            pct = (close - sma20) / sma20
            if pct > 0.05:
                sc, interp = 0.6, "Well above 20-day SMA (bullish)"
            elif pct > 0:
                sc, interp = 0.3, "Above 20-day SMA"
            elif pct > -0.05:
                sc, interp = -0.3, "Below 20-day SMA"
            else:
                sc, interp = -0.6, "Well below 20-day SMA (bearish)"
            signals.append(TechnicalSignal("SMA 20", round(sma20, 2), sc, 0.8, interp, "Trend"))

        sma50 = last.get("SMA_50")
        if sma50 is not None and not pd.isna(sma50):
            pct = (close - sma50) / sma50
            if pct > 0.08:
                sc, interp = 0.6, "Well above 50-day SMA (bullish)"
            elif pct > 0:
                sc, interp = 0.3, "Above 50-day SMA"
            elif pct > -0.08:
                sc, interp = -0.3, "Below 50-day SMA"
            else:
                sc, interp = -0.6, "Well below 50-day SMA (bearish)"
            signals.append(TechnicalSignal("SMA 50", round(sma50, 2), sc, 1.0, interp, "Trend"))

        sma200 = last.get("SMA_200")
        if sma200 is not None and not pd.isna(sma200):
            pct = (close - sma200) / sma200
            if pct > 0.1:
                sc, interp = 0.7, "Well above 200-day SMA (strong uptrend)"
            elif pct > 0:
                sc, interp = 0.3, "Above 200-day SMA (uptrend)"
            elif pct > -0.1:
                sc, interp = -0.4, "Below 200-day SMA (downtrend)"
            else:
                sc, interp = -0.7, "Well below 200-day SMA (strong downtrend)"
            signals.append(TechnicalSignal("SMA 200", round(sma200, 2), sc, 1.2, interp, "Trend"))

        # Golden/Death Cross
        if sma50 is not None and sma200 is not None and not pd.isna(sma50) and not pd.isna(sma200):
            prev = df.iloc[-5] if len(df) > 5 else df.iloc[0]
            prev_sma50 = prev.get("SMA_50")
            prev_sma200 = prev.get("SMA_200")
            if prev_sma50 is not None and prev_sma200 is not None:
                if sma50 > sma200 and prev_sma50 <= prev_sma200:
                    signals.append(TechnicalSignal("Golden Cross", 1, 0.8, 1.5, "SMA50 crossed above SMA200 (bullish)", "Trend"))
                elif sma50 < sma200 and prev_sma50 >= prev_sma200:
                    signals.append(TechnicalSignal("Death Cross", -1, -0.8, 1.5, "SMA50 crossed below SMA200 (bearish)", "Trend"))
                elif sma50 > sma200:
                    signals.append(TechnicalSignal("MA Alignment", 1, 0.3, 0.7, "SMA50 above SMA200 (bullish structure)", "Trend"))
                else:
                    signals.append(TechnicalSignal("MA Alignment", -1, -0.3, 0.7, "SMA50 below SMA200 (bearish structure)", "Trend"))

        return signals

    def _score_rsi(self, df: pd.DataFrame) -> list[TechnicalSignal]:
        rsi = df["RSI"].iloc[-1]
        if rsi is None or pd.isna(rsi):
            return []

        if rsi > 80:
            sc, interp = -0.8, "Extremely overbought"
        elif rsi > 70:
            sc, interp = -0.5, "Overbought territory"
        elif rsi > 60:
            sc, interp = 0.2, "Bullish momentum"
        elif rsi > 40:
            sc, interp = 0.0, "Neutral momentum"
        elif rsi > 30:
            sc, interp = -0.3, "Bearish momentum"
        elif rsi > 20:
            sc, interp = 0.4, "Oversold (potential bounce)"
        else:
            sc, interp = 0.6, "Extremely oversold (reversal likely)"

        return [TechnicalSignal("RSI (14)", round(rsi, 1), sc, 1.3, interp, "Momentum")]

    def _score_macd(self, df: pd.DataFrame) -> list[TechnicalSignal]:
        signals = []
        last = df.iloc[-1]

        macd = last.get("MACD")
        macd_signal = last.get("MACD_Signal")
        macd_hist = last.get("MACD_Hist")

        if macd is not None and macd_signal is not None and not pd.isna(macd) and not pd.isna(macd_signal):
            if macd > macd_signal and macd > 0:
                sc, interp = 0.7, "MACD bullish above zero line"
            elif macd > macd_signal:
                sc, interp = 0.3, "MACD bullish crossover"
            elif macd < macd_signal and macd < 0:
                sc, interp = -0.7, "MACD bearish below zero line"
            else:
                sc, interp = -0.3, "MACD bearish crossover"
            signals.append(TechnicalSignal("MACD", round(macd, 4), sc, 1.2, interp, "Momentum"))

        if macd_hist is not None and not pd.isna(macd_hist) and len(df) > 2:
            prev_hist = df["MACD_Hist"].iloc[-2]
            if not pd.isna(prev_hist):
                if macd_hist > 0 and macd_hist > prev_hist:
                    sc, interp = 0.4, "MACD histogram expanding (bullish)"
                elif macd_hist > 0:
                    sc, interp = 0.1, "MACD histogram contracting"
                elif macd_hist < prev_hist:
                    sc, interp = -0.4, "MACD histogram expanding (bearish)"
                else:
                    sc, interp = -0.1, "MACD histogram recovering"
                signals.append(TechnicalSignal("MACD Histogram", round(macd_hist, 4), sc, 0.7, interp, "Momentum"))

        return signals

    def _score_bollinger(self, df: pd.DataFrame, close: float) -> list[TechnicalSignal]:
        last = df.iloc[-1]
        bb_upper = last.get("BB_Upper")
        bb_lower = last.get("BB_Lower")
        bb_middle = last.get("BB_Middle")

        if bb_upper is None or bb_lower is None or pd.isna(bb_upper) or pd.isna(bb_lower):
            return []

        bb_range = bb_upper - bb_lower
        if bb_range == 0:
            return []

        position = (close - bb_lower) / bb_range

        if position > 1.0:
            sc, interp = -0.5, "Price above upper Bollinger Band (overbought)"
        elif position > 0.8:
            sc, interp = -0.2, "Price near upper Bollinger Band"
        elif position > 0.5:
            sc, interp = 0.2, "Price above middle band (bullish)"
        elif position > 0.2:
            sc, interp = -0.2, "Price below middle band"
        elif position > 0.0:
            sc, interp = 0.3, "Price near lower band (potential bounce)"
        else:
            sc, interp = 0.4, "Price below lower Bollinger Band (oversold)"

        return [TechnicalSignal("Bollinger Position", round(position, 2), sc, 0.9, interp, "Volatility")]

    def _score_stochastic(self, df: pd.DataFrame) -> list[TechnicalSignal]:
        last = df.iloc[-1]
        k = last.get("Stoch_K")
        d = last.get("Stoch_D")

        if k is None or pd.isna(k):
            return []

        if k > 80 and d is not None and k < d:
            sc, interp = -0.6, "Stochastic overbought with bearish crossover"
        elif k > 80:
            sc, interp = -0.3, "Stochastic overbought"
        elif k < 20 and d is not None and k > d:
            sc, interp = 0.6, "Stochastic oversold with bullish crossover"
        elif k < 20:
            sc, interp = 0.3, "Stochastic oversold"
        elif d is not None and k > d:
            sc, interp = 0.2, "Stochastic bullish crossover"
        elif d is not None:
            sc, interp = -0.2, "Stochastic bearish crossover"
        else:
            sc, interp = 0.0, "Stochastic neutral"

        return [TechnicalSignal("Stochastic", round(k, 1), sc, 0.8, interp, "Momentum")]

    def _score_volume(self, df: pd.DataFrame) -> list[TechnicalSignal]:
        signals = []
        last = df.iloc[-1]

        vol = last.get("Volume")
        vol_sma = last.get("Volume_SMA_20")

        if vol is not None and vol_sma is not None and not pd.isna(vol) and not pd.isna(vol_sma) and vol_sma > 0:
            vol_ratio = vol / vol_sma
            close_chg = 0
            if len(df) > 1:
                prev_close = df["Close"].iloc[-2]
                if prev_close > 0:
                    close_chg = (last["Close"] - prev_close) / prev_close

            if vol_ratio > 1.5 and close_chg > 0:
                sc, interp = 0.6, "High volume with price increase (conviction)"
            elif vol_ratio > 1.5 and close_chg < 0:
                sc, interp = -0.6, "High volume with price decrease (selling pressure)"
            elif vol_ratio > 1.0:
                sc, interp = 0.1 if close_chg >= 0 else -0.1, "Above-average volume"
            else:
                sc, interp = 0.0, "Below-average volume"
            signals.append(TechnicalSignal("Volume", round(vol_ratio, 2), sc, 0.7, interp, "Volume"))

        obv_series = df["OBV"].dropna()
        if len(obv_series) >= 20:
            obv_now = obv_series.iloc[-1]
            obv_20ago = obv_series.iloc[-20]
            close_now = df["Close"].iloc[-1]
            close_20ago = df["Close"].iloc[-20]

            obv_trend = obv_now > obv_20ago
            price_trend = close_now > close_20ago

            if obv_trend and price_trend:
                sc, interp = 0.4, "OBV confirms uptrend"
            elif obv_trend and not price_trend:
                sc, interp = 0.5, "OBV bullish divergence (accumulation)"
            elif not obv_trend and price_trend:
                sc, interp = -0.5, "OBV bearish divergence (distribution)"
            else:
                sc, interp = -0.3, "OBV confirms downtrend"
            signals.append(TechnicalSignal("OBV Trend", round(obv_now, 0), sc, 0.6, interp, "Volume"))

        return signals

    def _score_atr(self, df: pd.DataFrame, close: float) -> list[TechnicalSignal]:
        atr = df["ATR"].iloc[-1]
        if atr is None or pd.isna(atr) or close == 0:
            return []

        atr_pct = atr / close * 100
        if atr_pct > 5:
            sc, interp = -0.2, f"Very high volatility ({atr_pct:.1f}% ATR)"
        elif atr_pct > 3:
            sc, interp = -0.1, f"Elevated volatility ({atr_pct:.1f}% ATR)"
        elif atr_pct > 1.5:
            sc, interp = 0.1, f"Normal volatility ({atr_pct:.1f}% ATR)"
        else:
            sc, interp = 0.1, f"Low volatility ({atr_pct:.1f}% ATR)"

        return [TechnicalSignal("ATR", round(atr, 2), sc, 0.4, interp, "Volatility")]

    def _score_trend_strength(self, df: pd.DataFrame) -> list[TechnicalSignal]:
        adx = df["ADX"].iloc[-1]
        if adx is None or pd.isna(adx):
            return []

        if adx > 50:
            sc, interp = 0.3, "Very strong trend"
        elif adx > 25:
            sc, interp = 0.2, "Strong trend"
        elif adx > 20:
            sc, interp = 0.0, "Weak trend"
        else:
            sc, interp = -0.1, "No clear trend (ranging market)"

        return [TechnicalSignal("ADX", round(adx, 1), sc, 0.6, interp, "Trend")]

    def _compute_support_resistance(self, df: pd.DataFrame) -> list[SupportResistanceLevel]:
        """Identify support and resistance levels using multiple methods:
        1. Swing highs/lows (local extrema)
        2. Fibonacci retracement of the period range
        3. Volume-weighted price clusters (VWAP zones)
        4. Round-number psychological levels
        5. Moving-average levels as dynamic S/R
        """
        if len(df) < 30:
            return []

        close = df["Close"].values
        high = df["High"].values
        low = df["Low"].values
        volume = df["Volume"].values.astype(float)
        current = float(close[-1])

        raw_levels: list[tuple[float, str, str]] = []  # (price, method, kind_hint)

        # --- 1. Swing highs & lows (local extrema over a window) ---
        window = max(5, len(df) // 20)
        for i in range(window, len(df) - window):
            if high[i] == max(high[i - window:i + window + 1]):
                raw_levels.append((float(high[i]), "Swing High", "resistance"))
            if low[i] == min(low[i - window:i + window + 1]):
                raw_levels.append((float(low[i]), "Swing Low", "support"))

        # --- 2. Fibonacci retracement ---
        period_high = float(high.max())
        period_low = float(low.min())
        fib_range = period_high - period_low
        if fib_range > 0:
            for ratio, label in [(0.236, "Fib 23.6%"), (0.382, "Fib 38.2%"),
                                  (0.500, "Fib 50%"), (0.618, "Fib 61.8%"),
                                  (0.786, "Fib 78.6%")]:
                fib_price = period_high - fib_range * ratio
                kind = "support" if fib_price < current else "resistance"
                raw_levels.append((fib_price, label, kind))

        # --- 3. Volume-weighted price clusters ---
        try:
            n_bins = 40
            price_range = period_high - period_low
            if price_range > 0:
                bin_size = price_range / n_bins
                bins = np.zeros(n_bins)
                for i in range(len(close)):
                    idx = int((close[i] - period_low) / bin_size)
                    idx = min(idx, n_bins - 1)
                    bins[idx] += volume[i]

                vol_threshold = np.percentile(bins[bins > 0], 75) if bins.sum() > 0 else 0
                for i in range(n_bins):
                    if bins[i] >= vol_threshold:
                        level_price = period_low + (i + 0.5) * bin_size
                        kind = "support" if level_price < current else "resistance"
                        raw_levels.append((level_price, "Volume Cluster", kind))
        except Exception:
            pass

        # --- 4. Round-number psychological levels ---
        if current > 0:
            magnitude = 10 ** max(0, int(np.log10(current)) - 1)
            round_base = max(1, magnitude)
            lower_round = int(current / round_base) * round_base
            for mult in range(-2, 4):
                level = lower_round + mult * round_base
                if level > 0 and abs(level - current) / current < 0.15:
                    kind = "support" if level < current else "resistance"
                    raw_levels.append((float(level), "Psychological", kind))

        # --- 5. Moving averages as dynamic S/R ---
        last = df.iloc[-1]
        for col, label in [("SMA_20", "SMA 20"), ("SMA_50", "SMA 50"),
                            ("SMA_200", "SMA 200"), ("EMA_12", "EMA 12")]:
            val = last.get(col)
            if val is not None and not pd.isna(val):
                kind = "support" if float(val) < current else "resistance"
                raw_levels.append((float(val), label, kind))

        # --- Cluster nearby levels and count touches ---
        if not raw_levels:
            return []

        tolerance = current * 0.015  # 1.5% proximity to cluster
        raw_levels.sort(key=lambda x: x[0])

        clusters: list[dict] = []
        for price, method, kind in raw_levels:
            merged = False
            for cluster in clusters:
                if abs(price - cluster["price"]) <= tolerance:
                    cluster["methods"].add(method)
                    cluster["count"] += 1
                    cluster["price"] = (cluster["price"] * (cluster["count"] - 1) + price) / cluster["count"]
                    merged = True
                    break
            if not merged:
                clusters.append({"price": price, "kind": kind, "methods": {method}, "count": 1})

        # Score and convert
        levels: list[SupportResistanceLevel] = []
        for c in clusters:
            if abs(c["price"] - current) / current < 0.002:
                continue  # skip levels at current price

            kind = "support" if c["price"] < current else "resistance"
            n_methods = len(c["methods"])
            touches = c["count"]

            if n_methods >= 3 or touches >= 4:
                strength = "strong"
            elif n_methods >= 2 or touches >= 2:
                strength = "moderate"
            else:
                strength = "weak"

            levels.append(SupportResistanceLevel(
                price=round(c["price"], 2),
                kind=kind,
                strength=strength,
                method=" + ".join(sorted(c["methods"])),
                touches=touches,
            ))

        # Sort: supports descending (nearest first), resistances ascending
        supports = sorted([l for l in levels if l.kind == "support"],
                          key=lambda l: l.price, reverse=True)
        resistances = sorted([l for l in levels if l.kind == "resistance"],
                             key=lambda l: l.price)

        # Keep top levels by strength & proximity
        top_supports = sorted(supports, key=lambda l: ({"strong": 0, "moderate": 1, "weak": 2}[l.strength], abs(current - l.price)))[:5]
        top_resistances = sorted(resistances, key=lambda l: ({"strong": 0, "moderate": 1, "weak": 2}[l.strength], abs(current - l.price)))[:5]

        result = sorted(top_supports + top_resistances, key=lambda l: l.price)
        return result

    def get_chart_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return the dataframe with all computed indicators for charting."""
        df = df.copy()
        self._compute_indicators(df)
        return df
