"""Risk management engine — position sizing, stop loss strategies, and risk/reward analysis."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, List

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class StopLossLevel:
    method: str
    stop_price: float
    distance_pct: float  # % below entry
    risk_per_share: float
    description: str


@dataclass
class PositionSize:
    method: str
    shares: int
    position_value: float
    risk_amount: float
    pct_of_portfolio: float
    description: str


@dataclass
class RiskRewardScenario:
    label: str
    target_price: float
    reward_pct: float
    risk_pct: float
    ratio: float  # reward / risk


@dataclass
class RiskProfile:
    entry_price: float
    stop_losses: List[StopLossLevel] = field(default_factory=list)
    position_sizes: List[PositionSize] = field(default_factory=list)
    risk_reward: List[RiskRewardScenario] = field(default_factory=list)
    trailing_stops: dict = field(default_factory=dict)
    volatility_metrics: dict = field(default_factory=dict)
    max_drawdown: Optional[float] = None
    sharpe_approx: Optional[float] = None
    sortino: Optional[float] = None
    calmar: Optional[float] = None
    var_95: Optional[float] = None
    cvar_95: Optional[float] = None
    alpha_vs_spy: Optional[float] = None
    beta_vs_spy: Optional[float] = None


# ---------------------------------------------------------------------------
# Risk management calculator
# ---------------------------------------------------------------------------
class RiskManager:

    def analyze(
        self,
        df: pd.DataFrame,
        entry_price: float,
        account_size: float = 100_000,
        risk_pct: float = 1.0,
        target_prices: Optional[List[float]] = None,
    ) -> RiskProfile:
        profile = RiskProfile(entry_price=entry_price)

        atr_14 = self._compute_atr(df, period=14)
        atr_21 = self._compute_atr(df, period=21)
        daily_returns = df["Close"].pct_change().dropna()

        profile.volatility_metrics = {
            "atr_14": atr_14,
            "atr_21": atr_21,
            "daily_volatility": float(daily_returns.std()) if len(daily_returns) > 0 else 0,
            "annualized_volatility": float(daily_returns.std() * math.sqrt(252)) if len(daily_returns) > 0 else 0,
            "avg_daily_range": float((df["High"] - df["Low"]).tail(20).mean()) if len(df) >= 20 else 0,
        }

        profile.stop_losses = self._compute_stop_losses(df, entry_price, atr_14)
        profile.trailing_stops = self._compute_trailing_stops(entry_price, atr_14, df)

        risk_amount = account_size * (risk_pct / 100)
        profile.position_sizes = self._compute_position_sizes(
            entry_price, account_size, risk_amount, atr_14, daily_returns, profile.stop_losses,
        )

        profile.risk_reward = self._compute_risk_reward(
            entry_price, profile.stop_losses, target_prices, df,
        )

        profile.max_drawdown = self._compute_max_drawdown(df)
        profile.sharpe_approx = self._compute_sharpe(daily_returns)
        profile.sortino = self._compute_sortino(daily_returns)
        profile.calmar = self._compute_calmar(daily_returns, profile.max_drawdown)
        profile.var_95 = self._compute_var(daily_returns)
        profile.cvar_95 = self._compute_cvar(daily_returns)

        return profile

    # ----- ATR -----
    def _compute_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        if len(df) < period + 1:
            return 0.0
        high = df["High"]
        low = df["Low"]
        close = df["Close"].shift(1)
        tr = pd.concat([
            high - low,
            (high - close).abs(),
            (low - close).abs(),
        ], axis=1).max(axis=1)
        return float(tr.rolling(period).mean().iloc[-1])

    # ----- Stop Losses -----
    def _compute_stop_losses(
        self, df: pd.DataFrame, entry: float, atr: float,
    ) -> List[StopLossLevel]:
        stops = []

        # 1. ATR-based (2x ATR)
        if atr > 0:
            atr2_stop = entry - 2 * atr
            stops.append(StopLossLevel(
                method="ATR (2x)",
                stop_price=round(atr2_stop, 2),
                distance_pct=round((entry - atr2_stop) / entry * 100, 2),
                risk_per_share=round(entry - atr2_stop, 2),
                description=f"2x ATR ({atr:.2f}) below entry — adapts to current volatility",
            ))

            # 1b. ATR-based (3x ATR) — wider
            atr3_stop = entry - 3 * atr
            stops.append(StopLossLevel(
                method="ATR (3x)",
                stop_price=round(atr3_stop, 2),
                distance_pct=round((entry - atr3_stop) / entry * 100, 2),
                risk_per_share=round(entry - atr3_stop, 2),
                description=f"3x ATR — wider stop for volatile stocks, reduces whipsaw risk",
            ))

        # 2. Percentage-based stops
        for pct in [2, 5, 8]:
            pct_stop = entry * (1 - pct / 100)
            stops.append(StopLossLevel(
                method=f"Fixed {pct}%",
                stop_price=round(pct_stop, 2),
                distance_pct=pct,
                risk_per_share=round(entry - pct_stop, 2),
                description=f"Simple {pct}% stop — straightforward, ignores volatility",
            ))

        # 3. Support-level based (recent swing low)
        if len(df) >= 20:
            recent_low_20 = float(df["Low"].tail(20).min())
            dist = round((entry - recent_low_20) / entry * 100, 2)
            stops.append(StopLossLevel(
                method="20-Day Low",
                stop_price=round(recent_low_20, 2),
                distance_pct=dist,
                risk_per_share=round(entry - recent_low_20, 2),
                description=f"Below 20-day swing low — respects recent price structure",
            ))

        if len(df) >= 50:
            recent_low_50 = float(df["Low"].tail(50).min())
            dist = round((entry - recent_low_50) / entry * 100, 2)
            stops.append(StopLossLevel(
                method="50-Day Low",
                stop_price=round(recent_low_50, 2),
                distance_pct=dist,
                risk_per_share=round(entry - recent_low_50, 2),
                description=f"Below 50-day swing low — wider support for swing trades",
            ))

        # 4. Moving average stop
        if len(df) >= 50:
            sma50 = float(df["Close"].rolling(50).mean().iloc[-1])
            if sma50 < entry:
                dist = round((entry - sma50) / entry * 100, 2)
                stops.append(StopLossLevel(
                    method="Below SMA 50",
                    stop_price=round(sma50, 2),
                    distance_pct=dist,
                    risk_per_share=round(entry - sma50, 2),
                    description="Close below 50-day SMA signals trend reversal",
                ))

        return sorted(stops, key=lambda s: s.stop_price, reverse=True)

    # ----- Trailing Stops -----
    def _compute_trailing_stops(self, entry: float, atr: float, df: pd.DataFrame) -> dict:
        result = {}
        if atr > 0:
            result["atr_trailing"] = {
                "method": "ATR Trailing (2x)",
                "initial_stop": round(entry - 2 * atr, 2),
                "trail_amount": round(2 * atr, 2),
                "description": "Trail stop by 2x ATR below the highest close since entry. "
                               "Locks in profits while giving room for normal volatility.",
            }
            result["atr_trailing_tight"] = {
                "method": "ATR Trailing (1.5x)",
                "initial_stop": round(entry - 1.5 * atr, 2),
                "trail_amount": round(1.5 * atr, 2),
                "description": "Tighter trailing stop (1.5x ATR) — captures more profit "
                               "but higher chance of being stopped out on pullbacks.",
            }

        for pct in [5, 10, 15]:
            result[f"pct_trailing_{pct}"] = {
                "method": f"{pct}% Trailing",
                "initial_stop": round(entry * (1 - pct / 100), 2),
                "trail_amount_pct": pct,
                "description": f"Trail stop {pct}% below the highest price since entry.",
            }

        if len(df) >= 20:
            chandelier_stop = float(df["High"].tail(22).max()) - 3 * atr
            result["chandelier"] = {
                "method": "Chandelier Exit (3x ATR)",
                "stop_price": round(chandelier_stop, 2),
                "description": "Highest high of last 22 days minus 3x ATR — "
                               "classic trend-following trailing stop.",
            }

        return result

    # ----- Position Sizing -----
    def _compute_position_sizes(
        self,
        entry: float,
        account_size: float,
        risk_amount: float,
        atr: float,
        daily_returns: pd.Series,
        stop_losses: List[StopLossLevel],
    ) -> List[PositionSize]:
        sizes = []

        # 1. Fixed risk per trade (using ATR 2x stop)
        atr_stop = next((s for s in stop_losses if s.method == "ATR (2x)"), None)
        if atr_stop and atr_stop.risk_per_share > 0:
            shares = int(risk_amount / atr_stop.risk_per_share)
            if shares > 0:
                sizes.append(PositionSize(
                    method="Fixed Risk (ATR Stop)",
                    shares=shares,
                    position_value=round(shares * entry, 2),
                    risk_amount=round(shares * atr_stop.risk_per_share, 2),
                    pct_of_portfolio=round(shares * entry / account_size * 100, 2),
                    description=f"Risk ${risk_amount:,.0f} per trade with 2x ATR stop — "
                                f"sizes position so max loss equals your risk tolerance.",
                ))

        # 2. Fixed risk with percentage stop
        pct5_stop = next((s for s in stop_losses if s.method == "Fixed 5%"), None)
        if pct5_stop and pct5_stop.risk_per_share > 0:
            shares = int(risk_amount / pct5_stop.risk_per_share)
            if shares > 0:
                sizes.append(PositionSize(
                    method="Fixed Risk (5% Stop)",
                    shares=shares,
                    position_value=round(shares * entry, 2),
                    risk_amount=round(shares * pct5_stop.risk_per_share, 2),
                    pct_of_portfolio=round(shares * entry / account_size * 100, 2),
                    description=f"Risk ${risk_amount:,.0f} per trade with 5% stop — "
                                f"simpler approach suitable for less volatile stocks.",
                ))

        # 3. Volatility-based (ATR sizing)
        if atr > 0:
            risk_per_atr_unit = risk_amount / (2 * atr)
            shares = int(risk_per_atr_unit)
            if shares > 0:
                sizes.append(PositionSize(
                    method="Volatility-Based (ATR)",
                    shares=shares,
                    position_value=round(shares * entry, 2),
                    risk_amount=round(shares * 2 * atr, 2),
                    pct_of_portfolio=round(shares * entry / account_size * 100, 2),
                    description="Sizes position inversely to volatility — "
                                "smaller positions in volatile stocks, larger in calm ones.",
                ))

        # 4. Equal-weight portfolio position
        max_positions = [10, 15, 20]
        for n in max_positions:
            alloc = account_size / n
            shares = int(alloc / entry) if entry > 0 else 0
            if shares > 0:
                sizes.append(PositionSize(
                    method=f"Equal Weight (1/{n})",
                    shares=shares,
                    position_value=round(shares * entry, 2),
                    risk_amount=round(shares * entry * 0.05, 2),
                    pct_of_portfolio=round(shares * entry / account_size * 100, 2),
                    description=f"Equal allocation assuming {n}-stock portfolio "
                                f"(~{100/n:.1f}% each).",
                ))

        # 5. Kelly Criterion (simplified)
        if len(daily_returns) >= 60:
            wins = daily_returns[daily_returns > 0]
            losses = daily_returns[daily_returns < 0]
            if len(wins) > 0 and len(losses) > 0:
                win_rate = len(wins) / len(daily_returns)
                avg_win = float(wins.mean())
                avg_loss = float(abs(losses.mean()))
                if avg_loss > 0:
                    kelly_pct = win_rate - ((1 - win_rate) / (avg_win / avg_loss))
                    kelly_pct = max(0, min(kelly_pct, 0.25))  # cap at 25%
                    half_kelly = kelly_pct / 2  # half-Kelly is safer
                    kelly_value = account_size * half_kelly
                    shares = int(kelly_value / entry) if entry > 0 else 0
                    if shares > 0:
                        sizes.append(PositionSize(
                            method="Half-Kelly Criterion",
                            shares=shares,
                            position_value=round(shares * entry, 2),
                            risk_amount=round(shares * entry * avg_loss * 5, 2),
                            pct_of_portfolio=round(half_kelly * 100, 2),
                            description=f"Kelly fraction: {kelly_pct:.1%}, using half-Kelly ({half_kelly:.1%}) "
                                        f"for safety. Based on {len(daily_returns)}-day win rate of {win_rate:.0%}.",
                        ))

        return sizes

    # ----- Risk/Reward -----
    def _compute_risk_reward(
        self,
        entry: float,
        stop_losses: List[StopLossLevel],
        target_prices: Optional[List[float]],
        df: pd.DataFrame,
    ) -> List[RiskRewardScenario]:
        scenarios = []
        primary_stop = next((s for s in stop_losses if s.method == "ATR (2x)"), None)
        if primary_stop is None and stop_losses:
            primary_stop = stop_losses[0]
        if primary_stop is None:
            return scenarios

        risk_pct = primary_stop.distance_pct

        auto_targets = []
        if target_prices:
            auto_targets.extend([(f"Target ${t:.0f}", t) for t in target_prices])
        else:
            for mult in [1.5, 2.0, 3.0]:
                target = entry + (entry - primary_stop.stop_price) * mult
                auto_targets.append((f"{mult:.1f}R Target", target))

        if len(df) >= 50:
            high_52w = float(df["High"].tail(252).max()) if len(df) >= 252 else float(df["High"].max())
            if high_52w > entry * 1.01:
                auto_targets.append(("52-Week High", high_52w))

        for label, target in auto_targets:
            if target <= entry:
                continue
            reward_pct = (target - entry) / entry * 100
            ratio = reward_pct / risk_pct if risk_pct > 0 else 0
            scenarios.append(RiskRewardScenario(
                label=label,
                target_price=round(target, 2),
                reward_pct=round(reward_pct, 2),
                risk_pct=round(risk_pct, 2),
                ratio=round(ratio, 2),
            ))

        return sorted(scenarios, key=lambda s: s.ratio, reverse=True)

    # ----- Drawdown -----
    def _compute_max_drawdown(self, df: pd.DataFrame) -> float:
        if len(df) < 2:
            return 0.0
        cummax = df["Close"].cummax()
        drawdown = (df["Close"] - cummax) / cummax
        return float(drawdown.min()) * 100

    # ----- Sharpe -----
    def _compute_sharpe(self, daily_returns: pd.Series, risk_free_annual: float = 0.05) -> float:
        if len(daily_returns) < 30:
            return 0.0
        mean_daily = float(daily_returns.mean())
        std_daily = float(daily_returns.std())
        if std_daily == 0:
            return 0.0
        rf_daily = risk_free_annual / 252
        return round((mean_daily - rf_daily) / std_daily * math.sqrt(252), 2)

    def _compute_sortino(self, daily_returns: pd.Series, risk_free_annual: float = 0.05) -> float:
        if len(daily_returns) < 30:
            return 0.0
        rf_daily = risk_free_annual / 252
        excess = daily_returns - rf_daily
        downside = excess[excess < 0]
        if len(downside) == 0:
            return 0.0
        downside_std = float(downside.std())
        if downside_std == 0:
            return 0.0
        mean_excess = float(excess.mean())
        return round(mean_excess / downside_std * math.sqrt(252), 2)

    def _compute_calmar(self, daily_returns: pd.Series, max_drawdown: Optional[float]) -> Optional[float]:
        if not max_drawdown or max_drawdown >= 0 or len(daily_returns) < 30:
            return None
        annual_return = float(daily_returns.mean()) * 252
        return round(annual_return / abs(max_drawdown / 100), 2)

    def _compute_var(self, daily_returns: pd.Series, confidence: float = 0.95) -> Optional[float]:
        if len(daily_returns) < 30:
            return None
        return round(float(np.percentile(daily_returns, (1 - confidence) * 100)), 4)

    def _compute_cvar(self, daily_returns: pd.Series, confidence: float = 0.95) -> Optional[float]:
        if len(daily_returns) < 30:
            return None
        var = np.percentile(daily_returns, (1 - confidence) * 100)
        tail = daily_returns[daily_returns <= var]
        if len(tail) == 0:
            return None
        return round(float(tail.mean()), 4)
