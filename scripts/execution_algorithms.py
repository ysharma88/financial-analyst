#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum


class ExecutionAlgo(Enum):
    VWAP = "vwap"
    TWAP = "twap"
    POV = "pov"
    IS = "implementation_shortfall"


@dataclass
class MarketContext:
    liquidity: str
    volatility: str
    urgency: str
    order_size_pct_adv: float = 0.0


@dataclass
class Slice:
    quantity: int
    timestamp: datetime
    algo: str
    reason: str


def select_algorithm(context: MarketContext) -> ExecutionAlgo:
    liq = context.liquidity.lower()
    vol = context.volatility.lower()
    urg = context.urgency.lower()
    if urg == "high":
        return ExecutionAlgo.IS
    if liq == "low":
        return ExecutionAlgo.TWAP
    if urg == "medium":
        return ExecutionAlgo.POV
    if liq == "high" and vol == "low":
        return ExecutionAlgo.VWAP
    return ExecutionAlgo.TWAP


def execute(algo: ExecutionAlgo, total_quantity: int, start: datetime | None = None) -> list[Slice]:
    start = start or datetime.now()
    if total_quantity <= 0:
        return []
    if algo == ExecutionAlgo.IS:
        q1 = int(total_quantity * 0.6)
        q2 = total_quantity - q1
        return [
            Slice(q1, start, "IS", "Front-load"),
            Slice(q2, start + timedelta(minutes=2), "IS", "Follow-through"),
        ]
    # TWAP/VWAP/POV simplified into 5 slices
    per = total_quantity // 5
    rem = total_quantity % 5
    out: list[Slice] = []
    for i in range(5):
        qty = per + (1 if i < rem else 0)
        if qty > 0:
            out.append(Slice(qty, start + timedelta(minutes=5 * i), algo.value.upper(), f"Slice {i+1}/5"))
    return out

