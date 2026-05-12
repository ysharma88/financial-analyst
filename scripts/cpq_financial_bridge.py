#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DealContext:
    pipeline_value: float
    conversion_rate_14d: float
    deal_velocity: str
    margin_pct: float
    discount_pct: float
    days_in_pipeline: int


def should_discount(ctx: DealContext, max_discount_pct: float = 15.0) -> tuple[bool, str]:
    if ctx.discount_pct >= max_discount_pct:
        return False, "Already at max discount"
    if ctx.deal_velocity.lower() == "slow" and ctx.days_in_pipeline > 14:
        return True, "Slow deal, increase discount"
    if ctx.margin_pct < 0.2 and ctx.pipeline_value > 50_000:
        return True, "Strategic high-value discount"
    return False, "No discount change"

