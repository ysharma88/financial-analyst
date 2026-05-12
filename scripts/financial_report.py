#!/usr/bin/env python3
from __future__ import annotations


def format_dashboard(result: dict) -> str:
    m = result.get("metrics", {})
    lines = [
        "=" * 44,
        f"FINANCIAL STRATEGY  {result.get('symbol', '')}",
        "=" * 44,
        f"Signal: {result.get('signal', '').upper()}",
        f"Reason: {result.get('reason', '')}",
        f"Price: {m.get('price')}",
        f"RSI: {m.get('rsi')} | Momentum: {m.get('momentum_pct')}%",
        "-" * 44,
    ]
    if result.get("trade"):
        t = result["trade"]
        lines.append(f"Trade: {t['side']} {t['quantity']} @ {t['price']:.2f}")
    lines.append("=" * 44)
    return "\n".join(lines)

