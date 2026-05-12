#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
sys.path.insert(0, _project_root)

from scripts.data_feed import get_data
from scripts.execution_algorithms import ExecutionAlgo, MarketContext, execute, select_algorithm
from scripts.financial_report import format_dashboard
from scripts.financial_strategy import run_strategy


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("symbol", nargs="?", default="AAPL")
    p.add_argument("--data-source", default="yahoo", choices=["yahoo", "alphavantage", "demo"])
    p.add_argument("--period", default="3mo")
    p.add_argument("--interval", default="1d")
    p.add_argument("--json", action="store_true")
    p.add_argument("--exec-algo", choices=["auto", "vwap", "twap", "pov", "is"], default="auto")
    p.add_argument("--liquidity", default="high", choices=["low", "medium", "high"])
    p.add_argument("--volatility", default="medium", choices=["low", "medium", "high"])
    p.add_argument("--urgency", default="low", choices=["low", "medium", "high"])
    args = p.parse_args()

    if args.data_source == "demo":
        import random
        prices = [100 + random.gauss(0, 2) for _ in range(80)]
        volumes = [1_000_000 + random.randint(-100_000, 100_000) for _ in range(80)]
    else:
        out = get_data(args.symbol, source=args.data_source, period=args.period, interval=args.interval, api_key=os.getenv("ALPHA_VANTAGE_API_KEY"))
        if out is None:
            raise SystemExit("No market data available.")
        prices, volumes = out

    result = run_strategy(args.symbol, prices, volumes, position=None, alert_on_signal=True)
    ctx = MarketContext(args.liquidity, args.volatility, args.urgency)
    algo = select_algorithm(ctx) if args.exec_algo == "auto" else ExecutionAlgo("implementation_shortfall" if args.exec_algo == "is" else args.exec_algo)
    result["execution_algo"] = algo.value
    result["execution_slices"] = [s.__dict__ for s in execute(algo, int((result.get("trade") or {}).get("quantity", 100)))]

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(format_dashboard(result))


if __name__ == "__main__":
    main()

