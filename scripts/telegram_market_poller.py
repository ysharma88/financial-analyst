#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
sys.path.insert(0, _project_root)

from scripts.alert_metrics import (
    DEFAULT_ALERT_THRESHOLDS,
    FundamentalSnapshot,
    InstitutionalSnapshot,
    MacroSnapshot,
    TechnicalSnapshot,
    evaluate_all_alerts,
)
from scripts.data_feed import get_data
from scripts.financial_strategy import run_strategy

try:
    import yfinance as yf
except Exception:
    yf = None


@dataclass
class Snapshot:
    signal: str
    price: float
    momentum_pct: float | None
    atr: float | None


def send_telegram(msg: str, token: str, chat_id: str) -> bool:
    data = urllib.parse.urlencode({"chat_id": chat_id, "text": msg}).encode()
    req = urllib.request.Request(f"https://api.telegram.org/bot{token}/sendMessage", data=data, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status == 200
    except Exception:
        return False


def load_symbols_from_tracker(path: str) -> list[str]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        out = []
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                s = row.get("symbol") or row.get("ticker") or row.get("Symbol")
                if s:
                    out.append(s.strip().upper())
        return out
    if ext == ".json":
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            return [str(x.get("symbol", x)).upper() if isinstance(x, dict) else str(x).upper() for x in obj]
        if isinstance(obj, dict):
            vals = obj.get("symbols") or obj.get("watchlist") or obj.get("portfolio") or []
            return [str(x.get("symbol", x)).upper() if isinstance(x, dict) else str(x).upper() for x in vals]
    with open(path, encoding="utf-8") as f:
        return [x.strip().upper() for x in f.read().replace("\n", ",").split(",") if x.strip()]


def default_ui_tracker_path() -> str:
    return os.path.join(_project_root, "config", "ui_tracked_stocks.json")


def default_alert_config_path() -> str:
    return os.path.join(_project_root, "config", "alert_settings.json")


def default_telegram_config_path() -> str:
    return os.path.join(_project_root, "config", "telegram_credentials.json")


def load_alert_settings(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def merge_thresholds(runtime_cfg: dict) -> dict:
    merged = {
        "fundamental": dict(DEFAULT_ALERT_THRESHOLDS["fundamental"]),
        "technical": dict(DEFAULT_ALERT_THRESHOLDS["technical"]),
        "macro": dict(DEFAULT_ALERT_THRESHOLDS["macro"]),
        "institutional": dict(DEFAULT_ALERT_THRESHOLDS["institutional"]),
    }
    for group in ("fundamental", "technical", "macro", "institutional"):
        vals = runtime_cfg.get(group, {})
        if isinstance(vals, dict):
            merged[group].update(vals)
    return merged


def fetch_symbol_fundamental(symbol: str) -> FundamentalSnapshot:
    if yf is None:
        return FundamentalSnapshot()
    try:
        info = yf.Ticker(symbol).info or {}
        debt_to_equity = info.get("debtToEquity")
        if debt_to_equity is not None:
            debt_to_equity = float(debt_to_equity) / 100.0
        roe = info.get("returnOnEquity")
        payout = info.get("payoutRatio")
        peg = info.get("pegRatio")
        ps = info.get("priceToSalesTrailing12Months")
        roic = info.get("returnOnAssets")
        insider = bool(info.get("heldPercentInsiders", 0) and info.get("heldPercentInsiders", 0) < 0.01)
        return FundamentalSnapshot(
            debt_to_equity=float(debt_to_equity) if debt_to_equity is not None else None,
            roe=float(roe) if roe is not None else None,
            dividend_payout_ratio=float(payout) if payout is not None else None,
            peg_ratio=float(peg) if peg is not None else None,
            price_to_sales=float(ps) if ps is not None else None,
            roic=float(roic) if roic is not None else None,
            insider_cluster_selling=insider,
        )
    except Exception:
        return FundamentalSnapshot()


def fetch_macro_snapshot(runtime_cfg: dict) -> MacroSnapshot:
    vix = None
    us10 = None
    us2 = None
    if yf is not None:
        try:
            vix_hist = yf.Ticker("^VIX").history(period="5d", interval="1d")
            if not vix_hist.empty:
                vix = float(vix_hist["Close"].iloc[-1])
        except Exception:
            pass
        try:
            tnx_hist = yf.Ticker("^TNX").history(period="5d", interval="1d")
            if not tnx_hist.empty:
                us10 = float(tnx_hist["Close"].iloc[-1]) / 10.0
        except Exception:
            pass
        try:
            twoy_hist = yf.Ticker("^UST2Y").history(period="5d", interval="1d")
            if not twoy_hist.empty:
                us2 = float(twoy_hist["Close"].iloc[-1]) / 10.0
        except Exception:
            pass
    macro_cfg = runtime_cfg.get("macro_inputs", {}) if isinstance(runtime_cfg.get("macro_inputs"), dict) else {}
    return MacroSnapshot(
        vix=vix,
        buffett_indicator_pct=macro_cfg.get("buffett_indicator_pct"),
        us10y=us10 if us10 is not None else macro_cfg.get("us10y"),
        us2y=us2 if us2 is not None else macro_cfg.get("us2y"),
        inflation_surprise_pct=macro_cfg.get("inflation_surprise_pct"),
    )


def load_telegram_credentials(path: str) -> tuple[str, str]:
    if not os.path.exists(path):
        return "", ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            return str(payload.get("bot_token", "")).strip(), str(payload.get("chat_id", "")).strip()
    except Exception:
        pass
    return "", ""


def detect(prev: Snapshot | None, curr: Snapshot, px_jump: float, atr_jump: float, mom_jump: float) -> str | None:
    if prev is None:
        return None
    if prev.signal != curr.signal and curr.signal != "hold":
        return f"Signal flip {prev.signal}->{curr.signal}"
    if prev.price:
        move = abs((curr.price - prev.price) / prev.price) * 100
        if move >= px_jump:
            return f"Price jump {move:.2f}%"
    if prev.atr and curr.atr and prev.atr > 0:
        a = abs((curr.atr - prev.atr) / prev.atr) * 100
        if a >= atr_jump:
            return f"ATR spike {a:.1f}%"
    if prev.momentum_pct is not None and curr.momentum_pct is not None:
        m = abs(curr.momentum_pct - prev.momentum_pct)
        if m >= mom_jump:
            return f"Momentum shift {m:.2f} pts"
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", default="AAPL,MSFT,NVDA")
    p.add_argument("--tracker-file", default="")
    p.add_argument("--data-source", default="yahoo", choices=["yahoo", "alphavantage", "demo"])
    p.add_argument("--period", default="5d")
    p.add_argument("--interval", default="1m")
    p.add_argument("--poll-seconds", type=int, default=300)
    p.add_argument("--price-jump-threshold-pct", type=float, default=1.2)
    p.add_argument("--atr-spike-threshold-pct", type=float, default=20.0)
    p.add_argument("--momentum-spike-abs-pct", type=float, default=2.5)
    p.add_argument("--alert-config-file", default="", help="Path to alert settings JSON file")
    p.add_argument("--telegram-config-file", default="", help="Path to telegram credentials JSON file")
    p.add_argument("--enable-advanced-metrics", action="store_true", help="Enable fundamental/macro/institutional threshold alerts")
    p.add_argument("--max-cycles", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if args.tracker_file:
        symbols = load_symbols_from_tracker(args.tracker_file)
    else:
        ui_tracker = default_ui_tracker_path()
        if os.path.exists(ui_tracker):
            symbols = load_symbols_from_tracker(ui_tracker)
            print(f"Using UI tracker list from: {ui_tracker}")
        else:
            symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    cfg_bot, cfg_chat = load_telegram_credentials(args.telegram_config_file or default_telegram_config_path())
    token = os.getenv("TELEGRAM_BOT_TOKEN", "") or cfg_bot
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "") or cfg_chat
    if not args.dry_run and (not token or not chat_id):
        raise SystemExit("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.")

    prev: dict[str, Snapshot] = {}
    last_advanced_alerts: dict[str, set[str]] = {}
    cycles = 0
    print(f"[{datetime.now().isoformat()}] Polling started for: {', '.join(symbols)}")
    while True:
        alert_cfg_path = args.alert_config_file or default_alert_config_path()
        runtime_cfg = load_alert_settings(alert_cfg_path)
        thresholds = merge_thresholds(runtime_cfg)
        poll_seconds = int(runtime_cfg.get("poll_seconds", args.poll_seconds))
        price_jump_threshold_pct = float(runtime_cfg.get("price_jump_threshold_pct", args.price_jump_threshold_pct))
        atr_spike_threshold_pct = float(runtime_cfg.get("atr_spike_threshold_pct", args.atr_spike_threshold_pct))
        momentum_spike_abs_pct = float(runtime_cfg.get("momentum_spike_abs_pct", args.momentum_spike_abs_pct))
        macro_snapshot = fetch_macro_snapshot(runtime_cfg) if args.enable_advanced_metrics else MacroSnapshot()

        for sym in symbols:
            out = get_data(sym, source=args.data_source, period=args.period, interval=args.interval, api_key=os.getenv("ALPHA_VANTAGE_API_KEY"))
            if out is None:
                continue
            prices, volumes = out
            res = run_strategy(sym, prices, volumes, position=None, alert_on_signal=False)
            m = res.get("metrics", {})
            cur = Snapshot(res.get("signal", "hold"), float(m.get("price") or 0), m.get("momentum_pct"), m.get("atr"))
            reason = detect(prev.get(sym), cur, price_jump_threshold_pct, atr_spike_threshold_pct, momentum_spike_abs_pct)
            if reason:
                txt = f"{sym}: {reason} | signal={res.get('signal')} | price={m.get('price')}"
                if args.dry_run:
                    print(txt)
                else:
                    send_telegram(txt, token, chat_id)

            if args.enable_advanced_metrics:
                fundamental = fetch_symbol_fundamental(sym)
                technical = TechnicalSnapshot(prices=prices, rsi=m.get("rsi"), atr=m.get("atr"))
                institutional_inputs = runtime_cfg.get("institutional_inputs", {}) if isinstance(runtime_cfg.get("institutional_inputs"), dict) else {}
                institutional = InstitutionalSnapshot(
                    ownership_change_pct=institutional_inputs.get("ownership_change_pct"),
                    sentiment_score=institutional_inputs.get("sentiment_score"),
                    earnings_revision_percentile=institutional_inputs.get("earnings_revision_percentile"),
                )
                adv_alerts = evaluate_all_alerts(
                    fundamental=fundamental,
                    technical=technical,
                    macro=macro_snapshot,
                    institutional=institutional,
                    thresholds=thresholds,
                )
                current_alert_set = set(adv_alerts)
                new_alerts = [a for a in adv_alerts if a not in last_advanced_alerts.get(sym, set())]
                for a in new_alerts:
                    adv_msg = f"{sym}: {a} | price={m.get('price')}"
                    if args.dry_run:
                        print(adv_msg)
                    else:
                        send_telegram(adv_msg, token, chat_id)
                last_advanced_alerts[sym] = current_alert_set
            prev[sym] = cur
        cycles += 1
        if args.max_cycles and cycles >= args.max_cycles:
            break
        time.sleep(max(10, poll_seconds))


if __name__ == "__main__":
    main()

