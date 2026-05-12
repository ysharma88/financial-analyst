#!/usr/bin/env python3
"""Support & Resistance level analysis for buy candidates."""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_fetcher import StockDataFetcher
from technical_analysis import TechnicalAnalyzer
from risk_management import RiskManager

CANDIDATES = {
    "META": "META",
    "MSFT": "MSFT",
    "NFLX": "NFLX",
    "TEAM": "TEAM",
    "TSLA": "TSLA",
    "BABA": "BABA",
    "IBKR": "IBKR",
}

def fmt(v, prefix="$", decimals=2):
    if v is None: return "N/A"
    return f"{prefix}{v:.{decimals}f}"

def pct(v):
    if v is None: return "N/A"
    return f"{v*100:+.1f}%"

def run():
    print("\n" + "="*90)
    print("  SUPPORT & RESISTANCE — BUY LEVEL ANALYSIS")
    print("  Live data | " + __import__('time').strftime('%Y-%m-%d %H:%M'))
    print("="*90)

    for ticker, name in CANDIDATES.items():
        print(f"\n{'━'*90}")
        print(f"  {ticker} — {name}")
        print(f"{'━'*90}")

        try:
            fetcher = StockDataFetcher(ticker)
            hist    = fetcher.get_history(period="1y")
            fund    = fetcher.get_fundamental_data()
            price   = fund.get("price", 0) or 0

            if hist is None or len(hist) < 20:
                print("  [WARN] Insufficient price history.")
                continue

            tech   = TechnicalAnalyzer().analyze(hist)
            risk   = RiskManager().analyze(hist, entry_price=price)

            print(f"  Current Price : {fmt(price)}")
            print(f"  52-wk High    : {fmt(fund.get('fifty_two_week_high'))}"
                  f"   52-wk Low : {fmt(fund.get('fifty_two_week_low'))}")
            print(f"  50-day MA     : {fmt(fund.get('fifty_day_avg'))}"
                  f"   200-day MA: {fmt(fund.get('two_hundred_day_avg'))}")
            print(f"  Technical Score: {tech.overall_score:+.3f}  Signal: {tech.signal}")

            # ── Support levels ────────────────────────────────────────────
            sr = getattr(tech, 'support_resistance', []) or []
            supports    = sorted([l for l in sr if l.kind == 'support'    and l.price < price],
                                 key=lambda l: l.price, reverse=True)
            resistances = sorted([l for l in sr if l.kind == 'resistance' and l.price > price],
                                 key=lambda l: l.price)

            print(f"\n  ── SUPPORT LEVELS (buy zones, below current price) ──────────────────")
            if supports:
                for i, s in enumerate(supports[:5], 1):
                    drop = (s.price - price) / price * 100
                    print(f"  S{i}  ${s.price:>8.2f}  ({drop:+.1f}% from current)"
                          f"  Strength: {s.strength:<8}  Method: {s.method}")
            else:
                # Fallback: compute from MAs and ATR
                ma50  = fund.get('fifty_day_avg')
                ma200 = fund.get('two_hundred_day_avg')
                low52 = fund.get('fifty_two_week_low')
                if ma50:
                    d = (ma50-price)/price*100
                    print(f"  S1  ${ma50:>8.2f}  ({d:+.1f}% from current)  Strength: KEY      Method: 50-day MA")
                if ma200:
                    d = (ma200-price)/price*100
                    print(f"  S2  ${ma200:>8.2f}  ({d:+.1f}% from current)  Strength: MAJOR    Method: 200-day MA")
                if low52:
                    d = (low52-price)/price*100
                    print(f"  S3  ${low52:>8.2f}  ({d:+.1f}% from current)  Strength: EXTREME  Method: 52-wk Low")

            print(f"\n  ── RESISTANCE LEVELS (sell/take-profit zones, above current) ────────")
            if resistances:
                for i, r in enumerate(resistances[:5], 1):
                    rise = (r.price - price) / price * 100
                    print(f"  R{i}  ${r.price:>8.2f}  ({rise:+.1f}% from current)"
                          f"  Strength: {r.strength:<8}  Method: {r.method}")
            else:
                high52 = fund.get('fifty_two_week_high')
                target = fund.get('target_mean_price')
                if high52:
                    d = (high52-price)/price*100
                    print(f"  R1  ${high52:>8.2f}  ({d:+.1f}% from current)  Strength: MAJOR    Method: 52-wk High")
                if target:
                    d = (target-price)/price*100
                    print(f"  R2  ${target:>8.2f}  ({d:+.1f}% from current)  Strength: ANALYST  Method: Analyst Consensus Target")

            # ── Stop-loss levels ──────────────────────────────────────────
            print(f"\n  ── STOP-LOSS LEVELS ─────────────────────────────────────────────────")
            if risk.stop_losses:
                for sl in sorted(risk.stop_losses, key=lambda s: s.stop_price, reverse=True)[:3]:
                    print(f"  Stop  ${sl.stop_price:>8.2f}  ({sl.distance_pct:+.1f}% from current)"
                          f"  Method: {sl.method}")
            else:
                atr = risk.volatility_metrics.get('atr_14')
                if atr and price:
                    stop = price - 2*atr
                    print(f"  Stop  ${stop:>8.2f}  (2×ATR = {fmt(atr)} below current)  Method: ATR-based")

            # ── Risk metrics ─────────────────────────────────────────────
            vol = risk.volatility_metrics.get('annualized_volatility')
            atr = risk.volatility_metrics.get('atr_14')
            print(f"\n  ── RISK METRICS ─────────────────────────────────────────────────────")
            print(f"  Ann. Volatility: {pct(vol)}   ATR(14): {fmt(atr)}   Sharpe: {fmt(risk.sharpe_approx)}")

            # ── Analyst target context ────────────────────────────────────
            target = fund.get('target_mean_price')
            t_high = fund.get('target_high_price')
            t_low  = fund.get('target_low_price')
            rec    = fund.get('recommendation_key', 'N/A')
            if target and price:
                upside = (target - price)/price*100
                print(f"\n  ── ANALYST TARGETS ──────────────────────────────────────────────────")
                print(f"  Consensus: {fmt(target)} ({upside:+.1f}% upside)  |"
                      f"  High: {fmt(t_high)}  Low: {fmt(t_low)}  |  Rating: {rec}")

            # ── BUY ZONE RECOMMENDATION ───────────────────────────────────
            print(f"\n  ── BUY ZONE RECOMMENDATION ──────────────────────────────────────────")
            if supports:
                s1 = supports[0].price
                s2 = supports[1].price if len(supports) > 1 else None
                ideal  = round(s1 * 1.005, 2)   # just above nearest support
                aggressive = round(price * 0.98, 2)   # 2% below current = market order zone
                print(f"  IDEAL ENTRY    : ${ideal:.2f}  (just above S1 support ${s1:.2f})")
                if s2:
                    print(f"  PATIENT ENTRY  : ${s2:.2f}  (pullback to S2 support — better risk/reward)")
                print(f"  AGGRESSIVE ENTRY: ${aggressive:.2f}  (within 2% of current — for immediate exposure)")
            else:
                ma50 = fund.get('fifty_day_avg')
                if ma50 and ma50 < price:
                    print(f"  IDEAL ENTRY    : ${ma50:.2f}  (50-day MA — classic dip-buy level)")
                    print(f"  AGGRESSIVE ENTRY: ${price*0.98:.2f}  (within 2% of current)")
                else:
                    print(f"  AGGRESSIVE ENTRY: ${price:.2f}  (current price — enter now)")

            if risk.stop_losses:
                best_stop = min(risk.stop_losses, key=lambda s: s.stop_price)
                if price and best_stop.stop_price:
                    risk_pct = (price - best_stop.stop_price) / price * 100
                    print(f"  STOP-LOSS      : ${best_stop.stop_price:.2f}  ({risk_pct:.1f}% below entry)")
            if target and price:
                print(f"  TAKE PROFIT T1 : ${target:.2f}  (analyst consensus, {(target-price)/price*100:+.1f}%)")
                if t_high:
                    print(f"  TAKE PROFIT T2 : ${t_high:.2f}  (analyst high target, {(t_high-price)/price*100:+.1f}%)")

        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback; traceback.print_exc()

    print("\n" + "="*90)
    print("  END OF SUPPORT & RESISTANCE REPORT")
    print("="*90 + "\n")

if __name__ == "__main__":
    run()
