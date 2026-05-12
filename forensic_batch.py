#!/usr/bin/env python3
"""Batch forensic analysis for a list of tickers using all project modules."""

from __future__ import annotations
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_fetcher import StockDataFetcher
from fundamental_analysis import FundamentalAnalyzer
from technical_analysis import TechnicalAnalyzer
from macro_analysis import MacroAnalyzer
from sector_analysis import SectorAnalyzer
from news_sentiment import NewsSentimentAnalyzer
from governance_redflags import GovernanceAnalyzer
from risk_management import RiskManager
from reasoning_engine import ReasoningEngine
from quality_scores import QualityScorer
from dcf_model import DCFValuator
from options_analysis import OptionsAnalyzer
from peer_comps import PeerComparator

TICKERS = ["META", "MSFT", "NFLX", "TEAM", "TSLA", "BABA", "IBKR"]

HOLDINGS = ["META", "MSFT", "IBKR", "NFLX", "TEAM", "TSLA", "BABA"]

def fmt_pct(v):
    if v is None:
        return "N/A"
    return f"{v*100:.1f}%"

def fmt_num(v, decimals=2):
    if v is None:
        return "N/A"
    return f"{v:.{decimals}f}"

def run_analysis(ticker: str, macro_result, sector_result):
    print(f"\n{'='*70}")
    print(f"  Analyzing {ticker}...")
    print(f"{'='*70}")

    try:
        fetcher = StockDataFetcher(ticker)
        if not fetcher.validate():
            print(f"  [WARN] No data for {ticker}, skipping.")
            return None

        fund_data = fetcher.get_fundamental_data()
        history = fetcher.get_history(period="1y")
        company_info = {
            "name": fetcher.get_company_name(),
            "ticker": ticker,
            "sector": fetcher.get_sector(),
            "industry": fetcher.get_industry(),
            "price": fund_data.get("price", 0),
        }

        # Run all analyzers
        fund_result = FundamentalAnalyzer().analyze(fund_data, sector=fetcher.get_sector())
        tech_result = TechnicalAnalyzer().analyze(history)
        news_result = NewsSentimentAnalyzer().analyze(ticker)

        governance = GovernanceAnalyzer().analyze(
            info=fetcher.info,
            company_officers=fetcher.get_company_officers(),
            earnings_history=fetcher.get_earnings_history(),
            governance_scores=fetcher.get_governance_scores(),
            insider_purchases=fetcher.get_insider_purchases(),
        )

        entry_price = fund_data.get("price", 0) or 0
        risk_profile = RiskManager().analyze(history, entry_price=entry_price)

        # Quality scores (Piotroski, Altman, Beneish)
        try:
            quality = QualityScorer().compute(
                info=fetcher.info,
                income_stmt=fetcher.get_income_statement(),
                balance_sheet=fetcher.get_balance_sheet(),
                cashflow=fetcher.get_cashflow_statement(),
            )
        except Exception:
            quality = None

        # DCF intrinsic value
        try:
            dcf = DCFValuator().compute(
                info=fetcher.info,
                income_stmt=fetcher.get_income_statement(),
                balance_sheet=fetcher.get_balance_sheet(),
                cashflow=fetcher.get_cashflow_statement(),
            )
        except Exception:
            dcf = None

        # Options market signals
        try:
            options = OptionsAnalyzer().analyze(ticker=ticker, history=history)
        except Exception:
            options = None

        # Peer comparables
        try:
            peers = PeerComparator().compare(
                subject_ticker=ticker,
                subject_info=fetcher.info,
                sector=fetcher.get_sector(),
            )
        except Exception:
            peers = None

        from recommendation_engine import HolisticRecommendation
        recommendation = HolisticRecommendation(fund_result, tech_result)

        # Score this stock's sector in the sector result
        stock_sector_result = None
        if sector_result is not None:
            import copy
            stock_sector_result = copy.deepcopy(sector_result)
            stock_sector_result = SectorAnalyzer().score_stock_sector(stock_sector_result, fetcher.get_sector())

        verdict = ReasoningEngine().synthesize(
            company_info=company_info,
            fund_data=fund_data,
            fund_result=fund_result,
            tech_result=tech_result,
            recommendation=recommendation,
            governance=governance,
            macro_result=macro_result,
            sector_result=stock_sector_result,
            news_result=news_result,
            risk_profile=risk_profile,
        )

        return {
            "ticker": ticker,
            "name": company_info["name"],
            "sector": company_info["sector"],
            "price": fund_data.get("price", 0),
            "market_cap": fund_data.get("market_cap"),
            "verdict": verdict,
            "fund_data": fund_data,
            "fund_result": fund_result,
            "tech_result": tech_result,
            "governance": governance,
            "risk_profile": risk_profile,
            "news_result": news_result,
            "quality": quality,
            "dcf": dcf,
            "options": options,
            "peers": peers,
        }
    except Exception as e:
        print(f"  [ERROR] {ticker}: {e}")
        import traceback
        traceback.print_exc()
        return None


def print_report(results):
    print("\n")
    print("=" * 90)
    print("  FORENSIC PORTFOLIO ANALYSIS REPORT")
    print(f"  Generated: {time.strftime('%Y-%m-%d %H:%M')}")
    print("=" * 90)

    # Sort by composite score descending
    valid = [r for r in results if r is not None]
    valid.sort(key=lambda r: r["verdict"].composite_score, reverse=True)

    # --- SUMMARY SCOREBOARD ---
    print("\n┌─ SUMMARY SCOREBOARD ────────────────────────────────────────────────────────────┐")
    print(f"  {'Ticker':<7} {'Name':<22} {'Sector':<20} {'Price':>8} {'Score':>7} {'Action':<14} {'Conf':<8} {'Hold?'}")
    print(f"  {'-'*6} {'-'*21} {'-'*19} {'-'*8} {'-'*7} {'-'*13} {'-'*7} {'-'*5}")
    for r in valid:
        v = r["verdict"]
        held = "  *** HELD ***" if r["ticker"] in HOLDINGS else ""
        price_str = f"${r['price']:.2f}" if r['price'] else "N/A"
        print(f"  {r['ticker']:<7} {r['name'][:21]:<22} {r['sector'][:19]:<20} {price_str:>8} {v.composite_score:>+7.3f} {v.action:<14} {v.confidence:<8}{held}")
    print("└─────────────────────────────────────────────────────────────────────────────────┘")

    # --- DETAILED REPORT PER STOCK ---
    for r in valid:
        v = r["verdict"]
        fd = r["fund_data"]
        fr = r["fund_result"]
        tr = r["tech_result"]
        gov = r["governance"]
        rp = r["risk_profile"]
        news = r["news_result"]
        held_flag = " [CURRENTLY HELD]" if r["ticker"] in HOLDINGS else ""

        print(f"\n\n{'━'*90}")
        print(f"  {r['ticker']} — {r['name']}{held_flag}")
        print(f"  {r['sector']} | Price: ${r['price']:.2f}" if r['price'] else f"  {r['sector']}")
        print(f"{'━'*90}")

        # Verdict banner
        print(f"\n  VERDICT: {v.action}  |  Score: {v.composite_score:+.3f}  |  Confidence: {v.confidence}")
        print(f"\n  THESIS:")
        print(f"  {v.thesis}")

        # Reasoning chain
        print(f"\n  REASONING CHAIN:")
        print(f"  {'Pillar':<14} {'Signal':<10} {'Score':>7}  Detail")
        print(f"  {'-'*13} {'-'*9} {'-'*7}  {'-'*40}")
        for step in v.reasoning_chain:
            detail_short = step.detail[:60].rstrip() + ("..." if len(step.detail) > 60 else "")
            print(f"  {step.pillar:<14} {step.signal:<10} {step.score:>+7.2f}  {detail_short}")

        # Key metrics
        print(f"\n  KEY METRICS:")
        pe = fd.get('pe_trailing')
        pe_fwd = fd.get('pe_forward')
        peg = fd.get('peg_ratio')
        pb = fd.get('pb_ratio')
        ps = fd.get('ps_ratio')
        ev_eb = fd.get('ev_ebitda')
        roe = fd.get('roe')
        roa = fd.get('roa')
        roic = fd.get('roic')
        wacc = fd.get('wacc')
        gm = fd.get('gross_margin')
        om = fd.get('operating_margin')
        nm = fd.get('net_margin')
        rev_gr = fd.get('revenue_growth')
        earn_gr = fd.get('earnings_growth')
        d2e = fd.get('debt_to_equity')
        beta = fd.get('beta')
        target = fd.get('target_mean_price')
        price = r['price']

        upside_str = ""
        if target and price and price > 0:
            upside = (target - price) / price * 100
            upside_str = f"${target:.2f} ({upside:+.1f}%)"

        mc = fd.get('market_cap')
        mc_str = f"${mc/1e12:.2f}T" if mc and mc >= 1e12 else (f"${mc/1e9:.1f}B" if mc else "N/A")

        print(f"  Market Cap   : {mc_str:<14}  P/E (trail/fwd): {fmt_num(pe,1)} / {fmt_num(pe_fwd,1)}")
        print(f"  PEG Ratio    : {fmt_num(peg,2):<14}  P/B: {fmt_num(pb,2)}   P/S: {fmt_num(ps,2)}   EV/EBITDA: {fmt_num(ev_eb,1)}")
        print(f"  Gross Margin : {fmt_pct(gm):<14}  Oper Margin: {fmt_pct(om)}   Net Margin: {fmt_pct(nm)}")
        print(f"  ROE          : {fmt_pct(roe):<14}  ROA: {fmt_pct(roa)}   ROIC: {fmt_pct(roic)}   WACC: {fmt_pct(wacc)}")
        print(f"  Rev Growth   : {fmt_pct(rev_gr):<14}  Earnings Growth: {fmt_pct(earn_gr)}")
        print(f"  Debt/Equity  : {fmt_num(d2e,2):<14}  Beta: {fmt_num(beta,2)}")
        print(f"  Analyst Target: {upside_str if upside_str else 'N/A':<14}  Rec: {fd.get('recommendation_key','N/A')}")

        # Governance
        if gov:
            crit = [f for f in gov.red_flags if f.severity == "CRITICAL"]
            high = [f for f in gov.red_flags if f.severity == "HIGH"]
            med  = [f for f in gov.red_flags if f.severity == "MEDIUM"]
            print(f"\n  GOVERNANCE: Risk Score={gov.risk_score:.0f}/100  Level={gov.overall_risk_level}")
            if crit:
                print(f"  ⚠ CRITICAL FLAGS: {', '.join(f.title for f in crit[:3])}")
            if high:
                print(f"  ▲ HIGH FLAGS: {', '.join(f.title for f in high[:3])}")
            if med:
                print(f"  ● MEDIUM FLAGS: {', '.join(f.title for f in med[:3])}")
            if not crit and not high and not med:
                print(f"  ✓ No material governance red flags")

        # Risk
        if rp:
            md = rp.max_drawdown
            sharpe = rp.sharpe_approx
            vol = rp.volatility_metrics.get("annualized_volatility")
            print(f"\n  RISK PROFILE:")
            print(f"  Max Drawdown: {fmt_pct(md)}  |  Sharpe: {fmt_num(sharpe,2)}  |  Ann. Vol: {fmt_pct(vol)}")
            if rp.stop_losses:
                stops = sorted(rp.stop_losses, key=lambda s: s.stop_price)
                sl = stops[0]
                print(f"  Stop-Loss (ATR/Support): ${sl.stop_price:.2f}  ({sl.distance_pct:.1f}% below current)")

        # Quality scores
        quality = r.get("quality")
        if quality:
            pf = quality.piotroski_f
            az = quality.altman_z
            bm = quality.beneish_m
            print(f"\n  QUALITY SCORES:")
            if pf is not None:
                label = quality.piotroski_label or ("Strong" if pf >= 7 else ("Weak" if pf <= 2 else "Moderate"))
                print(f"  Piotroski F-Score : {pf}/9  ({label})")
            if az is not None:
                print(f"  Altman Z-Score    : {az:.2f}  ({quality.altman_zone})")
            if bm is not None:
                flag = "  ⚠ MANIPULATION RISK" if quality.beneish_flag else "  ✓ Clean"
                print(f"  Beneish M-Score   : {bm:.2f}{flag}")

        # DCF intrinsic value
        dcf = r.get("dcf")
        if dcf and dcf.intrinsic_value and not dcf.error:
            print(f"\n  DCF VALUATION:")
            print(f"  Intrinsic Value   : ${dcf.intrinsic_value:.2f}  ({dcf.valuation_label})")
            if dcf.upside_pct is not None:
                print(f"  Upside/Downside   : {dcf.upside_pct*100:+.1f}%  (MoS: {dcf.margin_of_safety*100:.1f}%)")
            print(f"  FCF Base          : ${dcf.fcf_base/1e9:.2f}B  |  WACC: {dcf.wacc*100:.1f}%  |  "
                  f"Growth: {dcf.growth_stage1*100:.1f}% → {dcf.growth_stage2*100:.1f}%")
        elif dcf and dcf.error:
            print(f"\n  DCF: {dcf.error}")

        # Options signals
        opts = r.get("options")
        if opts and not opts.error:
            print(f"\n  OPTIONS SIGNALS  ({opts.nearest_expiry}):")
            if opts.avg_iv is not None:
                print(f"  ATM IV            : {opts.avg_iv*100:.1f}%  |  "
                      f"Realized Vol: {opts.realized_vol_30d*100:.1f}%" if opts.realized_vol_30d else
                      f"  ATM IV            : {opts.avg_iv*100:.1f}%")
            if opts.iv_rank is not None:
                print(f"  IV Rank           : {opts.iv_rank:.0f}/100  |  IV Percentile: {opts.iv_percentile:.0f}%")
            if opts.put_call_oi_ratio is not None:
                print(f"  P/C OI Ratio      : {opts.put_call_oi_ratio:.2f}  |  P/C Vol Ratio: {opts.put_call_vol_ratio:.2f}" if opts.put_call_vol_ratio else
                      f"  P/C OI Ratio      : {opts.put_call_oi_ratio:.2f}")
            if opts.max_pain_price:
                print(f"  Max Pain          : ${opts.max_pain_price:.2f}")
            print(f"  Signal            : {opts.signal}")
        elif opts and opts.error:
            print(f"\n  OPTIONS: {opts.error}")

        # Peer comps
        peers = r.get("peers")
        if peers and not peers.error and len(peers.peers) > 1:
            print(f"\n  PEER COMPARABLES  (sector: {peers.sector}):")
            subj_ticker = r["ticker"]
            header = f"  {'Ticker':<8} {'P/E':>6} {'Fwd P/E':>7} {'EV/EBITDA':>9} {'P/S':>5} {'ROE':>7} {'Rev Gr':>7} {'Net Mgn':>7}"
            print(header)
            print(f"  {'-'*7} {'-'*6} {'-'*7} {'-'*9} {'-'*5} {'-'*7} {'-'*7} {'-'*7}")
            for p in peers.peers[:9]:
                marker = " *" if p.is_subject else ""
                print(f"  {p.ticker:<8} "
                      f"{fmt_num(p.pe_trailing,1):>6} "
                      f"{fmt_num(p.pe_forward,1):>7} "
                      f"{fmt_num(p.ev_ebitda,1):>9} "
                      f"{fmt_num(p.ps_ratio,1):>5} "
                      f"{fmt_pct(p.roe):>7} "
                      f"{fmt_pct(p.revenue_growth):>7} "
                      f"{fmt_pct(p.net_margin):>7}"
                      f"{marker}")
            # Percentile highlights
            pct = peers.percentile_ranks
            if pct:
                highlights = []
                for k, label in [("pe_forward", "Fwd P/E"), ("roe", "ROE"), ("revenue_growth", "Rev Growth"), ("net_margin", "Net Margin")]:
                    if k in pct:
                        highlights.append(f"{label}: {pct[k]:.0f}th pct")
                if highlights:
                    print(f"  Subject percentile ranks: {' | '.join(highlights)}")

        # Bull / Bear
        print(f"\n  BULL CASE: {v.bull_case[:120]}")
        print(f"  BEAR CASE: {v.bear_case[:120]}")

        if v.catalysts:
            print(f"\n  CATALYSTS: {' | '.join(v.catalysts[:4])}")
        if v.key_risks:
            print(f"  KEY RISKS: {' | '.join(v.key_risks[:4])}")

        if v.action_plan:
            print(f"\n  ACTION PLAN:")
            print(f"  {v.action_plan[:300]}")

    # --- PORTFOLIO RECOMMENDATION ---
    print(f"\n\n{'='*90}")
    print("  PORTFOLIO SWITCH RECOMMENDATIONS")
    print(f"  Current Holdings: {', '.join(sorted(HOLDINGS))}")
    print(f"{'='*90}")

    held_results = {r["ticker"]: r for r in valid if r["ticker"] in HOLDINGS}
    other_results = {r["ticker"]: r for r in valid if r["ticker"] not in HOLDINGS}

    print("\n  CURRENT HOLDINGS ASSESSMENT:")
    for ticker in sorted(HOLDINGS):
        if ticker in held_results:
            r = held_results[ticker]
            v = r["verdict"]
            flag = ""
            if v.action in ("STRONG SELL", "SELL"):
                flag = "  <<< CONSIDER EXITING"
            elif v.action == "HOLD":
                flag = "  (hold, watch)"
            elif v.action in ("BUY", "STRONG BUY"):
                flag = "  <<< ADD TO POSITION"
            print(f"  {ticker:<6} {v.action:<14} Score: {v.composite_score:+.3f}  Conf: {v.confidence}{flag}")
        else:
            print(f"  {ticker:<6} No data available")

    print("\n  NON-HELD OPPORTUNITIES (RANKED BY SCORE):")
    candidates = sorted(other_results.values(), key=lambda r: r["verdict"].composite_score, reverse=True)
    for r in candidates:
        v = r["verdict"]
        flag = ""
        if v.action in ("STRONG BUY", "BUY"):
            flag = "  <<< CONSIDER ADDING"
        print(f"  {r['ticker']:<6} {v.action:<14} Score: {v.composite_score:+.3f}  Conf: {v.confidence}  {r['name'][:30]}{flag}")

    # Switch Matrix
    print("\n  SWITCH MATRIX (From → To):")
    sells = [r for r in valid if r["ticker"] in HOLDINGS and r["verdict"].action in ("SELL", "STRONG SELL")]
    buys = [r for r in valid if r["ticker"] not in HOLDINGS and r["verdict"].action in ("BUY", "STRONG BUY")]

    if not sells and not buys:
        print("  No immediate forced switches — review HOLDs below for selective rotation.")
    else:
        if sells:
            print(f"  REDUCE/EXIT: {', '.join(r['ticker'] for r in sells)}")
        if buys:
            print(f"  ENTER/ADD:   {', '.join(r['ticker'] for r in buys)}")
        if sells and buys:
            for s in sells:
                for b in buys:
                    print(f"  → Switch {s['ticker']} (score {s['verdict'].composite_score:+.3f}) "
                          f"→ {b['ticker']} (score {b['verdict'].composite_score:+.3f})")

    print(f"\n  OVERALL PORTFOLIO QUALITY SCORE: ", end="")
    held_scores = [held_results[t]["verdict"].composite_score for t in HOLDINGS if t in held_results]
    if held_scores:
        avg = sum(held_scores) / len(held_scores)
        print(f"{avg:+.3f} (avg of held positions)")
    else:
        print("N/A")

    print("\n" + "="*90)
    print("  END OF REPORT")
    print("="*90 + "\n")


def main():
    print("Fetching shared macro & sector context (1x only)...")
    try:
        macro_result = MacroAnalyzer().analyze()
    except Exception as e:
        print(f"[WARN] Macro analysis failed: {e}")
        macro_result = None

    try:
        cycle_phase = "Expansion"
        rates_rising = False
        if macro_result and macro_result.cycle:
            cycle_phase = macro_result.cycle.phase
        if macro_result:
            for ind in macro_result.indicators:
                if "rate" in ind.name.lower() and ind.signal == "bearish":
                    rates_rising = True
                    break
        sector_result = SectorAnalyzer().analyze(cycle_phase=cycle_phase, rates_rising=rates_rising)
    except Exception as e:
        print(f"[WARN] Sector analysis failed: {e}")
        sector_result = None

    results = []
    for ticker in TICKERS:
        r = run_analysis(ticker, macro_result, sector_result)
        results.append(r)
        time.sleep(1)  # gentle throttle between tickers

    print_report(results)


if __name__ == "__main__":
    main()
