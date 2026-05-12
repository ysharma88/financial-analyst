"""Alternative Data Analytics — 5 institutional-grade layers built on free public data.

Layer 1 — News Sentiment:
    Scores recent headlines using VADER sentiment analysis.
    Flags when composite score crosses ±0.225 threshold (institutional trigger level).

Layer 2 — Insider & Regulatory Forensics:
    Parses yfinance insider_transactions (Form 4 proxy).
    Detects net buy/sell pressure, plan adoption signals, and cluster exit patterns.

Layer 3 — 13F Cluster Buying:
    Identifies when 3+ top institutional holders initiated new positions in the same quarter.
    Cross-references quarterly pct_change to flag high-conviction consensus accumulation.

Layer 4 — Vanna & Charm Flow:
    Computes second-order options Greeks from the live options chain.
    Net Vanna > 0 means vol-crush events (post-FOMC/CPI) force dealers to BUY mechanically.
    Net Charm > 0 means daily theta decay forces dealers to BUY, providing passive support.

Layer 5 — Political Alpha (Congressional Trades):
    Fetches House & Senate stock disclosures via public APIs (no key required).
    Flags recent member trades in the same ticker — early signal for regulatory events.
"""

from __future__ import annotations

import logging
import math
import urllib.request
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("alternative_data")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SentimentResult:
    composite_score: float          # -1 to +1
    article_count: int
    bullish_count: int
    bearish_count: int
    triggered: bool                  # crossed ±0.225 institutional threshold
    direction: str                   # "Bullish", "Bearish", "Neutral"
    top_headlines: list[dict]        # [{title, score, published}]
    error: Optional[str] = None


@dataclass
class InsiderTrade:
    name: str
    title: str
    transaction: str                 # "Buy" / "Sell" / "Sale"
    shares: int
    value: float
    date: str
    is_cluster_exit: bool = False    # same person filed multiple sells < 30 days


@dataclass
class InsiderFlowResult:
    net_shares_bought: int           # positive = net buying
    net_value_bought: float
    buy_count: int
    sell_count: int
    cluster_exit_flag: bool          # ≥2 insiders selling in same 30-day window
    signal: str                      # "Accumulating", "Distributing", "Mixed", "No Activity"
    recent_trades: list[InsiderTrade]
    error: Optional[str] = None


@dataclass
class ClusterHolder:
    name: str
    pct_held: float
    pct_change: float
    is_new_position: bool            # large positive change suggests new entry


@dataclass
class ClusterBuyingResult:
    cluster_count: int               # number of top funds with new/growing positions
    consensus_signal: str            # "High Conviction Buy", "Moderate Accumulation", "Neutral", "Distribution"
    new_entrants: list[ClusterHolder]
    exits: list[ClusterHolder]
    avg_position_change: float
    error: Optional[str] = None


@dataclass
class CongressionalTrade:
    member: str
    chamber: str                     # "House" / "Senate"
    party: str
    transaction: str                 # "Purchase" / "Sale"
    amount_range: str                # e.g. "$1,001 - $15,000"
    trade_date: str
    disclosure_date: str
    days_to_disclose: int            # delay in filing — longer = more suspicious


@dataclass
class CongressionalResult:
    trades: list[CongressionalTrade]
    net_congressional_bias: str      # "Buying", "Selling", "Mixed", "None"
    member_count: int
    alpha_signal: str                # "Strong Alpha Signal", "Weak Signal", "No Signal"
    error: Optional[str] = None


@dataclass
class VannaCharmResult:
    net_vanna: float                 # $ notional; positive = vol-crush → dealer buying
    net_charm: float                 # $ notional; positive = time-decay → dealer buying
    vanna_signal: str                # "Vol-Crush Rally Fuel", "Vol-Crush Selloff Risk", "Neutral"
    charm_signal: str                # "Theta Supports Price", "Theta Pressures Price", "Neutral"
    vanna_flip_vol: Optional[float]  # implied vol level where vanna switches sign
    expiry_used: str
    error: Optional[str] = None


@dataclass
class AlternativeDataResult:
    ticker: str
    sentiment: Optional[SentimentResult] = None
    insider_flow: Optional[InsiderFlowResult] = None
    cluster_buying: Optional[ClusterBuyingResult] = None
    congressional: Optional[CongressionalResult] = None
    vanna_charm: Optional[VannaCharmResult] = None
    overall_signal: str = "Insufficient Data"
    signal_count: int = 0            # how many layers are bullish
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Layer 1: News Sentiment
# ---------------------------------------------------------------------------

def _vader_score(text: str) -> float:
    """Lightweight keyword-based sentiment scorer when VADER unavailable."""
    text_lower = text.lower()
    bullish = ["beat", "record", "growth", "upgrade", "strong", "surge", "rally",
               "profit", "revenue", "outperform", "buy", "positive", "raise", "gain",
               "expansion", "accelerat", "breakout", "momentum", "innovative", "partnership"]
    bearish = ["miss", "loss", "downgrade", "decline", "fall", "weak", "sell",
               "underperform", "cut", "reduce", "risk", "concern", "lawsuit", "probe",
               "investigation", "fraud", "breach", "layoff", "restructur", "warning"]
    score = sum(0.15 for w in bullish if w in text_lower)
    score -= sum(0.15 for w in bearish if w in text_lower)
    return max(-1.0, min(1.0, score))


def compute_news_sentiment(ticker: str) -> SentimentResult:
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        news = t.news or []

        # Try VADER first, fall back to keyword scorer
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            analyzer = SentimentIntensityAnalyzer()
            score_fn = lambda text: analyzer.polarity_scores(text)["compound"]
        except ImportError:
            score_fn = _vader_score

        scored = []
        cutoff = datetime.now() - timedelta(days=30)
        for item in news[:30]:
            title = item.get("title", "")
            pub_ts = item.get("providerPublishTime", 0)
            pub_dt = datetime.fromtimestamp(pub_ts) if pub_ts else datetime.now()
            if pub_dt < cutoff:
                continue
            s = score_fn(title)
            scored.append({
                "title": title,
                "score": round(s, 3),
                "published": pub_dt.strftime("%Y-%m-%d"),
                "url": item.get("link", ""),
            })

        if not scored:
            return SentimentResult(0, 0, 0, 0, False, "Neutral", [], error="No recent news")

        scores = [a["score"] for a in scored]
        composite = float(np.mean(scores))
        bullish_count = sum(1 for s in scores if s > 0.05)
        bearish_count = sum(1 for s in scores if s < -0.05)
        triggered = abs(composite) >= 0.225
        direction = "Bullish" if composite > 0.05 else "Bearish" if composite < -0.05 else "Neutral"

        top = sorted(scored, key=lambda x: abs(x["score"]), reverse=True)[:5]

        return SentimentResult(
            composite_score=round(composite, 3),
            article_count=len(scored),
            bullish_count=bullish_count,
            bearish_count=bearish_count,
            triggered=triggered,
            direction=direction,
            top_headlines=top,
        )
    except Exception as e:
        logger.warning("News sentiment failed for %s: %s", ticker, e)
        return SentimentResult(0, 0, 0, 0, False, "Neutral", [], error=str(e))


# ---------------------------------------------------------------------------
# Layer 2: Insider Flow & Regulatory Forensics
# ---------------------------------------------------------------------------

def compute_insider_flow(ticker: str) -> InsiderFlowResult:
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)

        txns = t.insider_transactions
        if txns is None or len(txns) == 0:
            return InsiderFlowResult(0, 0, 0, 0, False, "No Activity", [])

        # Normalize column names (yfinance column names vary)
        df = txns.copy()
        df.columns = [c.strip() for c in df.columns]

        cutoff = datetime.now() - timedelta(days=90)
        trades: list[InsiderTrade] = []
        net_shares = 0
        net_value = 0.0
        buy_count = 0
        sell_count = 0

        sell_dates: list[datetime] = []

        for _, row in df.iterrows():
            try:
                raw_date = row.get("Start Date") or row.get("Date") or row.get("startDate")
                if raw_date is None:
                    continue
                if hasattr(raw_date, "to_pydatetime"):
                    trade_dt = raw_date.to_pydatetime()
                elif isinstance(raw_date, str):
                    trade_dt = datetime.strptime(raw_date[:10], "%Y-%m-%d")
                else:
                    trade_dt = datetime.combine(raw_date, datetime.min.time())

                if trade_dt < cutoff:
                    continue

                txn_text = str(row.get("Transaction", row.get("transaction", ""))).lower()
                shares_raw = row.get("Shares", row.get("shares", 0)) or 0
                value_raw = row.get("Value", row.get("value", 0)) or 0
                name = str(row.get("Insider", row.get("insider", row.get("Name", ""))))
                title = str(row.get("Position", row.get("position", row.get("Title", ""))))

                try:
                    shares = abs(int(float(str(shares_raw).replace(",", ""))))
                except Exception:
                    shares = 0
                try:
                    value = abs(float(str(value_raw).replace(",", "").replace("$", "")))
                except Exception:
                    value = 0.0

                is_buy = any(k in txn_text for k in ["purchase", "buy", "acquisition", "award", "grant"])
                is_sell = any(k in txn_text for k in ["sale", "sell", "disposed", "automatic"])

                if is_buy:
                    net_shares += shares
                    net_value += value
                    buy_count += 1
                    label = "Buy"
                elif is_sell:
                    net_shares -= shares
                    net_value -= value
                    sell_count += 1
                    sell_dates.append(trade_dt)
                    label = "Sale"
                else:
                    label = "Other"

                trades.append(InsiderTrade(
                    name=name, title=title, transaction=label,
                    shares=shares, value=value,
                    date=trade_dt.strftime("%Y-%m-%d"),
                ))
            except Exception:
                continue

        # Cluster exit: ≥2 different insiders selling within any 30-day window
        cluster_exit = False
        if len(sell_dates) >= 2:
            sell_dates_sorted = sorted(sell_dates)
            for i in range(len(sell_dates_sorted) - 1):
                if (sell_dates_sorted[i + 1] - sell_dates_sorted[i]).days <= 30:
                    cluster_exit = True
                    break

        if buy_count == 0 and sell_count == 0:
            signal = "No Activity"
        elif buy_count >= 2 and net_shares > 0:
            signal = "Accumulating"
        elif sell_count >= 2 and net_shares < 0:
            if cluster_exit:
                signal = "Cluster Exit — High Alert"
            else:
                signal = "Distributing"
        else:
            signal = "Mixed"

        return InsiderFlowResult(
            net_shares_bought=net_shares,
            net_value_bought=round(net_value, 0),
            buy_count=buy_count,
            sell_count=sell_count,
            cluster_exit_flag=cluster_exit,
            signal=signal,
            recent_trades=trades[:15],
        )
    except Exception as e:
        logger.warning("Insider flow failed for %s: %s", ticker, e)
        return InsiderFlowResult(0, 0, 0, 0, False, "No Activity", [], error=str(e))


# ---------------------------------------------------------------------------
# Layer 3: 13F Cluster Buying
# ---------------------------------------------------------------------------

def compute_cluster_buying(ticker: str) -> ClusterBuyingResult:
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        ih = t.institutional_holders

        if ih is None or len(ih) == 0:
            return ClusterBuyingResult(0, "Neutral", [], [], 0.0, error="No institutional data")

        new_entrants: list[ClusterHolder] = []
        exits: list[ClusterHolder] = []
        changes = []

        for _, row in ih.head(20).iterrows():
            name = str(row.get("Holder", ""))
            pct = float(row.get("pctHeld", 0) or 0)
            chg = float(row.get("pctChange", 0) or 0)
            changes.append(chg)

            if chg >= 0.25:  # 25%+ increase in position = likely new entry or large add
                new_entrants.append(ClusterHolder(name=name, pct_held=pct, pct_change=chg, is_new_position=chg >= 0.5))
            elif chg <= -0.25:  # 25%+ reduction
                exits.append(ClusterHolder(name=name, pct_held=pct, pct_change=chg, is_new_position=False))

        avg_chg = float(np.mean(changes)) if changes else 0.0
        cluster_count = len([h for h in new_entrants if h.is_new_position])

        if cluster_count >= 3:
            signal = "High Conviction Buy — 3+ Funds Initiated"
        elif len(new_entrants) >= 3:
            signal = "Moderate Accumulation"
        elif len(exits) >= 3:
            signal = "Distribution — Institutional Selling"
        elif avg_chg > 0.05:
            signal = "Mild Accumulation"
        elif avg_chg < -0.05:
            signal = "Mild Distribution"
        else:
            signal = "Neutral"

        return ClusterBuyingResult(
            cluster_count=cluster_count,
            consensus_signal=signal,
            new_entrants=new_entrants[:10],
            exits=exits[:5],
            avg_position_change=round(avg_chg, 4),
        )
    except Exception as e:
        logger.warning("Cluster buying failed for %s: %s", ticker, e)
        return ClusterBuyingResult(0, "Neutral", [], [], 0.0, error=str(e))


# ---------------------------------------------------------------------------
# Layer 4: Vanna & Charm
# ---------------------------------------------------------------------------

def _bs_greeks(S: float, K: float, T: float, r: float, sigma: float):
    """Returns (delta, gamma, vanna, charm) for a call option."""
    try:
        from scipy.stats import norm
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return 0, 0, 0, 0
        sqrt_T = math.sqrt(T)
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        npdf_d1 = norm.pdf(d1)

        delta = norm.cdf(d1)
        gamma = npdf_d1 / (S * sigma * sqrt_T)
        # Vanna = ∂delta/∂sigma = -d2/sigma * N'(d1) (same for calls and puts)
        vanna = -npdf_d1 * d2 / sigma
        # Charm = ∂delta/∂t (annualised, sign: positive = delta increases over time)
        charm = -npdf_d1 * (2 * r * T - d2 * sigma * sqrt_T) / (2 * T * sigma * sqrt_T)
        return delta, gamma, vanna, charm
    except Exception:
        return 0, 0, 0, 0


def compute_vanna_charm(ticker: str, current_price: float) -> VannaCharmResult:
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        expiries = t.options
        if not expiries:
            return VannaCharmResult(0, 0, "Neutral", "Neutral", None, "", error="No options data")

        today = date.today()
        expiry = None
        for exp in expiries:
            exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
            days = (exp_date - today).days
            if 3 <= days <= 45:  # prefer 3–45 DTE where vanna/charm effects are strongest
                expiry = exp
                break
        if expiry is None:
            expiry = expiries[0]

        chain = t.option_chain(expiry)
        exp_date = datetime.strptime(expiry, "%Y-%m-%d").date()
        T = max((exp_date - today).days / 365.0, 1 / 365.0)
        r = 0.045

        net_vanna = 0.0
        net_charm = 0.0

        for _, row in chain.calls.iterrows():
            iv = float(row.get("impliedVolatility", 0) or 0)
            oi = int(row.get("openInterest", 0) or 0)
            K = float(row["strike"])
            if iv > 0.001 and oi > 0:
                _, _, vanna, charm = _bs_greeks(current_price, K, T, r, iv)
                # Dealer short calls → positive vanna/charm sign (they benefit from vol drop)
                multiplier = oi * 100 * current_price
                net_vanna += vanna * multiplier
                net_charm += charm * multiplier

        for _, row in chain.puts.iterrows():
            iv = float(row.get("impliedVolatility", 0) or 0)
            oi = int(row.get("openInterest", 0) or 0)
            K = float(row["strike"])
            if iv > 0.001 and oi > 0:
                _, _, vanna, charm = _bs_greeks(current_price, K, T, r, iv)
                # Dealer long puts → negative sign (they are forced sellers when vol drops)
                multiplier = oi * 100 * current_price
                net_vanna -= vanna * multiplier
                net_charm -= charm * multiplier

        threshold = abs(current_price) * 1000  # scale-relative threshold
        if net_vanna > threshold:
            vanna_signal = "Vol-Crush Rally Fuel — Dealers Must Buy"
        elif net_vanna < -threshold:
            vanna_signal = "Vol-Crush Selloff Risk — Dealers Must Sell"
        else:
            vanna_signal = "Neutral Vanna"

        if net_charm > threshold:
            charm_signal = "Theta Decay Supports Price"
        elif net_charm < -threshold:
            charm_signal = "Theta Decay Pressures Price"
        else:
            charm_signal = "Neutral Charm"

        return VannaCharmResult(
            net_vanna=round(net_vanna, 0),
            net_charm=round(net_charm, 0),
            vanna_signal=vanna_signal,
            charm_signal=charm_signal,
            vanna_flip_vol=None,
            expiry_used=expiry,
        )
    except Exception as e:
        logger.warning("Vanna/charm failed for %s: %s", ticker, e)
        return VannaCharmResult(0, 0, "Neutral", "Neutral", None, "", error=str(e))


# ---------------------------------------------------------------------------
# Layer 5: Political Alpha — Congressional Trades
# ---------------------------------------------------------------------------

_HOUSE_API = "https://house-stock-watcher-data.s3-us-east-2.amazonaws.com/data/all_transactions.json"
_SENATE_API = "https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&forms=4&dateRange=custom&startdt={start}"

def _fetch_house_trades(ticker: str) -> list[CongressionalTrade]:
    """Fetches House member stock disclosures from the public House Stock Watcher dataset."""
    trades = []
    try:
        req = urllib.request.Request(
            _HOUSE_API,
            headers={"User-Agent": "FinancialAnalyst/1.0 (educational use)"}
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode())

        ticker_upper = ticker.upper()
        cutoff = datetime.now() - timedelta(days=180)

        for item in data:
            if not isinstance(item, dict):
                continue
            tickers_field = str(item.get("ticker", ""))
            if ticker_upper not in [t.strip().upper() for t in tickers_field.split(",")]:
                continue
            try:
                trade_date_str = item.get("transaction_date", "") or item.get("disclosure_date", "")
                trade_dt = datetime.strptime(trade_date_str[:10], "%Y-%m-%d")
                if trade_dt < cutoff:
                    continue
                disclosure_str = item.get("disclosure_date", trade_date_str)
                disc_dt = datetime.strptime(disclosure_str[:10], "%Y-%m-%d")
                days_delay = max(0, (disc_dt - trade_dt).days)

                trades.append(CongressionalTrade(
                    member=item.get("representative", "Unknown"),
                    chamber="House",
                    party=item.get("party", ""),
                    transaction=item.get("type", ""),
                    amount_range=item.get("amount", ""),
                    trade_date=trade_dt.strftime("%Y-%m-%d"),
                    disclosure_date=disc_dt.strftime("%Y-%m-%d"),
                    days_to_disclose=days_delay,
                ))
            except Exception:
                continue
    except Exception as e:
        logger.debug("House trades fetch failed: %s", e)
    return trades


def compute_congressional_alpha(ticker: str) -> CongressionalResult:
    try:
        trades = _fetch_house_trades(ticker)

        if not trades:
            return CongressionalResult(
                trades=[], net_congressional_bias="None",
                member_count=0, alpha_signal="No Signal"
            )

        buy_count = sum(1 for t in trades if "purchase" in t.transaction.lower() or "buy" in t.transaction.lower())
        sell_count = sum(1 for t in trades if "sale" in t.transaction.lower() or "sell" in t.transaction.lower())
        member_count = len(set(t.member for t in trades))

        if buy_count > sell_count * 2:
            bias = "Buying"
        elif sell_count > buy_count * 2:
            bias = "Selling"
        elif buy_count > 0 or sell_count > 0:
            bias = "Mixed"
        else:
            bias = "None"

        # Alpha signal: strong if multiple members buying; even stronger if filing delay is long
        avg_delay = np.mean([t.days_to_disclose for t in trades]) if trades else 0
        if member_count >= 3 and bias == "Buying":
            alpha_signal = "Strong Alpha Signal — Multi-Member Accumulation"
        elif member_count >= 2 and bias == "Buying":
            alpha_signal = "Moderate Alpha Signal — Congressional Buying"
        elif avg_delay > 30 and len(trades) > 0:
            alpha_signal = "Late Disclosure Flag — Review for Timing"
        else:
            alpha_signal = "Weak / No Signal"

        return CongressionalResult(
            trades=sorted(trades, key=lambda x: x.trade_date, reverse=True)[:20],
            net_congressional_bias=bias,
            member_count=member_count,
            alpha_signal=alpha_signal,
        )
    except Exception as e:
        logger.warning("Congressional alpha failed for %s: %s", ticker, e)
        return CongressionalResult([], "None", 0, "No Signal", error=str(e))


# ---------------------------------------------------------------------------
# Aggregate signal
# ---------------------------------------------------------------------------

def _aggregate_signal(result: AlternativeDataResult) -> tuple[str, int]:
    bullish = 0
    bearish = 0

    if result.sentiment and not result.sentiment.error:
        if result.sentiment.direction == "Bullish" and result.sentiment.triggered:
            bullish += 1
        elif result.sentiment.direction == "Bearish" and result.sentiment.triggered:
            bearish += 1

    if result.insider_flow and not result.insider_flow.error:
        if result.insider_flow.signal == "Accumulating":
            bullish += 1
        elif "Distributing" in result.insider_flow.signal or "Exit" in result.insider_flow.signal:
            bearish += 1

    if result.cluster_buying and not result.cluster_buying.error:
        if "Buy" in result.cluster_buying.consensus_signal or "Accumulation" in result.cluster_buying.consensus_signal:
            bullish += 1
        elif "Distribution" in result.cluster_buying.consensus_signal:
            bearish += 1

    if result.congressional and not result.congressional.error:
        if "Strong" in result.congressional.alpha_signal and result.congressional.net_congressional_bias == "Buying":
            bullish += 1

    if result.vanna_charm and not result.vanna_charm.error:
        if "Rally Fuel" in result.vanna_charm.vanna_signal or "Supports" in result.vanna_charm.charm_signal:
            bullish += 1
        elif "Selloff" in result.vanna_charm.vanna_signal or "Pressures" in result.vanna_charm.charm_signal:
            bearish += 1

    net = bullish - bearish
    signal_count = bullish

    if net >= 3:
        return "Strong Alternative Buy Signal", signal_count
    elif net == 2:
        return "Moderate Buy Signal", signal_count
    elif net == 1:
        return "Mild Bullish Lean", signal_count
    elif net == 0 and (bullish + bearish) > 0:
        return "Mixed / Conflicting", signal_count
    elif net == -1:
        return "Mild Bearish Lean", signal_count
    elif net <= -2:
        return "Strong Alternative Sell Signal", signal_count
    else:
        return "Insufficient Data", signal_count


# ---------------------------------------------------------------------------
# TTL constants (seconds) — each layer has its own optimal refresh window
# ---------------------------------------------------------------------------

TTL = {
    "sentiment":     6 * 3600,          # 6 hours  — news breaks intraday
    "insider":       24 * 3600,         # 24 hours — Form 4 filed within 2 business days
    "cluster":       90 * 24 * 3600,    # 90 days  — 13F data only changes quarterly
    "vanna_charm":   24 * 3600,         # 24 hours — options OI refreshes daily
    "congressional": 7 * 24 * 3600,    # 7 days   — STOCK Act filings trickle weekly
}

LAYER_LABELS = {
    "sentiment":     "News Sentiment",
    "insider":       "Insider Flow",
    "cluster":       "13F Cluster Buying",
    "vanna_charm":   "Vanna & Charm",
    "congressional": "Political Alpha",
}


# ---------------------------------------------------------------------------
# Main entry point (uncached — always fetches live)
# ---------------------------------------------------------------------------

def analyze(ticker: str, price: float) -> AlternativeDataResult:
    result = AlternativeDataResult(ticker=ticker)

    try:
        result.sentiment = compute_news_sentiment(ticker)
    except Exception as e:
        result.sentiment = SentimentResult(0, 0, 0, 0, False, "Neutral", [], error=str(e))

    try:
        result.insider_flow = compute_insider_flow(ticker)
    except Exception as e:
        result.insider_flow = InsiderFlowResult(0, 0, 0, 0, False, "No Activity", [], error=str(e))

    try:
        result.cluster_buying = compute_cluster_buying(ticker)
    except Exception as e:
        result.cluster_buying = ClusterBuyingResult(0, "Neutral", [], [], 0.0, error=str(e))

    try:
        result.congressional = compute_congressional_alpha(ticker)
    except Exception as e:
        result.congressional = CongressionalResult([], "None", 0, "No Signal", error=str(e))

    try:
        result.vanna_charm = compute_vanna_charm(ticker, price)
    except Exception as e:
        result.vanna_charm = VannaCharmResult(0, 0, "Neutral", "Neutral", None, "", error=str(e))

    result.overall_signal, result.signal_count = _aggregate_signal(result)
    return result


# ---------------------------------------------------------------------------
# Cached entry point — respects per-layer TTL, returns staleness metadata
# ---------------------------------------------------------------------------

def cached_analyze(ticker: str, price: float,
                   force_layers: Optional[list[str]] = None) -> dict:
    """
    Fetch alternative data with per-layer TTL caching.

    Returns a dict:
      {
        "result":    AlternativeDataResult,
        "staleness": {layer_key: {"age_seconds": float, "status": str, "label": str}},
        "cache_hit": {layer_key: bool},
      }

    force_layers: list of layer keys to bypass cache and re-fetch live.
    """
    import cache as _cache

    force = set(force_layers or [])
    t = ticker.upper()

    staleness: dict[str, dict] = {}
    cache_hit: dict[str, bool] = {}
    result = AlternativeDataResult(ticker=ticker)

    # --- Layer 1: Sentiment ---
    key = "sentiment"
    cached = None if key in force else _cache.get_ttl(t, f"altdata_{key}", TTL[key])
    if cached is not None:
        result.sentiment = cached
        cache_hit[key] = True
    else:
        try:
            result.sentiment = compute_news_sentiment(ticker)
        except Exception as e:
            result.sentiment = SentimentResult(0, 0, 0, 0, False, "Neutral", [], error=str(e))
        _cache.set_ttl(t, f"altdata_{key}", result.sentiment)
        cache_hit[key] = False

    # --- Layer 2: Insider ---
    key = "insider"
    cached = None if key in force else _cache.get_ttl(t, f"altdata_{key}", TTL[key])
    if cached is not None:
        result.insider_flow = cached
        cache_hit[key] = True
    else:
        try:
            result.insider_flow = compute_insider_flow(ticker)
        except Exception as e:
            result.insider_flow = InsiderFlowResult(0, 0, 0, 0, False, "No Activity", [], error=str(e))
        _cache.set_ttl(t, f"altdata_{key}", result.insider_flow)
        cache_hit[key] = False

    # --- Layer 3: Cluster ---
    key = "cluster"
    cached = None if key in force else _cache.get_ttl(t, f"altdata_{key}", TTL[key])
    if cached is not None:
        result.cluster_buying = cached
        cache_hit[key] = True
    else:
        try:
            result.cluster_buying = compute_cluster_buying(ticker)
        except Exception as e:
            result.cluster_buying = ClusterBuyingResult(0, "Neutral", [], [], 0.0, error=str(e))
        _cache.set_ttl(t, f"altdata_{key}", result.cluster_buying)
        cache_hit[key] = False

    # --- Layer 4: Vanna/Charm ---
    key = "vanna_charm"
    cached = None if key in force else _cache.get_ttl(t, f"altdata_{key}", TTL[key])
    if cached is not None:
        result.vanna_charm = cached
        cache_hit[key] = True
    else:
        try:
            result.vanna_charm = compute_vanna_charm(ticker, price)
        except Exception as e:
            result.vanna_charm = VannaCharmResult(0, 0, "Neutral", "Neutral", None, "", error=str(e))
        _cache.set_ttl(t, f"altdata_{key}", result.vanna_charm)
        cache_hit[key] = False

    # --- Layer 5: Congressional ---
    key = "congressional"
    cached = None if key in force else _cache.get_ttl(t, f"altdata_{key}", TTL[key])
    if cached is not None:
        result.congressional = cached
        cache_hit[key] = True
    else:
        try:
            result.congressional = compute_congressional_alpha(ticker)
        except Exception as e:
            result.congressional = CongressionalResult([], "None", 0, "No Signal", error=str(e))
        _cache.set_ttl(t, f"altdata_{key}", result.congressional)
        cache_hit[key] = False

    result.overall_signal, result.signal_count = _aggregate_signal(result)

    # Build staleness metadata
    for layer_key in TTL:
        age = _cache.get_age_seconds(t, f"altdata_{layer_key}")
        ttl = TTL[layer_key]
        if age is None:
            status = "unknown"
            badge = "⚪ No Data"
        else:
            pct = age / ttl
            if pct < 0.33:
                status = "fresh"
                badge = f"🟢 Fresh ({_fmt_age(age)})"
            elif pct < 0.75:
                status = "aging"
                badge = f"🟡 Aging ({_fmt_age(age)})"
            else:
                status = "stale"
                badge = f"🔴 Stale ({_fmt_age(age)})"
        staleness[layer_key] = {
            "age_seconds": age,
            "status": status,
            "badge": badge,
            "ttl_seconds": ttl,
            "label": LAYER_LABELS[layer_key],
        }

    return {"result": result, "staleness": staleness, "cache_hit": cache_hit}


def _fmt_age(seconds: Optional[float]) -> str:
    if seconds is None:
        return "?"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s ago"
    if seconds < 3600:
        return f"{seconds // 60}m ago"
    if seconds < 86400:
        return f"{seconds // 3600}h ago"
    return f"{seconds // 86400}d ago"
