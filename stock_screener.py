"""Stock screener — finds top stocks within a sector using quantitative metrics.

Fetches top companies from yfinance Sector API, then enriches each with
fundamental data to rank by ROIC, P/E, debt-to-equity, margins, and growth.
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict

import pandas as pd
import numpy as np
import yfinance as yf

logger = logging.getLogger(__name__)

SECTOR_KEYS = {
    "Technology": "technology",
    "Healthcare": "healthcare",
    "Financial Services": "financial-services",
    "Financials": "financial-services",
    "Consumer Cyclical": "consumer-cyclical",
    "Consumer Discretionary": "consumer-cyclical",
    "Consumer Defensive": "consumer-defensive",
    "Consumer Staples": "consumer-defensive",
    "Industrials": "industrials",
    "Energy": "energy",
    "Utilities": "utilities",
    "Basic Materials": "basic-materials",
    "Materials": "basic-materials",
    "Real Estate": "real-estate",
    "Communication Services": "communication-services",
}


@dataclass
class ScreenedStock:
    ticker: str
    name: str
    analyst_rating: str
    market_weight: float
    price: Optional[float] = None
    market_cap: Optional[float] = None
    pe_trailing: Optional[float] = None
    pe_forward: Optional[float] = None
    peg_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    roe: Optional[float] = None
    roa: Optional[float] = None
    roic: Optional[float] = None
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    net_margin: Optional[float] = None
    revenue_growth: Optional[float] = None
    earnings_growth: Optional[float] = None
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    free_cashflow: Optional[float] = None
    dividend_yield: Optional[float] = None
    beta: Optional[float] = None
    composite_score: float = 0.0


@dataclass
class ScreenerResult:
    sector: str
    total_found: int = 0
    stocks: List[ScreenedStock] = field(default_factory=list)
    titans: List[ScreenedStock] = field(default_factory=list)  # top ranked
    filters_applied: Dict = field(default_factory=dict)


class StockScreener:

    def screen_sector(
        self,
        sector: str,
        max_stocks: int = 25,
        min_market_cap: float = 0,
        max_pe: Optional[float] = None,
        max_debt_equity: Optional[float] = None,
        min_roe: Optional[float] = None,
        min_revenue_growth: Optional[float] = None,
    ) -> ScreenerResult:
        result = ScreenerResult(sector=sector)
        result.filters_applied = {
            "min_market_cap": min_market_cap,
            "max_pe": max_pe,
            "max_debt_equity": max_debt_equity,
            "min_roe": min_roe,
            "min_revenue_growth": min_revenue_growth,
        }

        sector_key = SECTOR_KEYS.get(sector)
        if not sector_key:
            for k, v in SECTOR_KEYS.items():
                if sector.lower() in k.lower():
                    sector_key = v
                    break

        if not sector_key:
            return result

        # Fetch top companies
        try:
            yf_sector = yf.Sector(sector_key)
            top_df = yf_sector.top_companies
            if top_df is None or len(top_df) == 0:
                return result
        except Exception as e:
            logger.error("Failed to fetch sector %s: %s", sector_key, e)
            return result

        result.total_found = len(top_df)
        tickers_to_fetch = top_df.index.tolist()[:max_stocks]

        # Enrich with fundamental data
        for ticker_sym in tickers_to_fetch:
            try:
                row = top_df.loc[ticker_sym]
                stock = ScreenedStock(
                    ticker=ticker_sym,
                    name=row.get("name", ticker_sym),
                    analyst_rating=str(row.get("rating", "N/A")),
                    market_weight=float(row.get("market weight", 0)),
                )

                info = self._fetch_info(ticker_sym)
                if info:
                    stock.price = info.get("currentPrice", info.get("regularMarketPrice"))
                    stock.market_cap = info.get("marketCap")
                    stock.pe_trailing = info.get("trailingPE")
                    stock.pe_forward = info.get("forwardPE")
                    stock.peg_ratio = info.get("pegRatio")
                    stock.pb_ratio = info.get("priceToBook")
                    stock.roe = info.get("returnOnEquity")
                    stock.roa = info.get("returnOnAssets")
                    stock.gross_margin = info.get("grossMargins")
                    stock.operating_margin = info.get("operatingMargins")
                    stock.net_margin = info.get("profitMargins")
                    stock.revenue_growth = info.get("revenueGrowth")
                    stock.earnings_growth = info.get("earningsGrowth")
                    stock.debt_to_equity = info.get("debtToEquity")
                    stock.current_ratio = info.get("currentRatio")
                    stock.free_cashflow = info.get("freeCashflow")
                    stock.dividend_yield = info.get("dividendYield")
                    stock.beta = info.get("beta")

                    # Compute ROIC approximation: NOPAT / Invested Capital
                    ebit = info.get("ebitda")
                    tax_rate = 0.21
                    total_debt = info.get("totalDebt", 0) or 0
                    total_equity = info.get("marketCap", 0) or 0
                    cash = info.get("totalCash", 0) or 0
                    invested_capital = total_debt + total_equity - cash
                    if ebit and invested_capital > 0:
                        stock.roic = (ebit * (1 - tax_rate)) / invested_capital

                result.stocks.append(stock)
                time.sleep(0.4)

            except Exception as e:
                logger.warning("Failed to enrich %s: %s", ticker_sym, e)
                continue

        # Apply filters
        filtered = result.stocks[:]

        if min_market_cap > 0:
            filtered = [s for s in filtered if s.market_cap and s.market_cap >= min_market_cap]

        if max_pe is not None:
            filtered = [s for s in filtered if s.pe_trailing is None or (s.pe_trailing > 0 and s.pe_trailing <= max_pe)]

        if max_debt_equity is not None:
            filtered = [s for s in filtered if s.debt_to_equity is None or s.debt_to_equity <= max_debt_equity]

        if min_roe is not None:
            filtered = [s for s in filtered if s.roe is not None and s.roe >= min_roe / 100]

        if min_revenue_growth is not None:
            filtered = [s for s in filtered if s.revenue_growth is not None and s.revenue_growth >= min_revenue_growth / 100]

        # Score and rank
        for stock in filtered:
            stock.composite_score = self._compute_score(stock)

        filtered.sort(key=lambda s: s.composite_score, reverse=True)
        result.stocks = filtered
        result.titans = filtered[:10]

        return result

    def _fetch_info(self, ticker: str) -> Optional[dict]:
        for attempt in range(3):
            try:
                info = yf.Ticker(ticker).info
                if info:
                    return info
            except Exception as e:
                if "RateLimit" in type(e).__name__:
                    time.sleep(2 * (attempt + 1))
                else:
                    return None
        return None

    def _compute_score(self, s: ScreenedStock) -> float:
        """Composite score from 0-100 based on quality metrics."""
        score = 50.0  # base

        # ROE (higher is better, max contribution ±15)
        if s.roe is not None:
            roe_pct = s.roe * 100 if abs(s.roe) < 2 else s.roe
            if roe_pct > 25:
                score += 15
            elif roe_pct > 15:
                score += 10
            elif roe_pct > 8:
                score += 3
            elif roe_pct < 0:
                score -= 10

        # ROIC (higher is better, max ±12)
        if s.roic is not None:
            roic_pct = s.roic * 100
            if roic_pct > 20:
                score += 12
            elif roic_pct > 12:
                score += 8
            elif roic_pct > 6:
                score += 3
            elif roic_pct < 0:
                score -= 8

        # P/E (lower is better for value, max ±10)
        if s.pe_trailing is not None and s.pe_trailing > 0:
            if s.pe_trailing < 12:
                score += 10
            elif s.pe_trailing < 20:
                score += 5
            elif s.pe_trailing < 35:
                score += 0
            elif s.pe_trailing < 60:
                score -= 5
            else:
                score -= 8

        # PEG (lower is better, max ±8)
        if s.peg_ratio is not None and s.peg_ratio > 0:
            if s.peg_ratio < 1.0:
                score += 8
            elif s.peg_ratio < 1.5:
                score += 4
            elif s.peg_ratio > 3:
                score -= 5

        # Debt/Equity (lower is better, max ±10)
        if s.debt_to_equity is not None:
            de = s.debt_to_equity / 100 if s.debt_to_equity > 10 else s.debt_to_equity
            if de < 0.3:
                score += 10
            elif de < 0.7:
                score += 5
            elif de < 1.2:
                score += 0
            elif de < 2.5:
                score -= 5
            else:
                score -= 10

        # Revenue growth (max ±8)
        if s.revenue_growth is not None:
            rg = s.revenue_growth * 100 if abs(s.revenue_growth) < 5 else s.revenue_growth
            if rg > 25:
                score += 8
            elif rg > 10:
                score += 5
            elif rg > 3:
                score += 2
            elif rg < 0:
                score -= 5

        # Net margin (max ±8)
        if s.net_margin is not None:
            nm = s.net_margin * 100 if abs(s.net_margin) < 2 else s.net_margin
            if nm > 25:
                score += 8
            elif nm > 12:
                score += 5
            elif nm > 5:
                score += 2
            elif nm < 0:
                score -= 8

        # Analyst rating bonus
        rating_bonus = {
            "Strong Buy": 5, "Buy": 3, "Hold": 0, "Sell": -3, "Strong Sell": -5,
        }
        score += rating_bonus.get(s.analyst_rating, 0)

        return round(max(0, min(100, score)), 1)
