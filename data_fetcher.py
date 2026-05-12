"""Data fetching layer using yfinance with retry logic for rate limiting."""

from __future__ import annotations

import math
import time
import logging

import yfinance as yf
import pandas as pd
import numpy as np

import cache as _cache

logger = logging.getLogger(__name__)

MAX_RETRIES = 4
BASE_DELAY = 2  # seconds


def _retry(func, *args, **kwargs):
    """Execute a callable with exponential backoff on rate-limit errors."""
    for attempt in range(MAX_RETRIES):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            is_rate_limit = "RateLimit" in type(e).__name__ or "429" in str(e)
            if is_rate_limit and attempt < MAX_RETRIES - 1:
                wait = BASE_DELAY * (2 ** attempt)
                logger.warning("Rate limited (attempt %d/%d), waiting %ds…", attempt + 1, MAX_RETRIES, wait)
                time.sleep(wait)
            elif is_rate_limit:
                logger.error("Rate limited after %d retries, returning fallback.", MAX_RETRIES)
                return None
            else:
                raise


class StockDataFetcher:
    """Fetches and caches stock data from Yahoo Finance."""

    def __init__(self, ticker: str, force_live: bool = False):
        self.ticker = ticker.upper()
        if force_live:
            # Clear yfinance's HTTP lru_cache so the new Ticker fetches live data.
            # yfinance caches at two levels in long-running processes:
            #   - YfData.cache_get(): lru_cache on HTTP responses (maxsize=64)
            #   - Ticker._quote._already_fetched: per-instance flag preventing re-fetch
            # Both must be reset. Creating a new Ticker object handles the second;
            # clearing cache_get handles the first.
            try:
                import yfinance.data as _yfdata
                singleton = _yfdata.YfData._instances.get(_yfdata.YfData)
                if singleton is not None and hasattr(singleton, 'cache_get'):
                    singleton.cache_get.cache_clear()
            except Exception:
                pass
        # Always create a fresh Ticker instance — never reuse across calls
        self.stock = yf.Ticker(self.ticker)
        # Reset _already_fetched on the quote scraper so info is re-fetched
        if force_live:
            try:
                self.stock._quote._already_fetched = False
                self.stock._quote._info = None
                self.stock._quote._retired_info = None
            except Exception:
                pass
        self._info = None
        self._history = None

    @property
    def info(self) -> dict:
        if self._info is None:
            cached = _cache.get(self.ticker, "info")
            if cached is not None:
                self._info = cached
            else:
                result = _retry(lambda: self.stock.info)
                self._info = result if isinstance(result, dict) else {}
                if self._info:
                    _cache.set(self.ticker, "info", self._info)
        return self._info

    def get_history(self, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """Fetch historical OHLCV data."""
        cache_key = f"history_{period}_{interval}"
        if self._history is None:
            cached = _cache.get(self.ticker, cache_key)
            if cached is not None:
                self._history = cached
                return self._history
        result = _retry(lambda: self.stock.history(period=period, interval=interval))
        if result is not None and len(result) > 0:
            self._history = result
            _cache.set(self.ticker, cache_key, result)
        elif self._history is None:
            self._history = pd.DataFrame()
        return self._history

    def get_company_name(self) -> str:
        return self.info.get("longName", self.ticker)

    def get_sector(self) -> str:
        return self.info.get("sector", "N/A")

    def get_industry(self) -> str:
        return self.info.get("industry", "N/A")

    def get_current_price(self) -> float:
        return self.info.get("currentPrice", self.info.get("regularMarketPrice", 0))

    def get_market_cap(self) -> float:
        return self.info.get("marketCap", 0)

    def get_fundamental_data(self) -> dict:
        """Extract all fundamental metrics from stock info."""
        i = self.info
        return {
            "price": self.get_current_price(),
            "market_cap": i.get("marketCap", None),
            "enterprise_value": i.get("enterpriseValue", None),

            "pe_trailing": i.get("trailingPE", None),
            "pe_forward": i.get("forwardPE", None),
            "peg_ratio": i.get("pegRatio", None),
            "pb_ratio": i.get("priceToBook", None),
            "ps_ratio": i.get("priceToSalesTrailing12Months", None),
            "ev_ebitda": i.get("enterpriseToEbitda", None),
            "ev_revenue": i.get("enterpriseToRevenue", None),

            "gross_margin": i.get("grossMargins", None),
            "operating_margin": i.get("operatingMargins", None),
            "net_margin": i.get("profitMargins", None),
            "roe": i.get("returnOnEquity", None),
            "roa": i.get("returnOnAssets", None),

            "revenue_growth": i.get("revenueGrowth", None),
            "earnings_growth": i.get("earningsGrowth", None),
            "earnings_quarterly_growth": i.get("earningsQuarterlyGrowth", None),

            "debt_to_equity": i.get("debtToEquity", None),
            "current_ratio": i.get("currentRatio", None),
            "quick_ratio": i.get("quickRatio", None),
            "total_debt": i.get("totalDebt", None),
            "total_cash": i.get("totalCash", None),
            "free_cashflow": i.get("freeCashflow", None),
            "operating_cashflow": i.get("operatingCashflow", None),

            "dividend_yield": i.get("dividendYield", None),
            "payout_ratio": i.get("payoutRatio", None),
            "five_year_avg_dividend_yield": i.get("fiveYearAvgDividendYield", None),

            "eps_trailing": i.get("trailingEps", None),
            "eps_forward": i.get("forwardEps", None),
            "book_value": i.get("bookValue", None),
            "revenue_per_share": i.get("revenuePerShare", None),

            "target_mean_price": i.get("targetMeanPrice", None),
            "target_high_price": i.get("targetHighPrice", None),
            "target_low_price": i.get("targetLowPrice", None),
            "recommendation_key": i.get("recommendationKey", None),
            "num_analyst_opinions": i.get("numberOfAnalystOpinions", None),

            "beta": i.get("beta", None),

            # ROIC & WACC (computed)
            "roic": self._compute_roic(),
            "wacc": self._compute_wacc(),

            "fifty_two_week_high": i.get("fiftyTwoWeekHigh", None),
            "fifty_two_week_low": i.get("fiftyTwoWeekLow", None),
            "fifty_day_avg": i.get("fiftyDayAverage", None),
            "two_hundred_day_avg": i.get("twoHundredDayAverage", None),
            "avg_volume": i.get("averageVolume", None),
        }

    def _compute_roic(self):
        """ROIC = NOPAT / Invested Capital.
        Uses income statement EBIT and balance sheet Invested Capital when
        available, falls back to info-level approximation."""
        try:
            inc = _retry(lambda: self.stock.income_stmt)
            bs = _retry(lambda: self.stock.balance_sheet)

            if inc is not None and len(inc) > 0 and bs is not None and len(bs) > 0:
                ebit = None
                for label in ["EBIT", "Operating Income"]:
                    if label in inc.index:
                        ebit = inc.loc[label].iloc[0]
                        break

                tax_rate = 0.21
                if "Tax Rate For Calcs" in inc.index:
                    tr = inc.loc["Tax Rate For Calcs"].iloc[0]
                    if tr and not (isinstance(tr, float) and math.isnan(tr)):
                        tax_rate = float(tr)

                invested_capital = None
                if "Invested Capital" in bs.index:
                    invested_capital = bs.loc["Invested Capital"].iloc[0]
                else:
                    total_debt = bs.loc["Total Debt"].iloc[0] if "Total Debt" in bs.index else 0
                    equity = bs.loc["Stockholders Equity"].iloc[0] if "Stockholders Equity" in bs.index else 0
                    if total_debt or equity:
                        invested_capital = (total_debt or 0) + (equity or 0)

                if ebit and invested_capital and invested_capital > 0:
                    nopat = float(ebit) * (1 - tax_rate)
                    return nopat / float(invested_capital)
        except Exception:
            pass

        # Fallback from info dict
        i = self.info
        ebitda = i.get("ebitda")
        total_debt = i.get("totalDebt", 0) or 0
        equity = i.get("marketCap", 0) or 0
        cash = i.get("totalCash", 0) or 0
        ic = total_debt + equity - cash
        if ebitda and ic > 0:
            return (ebitda * 0.79) / ic
        return None

    def _compute_wacc(self):
        """WACC = (E/V)*Re + (D/V)*Rd*(1-Tc).
        Re from CAPM: Rf + beta*(Rm - Rf). Uses 10Y yield proxy or 4.5% default."""
        try:
            i = self.info
            beta = i.get("beta", 1.0)
            if beta is None:
                beta = 1.0

            market_cap = i.get("marketCap", 0) or 0
            total_debt = i.get("totalDebt", 0) or 0
            if market_cap == 0:
                return None

            total_value = market_cap + total_debt

            # Risk-free rate proxy
            risk_free = 0.045
            try:
                tnx = yf.Ticker("^TNX")
                tnx_hist = tnx.history(period="5d")
                if tnx_hist is not None and len(tnx_hist) > 0:
                    risk_free = tnx_hist["Close"].iloc[-1] / 100
            except Exception:
                pass

            equity_risk_premium = 0.055
            cost_of_equity = risk_free + beta * equity_risk_premium

            # Cost of debt approximation
            inc = _retry(lambda: self.stock.income_stmt)
            interest_expense = None
            if inc is not None and len(inc) > 0 and "Interest Expense" in inc.index:
                ie = inc.loc["Interest Expense"].iloc[0]
                if ie and not (isinstance(ie, float) and math.isnan(ie)):
                    interest_expense = abs(float(ie))

            if interest_expense and total_debt > 0:
                cost_of_debt = interest_expense / total_debt
            else:
                cost_of_debt = risk_free + 0.015

            tax_rate = 0.21
            if inc is not None and len(inc) > 0 and "Tax Rate For Calcs" in inc.index:
                tr = inc.loc["Tax Rate For Calcs"].iloc[0]
                if tr and not (isinstance(tr, float) and math.isnan(tr)):
                    tax_rate = float(tr)

            e_weight = market_cap / total_value
            d_weight = total_debt / total_value
            wacc = e_weight * cost_of_equity + d_weight * cost_of_debt * (1 - tax_rate)
            return wacc
        except Exception:
            return None

    def get_analyst_price_targets(self) -> dict:
        try:
            result = _retry(lambda: self.stock.analyst_price_targets)
            return result if isinstance(result, dict) else {}
        except Exception:
            return {}

    def get_recommendations_summary(self) -> pd.DataFrame:
        try:
            rec = _retry(lambda: self.stock.recommendations)
            if rec is not None and len(rec) > 0:
                return rec
        except Exception:
            pass
        return pd.DataFrame()

    def get_upgrades_downgrades(self, limit: int = 20) -> pd.DataFrame:
        try:
            ud = _retry(lambda: self.stock.upgrades_downgrades)
            if ud is not None and len(ud) > 0:
                return ud.head(limit)
        except Exception:
            pass
        return pd.DataFrame()

    def get_institutional_holders(self) -> pd.DataFrame:
        try:
            ih = _retry(lambda: self.stock.institutional_holders)
            if ih is not None and len(ih) > 0:
                return ih
        except Exception:
            pass
        return pd.DataFrame()

    def get_mutualfund_holders(self) -> pd.DataFrame:
        try:
            mfh = _retry(lambda: self.stock.mutualfund_holders)
            if mfh is not None and len(mfh) > 0:
                return mfh
        except Exception:
            pass
        return pd.DataFrame()

    def get_major_holders(self) -> pd.DataFrame:
        try:
            mh = _retry(lambda: self.stock.major_holders)
            if mh is not None and len(mh) > 0:
                return mh
        except Exception:
            pass
        return pd.DataFrame()

    def get_income_statement(self) -> pd.DataFrame:
        try:
            cached = _cache.get(self.ticker, "income_stmt")
            if cached is not None:
                return cached
            result = _retry(lambda: self.stock.income_stmt)
            if result is not None and len(result) > 0:
                _cache.set(self.ticker, "income_stmt", result)
                return result
        except Exception:
            pass
        return pd.DataFrame()

    def get_balance_sheet(self) -> pd.DataFrame:
        try:
            cached = _cache.get(self.ticker, "balance_sheet")
            if cached is not None:
                return cached
            result = _retry(lambda: self.stock.balance_sheet)
            if result is not None and len(result) > 0:
                _cache.set(self.ticker, "balance_sheet", result)
                return result
        except Exception:
            pass
        return pd.DataFrame()

    def get_cashflow_statement(self) -> pd.DataFrame:
        try:
            cached = _cache.get(self.ticker, "cashflow_stmt")
            if cached is not None:
                return cached
            result = _retry(lambda: self.stock.cashflow)
            if result is not None and len(result) > 0:
                _cache.set(self.ticker, "cashflow_stmt", result)
                return result
        except Exception:
            pass
        return pd.DataFrame()

    def get_earnings_history(self) -> pd.DataFrame:
        try:
            result = _retry(lambda: self.stock.earnings_history)
            if result is not None and len(result) > 0:
                return result
        except Exception:
            pass
        return pd.DataFrame()

    def get_earnings_estimate(self) -> pd.DataFrame:
        try:
            result = _retry(lambda: self.stock.earnings_estimate)
            if result is not None and len(result) > 0:
                return result
        except Exception:
            pass
        return pd.DataFrame()

    def get_insider_purchases(self) -> pd.DataFrame:
        try:
            result = _retry(lambda: self.stock.insider_purchases)
            if result is not None and len(result) > 0:
                return result
        except Exception:
            pass
        return pd.DataFrame()

    def get_insider_roster(self) -> pd.DataFrame:
        try:
            result = _retry(lambda: self.stock.insider_roster_holders)
            if result is not None and len(result) > 0:
                return result
        except Exception:
            pass
        return pd.DataFrame()

    def get_company_officers(self) -> list:
        return self.info.get("companyOfficers", [])

    def get_calendar(self) -> dict:
        """Fetch next earnings date, ex-dividend date from yfinance calendar."""
        try:
            cal = _retry(lambda: self.stock.calendar)
            if isinstance(cal, dict):
                return cal
        except Exception:
            pass
        return {}

    def get_financial_trends(self) -> dict:
        """Return YoY and QoQ trends from multi-period financial statements."""
        trends = {}
        try:
            inc = self.get_income_statement()
            bs = self.get_balance_sheet()
            cf = self.get_cashflow_statement()

            def series(df, *labels):
                for lbl in labels:
                    if lbl in df.index:
                        return df.loc[lbl]
                return None

            def pct_chg(s):
                if s is None or len(s) < 2:
                    return None
                vals = [v for v in s.values if v is not None and not (isinstance(v, float) and math.isnan(v))]
                if len(vals) < 2 or vals[1] == 0:
                    return None
                return round((vals[0] - vals[1]) / abs(vals[1]), 4)

            if not inc.empty:
                rev_s   = series(inc, "Total Revenue")
                gp_s    = series(inc, "Gross Profit")
                op_s    = series(inc, "Operating Income", "EBIT")
                ni_s    = series(inc, "Net Income")
                ebitda_s = series(inc, "EBITDA", "Normalized EBITDA")

                trends["revenue_yoy"]         = pct_chg(rev_s)
                trends["gross_profit_yoy"]    = pct_chg(gp_s)
                trends["operating_income_yoy"] = pct_chg(op_s)
                trends["net_income_yoy"]      = pct_chg(ni_s)

                # 4-period time series for charts
                if rev_s is not None:
                    trends["revenue_series"]  = {str(k)[:10]: v for k, v in rev_s.head(4).items() if v is not None}
                if ni_s is not None:
                    trends["net_income_series"] = {str(k)[:10]: v for k, v in ni_s.head(4).items() if v is not None}
                if op_s is not None:
                    trends["operating_income_series"] = {str(k)[:10]: v for k, v in op_s.head(4).items() if v is not None}

            if not bs.empty:
                debt_s   = series(bs, "Total Debt")
                equity_s = series(bs, "Stockholders Equity")
                cash_s   = series(bs, "Cash And Cash Equivalents")
                trends["debt_yoy"]   = pct_chg(debt_s)
                trends["equity_yoy"] = pct_chg(equity_s)

            if not cf.empty:
                fcf_s = series(cf, "Free Cash Flow")
                ocf_s = series(cf, "Operating Cash Flow", "Cash Flow From Continuing Operating Activities")
                if fcf_s is not None:
                    trends["fcf_series"] = {str(k)[:10]: v for k, v in fcf_s.head(4).items() if v is not None}
                    trends["fcf_yoy"] = pct_chg(fcf_s)
                if ocf_s is not None:
                    trends["ocf_yoy"] = pct_chg(ocf_s)

        except Exception as e:
            logger.warning("Financial trends extraction failed: %s", e)
        return trends

    def get_governance_scores(self) -> dict:
        i = self.info
        return {
            "audit_risk": i.get("auditRisk"),
            "board_risk": i.get("boardRisk"),
            "compensation_risk": i.get("compensationRisk"),
            "shareholder_rights_risk": i.get("shareHolderRightsRisk"),
            "overall_risk": i.get("overallRisk"),
        }

    def validate(self) -> bool:
        """Check if the ticker is valid and has data."""
        try:
            hist = _retry(lambda: self.stock.history(period="5d"))
            return hist is not None and len(hist) > 0
        except Exception:
            return False
