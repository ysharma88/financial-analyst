"""Peer comparable company analysis.

Fetches the top peers from the same sector, pulls their key valuation/
profitability multiples, and ranks the subject stock relative to them.
Uses the existing StockScreener infrastructure (yf.Sector).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np
import yfinance as yf

logger = logging.getLogger(__name__)


@dataclass
class PeerStock:
    ticker: str
    name: str
    price: Optional[float] = None
    market_cap: Optional[float] = None
    pe_trailing: Optional[float] = None
    pe_forward: Optional[float] = None
    ev_ebitda: Optional[float] = None
    ps_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    peg_ratio: Optional[float] = None
    roe: Optional[float] = None
    net_margin: Optional[float] = None
    revenue_growth: Optional[float] = None
    gross_margin: Optional[float] = None
    debt_to_equity: Optional[float] = None
    is_subject: bool = False   # True = the stock being analyzed


@dataclass
class PeerCompsResult:
    subject_ticker: str
    sector: str
    peers: List[PeerStock] = field(default_factory=list)
    # Percentile ranks for subject (0=lowest, 100=highest in peer group)
    percentile_ranks: dict = field(default_factory=dict)
    # Sector medians
    medians: dict = field(default_factory=dict)
    error: str = ""


MULTIPLES = [
    ("pe_trailing",    "P/E (TTM)"),
    ("pe_forward",     "Fwd P/E"),
    ("ev_ebitda",      "EV/EBITDA"),
    ("ps_ratio",       "P/S"),
    ("pb_ratio",       "P/B"),
    ("peg_ratio",      "PEG"),
    ("roe",            "ROE"),
    ("net_margin",     "Net Margin"),
    ("gross_margin",   "Gross Margin"),
    ("revenue_growth", "Rev Growth"),
    ("debt_to_equity", "D/E"),
]


class PeerComparator:

    def compare(
        self,
        subject_ticker: str,
        subject_info: dict,
        sector: str,
        max_peers: int = 8,
    ) -> PeerCompsResult:
        result = PeerCompsResult(subject_ticker=subject_ticker.upper(), sector=sector)

        # Build subject PeerStock from already-fetched info dict
        subj = self._build_from_info(subject_ticker.upper(), subject_info)
        subj.is_subject = True

        # Fetch sector peers
        peers = self._fetch_peers(sector, subject_ticker, max_peers)
        if not peers:
            result.error = "Could not fetch peer data (rate limit or unsupported sector)"
            result.peers = [subj]
            return result

        result.peers = [subj] + peers

        # Compute medians and percentile ranks for subject
        for field_name, _ in MULTIPLES:
            vals = [getattr(p, field_name) for p in peers
                    if getattr(p, field_name) is not None
                    and getattr(p, field_name) > 0]
            if not vals:
                continue
            result.medians[field_name] = float(np.median(vals))
            subj_val = getattr(subj, field_name)
            if subj_val is not None and subj_val > 0:
                # For valuation multiples lower = better; for margins/growth higher = better
                lower_is_better = field_name in ("pe_trailing", "pe_forward", "ev_ebitda",
                                                  "ps_ratio", "pb_ratio", "peg_ratio", "debt_to_equity")
                all_vals = vals + [subj_val]
                rank = sorted(all_vals).index(subj_val) / (len(all_vals) - 1) * 100 if len(all_vals) > 1 else 50
                result.percentile_ranks[field_name] = round(100 - rank if lower_is_better else rank, 1)

        return result

    def _build_from_info(self, ticker: str, info: dict) -> PeerStock:
        return PeerStock(
            ticker=ticker,
            name=info.get("longName", ticker),
            price=info.get("currentPrice") or info.get("regularMarketPrice"),
            market_cap=info.get("marketCap"),
            pe_trailing=info.get("trailingPE"),
            pe_forward=info.get("forwardPE"),
            ev_ebitda=info.get("enterpriseToEbitda"),
            ps_ratio=info.get("priceToSalesTrailing12Months"),
            pb_ratio=info.get("priceToBook"),
            peg_ratio=info.get("pegRatio"),
            roe=info.get("returnOnEquity"),
            net_margin=info.get("profitMargins"),
            gross_margin=info.get("grossMargins"),
            revenue_growth=info.get("revenueGrowth"),
            debt_to_equity=info.get("debtToEquity"),
        )

    def _fetch_peers(self, sector: str, exclude_ticker: str, max_peers: int) -> List[PeerStock]:
        from stock_screener import SECTOR_KEYS
        sector_key = SECTOR_KEYS.get(sector)
        if not sector_key:
            return []
        try:
            yf_sector = yf.Sector(sector_key)
            top_df = yf_sector.top_companies
            if top_df is None or len(top_df) == 0:
                return []
        except Exception as e:
            logger.warning("Peer sector fetch failed: %s", e)
            return []

        peers = []
        exclude = exclude_ticker.upper()
        for sym in top_df.index.tolist():
            if sym.upper() == exclude:
                continue
            if len(peers) >= max_peers:
                break
            try:
                info = yf.Ticker(sym).info
                if not info or not isinstance(info, dict):
                    continue
                p = self._build_from_info(sym, info)
                if p.market_cap and p.market_cap > 0:
                    peers.append(p)
                time.sleep(0.3)
            except Exception as e:
                logger.debug("Peer fetch failed for %s: %s", sym, e)
                continue
        return peers
