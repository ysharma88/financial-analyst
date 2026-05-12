"""Fundamental analysis engine - evaluates intrinsic value through financial metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MetricScore:
    name: str
    value: Optional[float]
    score: float  # -1.0 (very bearish) to +1.0 (very bullish)
    weight: float
    interpretation: str
    category: str


@dataclass
class FundamentalResult:
    scores: list[MetricScore] = field(default_factory=list)
    overall_score: float = 0.0  # -1.0 to +1.0
    signal: str = "NEUTRAL"
    summary: str = ""

    def compute_overall(self):
        valid = [s for s in self.scores if s.value is not None]
        if not valid:
            self.overall_score = 0.0
            self.signal = "NEUTRAL"
            self.summary = "Insufficient fundamental data available."
            return

        total_weight = sum(s.weight for s in valid)
        if total_weight == 0:
            self.overall_score = 0.0
        else:
            self.overall_score = sum(s.score * s.weight for s in valid) / total_weight

        if self.overall_score >= 0.5:
            self.signal = "STRONG BUY"
        elif self.overall_score >= 0.2:
            self.signal = "BUY"
        elif self.overall_score >= -0.2:
            self.signal = "HOLD"
        elif self.overall_score >= -0.5:
            self.signal = "SELL"
        else:
            self.signal = "STRONG SELL"

        strengths = [s.name for s in valid if s.score > 0.3]
        weaknesses = [s.name for s in valid if s.score < -0.3]

        parts = []
        if strengths:
            parts.append(f"Strengths: {', '.join(strengths[:4])}")
        if weaknesses:
            parts.append(f"Concerns: {', '.join(weaknesses[:4])}")
        self.summary = ". ".join(parts) if parts else "Mixed fundamental signals."

    @property
    def category_scores(self) -> dict[str, float]:
        categories: dict[str, list] = {}
        for s in self.scores:
            if s.value is not None:
                categories.setdefault(s.category, []).append(s.score)
        return {cat: sum(vals) / len(vals) for cat, vals in categories.items()}


class FundamentalAnalyzer:
    """Scores stocks on fundamental metrics using sector-aware thresholds."""

    SECTOR_PE_BENCHMARKS = {
        "Technology": 30,
        "Healthcare": 25,
        "Financial Services": 15,
        "Consumer Cyclical": 20,
        "Consumer Defensive": 22,
        "Industrials": 20,
        "Energy": 12,
        "Utilities": 18,
        "Real Estate": 20,
        "Communication Services": 22,
        "Basic Materials": 15,
    }

    def analyze(self, data: dict, sector: str = "N/A") -> FundamentalResult:
        result = FundamentalResult()

        result.scores.extend(self._score_valuation(data, sector))
        result.scores.extend(self._score_profitability(data))
        result.scores.extend(self._score_capital_efficiency(data))
        result.scores.extend(self._score_growth(data))
        result.scores.extend(self._score_financial_health(data))
        result.scores.extend(self._score_dividends(data))
        result.scores.extend(self._score_analyst_sentiment(data))

        result.compute_overall()
        return result

    def _score_valuation(self, d: dict, sector: str) -> list[MetricScore]:
        scores = []
        benchmark_pe = self.SECTOR_PE_BENCHMARKS.get(sector, 20)

        pe = d.get("pe_trailing")
        if pe is not None and pe > 0:
            if pe < benchmark_pe * 0.6:
                sc, interp = 0.8, "Significantly undervalued"
            elif pe < benchmark_pe * 0.85:
                sc, interp = 0.4, "Moderately undervalued"
            elif pe < benchmark_pe * 1.15:
                sc, interp = 0.0, "Fairly valued"
            elif pe < benchmark_pe * 1.5:
                sc, interp = -0.4, "Moderately overvalued"
            else:
                sc, interp = -0.8, "Significantly overvalued"
            scores.append(MetricScore("P/E Ratio", pe, sc, 1.5, interp, "Valuation"))

        fwd_pe = d.get("pe_forward")
        if fwd_pe is not None and fwd_pe > 0:
            if pe and fwd_pe < pe:
                sc, interp = 0.3, "Earnings expected to grow"
            elif pe and fwd_pe > pe * 1.2:
                sc, interp = -0.3, "Earnings expected to decline"
            else:
                sc, interp = 0.0, "Stable forward earnings"
            scores.append(MetricScore("Forward P/E", fwd_pe, sc, 0.8, interp, "Valuation"))

        peg = d.get("peg_ratio")
        if peg is not None and peg > 0:
            if peg < 0.8:
                sc, interp = 0.8, "Excellent growth-adjusted value"
            elif peg < 1.0:
                sc, interp = 0.5, "Good growth-adjusted value"
            elif peg < 1.5:
                sc, interp = 0.1, "Fairly priced for growth"
            elif peg < 2.5:
                sc, interp = -0.3, "Expensive relative to growth"
            else:
                sc, interp = -0.7, "Very expensive relative to growth"
            scores.append(MetricScore("PEG Ratio", peg, sc, 1.2, interp, "Valuation"))

        pb = d.get("pb_ratio")
        if pb is not None and pb > 0:
            if pb < 1.0:
                sc, interp = 0.6, "Trading below book value"
            elif pb < 3.0:
                sc, interp = 0.2, "Reasonable price-to-book"
            elif pb < 8.0:
                sc, interp = -0.1, "Premium price-to-book"
            else:
                sc, interp = -0.5, "Very high price-to-book"
            scores.append(MetricScore("P/B Ratio", pb, sc, 0.7, interp, "Valuation"))

        ps = d.get("ps_ratio")
        if ps is not None and ps > 0:
            if ps < 1.0:
                sc, interp = 0.6, "Cheap relative to revenue"
            elif ps < 3.0:
                sc, interp = 0.2, "Reasonable price-to-sales"
            elif ps < 10.0:
                sc, interp = -0.2, "Elevated price-to-sales"
            else:
                sc, interp = -0.6, "Very expensive on sales basis"
            scores.append(MetricScore("P/S Ratio", ps, sc, 0.6, interp, "Valuation"))

        ev_ebitda = d.get("ev_ebitda")
        if ev_ebitda is not None and ev_ebitda > 0:
            if ev_ebitda < 8:
                sc, interp = 0.6, "Attractively valued"
            elif ev_ebitda < 14:
                sc, interp = 0.2, "Fairly valued"
            elif ev_ebitda < 25:
                sc, interp = -0.2, "Premium valuation"
            else:
                sc, interp = -0.6, "Expensive valuation"
            scores.append(MetricScore("EV/EBITDA", ev_ebitda, sc, 1.0, interp, "Valuation"))

        return scores

    def _score_profitability(self, d: dict) -> list[MetricScore]:
        scores = []

        roe = d.get("roe")
        if roe is not None:
            roe_pct = roe * 100 if abs(roe) < 1 else roe
            if roe_pct > 25:
                sc, interp = 0.8, "Excellent return on equity"
            elif roe_pct > 15:
                sc, interp = 0.4, "Good return on equity"
            elif roe_pct > 8:
                sc, interp = 0.0, "Average return on equity"
            elif roe_pct > 0:
                sc, interp = -0.4, "Below-average return on equity"
            else:
                sc, interp = -0.8, "Negative return on equity"
            scores.append(MetricScore("ROE", roe, sc, 1.2, interp, "Profitability"))

        roa = d.get("roa")
        if roa is not None:
            roa_pct = roa * 100 if abs(roa) < 1 else roa
            if roa_pct > 15:
                sc, interp = 0.7, "Excellent asset efficiency"
            elif roa_pct > 8:
                sc, interp = 0.3, "Good asset efficiency"
            elif roa_pct > 3:
                sc, interp = 0.0, "Average asset efficiency"
            elif roa_pct > 0:
                sc, interp = -0.3, "Low asset efficiency"
            else:
                sc, interp = -0.7, "Negative returns on assets"
            scores.append(MetricScore("ROA", roa, sc, 0.8, interp, "Profitability"))

        gm = d.get("gross_margin")
        if gm is not None:
            gm_pct = gm * 100 if abs(gm) < 2 else gm
            if gm_pct > 60:
                sc, interp = 0.7, "Strong pricing power"
            elif gm_pct > 40:
                sc, interp = 0.3, "Healthy gross margins"
            elif gm_pct > 20:
                sc, interp = 0.0, "Moderate gross margins"
            else:
                sc, interp = -0.4, "Thin gross margins"
            scores.append(MetricScore("Gross Margin", gm, sc, 0.8, interp, "Profitability"))

        om = d.get("operating_margin")
        if om is not None:
            om_pct = om * 100 if abs(om) < 2 else om
            if om_pct > 30:
                sc, interp = 0.7, "Excellent operational efficiency"
            elif om_pct > 15:
                sc, interp = 0.3, "Good operational efficiency"
            elif om_pct > 5:
                sc, interp = 0.0, "Moderate operational efficiency"
            elif om_pct > 0:
                sc, interp = -0.3, "Thin operating margins"
            else:
                sc, interp = -0.7, "Operating at a loss"
            scores.append(MetricScore("Operating Margin", om, sc, 0.9, interp, "Profitability"))

        nm = d.get("net_margin")
        if nm is not None:
            nm_pct = nm * 100 if abs(nm) < 2 else nm
            if nm_pct > 25:
                sc, interp = 0.6, "Highly profitable"
            elif nm_pct > 10:
                sc, interp = 0.3, "Good profitability"
            elif nm_pct > 3:
                sc, interp = 0.0, "Moderate profitability"
            elif nm_pct > 0:
                sc, interp = -0.3, "Low profitability"
            else:
                sc, interp = -0.6, "Net loss"
            scores.append(MetricScore("Net Margin", nm, sc, 0.7, interp, "Profitability"))

        return scores

    def _score_capital_efficiency(self, d: dict) -> list[MetricScore]:
        """Score ROIC, WACC, ROIC-WACC spread, and FCF Yield."""
        scores = []

        # --- ROIC ---
        roic = d.get("roic")
        if roic is not None:
            roic_pct = roic * 100 if abs(roic) < 2 else roic
            if roic_pct > 25:
                sc, interp = 0.9, "Exceptional capital efficiency — wide economic moat"
            elif roic_pct > 15:
                sc, interp = 0.5, "Strong ROIC — good competitive advantage"
            elif roic_pct > 8:
                sc, interp = 0.1, "Adequate capital returns"
            elif roic_pct > 0:
                sc, interp = -0.3, "Below cost-of-capital territory"
            else:
                sc, interp = -0.8, "Destroying shareholder value"
            scores.append(MetricScore("ROIC", roic, sc, 1.4, interp, "Capital Efficiency"))

        # --- WACC ---
        wacc = d.get("wacc")
        if wacc is not None:
            wacc_pct = wacc * 100 if abs(wacc) < 1 else wacc
            if wacc_pct < 7:
                sc, interp = 0.3, "Low cost of capital — favorable financing"
            elif wacc_pct < 10:
                sc, interp = 0.1, "Moderate cost of capital"
            elif wacc_pct < 14:
                sc, interp = -0.2, "Elevated cost of capital"
            else:
                sc, interp = -0.5, "High cost of capital — risky profile"
            scores.append(MetricScore("WACC", wacc, sc, 0.6, interp, "Capital Efficiency"))

        # --- ROIC - WACC Spread ---
        if roic is not None and wacc is not None:
            roic_v = roic * 100 if abs(roic) < 2 else roic
            wacc_v = wacc * 100 if abs(wacc) < 1 else wacc
            spread = roic_v - wacc_v
            spread_dec = spread / 100
            if spread > 15:
                sc, interp = 0.9, "Massive value creation — ROIC far exceeds WACC"
            elif spread > 5:
                sc, interp = 0.5, "Solid value creation — ROIC comfortably above WACC"
            elif spread > 0:
                sc, interp = 0.1, "Marginal value creation"
            elif spread > -5:
                sc, interp = -0.4, "Slight value destruction — ROIC below WACC"
            else:
                sc, interp = -0.8, "Significant value destruction"
            scores.append(MetricScore("ROIC-WACC Spread", spread_dec, sc, 1.3, interp, "Capital Efficiency"))

        # --- FCF Yield ---
        fcf = d.get("free_cashflow")
        mc = d.get("market_cap")
        if fcf is not None and mc and mc > 0:
            fcf_yield = fcf / mc
            if fcf_yield > 0.08:
                sc, interp = 0.8, "Excellent FCF yield — strong cash machine"
            elif fcf_yield > 0.05:
                sc, interp = 0.5, "Healthy FCF yield"
            elif fcf_yield > 0.02:
                sc, interp = 0.1, "Moderate FCF yield"
            elif fcf_yield > 0:
                sc, interp = -0.2, "Low FCF yield"
            else:
                sc, interp = -0.7, "Negative FCF — cash burn"
            scores.append(MetricScore("FCF Yield", fcf_yield, sc, 1.2, interp, "Capital Efficiency"))

        return scores

    def _score_growth(self, d: dict) -> list[MetricScore]:
        scores = []

        rg = d.get("revenue_growth")
        if rg is not None:
            rg_pct = rg * 100 if abs(rg) < 5 else rg
            if rg_pct > 30:
                sc, interp = 0.8, "Exceptional revenue growth"
            elif rg_pct > 15:
                sc, interp = 0.5, "Strong revenue growth"
            elif rg_pct > 5:
                sc, interp = 0.2, "Moderate revenue growth"
            elif rg_pct > 0:
                sc, interp = -0.1, "Slow revenue growth"
            else:
                sc, interp = -0.6, "Revenue declining"
            scores.append(MetricScore("Revenue Growth", rg, sc, 1.3, interp, "Growth"))

        eg = d.get("earnings_growth")
        if eg is not None:
            eg_pct = eg * 100 if abs(eg) < 5 else eg
            if eg_pct > 40:
                sc, interp = 0.8, "Exceptional earnings growth"
            elif eg_pct > 15:
                sc, interp = 0.4, "Strong earnings growth"
            elif eg_pct > 5:
                sc, interp = 0.1, "Moderate earnings growth"
            elif eg_pct > 0:
                sc, interp = -0.1, "Sluggish earnings growth"
            else:
                sc, interp = -0.6, "Earnings declining"
            scores.append(MetricScore("Earnings Growth", eg, sc, 1.2, interp, "Growth"))

        eqg = d.get("earnings_quarterly_growth")
        if eqg is not None:
            eqg_pct = eqg * 100 if abs(eqg) < 5 else eqg
            if eqg_pct > 30:
                sc, interp = 0.6, "Strong quarterly momentum"
            elif eqg_pct > 10:
                sc, interp = 0.3, "Positive quarterly trend"
            elif eqg_pct > 0:
                sc, interp = 0.0, "Flat quarterly performance"
            else:
                sc, interp = -0.5, "Quarterly earnings declining"
            scores.append(MetricScore("Quarterly Earnings Growth", eqg, sc, 0.8, interp, "Growth"))

        return scores

    def _score_financial_health(self, d: dict) -> list[MetricScore]:
        scores = []

        de = d.get("debt_to_equity")
        if de is not None and de >= 0:
            de_ratio = de / 100 if de > 10 else de
            if de_ratio < 0.3:
                sc, interp = 0.7, "Very low leverage"
            elif de_ratio < 0.6:
                sc, interp = 0.3, "Conservative leverage"
            elif de_ratio < 1.0:
                sc, interp = 0.0, "Moderate leverage"
            elif de_ratio < 2.0:
                sc, interp = -0.4, "High leverage"
            else:
                sc, interp = -0.8, "Extremely high leverage"
            scores.append(MetricScore("Debt/Equity", de, sc, 1.1, interp, "Financial Health"))

        cr = d.get("current_ratio")
        if cr is not None:
            if cr > 2.5:
                sc, interp = 0.5, "Very strong liquidity"
            elif cr > 1.5:
                sc, interp = 0.3, "Healthy liquidity"
            elif cr > 1.0:
                sc, interp = 0.0, "Adequate liquidity"
            elif cr > 0.7:
                sc, interp = -0.4, "Tight liquidity"
            else:
                sc, interp = -0.7, "Liquidity concern"
            scores.append(MetricScore("Current Ratio", cr, sc, 0.8, interp, "Financial Health"))

        fcf = d.get("free_cashflow")
        ocf = d.get("operating_cashflow")
        mc = d.get("market_cap")
        if fcf is not None and mc and mc > 0:
            fcf_yield = fcf / mc
            if fcf_yield > 0.08:
                sc, interp = 0.7, "Excellent FCF generation"
            elif fcf_yield > 0.04:
                sc, interp = 0.3, "Good FCF yield"
            elif fcf_yield > 0.01:
                sc, interp = 0.0, "Moderate FCF yield"
            elif fcf_yield > 0:
                sc, interp = -0.2, "Low FCF yield"
            else:
                sc, interp = -0.6, "Negative free cash flow"
            scores.append(MetricScore("FCF Yield", fcf_yield, sc, 1.0, interp, "Financial Health"))

        return scores

    def _score_dividends(self, d: dict) -> list[MetricScore]:
        scores = []

        dy = d.get("dividend_yield")
        if dy is not None and dy > 0:
            dy_pct = dy * 100 if dy < 1 else dy
            if dy_pct > 5:
                sc, interp = 0.3, "High yield (check sustainability)"
            elif dy_pct > 2.5:
                sc, interp = 0.5, "Attractive dividend yield"
            elif dy_pct > 1.0:
                sc, interp = 0.3, "Moderate dividend yield"
            else:
                sc, interp = 0.1, "Small dividend yield"
            scores.append(MetricScore("Dividend Yield", dy, sc, 0.6, interp, "Dividends"))

            pr = d.get("payout_ratio")
            if pr is not None:
                if pr < 0.4:
                    sc, interp = 0.5, "Sustainable payout with room to grow"
                elif pr < 0.6:
                    sc, interp = 0.3, "Healthy payout ratio"
                elif pr < 0.8:
                    sc, interp = 0.0, "Elevated payout ratio"
                else:
                    sc, interp = -0.5, "Payout may not be sustainable"
                scores.append(MetricScore("Payout Ratio", pr, sc, 0.4, interp, "Dividends"))

        return scores

    def _score_analyst_sentiment(self, d: dict) -> list[MetricScore]:
        scores = []

        price = d.get("price", 0)
        target = d.get("target_mean_price")
        if target and price and price > 0:
            upside = (target - price) / price
            if upside > 0.3:
                sc, interp = 0.7, f"Analysts see {upside:.0%} upside"
            elif upside > 0.1:
                sc, interp = 0.3, f"Analysts see {upside:.0%} upside"
            elif upside > -0.05:
                sc, interp = 0.0, "Near analyst consensus target"
            elif upside > -0.15:
                sc, interp = -0.3, f"Analysts see {abs(upside):.0%} downside"
            else:
                sc, interp = -0.6, f"Analysts see {abs(upside):.0%} downside"
            scores.append(MetricScore("Analyst Target", target, sc, 0.7, interp, "Analyst Sentiment"))

        rec = d.get("recommendation_key")
        if rec:
            rec_map = {
                "strong_buy": (0.8, "Analyst consensus: Strong Buy"),
                "buy": (0.4, "Analyst consensus: Buy"),
                "hold": (0.0, "Analyst consensus: Hold"),
                "sell": (-0.5, "Analyst consensus: Sell"),
                "strong_sell": (-0.8, "Analyst consensus: Strong Sell"),
            }
            sc, interp = rec_map.get(rec, (0.0, f"Analyst consensus: {rec}"))
            scores.append(MetricScore("Analyst Consensus", None, sc, 0.5, interp, "Analyst Sentiment"))
            scores[-1].value = rec

        return scores
