"""Multi-factor reasoning engine.

Synthesises every analysis module into one coherent, narrative-driven
investment verdict. The engine does NOT simply average scores — it applies
a decision-tree style reasoning chain:

1. Governance gate  — severe red flags can override everything else.
2. Macro context    — positions the recommendation within the cycle.
3. Sector alignment — checks if the sector has tailwinds or headwinds.
4. Fundamental case — valuation, profitability, capital efficiency.
5. Technical timing — momentum, trend, volume confirmation.
6. Sentiment check  — news tone and analyst consensus.
7. Risk calibration — volatility, drawdown, position sizing guidance.

The output is a structured verdict with a narrative "reasoning chain"
that explains *why* in plain English.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Optional, List

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data-classes for the verdict
# ---------------------------------------------------------------------------
@dataclass
class ReasoningStep:
    pillar: str         # e.g. "Governance", "Macro", "Fundamental"
    signal: str         # BULLISH / BEARISH / NEUTRAL / CAUTION
    weight: float       # 0-1 contribution
    headline: str       # one-liner
    detail: str         # 2-3 sentence explanation
    score: float        # -1 to +1


@dataclass
class InvestmentVerdict:
    action: str = "HOLD"                    # STRONG BUY / BUY / HOLD / SELL / STRONG SELL
    confidence: str = "LOW"                 # LOW / MEDIUM / HIGH
    composite_score: float = 0.0            # -1 to +1
    reasoning_chain: List[ReasoningStep] = field(default_factory=list)
    thesis: str = ""                        # 3-5 sentence investment thesis
    bull_case: str = ""
    bear_case: str = ""
    catalysts: List[str] = field(default_factory=list)
    key_risks: List[str] = field(default_factory=list)
    action_plan: str = ""                   # practical next steps
    score_color: str = "#FFD54F"


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class ReasoningEngine:
    """Builds a multi-factor investment verdict with transparent reasoning."""

    PILLAR_WEIGHTS = {
        "Governance":  0.12,
        "Macro":       0.10,
        "Sector":      0.10,
        "Fundamental": 0.25,
        "Technical":   0.18,
        "Sentiment":   0.10,
        "Risk":        0.15,
    }

    def synthesize(
        self,
        company_info: dict,
        fund_data: dict,
        fund_result,                # FundamentalResult
        tech_result,                # TechnicalResult
        recommendation,             # HolisticRecommendation
        governance=None,            # GovernanceResult
        macro_result=None,          # MacroResult
        sector_result=None,         # SectorResult
        news_result=None,           # NewsSentimentResult
        risk_profile=None,          # RiskProfile
        sr_levels=None,             # list[SupportResistanceLevel]
    ) -> InvestmentVerdict:
        verdict = InvestmentVerdict()
        steps: List[ReasoningStep] = []

        # --- 1. GOVERNANCE GATE ---
        gov_step = self._reason_governance(governance)
        steps.append(gov_step)

        governance_veto = gov_step.score <= -0.7

        # --- 2. MACRO ---
        macro_step = self._reason_macro(macro_result)
        steps.append(macro_step)

        # --- 3. SECTOR ---
        sector_step = self._reason_sector(sector_result, company_info)
        steps.append(sector_step)

        # --- 4. FUNDAMENTAL ---
        fund_step = self._reason_fundamental(fund_result, fund_data)
        steps.append(fund_step)

        # --- 5. TECHNICAL ---
        tech_step = self._reason_technical(tech_result)
        steps.append(tech_step)

        # --- 6. SENTIMENT ---
        sent_step = self._reason_sentiment(news_result, fund_data)
        steps.append(sent_step)

        # --- 7. RISK ---
        risk_step = self._reason_risk(risk_profile, fund_data, governance)
        steps.append(risk_step)

        verdict.reasoning_chain = steps

        # --- COMPOSITE SCORE ---
        total_weight = 0.0
        weighted_sum = 0.0
        for step in steps:
            w = self.PILLAR_WEIGHTS.get(step.pillar, 0.10)
            weighted_sum += step.score * w
            total_weight += w

        raw_score = weighted_sum / total_weight if total_weight > 0 else 0
        verdict.composite_score = round(max(-1, min(1, raw_score)), 3)

        # Governance veto: cap at HOLD
        if governance_veto and verdict.composite_score > 0:
            verdict.composite_score = min(verdict.composite_score, -0.05)

        # --- ACTION ---
        s = verdict.composite_score
        if s >= 0.45:
            verdict.action = "STRONG BUY"
        elif s >= 0.18:
            verdict.action = "BUY"
        elif s >= -0.18:
            verdict.action = "HOLD"
        elif s >= -0.45:
            verdict.action = "SELL"
        else:
            verdict.action = "STRONG SELL"

        if governance_veto:
            if verdict.action in ("STRONG BUY", "BUY"):
                verdict.action = "HOLD"

        # --- CONFIDENCE ---
        signal_agreement = sum(1 for st in steps if st.score > 0.1) - sum(1 for st in steps if st.score < -0.1)
        abs_score = abs(verdict.composite_score)
        if abs(signal_agreement) >= 4 and abs_score > 0.25:
            verdict.confidence = "HIGH"
        elif abs(signal_agreement) >= 2 or abs_score > 0.15:
            verdict.confidence = "MEDIUM"
        else:
            verdict.confidence = "LOW"

        # --- COLOR ---
        if s >= 0.3:
            verdict.score_color = "#00C853"
        elif s >= 0.1:
            verdict.score_color = "#69F0AE"
        elif s >= -0.1:
            verdict.score_color = "#FFD54F"
        elif s >= -0.3:
            verdict.score_color = "#FF8A65"
        else:
            verdict.score_color = "#FF1744"

        # --- NARRATIVES ---
        name = company_info.get("name", "This company")
        ticker = company_info.get("ticker", "")
        sector = company_info.get("sector", "N/A")
        price = company_info.get("price", 0)

        claude_narratives = self._build_narratives_with_claude(verdict, steps, name, sector, price)
        if claude_narratives:
            verdict.thesis = claude_narratives.get("thesis") or self._build_thesis(verdict, steps, name, sector, price)
            verdict.bull_case = claude_narratives.get("bull_case") or self._build_bull_case(steps, name)
            verdict.bear_case = claude_narratives.get("bear_case") or self._build_bear_case(steps, name)
        else:
            verdict.thesis = self._build_thesis(verdict, steps, name, sector, price)
            verdict.bull_case = self._build_bull_case(steps, name)
            verdict.bear_case = self._build_bear_case(steps, name)
        verdict.catalysts = self._extract_catalysts(steps, news_result, macro_result)
        verdict.key_risks = self._extract_risks(steps, governance, risk_profile)
        # Use S/R levels from tech result if not passed directly
        if sr_levels is None and hasattr(tech_result, 'support_resistance'):
            sr_levels = tech_result.support_resistance

        verdict.action_plan = self._build_action_plan(verdict, price, risk_profile, fund_data, sr_levels)

        return verdict

    # ------------------------------------------------------------------
    # Pillar reasoning
    # ------------------------------------------------------------------
    def _reason_governance(self, gov) -> ReasoningStep:
        if gov is None:
            return ReasoningStep("Governance", "NEUTRAL", 0.12,
                                 "No governance data available",
                                 "Governance analysis could not be performed. Proceed with caution.",
                                 0.0)

        score_map = {"OK": 0.5, "LOW": 0.2, "MEDIUM": -0.1, "HIGH": -0.5, "CRITICAL": -0.9}
        score = score_map.get(gov.overall_risk_level, 0.0)

        critical_flags = [f for f in gov.red_flags if f.severity == "CRITICAL"]
        high_flags = [f for f in gov.red_flags if f.severity == "HIGH"]

        if critical_flags:
            signal = "BEARISH"
            headline = f"{len(critical_flags)} critical governance red flag(s) detected"
            issues = "; ".join(f.title for f in critical_flags[:3])
            detail = (f"Critical issues: {issues}. "
                      f"Governance risk score is {gov.risk_score:.0f}/100. "
                      f"These issues can override positive financials and should be investigated before investing.")
        elif high_flags:
            signal = "CAUTION"
            headline = f"{len(high_flags)} elevated governance concern(s)"
            issues = "; ".join(f.title for f in high_flags[:3])
            detail = (f"Elevated concerns: {issues}. "
                      f"While not disqualifying, these warrant deeper due diligence.")
        elif score >= 0.3:
            signal = "BULLISH"
            headline = "Clean governance profile"
            detail = "No material red flags. Management appears well-aligned with shareholders."
        else:
            signal = "NEUTRAL"
            headline = "Governance is acceptable"
            detail = "Minor concerns only. Standard corporate governance."

        # Earnings track bonus/penalty
        if gov.earnings_track:
            miss_count = sum(1 for e in gov.earnings_track if not e.beat)
            total = len(gov.earnings_track)
            beat_rate = (total - miss_count) / total
            if beat_rate >= 0.75:
                score += 0.15
                detail += f" Management has beaten estimates {total - miss_count}/{total} quarters — reliable execution."
            elif beat_rate < 0.5 and total >= 3:
                score -= 0.2
                detail += f" Management missed estimates {miss_count}/{total} quarters — unreliable guidance."

        # Remuneration
        if gov.mgmt_pay_pct is not None and gov.mgmt_pay_pct > 11:
            score -= 0.15
            detail += f" Executive pay at {gov.mgmt_pay_pct:.1f}% of net income exceeds the 11% threshold."

        score = max(-1, min(1, score))
        return ReasoningStep("Governance", signal, 0.12, headline, detail, round(score, 2))

    def _reason_macro(self, macro) -> ReasoningStep:
        if macro is None:
            return ReasoningStep("Macro", "NEUTRAL", 0.10,
                                 "Macro data unavailable",
                                 "Macroeconomic analysis could not be performed.",
                                 0.0)

        score = macro.overall_score / 100 if abs(macro.overall_score) > 1 else macro.overall_score

        cycle = macro.cycle
        if cycle:
            phase = cycle.phase
            if phase in ("Expansion", "Recovery"):
                signal = "BULLISH"
                headline = f"Business cycle in {phase} phase — growth-friendly"
                detail = (f"{cycle.description} "
                          f"Risk posture: {cycle.risk_posture}. "
                          f"Favored sectors: {', '.join(cycle.favored_sectors[:3])}.")
            elif phase == "Peak":
                signal = "CAUTION"
                headline = "Business cycle near Peak — late-cycle positioning needed"
                detail = (f"{cycle.description} "
                          f"Consider defensive tilts. Sectors to avoid: {', '.join(cycle.avoid_sectors[:3])}.")
            else:
                signal = "BEARISH"
                headline = f"Business cycle in {phase} phase — defensive posture"
                detail = (f"{cycle.description} "
                          f"Favor defensives: {', '.join(cycle.favored_sectors[:3])}.")
        else:
            signal = "NEUTRAL"
            headline = "Macro environment is mixed"
            detail = macro.summary or "Signals are conflicting."

        return ReasoningStep("Macro", signal, 0.10, headline, detail, round(max(-1, min(1, score)), 2))

    def _reason_sector(self, sector, company_info) -> ReasoningStep:
        if sector is None:
            return ReasoningStep("Sector", "NEUTRAL", 0.10,
                                 "Sector analysis unavailable",
                                 "Sector rotation analysis could not be performed.",
                                 0.0)

        stock_sector = company_info.get("sector", "")
        rank = sector.stock_sector_rank
        total = len(sector.sectors) if sector.sectors else 11
        sector_score_val = sector.stock_sector_score

        if rank is not None and rank <= 3:
            signal = "BULLISH"
            score = 0.6
            headline = f"{stock_sector} ranks #{rank}/{total} in sector rotation"
            detail = f"The stock's sector is among the top performers in the current cycle, providing strong tailwinds."
        elif rank is not None and rank <= 6:
            signal = "NEUTRAL"
            score = 0.1
            headline = f"{stock_sector} ranks #{rank}/{total} — mid-pack"
            detail = "Sector positioning is neither a clear tailwind nor headwind."
        elif rank is not None:
            signal = "BEARISH"
            score = -0.4
            headline = f"{stock_sector} ranks #{rank}/{total} — facing headwinds"
            detail = "The sector is underperforming in the current rotation, creating a drag."
        else:
            signal = "NEUTRAL"
            score = 0.0
            headline = "Sector ranking not available"
            detail = sector.rotation_recommendation or "Unable to determine sector positioning."

        return ReasoningStep("Sector", signal, 0.10, headline, detail, round(score, 2))

    def _reason_fundamental(self, fund_result, fund_data) -> ReasoningStep:
        score = fund_result.overall_score
        cats = fund_result.category_scores

        parts = []
        valuation = cats.get("Valuation", 0)
        profitability = cats.get("Profitability", 0)
        growth = cats.get("Growth", 0)
        health = cats.get("Financial Health", 0)
        cap_eff = cats.get("Capital Efficiency", 0)

        # Valuation
        if valuation > 0.3:
            parts.append("attractively valued")
        elif valuation < -0.3:
            parts.append("appears overvalued")

        # Profitability
        if profitability > 0.3:
            parts.append("highly profitable with strong margins")
        elif profitability < -0.3:
            parts.append("weak profitability")

        # Capital efficiency
        roic = fund_data.get("roic")
        wacc = fund_data.get("wacc")
        if roic and wacc:
            spread = (roic - wacc) * 100
            if spread > 10:
                parts.append(f"exceptional value creation (ROIC-WACC spread of {spread:.0f}%)")
            elif spread > 0:
                parts.append(f"positive value creation (ROIC exceeds WACC by {spread:.0f}%)")
            else:
                parts.append(f"destroying value (ROIC {spread:.0f}% below WACC)")

        # Growth
        if growth > 0.3:
            parts.append("strong growth trajectory")
        elif growth < -0.3:
            parts.append("declining growth")

        # Health
        if health < -0.3:
            parts.append("balance sheet stress with high leverage")

        if score >= 0.3:
            signal = "BULLISH"
            headline = "Strong fundamental case"
        elif score >= 0.0:
            signal = "NEUTRAL"
            headline = "Fundamentals are mixed but acceptable"
        elif score >= -0.3:
            signal = "CAUTION"
            headline = "Fundamental concerns present"
        else:
            signal = "BEARISH"
            headline = "Weak fundamentals"

        detail = f"The company is {', '.join(parts)}." if parts else fund_result.summary
        detail += f" Fundamental score: {score:+.2f}."

        return ReasoningStep("Fundamental", signal, 0.25, headline, detail, round(score, 2))

    def _reason_technical(self, tech_result) -> ReasoningStep:
        score = tech_result.overall_score
        cats = tech_result.category_scores

        trend = cats.get("Trend", 0)
        momentum = cats.get("Momentum", 0)
        volume = cats.get("Volume", 0)

        parts = []
        if trend > 0.3:
            parts.append("price is in a strong uptrend")
        elif trend < -0.3:
            parts.append("price is in a downtrend")

        if momentum > 0.3:
            parts.append("momentum is bullish")
        elif momentum < -0.3:
            parts.append("momentum is fading")

        if volume > 0.3:
            parts.append("volume confirms the move")
        elif volume < -0.3:
            parts.append("volume signals distribution")

        if score >= 0.3:
            signal = "BULLISH"
            headline = "Technicals confirm bullish setup"
        elif score >= 0.0:
            signal = "NEUTRAL"
            headline = "Technical picture is mixed"
        elif score >= -0.3:
            signal = "CAUTION"
            headline = "Technical weakness developing"
        else:
            signal = "BEARISH"
            headline = "Bearish technical structure"

        detail = "; ".join(parts).capitalize() + "." if parts else tech_result.summary
        detail += f" Technical score: {score:+.2f}."

        return ReasoningStep("Technical", signal, 0.18, headline, detail, round(score, 2))

    def _reason_sentiment(self, news, fund_data) -> ReasoningStep:
        parts = []
        score = 0.0

        if news and news.articles:
            news_score = news.overall_sentiment
            score += news_score * 0.6

            if news.bullish_count > news.bearish_count * 2:
                parts.append(f"news flow is decidedly positive ({news.bullish_count} bullish vs {news.bearish_count} bearish)")
            elif news.bearish_count > news.bullish_count * 2:
                parts.append(f"news flow is negative ({news.bearish_count} bearish vs {news.bullish_count} bullish)")
            else:
                parts.append(f"news flow is balanced ({news.bullish_count} bullish, {news.bearish_count} bearish)")

            if news.regulatory_count > 0:
                parts.append(f"{news.regulatory_count} regulatory headline(s) to monitor")

        # Analyst consensus
        rec_key = fund_data.get("recommendation_key", "")
        if rec_key:
            analyst_map = {"strong_buy": 0.5, "buy": 0.3, "hold": 0.0, "sell": -0.3, "strong_sell": -0.5}
            score += analyst_map.get(rec_key, 0) * 0.4
            parts.append(f"analyst consensus is {rec_key.replace('_', ' ')}")

        target = fund_data.get("target_mean_price")
        price = fund_data.get("price", 0)
        if target and price and price > 0:
            upside = (target - price) / price
            parts.append(f"mean price target implies {upside:+.0%}")

        if score > 0.2:
            signal = "BULLISH"
            headline = "Positive sentiment backdrop"
        elif score > 0.0:
            signal = "NEUTRAL"
            headline = "Sentiment is modestly positive"
        elif score > -0.2:
            signal = "NEUTRAL"
            headline = "Sentiment is mixed"
        else:
            signal = "BEARISH"
            headline = "Negative sentiment overhang"

        detail = "; ".join(parts).capitalize() + "." if parts else "No sentiment data available."
        return ReasoningStep("Sentiment", signal, 0.10, headline, detail, round(max(-1, min(1, score)), 2))

    def _reason_risk(self, risk_profile, fund_data, governance) -> ReasoningStep:
        score = 0.0
        parts = []

        if risk_profile:
            # Drawdown
            md = risk_profile.max_drawdown
            if md is not None:
                if md > -0.15:
                    score += 0.2
                    parts.append(f"max drawdown of {md:.0%} is manageable")
                elif md > -0.30:
                    parts.append(f"max drawdown of {md:.0%} is moderate")
                else:
                    score -= 0.3
                    parts.append(f"max drawdown of {md:.0%} signals high volatility")

            # Sharpe
            sharpe = risk_profile.sharpe_approx
            if sharpe is not None:
                if sharpe > 1.0:
                    score += 0.3
                    parts.append(f"Sharpe of {sharpe:.2f} indicates excellent risk-adjusted returns")
                elif sharpe > 0.5:
                    score += 0.1
                    parts.append(f"Sharpe of {sharpe:.2f} is acceptable")
                elif sharpe > 0:
                    parts.append(f"Sharpe of {sharpe:.2f} is below average")
                else:
                    score -= 0.3
                    parts.append(f"negative Sharpe of {sharpe:.2f} — returns don't compensate for risk")

            # Volatility
            vol = risk_profile.volatility_metrics.get("annualized_volatility")
            if vol:
                if vol < 0.20:
                    parts.append(f"low annualized volatility ({vol:.0%})")
                elif vol < 0.35:
                    parts.append(f"moderate volatility ({vol:.0%})")
                else:
                    score -= 0.15
                    parts.append(f"high volatility ({vol:.0%})")

        beta = fund_data.get("beta")
        if beta:
            if beta > 1.5:
                score -= 0.1
                parts.append(f"high beta of {beta:.2f} amplifies market moves")
            elif beta < 0.7:
                score += 0.1
                parts.append(f"low beta of {beta:.2f} provides some defensiveness")

        if score > 0.15:
            signal = "BULLISH"
            headline = "Favourable risk profile"
        elif score > -0.15:
            signal = "NEUTRAL"
            headline = "Risk profile is acceptable"
        else:
            signal = "BEARISH"
            headline = "Elevated risk profile"

        detail = "; ".join(parts).capitalize() + "." if parts else "Risk data unavailable."
        return ReasoningStep("Risk", signal, 0.15, headline, detail, round(max(-1, min(1, score)), 2))

    # ------------------------------------------------------------------
    # Narrative builders
    # ------------------------------------------------------------------

    _NARRATIVE_SYSTEM = (
        "You are a senior equity research analyst writing concise, insightful "
        "investment narratives. Use plain English, no fluff. Be specific — "
        "reference the pillar signals provided. Do not add disclaimers."
    )

    def _build_narratives_with_claude(
        self, verdict, steps, name: str, sector: str, price: float
    ) -> Optional[dict]:
        """Call Claude to generate thesis, bull case, and bear case in one shot.
        Returns dict with keys 'thesis', 'bull_case', 'bear_case', or None on failure.
        """
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return None
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)

            chain_text = "\n".join(
                f"- {s.pillar} [{s.signal}, score {s.score:+.2f}]: {s.headline}. {s.detail}"
                for s in steps
            )
            prompt = (
                f"Company: {name} | Sector: {sector} | Price: ${price:.2f}\n"
                f"Overall verdict: {verdict.action} (score {verdict.composite_score:+.2f}, confidence {verdict.confidence})\n\n"
                f"7-pillar reasoning chain:\n{chain_text}\n\n"
                "Write three sections in JSON with keys 'thesis', 'bull_case', 'bear_case'.\n"
                "'thesis': 3-5 sentence investment thesis explaining the overall verdict.\n"
                "'bull_case': 2-3 sentences describing the upside scenario.\n"
                "'bear_case': 2-3 sentences describing the key risks and downside scenario.\n"
                "Reply with only the JSON object."
            )

            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                system=[
                    {
                        "type": "text",
                        "text": self._NARRATIVE_SYSTEM,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```[a-z]*\n?", "", raw)
                raw = re.sub(r"\n?```$", "", raw)
            return json.loads(raw)
        except Exception as exc:
            logger.warning("Claude narrative generation failed: %s", exc)
            return None

    def _build_thesis(self, verdict, steps, name, sector, price) -> str:
        bullish = [s for s in steps if s.score > 0.15]
        bearish = [s for s in steps if s.score < -0.15]
        score = verdict.composite_score

        if score >= 0.3:
            opening = f"{name} presents a compelling investment case."
        elif score >= 0.1:
            opening = f"{name} offers a moderately attractive opportunity."
        elif score >= -0.1:
            opening = f"{name} is a hold — the risk/reward is balanced."
        elif score >= -0.3:
            opening = f"{name} faces headwinds that warrant caution."
        else:
            opening = f"{name} raises significant concerns across multiple dimensions."

        if bullish:
            bull_pillars = ", ".join(s.pillar for s in bullish[:3])
            support = f"Positive signals from {bull_pillars} support the case."
        else:
            support = "There are few clear positive catalysts at this time."

        if bearish:
            bear_pillars = ", ".join(s.pillar for s in bearish[:3])
            caution = f"However, {bear_pillars} {'raises' if len(bearish) == 1 else 'raise'} concerns."
        else:
            caution = "No major pillars are signalling clear risk."

        fund_step = next((s for s in steps if s.pillar == "Fundamental"), None)
        tech_step = next((s for s in steps if s.pillar == "Technical"), None)
        if fund_step and tech_step:
            if fund_step.score > 0.1 and tech_step.score > 0.1:
                alignment = "Fundamentals and technicals are aligned in a bullish direction, strengthening conviction."
            elif fund_step.score < -0.1 and tech_step.score < -0.1:
                alignment = "Both fundamentals and technicals are weak, reinforcing the bearish case."
            elif fund_step.score > 0.1 and tech_step.score < -0.1:
                alignment = "Fundamentals are positive but technicals have not confirmed — timing is premature."
            elif fund_step.score < -0.1 and tech_step.score > 0.1:
                alignment = "Technicals show strength but fundamentals are questionable — potential value trap."
            else:
                alignment = "Signals are mixed across fundamental and technical dimensions."
        else:
            alignment = ""

        parts = [opening, support, caution]
        if alignment:
            parts.append(alignment)
        return " ".join(parts)

    def _build_bull_case(self, steps, name) -> str:
        positives = sorted([s for s in steps if s.score > 0], key=lambda s: s.score, reverse=True)
        if not positives:
            return f"Limited bullish arguments for {name} at this time."
        points = [f"**{s.pillar}**: {s.headline}" for s in positives[:4]]
        return "If the bull case plays out: " + ". ".join(points) + "."

    def _build_bear_case(self, steps, name) -> str:
        negatives = sorted([s for s in steps if s.score < 0], key=lambda s: s.score)
        if not negatives:
            return f"Few bearish arguments against {name} at this time."
        points = [f"**{s.pillar}**: {s.headline}" for s in negatives[:4]]
        return "Key downside risks: " + ". ".join(points) + "."

    def _extract_catalysts(self, steps, news, macro) -> List[str]:
        catalysts = []

        fund = next((s for s in steps if s.pillar == "Fundamental"), None)
        if fund and fund.score > 0.2:
            catalysts.append("Earnings growth and margin expansion")
        if fund and "ROIC" in fund.detail and "value creation" in fund.detail.lower():
            catalysts.append("Superior capital allocation (ROIC > WACC)")

        if macro and macro.cycle:
            if macro.cycle.phase in ("Recovery", "Expansion"):
                catalysts.append(f"Macro tailwind — {macro.cycle.phase} phase favors growth")

        if news and news.regulatory_count > 0:
            catalysts.append("Regulatory developments (monitor closely)")

        tech = next((s for s in steps if s.pillar == "Technical"), None)
        if tech and tech.score > 0.2:
            catalysts.append("Technical breakout / momentum confirmation")

        sector = next((s for s in steps if s.pillar == "Sector"), None)
        if sector and sector.score > 0.3:
            catalysts.append("Sector rotation tailwinds")

        return catalysts[:6]

    def _extract_risks(self, steps, governance, risk_profile) -> List[str]:
        risks = []

        if governance:
            critical = [f for f in governance.red_flags if f.severity in ("CRITICAL", "HIGH")]
            for f in critical[:2]:
                risks.append(f"{f.title}: {f.detail[:80]}")

        for step in steps:
            if step.score < -0.2:
                risks.append(f"{step.pillar}: {step.headline}")

        if risk_profile and risk_profile.max_drawdown and risk_profile.max_drawdown < -0.25:
            risks.append(f"Historical max drawdown of {risk_profile.max_drawdown:.0%}")

        return risks[:6]

    def _build_action_plan(self, verdict, price, risk_profile, fund_data, sr_levels=None) -> str:
        action = verdict.action
        parts = []

        # Extract nearest support/resistance
        nearest_support = None
        nearest_resistance = None
        if sr_levels:
            supports = sorted([l for l in sr_levels if l.kind == "support" and l.price < price],
                              key=lambda l: l.price, reverse=True)
            resistances = sorted([l for l in sr_levels if l.kind == "resistance" and l.price > price],
                                 key=lambda l: l.price)
            if supports:
                nearest_support = supports[0]
            if resistances:
                nearest_resistance = resistances[0]

        if action in ("STRONG BUY", "BUY"):
            parts.append(f"**Entry**: Consider accumulating around ${price:.2f}." if price else "")

            if nearest_support:
                parts.append(
                    f"**Key support**: ${nearest_support.price:.2f} "
                    f"({nearest_support.strength}, {nearest_support.method}) — "
                    f"consider buying on pullbacks to this level."
                )

            if risk_profile and risk_profile.stop_losses:
                best_stop = min(risk_profile.stop_losses, key=lambda s: s.stop_price)
                parts.append(f"**Stop-loss**: Place at ${best_stop.stop_price:.2f} ({best_stop.distance_pct:.1f}% below entry, {best_stop.method}).")

            if nearest_resistance:
                parts.append(
                    f"**Near-term resistance**: ${nearest_resistance.price:.2f} "
                    f"({nearest_resistance.strength}) — watch for breakout above this level."
                )

            target = fund_data.get("target_mean_price")
            if target and price:
                parts.append(f"**Analyst target**: ${target:.2f} ({(target-price)/price:+.0%}).")

            if risk_profile and risk_profile.position_sizes:
                ps = risk_profile.position_sizes[0]
                parts.append(f"**Position size**: ~{ps.shares} shares ({ps.pct_of_portfolio:.1f}% of portfolio) using {ps.method}.")

            parts.append("**Timeframe**: Monitor for 3-6 months. Re-evaluate on earnings or macro shift.")

        elif action == "HOLD":
            parts.append("**No new position recommended** at current levels.")

            if nearest_support and nearest_resistance:
                parts.append(
                    f"**Trading range**: ${nearest_support.price:.2f} (support) → "
                    f"${nearest_resistance.price:.2f} (resistance). "
                    f"A break above resistance is bullish; a break below support is bearish."
                )
            elif nearest_support:
                parts.append(f"**Key support**: ${nearest_support.price:.2f} — a break below triggers re-evaluation.")
            elif nearest_resistance:
                parts.append(f"**Key resistance**: ${nearest_resistance.price:.2f} — a break above could turn bullish.")

            parts.append("If already long, hold with a trailing stop to protect gains.")
            parts.append("Wait for a clearer signal from fundamentals or technicals before acting.")

        else:  # SELL / STRONG SELL
            parts.append("**Reduce or exit** existing positions.")

            if nearest_support:
                parts.append(
                    f"**Key support**: ${nearest_support.price:.2f} — if this level breaks, "
                    f"expect accelerated downside."
                )

            if risk_profile and risk_profile.stop_losses:
                parts.append("If holding, tighten stops aggressively.")
            parts.append("Re-evaluate only if fundamentals materially improve or governance concerns are resolved.")

        return " ".join(p for p in parts if p)
