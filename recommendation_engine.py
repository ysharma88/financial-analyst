"""Holistic recommendation engine combining fundamental and technical analysis."""

from dataclasses import dataclass, field
from fundamental_analysis import FundamentalResult
from technical_analysis import TechnicalResult


@dataclass
class HolisticRecommendation:
    fundamental: FundamentalResult
    technical: TechnicalResult

    fundamental_weight: float = 0.50
    technical_weight: float = 0.50

    overall_score: float = 0.0
    recommendation: str = "HOLD"
    confidence: str = "LOW"
    summary: str = ""
    key_factors: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)

    def __post_init__(self):
        self._compute()

    def _compute(self):
        f_score = self.fundamental.overall_score
        t_score = self.technical.overall_score

        self.overall_score = (
            f_score * self.fundamental_weight +
            t_score * self.technical_weight
        )

        if self.overall_score >= 0.5:
            self.recommendation = "STRONG BUY"
        elif self.overall_score >= 0.2:
            self.recommendation = "BUY"
        elif self.overall_score >= -0.2:
            self.recommendation = "HOLD"
        elif self.overall_score >= -0.5:
            self.recommendation = "SELL"
        else:
            self.recommendation = "STRONG SELL"

        alignment = f_score * t_score
        abs_score = abs(self.overall_score)
        if alignment > 0 and abs_score > 0.3:
            self.confidence = "HIGH"
        elif alignment > 0 or abs_score > 0.2:
            self.confidence = "MEDIUM"
        else:
            self.confidence = "LOW"

        self.key_factors = self._extract_key_factors()
        self.risks = self._extract_risks()
        self.summary = self._build_summary()

    def _extract_key_factors(self) -> list[str]:
        factors = []

        all_signals = []
        for s in self.fundamental.scores:
            if s.value is not None:
                all_signals.append((abs(s.score * s.weight), s.interpretation, s.score > 0))
        for s in self.technical.signals:
            if s.value is not None:
                all_signals.append((abs(s.score * s.weight), s.interpretation, s.score > 0))

        all_signals.sort(reverse=True)

        positive = [interp for _, interp, is_pos in all_signals if is_pos][:4]
        negative = [interp for _, interp, is_pos in all_signals if not is_pos][:4]

        if positive:
            factors.append("**Positive drivers:** " + "; ".join(positive))
        if negative:
            factors.append("**Negative drivers:** " + "; ".join(negative))

        return factors

    def _extract_risks(self) -> list[str]:
        risks = []

        f_cats = self.fundamental.category_scores
        t_cats = self.technical.category_scores

        if f_cats.get("Valuation", 0) < -0.3:
            risks.append("Stock appears overvalued on fundamental metrics")
        if f_cats.get("Financial Health", 0) < -0.3:
            risks.append("Balance sheet shows elevated risk (high debt or low liquidity)")
        if f_cats.get("Growth", 0) < -0.3:
            risks.append("Growth metrics are declining")

        if t_cats.get("Momentum", 0) < -0.3:
            risks.append("Momentum indicators signal weakness")
        if t_cats.get("Trend", 0) < -0.3:
            risks.append("Price is in a downtrend across key moving averages")
        if t_cats.get("Volume", 0) < -0.3:
            risks.append("Volume patterns suggest distribution / selling pressure")

        f_score = self.fundamental.overall_score
        t_score = self.technical.overall_score
        if f_score * t_score < 0:
            if f_score > t_score:
                risks.append("Fundamentals are positive but technicals are weak - timing risk")
            else:
                risks.append("Technicals are positive but fundamentals are weak - potential value trap")

        return risks

    def _build_summary(self) -> str:
        parts = [
            f"Overall recommendation: **{self.recommendation}** (confidence: {self.confidence})",
            f"Combined score: {self.overall_score:+.2f} "
            f"(Fundamental: {self.fundamental.overall_score:+.2f}, "
            f"Technical: {self.technical.overall_score:+.2f})",
        ]

        if self.fundamental.signal == self.technical.signal:
            parts.append(
                f"Both fundamental and technical analysis agree: {self.fundamental.signal}"
            )
        else:
            parts.append(
                f"Fundamental signal: {self.fundamental.signal} | "
                f"Technical signal: {self.technical.signal}"
            )

        return "\n".join(parts)

    @property
    def score_color(self) -> str:
        if self.overall_score >= 0.3:
            return "#00C853"
        elif self.overall_score >= 0.1:
            return "#69F0AE"
        elif self.overall_score >= -0.1:
            return "#FFD54F"
        elif self.overall_score >= -0.3:
            return "#FF8A65"
        else:
            return "#FF1744"

    @property
    def recommendation_emoji(self) -> str:
        return {
            "STRONG BUY": "🟢",
            "BUY": "🟩",
            "HOLD": "🟡",
            "SELL": "🟧",
            "STRONG SELL": "🔴",
        }.get(self.recommendation, "⚪")
