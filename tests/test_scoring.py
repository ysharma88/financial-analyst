"""Pytest suite for scoring and weighting logic.

Tests cover:
- FundamentalResult.compute_overall() weighted averaging
- FundamentalAnalyzer sector-aware P/E scoring
- HolisticRecommendation score, label, and confidence computation
- ReasoningEngine PILLAR_WEIGHTS sum to 1.0
"""

from __future__ import annotations

import pytest

from fundamental_analysis import (
    FundamentalAnalyzer,
    FundamentalResult,
    MetricScore,
)
from recommendation_engine import HolisticRecommendation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_metric(name: str, score: float, weight: float, value=1.0, category: str = "Test") -> MetricScore:
    return MetricScore(name=name, value=value, score=score, weight=weight, interpretation="", category=category)


def _make_fundamental_result(scores: list[MetricScore]) -> FundamentalResult:
    r = FundamentalResult(scores=scores)
    r.compute_overall()
    return r


class _FakeFundamental:
    """Minimal FundamentalResult duck-type for HolisticRecommendation."""
    def __init__(self, overall_score: float):
        self.overall_score = overall_score
        self.signal = "NEUTRAL"
        self.scores = []
        self.category_scores = {}


class _FakeTechnical:
    """Minimal TechnicalResult duck-type for HolisticRecommendation."""
    def __init__(self, overall_score: float):
        self.overall_score = overall_score
        self.signal = "NEUTRAL"
        self.signals = []
        self.category_scores = {}


# ---------------------------------------------------------------------------
# FundamentalResult.compute_overall()
# ---------------------------------------------------------------------------

class TestComputeOverall:

    def test_single_metric_score_propagates(self):
        r = _make_fundamental_result([_make_metric("P/E", score=0.8, weight=1.0)])
        assert r.overall_score == pytest.approx(0.8)

    def test_weighted_average_two_metrics(self):
        scores = [
            _make_metric("A", score=1.0, weight=2.0),
            _make_metric("B", score=0.0, weight=2.0),
        ]
        r = _make_fundamental_result(scores)
        assert r.overall_score == pytest.approx(0.5)

    def test_weighted_average_unequal_weights(self):
        scores = [
            _make_metric("A", score=1.0, weight=3.0),
            _make_metric("B", score=-1.0, weight=1.0),
        ]
        r = _make_fundamental_result(scores)
        # (1.0*3 + -1.0*1) / 4 = 0.5
        assert r.overall_score == pytest.approx(0.5)

    def test_none_value_metrics_excluded(self):
        scores = [
            MetricScore("A", value=None, score=1.0, weight=5.0, interpretation="", category="X"),
            _make_metric("B", score=0.2, weight=1.0),
        ]
        r = _make_fundamental_result(scores)
        assert r.overall_score == pytest.approx(0.2)

    def test_all_none_values_gives_neutral(self):
        scores = [
            MetricScore("A", value=None, score=0.9, weight=1.0, interpretation="", category="X"),
        ]
        r = _make_fundamental_result(scores)
        assert r.overall_score == 0.0
        assert r.signal == "NEUTRAL"

    def test_signal_strong_buy_threshold(self):
        r = _make_fundamental_result([_make_metric("X", score=0.6, weight=1.0)])
        assert r.signal == "STRONG BUY"

    def test_signal_buy_threshold(self):
        r = _make_fundamental_result([_make_metric("X", score=0.3, weight=1.0)])
        assert r.signal == "BUY"

    def test_signal_hold_threshold(self):
        r = _make_fundamental_result([_make_metric("X", score=0.0, weight=1.0)])
        assert r.signal == "HOLD"

    def test_signal_sell_threshold(self):
        r = _make_fundamental_result([_make_metric("X", score=-0.3, weight=1.0)])
        assert r.signal == "SELL"

    def test_signal_strong_sell_threshold(self):
        r = _make_fundamental_result([_make_metric("X", score=-0.6, weight=1.0)])
        assert r.signal == "STRONG SELL"

    def test_category_scores_grouped_correctly(self):
        scores = [
            _make_metric("P/E", score=0.8, weight=1.0, category="Valuation"),
            _make_metric("P/B", score=0.4, weight=1.0, category="Valuation"),
            _make_metric("ROE", score=0.6, weight=1.0, category="Profitability"),
        ]
        r = _make_fundamental_result(scores)
        cats = r.category_scores
        assert cats["Valuation"] == pytest.approx(0.6)
        assert cats["Profitability"] == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# FundamentalAnalyzer sector-aware P/E scoring
# ---------------------------------------------------------------------------

class TestPEScoring:

    def setup_method(self):
        self.analyzer = FundamentalAnalyzer()

    def _pe_score(self, pe: float, sector: str) -> float:
        data = {"pe_trailing": pe}
        result = self.analyzer.analyze(data, sector=sector)
        pe_metric = next((s for s in result.scores if s.name == "P/E Ratio"), None)
        assert pe_metric is not None, "P/E Ratio metric not found"
        return pe_metric.score

    def test_tech_pe_significantly_undervalued(self):
        # Tech benchmark = 30; 30*0.6 = 18 → PE 15 should be < 18
        assert self._pe_score(15.0, "Technology") == pytest.approx(0.8)

    def test_tech_pe_fairly_valued(self):
        # 30*0.85=25.5 to 30*1.15=34.5 → PE 30 is in range
        assert self._pe_score(30.0, "Technology") == pytest.approx(0.0)

    def test_tech_pe_significantly_overvalued(self):
        # > 30*1.5 = 45
        assert self._pe_score(50.0, "Technology") == pytest.approx(-0.8)

    def test_energy_pe_significantly_undervalued(self):
        # Energy benchmark = 12; 12*0.6 = 7.2 → PE 5 is < 7.2
        assert self._pe_score(5.0, "Energy") == pytest.approx(0.8)

    def test_energy_pe_significantly_overvalued(self):
        # > 12*1.5 = 18
        assert self._pe_score(25.0, "Energy") == pytest.approx(-0.8)

    def test_negative_pe_excluded(self):
        data = {"pe_trailing": -10.0}
        result = self.analyzer.analyze(data)
        pe_metric = next((s for s in result.scores if s.name == "P/E Ratio"), None)
        assert pe_metric is None  # negative P/E should be skipped

    def test_zero_pe_excluded(self):
        data = {"pe_trailing": 0.0}
        result = self.analyzer.analyze(data)
        pe_metric = next((s for s in result.scores if s.name == "P/E Ratio"), None)
        assert pe_metric is None


# ---------------------------------------------------------------------------
# HolisticRecommendation scoring and confidence
# ---------------------------------------------------------------------------

class TestHolisticRecommendation:

    def _make(self, f_score: float, t_score: float, f_weight: float = 0.5, t_weight: float = 0.5):
        return HolisticRecommendation(
            fundamental=_FakeFundamental(f_score),
            technical=_FakeTechnical(t_score),
            fundamental_weight=f_weight,
            technical_weight=t_weight,
        )

    def test_equal_weights_average(self):
        rec = self._make(0.6, 0.4)
        assert rec.overall_score == pytest.approx(0.5)

    def test_custom_weights(self):
        rec = self._make(0.8, 0.2, f_weight=0.7, t_weight=0.3)
        expected = 0.8 * 0.7 + 0.2 * 0.3
        assert rec.overall_score == pytest.approx(expected)

    def test_strong_buy_label(self):
        rec = self._make(0.6, 0.6)
        assert rec.recommendation == "STRONG BUY"

    def test_buy_label(self):
        rec = self._make(0.3, 0.2)
        assert rec.recommendation == "BUY"

    def test_hold_label(self):
        rec = self._make(0.0, 0.0)
        assert rec.recommendation == "HOLD"

    def test_sell_label(self):
        rec = self._make(-0.3, -0.2)
        assert rec.recommendation == "SELL"

    def test_strong_sell_label(self):
        rec = self._make(-0.6, -0.6)
        assert rec.recommendation == "STRONG SELL"

    def test_high_confidence_when_aligned_and_strong(self):
        rec = self._make(0.5, 0.5)
        assert rec.confidence == "HIGH"

    def test_low_confidence_when_opposed(self):
        # f_score * t_score < 0 (product negative) → alignment < 0, abs_score < 0.2
        rec = self._make(0.1, -0.1)
        assert rec.confidence == "LOW"

    def test_medium_confidence_aligned_weak(self):
        # alignment > 0 but abs_score <= 0.3
        rec = self._make(0.15, 0.15)
        assert rec.confidence in ("MEDIUM", "LOW")

    def test_score_color_green_for_positive(self):
        rec = self._make(0.4, 0.4)
        assert rec.score_color in ("#00C853", "#69F0AE")

    def test_score_color_red_for_negative(self):
        rec = self._make(-0.4, -0.4)
        assert rec.score_color in ("#FF8A65", "#FF1744")


# ---------------------------------------------------------------------------
# ReasoningEngine pillar weights
# ---------------------------------------------------------------------------

class TestReasoningEnginePillarWeights:

    def test_weights_sum_to_one(self):
        from reasoning_engine import ReasoningEngine
        total = sum(ReasoningEngine.PILLAR_WEIGHTS.values())
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_all_seven_pillars_present(self):
        from reasoning_engine import ReasoningEngine
        expected = {"Governance", "Macro", "Sector", "Fundamental", "Technical", "Sentiment", "Risk"}
        assert set(ReasoningEngine.PILLAR_WEIGHTS.keys()) == expected

    def test_fundamental_has_highest_weight(self):
        from reasoning_engine import ReasoningEngine
        w = ReasoningEngine.PILLAR_WEIGHTS
        assert w["Fundamental"] == max(w.values())
