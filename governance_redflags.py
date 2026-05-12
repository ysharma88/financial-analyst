"""Governance & red-flag scanner.

Detects management quality issues:
- Pledged / concentrated insider shares
- Unrealistic forecasts (earnings misses vs estimates)
- Excessive management remuneration relative to net income
- ISS governance risk scores (audit, board, compensation, shareholder rights)
- Fraud / legal risk indicators
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Optional, List

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

SEVERITY_COLORS = {"CRITICAL": "#FF1744", "HIGH": "#FF5722", "MEDIUM": "#FFA726", "LOW": "#FFCA28", "OK": "#00C853"}


@dataclass
class RedFlag:
    category: str
    title: str
    severity: str         # CRITICAL, HIGH, MEDIUM, LOW, OK
    detail: str
    metric_value: Optional[float] = None
    threshold: Optional[float] = None


@dataclass
class OfficerPay:
    name: str
    title: str
    total_pay: Optional[float] = None
    pay_pct_of_net_income: Optional[float] = None


@dataclass
class EarningsMissRecord:
    quarter: str
    eps_actual: Optional[float] = None
    eps_estimate: Optional[float] = None
    surprise_pct: Optional[float] = None
    beat: bool = True


@dataclass
class GovernanceResult:
    red_flags: List[RedFlag] = field(default_factory=list)
    governance_scores: dict = field(default_factory=dict)
    officers: List[OfficerPay] = field(default_factory=list)
    earnings_track: List[EarningsMissRecord] = field(default_factory=list)
    total_mgmt_pay: float = 0.0
    mgmt_pay_pct: Optional[float] = None
    insider_ownership_pct: Optional[float] = None
    shares_short_pct: Optional[float] = None
    overall_risk_level: str = "UNKNOWN"
    risk_score: float = 0.0  # 0 (clean) to 100 (severe)
    summary: str = ""


class GovernanceAnalyzer:
    REMUNERATION_THRESHOLD = 0.11  # 11% of net income

    def analyze(
        self,
        info: dict,
        company_officers: list,
        earnings_history: pd.DataFrame,
        governance_scores: dict,
        insider_purchases: pd.DataFrame,
    ) -> GovernanceResult:
        result = GovernanceResult()
        result.governance_scores = governance_scores

        self._check_governance_scores(result, governance_scores)
        self._check_remuneration(result, info, company_officers)
        self._check_earnings_track_record(result, earnings_history)
        self._check_insider_activity(result, info, insider_purchases)
        self._check_legal_fraud(result, info)
        self._check_short_interest(result, info)
        self._compute_overall(result)

        return result

    def _check_governance_scores(self, result: GovernanceResult, scores: dict):
        """ISS governance risk scores: 1 (low risk) to 10 (high risk)."""
        mapping = {
            "audit_risk": ("Audit Risk", "Accounting and audit oversight"),
            "board_risk": ("Board Risk", "Board independence and effectiveness"),
            "compensation_risk": ("Compensation Risk", "Executive pay alignment with shareholders"),
            "shareholder_rights_risk": ("Shareholder Rights", "Protection of minority shareholders"),
            "overall_risk": ("Overall Governance", "Composite governance quality"),
        }

        for key, (title, desc) in mapping.items():
            val = scores.get(key)
            if val is None:
                continue
            if val >= 8:
                sev = "CRITICAL"
                detail = f"{desc} score {val}/10 — severe governance concern"
            elif val >= 6:
                sev = "HIGH"
                detail = f"{desc} score {val}/10 — elevated risk"
            elif val >= 4:
                sev = "MEDIUM"
                detail = f"{desc} score {val}/10 — moderate risk"
            else:
                sev = "OK"
                detail = f"{desc} score {val}/10 — well governed"

            result.red_flags.append(RedFlag(
                category="Governance",
                title=title,
                severity=sev,
                detail=detail,
                metric_value=val,
                threshold=6,
            ))

    def _check_remuneration(self, result: GovernanceResult, info: dict, officers: list):
        """Flag if total management pay exceeds threshold % of net income."""
        net_income = info.get("netIncomeToCommon", 0) or 0

        total_pay = 0
        for off in officers:
            pay = off.get("totalPay", 0) or 0
            total_pay += pay

            pct = (pay / net_income * 100) if net_income > 0 else None
            result.officers.append(OfficerPay(
                name=off.get("name", "Unknown"),
                title=off.get("title", "Unknown"),
                total_pay=pay if pay > 0 else None,
                pay_pct_of_net_income=round(pct, 3) if pct else None,
            ))

        result.total_mgmt_pay = total_pay

        if net_income > 0 and total_pay > 0:
            pct = total_pay / net_income
            result.mgmt_pay_pct = round(pct * 100, 2)

            if pct > self.REMUNERATION_THRESHOLD:
                sev = "CRITICAL" if pct > 0.20 else "HIGH"
                result.red_flags.append(RedFlag(
                    category="Remuneration",
                    title="Excessive Management Pay",
                    severity=sev,
                    detail=f"Total executive compensation is {pct:.1%} of net income "
                           f"(threshold: {self.REMUNERATION_THRESHOLD:.0%}). "
                           f"${total_pay:,.0f} vs ${net_income:,.0f} net income.",
                    metric_value=pct * 100,
                    threshold=self.REMUNERATION_THRESHOLD * 100,
                ))
            else:
                result.red_flags.append(RedFlag(
                    category="Remuneration",
                    title="Management Pay",
                    severity="OK",
                    detail=f"Total executive compensation is {pct:.2%} of net income — within norms.",
                    metric_value=pct * 100,
                    threshold=self.REMUNERATION_THRESHOLD * 100,
                ))
        elif net_income <= 0 and total_pay > 0:
            result.red_flags.append(RedFlag(
                category="Remuneration",
                title="Pay Despite Losses",
                severity="HIGH",
                detail=f"Management draws ${total_pay:,.0f} despite the company reporting a net loss.",
                metric_value=total_pay,
            ))

    def _check_earnings_track_record(self, result: GovernanceResult, earnings_history: pd.DataFrame):
        """Flag if management consistently misses earnings estimates."""
        if earnings_history is None or len(earnings_history) == 0:
            return

        miss_count = 0
        total = 0
        for idx, row in earnings_history.iterrows():
            actual = row.get("epsActual")
            estimate = row.get("epsEstimate")
            surprise = row.get("surprisePercent")
            quarter = str(idx)

            if actual is None or estimate is None:
                continue
            if isinstance(actual, float) and math.isnan(actual):
                continue

            total += 1
            beat = float(actual) >= float(estimate)
            if not beat:
                miss_count += 1

            result.earnings_track.append(EarningsMissRecord(
                quarter=quarter,
                eps_actual=float(actual),
                eps_estimate=float(estimate),
                surprise_pct=float(surprise) * 100 if surprise is not None and not (isinstance(surprise, float) and math.isnan(surprise)) else None,
                beat=beat,
            ))

        if total > 0:
            miss_rate = miss_count / total
            if miss_rate >= 0.75:
                sev = "CRITICAL"
                detail = f"Missed estimates {miss_count}/{total} quarters — management cannot deliver on guidance"
            elif miss_rate >= 0.5:
                sev = "HIGH"
                detail = f"Missed estimates {miss_count}/{total} quarters — unreliable forecasting"
            elif miss_rate > 0:
                sev = "LOW"
                detail = f"Missed estimates {miss_count}/{total} quarters — mostly reliable"
            else:
                sev = "OK"
                detail = f"Beat estimates all {total} quarters — strong execution"

            result.red_flags.append(RedFlag(
                category="Forecasting",
                title="Earnings Track Record",
                severity=sev,
                detail=detail,
                metric_value=miss_rate * 100,
                threshold=50,
            ))

    def _check_insider_activity(self, result: GovernanceResult, info: dict, insider_purchases: pd.DataFrame):
        """Check insider ownership and net activity for pledged-share risk proxy."""
        insider_pct = info.get("heldPercentInsiders")
        if insider_pct is not None:
            result.insider_ownership_pct = round(insider_pct * 100, 2)

            if insider_pct > 0.50:
                result.red_flags.append(RedFlag(
                    category="Insider/Pledging",
                    title="Highly Concentrated Insider Ownership",
                    severity="HIGH",
                    detail=f"Insiders hold {insider_pct:.1%} of shares — potential pledging or "
                           "forced-selling risk if collateral calls occur.",
                    metric_value=insider_pct * 100,
                    threshold=50,
                ))
            elif insider_pct > 0.30:
                result.red_flags.append(RedFlag(
                    category="Insider/Pledging",
                    title="Elevated Insider Ownership",
                    severity="MEDIUM",
                    detail=f"Insiders hold {insider_pct:.1%} — monitor for pledge disclosures.",
                    metric_value=insider_pct * 100,
                    threshold=30,
                ))
            else:
                result.red_flags.append(RedFlag(
                    category="Insider/Pledging",
                    title="Insider Ownership",
                    severity="OK",
                    detail=f"Insiders hold {insider_pct:.1%} — no concentration concern.",
                    metric_value=insider_pct * 100,
                ))

        # Net insider sell pressure
        if insider_purchases is not None and len(insider_purchases) > 0:
            try:
                net_row = insider_purchases[
                    insider_purchases.iloc[:, 0].str.contains("Net Shares", case=False, na=False)
                ]
                if len(net_row) > 0:
                    net_shares = net_row.iloc[0, 1]
                    if net_shares is not None and not pd.isna(net_shares):
                        net_val = float(net_shares)
                        if net_val < -100000:
                            result.red_flags.append(RedFlag(
                                category="Insider/Pledging",
                                title="Net Insider Selling",
                                severity="MEDIUM",
                                detail=f"Net insider sales of {abs(net_val):,.0f} shares in last 6 months.",
                                metric_value=net_val,
                            ))
            except Exception:
                pass

    def _check_legal_fraud(self, result: GovernanceResult, info: dict):
        """Flag elevated audit risk as a proxy for legal / fraud concerns.
        yfinance doesn't provide litigation data directly, but ISS audit risk
        captures accounting irregularity risk."""
        audit = info.get("auditRisk")
        if audit is not None and audit >= 7:
            result.red_flags.append(RedFlag(
                category="Legal/Fraud",
                title="Elevated Audit & Legal Risk",
                severity="CRITICAL" if audit >= 9 else "HIGH",
                detail=f"ISS audit risk score {audit}/10 — heightened risk of accounting "
                       "irregularities or pending regulatory issues. Investigate SEC filings.",
                metric_value=audit,
                threshold=7,
            ))

    def _check_short_interest(self, result: GovernanceResult, info: dict):
        """Elevated short interest can signal market skepticism about management."""
        short_pct = info.get("shortPercentOfFloat")
        if short_pct is not None:
            result.shares_short_pct = round(short_pct * 100, 2)
            if short_pct > 0.20:
                result.red_flags.append(RedFlag(
                    category="Market Sentiment",
                    title="Very High Short Interest",
                    severity="HIGH",
                    detail=f"{short_pct:.1%} of float shorted — significant market skepticism.",
                    metric_value=short_pct * 100,
                    threshold=20,
                ))
            elif short_pct > 0.10:
                result.red_flags.append(RedFlag(
                    category="Market Sentiment",
                    title="Elevated Short Interest",
                    severity="MEDIUM",
                    detail=f"{short_pct:.1%} of float shorted — some bearish bets.",
                    metric_value=short_pct * 100,
                    threshold=10,
                ))

    def _compute_overall(self, result: GovernanceResult):
        severity_weights = {"CRITICAL": 30, "HIGH": 18, "MEDIUM": 8, "LOW": 3, "OK": 0}
        total = 0
        for flag in result.red_flags:
            total += severity_weights.get(flag.severity, 0)

        result.risk_score = min(100, total)

        if result.risk_score >= 60:
            result.overall_risk_level = "CRITICAL"
            result.summary = "Multiple severe red flags detected — extreme caution advised."
        elif result.risk_score >= 35:
            result.overall_risk_level = "HIGH"
            result.summary = "Significant governance/management concerns identified."
        elif result.risk_score >= 15:
            result.overall_risk_level = "MEDIUM"
            result.summary = "Some areas of concern — due diligence recommended."
        elif result.risk_score > 0:
            result.overall_risk_level = "LOW"
            result.summary = "Minor concerns only — generally well-governed."
        else:
            result.overall_risk_level = "OK"
            result.summary = "No material red flags detected — clean governance profile."
