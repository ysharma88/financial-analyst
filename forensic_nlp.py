"""Forensic NLP — Earnings transcript deception markers + SEC EDGAR MD&A analysis.

Three sub-systems:

1. SEC EDGAR Fetcher
   - Pulls latest 10-K or 10-Q filing directly from EDGAR full-text search API (free, no key)
   - Extracts Item 7 (MD&A) and Item 1A (Risk Factors) sections

2. Loughran-McDonald Linguistic Forensics
   - Scores text against the academic LM financial word lists
   - Categories: Uncertainty, Litigious, Negation, Hedging, Positive, Negative
   - High uncertainty + high litigious = management hiding something
   - TATA cross-check: if LM negative is high AND TATA is high = double manipulation flag

3. Claude Forensic Synthesis
   - Passes MD&A + key metrics to Claude for deception marker analysis
   - Looks for: vague guidance, excessive qualifiers, non-answer patterns in Q&A
   - Returns structured verdict: Clean / Watch / Red Flag / Critical
"""

from __future__ import annotations

import logging
import math
import re
import urllib.request
import urllib.parse
import json
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger("forensic_nlp")

# ---------------------------------------------------------------------------
# Loughran-McDonald word lists (condensed academic lists)
# ---------------------------------------------------------------------------

LM_UNCERTAINTY = {
    "approximately", "appears", "believes", "cannot", "conceivable", "contingent",
    "could", "depends", "doubt", "estimate", "expected", "feels", "fluctuate",
    "indefinite", "indeterminate", "intend", "likely", "may", "might", "nearly",
    "no assurance", "normally", "on the other hand", "pending", "possible",
    "possibly", "predict", "projected", "roughly", "seems", "should", "sometime",
    "suggest", "uncertain", "uncertainty", "unclear", "unpredictable", "unusual",
    "variable", "various", "whether", "would",
}

LM_LITIGIOUS = {
    "abusive", "allegation", "allege", "alleged", "arbitration", "breach",
    "claim", "class action", "complaint", "contingency", "controversy",
    "criminal", "damages", "defendant", "deposition", "dispute", "enforcement",
    "fine", "fraud", "illegal", "indemnification", "infringement", "injunction",
    "investigation", "judgment", "lawsuit", "legal", "legislation", "liable",
    "litigation", "misrepresent", "negligence", "penalty", "plaintiff",
    "proceeding", "prosecution", "regulatory", "restatement", "sanction",
    "settlement", "statute", "subpoena", "sue", "verdict", "violation",
    "warrant", "whistleblower",
}

LM_NEGATIVE = {
    "abandon", "adverse", "against", "breach", "burden", "challenge", "close",
    "concern", "crisis", "critical", "damage", "decline", "decrease", "default",
    "deficit", "delay", "deteriorate", "difficulties", "disappoint", "discontinue",
    "downturn", "elimination", "fail", "failure", "harm", "headwind", "impair",
    "impairment", "inability", "inadequate", "inconsistent", "inferior",
    "insufficient", "interrupt", "loss", "low", "miss", "negative", "obsolete",
    "obstacle", "poor", "reduce", "reduction", "restructure", "risk", "severe",
    "shortfall", "slow", "struggle", "termination", "uncertainty", "unfavorable",
    "unprofitable", "vulnerability", "warn", "weakness", "worse", "writedown",
    "writeoff",
}

LM_POSITIVE = {
    "accelerate", "achievement", "advantage", "benefit", "confidence",
    "deliver", "distinguished", "effective", "efficient", "enhance",
    "exceed", "excellent", "exceptional", "expand", "favorable", "gain",
    "growth", "improve", "increase", "innovation", "leading", "momentum",
    "opportunity", "outperform", "positive", "progress", "record", "robust",
    "strong", "success", "superior", "sustained", "upside",
}

LM_HEDGING = {
    "although", "assuming", "caveat", "conditional", "contingent", "depending",
    "except", "however", "if", "in the event", "insofar", "nevertheless",
    "notwithstanding", "otherwise", "provided that", "regardless", "subject to",
    "unless", "until", "whereas", "while",
}

# Deception phrases: vague non-answers common in earnings calls
DECEPTION_PHRASES = [
    "we are not going to provide guidance",
    "we are not in a position to",
    "as we have said before",
    "i'll take that offline",
    "we don't break that out",
    "that's not something we disclose",
    "we are not going to comment",
    "we remain confident",
    "we are very excited",
    "we believe we are well positioned",
    "we look forward to",
    "we are committed to",
    "i would rather not speculate",
    "stay tuned",
    "we will see",
    "it's too early to say",
    "we are monitoring closely",
    "we are evaluating",
    "we are assessing",
]

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LMScores:
    uncertainty_ratio: float    # uncertainty words / total words
    litigious_ratio: float
    negative_ratio: float
    positive_ratio: float
    hedging_ratio: float
    sentiment_net: float        # positive_ratio - negative_ratio
    total_words: int
    deception_phrase_count: int
    deception_phrases_found: list[str]


@dataclass
class EdgarSection:
    form_type: str              # "10-K" or "10-Q"
    filed_date: str
    period: str
    mda_text: str               # Item 7 / MD&A
    risk_text: str              # Item 1A / Risk Factors
    accession: str
    word_count: int


@dataclass
class ForensicVerdict:
    verdict: str                # "Clean", "Watch", "Red Flag", "Critical"
    confidence: str             # "High", "Medium", "Low"
    score: float                # 0–100 (higher = more concerning)
    flags: list[str]            # specific issues found
    summary: str                # 2–3 sentence synthesis
    lm_scores: Optional[LMScores] = None


@dataclass
class ForensicNLPResult:
    ticker: str
    filing: Optional[EdgarSection] = None
    lm: Optional[LMScores] = None
    verdict: Optional[ForensicVerdict] = None
    ai_analysis: Optional[str] = None      # Claude's narrative
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# SEC EDGAR fetcher
# ---------------------------------------------------------------------------

_EDGAR_SEARCH = "https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&dateRange=custom&startdt={start}&forms={form}"
_EDGAR_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
_EDGAR_FILING_BASE = "https://www.sec.gov/Archives/edgar/full-index/"

_HEADERS = {"User-Agent": "FinancialAnalystTool/1.0 (educational; contact@example.com)"}


def _edgar_get(url: str, retries: int = 2) -> Optional[bytes]:
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=_HEADERS)
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.read()
        except Exception as e:
            if attempt < retries:
                time.sleep(1)
            else:
                logger.debug("EDGAR fetch failed %s: %s", url, e)
    return None


def _get_cik(ticker: str) -> Optional[str]:
    """Resolve ticker to CIK via EDGAR company search.

    Tries multiple form types so foreign filers (20-F) are found as well as
    domestic (10-K). Falls back to a type-less search as a last resort.
    """
    # Try with each major form type first — type-filtered searches return CIK
    # directly in the Atom feed even when only one result exists
    for form_type in ("10-K", "20-F", ""):
        type_param = f"&type={form_type}" if form_type else ""
        url = (
            f"https://www.sec.gov/cgi-bin/browse-edgar"
            f"?action=getcompany&company=&CIK={urllib.parse.quote(ticker.upper())}"
            f"{type_param}&dateb=&owner=include&count=5&search_text=&output=atom"
        )
        data = _edgar_get(url)
        if data:
            text = data.decode("utf-8", errors="ignore")
            m = re.search(r"CIK=(\d+)", text)
            if m:
                return m.group(1).zfill(10)

    # Last-resort: EDGAR full-text search index
    url2 = f"https://efts.sec.gov/LATEST/search-index?q=%22{urllib.parse.quote(ticker.upper())}%22&forms=10-K,20-F"
    data2 = _edgar_get(url2)
    if data2:
        try:
            j = json.loads(data2)
            hits = j.get("hits", {}).get("hits", [])
            if hits:
                cik = hits[0].get("_source", {}).get("entity_id", "")
                if cik:
                    return str(cik).zfill(10)
        except Exception:
            pass
    return None


def _get_latest_filing(cik: str, form: str = "10-K") -> Optional[dict]:
    """Get latest filing metadata for a CIK."""
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    data = _edgar_get(url)
    if not data:
        return None
    try:
        sub = json.loads(data)
        filings = sub.get("filings", {}).get("recent", {})
        forms = filings.get("form", [])
        dates = filings.get("filingDate", [])
        accessions = filings.get("accessionNumber", [])
        periods = filings.get("reportDate", [])

        for i, f in enumerate(forms):
            if f == form:
                return {
                    "form": f,
                    "date": dates[i] if i < len(dates) else "",
                    "accession": accessions[i].replace("-", "") if i < len(accessions) else "",
                    "accession_raw": accessions[i] if i < len(accessions) else "",
                    "period": periods[i] if i < len(periods) else "",
                    "cik": cik,
                }
    except Exception as e:
        logger.debug("Filing parse failed: %s", e)
    return None


def _fetch_filing_text(cik: str, accession: str) -> Optional[str]:
    """Fetch the primary document text of a filing from EDGAR.

    accession may arrive either as '0001858985-26-000008' (raw) or
    '000185898526000008' (no-dash). We normalise both.

    Strategy:
      1. Use submissions JSON to get the canonical primaryDocument filename.
         This avoids accidentally fetching exhibits listed first in the index.
      2. Fall back to the index HTML regex if submissions JSON fails.
      3. Final fallback: try the dashed-accession .txt bundle.
    """
    # Normalise: ensure we have both forms
    if "-" in accession:
        acc_dashed = accession
        acc_nodash = accession.replace("-", "")
    else:
        acc_nodash = accession
        acc_dashed = f"{accession[:10]}-{accession[10:12]}-{accession[12:]}"

    cik_int = int(cik)
    doc_name = None

    # 1. submissions JSON — most reliable: gives exact primaryDocument filename
    sub_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    sub_data = _edgar_get(sub_url)
    if sub_data:
        try:
            sub = json.loads(sub_data)
            filings = sub.get("filings", {}).get("recent", {})
            accessions_list = filings.get("accessionNumber", [])
            primary_docs = filings.get("primaryDocument", [])
            for i, a in enumerate(accessions_list):
                if a.replace("-", "") == acc_nodash and i < len(primary_docs):
                    prim = primary_docs[i]
                    if prim:
                        doc_name = f"/Archives/edgar/data/{cik_int}/{acc_nodash}/{prim}"
                    break
        except Exception:
            pass

    # 2. Index HTML fallback — first .htm link in the filing index
    if not doc_name:
        idx_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/{acc_dashed}-index.htm"
        data = _edgar_get(idx_url)
        if data:
            text = data.decode("utf-8", errors="ignore")
            m = re.search(r'href="(/Archives/edgar/data/\d+/\d+/([^"]+\.(?:htm|txt)))"', text, re.I)
            if m:
                doc_name = m.group(1)

    # 3. Final fallback: the full submission text bundle
    if not doc_name:
        doc_name = f"/Archives/edgar/data/{cik_int}/{acc_nodash}/{acc_dashed}.txt"

    full_url = f"https://www.sec.gov{doc_name}"
    doc_data = _edgar_get(full_url)
    if doc_data:
        return doc_data.decode("utf-8", errors="ignore")
    return None


def _extract_section(text: str, section_patterns: list[str], end_patterns: list[str],
                     max_chars: int = 12000) -> str:
    """Extract a named section from a filing by regex pattern matching.

    Large filings (20-F, 10-K) repeat section headers in a table of contents,
    sub-section cross-references, and the actual body. We collect all matching
    positions and return the candidate whose body has the most words, subject
    to a minimum of 50 words. This picks the real section over TOC entries.
    """
    text_lower = text.lower()
    candidates = []
    for pat in section_patterns:
        for m in re.finditer(pat, text_lower):
            candidates.append(m.start())

    if not candidates:
        return ""

    candidates.sort()

    def _clean(raw: str) -> str:
        raw = re.sub(r"<[^>]+>", " ", raw)
        raw = re.sub(r"&#\d+;", " ", raw)
        raw = re.sub(r"&amp;", "&", raw)
        raw = re.sub(r"&lt;", "<", raw)
        raw = re.sub(r"&gt;", ">", raw)
        raw = re.sub(r"&nbsp;", " ", raw)
        return re.sub(r"\s+", " ", raw).strip()

    best_text = ""
    best_word_count = 0

    for start in candidates:
        end = len(text)
        for pat in end_patterns:
            m = re.search(pat, text_lower[start + 100:])
            if m:
                end = start + 100 + m.start()
                break

        raw = _clean(text[start:min(end, start + max_chars)])
        wc = len(raw.split())
        if wc > best_word_count:
            best_word_count = wc
            best_text = raw

    return best_text if best_word_count >= 50 else ""


def fetch_edgar_filing(ticker: str, prefer_form: str = "10-K") -> Optional[EdgarSection]:
    """Fetch latest annual filing from EDGAR and extract MD&A + Risk sections.

    Form priority: prefer_form (10-K) → 20-F (foreign private issuers) → 10-Q.
    Foreign companies like ONON (Swiss) file 20-F instead of 10-K.
    """
    try:
        cik = _get_cik(ticker)
        if not cik:
            logger.info("Could not resolve CIK for %s", ticker)
            return None

        filing_meta = _get_latest_filing(cik, prefer_form)
        if not filing_meta:
            # Foreign private issuers file 20-F instead of 10-K
            filing_meta = _get_latest_filing(cik, "20-F")
        if not filing_meta:
            # Fall back to 10-Q for companies with no recent annual filing
            filing_meta = _get_latest_filing(cik, "10-Q")
        if not filing_meta:
            return None

        text = _fetch_filing_text(cik, filing_meta["accession"])
        if not text or len(text) < 1000:
            return None

        form = filing_meta["form"]

        # MD&A: Item 7 in 10-K, Item 5 in 20-F, Item 2 in 10-Q
        mda_starts = [
            r"item\s+7[\.\s]+management.{0,30}discussion",
            r"item\s+5[\.\s]+operating.{0,30}financial\s+review",   # 20-F
            r"item\s+5[\.\s]+management.{0,30}discussion",           # 20-F alt
            r"item\s+2[\.\s]+management.{0,30}discussion",
            r"management.s discussion and analysis",
            r"operating and financial review",                        # 20-F narrative
        ]
        mda_ends = [
            r"item\s+7a[\.\s]+quantitative",
            r"item\s+8[\.\s]+financial",
            r"item\s+5a[\.\s]+",
            r"item\s+6[\.\s]+",                                      # 20-F next section
            r"item\s+3[\.\s]+quantitative",
        ]
        mda_text = _extract_section(text, mda_starts, mda_ends, max_chars=15000)

        # Risk Factors: Item 1A in 10-K/10-Q, Item 3D in 20-F
        risk_starts = [
            r"item\s+1a[\.\s]+risk\s+factor",
            r"item\s+3[d\.\s]+risk\s+factor",                       # 20-F
            r"item\s+3[\.\s]+key\s+information.{0,50}risk",         # 20-F alt
            r"risk\s+factors",
        ]
        risk_ends = [
            r"item\s+1b[\.\s]+",
            r"item\s+2[\.\s]+",
            r"item\s+4[\.\s]+",                                      # 20-F
            r"unresolved staff comments",
        ]
        risk_text = _extract_section(text, risk_starts, risk_ends, max_chars=8000)

        word_count = len((mda_text + " " + risk_text).split())

        return EdgarSection(
            form_type=filing_meta["form"],
            filed_date=filing_meta["date"],
            period=filing_meta["period"],
            mda_text=mda_text,
            risk_text=risk_text,
            accession=filing_meta["accession_raw"],
            word_count=word_count,
        )
    except Exception as e:
        logger.warning("EDGAR fetch failed for %s: %s", ticker, e)
        return None


# ---------------------------------------------------------------------------
# Loughran-McDonald scoring
# ---------------------------------------------------------------------------

def score_lm(text: str) -> LMScores:
    """Score text against Loughran-McDonald financial word lists."""
    if not text:
        return LMScores(0, 0, 0, 0, 0, 0, 0, 0, [])

    words = re.findall(r"\b[a-z]+\b", text.lower())
    total = max(len(words), 1)
    word_set = set(words)

    uncertainty = sum(1 for w in words if w in LM_UNCERTAINTY)
    litigious = sum(1 for w in words if w in LM_LITIGIOUS)
    negative = sum(1 for w in words if w in LM_NEGATIVE)
    positive = sum(1 for w in words if w in LM_POSITIVE)
    hedging = sum(1 for w in words if w in LM_HEDGING)

    text_lower = text.lower()
    found_phrases = [p for p in DECEPTION_PHRASES if p in text_lower]

    return LMScores(
        uncertainty_ratio=round(uncertainty / total, 4),
        litigious_ratio=round(litigious / total, 4),
        negative_ratio=round(negative / total, 4),
        positive_ratio=round(positive / total, 4),
        hedging_ratio=round(hedging / total, 4),
        sentiment_net=round((positive - negative) / total, 4),
        total_words=total,
        deception_phrase_count=len(found_phrases),
        deception_phrases_found=found_phrases[:10],
    )


# ---------------------------------------------------------------------------
# Rule-based forensic verdict (no API needed)
# ---------------------------------------------------------------------------

def _rule_based_verdict(lm: LMScores, tata: Optional[float] = None) -> ForensicVerdict:
    """Compute forensic verdict purely from LM scores + TATA ratio."""
    flags = []
    score = 0.0

    # Uncertainty above typical range (>3.5% of words)
    if lm.uncertainty_ratio > 0.045:
        flags.append(f"Extreme uncertainty language ({lm.uncertainty_ratio:.1%} of words) — management avoiding commitments")
        score += 25
    elif lm.uncertainty_ratio > 0.035:
        flags.append(f"Elevated uncertainty language ({lm.uncertainty_ratio:.1%} of words)")
        score += 12

    # Litigious language
    if lm.litigious_ratio > 0.025:
        flags.append(f"High legal/litigious language ({lm.litigious_ratio:.1%}) — significant legal exposure likely")
        score += 20
    elif lm.litigious_ratio > 0.015:
        flags.append(f"Moderate legal language ({lm.litigious_ratio:.1%})")
        score += 10

    # Negative sentiment dominance
    if lm.sentiment_net < -0.015:
        flags.append(f"Negative sentiment dominates MD&A (net {lm.sentiment_net:+.3f}) — forward guidance pessimistic")
        score += 15
    elif lm.sentiment_net > 0.025:
        flags.append(f"Unusually positive language (net {lm.sentiment_net:+.3f}) — possible spin/over-promotion")
        score += 8

    # Deception phrases
    if lm.deception_phrase_count >= 4:
        flags.append(f"{lm.deception_phrase_count} evasive management phrases detected — pattern of non-disclosure")
        score += 20
    elif lm.deception_phrase_count >= 2:
        flags.append(f"{lm.deception_phrase_count} evasive phrases found: {', '.join(lm.deception_phrases_found[:2])}")
        score += 10

    # Hedging excess
    if lm.hedging_ratio > 0.04:
        flags.append(f"Excessive hedging language ({lm.hedging_ratio:.1%}) — no firm commitments made")
        score += 10

    # TATA cross-check: high accruals + high linguistic red flags = double flag
    if tata is not None and tata > 0.05:
        flags.append(f"TATA = {tata:.3f} (>0.05): earnings not backed by cash flow — accrual manipulation risk")
        score += 20
        if lm.negative_ratio > 0.02 and lm.uncertainty_ratio > 0.03:
            flags.append("CRITICAL: High TATA + high linguistic uncertainty = double manipulation signal")
            score += 15

    score = min(score, 100)

    if score >= 65:
        verdict = "Critical"
        confidence = "High"
    elif score >= 40:
        verdict = "Red Flag"
        confidence = "Medium" if len(flags) >= 2 else "Low"
    elif score >= 20:
        verdict = "Watch"
        confidence = "Medium"
    else:
        verdict = "Clean"
        confidence = "High"

    # Build summary
    if verdict == "Critical":
        summary = (
            f"Multiple forensic signals converge: {len(flags)} red flags detected. "
            f"Uncertainty ratio {lm.uncertainty_ratio:.1%} and {lm.deception_phrase_count} evasive phrases "
            f"suggest management is obscuring material information. Cross-verify with cash flow statements."
        )
    elif verdict == "Red Flag":
        summary = (
            f"{len(flags)} warning signals in MD&A language. "
            f"Elevated {'litigious' if lm.litigious_ratio > 0.015 else 'uncertainty'} language "
            f"warrants deeper due diligence before position sizing."
        )
    elif verdict == "Watch":
        summary = (
            f"Mild linguistic anomalies detected — below threshold for concern but worth monitoring. "
            f"Sentiment net {lm.sentiment_net:+.3f}, uncertainty {lm.uncertainty_ratio:.1%}."
        )
    else:
        summary = (
            f"MD&A language appears consistent with transparent disclosure. "
            f"Sentiment net {lm.sentiment_net:+.3f}, no significant deception markers detected."
        )

    return ForensicVerdict(
        verdict=verdict,
        confidence=confidence,
        score=round(score, 1),
        flags=flags,
        summary=summary,
        lm_scores=lm,
    )


# ---------------------------------------------------------------------------
# Claude forensic synthesis (optional — enhances rule-based verdict)
# ---------------------------------------------------------------------------

def _claude_forensic_analysis(ticker: str, mda_excerpt: str, lm: LMScores,
                               quality_scores: Optional[dict] = None) -> Optional[str]:
    """Pass MD&A + LM scores to Claude for deep forensic synthesis."""
    try:
        import anthropic
        import os
        key = os.getenv("ANTHROPIC_API_KEY", "")
        if not key:
            return None

        client = anthropic.Anthropic(api_key=key)

        qs_context = ""
        if quality_scores:
            beneish = quality_scores.get("beneish_m")
            piotroski = quality_scores.get("piotroski_f")
            altman = quality_scores.get("altman_z")
            qs_context = (
                f"\nQuantitative forensic scores already computed:\n"
                f"- Beneish M-Score: {beneish} (flag if > -1.78)\n"
                f"- Piotroski F-Score: {piotroski}/9\n"
                f"- Altman Z-Score: {altman}\n"
            )

        prompt = f"""You are a forensic financial analyst specializing in detecting earnings manipulation,
management deception, and disclosure quality issues in SEC filings.

Analyze the following MD&A excerpt from {ticker}'s latest filing and identify deception markers.

LINGUISTIC SCORES (Loughran-McDonald):
- Uncertainty ratio: {lm.uncertainty_ratio:.2%} (flag if >3.5%)
- Litigious ratio: {lm.litigious_ratio:.2%} (flag if >1.5%)
- Negative ratio: {lm.negative_ratio:.2%}
- Positive ratio: {lm.positive_ratio:.2%}
- Hedging ratio: {lm.hedging_ratio:.2%}
- Net sentiment: {lm.sentiment_net:+.4f}
- Evasive phrases found: {lm.deception_phrase_count} ({', '.join(lm.deception_phrases_found[:3]) if lm.deception_phrases_found else 'none'})
{qs_context}

MD&A EXCERPT (first 3000 chars):
{mda_excerpt[:3000]}

Provide a forensic analysis covering:
1. TONE ANALYSIS: Is management language unusually vague, defensive, or over-promotional?
2. GUIDANCE QUALITY: Are forward-looking statements specific or deliberately evasive?
3. RISK DISCLOSURE: Are risks buried in boilerplate or specifically described?
4. CONSISTENCY CHECK: Does the language match the quantitative scores above?
5. VERDICT: Clean / Watch / Red Flag / Critical — and your single most important finding.

Be specific. Quote phrases from the text where relevant. Keep response under 300 words."""

        resp = client.messages.create(
            model="claude-opus-4-7",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()
    except Exception as e:
        logger.debug("Claude forensic analysis failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def analyze(ticker: str, quality_scores: Optional[dict] = None) -> ForensicNLPResult:
    """
    Full forensic NLP pipeline:
    1. Fetch latest 10-K/10-Q from EDGAR
    2. Score MD&A + Risk Factors with Loughran-McDonald word lists
    3. Cross-check with TATA (from quality_scores if provided)
    4. Run Claude forensic synthesis if API key available
    """
    result = ForensicNLPResult(ticker=ticker)

    try:
        filing = fetch_edgar_filing(ticker)
        result.filing = filing

        if not filing or (not filing.mda_text and not filing.risk_text):
            result.error = "Could not fetch EDGAR filing text"
            # Still run LM on empty — returns zeroed scores
            result.lm = LMScores(0, 0, 0, 0, 0, 0, 0, 0, [])
            result.verdict = ForensicVerdict(
                verdict="Unknown", confidence="Low", score=0,
                flags=["EDGAR filing text unavailable — cannot perform linguistic analysis"],
                summary="Filing text could not be retrieved. Manual review recommended.",
            )
            return result

        combined_text = filing.mda_text + " " + filing.risk_text
        lm = score_lm(combined_text)
        result.lm = lm

        # Get TATA from quality_scores if passed in
        tata = None
        if quality_scores and isinstance(quality_scores, dict):
            tata = quality_scores.get("tata")

        result.verdict = _rule_based_verdict(lm, tata)

        # Claude enhancement
        if filing.mda_text:
            result.ai_analysis = _claude_forensic_analysis(
                ticker, filing.mda_text, lm, quality_scores
            )

    except Exception as e:
        logger.warning("Forensic NLP failed for %s: %s", ticker, e)
        result.error = str(e)

    return result


# ---------------------------------------------------------------------------
# Cached entry point
# ---------------------------------------------------------------------------

def cached_analyze(ticker: str, quality_scores: Optional[dict] = None) -> ForensicNLPResult:
    """TTL-cached version — 7 day TTL (10-K filed quarterly, rarely changes)."""
    import cache as _cache
    TTL = 7 * 24 * 3600
    cached = _cache.get_ttl(ticker.upper(), "forensic_nlp", TTL)
    if cached is not None:
        return cached
    result = analyze(ticker, quality_scores)
    _cache.set_ttl(ticker.upper(), "forensic_nlp", result)
    return result
