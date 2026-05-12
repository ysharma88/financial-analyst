"""News & sentiment analysis — fetches headlines and scores sentiment.

Uses the Claude API (claude-sonnet-4-6) when ANTHROPIC_API_KEY is set, scoring
all headlines in a single batched call. Falls back to the original keyword-based
engine when the key is absent or the API call fails.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import yfinance as yf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keyword-based fallback lexicon (finance-specific)
# ---------------------------------------------------------------------------
BULLISH_WORDS = {
    "beat", "beats", "surge", "surges", "soar", "soars", "rally", "rallies",
    "upgrade", "upgrades", "upgraded", "outperform", "buy", "bullish",
    "record", "high", "growth", "grows", "profit", "profitable", "strong",
    "boost", "boosts", "boosted", "gain", "gains", "positive", "optimistic",
    "breakout", "momentum", "recovery", "recovers", "innovation", "expand",
    "expansion", "dividend", "buyback", "repurchase", "approval", "approved",
    "partnership", "deal", "acquisition", "acquires", "launch", "launches",
    "exceed", "exceeds", "exceeded", "expectations", "upside", "raise",
    "raises", "raised", "accelerate", "accelerates", "robust", "resilient",
    "opportunity", "opportunities", "revenue", "earnings", "beat",
    "overweight", "top pick", "best", "winner", "winning",
}

BEARISH_WORDS = {
    "miss", "misses", "fall", "falls", "decline", "declines", "drop", "drops",
    "downgrade", "downgrades", "downgraded", "underperform", "sell", "bearish",
    "low", "loss", "losses", "weak", "weakness", "risk", "risks", "risky",
    "crash", "crashes", "plunge", "plunges", "warning", "warns", "concern",
    "concerns", "recession", "slowdown", "layoff", "layoffs", "cut", "cuts",
    "lawsuit", "sued", "fine", "fined", "penalty", "investigation", "probe",
    "fraud", "scandal", "bankruptcy", "debt", "default", "tariff", "tariffs",
    "sanctions", "ban", "banned", "restriction", "restrictions", "delay",
    "delayed", "shortage", "shortages", "recall", "inflation", "overvalued",
    "bubble", "underweight", "worst", "loser", "losing", "negative",
    "disappointing", "disappointed", "below", "miss", "missed",
}

REGULATORY_WORDS = {
    "regulation", "regulatory", "sec", "fda", "ftc", "antitrust", "compliance",
    "legislation", "bill", "law", "policy", "mandate", "ruling", "court",
    "hearing", "subpoena", "audit", "oversight", "enforcement", "approval",
    "clearance", "patent", "license", "permit",
}


@dataclass
class NewsItem:
    title: str
    summary: str
    publisher: str
    published_at: str
    url: str
    sentiment_score: float  # -1 to +1
    sentiment_label: str  # "Bullish", "Bearish", "Neutral"
    is_regulatory: bool
    key_topics: List[str] = field(default_factory=list)


@dataclass
class NewsSentimentResult:
    ticker: str
    company_name: str
    articles: List[NewsItem] = field(default_factory=list)
    overall_sentiment: float = 0.0  # -1 to +1
    sentiment_label: str = "Neutral"
    bullish_count: int = 0
    bearish_count: int = 0
    neutral_count: int = 0
    regulatory_count: int = 0
    summary: str = ""


# ---------------------------------------------------------------------------
# Claude-powered batch scorer
# ---------------------------------------------------------------------------

_CLAUDE_SYSTEM = """\
You are a financial news sentiment analyst. Given a list of news headlines and \
summaries about a stock, score each one for sentiment from the perspective of \
a stock investor.

Return a JSON array (one object per article, same order as input) with exactly \
these fields:
  "score": float between -1.0 (very bearish) and +1.0 (very bullish)
  "label": one of "Bullish", "Slightly Bullish", "Neutral", "Slightly Bearish", "Bearish"
  "is_regulatory": true/false — does this article mention regulatory/legal matters?
  "topics": array of up to 4 strings from: \
["Earnings","Regulation","M&A","Product","Leadership","Market","Dividend","Analyst"]

Reply with only the JSON array and nothing else.\
"""


def _score_batch_with_claude(articles: list[dict]) -> Optional[list[dict]]:
    """Send all headlines to Claude in one call; return per-article dicts or None."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        lines = []
        for i, a in enumerate(articles, 1):
            lines.append(f"{i}. Title: {a['title']}\n   Summary: {a['summary'][:200]}")
        user_content = "\n\n".join(lines)

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            system=[
                {
                    "type": "text",
                    "text": _CLAUDE_SYSTEM,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_content}],
        )
        raw = response.content[0].text.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
        return json.loads(raw)
    except Exception as exc:
        logger.warning("Claude sentiment scoring failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------
class NewsSentimentAnalyzer:

    def analyze(self, ticker: str, company_name: str = "") -> NewsSentimentResult:
        result = NewsSentimentResult(ticker=ticker, company_name=company_name)

        try:
            stock = yf.Ticker(ticker)
            raw_news = stock.news
            if not raw_news:
                result.summary = "No recent news available."
                return result
        except Exception as e:
            logger.error("Failed to fetch news for %s: %s", ticker, e)
            result.summary = "Could not fetch news data."
            return result

        # Parse articles first (without scores)
        parsed: list[dict] = []
        for item in raw_news:
            content = item.get("content", {})
            title = content.get("title", "")
            summary_text = content.get("summary", "")
            pub_date = content.get("pubDate", "")
            provider = content.get("provider", {})
            publisher_name = provider.get("displayName", "") if isinstance(provider, dict) else str(provider)
            url = content.get("canonicalUrl", {})
            if isinstance(url, dict):
                url = url.get("url", "")
            if not title:
                continue
            parsed.append({
                "title": title,
                "summary": summary_text,
                "publisher": publisher_name,
                "published_at": self._format_date(pub_date),
                "url": str(url),
            })

        if not parsed:
            result.summary = "No analyzable news found."
            return result

        # Score with Claude; fall back to keyword engine
        claude_scores = _score_batch_with_claude(parsed)

        for i, p in enumerate(parsed):
            if claude_scores and i < len(claude_scores):
                cs = claude_scores[i]
                score = float(cs.get("score", 0.0))
                score = max(-1.0, min(1.0, score))
                label = cs.get("label", "Neutral")
                is_reg = bool(cs.get("is_regulatory", False))
                topics = list(cs.get("topics", []))[:4]
            else:
                score, label = self._score_text_keyword(p["title"], p["summary"])
                is_reg = self._check_regulatory(p["title"], p["summary"])
                topics = self._extract_topics(p["title"], p["summary"])

            # Normalise label to 3-way for counting
            simple_label = (
                "Bullish" if "Bullish" in label and "Slightly" not in label
                else "Bearish" if "Bearish" in label and "Slightly" not in label
                else "Neutral" if label == "Neutral"
                else "Bullish" if label == "Slightly Bullish"
                else "Bearish"
            )

            news_item = NewsItem(
                title=p["title"],
                summary=p["summary"][:300] if p["summary"] else "",
                publisher=p["publisher"],
                published_at=p["published_at"],
                url=p["url"],
                sentiment_score=round(score, 3),
                sentiment_label=label,
                is_regulatory=is_reg,
                key_topics=topics,
            )
            result.articles.append(news_item)

            if simple_label == "Bullish":
                result.bullish_count += 1
            elif simple_label == "Bearish":
                result.bearish_count += 1
            else:
                result.neutral_count += 1
            if is_reg:
                result.regulatory_count += 1

        if result.articles:
            scores = [a.sentiment_score for a in result.articles]
            weights = list(range(len(scores), 0, -1))
            result.overall_sentiment = round(
                sum(s * w for s, w in zip(scores, weights)) / sum(weights), 3
            )

            if result.overall_sentiment > 0.15:
                result.sentiment_label = "Bullish"
            elif result.overall_sentiment > 0.05:
                result.sentiment_label = "Slightly Bullish"
            elif result.overall_sentiment > -0.05:
                result.sentiment_label = "Neutral"
            elif result.overall_sentiment > -0.15:
                result.sentiment_label = "Slightly Bearish"
            else:
                result.sentiment_label = "Bearish"

            total = len(result.articles)
            engine = "AI" if claude_scores else "keyword"
            result.summary = (
                f"{total} articles analyzed ({engine}) — "
                f"{result.bullish_count} bullish, {result.bearish_count} bearish, "
                f"{result.neutral_count} neutral"
                f"{f', {result.regulatory_count} regulatory' if result.regulatory_count else ''}"
            )
        else:
            result.summary = "No analyzable news found."

        return result

    # ------------------------------------------------------------------
    # Keyword-based fallback (original logic)
    # ------------------------------------------------------------------

    def _score_text_keyword(self, title: str, summary: str) -> tuple:
        text = (title + " " + summary).lower()
        words = set(re.findall(r'\b\w+\b', text))

        bull_hits = words & BULLISH_WORDS
        bear_hits = words & BEARISH_WORDS

        bull_count = len(bull_hits)
        bear_count = len(bear_hits)

        title_words = set(re.findall(r'\b\w+\b', title.lower()))
        bull_count += len(title_words & BULLISH_WORDS)
        bear_count += len(title_words & BEARISH_WORDS)

        total = bull_count + bear_count
        if total == 0:
            return 0.0, "Neutral"

        score = (bull_count - bear_count) / total
        score = max(-1.0, min(1.0, score))

        if score > 0.15:
            label = "Bullish"
        elif score < -0.15:
            label = "Bearish"
        else:
            label = "Neutral"

        return round(score, 3), label

    def _check_regulatory(self, title: str, summary: str) -> bool:
        text = (title + " " + summary).lower()
        words = set(re.findall(r'\b\w+\b', text))
        return len(words & REGULATORY_WORDS) > 0

    def _extract_topics(self, title: str, summary: str) -> List[str]:
        text = (title + " " + summary).lower()
        topics = []

        topic_keywords = {
            "Earnings": {"earnings", "revenue", "profit", "eps", "quarterly", "annual"},
            "Regulation": REGULATORY_WORDS,
            "M&A": {"acquisition", "merger", "deal", "buyout", "takeover", "acquires"},
            "Product": {"launch", "product", "release", "innovation", "patent", "technology"},
            "Leadership": {"ceo", "cfo", "executive", "management", "board", "appoint"},
            "Market": {"market", "stock", "shares", "trading", "rally", "crash", "volatility"},
            "Dividend": {"dividend", "buyback", "repurchase", "payout"},
            "Analyst": {"analyst", "upgrade", "downgrade", "target", "rating", "forecast"},
        }

        words = set(re.findall(r'\b\w+\b', text))
        for topic, keywords in topic_keywords.items():
            if words & keywords:
                topics.append(topic)

        return topics[:4]

    def _format_date(self, date_str: str) -> str:
        if not date_str:
            return ""
        try:
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            return dt.strftime("%b %d, %Y %H:%M")
        except Exception:
            return date_str[:16]
