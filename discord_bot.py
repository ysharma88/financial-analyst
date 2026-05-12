"""Discord bot for the Financial Analyst app.

Polls a Discord channel for commands and responds with AI-powered analysis.
Also posts an automatic daily portfolio summary from TWS at a configurable time,
and earnings countdown alerts at 7-day, 2-day, and 1-day thresholds.

Commands (prefix: !):
  !analyze  <TICKER>   Full fundamental + technical summary
  !price    <TICKER>   Quick current price + day change
  !dcf      <TICKER>   DCF intrinsic value estimate
  !score    <TICKER>   Quality scores (Piotroski, Altman, Beneish)
  !portfolio           Daily portfolio summary from TWS (on demand)
  !earnings            Show upcoming earnings calendar for portfolio (next 30 days)
  !help                List available commands

Env vars:
  DISCORD_BOT_TOKEN      — required
  DISCORD_CHANNEL_ID     — optional, restrict to one channel
  DAILY_SUMMARY_TIME     — HH:MM in 24h (default "16:30", after US market close)
  EARNINGS_ALERT_TIME    — HH:MM in 24h (default "09:00", morning before market open)
  PORTFOLIO_TICKERS      — comma-separated fallback if TWS unavailable (e.g. "AAPL,NVDA")
  TWS_HOST               — default 127.0.0.1
  TWS_PORT               — default 7497 (paper); 7496 = live
  TWS_CLIENT_ID          — default 10

Run standalone:
    python3 discord_bot.py
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import traceback
from datetime import datetime, time as dtime
from typing import Optional

import discord
from discord.ext import tasks
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("discord_bot")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

COMMAND_PREFIX = "!"
MAX_MESSAGE_LEN = 1900

# Daily summary fires at this local time (HH:MM, 24h)
_summary_time_str = os.getenv("DAILY_SUMMARY_TIME", "16:30")
try:
    _h, _m = _summary_time_str.split(":")
    DAILY_SUMMARY_TIME = dtime(int(_h), int(_m))
except Exception:
    DAILY_SUMMARY_TIME = dtime(16, 30)

# Earnings alerts fire at this local time (default 09:00, before market open)
_alert_time_str = os.getenv("EARNINGS_ALERT_TIME", "09:00")
try:
    _ah, _am = _alert_time_str.split(":")
    EARNINGS_ALERT_TIME = dtime(int(_ah), int(_am))
except Exception:
    EARNINGS_ALERT_TIME = dtime(9, 0)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_pct(v, decimals=1) -> str:
    if v is None:
        return "N/A"
    return f"{v * 100:.{decimals}f}%"

def _fmt_num(v, decimals=2) -> str:
    if v is None:
        return "N/A"
    try:
        return f"{v:,.{decimals}f}"
    except Exception:
        return str(v)

def _fmt_large(v) -> str:
    if v is None:
        return "N/A"
    try:
        v = float(v)
        if v >= 1e12:
            return f"${v/1e12:.2f}T"
        if v >= 1e9:
            return f"${v/1e9:.2f}B"
        if v >= 1e6:
            return f"${v/1e6:.2f}M"
        return f"${v:,.0f}"
    except Exception:
        return str(v)

def _chunk(text: str) -> list[str]:
    chunks = []
    while len(text) > MAX_MESSAGE_LEN:
        split_at = text.rfind("\n", 0, MAX_MESSAGE_LEN)
        if split_at == -1:
            split_at = MAX_MESSAGE_LEN
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")
    if text:
        chunks.append(text)
    return chunks


# ---------------------------------------------------------------------------
# Stock analysis commands
# ---------------------------------------------------------------------------

def _bust_cache(ticker: str) -> None:
    """Invalidate all cache layers so Discord commands always get live data.

    Four layers cleared:
      1. Our SQLite daily cache (data_fetcher info/history keyed by date)
      2. Our SQLite TTL cache (alt data, forensic, inst risk)
      3. yfinance YfData.cache_get() lru_cache — HTTP response cache in-process
      4. yfinance Ticker._quote._info / _already_fetched — per-ticker info cache
    """
    import cache as _cache_mod
    _cache_mod.invalidate(ticker)
    _cache_mod.invalidate_ttl(ticker)

    try:
        import yfinance.data as _yfdata
        singleton = _yfdata.YfData._instances.get(_yfdata.YfData)
        if singleton is not None and hasattr(singleton, 'cache_get'):
            singleton.cache_get.cache_clear()
    except Exception:
        pass


def _run_price(ticker: str) -> str:
    import yfinance as yf
    from data_fetcher import StockDataFetcher
    _bust_cache(ticker)
    f = StockDataFetcher(ticker, force_live=True)
    if not f.validate():
        return f"❌ Unknown ticker **{ticker}**."
    # Live price via fast_info for lowest latency
    try:
        fi = yf.Ticker(ticker).fast_info
        price = float(fi.last_price)
        prev_close = float(fi.previous_close)
        pct = (price - prev_close) / prev_close * 100
        arrow = "📈" if pct >= 0 else "📉"
        change_str = f"  {arrow} {pct:+.2f}% vs prev close"
    except Exception:
        price = f.get_current_price()
        hist = f.get_history(period="5d", interval="1d")
        change_str = ""
        if hist is not None and len(hist) >= 2:
            prev = float(hist["Close"].iloc[-2])
            curr = float(hist["Close"].iloc[-1])
            pct = (curr - prev) / prev * 100
            arrow = "📈" if pct >= 0 else "📉"
            change_str = f"  {arrow} {pct:+.2f}% vs yesterday"
    name = f.get_company_name()
    sector = f.get_sector()
    mktcap = _fmt_large(f.get_market_cap())
    from datetime import datetime
    ts = datetime.now().strftime("%H:%M:%S")
    return (
        f"**{ticker} — {name}**\n"
        f"💲 Price: **${price:.2f}**{change_str}\n"
        f"🏭 Sector: {sector}  |  Market Cap: {mktcap}\n"
        f"🕐 Live as of {ts}"
    )


def _run_dcf(ticker: str) -> str:
    from data_fetcher import StockDataFetcher
    from dcf_model import DCFValuator
    _bust_cache(ticker)
    f = StockDataFetcher(ticker, force_live=True)
    if not f.validate():
        return f"❌ Unknown ticker **{ticker}**."
    result = DCFValuator().compute(
        info=f.info,
        income_stmt=f.get_income_statement(),
        balance_sheet=f.get_balance_sheet(),
        cashflow=f.get_cashflow_statement(),
    )
    if result is None or result.intrinsic_value is None:
        return f"⚠️ Could not compute DCF for **{ticker}** — insufficient data."
    iv = result.intrinsic_value
    cp = result.current_price or 0
    mos = result.margin_of_safety or 0
    upside = result.upside_pct or 0
    label = result.valuation_label or "Unknown"
    emoji = "🟢" if mos > 10 else "🔴" if mos < -10 else "🟡"
    lines = [
        f"**{ticker} — DCF Intrinsic Value**",
        f"{emoji} **{label}**",
        f"  Intrinsic Value : **${iv:.2f}**",
        f"  Current Price   : ${cp:.2f}",
        f"  Margin of Safety: {mos:+.1f}%",
        f"  Upside / (Down) : {upside:+.1f}%",
    ]
    if result.sensitivity:
        lines.append("\n📊 Sensitivity (Base WACC, Growth ±):")
        added = 0
        for (wd, gd), iv_val in result.sensitivity.items():
            if wd == 0 and iv_val is not None:
                lines.append(f"  {gd:+.1f}% growth: ${iv_val:.2f}")
                added += 1
            if added >= 3:
                break
    return "\n".join(lines)


def _run_score(ticker: str) -> str:
    from data_fetcher import StockDataFetcher
    from quality_scores import QualityScorer
    _bust_cache(ticker)
    f = StockDataFetcher(ticker, force_live=True)
    if not f.validate():
        return f"❌ Unknown ticker **{ticker}**."
    qs = QualityScorer().compute(
        info=f.info,
        income_stmt=f.get_income_statement(),
        balance_sheet=f.get_balance_sheet(),
        cashflow=f.get_cashflow_statement(),
    )
    lines = [f"**{ticker} — Quality Scores**"]
    if qs.piotroski_f is not None:
        bar = "🟢" * qs.piotroski_f + "⬜" * (9 - qs.piotroski_f)
        lines.append(f"\n🏆 Piotroski F-Score: **{qs.piotroski_f}/9** {bar}")
        lines.append(f"   Label: {qs.piotroski_label}")
    else:
        lines.append("\n🏆 Piotroski F-Score: N/A")
    if qs.altman_z is not None:
        zone_emoji = {"Safe": "🟢", "Grey": "🟡", "Distress": "🔴"}.get(qs.altman_zone, "⚪")
        lines.append(f"\n🧮 Altman Z-Score: **{qs.altman_z:.2f}** {zone_emoji} {qs.altman_zone or ''}")
    else:
        lines.append("\n🧮 Altman Z-Score: N/A")
    if qs.beneish_m is not None:
        flag_emoji = "🚨" if qs.beneish_flag else "✅"
        flag_text = "Possible manipulation" if qs.beneish_flag else "Clean"
        lines.append(f"\n🔍 Beneish M-Score: **{qs.beneish_m:.2f}** {flag_emoji} {flag_text}")
    else:
        lines.append("\n🔍 Beneish M-Score: N/A")
    return "\n".join(lines)


def _run_analyze(ticker: str) -> str:
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from data_fetcher import StockDataFetcher
    from fundamental_analysis import FundamentalAnalyzer
    from technical_analysis import TechnicalAnalyzer
    from recommendation_engine import HolisticRecommendation
    from risk_management import RiskManager
    from reasoning_engine import ReasoningEngine

    _bust_cache(ticker)
    f = StockDataFetcher(ticker, force_live=True)
    if not f.validate():
        return f"❌ Unknown ticker **{ticker}**."

    try:
        name = f.get_company_name()
        sector = f.get_sector()
        price = f.get_current_price()
        history = f.get_history(period="1y")
        fund = f.get_fundamental_data()

        fund_result = FundamentalAnalyzer().analyze(fund, sector)
        tech_result = TechnicalAnalyzer().analyze(history)
        risk_result = RiskManager().analyze(history, price)

        rec = HolisticRecommendation(
            fundamental=fund_result,
            technical=tech_result,
            fundamental_weight=0.5,
            technical_weight=0.5,
        )
        reasoning = ReasoningEngine(
            ticker=ticker,
            fundamental_result=fund_result,
            technical_result=tech_result,
            risk_result=risk_result,
            recommendation=rec,
            news_result=None,
            governance_result=None,
        ).synthesize()

        score = rec.overall_score
        verdict = rec.recommendation
        verdict_emoji = {
            "STRONG BUY": "🟢🟢", "BUY": "🟢", "HOLD": "🟡",
            "SELL": "🔴", "STRONG SELL": "🔴🔴",
        }.get(verdict, "⚪")

        pe = fund.get("pe_trailing")
        roe = fund.get("roe")
        de = fund.get("debt_to_equity")
        rev_growth = fund.get("revenue_growth")
        net_margin = fund.get("net_margin")
        mktcap = _fmt_large(fund.get("market_cap"))

        from datetime import datetime
        ts = datetime.now().strftime("%H:%M:%S")

        lines = [
            f"**{ticker} — {name}** ({sector})",
            f"💲 ${price:.2f}  |  Market Cap: {mktcap}  |  🕐 {ts}",
            "",
            f"## {verdict_emoji} Verdict: **{verdict}** (Score: {score:.0f}/100)",
            "",
            "**Key Metrics**",
            f"  P/E: {_fmt_num(pe)} | ROE: {_fmt_pct(roe)} | D/E: {_fmt_num(de)} | Net Margin: {_fmt_pct(net_margin)} | Rev Growth: {_fmt_pct(rev_growth)}",
            "",
        ]
        if tech_result:
            sig = getattr(tech_result, "signal", None) or "N/A"
            lines.append(f"**Technical Signal:** {sig}")
        if risk_result:
            sharpe = getattr(risk_result, "sharpe_approx", None)
            max_dd = getattr(risk_result, "max_drawdown", None)
            lines.append(f"**Risk:** Sharpe {_fmt_num(sharpe)} | Max DD {_fmt_num(max_dd)}%")
        if reasoning:
            thesis = getattr(reasoning, "thesis", None)
            bull = getattr(reasoning, "bull_case", None)
            bear = getattr(reasoning, "bear_case", None)
            if thesis:
                lines.append(f"\n**Investment Thesis:**\n{thesis}")
            if bull:
                lines.append(f"\n🟢 **Bull Case:**\n{bull}")
            if bear:
                lines.append(f"\n🔴 **Bear Case:**\n{bear}")
        lines.append("\n_Not financial advice. Live data fetched at time of request._")
        return "\n".join(lines)

    except Exception as e:
        logger.error("Analysis failed for %s: %s", ticker, traceback.format_exc())
        return f"⚠️ Analysis error for **{ticker}**: {e}"


# ---------------------------------------------------------------------------
# TWS portfolio command
# ---------------------------------------------------------------------------

def _run_portfolio() -> list[str]:
    """Fetch TWS portfolio and return list of message chunks."""
    from tws_portfolio import fetch_portfolio_summary, format_discord_summary, build_ai_commentary
    summary = fetch_portfolio_summary()
    commentary = build_ai_commentary(summary)
    return format_discord_summary(summary, commentary)


# ---------------------------------------------------------------------------
# Earnings calendar command + alert runner
# ---------------------------------------------------------------------------

def _run_earnings_calendar() -> str:
    """Return upcoming earnings for all portfolio tickers (next 30 days)."""
    from earnings_alerts import get_upcoming_calendar
    entries = get_upcoming_calendar()
    if not entries:
        return "📅 No earnings found for portfolio tickers in the next 30 days."

    timing_emoji = {"BMO": "🌅", "AMC": "🌙", "unconfirmed": "🕐"}
    urgency_emoji = {0: "🚨", 1: "🚨", 2: "⚠️", 3: "⚠️", 4: "📅", 5: "📅", 6: "📅", 7: "📅"}

    lines = ["**📅 Upcoming Earnings — Portfolio (next 30 days)**", ""]
    for ticker, ed, timing, days in entries:
        te = timing_emoji.get(timing, "🕐")
        ue = urgency_emoji.get(days, "📅")
        day_str = "TODAY" if days == 0 else f"in {days}d"
        lines.append(f"{ue} **{ticker}** — {ed.strftime('%b %d')} ({day_str})  {te} {timing}")

    lines.append("\n_Dates from Yahoo Finance. Confirm with IR calendar._")
    return "\n".join(lines)


def _run_earnings_alerts() -> list[str]:
    """Check for due earnings alerts (7/2/1-day thresholds) and return formatted messages."""
    from earnings_alerts import check_due_alerts, format_alert, _chunk as _ea_chunk
    alerts = check_due_alerts()
    if not alerts:
        return []
    messages = []
    for alert in alerts:
        messages.append(format_alert(alert))
    return messages


# ---------------------------------------------------------------------------
# Discord Bot
# ---------------------------------------------------------------------------

class FinancialBot(discord.Client):
    def __init__(self, allowed_channel_id: Optional[int] = None, **kwargs):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents, **kwargs)
        self.allowed_channel_id = allowed_channel_id
        self._last_summary_date: Optional[str] = None
        self._last_alerts_date: Optional[str] = None

    async def setup_hook(self):
        self._daily_summary_loop.start()
        self._earnings_alert_loop.start()

    async def on_ready(self):
        logger.info("Discord bot logged in as %s (id=%s)", self.user, self.user.id)
        ch = f"channel {self.allowed_channel_id}" if self.allowed_channel_id else "all channels"
        logger.info(
            "Listening on %s | Daily summary at %s | Earnings alerts at %s",
            ch, DAILY_SUMMARY_TIME.strftime("%H:%M"), EARNINGS_ALERT_TIME.strftime("%H:%M"),
        )

    @tasks.loop(minutes=1)
    async def _daily_summary_loop(self):
        """Post portfolio summary once per day at DAILY_SUMMARY_TIME."""
        now = datetime.now()
        today_str = now.strftime("%Y-%m-%d")
        if (
            now.hour == DAILY_SUMMARY_TIME.hour
            and now.minute == DAILY_SUMMARY_TIME.minute
            and self._last_summary_date != today_str
        ):
            self._last_summary_date = today_str
            channel = self._get_target_channel()
            if channel is None:
                logger.warning("Daily summary: no target channel found")
                return
            logger.info("Posting daily portfolio summary to channel %s", channel.id)
            await self._send_portfolio(channel)

    @tasks.loop(minutes=1)
    async def _earnings_alert_loop(self):
        """Check for due earnings alerts once per day at EARNINGS_ALERT_TIME.

        Fires at 7-day, 2-day, and 1-day thresholds before earnings date.
        Each alert includes: date/timing, EPS/revenue estimates, options-implied
        expected move, beat/miss history, key metrics, and AI briefing.
        """
        now = datetime.now()
        today_str = now.strftime("%Y-%m-%d")
        if (
            now.hour == EARNINGS_ALERT_TIME.hour
            and now.minute == EARNINGS_ALERT_TIME.minute
            and self._last_alerts_date != today_str
        ):
            self._last_alerts_date = today_str
            channel = self._get_target_channel()
            if channel is None:
                logger.warning("Earnings alerts: no target channel found")
                return
            logger.info("Running earnings alert scan at %s", now.strftime("%H:%M"))
            loop = asyncio.get_event_loop()
            try:
                messages = await loop.run_in_executor(None, _run_earnings_alerts)
            except Exception as e:
                logger.error("Earnings alert scan failed: %s", traceback.format_exc())
                messages = [f"⚠️ Earnings alert error: {e}"]
            if messages:
                for msg in messages:
                    for chunk in _chunk(msg):
                        await channel.send(chunk)
            else:
                logger.info("Earnings alerts: no alerts due today")

    @_daily_summary_loop.before_loop
    async def _before_loop(self):
        await self.wait_until_ready()

    @_earnings_alert_loop.before_loop
    async def _before_alerts_loop(self):
        await self.wait_until_ready()

    def _get_target_channel(self) -> Optional[discord.TextChannel]:
        if self.allowed_channel_id:
            ch = self.get_channel(self.allowed_channel_id)
            if ch:
                return ch
        # Fall back to first text channel in first guild
        for guild in self.guilds:
            for ch in guild.text_channels:
                if ch.permissions_for(guild.me).send_messages:
                    return ch
        return None

    async def _send_portfolio(self, channel: discord.TextChannel):
        loop = asyncio.get_event_loop()
        try:
            chunks = await loop.run_in_executor(None, _run_portfolio)
        except Exception as e:
            logger.error("Portfolio fetch error: %s", traceback.format_exc())
            chunks = [f"⚠️ Portfolio error: {e}"]
        for chunk in chunks:
            for msg in _chunk(chunk):
                await channel.send(msg)

    async def on_message(self, message: discord.Message):
        if message.author == self.user:
            return
        if self.allowed_channel_id and message.channel.id != self.allowed_channel_id:
            return

        content = message.content.strip()
        if not content.startswith(COMMAND_PREFIX):
            return

        parts = content[len(COMMAND_PREFIX):].split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip().upper() if len(parts) > 1 else ""

        logger.info("Command: %r arg: %r from %s", cmd, arg, message.author)

        if cmd == "help":
            await message.channel.send(
                "**Financial Analyst Bot — Commands**\n"
                "```\n"
                "!analyze  <TICKER>  Full analysis + AI thesis\n"
                "!price    <TICKER>  Current price + day change\n"
                "!dcf      <TICKER>  DCF intrinsic value\n"
                "!score    <TICKER>  Quality scores (Piotroski / Altman / Beneish)\n"
                "!portfolio          Live portfolio summary from TWS\n"
                "!earnings           Upcoming earnings calendar (next 30 days)\n"
                "!earnings <TICKER>  Full earnings alert for any ticker\n"
                "!help               Show this message\n"
                "```\n"
                f"Daily summary: **{DAILY_SUMMARY_TIME.strftime('%H:%M')}**  |  "
                f"Earnings alerts: **{EARNINGS_ALERT_TIME.strftime('%H:%M')}** (7d/2d/1d before).\n"
                "Example: `!analyze AAPL`"
            )
            return

        if cmd == "portfolio":
            async with message.channel.typing():
                await self._send_portfolio(message.channel)
            return

        if cmd == "earnings":
            # !earnings          → upcoming calendar for portfolio (next 30 days)
            # !earnings TICKER   → on-demand alert for a specific ticker
            async with message.channel.typing():
                loop = asyncio.get_event_loop()
                if arg:
                    # On-demand alert for a specific ticker
                    def _single_alert(ticker):
                        from earnings_alerts import _get_earnings_date, build_alert, format_alert
                        ed, timing = _get_earnings_date(ticker)
                        if ed is None:
                            return f"📅 No upcoming earnings found for **{ticker}**."
                        from datetime import date
                        days = (ed - date.today()).days
                        alert = build_alert(ticker, ed, timing)
                        return format_alert(alert)
                    try:
                        reply = await loop.run_in_executor(None, _single_alert, arg)
                    except Exception as e:
                        logger.error("Earnings alert error: %s", traceback.format_exc())
                        reply = f"⚠️ Earnings alert error for **{arg}**: {e}"
                    for chunk in _chunk(reply):
                        await message.channel.send(chunk)
                else:
                    # Portfolio calendar
                    try:
                        reply = await loop.run_in_executor(None, _run_earnings_calendar)
                    except Exception as e:
                        logger.error("Earnings calendar error: %s", traceback.format_exc())
                        reply = f"⚠️ Earnings calendar error: {e}"
                    for chunk in _chunk(reply):
                        await message.channel.send(chunk)
            return

        if cmd in ("analyze", "price", "dcf", "score"):
            if not arg:
                await message.channel.send(f"⚠️ Usage: `!{cmd} <TICKER>`  e.g. `!{cmd} AAPL`")
                return

            async with message.channel.typing():
                loop = asyncio.get_event_loop()
                try:
                    handler = {
                        "analyze": _run_analyze,
                        "price": _run_price,
                        "dcf": _run_dcf,
                        "score": _run_score,
                    }[cmd]
                    reply = await loop.run_in_executor(None, handler, arg)
                except Exception as e:
                    logger.error("Unhandled error: %s", traceback.format_exc())
                    reply = f"⚠️ Unexpected error: {e}"

            for chunk in _chunk(reply):
                await message.channel.send(chunk)
            return

        await message.channel.send(
            f"Unknown command `!{cmd}`. Type `!help` to see available commands."
        )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def run_bot(token: str, channel_id: Optional[int] = None):
    client = FinancialBot(allowed_channel_id=channel_id)
    client.run(token, log_handler=None)


if __name__ == "__main__":
    token = os.getenv("DISCORD_BOT_TOKEN", "").strip()
    if not token:
        print("ERROR: DISCORD_BOT_TOKEN not set in environment or .env file")
        sys.exit(1)
    channel_id_str = os.getenv("DISCORD_CHANNEL_ID", "").strip()
    channel_id = int(channel_id_str) if channel_id_str.isdigit() else None
    run_bot(token, channel_id)
