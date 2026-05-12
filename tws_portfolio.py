"""Fetch daily portfolio summary from Interactive Brokers TWS / IB Gateway.

Uses ib_insync for clean async access. Retrieves:
  - Account summary (NLV, buying power, margin)
  - All open positions with P&L
  - Daily P&L totals

Requires TWS or IB Gateway running locally with API enabled:
  TWS: Edit → Global Configuration → API → Settings → Enable ActiveX and Socket Clients
  IB Gateway: same setting on startup screen.

Default port: 7497 (TWS paper), 7496 (TWS live), 4002 (Gateway paper), 4001 (Gateway live)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("tws_portfolio")

TWS_HOST = os.getenv("TWS_HOST", "127.0.0.1")
TWS_PORT = int(os.getenv("TWS_PORT", "7497"))   # 7497 = paper, 7496 = live
TWS_CLIENT_ID = int(os.getenv("TWS_CLIENT_ID", "10"))
TWS_TIMEOUT = 15  # seconds to wait for TWS responses


@dataclass
class PositionRow:
    symbol: str
    sec_type: str
    exchange: str
    currency: str
    quantity: float
    avg_cost: float
    market_price: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float
    daily_pnl: float = 0.0
    pct_change_today: float = 0.0


@dataclass
class AccountSummary:
    net_liquidation: float = 0.0
    total_cash: float = 0.0
    buying_power: float = 0.0
    gross_position_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    daily_pnl: float = 0.0
    currency: str = "USD"


@dataclass
class PortfolioSummary:
    account: AccountSummary = field(default_factory=AccountSummary)
    positions: list[PositionRow] = field(default_factory=list)
    error: Optional[str] = None


def fetch_portfolio_summary() -> PortfolioSummary:
    """Connect to TWS, pull account + position data, disconnect, return result."""
    try:
        from ib_insync import IB, util
        util.logToConsole(logging.WARNING)
    except ImportError:
        return PortfolioSummary(error="ib_insync not installed. Run: pip install ib_insync")

    ib = IB()
    result = PortfolioSummary()

    try:
        ib.connect(TWS_HOST, TWS_PORT, clientId=TWS_CLIENT_ID, timeout=TWS_TIMEOUT)
        logger.info("Connected to TWS at %s:%s", TWS_HOST, TWS_PORT)
    except Exception as e:
        result.error = (
            f"Cannot connect to TWS at {TWS_HOST}:{TWS_PORT}. "
            f"Make sure TWS/IB Gateway is running and API is enabled. ({e})"
        )
        return result

    try:
        # --- Account summary ---
        acct_tags = [
            "NetLiquidation", "TotalCashValue", "BuyingPower",
            "GrossPositionValue", "UnrealizedPnL", "RealizedPnL",
        ]
        acct_vals = ib.accountSummary()
        acct_map: dict[str, float] = {}
        currency = "USD"
        for av in acct_vals:
            if av.tag in acct_tags:
                try:
                    acct_map[av.tag] = float(av.value)
                    if av.currency:
                        currency = av.currency
                except (ValueError, TypeError):
                    pass

        result.account = AccountSummary(
            net_liquidation=acct_map.get("NetLiquidation", 0),
            total_cash=acct_map.get("TotalCashValue", 0),
            buying_power=acct_map.get("BuyingPower", 0),
            gross_position_value=acct_map.get("GrossPositionValue", 0),
            unrealized_pnl=acct_map.get("UnrealizedPnL", 0),
            realized_pnl=acct_map.get("RealizedPnL", 0),
            currency=currency,
        )

        # --- Daily P&L ---
        try:
            pnl_obj = ib.reqPnL(ib.managedAccounts()[0])
            ib.sleep(1)
            result.account.daily_pnl = float(pnl_obj.dailyPnL or 0)
        except Exception:
            pass

        # --- Positions ---
        positions = ib.positions()
        if positions:
            # Request P&L for each position
            account_id = ib.managedAccounts()[0] if ib.managedAccounts() else ""
            pnl_single_map: dict[int, object] = {}
            for pos in positions:
                try:
                    conId = pos.contract.conId
                    pnl_s = ib.reqPnLSingle(account_id, "", conId)
                    pnl_single_map[conId] = pnl_s
                except Exception:
                    pass

            if pnl_single_map:
                ib.sleep(1.5)

            # Get current market prices via snapshot
            contracts = [pos.contract for pos in positions]
            try:
                tickers = ib.reqTickers(*contracts)
                price_map = {t.contract.conId: t.marketPrice() for t in tickers if t.contract}
            except Exception:
                price_map = {}

            for pos in positions:
                conId = pos.contract.conId
                qty = float(pos.position)
                avg = float(pos.avgCost)
                mkt_price = price_map.get(conId) or 0.0
                if mkt_price <= 0:
                    mkt_price = avg  # fallback
                mkt_value = qty * mkt_price
                unrealized = (mkt_price - avg) * qty

                pnl_s = pnl_single_map.get(conId)
                realized = float(getattr(pnl_s, "realizedPnL", 0) or 0)
                daily = float(getattr(pnl_s, "dailyPnL", 0) or 0)
                pct_today = (daily / abs(mkt_value) * 100) if abs(mkt_value) > 0 else 0.0

                result.positions.append(PositionRow(
                    symbol=pos.contract.symbol,
                    sec_type=pos.contract.secType,
                    exchange=pos.contract.exchange or "",
                    currency=pos.contract.currency or "",
                    quantity=qty,
                    avg_cost=avg,
                    market_price=mkt_price,
                    market_value=mkt_value,
                    unrealized_pnl=unrealized,
                    realized_pnl=realized,
                    daily_pnl=daily,
                    pct_change_today=pct_today,
                ))

        # Sort by abs(market_value) descending
        result.positions.sort(key=lambda p: abs(p.market_value), reverse=True)

    except Exception as e:
        logger.error("Error fetching portfolio: %s", e, exc_info=True)
        result.error = str(e)
    finally:
        try:
            ib.disconnect()
        except Exception:
            pass

    return result


def format_discord_summary(summary: PortfolioSummary, ai_commentary: str = "") -> list[str]:
    """Format portfolio summary as Discord message chunks."""
    from datetime import datetime
    now = datetime.now().strftime("%b %d, %Y %H:%M")

    if summary.error:
        return [f"⚠️ **TWS Portfolio Error**\n{summary.error}"]

    a = summary.account
    cur = a.currency

    # Emoji for daily P&L
    def pnl_emoji(v: float) -> str:
        return "📈" if v > 0 else "📉" if v < 0 else "➡️"

    def fmt(v: float, prefix="$", decimals=2) -> str:
        sign = "+" if v > 0 else ""
        return f"{sign}{prefix}{v:,.{decimals}f}"

    lines = [
        f"## 📊 Daily Portfolio Summary — {now}",
        "",
        "**Account Overview**",
        f"```",
        f"Net Liquidation   : {cur} {a.net_liquidation:>14,.2f}",
        f"Gross Position    : {cur} {a.gross_position_value:>14,.2f}",
        f"Total Cash        : {cur} {a.total_cash:>14,.2f}",
        f"Buying Power      : {cur} {a.buying_power:>14,.2f}",
        f"```",
        "",
        "**P&L Summary**",
        f"{pnl_emoji(a.daily_pnl)} Daily P&L       : **{fmt(a.daily_pnl)}**",
        f"{pnl_emoji(a.unrealized_pnl)} Unrealized P&L  : {fmt(a.unrealized_pnl)}",
        f"{pnl_emoji(a.realized_pnl)} Realized P&L    : {fmt(a.realized_pnl)}",
    ]

    msg1 = "\n".join(lines)

    # Positions table
    if summary.positions:
        pos_lines = ["\n**Open Positions**", "```"]
        header = f"{'Symbol':<8} {'Qty':>8} {'Price':>10} {'Mkt Val':>12} {'Day P&L':>10} {'Day%':>7} {'Unreal':>12}"
        pos_lines.append(header)
        pos_lines.append("-" * len(header))
        for p in summary.positions:
            sign_d = "+" if p.daily_pnl >= 0 else ""
            sign_u = "+" if p.unrealized_pnl >= 0 else ""
            pos_lines.append(
                f"{p.symbol:<8} {p.quantity:>8,.0f} {p.market_price:>10,.2f} "
                f"{p.market_value:>12,.2f} {sign_d}{p.daily_pnl:>9,.2f} "
                f"{p.pct_change_today:>+6.1f}% {sign_u}{p.unrealized_pnl:>11,.2f}"
            )
        pos_lines.append("```")
        msg2 = "\n".join(pos_lines)
    else:
        msg2 = "\n_No open positions._"

    chunks = [msg1, msg2]

    if ai_commentary:
        chunks.append(f"\n🤖 **AI Commentary**\n{ai_commentary}")

    chunks.append("\n_Data from Interactive Brokers TWS. Not financial advice._")
    return chunks


def build_ai_commentary(summary: PortfolioSummary) -> str:
    """Ask Claude for a brief commentary on today's portfolio moves."""
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key or summary.error:
        return ""

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        a = summary.account
        pos_lines = []
        for p in summary.positions[:10]:  # cap at 10 for prompt size
            pos_lines.append(
                f"  {p.symbol}: qty={p.quantity:.0f}, mkt_val=${p.market_value:,.0f}, "
                f"day_pnl=${p.daily_pnl:,.0f} ({p.pct_change_today:+.1f}%), "
                f"unrealized=${p.unrealized_pnl:,.0f}"
            )

        prompt = f"""Portfolio daily summary:
Net Liquidation: ${a.net_liquidation:,.0f}
Daily P&L: ${a.daily_pnl:,.0f}
Unrealized P&L: ${a.unrealized_pnl:,.0f}

Positions:
{chr(10).join(pos_lines)}

Write a concise 3-5 sentence commentary on today's portfolio performance.
Highlight the biggest movers, any concentration risks, and one actionable observation.
Be direct and professional. No disclaimers."""

        response = client.messages.create(
            model="claude-opus-4-7",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()
    except Exception as e:
        logger.warning("Claude commentary failed: %s", e)
        return ""
