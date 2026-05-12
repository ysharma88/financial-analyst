"""SQLite-backed cache for yfinance and alternative data.

Two cache tiers:
  - Daily cache  (original): keyed by (ticker, data_type), invalidated on date change.
  - TTL cache    (new):       keyed by (ticker, data_type), invalidated after N seconds.
    Stores a Unix timestamp alongside the payload so each layer can have its own TTL.
    Also exposes get_age_seconds() so the UI can render staleness badges.
"""

from __future__ import annotations

import logging
import os
import pickle
import sqlite3
import time
from datetime import date
from typing import Any, Optional

logger = logging.getLogger(__name__)

_DB_PATH = os.path.join(os.path.dirname(__file__), "cache", "yfinance_cache.db")


def _conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
    con = sqlite3.connect(_DB_PATH, check_same_thread=False)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS cache (
            ticker    TEXT NOT NULL,
            data_type TEXT NOT NULL,
            date_str  TEXT NOT NULL,
            payload   BLOB NOT NULL,
            PRIMARY KEY (ticker, data_type)
        )
        """
    )
    # TTL table — stores Unix timestamp instead of calendar date
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS cache_ttl (
            ticker      TEXT NOT NULL,
            data_type   TEXT NOT NULL,
            fetched_at  REAL NOT NULL,
            payload     BLOB NOT NULL,
            PRIMARY KEY (ticker, data_type)
        )
        """
    )
    con.commit()
    return con


# ---------------------------------------------------------------------------
# Daily cache (original API — unchanged)
# ---------------------------------------------------------------------------

def get(ticker: str, data_type: str) -> Optional[Any]:
    """Return cached value for today, or None if stale/missing."""
    today = date.today().isoformat()
    try:
        con = _conn()
        row = con.execute(
            "SELECT date_str, payload FROM cache WHERE ticker=? AND data_type=?",
            (ticker.upper(), data_type),
        ).fetchone()
        con.close()
        if row and row[0] == today:
            return pickle.loads(row[1])
    except Exception as exc:
        logger.warning("Cache read failed (%s/%s): %s", ticker, data_type, exc)
    return None


def set(ticker: str, data_type: str, value: Any) -> None:
    """Write value into cache with today's date."""
    today = date.today().isoformat()
    try:
        blob = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        con = _conn()
        con.execute(
            """
            INSERT INTO cache (ticker, data_type, date_str, payload)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(ticker, data_type) DO UPDATE SET
                date_str=excluded.date_str,
                payload=excluded.payload
            """,
            (ticker.upper(), data_type, today, blob),
        )
        con.commit()
        con.close()
    except Exception as exc:
        logger.warning("Cache write failed (%s/%s): %s", ticker, data_type, exc)


def invalidate(ticker: str, data_type: Optional[str] = None) -> None:
    """Remove one or all cache entries for a ticker (both tables)."""
    try:
        con = _conn()
        if data_type:
            con.execute("DELETE FROM cache WHERE ticker=? AND data_type=?", (ticker.upper(), data_type))
            con.execute("DELETE FROM cache_ttl WHERE ticker=? AND data_type=?", (ticker.upper(), data_type))
        else:
            con.execute("DELETE FROM cache WHERE ticker=?", (ticker.upper(),))
            con.execute("DELETE FROM cache_ttl WHERE ticker=?", (ticker.upper(),))
        con.commit()
        con.close()
    except Exception as exc:
        logger.warning("Cache invalidate failed (%s): %s", ticker, exc)


def purge_stale() -> int:
    """Delete daily entries older than today. Returns number of rows removed."""
    today = date.today().isoformat()
    try:
        con = _conn()
        cur = con.execute("DELETE FROM cache WHERE date_str < ?", (today,))
        count = cur.rowcount
        con.commit()
        con.close()
        return count
    except Exception as exc:
        logger.warning("Cache purge failed: %s", exc)
        return 0


# ---------------------------------------------------------------------------
# TTL cache (new — per-layer expiry)
# ---------------------------------------------------------------------------

def get_ttl(ticker: str, data_type: str, ttl_seconds: int) -> Optional[Any]:
    """Return cached value if fetched within ttl_seconds, else None."""
    try:
        con = _conn()
        row = con.execute(
            "SELECT fetched_at, payload FROM cache_ttl WHERE ticker=? AND data_type=?",
            (ticker.upper(), data_type),
        ).fetchone()
        con.close()
        if row:
            age = time.time() - row[0]
            if age <= ttl_seconds:
                return pickle.loads(row[1])
    except Exception as exc:
        logger.warning("TTL cache read failed (%s/%s): %s", ticker, data_type, exc)
    return None


def set_ttl(ticker: str, data_type: str, value: Any) -> None:
    """Write value into TTL cache with current Unix timestamp."""
    try:
        blob = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        con = _conn()
        con.execute(
            """
            INSERT INTO cache_ttl (ticker, data_type, fetched_at, payload)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(ticker, data_type) DO UPDATE SET
                fetched_at=excluded.fetched_at,
                payload=excluded.payload
            """,
            (ticker.upper(), data_type, time.time(), blob),
        )
        con.commit()
        con.close()
    except Exception as exc:
        logger.warning("TTL cache write failed (%s/%s): %s", ticker, data_type, exc)


def get_age_seconds(ticker: str, data_type: str) -> Optional[float]:
    """Return seconds since last TTL cache write, or None if no entry."""
    try:
        con = _conn()
        row = con.execute(
            "SELECT fetched_at FROM cache_ttl WHERE ticker=? AND data_type=?",
            (ticker.upper(), data_type),
        ).fetchone()
        con.close()
        if row:
            return time.time() - row[0]
    except Exception as exc:
        logger.warning("Cache age lookup failed (%s/%s): %s", ticker, data_type, exc)
    return None


def invalidate_ttl(ticker: str, data_type: Optional[str] = None) -> None:
    """Force-expire TTL cache entries so the next call re-fetches."""
    try:
        con = _conn()
        if data_type:
            con.execute("DELETE FROM cache_ttl WHERE ticker=? AND data_type=?", (ticker.upper(), data_type))
        else:
            con.execute("DELETE FROM cache_ttl WHERE ticker=?", (ticker.upper(),))
        con.commit()
        con.close()
    except Exception as exc:
        logger.warning("TTL invalidate failed (%s): %s", ticker, exc)
