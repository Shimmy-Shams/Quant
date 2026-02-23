"""
Earnings Calendar — Tier 2 News Event Integration

Fetches and caches earnings announcement dates so the strategy can:
  1. Avoid entering positions within N days of an earnings report
  2. Exit existing positions before earnings if configured

Data sources (tried in order):
  - yfinance .get_earnings_dates()  (free, reliable for forward-looking)
  - Fallback: Alpha Vantage EARNINGS_CALENDAR (free tier: 25 req/day)

Cache strategy:
  - Earnings dates are cached to a local JSON file (data/snapshots/earnings_cache.json)
  - Cache is keyed by symbol and refreshed if the last fetch is > 7 days old
  - For backtesting, we pre-build a historical earnings map from yfinance quarterly
    earnings data, keyed by (symbol, date) for O(1) lookups

Usage:
    from data.earnings_calendar import EarningsCalendar

    cal = EarningsCalendar(project_root)
    # Live: is there earnings within 2 days?
    if cal.has_upcoming_earnings("AAPL", within_days=2):
        skip_entry()

    # Backtest: build a date→set map for fast lookups
    blackout = cal.build_backtest_blackout(symbols, blackout_days=2)
    if date in blackout and symbol in blackout[date]:
        skip_entry()
"""

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd

logger = logging.getLogger("earnings_calendar")


class EarningsCalendar:
    """Manages earnings date lookups for both live trading and backtesting."""

    def __init__(self, project_root: Path, cache_max_age_days: int = 7):
        self.project_root = project_root
        self.cache_file = project_root / "data" / "snapshots" / "earnings_cache.json"
        self.historical_cache_file = (
            project_root / "data" / "snapshots" / "earnings_historical_cache.json"
        )
        self.cache_max_age_days = cache_max_age_days
        self._cache: Dict = {}
        self._historical_cache: Dict[str, List[str]] = {}
        self._load_cache()
        self._load_historical_cache()

    # ─── Cache I/O ─────────────────────────────────────────────────────

    def _load_cache(self) -> None:
        """Load cached earnings dates from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r") as f:
                    self._cache = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load earnings cache: {e}")
                self._cache = {}

    def _save_cache(self) -> None:
        """Persist cache to disk."""
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.cache_file, "w") as f:
                json.dump(self._cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save earnings cache: {e}")

    def _load_historical_cache(self) -> None:
        """Load historical earnings cache (for backtesting — stable data)."""
        if self.historical_cache_file.exists():
            try:
                with open(self.historical_cache_file, "r") as f:
                    self._historical_cache = json.load(f)
                logger.info(
                    f"Loaded historical earnings cache: {len(self._historical_cache)} symbols"
                )
            except Exception as e:
                logger.warning(f"Could not load historical earnings cache: {e}")
                self._historical_cache = {}

    def _save_historical_cache(self) -> None:
        """Persist historical earnings cache to disk."""
        self.historical_cache_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.historical_cache_file, "w") as f:
                json.dump(self._historical_cache, f, indent=2)
            logger.info(
                f"Saved historical earnings cache: {len(self._historical_cache)} symbols"
            )
        except Exception as e:
            logger.warning(f"Could not save historical earnings cache: {e}")

    def _is_stale(self, symbol: str) -> bool:
        """Check if a symbol's cache entry is stale."""
        entry = self._cache.get(symbol)
        if not entry or "fetched_at" not in entry:
            return True
        fetched = datetime.fromisoformat(entry["fetched_at"])
        return (datetime.now() - fetched).days > self.cache_max_age_days

    # ─── Live Fetching ─────────────────────────────────────────────────

    def fetch_upcoming_earnings(self, symbol: str) -> Optional[List[str]]:
        """
        Fetch upcoming earnings dates for a symbol using yfinance.

        Returns:
            List of date strings (YYYY-MM-DD) for upcoming earnings, or None on error.
        """
        try:
            import yfinance as yf

            t = yf.Ticker(symbol)
            # get_earnings_dates returns both past and future dates
            ed = t.get_earnings_dates(limit=8)
            if ed is None or ed.empty:
                return []

            # Filter to future dates only
            now = pd.Timestamp.now(tz="America/New_York").normalize()
            future_dates = []
            for dt in ed.index:
                d = pd.Timestamp(dt)
                if d.tz is not None:
                    d = d.tz_convert("America/New_York").normalize()
                else:
                    d = d.normalize()
                if d >= now - pd.Timedelta(days=1):
                    future_dates.append(d.strftime("%Y-%m-%d"))

            # Update cache
            self._cache[symbol] = {
                "upcoming": future_dates,
                "fetched_at": datetime.now().isoformat(),
            }
            self._save_cache()
            return future_dates

        except Exception as e:
            logger.warning(f"Could not fetch earnings for {symbol}: {e}")
            return None

    def has_upcoming_earnings(
        self, symbol: str, within_days: int = 2, reference_date: Optional[str] = None
    ) -> bool:
        """
        Check if a symbol has earnings within N trading days.

        Args:
            symbol: Stock ticker
            within_days: Number of calendar days to look ahead
            reference_date: Check relative to this date (YYYY-MM-DD).
                            Defaults to today.

        Returns:
            True if earnings are within the window.
        """
        # Refresh cache if stale
        if self._is_stale(symbol):
            self.fetch_upcoming_earnings(symbol)

        entry = self._cache.get(symbol, {})
        upcoming = entry.get("upcoming", [])
        if not upcoming:
            return False

        ref = (
            pd.Timestamp(reference_date)
            if reference_date
            else pd.Timestamp.now().normalize()
        )
        window_end = ref + pd.Timedelta(days=within_days)

        for date_str in upcoming:
            d = pd.Timestamp(date_str)
            if ref - pd.Timedelta(days=1) <= d <= window_end:
                return True
        return False

    def refresh_batch(self, symbols: List[str], delay: float = 0.3) -> int:
        """
        Refresh earnings dates for multiple symbols.

        Args:
            symbols: List of tickers to refresh.
            delay: Seconds between API calls (rate limiting).

        Returns:
            Number of symbols successfully refreshed.
        """
        refreshed = 0
        for sym in symbols:
            if self._is_stale(sym):
                result = self.fetch_upcoming_earnings(sym)
                if result is not None:
                    refreshed += 1
                time.sleep(delay)
        if refreshed:
            logger.info(f"Refreshed earnings dates for {refreshed}/{len(symbols)} symbols")
        return refreshed

    # ─── Backtest Support ──────────────────────────────────────────────

    def build_backtest_blackout(
        self,
        symbols: List[str],
        price_dates: pd.DatetimeIndex,
        blackout_days: int = 2,
    ) -> Dict[pd.Timestamp, Set[str]]:
        """
        Build a date → set-of-symbols blackout map for backtesting.

        For each historical earnings date, marks [earnings_date - blackout_days, earnings_date]
        as blackout for that symbol (no new entries allowed).

        Uses yfinance quarterly earnings data for historical dates.

        Args:
            symbols: List of tickers in the backtest universe.
            price_dates: DatetimeIndex of the backtest period.
            blackout_days: Number of calendar days before earnings to block.

        Returns:
            Dict mapping each date to the set of symbols in blackout.
        """
        blackout: Dict[pd.Timestamp, Set[str]] = {}
        start_date = price_dates[0]
        end_date = price_dates[-1]

        # Also check for exit-before-earnings: we mark [earn-blackout, earn] as blackout
        # The entry loop skips symbols in blackout; the exit loop can force-exit.

        logger.info(
            f"Building earnings blackout map for {len(symbols)} symbols "
            f"({start_date.date()} → {end_date.date()}, blackout={blackout_days}d)"
        )

        fetched = 0
        cache_hits = 0
        fetched_fresh = 0
        for i, sym in enumerate(symbols):
            # Check persistent historical cache first
            if sym in self._historical_cache:
                earnings_dates = self._historical_cache[sym]
                cache_hits += 1
            else:
                earnings_dates = self._fetch_historical_earnings(sym)
                if earnings_dates:
                    self._historical_cache[sym] = earnings_dates
                fetched_fresh += 1
                # Rate-limit API calls (0.35s between fresh fetches)
                if fetched_fresh % 10 == 0:
                    time.sleep(0.5)
                elif fetched_fresh > 0:
                    time.sleep(0.35)
                # Progress logging every 50 symbols
                if fetched_fresh % 50 == 0 and fetched_fresh > 0:
                    logger.info(f"  ... fetched {fetched_fresh}/{len(symbols)} symbols")

            if not earnings_dates:
                continue
            fetched += 1

            for ed in earnings_dates:
                earn_ts = pd.Timestamp(ed)
                if earn_ts < start_date - pd.Timedelta(days=blackout_days):
                    continue
                if earn_ts > end_date:
                    continue

                # Mark blackout window
                for offset in range(blackout_days + 1):
                    blackout_date = earn_ts - pd.Timedelta(days=offset)
                    blackout_date = blackout_date.normalize()
                    if blackout_date not in blackout:
                        blackout[blackout_date] = set()
                    blackout[blackout_date].add(sym)

        # Save historical cache if we fetched any fresh data
        if fetched_fresh > 0:
            self._save_historical_cache()

        logger.info(
            f"Earnings blackout built: {fetched}/{len(symbols)} symbols "
            f"({cache_hits} cached, {fetched_fresh} fresh fetched), "
            f"{len(blackout)} blackout dates"
        )
        return blackout

    def _fetch_historical_earnings(self, symbol: str) -> List[str]:
        """
        Fetch historical earnings dates for a symbol.

        Requires `lxml` package (pip install lxml) for HTML parsing.
        Returns list of date strings (YYYY-MM-DD).
        """
        try:
            import yfinance as yf

            # yfinance expects hyphens not underscores (BRK_B → BRK-B)
            yf_symbol = symbol.replace("_", "-")
            t = yf.Ticker(yf_symbol)

            # Primary: get_earnings_dates (scrapes Yahoo Finance earnings page)
            # Needs lxml installed. limit=80 gives ~100 entries (~25 years)
            try:
                ed = t.get_earnings_dates(limit=80)
                if ed is not None and not ed.empty:
                    dates = []
                    for dt in ed.index:
                        d = pd.Timestamp(dt)
                        if d.tz is not None:
                            d = d.tz_localize(None)
                        dates.append(d.strftime("%Y-%m-%d"))
                    return dates
            except ImportError:
                logger.warning(
                    "lxml not installed — get_earnings_dates() will fail. "
                    "Run: pip install lxml"
                )
            except Exception as e:
                logger.debug(f"{symbol}: get_earnings_dates failed: {e}")

            # Fallback: quarterly_income_stmt column dates
            # These are fiscal quarter-end dates (not announcement dates)
            # so they're offset by ~6 weeks, but better than nothing
            try:
                qi = t.quarterly_income_stmt
                if qi is not None and not qi.empty:
                    dates = []
                    for dt in qi.columns:
                        d = pd.Timestamp(dt)
                        if d.tz is not None:
                            d = d.tz_localize(None)
                        dates.append(d.strftime("%Y-%m-%d"))
                    if dates:
                        logger.debug(f"{symbol}: using quarterly_income_stmt fallback ({len(dates)} dates)")
                        return dates
            except Exception:
                pass

            return []
        except Exception as e:
            logger.debug(f"Could not fetch historical earnings for {symbol}: {e}")
            return []

    # ─── Utility ───────────────────────────────────────────────────────

    def get_next_earnings_date(self, symbol: str) -> Optional[str]:
        """Get the next upcoming earnings date string for a symbol."""
        if self._is_stale(symbol):
            self.fetch_upcoming_earnings(symbol)
        entry = self._cache.get(symbol, {})
        upcoming = entry.get("upcoming", [])
        if not upcoming:
            return None
        now = pd.Timestamp.now().normalize()
        for d in sorted(upcoming):
            if pd.Timestamp(d) >= now:
                return d
        return None
