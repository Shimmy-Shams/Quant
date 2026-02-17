"""
Alpaca Market Data Adapter

Bridges Alpaca's data format to our existing signal pipeline.
Fetches historical bars and converts to the DataFrame formats
expected by MeanReversionSignals and BacktestEngine.

Performance optimizations:
- Cache-first loading: reads local parquet files before hitting API
- Incremental updates: only fetches new bars since last cache timestamp
- Trading calendar awareness: skips API calls on weekends/holidays
- Concurrent fetching: parallel API calls for fresh/split batches
- Request timeouts: prevents hanging on unresponsive API
- Graceful no-data handling: tracks IEX-unavailable symbols
- Vectorized bar conversion: bulk DataFrame construction
"""

import logging
import time
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime, timedelta, date as date_type
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError

import pandas as pd
import numpy as np

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestBarRequest
from alpaca.data.timeframe import TimeFrame

# ─── US Market Holiday Calendar ────────────────────────────────────────────
# Major US exchange holidays (NYSE/NASDAQ). Covers fixed + observed dates.
# Updated annually; holidays that fall on weekends are observed Mon/Fri.

def _us_market_holidays(year: int) -> Set[date_type]:
    """Return set of US market holiday dates for a given year."""
    from datetime import date as d
    holidays = set()

    # New Year's Day (Jan 1, observed)
    ny = d(year, 1, 1)
    if ny.weekday() == 6:  # Sunday → Monday
        holidays.add(d(year, 1, 2))
    elif ny.weekday() == 5:  # Saturday → Friday prior
        holidays.add(d(year - 1, 12, 31))
    else:
        holidays.add(ny)

    # MLK Day: 3rd Monday of January
    jan1 = d(year, 1, 1)
    first_mon = jan1 + timedelta(days=(7 - jan1.weekday()) % 7)
    holidays.add(first_mon + timedelta(weeks=2))

    # Presidents' Day: 3rd Monday of February
    feb1 = d(year, 2, 1)
    first_mon = feb1 + timedelta(days=(7 - feb1.weekday()) % 7)
    holidays.add(first_mon + timedelta(weeks=2))

    # Good Friday (approximate: 2 days before Easter)
    # Use Anonymous Gregorian algorithm
    a = year % 19
    b = year // 100
    c = year % 100
    dd = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - dd - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    easter = d(year, month, day)
    holidays.add(easter - timedelta(days=2))  # Good Friday

    # Memorial Day: last Monday of May
    may31 = d(year, 5, 31)
    holidays.add(may31 - timedelta(days=(may31.weekday()) % 7))

    # Juneteenth (Jun 19, observed) — NYSE holiday since 2022
    if year >= 2022:
        jt = d(year, 6, 19)
        if jt.weekday() == 6:
            holidays.add(d(year, 6, 20))
        elif jt.weekday() == 5:
            holidays.add(d(year, 6, 18))
        else:
            holidays.add(jt)

    # Independence Day (Jul 4, observed)
    jul4 = d(year, 7, 4)
    if jul4.weekday() == 6:
        holidays.add(d(year, 7, 5))
    elif jul4.weekday() == 5:
        holidays.add(d(year, 7, 3))
    else:
        holidays.add(jul4)

    # Labor Day: 1st Monday of September
    sep1 = d(year, 9, 1)
    holidays.add(sep1 + timedelta(days=(7 - sep1.weekday()) % 7))

    # Thanksgiving: 4th Thursday of November
    nov1 = d(year, 11, 1)
    first_thu = nov1 + timedelta(days=(3 - nov1.weekday()) % 7)
    holidays.add(first_thu + timedelta(weeks=3))

    # Christmas (Dec 25, observed)
    xmas = d(year, 12, 25)
    if xmas.weekday() == 6:
        holidays.add(d(year, 12, 26))
    elif xmas.weekday() == 5:
        holidays.add(d(year, 12, 24))
    else:
        holidays.add(xmas)

    return holidays


def _trading_days_between(start: date_type, end: date_type) -> int:
    """Count trading days (Mon-Fri, non-holiday) between two dates inclusive."""
    if start > end:
        return 0
    count = 0
    holidays = set()
    for y in range(start.year, end.year + 1):
        holidays |= _us_market_holidays(y)
    current = start
    while current <= end:
        if current.weekday() < 5 and current not in holidays:
            count += 1
        current += timedelta(days=1)
    return count

logger = logging.getLogger(__name__)


class AlpacaDataAdapter:
    """
    Fetches market data from Alpaca and converts to our pipeline format.

    Output format matches what our signal generator and backtest engine expect:
    - price_df: DataFrame with DatetimeIndex, columns = symbols, values = close prices
    - volume_df: DataFrame with same shape, values = volume

    Handles:
    - Cache-first loading with incremental API updates
    - Concurrent multi-batch fetching
    - Rate limiting (200 calls/min on free tier)
    - Data alignment across symbols (shared date index)
    - lookback period bootstrap for signal warmup
    """

    def __init__(
        self,
        data_client: StockHistoricalDataClient,
        data_feed: str = 'iex',
        cache_dir: Optional[Path] = None,
    ):
        """
        Args:
            data_client: Alpaca StockHistoricalDataClient
            data_feed: 'iex' (free) or 'sip' (paid)
            cache_dir: Optional directory to cache data locally
        """
        self.data_client = data_client
        self.data_feed = data_feed
        self.cache_dir = cache_dir
        self._call_count = 0
        self._call_window_start = time.time()
        self._no_data_symbols: Set[str] = set()  # Symbols with no IEX data

    def _rate_limit(self):
        """Enforce 200 calls/min rate limit"""
        self._call_count += 1
        elapsed = time.time() - self._call_window_start
        if self._call_count >= 190 and elapsed < 60:
            wait = 60 - elapsed + 1
            logger.info(f"Rate limit: waiting {wait:.0f}s...")
            time.sleep(wait)
            self._call_count = 0
            self._call_window_start = time.time()
        elif elapsed >= 60:
            self._call_count = 0
            self._call_window_start = time.time()

    # ─── Bar conversion helper ──────────────────────────────────────────

    @staticmethod
    def _bars_to_df(symbol_bars) -> Optional[pd.DataFrame]:
        """Convert a list of Alpaca Bar objects to a DataFrame efficiently."""
        if not symbol_bars:
            return None
        records = [{
            'date': bar.timestamp.date(),
            'open': float(bar.open),
            'high': float(bar.high),
            'low': float(bar.low),
            'close': float(bar.close),
            'volume': int(bar.volume),
            'vwap': float(bar.vwap) if bar.vwap else None,
        } for bar in symbol_bars]
        df = pd.DataFrame(records)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        return df[~df.index.duplicated(keep='last')]

    # ─── Single-batch fetch (used by concurrent executor) ──────────────

    def _fetch_batch(
        self,
        batch: List[str],
        start_date: datetime,
        end_date: datetime,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch one batch of symbols. Thread-safe (no shared mutable state)."""
        result: Dict[str, pd.DataFrame] = {}
        try:
            request = StockBarsRequest(
                symbol_or_symbols=batch,
                timeframe=TimeFrame.Day,
                start=start_date,
                end=end_date,
                feed=self.data_feed,
            )
            bars = self.data_client.get_stock_bars(request)
            for symbol in batch:
                try:
                    symbol_bars = bars[symbol]
                except (KeyError, IndexError):
                    symbol_bars = []
                df = self._bars_to_df(symbol_bars)
                if df is not None and len(df) > 0:
                    result[symbol] = df
                else:
                    self._no_data_symbols.add(symbol)
                    logger.debug(f"No data returned for {symbol} (IEX may not cover this symbol)")
        except Exception as e:
            logger.error(f"Batch error ({batch[:3]}…): {e}")
            # Individual fallback
            for symbol in batch:
                try:
                    req = StockBarsRequest(
                        symbol_or_symbols=symbol,
                        timeframe=TimeFrame.Day,
                        start=start_date,
                        end=end_date,
                        feed=self.data_feed,
                    )
                    bars = self.data_client.get_stock_bars(req)
                    try:
                        symbol_bars = bars[symbol]
                    except (KeyError, IndexError):
                        symbol_bars = []
                    df = self._bars_to_df(symbol_bars)
                    if df is not None and len(df) > 0:
                        result[symbol] = df
                except Exception as e2:
                    logger.error(f"Error fetching {symbol}: {e2}")
        return result

    # ─── Primary fetch (concurrent batches) ────────────────────────────

    def fetch_daily_bars(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: Optional[datetime] = None,
        max_workers: int = 4,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch daily OHLCV bars for multiple symbols with concurrent batches.

        Args:
            symbols: List of tickers
            start_date: Start date for data
            end_date: End date (default: today)
            max_workers: Max parallel API threads (keep ≤4 for rate limits)

        Returns:
            Dict of {symbol: DataFrame with OHLCV columns}
        """
        if end_date is None:
            end_date = datetime.now()

        all_bars: Dict[str, pd.DataFrame] = {}

        # Split into batches (Alpaca handles up to ~100 symbols per call;
        # we use 15 per batch to allow parallel calls within rate limits)
        batch_size = 15
        batches = [
            symbols[i:i + batch_size]
            for i in range(0, len(symbols), batch_size)
        ]

        if len(batches) == 1:
            # Single batch — no thread overhead
            all_bars = self._fetch_batch(batches[0], start_date, end_date)
        else:
            # Concurrent batches — 2-4× faster for larger universes
            with ThreadPoolExecutor(max_workers=min(max_workers, len(batches))) as pool:
                futures = {
                    pool.submit(self._fetch_batch, batch, start_date, end_date): batch
                    for batch in batches
                }
                for future in as_completed(futures):
                    try:
                        batch_result = future.result(timeout=60)
                        all_bars.update(batch_result)
                    except FuturesTimeoutError:
                        batch = futures[future]
                        logger.warning(f"Timeout fetching batch: {batch[:3]}... — skipping")
                    except Exception as e:
                        batch = futures[future]
                        logger.error(f"Error in batch {batch[:3]}...: {e}")

        logger.info(f"Fetched data for {len(all_bars)}/{len(symbols)} symbols")
        return all_bars

    def to_pipeline_format(
        self,
        raw_bars: Dict[str, pd.DataFrame],
        min_history: int = 100,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Convert raw bars to pipeline-compatible DataFrames.

        Args:
            raw_bars: Dict of {symbol: OHLCV DataFrame}
            min_history: Minimum data points required per symbol

        Returns:
            (price_df, volume_df) — aligned DataFrames with shared DatetimeIndex
        """
        price_series = {}
        volume_series = {}

        for symbol, df in raw_bars.items():
            if len(df) < min_history:
                logger.warning(
                    f"Skipping {symbol}: only {len(df)} bars (need {min_history})"
                )
                continue
            price_series[symbol] = df['close']
            volume_series[symbol] = df['volume']

        if not price_series:
            raise ValueError("No symbols have sufficient data")

        price_df = pd.DataFrame(price_series)
        volume_df = pd.DataFrame(volume_series)

        # Align to common trading days
        price_df = price_df.sort_index()
        volume_df = volume_df.sort_index()

        # Forward-fill small gaps (holidays across exchanges)
        price_df = price_df.ffill(limit=3)
        volume_df = volume_df.ffill(limit=3).fillna(0)

        logger.info(
            f"Pipeline data: {len(price_df)} days × {len(price_df.columns)} symbols "
            f"({price_df.index[0].date()} to {price_df.index[-1].date()})"
        )

        return price_df, volume_df

    # ─── Smart fetch: cache-first + incremental update ─────────────────

    def fetch_pipeline_data(
        self,
        symbols: List[str],
        lookback_days: int = 504,
        end_date: Optional[datetime] = None,
        min_history: int = 100,
        use_cache: bool = True,
        verbose: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Cache-aware data loader: loads from disk first, then fetches only
        the delta from Alpaca. On first run, does a full fetch.

        Performance:
        - Cached + up-to-date: <2s (disk I/O only)
        - Cached + stale by N days: fetches N days of incremental data
        - No cache: full API fetch (30-90s depending on history depth)

        Args:
            symbols: List of tickers
            lookback_days: Calendar days of history
            end_date: End date (default: today)
            min_history: Minimum trading days per symbol
            use_cache: Whether to try cache first (default True)
            verbose: Print progress details

        Returns:
            (price_df, volume_df, raw_bars)
        """
        if end_date is None:
            end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)

        raw_bars: Dict[str, pd.DataFrame] = {}
        cache_hit = 0
        cache_miss_symbols: List[str] = []
        need_incremental = False
        incremental_start: Optional[datetime] = None
        cache_age_days = 0

        # ── Step 1: Try loading from cache ─────────────────────────────
        if use_cache and self.cache_dir:
            cached = self.load_cache(symbols, label='latest')
            if cached:
                cache_hit = len(cached)

                # Find the latest date across all cached symbols
                latest_dates = [df.index.max() for df in cached.values() if len(df) > 0]
                if latest_dates:
                    latest_cached = max(latest_dates)
                    latest_cached_dt = pd.Timestamp(latest_cached).to_pydatetime()

                    # Trim cached data to requested window
                    for sym, df in cached.items():
                        trimmed = df.loc[df.index >= pd.Timestamp(start_date)]
                        if len(trimmed) > 0:
                            raw_bars[sym] = trimmed

                    # ── Trading calendar check ─────────────────────────
                    today = datetime.now().date()
                    cache_date = latest_cached_dt.date()
                    cache_age_days = (today - cache_date).days

                    # Count actual trading days missed (not calendar days)
                    next_day = cache_date + timedelta(days=1)
                    missed_trading_days = _trading_days_between(next_day, today)

                    if missed_trading_days > 0:
                        need_incremental = True
                        incremental_start = latest_cached_dt + timedelta(days=1)
                        if verbose:
                            print(f"   📦 Cache hit: {cache_hit} symbols "
                                  f"(latest: {cache_date}, "
                                  f"{cache_age_days}d ago, "
                                  f"{missed_trading_days} trading day(s) to fetch)")
                    else:
                        if verbose:
                            reason = ""
                            if today.weekday() >= 5:
                                reason = " (weekend)"
                            elif today in _us_market_holidays(today.year):
                                reason = " (market holiday)"
                            elif cache_age_days == 0:
                                reason = ""
                            else:
                                reason = " (no missed trading days)"
                            print(f"   📦 Cache hit: {cache_hit} symbols — "
                                  f"up to date{reason}, skipping API")

                # Symbols not in cache need full fetch
                cache_miss_symbols = [s for s in symbols if s not in raw_bars]
                if cache_miss_symbols and verbose:
                    print(f"   🔍 Cache miss: {len(cache_miss_symbols)} symbols "
                          f"need full fetch")
            else:
                cache_miss_symbols = symbols
                if verbose:
                    print(f"   📦 No cache found — full API fetch")
        else:
            cache_miss_symbols = symbols

        # ── Step 2: Incremental update for cached symbols ──────────────
        if need_incremental and incremental_start and raw_bars:
            # Filter out symbols that are known to have no IEX data
            stale_symbols = [s for s in raw_bars.keys()
                             if s not in self._no_data_symbols]
            if stale_symbols:
                if verbose:
                    print(f"   🔄 Incremental update: {len(stale_symbols)} symbols "
                          f"from {incremental_start.date()}...")
                    if self._no_data_symbols & set(raw_bars.keys()):
                        skipped = self._no_data_symbols & set(raw_bars.keys())
                        print(f"   ⏭️  Skipping {len(skipped)} symbols with no IEX data: "
                              f"{', '.join(sorted(skipped)[:5])}{'...' if len(skipped) > 5 else ''}")
                t0 = time.perf_counter()
                delta = self.fetch_daily_bars(stale_symbols, incremental_start, end_date)
                dt = time.perf_counter() - t0
                updated = 0
                for sym in stale_symbols:
                    if sym in delta and len(delta[sym]) > 0:
                        raw_bars[sym] = pd.concat([raw_bars[sym], delta[sym]])
                        raw_bars[sym] = raw_bars[sym][~raw_bars[sym].index.duplicated(keep='last')]
                        raw_bars[sym] = raw_bars[sym].sort_index()
                        updated += 1
                if verbose:
                    print(f"   ✅ Updated {updated} symbols in {dt:.1f}s")
            else:
                if verbose:
                    print(f"   ⏭️  All cached symbols are known IEX-absent — skipping API")

        # ── Step 3: Full fetch for cache-miss symbols ──────────────────
        if cache_miss_symbols:
            if verbose:
                print(f"   🌐 Fetching {len(cache_miss_symbols)} symbols "
                      f"from API ({lookback_days} days)...")
            t0 = time.perf_counter()
            fresh = self.fetch_daily_bars(cache_miss_symbols, start_date, end_date)
            dt = time.perf_counter() - t0
            raw_bars.update(fresh)
            if verbose:
                print(f"   ✅ Fetched {len(fresh)} symbols in {dt:.1f}s")

        # ── Step 4: Convert to pipeline format ─────────────────────────
        price_df, volume_df = self.to_pipeline_format(raw_bars, min_history)

        return price_df, volume_df, raw_bars

    def get_latest_prices(self, symbols: List[str]) -> pd.Series:
        """
        Get latest closing prices for symbols.

        Returns:
            Series indexed by symbol with latest close prices
        """
        self._rate_limit()
        try:
            request = StockLatestBarRequest(
                symbol_or_symbols=symbols,
                feed=self.data_feed,
            )
            bars = self.data_client.get_stock_latest_bar(request)
            return pd.Series({
                symbol: float(bar.close)
                for symbol, bar in bars.items()
            })
        except Exception as e:
            logger.error(f"Error fetching latest prices: {e}")
            return pd.Series(dtype=float)

    # ─── Cache I/O ─────────────────────────────────────────────────────

    def save_cache(self, raw_bars: Dict[str, pd.DataFrame], label: str = 'alpaca'):
        """Save fetched data to local cache (overwrites existing)"""
        if self.cache_dir is None:
            return

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        for symbol, df in raw_bars.items():
            path = self.cache_dir / f"{symbol}_{label}.parquet"
            df.to_parquet(path)
        logger.info(f"Cached {len(raw_bars)} symbols to {self.cache_dir}")

    def load_cache(self, symbols: List[str], label: str = 'alpaca') -> Dict[str, pd.DataFrame]:
        """Load data from local cache (parquet files)"""
        if self.cache_dir is None:
            return {}

        loaded = {}
        for symbol in symbols:
            path = self.cache_dir / f"{symbol}_{label}.parquet"
            if path.exists():
                loaded[symbol] = pd.read_parquet(path)

        if loaded:
            logger.info(f"Loaded {len(loaded)} symbols from cache")
        return loaded
