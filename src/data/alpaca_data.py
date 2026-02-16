"""
Alpaca Market Data Adapter

Bridges Alpaca's data format to our existing signal pipeline.
Fetches historical bars and converts to the DataFrame formats
expected by MeanReversionSignals and BacktestEngine.

Performance optimizations:
- Cache-first loading: reads local parquet files before hitting API
- Incremental updates: only fetches new bars since last cache timestamp
- Concurrent fetching: parallel API calls for fresh/split batches
- Vectorized bar conversion: bulk DataFrame construction
"""

import logging
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestBarRequest
from alpaca.data.timeframe import TimeFrame

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
                    logger.warning(f"No data returned for {symbol}")
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
                    batch_result = future.result()
                    all_bars.update(batch_result)

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
        incremental_start: Optional[datetime] = None

        # ── Step 1: Try loading from cache ─────────────────────────────
        if use_cache and self.cache_dir:
            cached = self.load_cache(symbols, label='latest')
            if cached:
                cache_hit = len(cached)

                # Find the latest date across all cached symbols
                latest_dates = [df.index.max() for df in cached.values() if len(df) > 0]
                if latest_dates:
                    latest_cached = max(latest_dates)
                    # Convert to datetime for comparison
                    latest_cached_dt = pd.Timestamp(latest_cached).to_pydatetime()

                    # Trim cached data to requested window
                    for sym, df in cached.items():
                        trimmed = df.loc[df.index >= pd.Timestamp(start_date)]
                        if len(trimmed) > 0:
                            raw_bars[sym] = trimmed

                    # Determine if we need incremental update
                    today = datetime.now().date()
                    cache_age_days = (today - latest_cached_dt.date()).days

                    if cache_age_days > 0:
                        # Need incremental update for all cached symbols
                        incremental_start = latest_cached_dt + timedelta(days=1)
                        if verbose:
                            print(f"   📦 Cache hit: {cache_hit} symbols "
                                  f"(latest: {latest_cached_dt.date()}, "
                                  f"{cache_age_days}d stale)")
                    else:
                        if verbose:
                            print(f"   📦 Cache hit: {cache_hit} symbols (up to date)")

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
        if incremental_start and raw_bars:
            stale_symbols = list(raw_bars.keys())
            if verbose:
                print(f"   🔄 Incremental update: {len(stale_symbols)} symbols "
                      f"from {incremental_start.date()}...")
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
