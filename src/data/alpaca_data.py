"""
Alpaca Market Data Adapter

Bridges Alpaca's data format to our existing signal pipeline.
Fetches historical bars and converts to the DataFrame formats
expected by MeanReversionSignals and BacktestEngine.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path

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

    def fetch_daily_bars(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch daily OHLCV bars for multiple symbols.

        Args:
            symbols: List of tickers
            start_date: Start date for data
            end_date: End date (default: today)

        Returns:
            Dict of {symbol: DataFrame with OHLCV columns}
        """
        if end_date is None:
            end_date = datetime.now()

        all_bars = {}

        # Alpaca supports multi-symbol requests (more efficient)
        batch_size = 30  # Free tier limit
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]

            self._rate_limit()

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
                    symbol_bars = bars[symbol] if symbol in bars else []
                    if symbol_bars:
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
                        # Remove duplicate dates (can happen with Alpaca data)
                        df = df[~df.index.duplicated(keep='last')]
                        all_bars[symbol] = df
                    else:
                        logger.warning(f"No data returned for {symbol}")

            except Exception as e:
                logger.error(f"Error fetching batch {batch}: {e}")
                # Try individual symbols as fallback
                for symbol in batch:
                    try:
                        self._rate_limit()
                        request = StockBarsRequest(
                            symbol_or_symbols=symbol,
                            timeframe=TimeFrame.Day,
                            start=start_date,
                            end=end_date,
                            feed=self.data_feed,
                        )
                        bars = self.data_client.get_stock_bars(request)
                        symbol_bars = bars[symbol] if symbol in bars else []
                        if symbol_bars:
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
                            df = df[~df.index.duplicated(keep='last')]
                            all_bars[symbol] = df
                    except Exception as e2:
                        logger.error(f"Error fetching {symbol}: {e2}")

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

    def fetch_pipeline_data(
        self,
        symbols: List[str],
        lookback_days: int = 504,
        end_date: Optional[datetime] = None,
        min_history: int = 100,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        One-call convenience: fetch + convert to pipeline format.

        Args:
            symbols: List of tickers
            lookback_days: Calendar days of history (504 ≈ 2 years trading days)
            end_date: End date (default: today)
            min_history: Minimum trading days per symbol

        Returns:
            (price_df, volume_df, raw_bars)
        """
        if end_date is None:
            end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)

        raw_bars = self.fetch_daily_bars(symbols, start_date, end_date)
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

    def save_cache(self, raw_bars: Dict[str, pd.DataFrame], label: str = 'alpaca'):
        """Save fetched data to local cache"""
        if self.cache_dir is None:
            return

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        for symbol, df in raw_bars.items():
            path = self.cache_dir / f"{symbol}_{label}.parquet"
            df.to_parquet(path)
        logger.info(f"Cached {len(raw_bars)} symbols to {self.cache_dir}")

    def load_cache(self, symbols: List[str], label: str = 'alpaca') -> Dict[str, pd.DataFrame]:
        """Load data from local cache"""
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
