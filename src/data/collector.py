"""
Historical data collector for IB Gateway.
Downloads OHLCV bars and stores as Parquet files.
Designed for efficient bulk download and incremental updates.
"""

import logging
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict

import pandas as pd
from ib_insync import IB, Stock, Contract, util

try:
    from data.universe_builder import get_unique_tickers, get_etf_tickers, INDEX_ETFS
except ImportError:
    from ..data.universe_builder import get_unique_tickers, get_etf_tickers, INDEX_ETFS


logger = logging.getLogger(__name__)


class DataCollector:
    """
    Collects historical market data from IB and stores as Parquet files.
    
    Directory structure:
        data/
          historical/
            daily/
              AAPL.parquet
              MSFT.parquet
              ...
            hourly/
              AAPL.parquet
              ...
          snapshots/
            options/
              AAPL_2026-02-12.parquet
              ...
    """

    # IB rate limits: max 60 requests per 10 minutes for historical data
    REQUEST_DELAY = 11  # seconds between requests (conservative)
    MAX_RETRIES = 3

    def __init__(self, ib: IB, data_dir: Optional[str] = None):
        """
        Args:
            ib: Connected IB instance (from IBConnection.ib)
            data_dir: Root directory for data storage.
                      Defaults to project_root/data/
        """
        self.ib = ib
        self.logger = logging.getLogger(__name__)

        if data_dir is None:
            # Default: project_root/data/
            src_dir = Path(__file__).resolve().parent.parent
            project_root = src_dir.parent
            self.data_dir = project_root / "data"
        else:
            self.data_dir = Path(data_dir)

        # Create directory structure
        self.daily_dir = self.data_dir / "historical" / "daily"
        self.hourly_dir = self.data_dir / "historical" / "hourly"
        self.options_dir = self.data_dir / "snapshots" / "options"
        
        for d in [self.daily_dir, self.hourly_dir, self.options_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def _make_contract(self, symbol: str, exchange: str = "SMART", 
                       currency: str = "USD") -> Stock:
        """Create an IB Stock contract."""
        return Stock(symbol, exchange, currency)

    def _qualify_contract(self, contract: Contract) -> bool:
        """Qualify a contract with IB to get full details."""
        try:
            qualified = self.ib.qualifyContracts(contract)
            return len(qualified) > 0
        except Exception as e:
            self.logger.warning(f"Failed to qualify {contract.symbol}: {e}")
            return False

    def fetch_historical_bars(
        self,
        symbol: str,
        duration: str = "2 Y",
        bar_size: str = "1 day",
        what_to_show: str = "TRADES",
        end_date: str = "",
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical OHLCV bars for a single symbol.

        Args:
            symbol: Ticker symbol (e.g., 'AAPL')
            duration: How far back ('1 D', '1 W', '1 M', '1 Y', '2 Y')
            bar_size: Bar granularity ('1 min', '5 mins', '1 hour', '1 day')
            what_to_show: Data type ('TRADES', 'MIDPOINT', 'BID', 'ASK')
            end_date: End date string. Empty = now.

        Returns:
            DataFrame with columns: date, open, high, low, close, volume, average, barCount
            Returns None if request fails.
        """
        contract = self._make_contract(symbol)
        if not self._qualify_contract(contract):
            self.logger.error(f"[{symbol}] Contract qualification failed")
            return None

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                bars = self.ib.reqHistoricalData(
                    contract,
                    endDateTime=end_date,
                    durationStr=duration,
                    barSizeSetting=bar_size,
                    whatToShow=what_to_show,
                    useRTH=True,  # Regular trading hours only
                    formatDate=1,
                )

                if not bars:
                    self.logger.warning(f"[{symbol}] No data returned (attempt {attempt})")
                    if attempt < self.MAX_RETRIES:
                        time.sleep(self.REQUEST_DELAY)
                    continue

                df = util.df(bars)
                df.insert(0, "symbol", symbol)
                
                # Ensure date column is datetime
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])

                self.logger.info(f"[{symbol}] Retrieved {len(df)} bars ({bar_size})")
                return df

            except Exception as e:
                self.logger.error(f"[{symbol}] Error (attempt {attempt}): {e}")
                if attempt < self.MAX_RETRIES:
                    time.sleep(self.REQUEST_DELAY)

        return None

    def save_parquet(self, df: pd.DataFrame, filepath: Path) -> bool:
        """Save DataFrame as Parquet file."""
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(filepath, index=False, engine="pyarrow")
            size_kb = filepath.stat().st_size / 1024
            self.logger.info(f"Saved {filepath.name} ({size_kb:.1f} KB, {len(df)} rows)")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save {filepath}: {e}")
            return False

    def load_parquet(self, filepath: Path) -> Optional[pd.DataFrame]:
        """Load DataFrame from Parquet file."""
        try:
            if filepath.exists():
                return pd.read_parquet(filepath, engine="pyarrow")
            return None
        except Exception as e:
            self.logger.error(f"Failed to load {filepath}: {e}")
            return None

    def has_data(self, symbol: str, min_age_days: int = 2) -> bool:
        """
        Check if data exists for a symbol and is reasonably fresh.
        
        Args:
            symbol: Ticker symbol
            min_age_days: Maximum staleness in days (default 2)
        
        Returns:
            True if data exists and is fresh enough
        """
        filepath = self.daily_dir / f"{symbol.replace(' ', '_')}.parquet"
        
        if not filepath.exists():
            return False
        
        try:
            df = self.load_parquet(filepath)
            if df is None or len(df) == 0:
                return False
            
            last_date = pd.to_datetime(df["date"]).max()
            days_stale = (datetime.now() - last_date).days
            
            return days_stale <= min_age_days
        except Exception:
            return False

    def filter_missing_tickers(self, tickers: List[str]) -> List[str]:
        """
        Filter out tickers that already have fresh data.
        
        Args:
            tickers: List of symbols to check
        
        Returns:
            List of symbols that need data collection
        """
        missing = []
        for ticker in tickers:
            if not self.has_data(ticker):
                missing.append(ticker)
        return missing

    def collect_daily_data(
        self,
        tickers: List[str] = None,
        duration: str = "2 Y",
        update_existing: bool = True,
    ) -> Dict[str, bool]:
        """
        Bulk download daily OHLCV data for all tickers.

        Args:
            tickers: List of symbols. If None, uses full universe.
            duration: How far back to fetch.
            update_existing: If True, skips tickers that already have recent data.

        Returns:
            Dictionary of {symbol: success_bool}
        """
        if tickers is None:
            tickers = get_unique_tickers()

        results = {}
        total = len(tickers)
        
        self.logger.info(f"Starting daily data collection for {total} tickers")
        print(f"\nCollecting daily data: {total} tickers, duration={duration}")
        print("=" * 60)

        for i, symbol in enumerate(tickers, 1):
            filepath = self.daily_dir / f"{symbol.replace(' ', '_')}.parquet"

            # Check if update is needed
            if update_existing and filepath.exists():
                existing = self.load_parquet(filepath)
                if existing is not None and len(existing) > 0:
                    last_date = pd.to_datetime(existing["date"]).max()
                    days_stale = (datetime.now() - last_date).days
                    if days_stale <= 1:
                        self.logger.info(f"[{symbol}] Already up to date")
                        print(f"  [{i:>3}/{total}] {symbol:<8} -- up to date")
                        results[symbol] = True
                        continue

            # Fetch data
            print(f"  [{i:>3}/{total}] {symbol:<8}", end="", flush=True)
            df = self.fetch_historical_bars(symbol, duration=duration, bar_size="1 day")

            if df is not None and len(df) > 0:
                # Merge with existing data if present
                if update_existing and filepath.exists():
                    existing = self.load_parquet(filepath)
                    if existing is not None:
                        df = pd.concat([existing, df]).drop_duplicates(
                            subset=["date", "symbol"], keep="last"
                        ).sort_values("date").reset_index(drop=True)

                success = self.save_parquet(df, filepath)
                results[symbol] = success
                print(f" -- {len(df)} bars saved")
            else:
                results[symbol] = False
                print(f" -- FAILED")

            # Rate limiting
            if i < total:
                time.sleep(self.REQUEST_DELAY)

        # Summary
        ok = sum(1 for v in results.values() if v)
        fail = sum(1 for v in results.values() if not v)
        print("=" * 60)
        print(f"Complete: {ok} succeeded, {fail} failed out of {total}")
        
        return results

    def collect_hourly_data(
        self,
        tickers: List[str] = None,
        duration: str = "20 D",
    ) -> Dict[str, bool]:
        """
        Download hourly bars. IB limits hourly data to ~30 days.

        Args:
            tickers: List of symbols. If None, uses ETFs only (for efficiency).
            duration: How far back (max ~30 days for hourly).

        Returns:
            Dictionary of {symbol: success_bool}
        """
        if tickers is None:
            tickers = get_etf_tickers()

        results = {}
        total = len(tickers)
        
        print(f"\nCollecting hourly data: {total} tickers, duration={duration}")
        print("=" * 60)

        for i, symbol in enumerate(tickers, 1):
            filepath = self.hourly_dir / f"{symbol.replace(' ', '_')}.parquet"
            
            print(f"  [{i:>3}/{total}] {symbol:<8}", end="", flush=True)
            df = self.fetch_historical_bars(symbol, duration=duration, bar_size="1 hour")

            if df is not None and len(df) > 0:
                success = self.save_parquet(df, filepath)
                results[symbol] = success
                print(f" -- {len(df)} bars saved")
            else:
                results[symbol] = False
                print(f" -- FAILED")

            if i < total:
                time.sleep(self.REQUEST_DELAY)

        ok = sum(1 for v in results.values() if v)
        print("=" * 60)
        print(f"Complete: {ok}/{total} succeeded")
        return results

    def get_stored_tickers(self, timeframe: str = "daily") -> List[str]:
        """Get list of tickers that have stored data."""
        data_dir = self.daily_dir if timeframe == "daily" else self.hourly_dir
        return sorted([
            f.stem.replace("_", " ") for f in data_dir.glob("*.parquet")
        ])

    def load_ticker_data(self, symbol: str, timeframe: str = "daily") -> Optional[pd.DataFrame]:
        """Load stored data for a specific ticker."""
        data_dir = self.daily_dir if timeframe == "daily" else self.hourly_dir
        filepath = data_dir / f"{symbol.replace(' ', '_')}.parquet"
        return self.load_parquet(filepath)

    def data_status(self) -> pd.DataFrame:
        """Get summary of all stored data."""
        rows = []
        for filepath in sorted(self.daily_dir.glob("*.parquet")):
            df = self.load_parquet(filepath)
            if df is not None:
                rows.append({
                    "symbol": filepath.stem.replace("_", " "),
                    "bars": len(df),
                    "start": pd.to_datetime(df["date"]).min().strftime("%Y-%m-%d"),
                    "end": pd.to_datetime(df["date"]).max().strftime("%Y-%m-%d"),
                    "size_kb": round(filepath.stat().st_size / 1024, 1),
                })

        if rows:
            return pd.DataFrame(rows)
        return pd.DataFrame(columns=["symbol", "bars", "start", "end", "size_kb"])
