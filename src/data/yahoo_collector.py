"""
Yahoo Finance historical data collector.
Downloads OHLCV bars and stores as Parquet files in IB-compatible format.
Designed for bulk historical data (20+ years) where IB API is limited to ~2 years.
"""

import logging
import time
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import yfinance as yf

try:
    from data.universe_builder import get_unique_tickers, get_etf_tickers
except ImportError:
    from ..data.universe_builder import get_unique_tickers, get_etf_tickers


logger = logging.getLogger(__name__)


class YahooCollector:
    """
    Collects historical market data from Yahoo Finance and stores as Parquet files.
    
    Format is IB-compatible: saves to data/historical/daily/ with same schema.
    
    Advantages over IB:
    - 20+ years of historical data (IB limited to ~2 years via API)
    - Faster bulk downloads (no rate limits)
    - No active TWS connection required
    
    Use for: Historical backtesting data
    Use IB for: Real-time quotes, live trading, options chains
    """

    def __init__(self, data_dir: Optional[str] = None):
        """
        Args:
            data_dir: Root directory for data storage.
                      Defaults to project_root/data/
        """
        self.logger = logging.getLogger(__name__)

        if data_dir is None:
            # Default: project_root/data/
            src_dir = Path(__file__).resolve().parent.parent
            project_root = src_dir.parent
            self.data_dir = project_root / "data"
        else:
            self.data_dir = Path(data_dir)

        # Same directory structure as IB collector
        self.daily_dir = self.data_dir / "historical" / "daily"
        self.daily_dir.mkdir(parents=True, exist_ok=True)

    def fetch_historical_bars(
        self,
        symbol: str,
        start_date: str = None,
        end_date: str = None,
        period: str = "20y",
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical OHLCV bars for a single symbol from Yahoo Finance.

        Args:
            symbol: Ticker symbol (e.g., 'AAPL')
            start_date: Start date 'YYYY-MM-DD' (optional, use period if None)
            end_date: End date 'YYYY-MM-DD' (optional, defaults to today)
            period: Period string if start_date not given ('1y', '5y', '20y', 'max')

        Returns:
            DataFrame with IB-compatible columns: symbol, date, open, high, low, close, volume
            Returns None if request fails.
        """
        try:
            ticker = yf.Ticker(symbol)
            
            # Download data
            if start_date:
                df = ticker.history(start=start_date, end=end_date, auto_adjust=False)
            else:
                df = ticker.history(period=period, auto_adjust=False)

            if df is None or len(df) == 0:
                self.logger.warning(f"[{symbol}] No data returned from Yahoo Finance")
                return None

            # Convert to IB-compatible format
            df = df.reset_index()
            
            # Rename columns to match IB format (lowercase)
            df = df.rename(columns={
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume',
            })
            
            # Add symbol column (IB format)
            df.insert(0, 'symbol', symbol)
            
            # Keep only columns that IB provides (drop Dividends, Stock Splits, Adj Close)
            keep_cols = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']
            df = df[[col for col in keep_cols if col in df.columns]]
            
            # Ensure date is datetime
            df['date'] = pd.to_datetime(df['date'])
            
            # Sort by date (oldest first)
            df = df.sort_values('date').reset_index(drop=True)
            
            self.logger.info(f"[{symbol}] Retrieved {len(df)} bars from Yahoo Finance")
            return df

        except Exception as e:
            self.logger.error(f"[{symbol}] Error fetching from Yahoo Finance: {e}")
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

    def has_data(self, symbol: str, min_years: int = 10) -> bool:
        """
        Check if data exists for a symbol and has sufficient history.
        
        Args:
            symbol: Ticker symbol
            min_years: Minimum years of history required (default 10)
        
        Returns:
            True if data exists with at least min_years of history
        """
        filepath = self.daily_dir / f"{symbol.replace(' ', '_')}.parquet"
        
        if not filepath.exists():
            return False
        
        try:
            df = self.load_parquet(filepath)
            if df is None or len(df) == 0:
                return False
            
            first_date = pd.to_datetime(df["date"]).min()
            last_date = pd.to_datetime(df["date"]).max()
            years_data = (last_date - first_date).days / 365.25
            
            return years_data >= min_years
        except Exception:
            return False

    def collect_historical_data(
        self,
        tickers: List[str] = None,
        period: str = "20y",
        update_existing: bool = False,
        max_workers: Union[int, str, None] = None,
    ) -> Dict[str, bool]:
        """
        Collect historical data for multiple tickers (bulk download).
        
        Args:
            tickers: List of symbols. If None, uses universe tickers.
            period: Period to download ('1y', '5y', '10y', '20y', 'max')
            update_existing: If True, re-download even if data exists
            max_workers: Number of parallel download threads.
                        - None or "auto": Auto-detect CPU cores (cpu_count())
                        - int: Specific number of workers
        
        Returns:
            Dictionary mapping symbol -> success boolean
        """
        if tickers is None:
            tickers = get_unique_tickers()
        
        # Auto-detect CPU cores if max_workers is None or "auto"
        if max_workers is None or max_workers == "auto":
            cpu_count = os.cpu_count() or 4  # Fallback to 4 if detection fails
            max_workers = cpu_count
            self.logger.info(f"Auto-detected {cpu_count} CPU cores, using {max_workers} workers")
        elif isinstance(max_workers, str):
            # Handle string numbers from config
            try:
                max_workers = int(max_workers)
            except ValueError:
                max_workers = os.cpu_count() or 4
                self.logger.warning(f"Invalid max_workers value, using auto-detected {max_workers}")
        
        results = {}
        total = len(tickers)
        
        print(f"\nYahoo Finance Historical Data Collection")
        print(f"Period: {period}")
        print(f"Tickers: {total}")
        print(f"Workers: {max_workers} (CPU cores: {os.cpu_count() or 'unknown'})")
        print("=" * 60)
        
        # Filter tickers if not updating existing
        if not update_existing:
            missing = [t for t in tickers if not self.has_data(t, min_years=10)]
            if len(missing) < total:
                print(f"\n{total - len(missing)} tickers already have 10+ years of data")
                print(f"Downloading {len(missing)} missing/incomplete tickers\n")
                tickers = missing
        
        if len(tickers) == 0:
            print("All tickers already have sufficient data!")
            return {}
        
        start_time = time.time()
        success_count = 0
        
        def download_one(symbol: str) -> tuple:
            """Download and save data for one symbol."""
            try:
                df = self.fetch_historical_bars(symbol, period=period)
                if df is None or len(df) == 0:
                    return (symbol, False, "No data returned")
                
                filepath = self.daily_dir / f"{symbol.replace(' ', '_')}.parquet"
                if self.save_parquet(df, filepath):
                    years = (df['date'].max() - df['date'].min()).days / 365.25
                    return (symbol, True, f"{len(df)} bars, {years:.1f} years")
                else:
                    return (symbol, False, "Save failed")
            except Exception as e:
                return (symbol, False, str(e))
        
        # Parallel download with thread pool
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(download_one, ticker): ticker for ticker in tickers}
            
            for i, future in enumerate(as_completed(futures), 1):
                symbol, success, msg = future.result()
                results[symbol] = success
                
                if success:
                    success_count += 1
                    status = "OK"
                else:
                    status = "FAIL"
                
                print(f"[{i}/{len(tickers)}] {symbol:<6} [{status}] {msg}")
        
        elapsed = time.time() - start_time
        
        print("\n" + "=" * 60)
        print(f"Complete: {success_count}/{len(tickers)} tickers downloaded")
        print(f"Time: {elapsed:.1f}s ({elapsed/len(tickers):.1f}s per ticker)")
        print(f"Failed: {len(tickers) - success_count}")
        
        if len(tickers) - success_count > 0:
            failed = [s for s, ok in results.items() if not ok]
            print(f"Failed tickers: {', '.join(failed[:10])}")
            if len(failed) > 10:
                print(f"  ... and {len(failed) - 10} more")
        
        return results

    def filter_missing_tickers(self, tickers: List[str], min_years: int = 10) -> List[str]:
        """
        Filter ticker list to only those missing sufficient historical data.
        
        Args:
            tickers: List of symbols to check
            min_years: Minimum years of history required
        
        Returns:
            List of tickers that need data collection
        """
        return [t for t in tickers if not self.has_data(t, min_years=min_years)]

    def data_status(self) -> pd.DataFrame:
        """
        Get summary of all stored ticker data.
        
        Returns:
            DataFrame with columns: symbol, bars, start_date, end_date, years, size_kb
        """
        files = list(self.daily_dir.glob("*.parquet"))
        
        if not files:
            return pd.DataFrame(columns=['symbol', 'bars', 'start_date', 'end_date', 'years', 'size_kb'])
        
        rows = []
        for filepath in files:
            try:
                df = self.load_parquet(filepath)
                if df is not None and len(df) > 0:
                    symbol = filepath.stem.replace('_', ' ')
                    start_date = df['date'].min()
                    end_date = df['date'].max()
                    years = (end_date - start_date).days / 365.25
                    size_kb = filepath.stat().st_size / 1024
                    
                    rows.append({
                        'symbol': symbol,
                        'bars': len(df),
                        'start_date': start_date.strftime('%Y-%m-%d'),
                        'end_date': end_date.strftime('%Y-%m-%d'),
                        'years': round(years, 1),
                        'size_kb': round(size_kb, 1),
                    })
            except Exception:
                continue
        
        status_df = pd.DataFrame(rows)
        if len(status_df) > 0:
            status_df = status_df.sort_values('symbol')
        
        return status_df

    def load_ticker_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Load historical data for a specific ticker.
        
        Args:
            symbol: Ticker symbol
        
        Returns:
            DataFrame with OHLCV data, or None if not found
        """
        filepath = self.daily_dir / f"{symbol.replace(' ', '_')}.parquet"
        return self.load_parquet(filepath)
