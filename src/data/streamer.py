"""
Real-time market data streaming from IB Gateway.
Subscribes to live quotes and streams tick data.
"""

import logging
import time
from datetime import datetime
from typing import Optional, List, Dict, Callable

import pandas as pd
from ib_insync import IB, Stock, Ticker, Contract, util

try:
    from data.universe_builder import get_etf_tickers
except ImportError:
    from ..data.universe_builder import get_etf_tickers


logger = logging.getLogger(__name__)


class DataStreamer:
    """
    Manages real-time market data subscriptions from IB.
    
    Provides:
    - Live quote snapshots for any symbol
    - Streaming subscriptions with callbacks
    - Consolidated quote view for watchlists
    """

    def __init__(self, ib: IB):
        """
        Args:
            ib: Connected IB instance (from IBConnection.ib)
        """
        self.ib = ib
        self.logger = logging.getLogger(__name__)
        self._subscriptions: Dict[str, Ticker] = {}

    def _make_contract(self, symbol: str, exchange: str = "SMART",
                       currency: str = "USD") -> Stock:
        """Create an IB Stock contract."""
        return Stock(symbol, exchange, currency)

    def get_snapshot(self, symbol: str) -> Optional[Dict]:
        """
        Get a single real-time quote snapshot for a symbol.

        Args:
            symbol: Ticker symbol

        Returns:
            Dictionary with quote data, or None on failure
        """
        contract = self._make_contract(symbol)
        try:
            self.ib.qualifyContracts(contract)
            ticker = self.ib.reqMktData(contract, snapshot=True)
            
            # Wait for data to arrive
            for _ in range(50):  # 5 seconds max
                self.ib.sleep(0.1)
                if ticker.last is not None or ticker.close is not None:
                    break

            result = {
                "symbol": symbol,
                "time": datetime.now().strftime("%H:%M:%S"),
                "last": ticker.last if ticker.last == ticker.last else None,  # NaN check
                "bid": ticker.bid if ticker.bid == ticker.bid else None,
                "ask": ticker.ask if ticker.ask == ticker.ask else None,
                "high": ticker.high if ticker.high == ticker.high else None,
                "low": ticker.low if ticker.low == ticker.low else None,
                "close": ticker.close if ticker.close == ticker.close else None,
                "volume": ticker.volume if ticker.volume == ticker.volume else None,
            }

            self.ib.cancelMktData(contract)
            return result

        except Exception as e:
            self.logger.error(f"[{symbol}] Snapshot failed: {e}")
            return None

    def get_quotes(self, symbols: List[str]) -> pd.DataFrame:
        """
        Get snapshot quotes for multiple symbols.

        Args:
            symbols: List of ticker symbols

        Returns:
            DataFrame with quote data for all symbols
        """
        rows = []
        total = len(symbols)
        
        print(f"Fetching quotes for {total} symbols...")
        for i, symbol in enumerate(symbols, 1):
            snapshot = self.get_snapshot(symbol)
            if snapshot:
                rows.append(snapshot)
            
            # Brief pause between requests
            if i < total:
                time.sleep(0.5)

        if rows:
            df = pd.DataFrame(rows)
            return df
        
        return pd.DataFrame()

    def subscribe(self, symbol: str) -> Optional[Ticker]:
        """
        Subscribe to streaming market data for a symbol.
        Data updates continuously until unsubscribed.

        Args:
            symbol: Ticker symbol

        Returns:
            Ticker object that updates in real-time, or None on failure
        """
        if symbol in self._subscriptions:
            self.logger.info(f"[{symbol}] Already subscribed")
            return self._subscriptions[symbol]

        contract = self._make_contract(symbol)
        try:
            self.ib.qualifyContracts(contract)
            ticker = self.ib.reqMktData(contract)
            self._subscriptions[symbol] = ticker
            self.logger.info(f"[{symbol}] Subscribed to streaming data")
            return ticker
        except Exception as e:
            self.logger.error(f"[{symbol}] Subscribe failed: {e}")
            return None

    def subscribe_many(self, symbols: List[str]) -> Dict[str, Ticker]:
        """Subscribe to multiple symbols."""
        results = {}
        for symbol in symbols:
            ticker = self.subscribe(symbol)
            if ticker:
                results[symbol] = ticker
            time.sleep(0.2)
        
        self.logger.info(f"Subscribed to {len(results)}/{len(symbols)} symbols")
        return results

    def unsubscribe(self, symbol: str):
        """Unsubscribe from streaming data for a symbol."""
        if symbol in self._subscriptions:
            contract = self._make_contract(symbol)
            self.ib.cancelMktData(contract)
            del self._subscriptions[symbol]
            self.logger.info(f"[{symbol}] Unsubscribed")

    def unsubscribe_all(self):
        """Unsubscribe from all streaming data."""
        symbols = list(self._subscriptions.keys())
        for symbol in symbols:
            self.unsubscribe(symbol)
        self.logger.info("Unsubscribed from all streams")

    def get_live_quotes(self) -> pd.DataFrame:
        """
        Get current data from all active subscriptions.
        Call this repeatedly to see live updates.

        Returns:
            DataFrame with latest quote data
        """
        if not self._subscriptions:
            return pd.DataFrame()

        # Let IB event loop process pending updates
        self.ib.sleep(0.1)

        rows = []
        for symbol, ticker in self._subscriptions.items():
            rows.append({
                "symbol": symbol,
                "last": ticker.last if ticker.last == ticker.last else None,
                "bid": ticker.bid if ticker.bid == ticker.bid else None,
                "ask": ticker.ask if ticker.ask == ticker.ask else None,
                "volume": int(ticker.volume) if ticker.volume == ticker.volume else None,
                "high": ticker.high if ticker.high == ticker.high else None,
                "low": ticker.low if ticker.low == ticker.low else None,
            })

        return pd.DataFrame(rows)

    @property
    def active_subscriptions(self) -> List[str]:
        """List of currently subscribed symbols."""
        return list(self._subscriptions.keys())

    def __repr__(self) -> str:
        count = len(self._subscriptions)
        return f"DataStreamer(active_subscriptions={count})"
