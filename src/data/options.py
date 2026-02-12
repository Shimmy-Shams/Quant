"""
Options data collector for IB Gateway.
Fetches options chains, Greeks, and implied volatility data.
Stores as Parquet snapshots for volatility analysis.

Performance: Uses batch qualifying + batch market data requests.
Filters to ATM +/- range to avoid wasting time on illiquid deep OTM strikes.
"""

import logging
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple

import pandas as pd
from ib_insync import IB, Stock, Option, Contract, util

try:
    from data.universe_builder import get_etf_tickers
except ImportError:
    from ..data.universe_builder import get_etf_tickers


logger = logging.getLogger(__name__)


class OptionsCollector:
    """
    Collects options chain data from IB and stores as Parquet snapshots.

    Performance optimizations:
    - Batch contract qualification (1 call per expiration, not per strike)
    - Batch market data requests (request all, wait once, collect all)
    - Strike filtering (ATM +/- range, skip deep OTM)
    - Delayed data fallback for paper trading accounts

    Use cases:
    - IV rank/percentile calculation for volatility strategies
    - Options strategy selection (which strikes/expirations to trade)
    - Volatility surface analysis
    """

    def __init__(self, ib: IB, data_dir: Optional[str] = None):
        """
        Args:
            ib: Connected IB instance
            data_dir: Root directory for data storage
        """
        self.ib = ib
        self.logger = logging.getLogger(__name__)

        if data_dir is None:
            src_dir = Path(__file__).resolve().parent.parent
            project_root = src_dir.parent
            self.data_dir = project_root / "data"
        else:
            self.data_dir = Path(data_dir)

        self.options_dir = self.data_dir / "snapshots" / "options"
        self.options_dir.mkdir(parents=True, exist_ok=True)

    def _make_stock_contract(self, symbol: str) -> Stock:
        """Create a stock contract."""
        return Stock(symbol, "SMART", "USD")

    def _get_stock_price(self, symbol: str) -> Optional[float]:
        """
        Get current stock price for strike filtering.
        Tries live first, falls back to delayed, then to last close from Parquet.
        """
        contract = self._make_stock_contract(symbol)

        try:
            self.ib.qualifyContracts(contract)
            ticker = self.ib.reqMktData(contract, "", snapshot=True)
            self.ib.sleep(2)

            price = None
            if ticker.last and ticker.last == ticker.last and ticker.last > 0:
                price = ticker.last
            elif ticker.close and ticker.close == ticker.close and ticker.close > 0:
                price = ticker.close
            elif ticker.marketPrice() and ticker.marketPrice() == ticker.marketPrice():
                price = ticker.marketPrice()

            self.ib.cancelMktData(contract)

            if price and price > 0:
                return price
        except Exception as e:
            self.logger.debug(f"[{symbol}] Live price failed: {e}")

        # Fallback: try local Parquet
        try:
            src_dir = Path(__file__).resolve().parent.parent
            project_root = src_dir.parent
            parquet_path = project_root / "data" / "historical" / "daily" / f"{symbol}.parquet"
            if parquet_path.exists():
                df = pd.read_parquet(parquet_path, engine="pyarrow")
                if not df.empty and "close" in df.columns:
                    return float(df["close"].iloc[-1])
        except Exception:
            pass

        return None

    def get_option_chains(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Get available option chains (expirations and strikes) for a symbol.

        Args:
            symbol: Ticker symbol (e.g., 'AAPL')

        Returns:
            DataFrame with expirations, strikes, and exchange info
        """
        contract = self._make_stock_contract(symbol)

        try:
            self.ib.qualifyContracts(contract)
            chains = self.ib.reqSecDefOptParams(
                contract.symbol,
                "",
                contract.secType,
                contract.conId
            )

            if not chains:
                self.logger.warning(f"[{symbol}] No option chains found")
                return None

            # Parse chain data
            rows = []
            for chain in chains:
                for exp in chain.expirations:
                    rows.append({
                        "symbol": symbol,
                        "exchange": chain.exchange,
                        "expiration": exp,
                        "strike_count": len(chain.strikes),
                        "min_strike": min(chain.strikes) if chain.strikes else None,
                        "max_strike": max(chain.strikes) if chain.strikes else None,
                    })

            df = pd.DataFrame(rows)
            self.logger.info(f"[{symbol}] Found {len(df)} option expirations")
            return df

        except Exception as e:
            self.logger.error(f"[{symbol}] Error fetching option chains: {e}")
            return None

    def _filter_strikes(
        self,
        strikes: list,
        stock_price: float,
        range_pct: float = 0.20,
    ) -> list:
        """
        Filter strikes to ATM +/- range.
        E.g., stock at $600, range 20% -> keep strikes $480-$720.
        This cuts 180+ strikes down to ~30-40 relevant ones.
        """
        if not stock_price or stock_price <= 0:
            return list(strikes)  # Can't filter without price

        min_strike = stock_price * (1 - range_pct)
        max_strike = stock_price * (1 + range_pct)
        filtered = [s for s in strikes if min_strike <= s <= max_strike]

        return filtered if filtered else list(strikes)  # Fallback to all if filter too aggressive

    def fetch_options_for_expiration(
        self,
        symbol: str,
        expiration: str,
        rights: List[str] = None,
        exchange: str = "SMART",
        stock_price: float = None,
        strike_range_pct: float = 0.20,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch options data for a specific expiration using BATCH operations.

        Performance: Qualifies all contracts at once, then requests all market
        data simultaneously, waits once, and collects all results.

        Args:
            symbol: Ticker symbol
            expiration: Expiration date (YYYYMMDD format)
            rights: List of option types, e.g. ['C', 'P']. Default both.
            exchange: Exchange (default 'SMART')
            stock_price: Current stock price (for strike filtering). Auto-fetched if None.
            strike_range_pct: ATM +/- range (0.20 = 20%)

        Returns:
            DataFrame with option chain data including Greeks and IV
        """
        if rights is None:
            rights = ["C", "P"]

        stock_contract = self._make_stock_contract(symbol)

        try:
            self.ib.qualifyContracts(stock_contract)

            # Get available strikes for this expiration
            chains = self.ib.reqSecDefOptParams(
                stock_contract.symbol, "", stock_contract.secType, stock_contract.conId
            )

            all_strikes = None
            for chain in chains:
                if expiration in chain.expirations and chain.exchange == exchange:
                    all_strikes = sorted(chain.strikes)
                    break

            # Fallback: try SMART exchange if exact match not found
            if not all_strikes:
                for chain in chains:
                    if expiration in chain.expirations:
                        all_strikes = sorted(chain.strikes)
                        exchange = chain.exchange
                        break

            if not all_strikes:
                self.logger.warning(f"[{symbol}] No strikes found for {expiration}")
                return None

            # Filter strikes to ATM range (huge performance gain)
            strikes = self._filter_strikes(all_strikes, stock_price, strike_range_pct)

            self.logger.info(
                f"[{symbol}] {expiration}: {len(strikes)} strikes "
                f"(filtered from {len(all_strikes)}, ATM +/-{strike_range_pct:.0%})"
            )

            # ----- BATCH QUALIFY all option contracts at once -----
            contracts = []
            for right in rights:
                for strike in strikes:
                    contracts.append(Option(symbol, expiration, strike, right, exchange))

            qualified = self.ib.qualifyContracts(*contracts)
            valid_contracts = [c for c in qualified if c.conId]

            if not valid_contracts:
                self.logger.warning(f"[{symbol}] No valid contracts for {expiration}")
                return None

            self.logger.info(
                f"[{symbol}] Qualified: {len(valid_contracts)}/{len(contracts)} contracts"
            )

            # ----- BATCH REQUEST market data for all contracts -----
            tickers_map = {}
            for contract in valid_contracts:
                ticker = self.ib.reqMktData(contract, "", snapshot=True)
                tickers_map[contract] = ticker

            # Single wait for all snapshots (instead of per-strike waits)
            # Give IB time to send all snapshots back
            wait_time = min(max(len(valid_contracts) * 0.15, 3), 15)
            self.ib.sleep(wait_time)

            # ----- COLLECT results from all tickers -----
            rows = []
            for contract, ticker in tickers_map.items():
                row = {
                    "symbol": symbol,
                    "expiration": expiration,
                    "strike": contract.strike,
                    "right": contract.right,
                    "last": ticker.last if (ticker.last and ticker.last == ticker.last) else None,
                    "bid": ticker.bid if (ticker.bid and ticker.bid == ticker.bid and ticker.bid >= 0) else None,
                    "ask": ticker.ask if (ticker.ask and ticker.ask == ticker.ask and ticker.ask >= 0) else None,
                    "close": ticker.close if (ticker.close and ticker.close == ticker.close) else None,
                    "volume": int(ticker.volume) if (ticker.volume and ticker.volume == ticker.volume and ticker.volume >= 0) else None,
                    "open_interest": int(ticker.avVolume) if (ticker.avVolume and ticker.avVolume == ticker.avVolume) else None,
                }

                # Greeks and IV
                greeks = ticker.modelGreeks
                if greeks:
                    row["iv"] = greeks.impliedVol if (greeks.impliedVol and greeks.impliedVol == greeks.impliedVol) else None
                    row["delta"] = greeks.delta if (greeks.delta and greeks.delta == greeks.delta) else None
                    row["gamma"] = greeks.gamma if (greeks.gamma and greeks.gamma == greeks.gamma) else None
                    row["theta"] = greeks.theta if (greeks.theta and greeks.theta == greeks.theta) else None
                    row["vega"] = greeks.vega if (greeks.vega and greeks.vega == greeks.vega) else None

                rows.append(row)
                self.ib.cancelMktData(contract)

            df = pd.DataFrame(rows)
            df["timestamp"] = datetime.now()

            # Count how many have actual data vs None
            has_data = df["bid"].notna().sum() + df["last"].notna().sum()
            self.logger.info(
                f"[{symbol}] {expiration}: {len(df)} contracts, "
                f"{has_data} with market data"
            )

            return df

        except Exception as e:
            self.logger.error(f"[{symbol}] Error fetching options for {expiration}: {e}")
            return None

    def collect_options_snapshot(
        self,
        symbols: List[str] = None,
        expirations_count: int = 3,
        include_calls: bool = True,
        include_puts: bool = True,
        strike_range_pct: float = 0.20,
    ) -> Dict[str, pd.DataFrame]:
        """
        Collect options snapshot for multiple symbols using batch operations.

        Performance vs old approach:
        - Old: ~36 min per symbol (sequential qualify + data per strike)
        - New: ~30-60 sec per symbol (batch qualify + batch data + strike filtering)

        Args:
            symbols: List of symbols. If None, uses ETFs.
            expirations_count: Number of nearest expirations to collect (default 3)
            include_calls: Include call options
            include_puts: Include put options
            strike_range_pct: ATM +/- range for strike filtering (0.20 = 20%)

        Returns:
            Dictionary mapping symbol to DataFrame with options data
        """
        if symbols is None:
            symbols = get_etf_tickers()

        # Try delayed data first (works without market data subscriptions)
        try:
            self.ib.reqMarketDataType(3)  # 3 = delayed data
            self.logger.info("Requesting delayed market data (type 3)")
        except Exception:
            pass

        rights = []
        if include_calls:
            rights.append("C")
        if include_puts:
            rights.append("P")

        results = {}

        print(f"\nCollecting options snapshots for {len(symbols)} symbols")
        print(f"  Strike range: ATM +/- {strike_range_pct:.0%}")
        print(f"  Expirations: nearest {expirations_count}")
        print(f"  Rights: {', '.join('Calls' if r == 'C' else 'Puts' for r in rights)}")
        print("=" * 60)

        total_start = time.time()

        for i, symbol in enumerate(symbols, 1):
            sym_start = time.time()
            print(f"\n[{i}/{len(symbols)}] {symbol}")

            # Get current price for strike filtering
            stock_price = self._get_stock_price(symbol)
            if stock_price:
                print(f"  Price: ${stock_price:.2f} (strikes: ${stock_price*(1-strike_range_pct):.0f}-${stock_price*(1+strike_range_pct):.0f})")
            else:
                print(f"  Price: unknown (using all strikes)")

            # Get available expirations
            chain_info = self.get_option_chains(symbol)
            if chain_info is None or len(chain_info) == 0:
                print(f"  No option chains available")
                continue

            # Take nearest expirations (skip same-day if it's after market close)
            today = datetime.now().strftime("%Y%m%d")
            expirations = sorted(chain_info["expiration"].unique())

            # Prefer expirations at least 1 day out for better data
            future_exps = [e for e in expirations if e > today]
            if len(future_exps) >= expirations_count:
                expirations = future_exps[:expirations_count]
            else:
                expirations = expirations[:expirations_count]

            print(f"  Expirations: {', '.join(expirations)}")

            symbol_data = []
            for exp in expirations:
                exp_data = self.fetch_options_for_expiration(
                    symbol=symbol,
                    expiration=exp,
                    rights=rights,
                    stock_price=stock_price,
                    strike_range_pct=strike_range_pct,
                )

                if exp_data is not None and len(exp_data) > 0:
                    symbol_data.append(exp_data)
                    calls_n = len(exp_data[exp_data["right"] == "C"]) if "C" in rights else 0
                    puts_n = len(exp_data[exp_data["right"] == "P"]) if "P" in rights else 0
                    has_bid = exp_data["bid"].notna().sum()
                    print(f"  {exp}: {calls_n}C + {puts_n}P = {len(exp_data)} contracts ({has_bid} with bids)")

                time.sleep(0.5)  # Brief pause between expirations

            if symbol_data:
                combined = pd.concat(symbol_data, ignore_index=True)
                results[symbol] = combined

                # Save snapshot
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = self.options_dir / f"{symbol}_{timestamp}.parquet"
                combined.to_parquet(filepath, index=False, engine="pyarrow")

                size_kb = filepath.stat().st_size / 1024
                sym_elapsed = time.time() - sym_start
                print(f"  Saved: {len(combined)} contracts ({size_kb:.1f} KB) in {sym_elapsed:.1f}s")

        # Restore live data type
        try:
            self.ib.reqMarketDataType(1)
        except Exception:
            pass

        total_elapsed = time.time() - total_start
        print("\n" + "=" * 60)
        print(f"Complete: {len(results)}/{len(symbols)} symbols in {total_elapsed:.0f}s")

        # Summary of data quality
        total_contracts = sum(len(df) for df in results.values())
        total_with_data = sum(df["bid"].notna().sum() for df in results.values())
        print(f"Total contracts: {total_contracts} ({total_with_data} with market data)")
        if total_with_data == 0:
            print("\n[!] No market data received.")
            print("    Paper accounts need market data subscriptions for options.")
            print("    Go to: Account Management -> Market Data Subscriptions")
            print("    Minimum: 'US Securities Snapshot and Futures Value Bundle' (~$10/mo)")

        return results

    def calculate_iv_rank(self, symbol: str, days_lookback: int = 252) -> Optional[Dict]:
        """
        Calculate IV rank and percentile for a symbol.
        Requires historical snapshots.

        Args:
            symbol: Ticker symbol
            days_lookback: Historical period for IV rank calculation

        Returns:
            Dictionary with current IV, IV rank, IV percentile
        """
        # Load recent snapshots
        snapshot_files = sorted(
            self.options_dir.glob(f"{symbol}_*.parquet")
        )

        if not snapshot_files:
            self.logger.warning(f"[{symbol}] No historical snapshots for IV rank")
            return None

        # Load latest
        latest = pd.read_parquet(snapshot_files[-1])
        
        # Get ATM IV (average of calls and puts near current price)
        # This is a simplified calculation - production would use more sophisticated methods
        current_iv = latest[latest["iv"].notna()]["iv"].median()

        # Load historical IVs (if available)
        historical_ivs = []
        for f in snapshot_files[-days_lookback:]:
            df = pd.read_parquet(f)
            iv_median = df[df["iv"].notna()]["iv"].median()
            if iv_median == iv_median:  # NaN check
                historical_ivs.append(iv_median)

        if not historical_ivs or current_iv != current_iv:
            return None

        # Calculate rank and percentile
        iv_min = min(historical_ivs)
        iv_max = max(historical_ivs)
        iv_rank = (current_iv - iv_min) / (iv_max - iv_min) if iv_max != iv_min else 0.5
        iv_percentile = sum(1 for iv in historical_ivs if iv < current_iv) / len(historical_ivs)

        return {
            "symbol": symbol,
            "current_iv": current_iv,
            "iv_min": iv_min,
            "iv_max": iv_max,
            "iv_rank": iv_rank,
            "iv_percentile": iv_percentile,
            "snapshots_count": len(historical_ivs),
        }

    def get_stored_snapshots(self) -> pd.DataFrame:
        """Get summary of stored options snapshots."""
        files = list(self.options_dir.glob("*.parquet"))
        
        if not files:
            return pd.DataFrame(columns=["symbol", "timestamp", "contracts", "size_kb"])

        rows = []
        for f in files:
            parts = f.stem.split("_")
            symbol = parts[0]
            timestamp = "_".join(parts[1:]) if len(parts) > 1 else "unknown"
            
            df = pd.read_parquet(f)
            rows.append({
                "symbol": symbol,
                "timestamp": timestamp,
                "contracts": len(df),
                "size_kb": round(f.stat().st_size / 1024, 1),
            })

        return pd.DataFrame(rows).sort_values(["symbol", "timestamp"])
