"""
Dynamic universe builder with multi-criteria filtering.

Single source of truth for all stock universe operations:
- Curated constituent lists (100 per index, 30 for Dow)
- Dynamic filtering with B+C performance optimization
- Local Parquet-based filtering (no slow IB calls for curated lists)
- Universe persistence via Parquet snapshots
"""

import logging
import time
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

import pandas as pd
from ib_insync import IB, Stock, Contract, util

logger = logging.getLogger(__name__)


# ============================================================================
# CURATED CONSTITUENT LISTS (100 per index, 30 for Dow)
# Pre-filtered for: volume > 1M, market cap > $1B, options available
# These are the fallback when live fetching is not implemented
# ============================================================================

SP500_CORE = [
    # Mega cap tech (10)
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "GOOG", "META", "TSLA", "AVGO", "ORCL",
    # Communication & Consumer (10)
    "NFLX", "DIS", "CMCSA", "VZ", "T", "TMUS", "NKE", "SBUX", "MCD", "BKNG",
    # Healthcare (20)
    "UNH", "LLY", "JNJ", "ABBV", "MRK", "TMO", "ABT", "DHR", "PFE", "BMY",
    "AMGN", "GILD", "CVS", "CI", "ISRG", "REGN", "VRTX", "ZTS", "SYK", "BSX",
    # Financials (10)
    "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "C", "SPGI", "BLK",
    # Industrials (10)
    "CAT", "BA", "HON", "UNP", "RTX", "GE", "LMT", "DE", "UPS", "ADP",
    # Consumer Discretionary (10)
    "HD", "WMT", "COST", "LOW", "TGT", "TJX", "F", "GM", "MAR", "ABNB",
    # Energy (10)
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "HES",
    # Tech hardware & software (10)
    "ADBE", "CRM", "CSCO", "ACN", "AMD", "QCOM", "INTC", "TXN", "NOW", "INTU",
    # Consumer Staples (5) + Materials (5)
    "PG", "PEP", "KO", "PM", "MO", "LIN", "APD", "SHW", "ECL", "DD",
    # Real Estate & Utilities (5)
    "AMT", "PLD", "NEE", "DUK", "SO",
]  # 100 total

NASDAQ_100_CORE = [
    # Mega tech (10)
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "GOOG", "META", "TSLA", "AVGO", "NFLX",
    # Software & cloud (20)
    "ADBE", "CRM", "ORCL", "CSCO", "INTU", "NOW", "SNPS", "CDNS", "TEAM", "WDAY",
    "DDOG", "FTNT", "PANW", "ANSS", "CRWD", "ZS", "OKTA", "MNDY", "ADSK", "ROP",
    # Semiconductors (17)
    "AMD", "QCOM", "INTC", "TXN", "AMAT", "ADI", "KLAC", "LRCX", "MCHP", "NXPI",
    "MU", "MRVL", "ON", "MPWR", "SWKS", "QRVO", "ALGN",
    # E-commerce & payments (6)
    "PYPL", "BKNG", "EBAY", "CPRT", "MELI", "DASH",
    # Biotech & healthcare (9)
    "AMGN", "GILD", "VRTX", "REGN", "BIIB", "ILMN", "MRNA", "SGEN", "EXAS",
    # Communication (6)
    "CMCSA", "TMUS", "CHTR", "ATVI", "EA", "TTWO",
    # Consumer (14)
    "COST", "SBUX", "ABNB", "MAR", "LULU", "ROST", "CTAS", "FAST", "PAYX", "ODFL",
    "PDD", "JD", "BIDU", "NTES",
    # Other (10)
    "HON", "ADP", "ISRG", "PCAR", "VRSK", "CTSH", "IDXX", "ZM", "DLTR", "KHC",
]  # ~92 unique (some overlap with S&P 500 -- that's correct)

DOW_30 = [
    "AAPL", "MSFT", "UNH", "GS", "HD", "CAT", "MCD", "AMGN", "V", "BA",
    "TRV", "HON", "AXP", "IBM", "JPM", "CRM", "CVX", "JNJ", "WMT", "PG",
    "MRK", "NKE", "MMM", "DIS", "KO", "DOW", "CSCO", "VZ", "INTC", "WBA",
]  # 30 total (all constituents)

RUSSELL_2000_CORE = [
    # High-liquidity small caps across sectors (100)
    "SMAR", "RYAN", "KVUE", "HQY", "CASY", "IBKR", "LECO", "QTWO", "TENB", "CHE",
    "ALKT", "EXPO", "NOG", "CRVL", "CADE", "CVCO", "AGM", "BMI", "PIPR", "NSIT",
    "ENSG", "SANM", "UFPI", "AAON", "PECO", "GATX", "WWD", "GTES", "CEIX", "HELE",
    "HNI", "RBC", "DY", "PRIM", "MLI", "AMWD", "PATK", "ATKR", "CRS", "CSWI",
    "SKYW", "IESC", "LSTR", "MATX", "BCPC", "FTDR", "AIT", "GFF", "HWKN", "HURN",
    "KFY", "ICFI", "SHOO", "GNTX", "SIG", "ABG", "KAI", "OMCL", "CENTA", "TRN",
    "DNOW", "CWEN", "USLM", "NHC", "SXI", "ANDE", "APOG", "WTS", "VSEC", "MUSA",
    "GVA", "ASTE", "GMS", "AEIS", "CXT", "CALM", "PLUS", "KRG", "BCC", "PPBI",
    "SFNC", "FORM", "NPO", "PLAB", "ASGN", "MTX", "ARCH", "CBT", "POWL", "ADTN",
    "CVLT", "TCS", "TILE", "FELE", "AIN", "SLGN", "CNS", "ESNT", "CSGS", "HCC",
]  # 100 total

# ETFs tracking each index (for hedging and index-level data)
INDEX_ETFS = {
    "SP500": "SPY",
    "NASDAQ": "QQQ",
    "DOW": "DIA",
    "RUSSELL": "IWM",
}

# Mapping from index name to curated list
_CURATED_LISTS = {
    "SP500": SP500_CORE,
    "NASDAQ": NASDAQ_100_CORE,
    "DOW": DOW_30,
    "RUSSELL": RUSSELL_2000_CORE,
}


# ============================================================================
# MODULE-LEVEL UTILITY FUNCTIONS
# ============================================================================

def get_universe(indices: List[str] = None) -> Dict[str, List[str]]:
    """
    Get the stock universe organized by index.

    Args:
        indices: List of index names ('SP500', 'NASDAQ', 'DOW', 'RUSSELL').
                 If None, returns all.
    """
    if indices is None:
        return dict(_CURATED_LISTS)
    return {k: v for k, v in _CURATED_LISTS.items() if k in indices}


def get_unique_tickers(indices: List[str] = None) -> List[str]:
    """
    Get deduplicated list of all tickers across selected indices.
    Always includes index ETFs.
    """
    universe = get_universe(indices)
    all_tickers = set()
    for tickers in universe.values():
        all_tickers.update(tickers)

    etfs = INDEX_ETFS.values() if indices is None else [
        INDEX_ETFS[k] for k in indices if k in INDEX_ETFS
    ]
    all_tickers.update(etfs)
    return sorted(all_tickers)


def get_etf_tickers() -> List[str]:
    """Get list of index ETF tickers: [SPY, QQQ, DIA, IWM]."""
    return list(INDEX_ETFS.values())


def summary() -> str:
    """Print universe summary."""
    universe = get_universe()
    unique = get_unique_tickers()
    lines = ["STOCK UNIVERSE SUMMARY", "=" * 40]
    for name, tickers in universe.items():
        etf = INDEX_ETFS.get(name, "N/A")
        lines.append(f"  {name:<10} {len(tickers):>3} stocks  (ETF: {etf})")
    lines.append(f"  {'TOTAL':<10} {len(unique):>3} unique tickers (incl. ETFs)")
    return "\n".join(lines)


# ============================================================================
# UNIVERSE BUILDER CLASS
# ============================================================================

class UniverseBuilder:
    """
    Dynamically builds a filtered stock universe from index constituents.

    Performance optimization (B+C approach):
    - B: Curated lists are pre-filtered -- skip slow IB API calls entirely
    - C: When local Parquet data exists, use it for volume/price filtering
         (instant local reads vs 2-3 sec per stock via IB)

    IB API calls are only made when:
    - Live constituent fetching is implemented (future)
    - Options availability needs verification (batch qualified)
    """

    # Index symbols for constituent lookup
    INDEX_SYMBOLS = {
        "SP500": "SPX",
        "NASDAQ": "NDX",
        "DOW": "DJI",
        "RUSSELL": "RUT",
    }

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

        self.universe_dir = self.data_dir / "universe"
        self.daily_dir = self.data_dir / "historical" / "daily"
        self.universe_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Constituent fetching
    # ------------------------------------------------------------------

    def get_index_constituents(self, index_name: str) -> List[str]:
        """
        Get constituents for an index.

        Currently returns curated lists (already 100 per index).
        When live fetching is implemented, this will pull from
        Wikipedia / financial APIs / IB contract search.

        Args:
            index_name: One of 'SP500', 'NASDAQ', 'DOW', 'RUSSELL'

        Returns:
            List of ticker symbols (exactly 100, or 30 for Dow)
        """
        # TODO: Implement live constituent fetching here
        # When implemented, this method should:
        # 1. Fetch live list from source
        # 2. Apply local Parquet filtering (approach C)
        # 3. Fall back to curated list on failure

        constituents = _CURATED_LISTS.get(index_name, [])

        if not constituents:
            self.logger.warning(f"Unknown index: {index_name}")

        return constituents

    # ------------------------------------------------------------------
    # Local Parquet-based filtering (Approach C -- fast)
    # ------------------------------------------------------------------

    def _filter_from_local_data(
        self,
        symbols: List[str],
        min_volume: float = 1_000_000,
        min_price: float = 10.0,
        max_price: float = 500.0,
    ) -> pd.DataFrame:
        """
        Filter and rank symbols using locally stored Parquet data.
        No IB API calls -- runs in seconds, not minutes.

        Reads last 20 trading days from each symbol's daily Parquet file
        to compute average volume and latest price.

        Args:
            symbols: List of ticker symbols to evaluate
            min_volume: Minimum 20-day average volume
            min_price: Minimum latest close price
            max_price: Maximum latest close price

        Returns:
            DataFrame with columns: symbol, price, avg_volume, volume_score, rank
        """
        results = []

        for symbol in symbols:
            filepath = self.daily_dir / f"{symbol.replace(' ', '_')}.parquet"

            if not filepath.exists():
                # No local data -- include with unknown metrics
                # (will be collected later, trust curated list)
                results.append({
                    "symbol": symbol,
                    "price": 0.0,
                    "avg_volume": 0.0,
                    "volume_score": 0.0,
                    "has_local_data": False,
                })
                continue

            try:
                df = pd.read_parquet(filepath, engine="pyarrow")
                if df.empty:
                    continue

                # Use last 20 trading days for averages
                recent = df.tail(20)
                avg_vol = recent["volume"].mean() if "volume" in recent.columns else 0
                last_close = recent["close"].iloc[-1] if "close" in recent.columns else 0

                # Apply price filter
                if last_close < min_price or last_close > max_price:
                    continue

                # Apply volume filter
                if avg_vol < min_volume:
                    continue

                results.append({
                    "symbol": symbol,
                    "price": round(last_close, 2),
                    "avg_volume": round(avg_vol, 0),
                    "volume_score": round(avg_vol / 1_000_000, 2),
                    "has_local_data": True,
                })

            except Exception as e:
                self.logger.warning(f"[{symbol}] Error reading local data: {e}")
                continue

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)
        df = df.sort_values("volume_score", ascending=False).reset_index(drop=True)
        df["rank"] = range(1, len(df) + 1)

        return df

    # ------------------------------------------------------------------
    # IB-based filtering (for future live fetching)
    # ------------------------------------------------------------------

    def _filter_from_ib(
        self,
        symbols: List[str],
        min_volume: float = 1_000_000,
        min_price: float = 10.0,
        max_price: float = 500.0,
        require_options: bool = True,
    ) -> pd.DataFrame:
        """
        Filter symbols using IB API calls. Slower but works without local data.
        Uses batch contract qualification for speed.

        Only called when local data is unavailable AND live fetching is active.
        """
        results = []
        total = len(symbols)

        print(f"  IB filtering {total} symbols (batch qualify)...")

        # Batch qualify all contracts at once (much faster than one-by-one)
        contracts = [Stock(s, "SMART", "USD") for s in symbols]
        try:
            qualified = self.ib.qualifyContracts(*contracts)
        except Exception as e:
            self.logger.error(f"Batch qualify failed: {e}")
            qualified = []

        qualified_map = {c.symbol: c for c in qualified if c.conId}

        for i, symbol in enumerate(symbols, 1):
            if symbol not in qualified_map:
                continue

            contract = qualified_map[symbol]

            try:
                # Get market data snapshot
                self.ib.reqMktData(contract, "", snapshot=True)
                self.ib.sleep(0.5)

                tickers = self.ib.reqTickers(contract)
                if not tickers:
                    self.ib.cancelMktData(contract)
                    continue

                ticker = tickers[0]
                price = ticker.last if (ticker.last and ticker.last == ticker.last) else ticker.close
                avg_vol = ticker.avVolume or ticker.volume or 0

                self.ib.cancelMktData(contract)

                if not price or price != price:
                    continue
                if price < min_price or price > max_price:
                    continue
                if avg_vol < min_volume:
                    continue

                results.append({
                    "symbol": symbol,
                    "price": round(price, 2),
                    "avg_volume": round(avg_vol, 0),
                    "volume_score": round(avg_vol / 1_000_000, 2),
                    "has_local_data": False,
                })

            except Exception as e:
                self.logger.warning(f"[{symbol}] IB filter error: {e}")
                continue

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)
        df = df.sort_values("volume_score", ascending=False).reset_index(drop=True)
        df["rank"] = range(1, len(df) + 1)
        return df

    # ------------------------------------------------------------------
    # Build universe
    # ------------------------------------------------------------------

    def build_universe(
        self,
        indices: List[str] = None,
        top_n_per_index: int = 100,
        save: bool = True,
        use_local_data: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        Build filtered universe from multiple indices.

        Performance optimization: Deduplicates tickers across indices and filters
        each unique ticker only ONCE (no redundant processing).

        Performance approach:
        - If local Parquet data exists: filter locally (seconds)
        - If no local data: trust curated lists as-is (instant)
        - IB API filtering only when live fetching is implemented

        Args:
            indices: List of index names. If None, uses all.
            top_n_per_index: Max stocks per index (100 default)
            save: Save results as Parquet
            use_local_data: Use local Parquet files for filtering (approach C)

        Returns:
            Dictionary mapping index name to DataFrame of selected stocks
        """
        if indices is None:
            indices = list(self.INDEX_SYMBOLS.keys())

        print(f"\nBuilding universe from {len(indices)} indices")
        print(f"Target: Top {top_n_per_index} per index")
        print("=" * 60)

        # ----------------------------------------------------------------
        # PHASE 1: Collect all constituents and build index membership map
        # ----------------------------------------------------------------
        index_constituents = {}
        all_tickers = set()
        
        for index_name in indices:
            constituents = self.get_index_constituents(index_name)
            index_constituents[index_name] = constituents
            all_tickers.update(constituents)

        unique_tickers = sorted(all_tickers)
        
        print(f"\nPhase 1: Constituent collection")
        print(f"  Total constituents: {sum(len(c) for c in index_constituents.values())}")
        print(f"  Unique tickers: {len(unique_tickers)}")
        print(f"  Deduplication saved: {sum(len(c) for c in index_constituents.values()) - len(unique_tickers)} redundant lookups")

        # ----------------------------------------------------------------
        # PHASE 2: Filter all unique tickers ONCE (no duplicates)
        # ----------------------------------------------------------------
        print(f"\nPhase 2: Filtering {len(unique_tickers)} unique tickers...")
        
        if use_local_data:
            # Check how many have local data
            has_data = sum(
                1 for s in unique_tickers
                if (self.daily_dir / f"{s.replace(' ', '_')}.parquet").exists()
            )
            
            has_data_pct = (has_data / len(unique_tickers)) * 100 if unique_tickers else 0
            
            # Always use local filtering - it handles mixed scenarios gracefully
            # Tickers WITH data: get real volume/price stats
            # Tickers WITHOUT data: included with zeros, rank last
            print(f"  Local data coverage: {has_data}/{len(unique_tickers)} ({has_data_pct:.0f}%)")
            all_filtered = self._filter_from_local_data(unique_tickers)
        else:
            # No local data mode - include all tickers with placeholder stats
            print(f"  Local data disabled - using curated order")
            all_filtered = pd.DataFrame({
                "symbol": unique_tickers,
                "price": 0.0,
                "avg_volume": 0.0,
                "volume_score": 0.0,
                "has_local_data": False,
            })
            all_filtered["rank"] = range(1, len(all_filtered) + 1)

        # Create lookup dict for fast access
        ticker_data = {row["symbol"]: row for _, row in all_filtered.iterrows()}
        
        print(f"  Filtered: {len(all_filtered)} tickers passed criteria")

        # ----------------------------------------------------------------
        # PHASE 3: Build per-index universe from filtered results
        # ----------------------------------------------------------------
        print(f"\nPhase 3: Building per-index universes")
        
        universe = {}
        
        for index_name in indices:
            print(f"\n[{index_name}]")
            
            constituents = index_constituents[index_name]
            print(f"  Constituents: {len(constituents)}")
            
            # Subset filtered data to this index's constituents
            index_filtered = []
            for ticker in constituents:
                if ticker in ticker_data:
                    row = ticker_data[ticker].copy()
                    index_filtered.append(row)
            
            if not index_filtered:
                print(f"  No stocks passed filters")
                continue
            
            # Convert to DataFrame and re-rank within this index
            filtered_df = pd.DataFrame(index_filtered)
            filtered_df = filtered_df.sort_values("volume_score", ascending=False).reset_index(drop=True)
            filtered_df["rank"] = range(1, len(filtered_df) + 1)
            
            # Take top N
            selected = filtered_df.head(top_n_per_index).copy()
            selected["index"] = index_name
            
            universe[index_name] = selected
            print(f"  Selected: {len(selected)} stocks")
            
            if save:
                timestamp = datetime.now().strftime("%Y%m%d")
                filepath = self.universe_dir / f"{index_name}_{timestamp}.parquet"
                selected.to_parquet(filepath, index=False, engine="pyarrow")
                size_kb = filepath.stat().st_size / 1024
                print(f"  Saved: {filepath.name} ({size_kb:.1f} KB)")

        total_stocks = sum(len(df) for df in universe.values())
        print("\n" + "=" * 60)
        print(f"Complete: {total_stocks} total stocks across {len(universe)} indices")
        print(f"Unique tickers: {len(set(t for df in universe.values() for t in df['symbol']))}")

        return universe

    # ------------------------------------------------------------------
    # Universe access
    # ------------------------------------------------------------------

    def get_universe_tickers(self, include_etfs: bool = True) -> List[str]:
        """
        Get list of all tickers in the built universe.
        Falls back to curated lists if no universe files exist.

        Args:
            include_etfs: Include index ETFs (SPY, QQQ, DIA, IWM)

        Returns:
            Sorted list of unique ticker symbols
        """
        # Try loading from built universe files
        universe_files = sorted(self.universe_dir.glob("*_*.parquet"))

        if universe_files:
            # Group by index, take latest for each
            latest_files = {}
            for f in universe_files:
                parts = f.stem.split("_")
                index_name = "_".join(parts[:-1])
                date = parts[-1]
                if index_name not in latest_files or date > latest_files[index_name][1]:
                    latest_files[index_name] = (f, date)

            all_tickers = set()
            for filepath, _ in latest_files.values():
                df = pd.read_parquet(filepath, engine="pyarrow")
                all_tickers.update(df["symbol"].tolist())
        else:
            # Fallback to curated lists
            self.logger.info("No universe files found -- using curated lists")
            all_tickers = set(get_unique_tickers())

        if include_etfs:
            all_tickers.update(INDEX_ETFS.values())

        return sorted(all_tickers)

    def summary(self) -> str:
        """Get summary of current universe."""
        tickers = self.get_universe_tickers(include_etfs=False)
        etfs = list(INDEX_ETFS.values())

        lines = ["DYNAMIC UNIVERSE SUMMARY", "=" * 40]
        lines.append(f"  Total Stocks:  {len(tickers)}")
        lines.append(f"  Index ETFs:    {len(etfs)}")
        lines.append(f"  Total Tickers: {len(tickers) + len(etfs)}")
        lines.append(f"  Storage:       {self.universe_dir}")

        # Show breakdown by index from saved files
        universe_files = sorted(self.universe_dir.glob("*_*.parquet"))
        if universe_files:
            lines.append("\nBreakdown (from saved universe):")
            latest = {}
            for f in universe_files:
                parts = f.stem.split("_")
                idx = "_".join(parts[:-1])
                date = parts[-1]
                if idx not in latest or date > latest[idx][1]:
                    latest[idx] = (f, date)

            for idx, (filepath, date) in sorted(latest.items()):
                df = pd.read_parquet(filepath, engine="pyarrow")
                lines.append(f"  {idx:<10} {len(df):>3} stocks (as of {date})")
        else:
            lines.append("\nBreakdown (from curated lists):")
            for idx, tickers_list in _CURATED_LISTS.items():
                lines.append(f"  {idx:<10} {len(tickers_list):>3} stocks")

        return "\n".join(lines)
