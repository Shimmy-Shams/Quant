"""
Shared trading pipeline — universe selection, data fetching, signal generation.

Used by both the interactive notebook (main_alpaca_trader.ipynb)
and the headless trader (main_trader.py).
"""

import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from strategy_config import ConfigLoader
from strategies.mean_reversion import MeanReversionSignals
from backtest.engine import BacktestConfig
from connection.alpaca_connection import AlpacaConnection, TradingMode
from data.alpaca_data import AlpacaDataAdapter
from execution.alpaca_executor import AlpacaExecutor
from execution.simulation import SimulationEngine

logger = logging.getLogger(__name__)

# Default project root: two levels up from this file (src/trading/pipeline.py → project/)
_DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ═══════════════════════════════════════════════════════════════════════════
# UNIVERSE SELECTION
# ═══════════════════════════════════════════════════════════════════════════

def select_universe(
    config: ConfigLoader,
    max_symbols: int = 30,
    project_root: Optional[Path] = None,
) -> List[str]:
    """
    Pick the top mean-reverting symbols from cached Hurst data,
    falling back to DOW 30 if no cache exists.

    This is the STARTUP selection — called once when the process starts.
    During live trading, refresh_universe_hurst() updates the universe
    daily with fresh Alpaca data.

    Args:
        config: Application configuration loader.
        max_symbols: Maximum number of symbols to include.
        project_root: Project root directory (for locating data/).

    Returns:
        List of ticker symbols.
    """
    from data.universe_builder import SP500_CORE, NASDAQ_100_CORE, DOW_30, RUSSELL_2000_CORE

    root = project_root or _DEFAULT_PROJECT_ROOT
    hurst_cache = root / "data" / "snapshots" / "hurst_rankings.csv"
    daily_dir = root / "data" / "historical" / "daily"

    # Option A: cached Hurst CSV
    if hurst_cache.exists():
        hurst_df = pd.read_csv(hurst_cache)
        universe = hurst_df.nsmallest(max_symbols, "hurst_exponent")["symbol"].tolist()
        logger.info(f"Loaded Hurst cache → {len(universe)} symbols")
        return universe

    # Option B: compute from local parquet files
    if daily_dir.exists() and list(daily_dir.glob("*.parquet")):
        logger.info("Computing Hurst exponents from local parquet files...")
        signal_gen = MeanReversionSignals(config.to_signal_config())
        hurst_results: Dict[str, float] = {}

        for pf in sorted(daily_dir.glob("*.parquet")):
            try:
                df = pd.read_parquet(pf)
                prices = df["close"] if "close" in df.columns else df.iloc[:, 0]
                if len(prices) >= 100:
                    h = signal_gen.calculate_hurst_exponent(prices)
                    if h is not None and h < 0.5:
                        hurst_results[pf.stem] = h
            except Exception:
                pass

        if hurst_results:
            ranked = sorted(hurst_results.items(), key=lambda x: x[1])
            universe = [s for s, _ in ranked[:max_symbols]]
            logger.info(f"Computed Hurst for {len(hurst_results)} symbols → top {len(universe)}")
            return universe

    # Fallback: DOW 30
    universe = DOW_30[:max_symbols]
    logger.warning(f"No Hurst data — falling back to DOW {len(universe)}")
    return universe


def refresh_universe_hurst(
    adapter: AlpacaDataAdapter,
    config: ConfigLoader,
    max_symbols: int = 60,
    project_root: Optional[Path] = None,
) -> List[str]:
    """
    Recompute Hurst exponents from FRESH Alpaca data for ALL candidate
    symbols, then return the top mean-reverters.

    Called at the start of each daily cycle so the universe evolves
    as stocks' mean-reversion characteristics change over time.

    Steps:
      1. Build the full candidate list (~269 symbols)
      2. Fetch 500 days of daily bars for all candidates from Alpaca
      3. Compute Hurst exponent for each (needs ≥100 data points)
      4. Keep only H < 0.5 (mean-reverting)
      5. Rank by H ascending, take top max_symbols
      6. Save updated hurst_rankings.csv for dashboard & next startup

    Args:
        adapter: Alpaca data adapter for fetching bars.
        config: Application configuration.
        max_symbols: Maximum universe size.
        project_root: Project root directory.

    Returns:
        Refreshed universe list.
    """
    from data.universe_builder import SP500_CORE, NASDAQ_100_CORE, DOW_30, RUSSELL_2000_CORE

    root = project_root or _DEFAULT_PROJECT_ROOT
    t0 = time.perf_counter()

    # 1. Full candidate pool
    all_candidates = sorted(set(SP500_CORE + NASDAQ_100_CORE + DOW_30 + RUSSELL_2000_CORE))
    logger.info(f"Hurst refresh: {len(all_candidates)} candidates")

    # 2. Fetch data for all candidates (cache-aware, 500-day window)
    try:
        price_df_all, _, _ = adapter.fetch_pipeline_data(
            symbols=all_candidates,
            lookback_days=500,
            use_cache=True,
            verbose=False,
        )
    except Exception as e:
        logger.warning(f"Hurst refresh fetch failed: {e} — keeping current universe")
        return select_universe(config, max_symbols=max_symbols, project_root=root)

    # 3–4. Compute Hurst for each symbol
    signal_gen = MeanReversionSignals(config.to_signal_config())
    hurst_results: Dict[str, float] = {}

    for sym in price_df_all.columns:
        prices = price_df_all[sym].dropna()
        if len(prices) < 100:
            continue
        try:
            h = signal_gen.calculate_hurst_exponent(prices)
            if h is not None and h < 0.5:
                hurst_results[sym] = h
        except Exception:
            pass

    if not hurst_results:
        logger.warning("Hurst refresh: no mean-reverting symbols found — keeping current universe")
        return select_universe(config, max_symbols=max_symbols, project_root=root)

    # 5. Rank and select top N
    ranked = sorted(hurst_results.items(), key=lambda x: x[1])
    universe = [s for s, _ in ranked[:max_symbols]]

    # 6. Save updated rankings CSV
    hurst_cache = root / "data" / "snapshots" / "hurst_rankings.csv"
    hurst_cache.parent.mkdir(parents=True, exist_ok=True)
    hurst_df = pd.DataFrame(ranked, columns=["symbol", "hurst_exponent"])
    hurst_df.to_csv(hurst_cache, index=False)

    elapsed = time.perf_counter() - t0
    logger.info(
        f"Hurst refresh done in {elapsed:.1f}s: "
        f"{len(hurst_results)}/{len(price_df_all.columns)} mean-reverting → top {len(universe)} selected  "
        f"(H range: {ranked[0][1]:.3f}–{ranked[min(len(ranked)-1, max_symbols-1)][1]:.3f})"
    )

    return universe


# ═══════════════════════════════════════════════════════════════════════════
# DATA FETCHING
# ═══════════════════════════════════════════════════════════════════════════

def fetch_data(
    adapter: AlpacaDataAdapter,
    universe: List[str],
    lookback_days: int = 500,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Fetch price + volume data with caching.

    Uses cache if <3 days old, otherwise fetches from Alpaca API.

    Args:
        adapter: Alpaca data adapter.
        universe: List of ticker symbols.
        lookback_days: Number of calendar days of history.

    Returns:
        (price_df, volume_df, raw_bars) tuple.
    """
    cache_dir = adapter.cache_dir

    # Try recent cache first (< 3 days old → skip API)
    cache_files = list(cache_dir.glob("*.parquet")) if cache_dir.exists() else []
    if cache_files:
        latest_cache = max(cache_files, key=lambda p: p.stat().st_mtime)
        age_days = (time.time() - latest_cache.stat().st_mtime) / 86400
        if age_days <= 3:
            logger.info(f"Cache is {age_days:.1f}d old — loading from disk")
            cached = adapter.load_cache(universe, label="latest")
            if cached:
                all_prices, all_volumes = {}, {}
                for sym, df in cached.items():
                    if len(df) >= 100:
                        all_prices[sym] = df["close"]
                        all_volumes[sym] = df["volume"]
                if all_prices:
                    return pd.DataFrame(all_prices), pd.DataFrame(all_volumes), cached

    logger.info(f"Fetching pipeline data for {len(universe)} symbols ({lookback_days}d)...")
    price_df, volume_df, raw_bars = adapter.fetch_pipeline_data(
        symbols=universe,
        lookback_days=lookback_days,
        use_cache=True,
        verbose=False,
    )
    adapter.save_cache(raw_bars, label="latest")
    return price_df, volume_df, raw_bars


# ═══════════════════════════════════════════════════════════════════════════
# SIGNAL GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def generate_signals(
    config: ConfigLoader,
    price_df: pd.DataFrame,
    volume_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run signal pipeline for all symbols.

    Args:
        config: Application configuration.
        price_df: Price DataFrame (date × symbol).
        volume_df: Volume DataFrame (date × symbol).

    Returns:
        (signal_df, zscore_df) tuple aligned to common index with price_df.
    """
    signal_config = config.to_signal_config()
    composite_weights = config.get_composite_weights()
    signal_gen = MeanReversionSignals(signal_config)

    all_signals, all_zscores = {}, {}

    for symbol in price_df.columns:
        if symbol not in volume_df.columns:
            continue
        prices = price_df[symbol].dropna()
        volumes = volume_df[symbol].dropna()
        if len(prices) < 100:
            continue

        composite, individual = signal_gen.generate_composite_signal(
            prices, volumes, weights=composite_weights
        )
        all_signals[symbol] = composite
        if "zscore" in individual:
            all_zscores[symbol] = individual["zscore"]

    signal_df = pd.DataFrame(all_signals)
    zscore_df = pd.DataFrame(all_zscores)

    # Align to common index
    common_idx = price_df.index.intersection(signal_df.index)
    signal_df = signal_df.loc[common_idx]
    zscore_df = zscore_df.loc[common_idx]

    return signal_df, zscore_df


# ═══════════════════════════════════════════════════════════════════════════
# FACTORY HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def build_executor(
    conn: AlpacaConnection,
    bt_config: BacktestConfig,
    mode: TradingMode,
) -> AlpacaExecutor:
    """
    Create an AlpacaExecutor configured for the given trading mode.

    Args:
        conn: Alpaca connection.
        bt_config: Backtest configuration (provides commission, sizing).
        mode: Trading mode (REPLAY/SHADOW/LIVE).

    Returns:
        Configured AlpacaExecutor.
    """
    return AlpacaExecutor(
        connection=conn,
        commission_pct=0.0 if mode == TradingMode.LIVE else bt_config.commission_pct,
        max_position_pct=bt_config.max_position_size,
        max_total_exposure=bt_config.max_total_exposure,
        stop_loss_pct=getattr(bt_config, "stop_loss_pct", None),
        take_profit_pct=getattr(bt_config, "take_profit_pct", None),
    )


def build_simulation(
    executor: AlpacaExecutor,
    bt_config: BacktestConfig,
    mode: TradingMode = TradingMode.SHADOW,
) -> SimulationEngine:
    """
    Create a SimulationEngine wired to the given executor.

    Args:
        executor: AlpacaExecutor instance.
        bt_config: Backtest configuration.
        mode: Trading mode (REPLAY/SHADOW/LIVE).

    Returns:
        Configured SimulationEngine.
    """
    return SimulationEngine(
        executor=executor,
        initial_capital=bt_config.initial_capital,
        commission_pct=bt_config.commission_pct if mode != TradingMode.LIVE else 0.0,
        slippage_pct=bt_config.slippage_pct if mode != TradingMode.LIVE else 0.0,
    )
