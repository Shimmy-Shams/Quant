#!/usr/bin/env python3
"""
Headless Mean-Reversion Trader — 24/7 Autonomous Operation

Standalone script for unattended deployment (Oracle Cloud, VPS, etc.).
Replaces the Jupyter notebook with a continuous trading loop.

Architecture:
  startup → data load → signal loop → graceful shutdown

Execution Timing (T+1 Open):
  Signals are generated from yesterday's close prices (fully settled overnight).
  Trades execute at 9:35 AM ET (market open + 5 min) at current market prices.
  This eliminates T+0 look-ahead bias while capturing most of the alpha.
  (Validated: Sharpe 6.00 at T+1 Open vs 6.99 at T+0 Close — 14% decay)

Modes:
  SHADOW  — Track hypothetical trades, no real orders (default)
  LIVE    — Submit real paper/live orders to Alpaca

Usage:
  python main_trader.py                    # Shadow mode (default)
  python main_trader.py --mode live        # Live-order mode
  python main_trader.py --mode shadow --interval 300  # 5-min shadow cycle

Deployment:
  # As systemd service (see deploy/ directory):
  sudo systemctl start quant-trader
"""

import sys
import os
import time
import signal
import logging
import argparse
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict

import numpy as np
import pandas as pd

# ─── Bootstrap module path ────────────────────────────────────────────────
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from strategy_config import ConfigLoader
from strategies.mean_reversion import MeanReversionSignals, SignalConfig
from backtest.engine import BacktestConfig
from connection.alpaca_connection import AlpacaConfig, AlpacaConnection, TradingMode
from data.alpaca_data import AlpacaDataAdapter
from execution.alpaca_executor import AlpacaExecutor
from execution.simulation import SimulationEngine
from execution.intraday_monitor import IntradayMonitor, IntradayMonitorConfig
from dashboard_generator import DashboardGenerator
from trading.pipeline import (
    select_universe,
    refresh_universe_hurst,
    fetch_data,
    generate_signals,
    build_executor,
    build_simulation,
)


# ═══════════════════════════════════════════════════════════════════════════
# GLOBALS
# ═══════════════════════════════════════════════════════════════════════════

SHUTDOWN_REQUESTED = False

logger = logging.getLogger("trader")


def setup_logging(log_dir: Path, level: str = "INFO") -> None:
    """Configure logging to both console and rotating file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"trader_{datetime.now():%Y%m%d}.log"

    fmt = "%(asctime)s | %(name)-12s | %(levelname)-7s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    # Root logger
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    root.addHandler(console)

    # File handler (daily rotation via name)
    fh = logging.FileHandler(log_file, mode="a")
    fh.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    root.addHandler(fh)

    logger.info(f"Logging to {log_file}")


def handle_shutdown(signum, frame):
    """Graceful shutdown handler for SIGINT / SIGTERM."""
    global SHUTDOWN_REQUESTED
    sig_name = signal.Signals(signum).name
    logger.info(f"Received {sig_name} — shutting down gracefully...")
    SHUTDOWN_REQUESTED = True


# ═══════════════════════════════════════════════════════════════════════════
# NOTE: Universe selection, data fetching, and signal generation are now
# in trading.pipeline (shared with the interactive notebook).
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
# DAILY EXECUTION LOGIC
# ═══════════════════════════════════════════════════════════════════════════


def _get_entry_dates_from_orders(conn: AlpacaConnection) -> Dict[str, pd.Timestamp]:
    """
    Query Alpaca order history to find the actual fill date for each
    currently-held position.  Returns {symbol: fill_timestamp}.

    Falls back to submitted_at if filled_at is unavailable.
    """
    entry_dates: Dict[str, pd.Timestamp] = {}
    try:
        # Fetch recent filled orders (buy-side for longs, sell-side for shorts)
        orders = conn.get_orders(status='closed', limit=100)
        # Build a map: for each symbol, find the most recent entry fill
        current_positions = {p['symbol'] for p in conn.get_positions()}
        for o in orders:
            sym = o.get('symbol')
            if sym not in current_positions or sym in entry_dates:
                continue
            if o.get('status') != 'filled':
                continue
            # Use filled_at (actual fill time), fall back to submitted_at
            ts_str = o.get('filled_at') or o.get('submitted_at', '')
            if ts_str and ts_str != 'None':
                try:
                    entry_dates[sym] = pd.Timestamp(ts_str).tz_localize(None)
                except Exception:
                    entry_dates[sym] = pd.Timestamp(ts_str[:19])
    except Exception as e:
        logger.warning(f"Could not fetch entry dates from orders: {e}")
    return entry_dates


def _retrofit_bracket_orders(
    conn: AlpacaConnection,
    bt_config: BacktestConfig,
) -> None:
    """
    On startup, check all open positions for missing stop-loss orders
    and submit standalone stop orders to protect them.

    This handles legacy positions entered before bracket order support
    was deployed, or positions whose bracket legs were cancelled.
    """
    if conn.config.trading_mode != TradingMode.LIVE:
        return

    stop_loss_pct = bt_config.stop_loss_pct
    short_sl_pct = bt_config.short_stop_loss_pct or stop_loss_pct
    take_profit_pct = bt_config.take_profit_pct

    if stop_loss_pct is None:
        return

    positions = conn.get_positions()
    if not positions:
        return

    # Get all open orders to check which positions already have stop orders
    open_orders = conn.get_orders(status='open', limit=200)
    symbols_with_stops = set()
    for o in open_orders:
        if o.get('type') in ('stop', 'stop_limit') and o.get('symbol'):
            symbols_with_stops.add(o['symbol'])

    retrofitted = 0
    for pos in positions:
        sym = pos['symbol']
        if sym in symbols_with_stops:
            continue  # Already has a stop order

        qty = abs(float(pos['qty']))
        entry_price = float(pos['avg_entry_price'])
        is_long = float(pos['qty']) > 0
        side_str = 'long' if is_long else 'short'

        # Compute stop price
        sl_pct = stop_loss_pct if is_long else short_sl_pct
        if is_long:
            stop_price = round(entry_price * (1 - sl_pct), 2)
            exit_side = 'sell'
        else:
            stop_price = round(entry_price * (1 + sl_pct), 2)
            exit_side = 'buy'

        try:
            order = conn.submit_stop_order(
                symbol=sym,
                qty=int(qty),
                side=exit_side,
                stop_price=stop_price,
            )
            logger.info(
                f"🛡️ Retrofitted stop-loss: {sym} {side_str} "
                f"qty={int(qty)} SL=${stop_price:.2f} "
                f"(entry=${entry_price:.2f}, {sl_pct:.0%}) → {order['status']}"
            )
            retrofitted += 1
        except Exception as e:
            logger.error(f"Failed to retrofit stop for {sym}: {e}")

    if retrofitted:
        logger.info(f"Retrofitted {retrofitted} stop-loss orders for unprotected positions")
    else:
        logger.info("All positions have stop-loss protection ✓")

def run_daily_cycle(
    conn: AlpacaConnection,
    adapter: AlpacaDataAdapter,
    config: ConfigLoader,
    bt_config: BacktestConfig,
    universe: list,
    mode: TradingMode,
    shadow_sim: Optional[SimulationEngine] = None,
    executor: Optional[AlpacaExecutor] = None,
) -> dict:
    """
    Execute one full trading cycle:
    1. Refresh data
    2. Generate signals
    3. Execute trades (shadow or live)

    Returns summary dict.
    """
    cycle_start = time.perf_counter()

    # ── 0. Refresh universe via fresh Hurst computation ──
    max_symbols = config.get('alpaca.max_universe_size', 60)
    universe = refresh_universe_hurst(adapter, config, max_symbols=max_symbols,
                                      project_root=PROJECT_ROOT)
    logger.info(f"Universe refreshed: {len(universe)} symbols → {', '.join(universe[:10])}...")

    # ── 1. Data (now for the refreshed universe) ──
    price_df, volume_df, raw_bars = fetch_data(adapter, universe)
    universe_active = list(price_df.columns)
    logger.info(f"Data ready: {len(universe_active)} symbols, "
                f"{len(price_df)} days ({price_df.index[-1].date()})")

    # ── 2. Signals ──
    signal_df, zscore_df = generate_signals(config, price_df, volume_df)
    today = signal_df.index[-1]
    today_signals = signal_df.loc[today].dropna()
    entry_count = (today_signals.abs() > bt_config.entry_threshold).sum()
    logger.info(f"Signals ({today.date()}): {len(today_signals)} valid, "
                f"{entry_count} above threshold")

    # ── 3. Execute ──
    result = {"date": today, "mode": mode.value}

    if mode == TradingMode.SHADOW:
        assert shadow_sim is not None
        day_result = shadow_sim.process_shadow_day(
            date=today,
            signal_df=signal_df,
            price_df=price_df,
            volume_df=volume_df,
            exit_signal_df=zscore_df,
            config=bt_config,
            verbose=True,
        )
        result.update(day_result)

        # Persist shadow state
        _save_shadow_state(shadow_sim)

    elif mode == TradingMode.LIVE:
        assert executor is not None

        # Get actual entry dates from Alpaca order history (Fix 3)
        entry_dates = _get_entry_dates_from_orders(conn)

        # Get current live positions
        current_positions: Dict[str, Dict] = {}
        for pos in conn.get_positions():
            sym = pos["symbol"]
            current_positions[sym] = {
                "qty": int(pos["qty"]),
                "side": "long" if int(pos["qty"]) > 0 else "short",
                "entry_price": float(pos["avg_entry_price"]),
                "entry_date": entry_dates.get(sym, pd.Timestamp.now() - pd.Timedelta(days=1)),
            }

        # Fetch real-time prices for exit evaluation (Fix 2)
        # Cached price_df may be stale, but live prices from Alpaca are current
        held_symbols = list(current_positions.keys())
        live_prices = conn.get_latest_trades(held_symbols) if held_symbols else {}

        # Overlay live prices onto the price DataFrame's last row
        # so exit logic sees actual current prices, not stale cache
        if live_prices:
            live_row = price_df.iloc[-1].copy()
            for sym, px in live_prices.items():
                if sym in live_row.index and px > 0:
                    live_row[sym] = px
            # Append as a new "today" row if the cache is stale
            cache_date = price_df.index[-1].date()
            real_today = pd.Timestamp.now().normalize().date()
            if cache_date < real_today:
                new_idx = pd.Timestamp(real_today)
                price_df.loc[new_idx] = live_row
                signal_df.loc[new_idx] = signal_df.iloc[-1]  # carry forward signals
                zscore_df.loc[new_idx] = zscore_df.iloc[-1]
                today = new_idx
                logger.info(
                    f"Overlaid {len(live_prices)} live prices onto stale cache "
                    f"({cache_date} → {real_today})"
                )
            else:
                # Cache is current day — just update the last row
                for sym, px in live_prices.items():
                    if sym in price_df.columns and px > 0:
                        price_df.iloc[-1][sym] = px

        decisions = executor.generate_decisions_from_signals(
            signal_df=signal_df,
            price_df=price_df,
            volume_df=volume_df,
            exit_signal_df=zscore_df,
            date=today,
            current_positions=current_positions,
            config=bt_config,
        )

        if decisions:
            current_prices = price_df.iloc[-1]
            results = executor.execute_decisions(decisions, current_prices)
            filled = sum(1 for r in results if r.status in ("filled", "submitted"))
            logger.info(f"Executed {len(decisions)} decisions → {filled} filled")
            for r in results:
                icon = "✅" if r.status in ("filled", "submitted") else "❌"
                logger.info(f"  {icon} {r.decision.symbol} {r.decision.action} "
                            f"x{r.decision.target_qty} → {r.status}")
            result["decisions"] = len(decisions)
            result["filled"] = filled
        else:
            logger.info("No trade signals today")
            result["decisions"] = 0

        # Account summary
        account = conn.get_account()
        logger.info(f"Account: ${account['portfolio_value']:,.2f} "
                     f"(cash: ${account['cash']:,.2f})")
        
        # Save live state for dashboard
        _save_live_state(conn)
        result["portfolio_value"] = account["portfolio_value"]
        result["cash"] = account["cash"]

    elapsed = time.perf_counter() - cycle_start
    result["cycle_seconds"] = round(elapsed, 1)
    logger.info(f"Cycle complete in {elapsed:.1f}s")
    return result


# ═══════════════════════════════════════════════════════════════════════════
# SHADOW STATE PERSISTENCE
# ═══════════════════════════════════════════════════════════════════════════

SHADOW_STATE_PATH = PROJECT_ROOT / "data" / "snapshots" / "shadow_state.csv"


def _save_shadow_state(sim: SimulationEngine) -> None:
    """Persist shadow positions to CSV."""
    if sim.positions:
        rows = []
        for sym, pos in sim.positions.items():
            rows.append({
                "symbol": pos.symbol,
                "qty": pos.qty,
                "side": pos.side,
                "entry_price": pos.entry_price,
                "entry_date": pos.entry_date,
                "signal_strength": pos.signal_strength,
                "current_price": pos.current_price,
            })
        pd.DataFrame(rows).to_csv(SHADOW_STATE_PATH, index=False)
        logger.info(f"Shadow state saved ({len(rows)} positions)")
    else:
        if SHADOW_STATE_PATH.exists():
            SHADOW_STATE_PATH.unlink()
        logger.info("No open positions — shadow state cleared")


# ═══════════════════════════════════════════════════════════════════════════
# SIGNAL CACHE (T+1 Two-Phase Architecture)
#
# Phase 1 (post-close ~4:10 PM): generate signals from Day T close → cache
# Phase 2 (9:35 AM T+1): load cached signals → execute trades
# ═══════════════════════════════════════════════════════════════════════════

SIGNAL_CACHE_BASE = PROJECT_ROOT / "data" / "snapshots" / "signal_cache"


def _signal_cache_dir(mode: TradingMode) -> Path:
    """Return mode-specific signal cache directory (live/ or shadow/)."""
    return SIGNAL_CACHE_BASE / mode.value


def _save_signal_cache(
    signal_df: pd.DataFrame,
    zscore_df: pd.DataFrame,
    price_df: pd.DataFrame,
    volume_df: pd.DataFrame,
    universe: list,
    mode: TradingMode = TradingMode.LIVE,
) -> None:
    """Cache signal generation results for T+1 morning execution (mode-specific)."""
    cache_dir = _signal_cache_dir(mode)
    cache_dir.mkdir(parents=True, exist_ok=True)

    signal_df.to_parquet(cache_dir / "signal_df.parquet")
    zscore_df.to_parquet(cache_dir / "zscore_df.parquet")
    price_df.to_parquet(cache_dir / "price_df.parquet")
    volume_df.to_parquet(cache_dir / "volume_df.parquet")

    metadata = {
        "signal_date": str(signal_df.index[-1].date()),
        "generated_at": datetime.now().isoformat(),
        "mode": mode.value,
        "universe": universe,
        "n_symbols": len(signal_df.columns),
        "n_days": len(signal_df),
    }
    with open(cache_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(
        f"Cached signals [{mode.value}] for {metadata['signal_date']}: "
        f"{metadata['n_symbols']} symbols, {metadata['n_days']} days"
    )


def _load_signal_cache(mode: TradingMode = TradingMode.LIVE) -> Optional[dict]:
    """
    Load cached signals for T+1 execution (mode-specific).

    Returns dict with signal_df, zscore_df, price_df, volume_df, universe,
    metadata — or None if no valid cache exists.
    """
    cache_dir = _signal_cache_dir(mode)
    meta_path = cache_dir / "metadata.json"
    if not meta_path.exists():
        logger.info(f"No signal cache found for [{mode.value}] mode")
        return None

    try:
        with open(meta_path) as f:
            metadata = json.load(f)

        # Check cache freshness (must be from last 3 calendar days)
        signal_date = pd.Timestamp(metadata["signal_date"])
        age_days = (pd.Timestamp.now().normalize() - signal_date).days
        if age_days > 3:
            logger.warning(f"Signal cache [{mode.value}] is {age_days} days old — ignoring stale cache")
            return None

        signal_df = pd.read_parquet(cache_dir / "signal_df.parquet")
        zscore_df = pd.read_parquet(cache_dir / "zscore_df.parquet")
        price_df = pd.read_parquet(cache_dir / "price_df.parquet")
        volume_df = pd.read_parquet(cache_dir / "volume_df.parquet")

        logger.info(
            f"Loaded cached signals [{mode.value}] from {metadata['signal_date']} "
            f"({metadata['n_symbols']} symbols, generated {metadata['generated_at'][:16]})"
        )

        return {
            "signal_df": signal_df,
            "zscore_df": zscore_df,
            "price_df": price_df,
            "volume_df": volume_df,
            "universe": metadata["universe"],
            "metadata": metadata,
        }
    except Exception as e:
        logger.warning(f"Failed to load signal cache [{mode.value}]: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# SIGNAL DETAIL LOGGING
# ═══════════════════════════════════════════════════════════════════════════

SIGNAL_HISTORY_PATH = PROJECT_ROOT / "data" / "snapshots" / "signal_history.json"
TRADE_HISTORY_PATH = PROJECT_ROOT / "data" / "snapshots" / "trade_history.json"


def _append_signal_history(
    date,
    mode: str,
    phase: str,
    today_signals: pd.Series,
    today_zscores: pd.Series,
    today_prices: pd.Series,
    entry_threshold: float,
) -> None:
    """
    Append today's signal snapshot to persistent signal_history.json.

    Tracked in git so local/codespace dev can access VM signal data
    for model improvement.
    """
    actionable = today_signals[today_signals.abs() > entry_threshold]

    signals_detail = []
    for sym, sig in actionable.items():
        signals_detail.append({
            "symbol": sym,
            "direction": "LONG" if sig < 0 else "SHORT",
            "signal": round(float(sig), 4),
            "z_score": round(float(today_zscores.get(sym, float('nan'))), 4),
            "price": round(float(today_prices.get(sym, float('nan'))), 2),
        })

    # Sort by absolute signal strength (strongest first)
    signals_detail.sort(key=lambda x: abs(x["signal"]), reverse=True)

    entry = {
        "date": str(date),
        "generated_at": datetime.now().isoformat()[:19],
        "mode": mode,
        "phase": phase,
        "entry_threshold": entry_threshold,
        "total_valid": int((~today_signals.isna()).sum()),
        "actionable_count": len(signals_detail),
        "signals": signals_detail,
    }

    # Load existing history
    history = []
    if SIGNAL_HISTORY_PATH.exists():
        try:
            with open(SIGNAL_HISTORY_PATH) as f:
                history = json.load(f)
        except (json.JSONDecodeError, Exception):
            history = []

    # Avoid duplicate entries for same date+mode+phase
    history = [
        h for h in history
        if not (h.get("date") == str(date) and h.get("mode") == mode and h.get("phase") == phase)
    ]

    history.append(entry)

    # Keep last 365 days of history
    history = history[-365:]

    SIGNAL_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SIGNAL_HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)

    logger.info(f"Signal history updated ({len(history)} entries)")


def _append_trade_history(
    date,
    mode: str,
    decisions_data: list,
    portfolio_value: float = 0,
    cash: float = 0,
) -> None:
    """
    Append today's trade execution results to persistent trade_history.json.

    Tracked in git for post-hoc analysis and model improvement.
    """
    entry = {
        "date": str(date),
        "executed_at": datetime.now().isoformat()[:19],
        "mode": mode,
        "decisions": decisions_data,
        "decision_count": len(decisions_data),
        "portfolio_value": round(portfolio_value, 2),
        "cash": round(cash, 2),
    }

    history = []
    if TRADE_HISTORY_PATH.exists():
        try:
            with open(TRADE_HISTORY_PATH) as f:
                history = json.load(f)
        except (json.JSONDecodeError, Exception):
            history = []

    # Avoid duplicates for same date+mode
    history = [h for h in history if not (h.get("date") == str(date) and h.get("mode") == mode)]

    history.append(entry)
    history = history[-365:]

    TRADE_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TRADE_HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)

    logger.info(f"Trade history updated ({len(history)} entries)")


def _log_signal_details(
    today_signals: pd.Series,
    today_zscores: pd.Series,
    today_prices: pd.Series,
    entry_threshold: float,
    date,
) -> None:
    """
    Log per-stock signal details for visibility in VM logs.

    Always prints a clear summary block showing:
      - Universe size, valid signals count, actionable count
      - Actionable signals with direction, strength, z-score, price
      - Top 5 nearest-to-threshold signals (even on quiet days)
    """
    valid_count = int((~today_signals.isna()).sum())
    actionable = today_signals[today_signals.abs() > entry_threshold].sort_values()

    # Always print the summary header
    logger.info(f"  {'='*52}")
    logger.info(f"  SIGNAL GENERATION SUMMARY — {date}")
    logger.info(f"  {'='*52}")
    logger.info(f"  Universe: {len(today_signals)} symbols | Valid: {valid_count} | Actionable: {len(actionable)} (threshold: {entry_threshold})")

    if len(actionable) > 0:
        # Split into LONG (negative signal = buy oversold) and SHORT (positive = sell overbought)
        longs = actionable[actionable < 0].sort_values()       # most negative first
        shorts = actionable[actionable > 0].sort_values(ascending=False)  # most positive first

        if len(longs) > 0:
            logger.info(f"  LONG candidates ({len(longs)}):")
            for sym, sig in longs.items():
                zscore = today_zscores.get(sym, float('nan'))
                price = today_prices.get(sym, float('nan'))
                logger.info(
                    f"    BUY  {sym:<6s} | signal={sig:+.3f} | z-score={zscore:+.2f} | price=${price:,.2f}"
                )

        if len(shorts) > 0:
            logger.info(f"  SHORT candidates ({len(shorts)}):")
            for sym, sig in shorts.items():
                zscore = today_zscores.get(sym, float('nan'))
                price = today_prices.get(sym, float('nan'))
                logger.info(
                    f"    SELL {sym:<6s} | signal={sig:+.3f} | z-score={zscore:+.2f} | price=${price:,.2f}"
                )

    # Always show top 5 closest-to-threshold (even when nothing is actionable)
    non_actionable = today_signals[
        (today_signals.abs() <= entry_threshold) & (~today_signals.isna())
    ]
    if len(non_actionable) > 0:
        ranked = non_actionable.reindex(
            non_actionable.abs().sort_values(ascending=False).index
        )
        top_n = ranked.head(5)
        logger.info(f"  Top {len(top_n)} nearest to threshold:")
        for sym, sig in top_n.items():
            pct = abs(sig) / entry_threshold * 100
            direction = "BUY " if sig < 0 else "SELL"
            zscore = today_zscores.get(sym, float('nan'))
            price = today_prices.get(sym, float('nan'))
            logger.info(
                f"    {direction} {sym:<6s} | signal={sig:+.3f} | z-score={zscore:+.2f} | price=${price:,.2f} | ({pct:.0f}% of threshold)"
            )
    logger.info(f"  {'='*52}")


# ═══════════════════════════════════════════════════════════════════════════
# TWO-PHASE EXECUTION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════


def run_signal_generation_phase(
    adapter: AlpacaDataAdapter,
    config: ConfigLoader,
    bt_config: BacktestConfig,
    universe: list,
    mode: TradingMode = TradingMode.LIVE,
) -> Optional[dict]:
    """
    Phase 1 (Post-Close): Generate and cache signals from today's close data.

    Called after market close (~4:10 PM ET). Results are cached for
    T+1 morning execution.

    Returns dict with signal_df, zscore_df, price_df, volume_df, universe
    or None on failure.
    """
    phase_start = time.perf_counter()
    logger.info("=" * 50)
    logger.info("SIGNAL GENERATION PHASE (Post-Close)")
    logger.info("=" * 50)

    try:
        # Step 0: Refresh universe via Hurst recomputation
        max_symbols = config.get('alpaca.max_universe_size', 60)
        universe = refresh_universe_hurst(
            adapter, config, max_symbols=max_symbols, project_root=PROJECT_ROOT
        )
        logger.info(f"Universe refreshed: {len(universe)} symbols → {', '.join(universe[:10])}...")

        # Step 1: Fetch data
        price_df, volume_df, raw_bars = fetch_data(adapter, universe)
        universe_active = list(price_df.columns)
        logger.info(
            f"Data ready: {len(universe_active)} symbols, "
            f"{len(price_df)} days ({price_df.index[-1].date()})"
        )

        # Step 2: Generate signals
        signal_df, zscore_df = generate_signals(config, price_df, volume_df)
        today = signal_df.index[-1]
        today_signals = signal_df.loc[today].dropna()
        today_zscores = zscore_df.loc[today].dropna()
        today_prices = price_df.loc[today].dropna()
        entry_count = (today_signals.abs() > bt_config.entry_threshold).sum()
        logger.info(
            f"Signals ({today.date()}): {len(today_signals)} valid, "
            f"{entry_count} above threshold ({bt_config.entry_threshold})"
        )

        # Log detailed signal breakdown
        _log_signal_details(today_signals, today_zscores, today_prices, bt_config.entry_threshold, today.date())

        # Persist to signal history (git-tracked for model improvement)
        _append_signal_history(
            date=today.date(), mode=mode.value, phase="generation",
            today_signals=today_signals, today_zscores=today_zscores,
            today_prices=today_prices, entry_threshold=bt_config.entry_threshold,
        )

        # Cache for T+1 execution (mode-specific directory)
        _save_signal_cache(signal_df, zscore_df, price_df, volume_df, universe_active, mode)

        elapsed = time.perf_counter() - phase_start
        logger.info(f"Signal generation phase complete in {elapsed:.1f}s")

        return {
            "signal_df": signal_df,
            "zscore_df": zscore_df,
            "price_df": price_df,
            "volume_df": volume_df,
            "universe": universe_active,
        }
    except Exception as e:
        logger.error(f"Signal generation phase failed: {e}", exc_info=True)
        return None


def run_trade_execution_phase(
    conn: AlpacaConnection,
    signal_cache: dict,
    bt_config: BacktestConfig,
    mode: TradingMode,
    executor: Optional[AlpacaExecutor] = None,
    shadow_sim: Optional[SimulationEngine] = None,
) -> dict:
    """
    Phase 2 (T+1 Open): Execute trades from cached signals.

    Called at 9:35 AM ET. Uses pre-computed signals from last night's
    close data, combined with current position state and live prices.

    Returns summary dict.
    """
    phase_start = time.perf_counter()
    logger.info("=" * 50)
    logger.info("TRADE EXECUTION PHASE (T+1 Open)")
    logger.info("=" * 50)

    signal_df = signal_cache["signal_df"].copy()
    zscore_df = signal_cache["zscore_df"].copy()
    price_df = signal_cache["price_df"].copy()
    volume_df = signal_cache["volume_df"].copy()

    today_signal = signal_df.index[-1]
    today_signals = signal_df.loc[today_signal].dropna()
    today_zscores = zscore_df.loc[today_signal].dropna()
    today_prices = price_df.loc[today_signal].dropna()
    entry_count = (today_signals.abs() > bt_config.entry_threshold).sum()
    logger.info(
        f"Loaded signals from {today_signal.date()}: "
        f"{len(today_signals)} valid, {entry_count} above threshold"
    )

    # Log detailed signal breakdown for execution visibility
    _log_signal_details(today_signals, today_zscores, today_prices, bt_config.entry_threshold, today_signal.date())

    # Persist to signal history (git-tracked for model improvement)
    _append_signal_history(
        date=today_signal.date(), mode=mode.value, phase="execution",
        today_signals=today_signals, today_zscores=today_zscores,
        today_prices=today_prices, entry_threshold=bt_config.entry_threshold,
    )

    result = {"date": str(today_signal.date()), "mode": mode.value, "phase": "execution"}

    if mode == TradingMode.SHADOW:
        assert shadow_sim is not None
        day_result = shadow_sim.process_shadow_day(
            date=today_signal,
            signal_df=signal_df,
            price_df=price_df,
            volume_df=volume_df,
            exit_signal_df=zscore_df,
            config=bt_config,
            verbose=True,
        )
        result.update(day_result)
        _save_shadow_state(shadow_sim)

    elif mode == TradingMode.LIVE:
        assert executor is not None

        # Get actual entry dates from Alpaca order history
        entry_dates = _get_entry_dates_from_orders(conn)

        # Get current live positions
        current_positions: Dict[str, Dict] = {}
        for pos in conn.get_positions():
            sym = pos["symbol"]
            current_positions[sym] = {
                "qty": int(pos["qty"]),
                "side": "long" if int(pos["qty"]) > 0 else "short",
                "entry_price": float(pos["avg_entry_price"]),
                "entry_date": entry_dates.get(sym, pd.Timestamp.now() - pd.Timedelta(days=1)),
            }

        # Fetch real-time prices for exit evaluation
        held_symbols = list(current_positions.keys())
        live_prices = conn.get_latest_trades(held_symbols) if held_symbols else {}

        # Overlay live prices onto the cached price DataFrame
        if live_prices:
            live_row = price_df.iloc[-1].copy()
            for sym, px in live_prices.items():
                if sym in live_row.index and px > 0:
                    live_row[sym] = px
            cache_date = price_df.index[-1].date()
            real_today = pd.Timestamp.now().normalize().date()
            if cache_date < real_today:
                new_idx = pd.Timestamp(real_today)
                price_df.loc[new_idx] = live_row
                signal_df.loc[new_idx] = signal_df.iloc[-1]  # carry forward signals
                zscore_df.loc[new_idx] = zscore_df.iloc[-1]
                today_signal = new_idx
                logger.info(
                    f"Overlaid {len(live_prices)} live prices onto cached data "
                    f"({cache_date} → {real_today})"
                )
            else:
                for sym, px in live_prices.items():
                    if sym in price_df.columns and px > 0:
                        price_df.iloc[-1][sym] = px

        decisions = executor.generate_decisions_from_signals(
            signal_df=signal_df,
            price_df=price_df,
            volume_df=volume_df,
            exit_signal_df=zscore_df,
            date=today_signal,
            current_positions=current_positions,
            config=bt_config,
        )

        if decisions:
            current_prices = price_df.iloc[-1]
            results = executor.execute_decisions(decisions, current_prices)
            filled = sum(1 for r in results if r.status in ("filled", "submitted"))
            logger.info(f"Executed {len(decisions)} decisions -> {filled} filled")
            decisions_data = []
            for r in results:
                status_ok = r.status in ("filled", "submitted")
                logger.info(
                    f"  {'OK' if status_ok else 'FAIL'} {r.decision.symbol} {r.decision.action} "
                    f"x{r.decision.target_qty} -> {r.status}"
                )
                decisions_data.append({
                    "symbol": r.decision.symbol,
                    "action": r.decision.action,
                    "qty": r.decision.target_qty,
                    "status": r.status,
                    "price": round(float(current_prices.get(r.decision.symbol, 0)), 2),
                })
            result["decisions"] = len(decisions)
            result["filled"] = filled
        else:
            logger.info("No trade signals today")
            result["decisions"] = 0
            decisions_data = []

        # Account summary
        account = conn.get_account()
        logger.info(
            f"Account: ${account['portfolio_value']:,.2f} "
            f"(cash: ${account['cash']:,.2f})"
        )

        _save_live_state(conn)
        result["portfolio_value"] = account["portfolio_value"]
        result["cash"] = account["cash"]

        # Persist trade history (git-tracked for model improvement)
        _append_trade_history(
            date=today_signal.date(), mode=mode.value,
            decisions_data=decisions_data,
            portfolio_value=float(account["portfolio_value"]),
            cash=float(account["cash"]),
        )

    elapsed = time.perf_counter() - phase_start
    result["execution_seconds"] = round(elapsed, 1)
    logger.info(f"Trade execution phase complete in {elapsed:.1f}s")
    return result


def _seed_equity_history(conn: AlpacaConnection) -> None:
    """Backfill equity_history.json from Alpaca portfolio history API."""
    equity_history_file = PROJECT_ROOT / "data" / "snapshots" / "equity_history.json"
    equity_history_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Load existing history
        history = []
        if equity_history_file.exists():
            with open(equity_history_file, "r") as f:
                history = json.load(f)

        existing_dates = {entry["timestamp"][:10] for entry in history}

        # Fetch daily history from Alpaca (last 3 months)
        alpaca_history = conn.get_portfolio_history(period="3M", timeframe="1D")

        if not alpaca_history:
            logger.info("No Alpaca portfolio history available for seeding")
            return

        # Add missing daily snapshots (end-of-day values)
        added = 0
        for point in alpaca_history:
            date_str = point["timestamp"][:10]
            if date_str not in existing_dates:
                history.append(point)
                existing_dates.add(date_str)
                added += 1

        if added > 0:
            # Sort by timestamp
            history.sort(key=lambda x: x["timestamp"])
            history = history[-2000:]

            with open(equity_history_file, "w") as f:
                json.dump(history, f)

            logger.info(
                f"Seeded equity history: {added} new daily points from Alpaca API "
                f"(total: {len(history)} points)"
            )
        else:
            logger.info(
                f"Equity history up to date ({len(history)} points)"
            )

    except Exception as e:
        logger.warning(f"Could not seed equity history: {e}")


def _post_trade_refresh(conn: AlpacaConnection, push: bool) -> None:
    """Dashboard refresh at 10:00 AM ET — captures post-trade state after 9:35 AM execution."""
    import pytz
    et = pytz.timezone("US/Eastern")
    now_et = datetime.now(et)

    target = now_et.replace(hour=10, minute=0, second=0, microsecond=0)

    if now_et >= target:
        logger.info("Already past 10:00 AM ET — skipping post-trade refresh")
        return

    wait_secs = (target - now_et).total_seconds()
    if wait_secs > 1800:  # >30 min — too early, skip
        logger.info(
            f"Post-trade refresh: {wait_secs:.0f}s until 10:00 AM ET — too far, skipping"
        )
        return

    logger.info(f"Post-trade refresh: waiting {wait_secs:.0f}s until 10:00 AM ET...")
    _interruptible_sleep(wait_secs)

    if SHUTDOWN_REQUESTED:
        return

    logger.info("Post-trade refresh: capturing post-execution state...")
    _save_live_state(conn)
    _generate_and_push_dashboard(push)
    logger.info("Post-trade refresh complete ✓")


def _post_close_refresh(conn: AlpacaConnection, push: bool) -> None:
    """Dashboard refresh at 4:05 PM ET — captures final closing data."""
    import pytz
    et = pytz.timezone("US/Eastern")
    now_et = datetime.now(et)

    target = now_et.replace(hour=16, minute=5, second=0, microsecond=0)

    if now_et >= target:
        logger.info("Already past 4:05 PM ET — skipping post-close refresh")
        return

    wait_secs = (target - now_et).total_seconds()

    logger.info(f"Post-close refresh: waiting {wait_secs / 3600:.1f}h until 4:05 PM ET...")
    _interruptible_sleep(wait_secs)

    if SHUTDOWN_REQUESTED:
        return

    logger.info("Post-close refresh: capturing final closing data...")
    _save_live_state(conn)
    _generate_and_push_dashboard(push)
    logger.info("Post-close refresh complete ✓")


def _save_live_state(conn: AlpacaConnection) -> None:
    """Save live trading state for dashboard."""
    state_file = PROJECT_ROOT / "data" / "snapshots" / "live_state.json"
    state_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Get account info
        account = conn.get_account()
        
        # Get current positions
        positions = []
        for pos in conn.get_positions():
            raw_qty = float(pos["qty"])
            positions.append({
                "symbol": pos["symbol"],
                "qty": abs(raw_qty),
                "side": "long" if raw_qty > 0 else "short",
                "entry_price": float(pos["avg_entry_price"]),
                "current_price": float(pos["current_price"]),
                "market_value": abs(float(pos["market_value"])),
                "unrealized_pl": float(pos["unrealized_pl"]),
                "unrealized_plpc": float(pos["unrealized_plpc"]) * 100,
            })
        
        # Get recent filled orders (last 50)
        trades = []
        try:
            closed_orders = conn.get_orders(status='closed', limit=50)
            for order in closed_orders:
                if order['status'] == 'filled' and order['filled_avg_price']:
                    trades.append({
                        "symbol": order["symbol"],
                        "side": order["side"],
                        "qty": float(order["qty"]) if order["qty"] else 0,
                        "filled_price": float(order["filled_avg_price"]),
                        "submitted_at": order["submitted_at"],
                        "order_id": order["id"],
                    })
        except Exception as e:
            logger.warning(f"Could not fetch order history: {e}")
        
        # Get open orders (stop-loss, take-profit, pending entries)
        open_orders = []
        try:
            for order in conn.get_orders(status='open', limit=50):
                open_orders.append({
                    "symbol": order["symbol"],
                    "side": order["side"],
                    "qty": float(order["qty"]) if order["qty"] else 0,
                    "type": order.get("type", "market"),
                    "stop_price": order.get("stop_price"),
                    "limit_price": order.get("limit_price"),
                    "order_class": order.get("order_class"),
                    "time_in_force": order.get("time_in_force"),
                    "status": order.get("status", "open"),
                    "submitted_at": order.get("submitted_at", "")[:19],
                    "order_id": order["id"],
                })
        except Exception as e:
            logger.warning(f"Could not fetch open orders: {e}")

        # Save snapshot
        equity_val = float(account["equity"])
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "mode": "live",
            "account": {
                "equity": equity_val,
                "cash": float(account["cash"]),
                "portfolio_value": float(account["portfolio_value"]),
                "buying_power": float(account["buying_power"]),
            },
            "positions": positions,
            "recent_trades": trades,
            "open_orders": open_orders,
        }
        
        with open(state_file, "w") as f:
            json.dump(snapshot, f, indent=2)
        
        # ── Accumulate equity history for dashboard chart ──
        equity_history_file = PROJECT_ROOT / "data" / "snapshots" / "equity_history.json"
        try:
            history = []
            if equity_history_file.exists():
                with open(equity_history_file, "r") as f:
                    history = json.load(f)
            
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
            # Avoid duplicate entries within the same minute
            if not history or history[-1].get("timestamp", "")[:16] != now_str:
                history.append({
                    "timestamp": now_str,
                    "equity": equity_val,
                })
                # Keep last 2000 data points (~30 days at 1 per minute trading hours)
                history = history[-2000:]
                
                with open(equity_history_file, "w") as f:
                    json.dump(history, f)
                    
        except Exception as e:
            logger.warning(f"Could not update equity history: {e}")
        
        logger.info(f"Live state saved ({len(positions)} positions, {len(trades)} trades)")

        # ── Save intraday equity for dashboard 1D chart ──
        try:
            intraday = conn.get_intraday_equity()
            if intraday:
                intraday_file = PROJECT_ROOT / "data" / "snapshots" / "intraday_equity.json"
                with open(intraday_file, "w") as f:
                    json.dump(intraday, f)
                logger.debug(f"Intraday equity saved ({len(intraday)} points)")
        except Exception as e:
            logger.warning(f"Could not save intraday equity: {e}")

    except Exception as e:
        logger.error(f"Failed to save live state: {e}")


def _restore_shadow_state(sim: SimulationEngine) -> None:
    """Restore shadow positions from CSV."""
    if not SHADOW_STATE_PATH.exists():
        return

    from execution.simulation import SimulatedPosition

    prev = pd.read_csv(SHADOW_STATE_PATH)
    if prev.empty:
        return

    for _, row in prev.iterrows():
        sim.positions[row["symbol"]] = SimulatedPosition(
            symbol=row["symbol"],
            qty=int(row["qty"]),
            side=row["side"],
            entry_price=float(row["entry_price"]),
            entry_date=pd.Timestamp(row["entry_date"]),
            signal_strength=float(row.get("signal_strength", 0)),
            current_price=float(row.get("current_price", row["entry_price"])),
        )
    sim.cash -= sum(
        p.entry_price * abs(p.qty)
        for p in sim.positions.values()
        if p.side == "long"
    )
    logger.info(f"Restored {len(sim.positions)} shadow positions")


# ═══════════════════════════════════════════════════════════════════════════
# SCHEDULING: MARKET-HOURS AWARENESS
# ═══════════════════════════════════════════════════════════════════════════

def is_market_day() -> bool:
    """Check if today is a US equity trading day (Mon-Fri, excluding holidays)."""
    from pandas.tseries.holiday import USFederalHolidayCalendar

    today = pd.Timestamp.now().normalize()
    if today.weekday() >= 5:
        return False

    holidays = USFederalHolidayCalendar().holidays(
        start=today - timedelta(days=1),
        end=today + timedelta(days=1),
    )
    return today not in holidays


def seconds_until_execution_window() -> float:
    """
    Seconds until 9:35 AM ET (market open + 5 min).
    
    T+1 Open execution: signals are generated from yesterday's close
    prices (fully settled overnight), and trades execute at market open.
    Running at 9:35 AM ensures the market is open and opening auction
    has settled for reliable fills.
    
    Returns 0 if in the execution window (9:35–10:00 AM ET).
    Returns positive if before the window.
    Returns negative if past the window.
    """
    import pytz
    et = pytz.timezone("US/Eastern")
    now_et = datetime.now(et)
    exec_time = now_et.replace(hour=9, minute=35, second=0, microsecond=0)
    window_end = now_et.replace(hour=10, minute=0, second=0, microsecond=0)

    if exec_time <= now_et <= window_end:
        return 0  # In execution window
    elif now_et < exec_time:
        return (exec_time - now_et).total_seconds()
    else:
        return -1  # Past window


def wait_for_execution_window(interval_sec: int) -> None:
    """
    Sleep until the execution window (9:35 AM ET — market open).
    
    T+1 Open execution: by running at market open, yesterday's daily
    bars are fully settled.  Signals computed from yesterday's close
    are executed at today's opening prices, eliminating T+0 look-ahead.
    
    - On market days: sleep until 9:35 AM ET
    - On weekends/holidays: sleep then re-check
    """
    while not SHUTDOWN_REQUESTED:
        if not is_market_day():
            logger.info("Non-trading day — sleeping until next check...")
            _interruptible_sleep(min(interval_sec, 3600))
            continue

        wait = seconds_until_execution_window()
        if wait == 0:
            return  # In execution window
        elif wait > 0:
            hours = wait / 3600
            if hours > 1:
                logger.info(f"Execution window in {hours:.1f}h (9:35 AM ET) — sleeping...")
            else:
                logger.info(f"Execution window in {wait/60:.0f}min — sleeping...")
            _interruptible_sleep(min(wait, 3600))  # Re-log every hour
        elif wait < 0:
            # Past window — sleep until tomorrow
            logger.info("Past execution window — sleeping until tomorrow...")
            _interruptible_sleep(min(interval_sec, 3600))
            continue


def _interruptible_sleep(seconds: float) -> None:
    """Sleep in 1-second chunks so shutdown signals are responsive."""
    end = time.time() + seconds
    while time.time() < end and not SHUTDOWN_REQUESTED:
        time.sleep(min(1.0, end - time.time()))


def _wait_for_post_close() -> None:
    """Wait until 4:10 PM ET for daily bars to settle before signal generation."""
    import pytz
    et = pytz.timezone("US/Eastern")
    now_et = datetime.now(et)
    target = now_et.replace(hour=16, minute=10, second=0, microsecond=0)

    if now_et >= target:
        return  # Already past 4:10 PM ET

    wait_secs = (target - now_et).total_seconds()
    logger.info(f"Waiting {wait_secs/60:.0f}min until 4:10 PM ET for daily bar settlement...")
    _interruptible_sleep(wait_secs)


def _generate_and_push_dashboard(auto_push: bool = False) -> None:
    """
    Generate static dashboard and optionally push to GitHub.
    
    Args:
        auto_push: If True, commits and pushes to GitHub
    """
    try:
        import subprocess
        
        # Generate dashboard on main branch (where live_state.json is available)
        generator = DashboardGenerator(PROJECT_ROOT)
        output_path = PROJECT_ROOT / "docs" / "index.html"
        
        if generator.generate(output_path):
            logger.info(f"Dashboard generated: {output_path}")
            
            if auto_push:
                # Read files into memory BEFORE switching branches
                # (working tree gets stashed during branch switch)
                dashboard_html = output_path.read_text()

                # Snapshot data files to bundle into dashboard-live
                data_snapshots = {}
                snapshot_dir = PROJECT_ROOT / "data" / "snapshots"
                for fname in [
                    "signal_history.json",
                    "trade_history.json",
                    "live_state.json",
                    "equity_history.json",
                    "intraday_equity.json",
                ]:
                    fpath = snapshot_dir / fname
                    if fpath.exists():
                        data_snapshots[fname] = fpath.read_text()
                
                # Push to dashboard-live branch (keeps main clean)
                def _git(*args, **kw):
                    return subprocess.run(
                        ["git", "-C", str(PROJECT_ROOT)] + list(args),
                        capture_output=True, timeout=kw.get("timeout", 15),
                    )

                def _ensure_main_branch():
                    """Force-switch back to main branch — always succeeds."""
                    try:
                        _git("checkout", "-f", "main")
                        # Try to pop stash; ignore if empty or conflicts
                        pop = _git("stash", "pop")
                        if pop.returncode != 0:
                            stderr = pop.stderr.decode().strip()
                            if "No stash entries" not in stderr:
                                logger.debug(f"Stash pop skipped: {stderr}")
                                _git("stash", "drop")
                    except Exception as e:
                        logger.warning(f"Branch recovery error: {e}")

                try:
                    # Stash any working changes, switch to dashboard-live
                    _git("stash", "--include-untracked")
                    
                    # Ensure dashboard-live branch exists locally
                    fetch_result = _git("fetch", "origin", "dashboard-live", timeout=30)
                    if fetch_result.returncode == 0:
                        _git("checkout", "dashboard-live")
                        _git("reset", "--hard", "origin/dashboard-live")
                    else:
                        # Branch doesn't exist yet — create orphan
                        _git("checkout", "--orphan", "dashboard-live")
                        _git("reset", "--hard")
                    
                    # Write dashboard HTML
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_text(dashboard_html)
                    _git("add", "docs/index.html")

                    # Write data snapshots (signal/trade history for dev access)
                    data_dir = PROJECT_ROOT / "data" / "snapshots"
                    data_dir.mkdir(parents=True, exist_ok=True)
                    for fname, content in data_snapshots.items():
                        (data_dir / fname).write_text(content)
                        _git("add", f"data/snapshots/{fname}")
                    
                    commit_msg = f"Update dashboard {datetime.now():%Y-%m-%d %H:%M}"
                    result = _git("commit", "-m", commit_msg)
                    
                    if result.returncode == 0:
                        push_result = _git("push", "origin", "dashboard-live", "--force", timeout=30)
                        if push_result.returncode == 0:
                            n_data = len(data_snapshots)
                            logger.info(f"Dashboard + {n_data} data files pushed to GitHub (dashboard-live branch)")
                        else:
                            logger.warning(f"Git push failed: {push_result.stderr.decode()}")
                    else:
                        logger.info("No dashboard changes to commit")
                    
                except subprocess.TimeoutExpired:
                    logger.warning("Git operation timed out")
                except Exception as e:
                    logger.warning(f"Dashboard push failed: {e}")
                finally:
                    # ALWAYS force-switch back to main (even on failure)
                    _ensure_main_branch()
                    # Restore data files — checkout from dashboard-live
                    # deletes them because they're tracked there but
                    # gitignored on main
                    for fname, content in data_snapshots.items():
                        try:
                            dest = snapshot_dir / fname
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            dest.write_text(content)
                        except Exception:
                            pass
        else:
            logger.warning("Dashboard generation failed")
            
    except Exception as e:
        logger.error(f"Dashboard error: {e}", exc_info=True)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Headless mean-reversion trader for autonomous 24/7 operation."
    )
    parser.add_argument(
        "--mode", choices=["shadow", "live"], default="shadow",
        help="Trading mode: shadow (no orders) or live (default: shadow)"
    )
    parser.add_argument(
        "--interval", type=int, default=0,
        help="Seconds between cycles. 0 = run once per market day (default: 0)"
    )
    parser.add_argument(
        "--log-level", default="INFO",
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Run a single cycle and exit (useful for cron)"
    )
    parser.add_argument(
        "--no-dashboard", action="store_true",
        help="Skip dashboard generation"
    )
    parser.add_argument(
        "--push-dashboard", action="store_true",
        help="Auto-commit and push dashboard to GitHub"
    )
    parser.add_argument(
        "--update-dashboard-only", action="store_true",
        help="Update dashboard with current Alpaca positions/prices and exit (no trading)"
    )
    parser.add_argument(
        "--generate-signals", action="store_true",
        help="Manually run signal generation phase and exit (for cache priming)"
    )
    return parser.parse_args()


def main():
    global SHUTDOWN_REQUESTED

    args = parse_args()

    # ── Dashboard-only mode: refresh state + generate + push, then exit ──
    if args.update_dashboard_only:
        log_dir = PROJECT_ROOT / "data" / "logs"
        setup_logging(log_dir, "INFO")
        alpaca_config = AlpacaConfig.from_env()
        alpaca_config.trading_mode = TradingMode.LIVE
        conn = AlpacaConnection(alpaca_config)
        conn.test_connection()
        _save_live_state(conn)
        _generate_and_push_dashboard(auto_push=True)
        logger.info("Dashboard updated and pushed (dashboard-only mode)")
        return

    # ── Manual signal generation: generate signals now and exit ──
    if args.generate_signals:
        mode = TradingMode.LIVE if args.mode == "live" else TradingMode.SHADOW
        log_dir = PROJECT_ROOT / "data" / "logs"
        setup_logging(log_dir, args.log_level)
        logger.info("=" * 60)
        logger.info(f"  MANUAL SIGNAL GENERATION — {mode.value.upper()} MODE")
        logger.info("=" * 60)

        config = ConfigLoader()
        bt_config = config.to_backtest_config()

        alpaca_config = AlpacaConfig.from_env()
        alpaca_config.trading_mode = mode
        conn = AlpacaConnection(alpaca_config)
        conn.test_connection()

        adapter = AlpacaDataAdapter(
            data_client=conn.data_client,
            data_feed=alpaca_config.data_feed,
            cache_dir=PROJECT_ROOT / "data" / "snapshots" / "alpaca_cache",
        )

        max_symbols = config.get('alpaca.max_universe_size', 60)
        universe = select_universe(config, max_symbols=max_symbols, project_root=PROJECT_ROOT)
        logger.info(f"Universe: {len(universe)} symbols")

        result = run_signal_generation_phase(
            adapter=adapter, config=config,
            bt_config=bt_config, universe=universe, mode=mode,
        )

        if result:
            logger.info(
                f"Signal generation complete — cached {len(result['universe'])} symbols "
                f"for {mode.value} mode T+1 execution"
            )
        else:
            logger.error("Signal generation FAILED")
            sys.exit(1)
        return

    mode = TradingMode.LIVE if args.mode == "live" else TradingMode.SHADOW

    # ── Logging ──
    log_dir = PROJECT_ROOT / "data" / "logs"
    setup_logging(log_dir, args.log_level)

    # ── Signal handlers ──
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    logger.info("=" * 60)
    logger.info(f"  MEAN REVERSION TRADER — {mode.value.upper()} MODE")
    logger.info(f"  Interval: {'once' if args.once else f'{args.interval}s' if args.interval else 'daily'}")
    logger.info(f"  PID: {os.getpid()}")
    logger.info("=" * 60)

    # ── Load config ──
    config = ConfigLoader()
    bt_config = config.to_backtest_config()
    signal_config = config.to_signal_config()

    logger.info(f"Config: entry={bt_config.entry_threshold}, "
                f"exit={bt_config.exit_threshold}, "
                f"stop={bt_config.stop_loss_pct}, "
                f"sizing={getattr(bt_config, 'position_size_method', 'equal_weight')}")

    # ── Connect to Alpaca ──
    alpaca_config = AlpacaConfig.from_env()
    alpaca_config.trading_mode = mode

    conn = AlpacaConnection(alpaca_config)
    conn.test_connection()

    # ── Seed equity history from Alpaca portfolio history ──
    _seed_equity_history(conn)

    # ── Retrofit stop-loss orders for unprotected positions (Fix 1) ──
    if mode == TradingMode.LIVE:
        _retrofit_bracket_orders(conn, bt_config)

    # ── Data adapter ──
    adapter = AlpacaDataAdapter(
        data_client=conn.data_client,
        data_feed=alpaca_config.data_feed,
        cache_dir=PROJECT_ROOT / "data" / "snapshots" / "alpaca_cache",
    )

    # ── Universe ──
    max_symbols = config.get('alpaca.max_universe_size', 60)
    universe = select_universe(config, max_symbols=max_symbols, project_root=PROJECT_ROOT)
    logger.info(f"Universe: {len(universe)} symbols → {', '.join(universe[:10])}...")

    # ── Build executor / simulation engine ──
    executor = build_executor(conn, bt_config, mode)

    shadow_sim = None
    if mode == TradingMode.SHADOW:
        shadow_sim = build_simulation(executor, bt_config, mode)
        _restore_shadow_state(shadow_sim)

    # ── Intraday monitor ──
    monitor = None
    intraday_cfg_dict = config.get('backtest.intraday_monitor', {})
    intraday_enabled = bool(intraday_cfg_dict.get('enabled', False)) if intraday_cfg_dict else False
    if intraday_enabled:
        monitor_config = IntradayMonitorConfig.from_dict(intraday_cfg_dict)
        monitor = IntradayMonitor(
            connection=conn,
            config=monitor_config,
            shadow_sim=shadow_sim,
            shutdown_flag=lambda: SHUTDOWN_REQUESTED,
        )
        logger.info(
            f"Intraday monitor enabled | poll={monitor_config.poll_interval_sec}s "
            f"| window={monitor_config.start_time_et}-{monitor_config.stop_time_et} ET"
        )

    # ── Main loop ──
    # Two-phase T+1 architecture:
    #   Phase 1 (post-close ~4:10 PM): Generate signals from Day T close → cache
    #   Phase 2 (9:35 AM T+1): Load cached signals → execute trades
    #   Between: Intraday monitor watches held positions (09:45–15:50)
    #
    # First run (no cache): skips execution, generates signals post-close.
    # Subsequent days: cached signals drive morning execution.
    cycle_count = 0
    last_trade_date = None
    last_signal_date = None

    while not SHUTDOWN_REQUESTED:
        try:
            # ── Non-trading day → sleep ──
            if not is_market_day():
                logger.info("Non-trading day — sleeping until next check...")
                _interruptible_sleep(3600)
                continue

            today = pd.Timestamp.now().normalize()

            # ═══════════════════════════════════════════════════════
            # --once and --interval modes: legacy full cycle
            # (signal gen + execution in one shot — for cron / testing)
            # ═══════════════════════════════════════════════════════
            if args.once or args.interval > 0:
                cycle_count += 1
                logger.info(f"\n{'─'*50}")
                logger.info(f"CYCLE {cycle_count} — {datetime.now():%Y-%m-%d %H:%M:%S}")
                logger.info(f"{'─'*50}")

                result = run_daily_cycle(
                    conn=conn, adapter=adapter, config=config,
                    bt_config=bt_config, universe=universe, mode=mode,
                    shadow_sim=shadow_sim, executor=executor,
                )
                last_trade_date = today
                logger.info(f"Cycle {cycle_count} result: {result}")

                if not args.no_dashboard:
                    _generate_and_push_dashboard(args.push_dashboard)

                if args.once:
                    logger.info("--once flag: exiting after single cycle")
                    break
                _interruptible_sleep(args.interval)
                continue

            # ═══════════════════════════════════════════════════════
            # PHASE 2: Trade Execution (9:35 AM ET)
            # Load cached signals from last night's post-close gen
            # ═══════════════════════════════════════════════════════
            exec_wait = seconds_until_execution_window()

            if exec_wait > 0:
                # Before execution window → sleep until it opens
                hours = exec_wait / 3600
                if hours > 1:
                    logger.info(f"Execution window in {hours:.1f}h (9:35 AM ET) — sleeping...")
                else:
                    logger.info(f"Execution window in {exec_wait/60:.0f}min — sleeping...")
                _interruptible_sleep(min(exec_wait, 3600))
                continue

            if exec_wait == 0 and last_trade_date != today:
                # In execution window → execute trades from cached signals
                signal_cache = _load_signal_cache(mode)

                if signal_cache is not None:
                    cycle_count += 1
                    logger.info(f"\n{'─'*50}")
                    logger.info(f"CYCLE {cycle_count} — {datetime.now():%Y-%m-%d %H:%M:%S}")
                    logger.info(f"{'─'*50}")

                    result = run_trade_execution_phase(
                        conn=conn, signal_cache=signal_cache,
                        bt_config=bt_config, mode=mode,
                        executor=executor, shadow_sim=shadow_sim,
                    )
                    last_trade_date = today
                    logger.info(f"Cycle {cycle_count} result: {result}")

                    # Immediate post-trade dashboard
                    if not args.no_dashboard:
                        _generate_and_push_dashboard(args.push_dashboard)
                        _post_trade_refresh(conn, args.push_dashboard)
                else:
                    logger.warning(
                        "No cached signals available — skipping trade execution. "
                        "Signals will be generated after market close today."
                    )
                    last_trade_date = today  # Don't re-check every hour

            elif exec_wait < 0 and last_trade_date != today:
                # Past execution window (e.g. service restarted mid-day)
                logger.info(
                    "Past execution window — skipping trade execution for today. "
                    "Will generate signals after market close."
                )
                last_trade_date = today

            # ═══════════════════════════════════════════════════════
            # INTRADAY MONITOR (09:45–15:50 ET)
            # Monitors held positions for dynamic exits
            # ═══════════════════════════════════════════════════════
            if monitor is not None:
                monitor.reset_session()
                monitor.run()  # Blocks until stop_time_et or shutdown
                if SHUTDOWN_REQUESTED:
                    break

            # ═══════════════════════════════════════════════════════
            # PHASE 1: Signal Generation (Post-Close ~4:10 PM ET)
            # Generate tomorrow's signals from today's close data
            # ═══════════════════════════════════════════════════════
            _wait_for_post_close()
            if SHUTDOWN_REQUESTED:
                break

            if last_signal_date != today:
                sig_result = run_signal_generation_phase(
                    adapter=adapter, config=config,
                    bt_config=bt_config, universe=universe, mode=mode,
                )
                if sig_result is not None:
                    last_signal_date = today
                    universe = sig_result["universe"]

            # ── Post-close dashboard refresh ──
            if not args.no_dashboard:
                _save_live_state(conn)
                _generate_and_push_dashboard(args.push_dashboard)

            # ── Sleep until next trading day ──
            logger.info("Daily cycle complete — sleeping until next trading day...")
            _interruptible_sleep(3600)

        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Cycle error: {e}", exc_info=True)
            _interruptible_sleep(300)

    # ── Shutdown ──
    logger.info("=" * 60)
    logger.info(f"  SHUTDOWN — {cycle_count} cycles completed")
    if shadow_sim:
        _save_shadow_state(shadow_sim)
        logger.info(f"  Final equity: ${shadow_sim.equity:,.2f}")
        logger.info(f"  Open positions: {len(shadow_sim.positions)}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
