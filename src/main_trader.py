#!/usr/bin/env python3
"""
Headless Mean-Reversion Trader — 24/7 Autonomous Operation

Standalone script for unattended deployment (Oracle Cloud, VPS, etc.).
Replaces the Jupyter notebook with a continuous trading loop.

Architecture:
  startup → data load → signal loop → graceful shutdown

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
from dashboard_generator import DashboardGenerator


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
# UNIVERSE SELECTION
# ═══════════════════════════════════════════════════════════════════════════

def select_universe(config: ConfigLoader, max_symbols: int = 30) -> list:
    """
    Pick the top mean-reverting symbols from cached Hurst data,
    falling back to DOW 30 if no cache exists.
    """
    from data.universe_builder import SP500_CORE, NASDAQ_100_CORE, DOW_30, RUSSELL_2000_CORE

    hurst_cache = PROJECT_ROOT / "data" / "snapshots" / "hurst_rankings.csv"
    daily_dir = PROJECT_ROOT / "data" / "historical" / "daily"

    if hurst_cache.exists():
        hurst_df = pd.read_csv(hurst_cache)
        universe = hurst_df.nsmallest(max_symbols, "hurst_exponent")["symbol"].tolist()
        logger.info(f"Loaded Hurst cache → {len(universe)} symbols")
        return universe

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

    # Fallback
    universe = DOW_30[:max_symbols]
    logger.warning(f"No Hurst data — falling back to DOW {len(universe)}")
    return universe


# ═══════════════════════════════════════════════════════════════════════════
# DATA & SIGNALS
# ═══════════════════════════════════════════════════════════════════════════

def fetch_data(
    adapter: AlpacaDataAdapter,
    universe: list,
    lookback_days: int = 365,
):
    """Fetch price+volume data with caching. Returns (price_df, volume_df, raw_bars)."""
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


def generate_signals(
    config: ConfigLoader,
    price_df: pd.DataFrame,
    volume_df: pd.DataFrame,
):
    """Run signal pipeline. Returns (signal_df, zscore_df)."""
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
# DAILY EXECUTION LOGIC
# ═══════════════════════════════════════════════════════════════════════════

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

    # ── 1. Data ──
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

        # Get current live positions
        current_positions: Dict[str, Dict] = {}
        for pos in conn.get_positions():
            current_positions[pos["symbol"]] = {
                "qty": int(pos["qty"]),
                "side": "long" if int(pos["qty"]) > 0 else "short",
                "entry_price": float(pos["avg_entry_price"]),
                "entry_date": pd.Timestamp.now() - pd.Timedelta(days=1),
            }

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
            positions.append({
                "symbol": pos["symbol"],
                "qty": int(pos["qty"]),
                "side": "long" if int(pos["qty"]) > 0 else "short",
                "entry_price": float(pos["avg_entry_price"]),
                "current_price": float(pos["current_price"]),
                "market_value": float(pos["market_value"]),
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
    Seconds until 3:55 PM ET (5 min before close).
    
    We execute near market close to match the backtest engine, which
    uses daily close prices for signal generation and entry/exit decisions.
    Running at close ensures today's full price bar is formed.
    
    Returns 0 if in the execution window (3:55–4:00 PM ET).
    Returns positive if before the window.
    Returns negative if past market close.
    """
    import pytz
    et = pytz.timezone("US/Eastern")
    now_et = datetime.now(et)
    exec_time = now_et.replace(hour=15, minute=55, second=0, microsecond=0)
    close_time = now_et.replace(hour=16, minute=0, second=0, microsecond=0)

    if exec_time <= now_et <= close_time:
        return 0  # In execution window
    elif now_et < exec_time:
        return (exec_time - now_et).total_seconds()
    else:
        return -1  # Past close


def wait_for_execution_window(interval_sec: int) -> None:
    """
    Sleep until the execution window (3:55 PM ET).
    
    This matches the backtest engine which uses daily close prices.
    By executing near close, the IEX daily bar is fully formed and
    signals reflect the same data the backtest used.
    
    - On market days: sleep until 3:55 PM ET
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
                logger.info(f"Execution window in {hours:.1f}h (3:55 PM ET) — sleeping...")
            else:
                logger.info(f"Execution window in {wait/60:.0f}min — sleeping...")
            _interruptible_sleep(min(wait, 3600))  # Re-log every hour
        elif wait < 0:
            # Past close — sleep until tomorrow
            logger.info("Past market close — sleeping until tomorrow...")
            _interruptible_sleep(min(interval_sec, 3600))
            continue


def _interruptible_sleep(seconds: float) -> None:
    """Sleep in 1-second chunks so shutdown signals are responsive."""
    end = time.time() + seconds
    while time.time() < end and not SHUTDOWN_REQUESTED:
        time.sleep(min(1.0, end - time.time()))


def _generate_and_push_dashboard(auto_push: bool = False) -> None:
    """
    Generate static dashboard and optionally push to GitHub.
    
    Args:
        auto_push: If True, commits and pushes to GitHub
    """
    try:
        import subprocess
        
        # Generate dashboard
        generator = DashboardGenerator(PROJECT_ROOT)
        output_path = PROJECT_ROOT / "docs" / "index.html"
        
        if generator.generate(output_path):
            logger.info(f"✅ Dashboard generated: {output_path}")
            
            if auto_push:
                # Push to dashboard-live branch (keeps main clean)
                try:
                    git = lambda *args, **kw: subprocess.run(
                        ["git", "-C", str(PROJECT_ROOT)] + list(args),
                        capture_output=True, timeout=kw.get("timeout", 15),
                    )
                    
                    # Stash any working changes, switch to dashboard-live
                    git("stash", "--include-untracked")
                    
                    # Ensure dashboard-live branch exists locally
                    fetch_result = git("fetch", "origin", "dashboard-live", timeout=30)
                    if fetch_result.returncode == 0:
                        git("checkout", "dashboard-live")
                        git("reset", "--hard", "origin/dashboard-live")
                    else:
                        # Branch doesn't exist yet — create orphan
                        git("checkout", "--orphan", "dashboard-live")
                        git("reset", "--hard")
                    
                    # Copy the generated dashboard into the branch
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    generator.generate(output_path)
                    
                    git("add", "docs/index.html")
                    commit_msg = f"Update dashboard {datetime.now():%Y-%m-%d %H:%M}"
                    result = git("commit", "-m", commit_msg)
                    
                    if result.returncode == 0:
                        push_result = git("push", "origin", "dashboard-live", "--force", timeout=30)
                        if push_result.returncode == 0:
                            logger.info("✅ Dashboard pushed to GitHub (dashboard-live branch)")
                        else:
                            logger.warning(f"⚠️  Git push failed: {push_result.stderr.decode()}")
                    else:
                        logger.info("ℹ️  No dashboard changes to commit")
                    
                    # Switch back to main
                    git("checkout", "main")
                    git("stash", "pop")
                        
                except subprocess.TimeoutExpired:
                    logger.warning("⚠️  Git operation timed out")
                    git("checkout", "main")
                    git("stash", "pop")
                except Exception as e:
                    logger.warning(f"⚠️  Git push failed: {e}")
                    try:
                        git("checkout", "main")
                        git("stash", "pop")
                    except Exception:
                        pass
        else:
            logger.warning("⚠️  Dashboard generation failed")
            
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

    # ── Data adapter ──
    adapter = AlpacaDataAdapter(
        data_client=conn.data_client,
        data_feed=alpaca_config.data_feed,
        cache_dir=PROJECT_ROOT / "data" / "snapshots" / "alpaca_cache",
    )

    # ── Universe ──
    universe = select_universe(config, max_symbols=30)
    logger.info(f"Universe: {len(universe)} symbols → {', '.join(universe[:10])}...")

    # ── Build executor / simulation engine ──
    executor = AlpacaExecutor(
        connection=conn,
        commission_pct=0.0 if mode == TradingMode.LIVE else bt_config.commission_pct,
        max_position_pct=bt_config.max_position_size,
        max_total_exposure=bt_config.max_total_exposure,
    )

    shadow_sim = None
    if mode == TradingMode.SHADOW:
        shadow_sim = SimulationEngine(
            executor=executor,
            initial_capital=bt_config.initial_capital,
            commission_pct=bt_config.commission_pct,
            slippage_pct=bt_config.slippage_pct,
        )
        _restore_shadow_state(shadow_sim)

    # ── Main loop ──
    cycle_count = 0
    last_trade_date = None

    while not SHUTDOWN_REQUESTED:
        try:
            # Wait for execution window (unless running on interval/once)
            if args.interval == 0 and not args.once:
                wait_for_execution_window(interval_sec=3600)
                if SHUTDOWN_REQUESTED:
                    break

            # Skip if already traded today
            today = pd.Timestamp.now().normalize()
            if last_trade_date == today and not args.once:
                logger.info(f"Already traded today ({today.date()}) — sleeping until tomorrow")
                _interruptible_sleep(3600)
                continue

            # ── Execute cycle ──
            cycle_count += 1
            logger.info(f"\n{'─'*50}")
            logger.info(f"CYCLE {cycle_count} — {datetime.now():%Y-%m-%d %H:%M:%S}")
            logger.info(f"{'─'*50}")

            result = run_daily_cycle(
                conn=conn,
                adapter=adapter,
                config=config,
                bt_config=bt_config,
                universe=universe,
                mode=mode,
                shadow_sim=shadow_sim,
                executor=executor,
            )

            last_trade_date = today
            logger.info(f"Cycle {cycle_count} result: {result}")

            # ── Generate Dashboard ──
            if not args.no_dashboard:
                _generate_and_push_dashboard(args.push_dashboard)

            # ── Exit or sleep ──
            if args.once:
                logger.info("--once flag: exiting after single cycle")
                break

            if args.interval > 0:
                logger.info(f"Sleeping {args.interval}s until next cycle...")
                _interruptible_sleep(args.interval)
            else:
                # Daily mode: sleep until tomorrow
                logger.info("Daily mode — sleeping until next trading day...")
                _interruptible_sleep(3600)  # Re-check hourly

        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Cycle error: {e}", exc_info=True)
            # Back off on errors, then retry
            _interruptible_sleep(min(300, args.interval or 300))

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
