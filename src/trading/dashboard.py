"""
Position monitoring dashboard.

Displays current positions, P&L, and signal status for all trading modes.
Extracted from notebook Cell 9 with explicit parameters (no globals()).
"""

import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from connection.alpaca_connection import AlpacaConnection, TradingMode
from backtest.engine import BacktestConfig
from execution.simulation import SimulationEngine

logger = logging.getLogger(__name__)


def show_dashboard(
    sim_engine: Optional[SimulationEngine] = None,
    connection: Optional[AlpacaConnection] = None,
    mode: Optional[TradingMode] = None,
    signal_df: Optional[pd.DataFrame] = None,
    entry_threshold: Optional[float] = None,
) -> None:
    """
    Display position monitoring dashboard.

    Args:
        sim_engine: SimulationEngine for shadow/replay positions.
        connection: AlpacaConnection for live positions.
        mode: Current trading mode.
        signal_df: Signal DataFrame for showing active entry signals.
        entry_threshold: Entry signal threshold (e.g. 1.43).
    """
    print(f"\n{'═' * 70}")
    print(f"  POSITION DASHBOARD — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'═' * 70}")

    if mode in (TradingMode.SHADOW, TradingMode.REPLAY) and sim_engine:
        # Shadow/Replay positions
        print(f"\n  Mode: {mode.value.upper()} | Equity: ${sim_engine.equity:,.2f} "
              f"| Cash: ${sim_engine.cash:,.2f}")
        print(f"  Open positions: {len(sim_engine.positions)}")
        print(f"  Completed trades: {len(sim_engine.completed_trades)}")

        if sim_engine.positions:
            print(f"\n  {'Symbol':8s} {'Side':6s} {'Qty':>6s} {'Entry':>10s} "
                  f"{'Current':>10s} {'P&L':>10s} {'P&L%':>8s} {'Days':>5s}")
            print(f"  {'─' * 65}")

            total_pnl = 0
            for sym, pos in sorted(sim_engine.positions.items()):
                days = (pd.Timestamp.now() - pos.entry_date).days
                pnl = pos.unrealized_pnl
                pnl_pct = pos.unrealized_pnl_pct * 100
                total_pnl += pnl
                icon = "🟢" if pnl >= 0 else "🔴"
                print(f"  {icon} {sym:6s} {pos.side:6s} {abs(pos.qty):6d} "
                      f"${pos.entry_price:9.2f} ${pos.current_price:9.2f} "
                      f"${pnl:9.2f} {pnl_pct:+7.2f}% {days:5d}")
            print(f"  {'─' * 65}")
            print(f"  {'Total':55s} ${total_pnl:9.2f}")

        # Trade history summary
        if sim_engine.completed_trades:
            wins = sum(1 for t in sim_engine.completed_trades if t.pnl > 0)
            losses = len(sim_engine.completed_trades) - wins
            avg_pnl = np.mean([t.pnl_pct * 100 for t in sim_engine.completed_trades])
            print(f"\n  Trade History: {wins}W / {losses}L | "
                  f"Win rate: {wins / (wins + losses) * 100:.1f}% | "
                  f"Avg P&L: {avg_pnl:+.2f}%")

    elif mode == TradingMode.LIVE and connection:
        # Live positions from Alpaca
        account = connection.get_account()
        positions = connection.get_positions()

        print(f"\n  Equity: ${account['portfolio_value']:,.2f} | "
              f"Cash: ${account['cash']:,.2f} | "
              f"Day trades: {account['daytrade_count']}/3")

        if positions:
            print(f"\n  {'Symbol':8s} {'Side':6s} {'Qty':>6s} {'Entry':>10s} "
                  f"{'Current':>10s} {'P&L':>10s} {'P&L%':>8s}")
            print(f"  {'─' * 60}")

            for pos in positions:
                qty = int(pos["qty"])
                side = "long" if qty > 0 else "short"
                entry = float(pos["avg_entry_price"])
                current = float(pos["current_price"])
                pnl = float(pos["unrealized_pl"])
                pnl_pct = float(pos["unrealized_plpc"]) * 100
                icon = "🟢" if pnl >= 0 else "🔴"
                print(f"  {icon} {pos['symbol']:6s} {side:6s} {abs(qty):6d} "
                      f"${entry:9.2f} ${current:9.2f} ${pnl:9.2f} {pnl_pct:+7.2f}%")
        else:
            print("\n  No open positions")

    # Show today's signals that hit thresholds
    if signal_df is not None and entry_threshold is not None:
        latest = signal_df.iloc[-1].dropna()
        entries = latest[latest.abs() > entry_threshold].sort_values()
        if len(entries) > 0:
            print(f"\n  📡 Active entry signals ({len(entries)}):")
            for sym, val in entries.items():
                direction = "SHORT" if val > 0 else "LONG "
                print(f"     {direction} {sym:6s}  signal={val:+.3f}")
