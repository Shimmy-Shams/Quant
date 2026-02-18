"""
Trading analysis — charts, reports, and comparison utilities.

Extracted from the interactive notebook for reuse.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from backtest.engine import BacktestConfig, BacktestEngine
from execution.simulation import SimulationEngine

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# SHADOW REPLAY CHART  (notebook Cell 5c)
# ═══════════════════════════════════════════════════════════════════════════

def plot_shadow_replay(
    results: dict,
    bt_config: BacktestConfig,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    n_symbols: int,
    save_path: Optional[Path] = None,
) -> None:
    """
    3-panel chart: equity curve, daily returns, cumulative trade P&L.

    Args:
        results: Shadow replay results dict from SimulationEngine.run_replay().
        bt_config: Backtest configuration (for initial capital, etc.).
        start_date: Replay window start.
        end_date: Replay window end.
        n_symbols: Number of symbols in the universe.
        save_path: If set, save the chart to this path.
    """
    equity = results.get("equity_curve")
    trades_df = results.get("trades_df")
    daily_returns = results.get("returns")
    trading_days = len(equity) if equity is not None else 0
    years = trading_days / 252

    fig, axes = plt.subplots(
        3, 1, figsize=(14, 12), height_ratios=[3, 1, 1],
        gridspec_kw={"hspace": 0.30},
    )

    # ── Panel 1: Equity curve ──
    ax1 = axes[0]
    if equity is not None and len(equity) > 0:
        ax1.plot(equity.index, equity.values, linewidth=1.4, color="#2196F3",
                 label="Shadow Equity")
        ax1.axhline(bt_config.initial_capital, ls="--", lw=0.8, color="gray",
                    alpha=0.6, label="Starting Capital")

        peak = equity.cummax()
        dd = (equity - peak) / peak
        ax1.fill_between(equity.index, equity.values, peak.values,
                         where=(equity < peak), alpha=0.15, color="red",
                         label="Drawdown")

        ax1.set_ylabel("Portfolio Value ($)")
        ax1.set_title(
            f"Shadow Replay — {start_date.date()} to {end_date.date()} "
            f"({trading_days}d)\n"
            f"Return: {results.get('total_return_pct', 0):.1f}%  |  "
            f"Sharpe: {results.get('sharpe_ratio', 0):.2f}  |  "
            f"Max DD: {results.get('max_drawdown_pct', 0):.1f}%  |  "
            f"Trades: {results.get('total_trades', 0)}  |  "
            f"Win Rate: {results.get('win_rate', 0):.0f}%",
            fontsize=11, fontweight="bold",
        )
        ax1.legend(loc="upper left", fontsize=9)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    else:
        ax1.text(0.5, 0.5, "No equity data", transform=ax1.transAxes,
                 ha="center", fontsize=14)

    # ── Panel 2: Daily returns ──
    ax2 = axes[1]
    if daily_returns is not None and len(daily_returns) > 0:
        colors = ["#4CAF50" if r >= 0 else "#F44336" for r in daily_returns]
        ax2.bar(daily_returns.index, daily_returns.values * 100, width=1,
                color=colors, alpha=0.7, linewidth=0)
        ax2.axhline(0, color="black", lw=0.5)
        ax2.set_ylabel("Daily Return (%)")
        ax2.set_title("Daily Returns", fontsize=10)
    else:
        ax2.text(0.5, 0.5, "No return data", transform=ax2.transAxes,
                 ha="center", fontsize=11)

    # ── Panel 3: Cumulative trade P&L ──
    ax3 = axes[2]
    if trades_df is not None and len(trades_df) > 0:
        cum_pnl = trades_df["pnl_pct"].cumsum()
        trade_nums = range(1, len(cum_pnl) + 1)
        ax3.plot(trade_nums, cum_pnl.values, marker="o", markersize=3,
                 linewidth=1.2, color="#FF9800")
        ax3.axhline(0, color="black", lw=0.5)
        ax3.set_xlabel("Trade #")
        ax3.set_ylabel("Cumulative P&L (%)")
        ax3.set_title(f"Cumulative Trade P&L ({len(trades_df)} trades)", fontsize=10)
        ax3.fill_between(trade_nums, cum_pnl.values, 0,
                         where=(cum_pnl.values >= 0), alpha=0.15, color="green")
        ax3.fill_between(trade_nums, cum_pnl.values, 0,
                         where=(cum_pnl.values < 0), alpha=0.15, color="red")
    else:
        ax3.text(0.5, 0.5, "No completed trades", transform=ax3.transAxes,
                 ha="center", fontsize=11)

    for ax in axes:
        ax.grid(True, alpha=0.3)
        if ax is not axes[2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
            ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"📊 Chart saved → {save_path}")
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════
# MONTHLY RETURNS TABLE  (notebook Cell 5c)
# ═══════════════════════════════════════════════════════════════════════════

def print_monthly_returns(daily_returns: pd.Series) -> None:
    """Print monthly return breakdown from daily returns series."""
    if daily_returns is None or len(daily_returns) == 0:
        return

    monthly = daily_returns.resample("ME").apply(
        lambda x: (1 + x).prod() - 1 if len(x) > 0 else 0
    ) * 100

    print(f"\n📅 Monthly Returns:")
    print(f"{'─' * 40}")
    for dt, ret in monthly.items():
        bar = "█" * int(abs(ret) * 2) if abs(ret) > 0.1 else "·"
        sign = "+" if ret >= 0 else ""
        color_icon = "🟢" if ret >= 0 else "🔴"
        print(f"   {dt.strftime('%Y-%m')}  {color_icon} {sign}{ret:6.2f}%  {bar}")
    print(f"{'─' * 40}")
    print(f"   Total       {monthly.sum():+.2f}%")


# ═══════════════════════════════════════════════════════════════════════════
# TRADE SUMMARY TABLE  (notebook Cell 5c)
# ═══════════════════════════════════════════════════════════════════════════

def print_trade_summary(trades_df: pd.DataFrame) -> None:
    """Print top/bottom trades from a trades DataFrame."""
    if trades_df is None or len(trades_df) == 0:
        return

    print(f"\n📋 Top/Bottom Trades:")
    print(f"{'─' * 65}")
    cols = ["symbol", "side", "entry_date", "exit_date", "pnl_pct"]
    avail_cols = [c for c in cols if c in trades_df.columns]
    if avail_cols:
        sorted_trades = trades_df.sort_values("pnl_pct", ascending=False)
        top5 = sorted_trades.head(5)
        bot5 = sorted_trades.tail(5)
        print("   Best:")
        for _, t in top5.iterrows():
            print(f"     {t.get('symbol', '?'):6s} {t.get('side', '?'):5s}  "
                  f"{t.get('pnl_pct', 0):+6.2f}%  "
                  f"{str(t.get('entry_date', ''))[:10]} → {str(t.get('exit_date', ''))[:10]}")
        print("   Worst:")
        for _, t in bot5.iterrows():
            print(f"     {t.get('symbol', '?'):6s} {t.get('side', '?'):5s}  "
                  f"{t.get('pnl_pct', 0):+6.2f}%  "
                  f"{str(t.get('entry_date', ''))[:10]} → {str(t.get('exit_date', ''))[:10]}")


# ═══════════════════════════════════════════════════════════════════════════
# REPLAY vs BACKTEST COMPARISON  (notebook Cell 7)
# ═══════════════════════════════════════════════════════════════════════════

def print_metrics_comparison(
    replay_results: dict,
    bt_results,
    replay_start: pd.Timestamp,
    replay_end: pd.Timestamp,
) -> None:
    """
    Print side-by-side comparison table: replay (live pipe) vs backtest engine.

    Args:
        replay_results: Results dict from SimulationEngine.run_replay().
        bt_results: BacktestResults from BacktestEngine.run_backtest().
        replay_start: Replay period start date.
        replay_end: Replay period end date.
    """
    r = replay_results
    bt = bt_results

    print(f"{'═' * 72}")
    print(f"  {'METRIC':<30s} {'REPLAY (Live Pipe)':>18s}  {'BACKTEST':>18s}")
    print(f"{'═' * 72}")

    rows = [
        ("Period",
         f"{replay_start.date()} → {replay_end.date()}",
         f"{replay_start.date()} → {replay_end.date()}"),
        ("Trading Days",
         f"{len(r['equity_curve'])}",
         f"{len(bt.equity_curve)}"),
        ("─── Returns ───", "", ""),
        ("Total Return",
         f"{r['total_return_pct']:+.2f}%",
         f"{bt.total_return * 100:+.2f}%"),
        ("Annualized Return",
         f"{(((1 + r['total_return_pct'] / 100) ** (252 / max(len(r['equity_curve']), 1))) - 1) * 100:+.2f}%",
         f"{bt.annualized_return * 100:+.2f}%"),
        ("Final Equity",
         f"${r['final_equity']:,.2f}",
         f"${bt.equity_curve.iloc[-1]:,.2f}"),
        ("─── Risk ───", "", ""),
        ("Sharpe Ratio",
         f"{r['sharpe_ratio']:.3f}",
         f"{bt.sharpe_ratio:.3f}"),
        ("Sortino Ratio", "n/a", f"{bt.sortino_ratio:.3f}"),
        ("Calmar Ratio", "n/a", f"{bt.calmar_ratio:.3f}"),
        ("Max Drawdown",
         f"{r['max_drawdown_pct']:.2f}%",
         f"{bt.max_drawdown * 100:.2f}%"),
        ("Max DD Duration", "—", f"{bt.max_drawdown_duration} days"),
        ("─── Trades ───", "", ""),
        ("Total Trades",
         f"{r['total_trades']}",
         f"{bt.total_trades}"),
        ("Win Rate",
         f"{r['win_rate']:.1f}%",
         f"{bt.win_rate * 100:.1f}%"),
        ("Avg Trade P&L",
         f"{r['avg_pnl_pct']:+.3f}%", "—"),
        ("Avg Winner",
         f"{r['avg_win_pct']:+.3f}%",
         f"{bt.avg_win * 100:+.3f}%"),
        ("Avg Loser",
         f"{r['avg_loss_pct']:+.3f}%",
         f"{bt.avg_loss * 100:+.3f}%"),
        ("Profit Factor", "—", f"{bt.profit_factor:.2f}"),
        ("EV Per Trade", "—", f"{bt.ev_per_trade * 100:.3f}%"),
        ("Avg Holding Days", "—", f"{bt.avg_holding_days:.1f}"),
        ("─── Exposure ───", "", ""),
        ("Avg Exposure", "—", f"{bt.avg_exposure * 100:.1f}%"),
        ("Max Positions", "—", f"{bt.max_positions}"),
        ("Open Positions (end)",
         f"{r['open_positions']}", "—"),
        ("Total Commission", "—", f"${bt.total_commission:,.2f}"),
    ]

    for label, val_r, val_bt in rows:
        if label.startswith("───"):
            print(f"  {label:<30s}")
        else:
            print(f"  {label:<30s} {val_r:>18s}  {val_bt:>18s}")

    print(f"{'═' * 72}")


def print_replay_trade_breakdown(replay_results: dict) -> None:
    """Print detailed trade breakdown from replay results."""
    r = replay_results
    tdf = r.get("trades_df")
    if tdf is None or len(tdf) == 0:
        return

    print(f"\n📋 Replay Trade Breakdown:")
    print(f"   Long trades : {(tdf['side'] == 'long').sum()}")
    print(f"   Short trades: {(tdf['side'] == 'short').sum()}")
    if "holding_days" in tdf.columns:
        print(f"   Avg holding  : {tdf['holding_days'].mean():.1f} days")
        print(f"   Med holding  : {tdf['holding_days'].median():.0f} days")
    print(f"   Best trade   : {tdf['pnl_pct'].max() * 100:+.2f}%  ({tdf.loc[tdf['pnl_pct'].idxmax(), 'symbol']})")
    print(f"   Worst trade  : {tdf['pnl_pct'].min() * 100:+.2f}%  ({tdf.loc[tdf['pnl_pct'].idxmin(), 'symbol']})")
    print(f"   Total P&L $  : ${tdf['pnl'].sum():,.2f}")
    print(f"   Commission $ : ${tdf['commission'].sum():,.2f}")

    # Monthly returns from equity curve
    eq = r.get("equity_curve")
    if eq is not None and len(eq) > 1:
        monthly = eq.resample("ME").last().pct_change().dropna() * 100
        if len(monthly) > 0:
            print(f"\n📅 Monthly Return Stats (Replay):")
            print(f"   Mean     : {monthly.mean():+.2f}%")
            print(f"   Median   : {monthly.median():+.2f}%")
            print(f"   Std Dev  : {monthly.std():.2f}%")
            print(f"   Best mo  : {monthly.max():+.2f}%")
            print(f"   Worst mo : {monthly.min():+.2f}%")
            print(f"   Positive : {(monthly > 0).sum()}/{len(monthly)} months")


def plot_equity_comparison(
    sim: SimulationEngine,
    bt_equity: pd.Series,
    replay_results: dict,
) -> None:
    """
    Plot replay vs backtest equity curves and print alignment metrics.

    Args:
        sim: SimulationEngine used for replay.
        bt_equity: Backtest equity curve series.
        replay_results: Replay results dict (for equity_curve).
    """
    comparison = sim.compare_with_backtest(
        backtest_equity=bt_equity,
        label_sim="Replay (Live Pipeline)",
        label_bt="Backtest Engine",
    )

    print(f"\n📊 Alignment Metrics:")
    print(f"   Correlation       : {comparison['correlation']:.4f}")
    print(f"   Tracking Error    : {comparison['tracking_error_pct']:.2f}% (annualized)")
    print(f"   Max Deviation     : {comparison['max_deviation']:.2f}")
    print(f"   Avg Deviation     : {comparison['avg_deviation']:.2f}")
    print(f"   Overlapping Days  : {comparison['common_days']}")

    if comparison["correlation"] > 0.95:
        print("\n✅ Strong match! Live pipeline is consistent with backtest.")
    elif comparison["correlation"] > 0.85:
        print("\n⚠️  Moderate match. Check for timing/sizing differences.")
    else:
        print("\n❌ Significant divergence — investigate before going live.")


# ═══════════════════════════════════════════════════════════════════════════
# EXPORT CHARTS  (notebook Cell 10)
# ═══════════════════════════════════════════════════════════════════════════

def plot_export_charts(snapshots_df: pd.DataFrame, mode_label: str) -> None:
    """
    Plot equity curve, daily P&L, and position count from daily snapshots.

    Args:
        snapshots_df: DataFrame with columns [equity, daily_pnl, n_positions].
        mode_label: Trading mode string for chart title.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Equity
    axes[0].plot(snapshots_df["equity"], linewidth=2, color="blue")
    axes[0].set_ylabel("Equity ($)")
    axes[0].set_title(f"Equity Curve — {mode_label.upper()} Mode")
    axes[0].grid(alpha=0.3)

    # Daily P&L
    colors = ["green" if x >= 0 else "red" for x in snapshots_df["daily_pnl"]]
    axes[1].bar(snapshots_df.index, snapshots_df["daily_pnl"], color=colors, alpha=0.7)
    axes[1].set_ylabel("Daily P&L ($)")
    axes[1].set_title("Daily P&L")
    axes[1].grid(alpha=0.3)

    # Position count
    axes[2].fill_between(snapshots_df.index, 0, snapshots_df["n_positions"],
                         alpha=0.5, color="purple")
    axes[2].set_ylabel("Positions")
    axes[2].set_title("Open Positions")
    axes[2].set_xlabel("Date")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
