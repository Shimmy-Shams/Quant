"""
Hold-Through REPLAY A/B Comparison — Before vs After.

Runs the SimulationEngine replay (day-by-day through the live pipeline) with:
  - Baseline  (hold_through_enabled=False)
  - Hold-Through (hold_through_enabled=True)

This is the most realistic test: it uses generate_decisions_from_signals()
exactly like the live trader does, processing one day at a time.
"""

import sys, time
from pathlib import Path
from dataclasses import asdict

import numpy as np
import pandas as pd

# ── project root on PATH ──
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from strategy_config import ConfigLoader
from strategies.mean_reversion import MeanReversionSignals, UniverseAnalyzer
from backtest.engine import BacktestConfig
from execution.simulation import SimulationEngine
from execution.alpaca_executor import AlpacaExecutor
from connection.alpaca_connection import AlpacaConfig, AlpacaConnection, TradingMode

# ============================================================================
#  DATA LOADING
# ============================================================================

print("=" * 90)
print("  HOLD-THROUGH REPLAY A/B COMPARISON")
print("=" * 90)

config = ConfigLoader(project_root / "config.yaml")
data_dir = project_root / "data" / "historical" / "daily"
parquet_files = list(data_dir.glob("*.parquet"))
print(f"\nLoading {len(parquet_files)} parquet files...")

all_data = {}
for f in parquet_files:
    sym = f.stem
    try:
        df = pd.read_parquet(f)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        all_data[sym] = df
    except Exception:
        pass

start_date, end_date = config.get_date_range()
if start_date or end_date:
    for sym in list(all_data):
        df = all_data[sym]
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
        if len(df) < 100:
            del all_data[sym]
        else:
            all_data[sym] = df

price_data = {s: df["close"] for s, df in all_data.items()}
volume_data = {s: df["volume"] for s, df in all_data.items()}

# Universe + signals
signal_config = config.to_signal_config()
analyzer = UniverseAnalyzer(signal_config)
analysis_df = analyzer.analyze_universe(price_data, min_history=config.get("data.min_history", 100))
mean_reverting = analysis_df[analysis_df["is_mean_reverting"]]["symbol"].tolist()

signal_gen = MeanReversionSignals(signal_config)
composite_weights = config.get_composite_weights()

print("Generating signals...")
all_signals, all_individual = {}, {}
for sym in mean_reverting:
    if sym in price_data and sym in volume_data:
        comp, ind = signal_gen.generate_composite_signal(price_data[sym], volume_data[sym], weights=composite_weights)
        all_signals[sym] = comp
        all_individual[sym] = ind

price_df = pd.DataFrame(price_data)
signal_df = pd.DataFrame(all_signals)
volume_df = pd.DataFrame(volume_data)
zscore_df = pd.DataFrame({s: ind["zscore"] for s, ind in all_individual.items() if "zscore" in ind})

common_idx = price_df.index.intersection(signal_df.index)
price_df = price_df.loc[common_idx]
signal_df = signal_df.loc[common_idx]
volume_df = volume_df.loc[common_idx]
zscore_df = zscore_df.loc[common_idx]

n_days = len(common_idx)
n_syms = len(signal_df.columns)
print(f"Full range: {common_idx[0].date()} → {common_idx[-1].date()}  ({n_days} days, {n_syms} symbols)\n")


# ============================================================================
#  REPLAY RUNNER
# ============================================================================

base_cfg = config.to_backtest_config()

def run_replay_variant(hold_through: bool, label: str, years: int = 2) -> dict:
    """Run a full replay with hold-through toggled."""
    d = asdict(base_cfg)
    d["hold_through_enabled"] = hold_through
    bt_config = BacktestConfig(**d)

    # Build dummy Alpaca connection for replay mode
    alpaca_cfg = AlpacaConfig(
        api_key='REPLAY',
        secret_key='REPLAY',
        paper=True,
        trading_mode=TradingMode.REPLAY,
    )
    conn = AlpacaConnection(alpaca_cfg)

    executor = AlpacaExecutor(
        connection=conn,
        commission_pct=bt_config.commission_pct,
        max_position_pct=bt_config.max_position_size,
        max_total_exposure=bt_config.max_total_exposure,
        stop_loss_pct=getattr(bt_config, 'stop_loss_pct', None),
        take_profit_pct=getattr(bt_config, 'take_profit_pct', None),
    )

    sim = SimulationEngine(
        executor=executor,
        initial_capital=1_000_000,
        commission_pct=bt_config.commission_pct,
        slippage_pct=getattr(bt_config, 'slippage_pct', 0.0005),
        commission_model=getattr(bt_config, 'commission_model', 'flat'),
    )

    # Replay window
    replay_end = signal_df.index.max()
    replay_start = replay_end - pd.DateOffset(years=years)

    print(f"\n  Running {label}...")
    print(f"  Window: {replay_start.date()} → {replay_end.date()}")
    t0 = time.perf_counter()

    results = sim.run_replay(
        price_df=price_df,
        signal_df=signal_df,
        volume_df=volume_df,
        exit_signal_df=zscore_df,
        config=bt_config,
        start_date=replay_start,
        end_date=replay_end,
        verbose=True,
    )
    elapsed = time.perf_counter() - t0
    print(f"  Completed in {elapsed:.1f}s")

    return results


# ============================================================================
#  RUN BOTH VARIANTS
# ============================================================================

WINDOWS = [1, 2, 3]  # years

all_results = {}
for years in WINDOWS:
    win_label = f"{years}yr"
    all_results[win_label] = {}
    for hold_through, label in [(False, "No Hold-Thru"), (True, "Hold-Through")]:
        results = run_replay_variant(hold_through, f"{label} ({win_label})", years=years)
        all_results[win_label][label] = results


# ============================================================================
#  COMPARISON TABLE
# ============================================================================

print("\n\n")
print("█" * 90)
print("  HOLD-THROUGH REPLAY COMPARISON RESULTS")
print("█" * 90)


def fmt(val, pct=False, plain=False, decimals=2):
    if plain:
        return f"{val}"
    elif pct:
        return f"{val:.2f}%"
    else:
        return f"{val:.{decimals}f}"


def delta_str(base_val, test_val, pct=False, invert=False):
    diff = test_val - base_val
    if pct:
        s = f"{diff:+.2f}pp"
    else:
        s = f"{diff:+.3f}"
    if abs(diff) < 1e-6:
        return f"{s} (=)"
    better = diff < 0 if invert else diff > 0
    return f"{s} {'✓' if better else '✗'}"


METRICS = [
    ("Total Trades",   "total_trades",      False, True,  False),
    ("Sharpe Ratio",   "sharpe_ratio",       False, False, False),
    ("Total Return",   "total_return_pct",   True,  False, False),
    ("Win Rate",       "win_rate",           True,  False, False),
    ("Max Drawdown",   "max_drawdown_pct",   True,  False, True),
    ("Avg Trade P&L",  "avg_pnl_pct",        True,  False, False),
    ("Avg Winner",     "avg_win_pct",         True,  False, False),
    ("Avg Loser",      "avg_loss_pct",        True,  False, True),
    ("Final Equity",   "final_equity",        False, False, False),
    ("Open Positions", "open_positions",      False, True,  False),
]

for win_label in all_results:
    results = all_results[win_label]
    baseline = results["No Hold-Thru"]
    holdthru = results["Hold-Through"]

    print(f"\n\n{'─' * 75}")
    print(f"  REPLAY — {win_label}")
    print(f"{'─' * 75}")

    col_w = 16
    header = f"  {'Metric':22s} {'No Hold-Thru':>{col_w}s} {'Hold-Through':>{col_w}s}  {'Δ HoldThru':>18s}"
    print(header)
    print("  " + "-" * 72)

    for m_label, m_key, is_pct, is_plain, is_inverted in METRICS:
        v_base = baseline.get(m_key, 0)
        v_test = holdthru.get(m_key, 0)

        row = f"  {m_label:22s}"
        row += f" {fmt(v_base, pct=is_pct, plain=is_plain):>{col_w}s}"
        row += f" {fmt(v_test, pct=is_pct, plain=is_plain):>{col_w}s}"
        row += f"  {delta_str(v_base, v_test, pct=is_pct, invert=is_inverted):>18s}"
        print(row)

    # Show hold-through events count
    ht_trades_df = holdthru.get('trades_df', pd.DataFrame())
    base_trades_df = baseline.get('trades_df', pd.DataFrame())
    if len(base_trades_df) > 0 and len(ht_trades_df) > 0:
        avoided = len(base_trades_df) - len(ht_trades_df)
        print(f"\n  Round-trips avoided by hold-through: ~{avoided}")


# ============================================================================
#  CROSS-WINDOW SUMMARY
# ============================================================================

print("\n\n")
print("=" * 75)
print("  CROSS-WINDOW REPLAY SUMMARY: HOLD-THROUGH IMPACT")
print("=" * 75)

summary_header = f"  {'Window':8s} {'Trades Δ':>10s} {'Sharpe Δ':>10s} {'Return Δ':>12s} {'DD Δ':>10s} {'WinRate Δ':>10s}"
print(summary_header)
print("  " + "-" * 62)

for win_label in all_results:
    base = all_results[win_label]["No Hold-Thru"]
    test = all_results[win_label]["Hold-Through"]

    trade_d = test['total_trades'] - base['total_trades']
    sharpe_d = test['sharpe_ratio'] - base['sharpe_ratio']
    ret_d = test['total_return_pct'] - base['total_return_pct']
    dd_d = test['max_drawdown_pct'] - base['max_drawdown_pct']
    wr_d = test['win_rate'] - base['win_rate']

    print(f"  {win_label:8s} {trade_d:>+10d} {sharpe_d:>+10.3f} {ret_d:>+11.2f}pp {dd_d:>+9.2f}pp {wr_d:>+9.2f}pp")

print("\n✓ = improvement,  ✗ = degradation")
print("Replay mode processes signals day-by-day through the live execution pipeline.")
print("Hold-through suppresses signal-based exits when same-direction re-entry would occur.")
print("Protective exits (stop-loss, trailing stop, take-profit, max holding, time decay) ALWAYS honored.")
print()
