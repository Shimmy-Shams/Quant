"""
Trailing Stop Parameter Sweep — Find optimal activation & trail values.

Sweeps:
  - trailing_stop_activation: [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]
  - trailing_stop_pct:        [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]
  - use_trailing_stop:        True vs False baseline

Runs on 1yr, 2yr, and Full windows. Outputs ranked tables by Sharpe.
"""

import sys, time
from pathlib import Path
from dataclasses import asdict

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from strategy_config import ConfigLoader
from strategies.mean_reversion import MeanReversionSignals, UniverseAnalyzer
from backtest.engine import BacktestEngine, BacktestConfig

# ============================================================================
#  DATA LOADING
# ============================================================================

print("=" * 100)
print("  TRAILING STOP PARAMETER SWEEP")
print("=" * 100)

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
#  PARAMETER GRID
# ============================================================================

ACTIVATION_VALUES = [0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]
TRAIL_VALUES      = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]

base_cfg = config.to_backtest_config()

# Build all configs: baseline (no trailing) + grid
configs = []

# Baseline: trailing stop OFF
d = asdict(base_cfg)
d["use_trailing_stop"] = False
configs.append(("OFF", 0.0, 0.0, BacktestConfig(**d)))

# Grid: all activation × trail combos (skip where trail < activation — makes no sense)
for act in ACTIVATION_VALUES:
    for trail in TRAIL_VALUES:
        d = asdict(base_cfg)
        d["use_trailing_stop"] = True
        d["trailing_stop_activation"] = act
        d["trailing_stop_pct"] = trail
        configs.append((f"act={act:.0%}/trail={trail:.0%}", act, trail, BacktestConfig(**d)))

print(f"Sweep: {len(configs)} configurations ({len(ACTIVATION_VALUES)} activations × {len(TRAIL_VALUES)} trails + baseline)")

# ============================================================================
#  TIME WINDOWS
# ============================================================================

end_dt = common_idx[-1]
WINDOWS = []
for label, years in [("1yr", 1), ("2yr", 2), ("Full", None)]:
    if years:
        cutoff = end_dt - pd.DateOffset(years=years)
        mask = common_idx >= cutoff
    else:
        mask = pd.Series(True, index=common_idx).values.astype(bool)
    WINDOWS.append((label, mask))

# ============================================================================
#  RUN SWEEP
# ============================================================================

results_rows = []
total_runs = len(WINDOWS) * len(configs)
run_num = 0

for win_label, win_mask in WINDOWS:
    p = price_df.loc[win_mask] if isinstance(win_mask, pd.Series) else price_df[win_mask]
    s = signal_df.loc[win_mask] if isinstance(win_mask, pd.Series) else signal_df[win_mask]
    v = volume_df.loc[win_mask] if isinstance(win_mask, pd.Series) else volume_df[win_mask]
    z = zscore_df.loc[win_mask] if isinstance(win_mask, pd.Series) else zscore_df[win_mask]

    t_win = time.perf_counter()
    for label, act, trail, cfg in configs:
        run_num += 1
        engine = BacktestEngine(cfg)
        res = engine.run_backtest(p, s, v, exit_signal_data=z)

        # Count trailing stop exits
        ts_exits = sum(1 for t in res.trades if t.exit_reason == 'trailing_stop')

        results_rows.append({
            'window': win_label,
            'label': label,
            'activation': act,
            'trail': trail,
            'sharpe': res.sharpe_ratio,
            'ann_return': res.annualized_return,
            'max_dd': res.max_drawdown,
            'win_rate': res.win_rate,
            'profit_factor': res.profit_factor,
            'sortino': res.sortino_ratio,
            'total_trades': res.total_trades,
            'ev_per_trade': res.ev_per_trade,
            'avg_holding': res.avg_holding_days,
            'ts_exits': ts_exits,
        })

        if run_num % 20 == 0 or run_num == total_runs:
            print(f"  [{run_num:3d}/{total_runs}] {win_label} | {label:25s} | "
                  f"Sharpe {res.sharpe_ratio:6.3f} | Trades {res.total_trades:5d} | TS exits {ts_exits}")

    elapsed = time.perf_counter() - t_win
    print(f"  {win_label} complete ({elapsed:.1f}s)\n")

df = pd.DataFrame(results_rows)

# ============================================================================
#  RESULTS — RANKED BY SHARPE PER WINDOW
# ============================================================================

print("\n\n")
print("█" * 100)
print("  TRAILING STOP SWEEP RESULTS — TOP 15 PER WINDOW (by Sharpe)")
print("█" * 100)

for win_label in df['window'].unique():
    wdf = df[df['window'] == win_label].copy()
    wdf = wdf.sort_values('sharpe', ascending=False).head(15)

    # Get baseline row for this window
    baseline = df[(df['window'] == win_label) & (df['label'] == 'OFF')].iloc[0]

    print(f"\n\n{'─' * 100}")
    print(f"  {win_label}  (Baseline: Sharpe={baseline['sharpe']:.3f}, Return={baseline['ann_return']*100:.2f}%, DD={baseline['max_dd']*100:.2f}%)")
    print(f"{'─' * 100}")

    header = (f"  {'Rank':4s} {'Activation':>10s} {'Trail':>8s} {'Sharpe':>8s} {'ΔSharpe':>8s} "
              f"{'Return':>9s} {'MaxDD':>8s} {'WinRate':>8s} {'PF':>6s} {'Trades':>7s} {'TS Exits':>9s} "
              f"{'EV/Trade':>9s} {'AvgHold':>8s}")
    print(header)
    print("  " + "-" * 98)

    for rank, (_, row) in enumerate(wdf.iterrows(), 1):
        d_sharpe = row['sharpe'] - baseline['sharpe']
        marker = " ◀ CURRENT" if row['label'] == f"act={base_cfg.trailing_stop_activation:.0%}/trail={base_cfg.trailing_stop_pct:.0%}" else ""
        if row['label'] == 'OFF':
            marker = " ◀ NO TRAIL"

        print(f"  {rank:3d}. {row['activation']*100:>8.1f}% {row['trail']*100:>7.1f}% "
              f"{row['sharpe']:>8.3f} {d_sharpe:>+8.3f} "
              f"{row['ann_return']*100:>8.2f}% {row['max_dd']*100:>7.2f}% "
              f"{row['win_rate']*100:>7.2f}% {row['profit_factor']:>6.2f} "
              f"{row['total_trades']:>7d} {row['ts_exits']:>9d} "
              f"{row['ev_per_trade']*100:>8.3f}% {row['avg_holding']:>7.2f}"
              f"{marker}")


# ============================================================================
#  HEATMAP — SHARPE BY ACTIVATION × TRAIL
# ============================================================================

print("\n\n")
print("=" * 100)
print("  SHARPE HEATMAPS BY ACTIVATION × TRAIL")
print("=" * 100)

for win_label in df['window'].unique():
    wdf = df[(df['window'] == win_label) & (df['label'] != 'OFF')]
    pivot = wdf.pivot_table(values='sharpe', index='activation', columns='trail', aggfunc='first')

    baseline_sharpe = df[(df['window'] == win_label) & (df['label'] == 'OFF')].iloc[0]['sharpe']

    print(f"\n  {win_label}  (baseline Sharpe = {baseline_sharpe:.3f})")
    print(f"  {'Activation↓ / Trail→':>20s}", end="")
    for t in pivot.columns:
        print(f"  {t*100:>5.0f}%", end="")
    print()
    print("  " + "-" * (22 + 7 * len(pivot.columns)))

    for act in pivot.index:
        print(f"  {act*100:>18.1f}%  ", end="")
        for t in pivot.columns:
            val = pivot.loc[act, t]
            if pd.isna(val):
                print(f"  {'--':>5s}", end="")
            else:
                d = val - baseline_sharpe
                marker = "+" if d > 0 else "-" if d < 0 else "="
                print(f"  {val:>5.2f}{marker}", end="")
        print()


# ============================================================================
#  CONSISTENCY CHECK — ARE TOP PARAMS STABLE ACROSS WINDOWS?
# ============================================================================

print("\n\n")
print("=" * 100)
print("  CROSS-WINDOW CONSISTENCY — Top 5 per window")
print("=" * 100)

for win_label in df['window'].unique():
    wdf = df[(df['window'] == win_label) & (df['label'] != 'OFF')]
    top5 = wdf.nlargest(5, 'sharpe')
    print(f"\n  {win_label}:")
    for _, row in top5.iterrows():
        print(f"    act={row['activation']*100:.1f}% trail={row['trail']*100:.1f}%  →  "
              f"Sharpe={row['sharpe']:.3f}  Return={row['ann_return']*100:.1f}%  "
              f"DD={row['max_dd']*100:.2f}%  WR={row['win_rate']*100:.1f}%  "
              f"TS_exits={row['ts_exits']}")


# ============================================================================
#  RECOMMENDATION
# ============================================================================

print("\n\n")
print("=" * 100)
print("  RECOMMENDATION")
print("=" * 100)

# Find params that appear in top 5 across ALL windows
from collections import Counter
all_top5_params = []
for win_label in df['window'].unique():
    wdf = df[(df['window'] == win_label) & (df['label'] != 'OFF')]
    top5 = wdf.nlargest(5, 'sharpe')
    for _, row in top5.iterrows():
        all_top5_params.append((row['activation'], row['trail']))

param_counts = Counter(all_top5_params)
n_windows = df['window'].nunique()

print(f"\n  Parameters appearing in Top 5 across multiple windows (of {n_windows}):")
for (act, trail), count in param_counts.most_common(10):
    avg_sharpe = df[(df['activation'] == act) & (df['trail'] == trail)]['sharpe'].mean()
    print(f"    act={act*100:.1f}% trail={trail*100:.1f}%  → in {count}/{n_windows} windows  avg Sharpe={avg_sharpe:.3f}")

# Current config comparison
curr_act = base_cfg.trailing_stop_activation
curr_trail = base_cfg.trailing_stop_pct
curr_rows = df[(df['activation'] == curr_act) & (df['trail'] == curr_trail)]
if len(curr_rows) > 0:
    print(f"\n  Current config (act={curr_act*100:.0f}% trail={curr_trail*100:.0f}%):")
    for _, row in curr_rows.iterrows():
        baseline_s = df[(df['window'] == row['window']) & (df['label'] == 'OFF')].iloc[0]['sharpe']
        print(f"    {row['window']}: Sharpe={row['sharpe']:.3f} (vs baseline {baseline_s:.3f}, Δ={row['sharpe']-baseline_s:+.3f})")

print()
