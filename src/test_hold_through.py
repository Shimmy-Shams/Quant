"""
Hold-Through A/B Comparison — Before vs After across multiple time windows.

Runs the BacktestEngine with:
  - Baseline (hold_through_enabled=False)
  - Hold-Through (hold_through_enabled=True)

Shows side-by-side performance table per window + delta analysis.
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
from backtest.engine import BacktestEngine, BacktestConfig

# ============================================================================
#  DATA LOADING
# ============================================================================

print("=" * 90)
print("  HOLD-THROUGH A/B COMPARISON")
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

sym_list = list(signal_df.columns)
n_days = len(common_idx)
n_syms = len(sym_list)
print(f"Full range: {common_idx[0].date()} → {common_idx[-1].date()}  ({n_days} days, {n_syms} symbols)\n")


# ============================================================================
#  CONFIG VARIANTS
# ============================================================================

base_cfg = config.to_backtest_config()

def make_cfg(hold_through: bool):
    """Create a BacktestConfig with hold_through toggled."""
    d = asdict(base_cfg)
    d["hold_through_enabled"] = hold_through
    return BacktestConfig(**d)

VARIANTS = [
    ("No Hold-Thru", make_cfg(hold_through=False)),
    ("Hold-Through", make_cfg(hold_through=True)),
]


# ============================================================================
#  TIME WINDOWS
# ============================================================================

end_dt = common_idx[-1]
WINDOWS = []
for label, years in [("1 Year", 1), ("2 Year", 2), ("3 Year", 3), ("5 Year", 5)]:
    cutoff = end_dt - pd.DateOffset(years=years)
    mask = common_idx >= cutoff
    if mask.sum() >= 60:
        WINDOWS.append((label, mask))
WINDOWS.append(("Full", pd.Series(True, index=common_idx).values.astype(bool)))


# ============================================================================
#  RUN BACKTESTS
# ============================================================================

def run_engine(cfg, p_df, s_df, v_df, z_df):
    engine = BacktestEngine(cfg)
    return engine.run_backtest(p_df, s_df, v_df, exit_signal_data=z_df)

all_results = {}
total_runs = len(WINDOWS) * len(VARIANTS)
run_num = 0

for win_label, win_mask in WINDOWS:
    all_results[win_label] = {}
    p = price_df.loc[win_mask] if isinstance(win_mask, pd.Series) else price_df[win_mask]
    s = signal_df.loc[win_mask] if isinstance(win_mask, pd.Series) else signal_df[win_mask]
    v = volume_df.loc[win_mask] if isinstance(win_mask, pd.Series) else volume_df[win_mask]
    z = zscore_df.loc[win_mask] if isinstance(win_mask, pd.Series) else zscore_df[win_mask]

    for var_label, var_cfg in VARIANTS:
        run_num += 1
        t0 = time.perf_counter()
        res = run_engine(var_cfg, p, s, v, z)
        elapsed = time.perf_counter() - t0
        all_results[win_label][var_label] = res
        print(f"  [{run_num:2d}/{total_runs}] {win_label:12s} | {var_label:13s} | "
              f"{res.total_trades:5d} trades | Sharpe {res.sharpe_ratio:6.3f} | {elapsed:.1f}s")


# ============================================================================
#  DISPLAY RESULTS
# ============================================================================

def fmt(val, pct=False, plain=False, decimals=3):
    if plain:
        return f"{val}"
    elif pct:
        return f"{val * 100:.2f}%"
    else:
        return f"{val:.{decimals}f}"


def delta_str(base_val, test_val, pct=False, invert=False):
    diff = test_val - base_val
    if pct:
        s = f"{diff * 100:+.2f}pp"
    else:
        s = f"{diff:+.3f}"
    if abs(diff) < 1e-6:
        return f"{s} (=)"
    better = diff < 0 if invert else diff > 0
    return f"{s} {'✓' if better else '✗'}"


print("\n\n")
print("█" * 90)
print("  HOLD-THROUGH COMPARISON RESULTS")
print("█" * 90)

METRICS = [
    ("Total Trades",      "total_trades",      False, True,  False),
    ("Sharpe Ratio",      "sharpe_ratio",       False, False, False),
    ("Ann. Return",       "annualized_return",  True,  False, False),
    ("Win Rate",          "win_rate",           True,  False, False),
    ("Max Drawdown",      "max_drawdown",       True,  False, True),
    ("Profit Factor",     "profit_factor",      False, False, False),
    ("Sortino Ratio",     "sortino_ratio",      False, False, False),
    ("EV Per Trade",      "ev_per_trade",       True,  False, False),
    ("Avg Holding (d)",   "avg_holding_days",   False, False, False),
    ("Avg Exposure",      "avg_exposure",       True,  False, False),
]

for win_label, _ in WINDOWS:
    results = all_results[win_label]
    baseline = results["No Hold-Thru"]
    holdthru = results["Hold-Through"]

    if isinstance(_, pd.Series):
        days = int(_.sum())
    else:
        days = int(_.sum()) if hasattr(_, 'sum') else len(price_df[_])

    print(f"\n\n{'─' * 70}")
    print(f"  {win_label}  ({days} trading days)")
    print(f"{'─' * 70}")

    col_w = 14
    header = f"  {'Metric':22s}"
    for var_label, _ in VARIANTS:
        header += f" {var_label:>{col_w}s}"
    header += f"  {'Δ HoldThru':>18s}"
    print(header)
    print("  " + "-" * (22 + (col_w + 1) * len(VARIANTS) + 20))

    for m_label, m_attr, is_pct, is_plain, is_inverted in METRICS:
        vals = []
        for var_label, _ in VARIANTS:
            v = getattr(results[var_label], m_attr, 0)
            vals.append(v)

        row = f"  {m_label:22s}"
        for v in vals:
            row += f" {fmt(v, pct=is_pct, plain=is_plain):>{col_w}s}"

        d = delta_str(vals[0], vals[1], pct=is_pct, invert=is_inverted)
        row += f"  {d:>18s}"
        print(row)


# ============================================================================
#  CROSS-WINDOW SUMMARY
# ============================================================================

print("\n\n")
print("=" * 70)
print("  CROSS-WINDOW SUMMARY: HOLD-THROUGH IMPACT")
print("=" * 70)

summary_header = f"  {'Window':12s} {'Trades Δ':>10s} {'Sharpe Δ':>10s} {'Return Δ':>10s} {'DD Δ':>10s} {'WinRate Δ':>10s}"
print(summary_header)
print("  " + "-" * 62)

for win_label, _ in WINDOWS:
    base = all_results[win_label]["No Hold-Thru"]
    test = all_results[win_label]["Hold-Through"]

    trade_d = test.total_trades - base.total_trades
    sharpe_d = test.sharpe_ratio - base.sharpe_ratio
    ret_d = (test.annualized_return - base.annualized_return) * 100
    dd_d = (test.max_drawdown - base.max_drawdown) * 100
    wr_d = (test.win_rate - base.win_rate) * 100

    print(f"  {win_label:12s} {trade_d:>+10d} {sharpe_d:>+10.3f} {ret_d:>+9.2f}pp {dd_d:>+9.2f}pp {wr_d:>+9.2f}pp")

print("\n✓ = improvement,  ✗ = degradation")
print("Hold-through suppresses signal-based exits when same-direction re-entry would occur.")
print("Protective exits (stop-loss, trailing stop, take-profit, max holding, time decay) are ALWAYS honored.")
print()
