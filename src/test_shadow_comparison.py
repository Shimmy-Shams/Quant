"""
Multi-Period Shadow A/B Comparison — Filters ON vs OFF across multiple time windows.

Runs the BacktestEngine on recent 1yr, 2yr, 3yr, 5yr, and Full (20yr) windows
with 4 configurations:
  - Baseline (no filters)
  - Tier 2 Only (Earnings Blackout)
  - Tier 1 Only (Sentiment Penalty)
  - Both Tiers

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
print("  MULTI-PERIOD SHADOW A/B COMPARISON")
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

def make_cfg(earnings=False, sentiment=False):
    """Create a BacktestConfig with specified filter settings."""
    d = asdict(base_cfg)
    d["earnings_blackout_enabled"] = earnings
    d["earnings_blackout_days"] = 2
    d["sentiment_penalty_enabled"] = sentiment
    d["sentiment_penalty_floor"] = 0.5
    d["sentiment_negative_threshold"] = -0.3
    d["sentiment_proxy_lookback"] = 5
    d["sentiment_proxy_drop_threshold"] = -0.08
    return BacktestConfig(**d)

VARIANTS = [
    ("Baseline",  make_cfg(earnings=False, sentiment=False)),
    ("T2 (Earn)", make_cfg(earnings=True,  sentiment=False)),
    ("T1 (Sent)", make_cfg(earnings=False, sentiment=True)),
    ("Both",      make_cfg(earnings=True,  sentiment=True)),
]


# ============================================================================
#  TIME WINDOWS
# ============================================================================

end_dt = common_idx[-1]
WINDOWS = []
for label, years in [("1 Year", 1), ("2 Year", 2), ("3 Year", 3), ("5 Year", 5)]:
    cutoff = end_dt - pd.DateOffset(years=years)
    mask = common_idx >= cutoff
    if mask.sum() >= 60:  # need at least 60 trading days
        WINDOWS.append((label, mask))
WINDOWS.append(("Full (20yr)", pd.Series(True, index=common_idx).values.astype(bool)))


# ============================================================================
#  RUN BACKTESTS
# ============================================================================

def run_engine(cfg, p_df, s_df, v_df, z_df):
    """Run backtest engine, return results."""
    engine = BacktestEngine(cfg)
    return engine.run_backtest(p_df, s_df, v_df, exit_signal_data=z_df)


# Store all results: results[window_label][variant_label] = BacktestResults
all_results = {}
total_runs = len(WINDOWS) * len(VARIANTS)
run_num = 0

for win_label, win_mask in WINDOWS:
    all_results[win_label] = {}
    p = price_df.loc[win_mask] if isinstance(win_mask, pd.Series) else price_df[win_mask]
    s = signal_df.loc[win_mask] if isinstance(win_mask, pd.Series) else signal_df[win_mask]
    v = volume_df.loc[win_mask] if isinstance(win_mask, pd.Series) else volume_df[win_mask]
    z = zscore_df.loc[win_mask] if isinstance(win_mask, pd.Series) else zscore_df[win_mask]

    trading_days = len(p)

    for var_label, var_cfg in VARIANTS:
        run_num += 1
        t0 = time.perf_counter()
        res = run_engine(var_cfg, p, s, v, z)
        elapsed = time.perf_counter() - t0
        all_results[win_label][var_label] = res
        print(f"  [{run_num:2d}/{total_runs}] {win_label:12s} | {var_label:10s} | "
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
    """Show delta. invert=True means lower is better (e.g. drawdown)."""
    diff = test_val - base_val
    if pct:
        s = f"{diff * 100:+.2f}pp"
    else:
        s = f"{diff:+.3f}"
    # Color indicator
    if abs(diff) < 1e-6:
        return f"{s} (=)"
    better = diff < 0 if invert else diff > 0
    return f"{s} {'✓' if better else '✗'}"


print("\n\n")
print("█" * 90)
print("  MULTI-PERIOD COMPARISON RESULTS")
print("█" * 90)

METRICS = [
    ("Total Trades",      "total_trades",      False, True,  False),
    ("Sharpe Ratio",      "sharpe_ratio",       False, False, False),
    ("Ann. Return",       "annualized_return",  True,  False, False),
    ("Win Rate",          "win_rate",           True,  False, False),
    ("Max Drawdown",      "max_drawdown",       True,  False, True),   # invert: lower DD is better
    ("Profit Factor",     "profit_factor",      False, False, False),
    ("Sortino Ratio",     "sortino_ratio",      False, False, False),
    ("EV Per Trade",      "ev_per_trade",       True,  False, False),
    ("Avg Holding (d)",   "avg_holding_days",   False, False, False),
    ("Avg Exposure",      "avg_exposure",       True,  False, False),
]

for win_label, _ in WINDOWS:
    results = all_results[win_label]
    baseline = results["Baseline"]

    if isinstance(_, pd.Series):
        days = int(_.sum())
    else:
        days = int(_.sum()) if hasattr(_, 'sum') else len(price_df[_])

    print(f"\n\n{'─' * 90}")
    print(f"  {win_label}  ({days} trading days)")
    print(f"{'─' * 90}")

    # Header
    col_w = 14
    header = f"  {'Metric':22s}"
    for var_label, _ in VARIANTS:
        header += f" {var_label:>{col_w}s}"
    header += f"  {'Δ Both vs Base':>18s}"
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

        # Delta: Both vs Baseline
        d = delta_str(vals[0], vals[3], pct=is_pct, invert=is_inverted)
        row += f"  {d:>18s}"
        print(row)

    # Filter diagnostics for "Both"
    both_res = results["Both"]
    diag = both_res.filter_diagnostics
    if diag:
        print(f"\n  Filter Diagnostics (Both):")
        t2_blocked = diag.get("earnings_blocked", 0)
        t1_reduced = diag.get("sentiment_reduced", 0)
        total_entries = diag.get("total_entries_evaluated", 0)
        print(f"    Entries evaluated: {total_entries}")
        print(f"    T2 blocked:       {t2_blocked} ({t2_blocked/max(total_entries,1)*100:.2f}%)")
        print(f"    T1 reduced:       {t1_reduced} ({t1_reduced/max(total_entries,1)*100:.2f}%)")


# ============================================================================
#  CROSS-WINDOW SUMMARY (key metric: Sharpe delta)
# ============================================================================

print("\n\n")
print("=" * 90)
print("  CROSS-WINDOW SUMMARY: Impact of Both Filters vs Baseline")
print("=" * 90)

header = f"  {'Window':14s} {'Base Sharpe':>12s} {'Both Sharpe':>12s} {'Δ Sharpe':>10s} " \
         f"{'Base Ret':>10s} {'Both Ret':>10s} {'Δ Ret':>10s} " \
         f"{'Base Trades':>12s} {'Both Trades':>12s}"
print(header)
print("  " + "-" * 100)

for win_label, _ in WINDOWS:
    b = all_results[win_label]["Baseline"]
    f_both = all_results[win_label]["Both"]

    d_sharpe = f_both.sharpe_ratio - b.sharpe_ratio
    d_ret = (f_both.annualized_return - b.annualized_return) * 100

    sharpe_ind = "✓" if abs(d_sharpe) < 0.3 else ("✗" if d_sharpe < -0.3 else "✓")
    ret_ind = "✓" if abs(d_ret) < 3.0 else ("✗" if d_ret < -3.0 else "✓")

    print(f"  {win_label:14s} "
          f"{b.sharpe_ratio:12.3f} {f_both.sharpe_ratio:12.3f} {d_sharpe:+9.3f} {sharpe_ind} "
          f"{b.annualized_return*100:9.2f}% {f_both.annualized_return*100:9.2f}% {d_ret:+9.2f}pp {ret_ind} "
          f"{b.total_trades:12d} {f_both.total_trades:12d}")


# ============================================================================
#  VERDICT
# ============================================================================

print("\n\n" + "=" * 90)
print("  VERDICT")
print("=" * 90)

# Look at recent windows (1yr, 2yr) for practical impact
recent_windows = [w for w in ["1 Year", "2 Year"] if w in all_results]
max_sharpe_drop = 0
max_return_drop = 0

for w in recent_windows:
    b = all_results[w]["Baseline"]
    f_both = all_results[w]["Both"]
    d_sharpe = b.sharpe_ratio - f_both.sharpe_ratio  # positive = Baseline was better
    d_ret = (b.annualized_return - f_both.annualized_return) * 100
    max_sharpe_drop = max(max_sharpe_drop, d_sharpe)
    max_return_drop = max(max_return_drop, d_ret)

if max_sharpe_drop <= 0.3 and max_return_drop <= 3.0:
    print("\n  ✅ PASS — Filters are safe to deploy.")
    print(f"     Max Sharpe drop in recent windows: {max_sharpe_drop:.3f} (threshold: 0.30)")
    print(f"     Max Return drop in recent windows: {max_return_drop:.2f}pp (threshold: 3.00pp)")
    print("     Performance remains competitive while adding earnings/sentiment protection.")
elif max_sharpe_drop <= 0.5 and max_return_drop <= 5.0:
    print("\n  ⚠️  MARGINAL — Filters have moderate cost. Consider tuning parameters.")
    print(f"     Max Sharpe drop: {max_sharpe_drop:.3f}")
    print(f"     Max Return drop: {max_return_drop:.2f}pp")
else:
    print("\n  ❌ FAIL — Filters have significant performance cost.")
    print(f"     Max Sharpe drop: {max_sharpe_drop:.3f}")
    print(f"     Max Return drop: {max_return_drop:.2f}pp")
    print("     Consider disabling or significantly loosening parameters.")

print()
