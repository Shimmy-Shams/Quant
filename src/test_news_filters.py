"""
News Filter A/B Test — Tier 1 (Sentiment Penalty) + Tier 2 (Earnings Blackout)
Runs backtest twice: baseline vs filters-enabled, prints comparison table.
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

# ── Load config ──
config = ConfigLoader(project_root / "config.yaml")

# ── Load data ──
data_dir = project_root / "data" / "historical" / "daily"
parquet_files = list(data_dir.glob("*.parquet"))
print(f"Found {len(parquet_files)} parquet files")

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
print(f"Loaded {len(all_data)} symbols after filtering")

# ── Universe analysis ──
signal_config = config.to_signal_config()
analyzer = UniverseAnalyzer(signal_config)
min_history = config.get("data.min_history", 100)
print("Analyzing universe...")
analysis_df = analyzer.analyze_universe(price_data, min_history=min_history)
mean_reverting = analysis_df[analysis_df["is_mean_reverting"]]["symbol"].tolist()
print(f"Mean-reverting universe: {len(mean_reverting)} symbols")

# ── Generate signals ──
signal_gen = MeanReversionSignals(signal_config)
composite_weights = config.get_composite_weights()
print("Generating signals...")
all_signals, all_individual = {}, {}
for sym in mean_reverting:
    if sym in price_data and sym in volume_data:
        comp, ind = signal_gen.generate_composite_signal(
            price_data[sym], volume_data[sym], weights=composite_weights
        )
        all_signals[sym] = comp
        all_individual[sym] = ind
print(f"Signals for {len(all_signals)} symbols")

# ── Build aligned DataFrames ──
price_df = pd.DataFrame(price_data)
signal_df = pd.DataFrame(all_signals)
volume_df = pd.DataFrame(volume_data)
zscore_df = pd.DataFrame({
    s: ind["zscore"] for s, ind in all_individual.items() if "zscore" in ind
})

common_idx = price_df.index.intersection(signal_df.index)
price_df = price_df.loc[common_idx]
signal_df = signal_df.loc[common_idx]
volume_df = volume_df.loc[common_idx]
zscore_df = zscore_df.loc[common_idx]

print(f"\nBacktest range: {common_idx.min().date()} → {common_idx.max().date()}  ({len(common_idx)} days)")
print(f"Symbols: {len(signal_df.columns)}")

# ── Helper ──
def run_variant(label, bt_cfg):
    print(f"\n{'='*60}")
    print(f"  Running: {label}")
    print(f"{'='*60}")
    t0 = time.perf_counter()
    engine = BacktestEngine(bt_cfg)
    res = engine.run_backtest(price_df, signal_df, volume_df, exit_signal_data=zscore_df)
    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.1f}s — {res.total_trades} trades, Sharpe {res.sharpe_ratio:.2f}")
    return res

# ── A: Baseline (filters OFF) ──
base_cfg = config.to_backtest_config()
base_dict = asdict(base_cfg)
base_dict["earnings_blackout_enabled"] = False
base_dict["sentiment_penalty_enabled"] = False
baseline_cfg = BacktestConfig(**base_dict)
baseline = run_variant("BASELINE (filters OFF)", baseline_cfg)

# ── B: Filters ON ──
filt_dict = asdict(base_cfg)
filt_dict["earnings_blackout_enabled"] = True
filt_dict["earnings_blackout_days"] = 2
filt_dict["sentiment_penalty_enabled"] = True
filt_dict["sentiment_penalty_floor"] = 0.5
filt_dict["sentiment_negative_threshold"] = -0.3
filt_dict["sentiment_proxy_lookback"] = 5
filt_dict["sentiment_proxy_drop_threshold"] = -0.08
filtered_cfg = BacktestConfig(**filt_dict)
filtered = run_variant("FILTERED (Tier 1 + Tier 2 ON)", filtered_cfg)

# ── Comparison ──
def fmt(v, pct=False, dollar=False, plain=False):
    if pct:
        return f"{v*100:.2f}%"
    if dollar:
        return f"${v:,.0f}"
    if plain:
        return f"{v}"
    return f"{v:.2f}"

print("\n")
print("=" * 78)
print("  NEWS FILTER A/B COMPARISON")
print("=" * 78)
header = f"{'Metric':30s} {'Baseline':>20s} {'Filtered':>20s} {'Delta':>10s}"
print(header)
print("-" * 78)

rows = [
    ("Total Return", baseline.total_return, filtered.total_return, True),
    ("Annualized Return", baseline.annualized_return, filtered.annualized_return, True),
    ("Sharpe Ratio", baseline.sharpe_ratio, filtered.sharpe_ratio, False),
    ("Sortino Ratio", baseline.sortino_ratio, filtered.sortino_ratio, False),
    ("Max Drawdown", baseline.max_drawdown, filtered.max_drawdown, True),
    ("Win Rate", baseline.win_rate, filtered.win_rate, True),
    ("Total Trades", float(baseline.total_trades), float(filtered.total_trades), False),
    ("Profit Factor", baseline.profit_factor, filtered.profit_factor, False),
    ("Avg Win", baseline.avg_win, filtered.avg_win, True),
    ("Avg Loss", baseline.avg_loss, filtered.avg_loss, True),
    ("Avg Holding Days", baseline.avg_holding_days, filtered.avg_holding_days, False),
    ("Avg Exposure", baseline.avg_exposure, filtered.avg_exposure, True),
    ("Final Equity", baseline.equity_curve.iloc[-1], filtered.equity_curve.iloc[-1], False),
]

for label, bv, fv, is_pct in rows:
    if label == "Total Trades":
        b_str = f"{int(bv)}"
        f_str = f"{int(fv)}"
        d_str = f"{int(fv - bv):+d}"
    elif label == "Final Equity":
        b_str = f"${bv:,.0f}"
        f_str = f"${fv:,.0f}"
        d_str = f"${fv - bv:+,.0f}"
    elif is_pct:
        b_str = f"{bv * 100:.2f}%"
        f_str = f"{fv * 100:.2f}%"
        d_str = f"{(fv - bv) * 100:+.2f}%"
    else:
        b_str = f"{bv:.2f}"
        f_str = f"{fv:.2f}"
        d_str = f"{fv - bv:+.2f}"
    print(f"{label:30s} {b_str:>20s} {f_str:>20s} {d_str:>10s}")

print("=" * 78)

# Trade-level detail: how many blocked by earnings? sentiment penalty effect?
if filtered.total_trades < baseline.total_trades:
    blocked = baseline.total_trades - filtered.total_trades
    print(f"\n  Trades blocked by filters: {blocked} ({blocked / baseline.total_trades * 100:.1f}%)")

# Exit reason comparison
print("\n  Exit reason breakdown:")
for res_obj, label in [(baseline, "Baseline"), (filtered, "Filtered")]:
    from collections import Counter
    reasons = Counter(t.exit_reason for t in res_obj.trades)
    parts = ", ".join(f"{r}: {c}" for r, c in reasons.most_common())
    print(f"    {label}: {parts}")

print("\nDone.")
