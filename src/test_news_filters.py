"""
News Filter Diagnostic Test — Validates Tier 1 (Sentiment) + Tier 2 (Earnings) end-to-end.

Three phases:
  Phase 1: Standalone data validation — call the modules directly, inspect raw data
  Phase 2: Engine-integrated A/B comparison with diagnostic counters
  Phase 3: Trade-level diff — which specific trades changed between baseline and filtered
"""

import sys, time, os
from pathlib import Path
from dataclasses import asdict
from collections import Counter

import numpy as np
import pandas as pd

# ── project root on PATH ──
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from strategy_config import ConfigLoader
from strategies.mean_reversion import MeanReversionSignals, UniverseAnalyzer
from backtest.engine import BacktestEngine, BacktestConfig

# ========================================================================================
#  DATA LOADING (shared by all phases)
# ========================================================================================

config = ConfigLoader(project_root / "config.yaml")
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

# Universe + signals
signal_config = config.to_signal_config()
analyzer = UniverseAnalyzer(signal_config)
analysis_df = analyzer.analyze_universe(price_data, min_history=config.get("data.min_history", 100))
mean_reverting = analysis_df[analysis_df["is_mean_reverting"]]["symbol"].tolist()
print(f"Mean-reverting universe: {len(mean_reverting)} symbols")

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
n_days, n_syms = len(common_idx), len(sym_list)
print(f"\nBacktest range: {common_idx[0].date()} -> {common_idx[-1].date()}  ({n_days} days, {n_syms} symbols)")


# ========================================================================================
#  PHASE 1: STANDALONE DATA VALIDATION
# ========================================================================================

print("\n")
print("=" * 80)
print("  PHASE 1: STANDALONE MODULE VALIDATION")
print("=" * 80)

# -- 1a. Tier 2: Earnings Calendar --
print("\n-- Tier 2: Earnings Calendar -----------------------------------------")
from data.earnings_calendar import EarningsCalendar

ec = EarningsCalendar(project_root)
t0 = time.perf_counter()
blackout_map = ec.build_backtest_blackout(
    symbols=sym_list,
    price_dates=common_idx,
    blackout_days=2,
)
t_ec = time.perf_counter() - t0

# Analyze what was returned
syms_with_data = set()
total_blackout_entries = 0
for dt, syms in blackout_map.items():
    syms_with_data.update(syms)
    total_blackout_entries += len(syms)

# Check which dates in the blackout_map actually align to trading days
aligned = 0
unaligned = 0
for dt in blackout_map:
    if dt in price_df.index:
        aligned += 1
    else:
        unaligned += 1

print(f"  Fetch time: {t_ec:.1f}s")
print(f"  Symbols with ANY earnings data: {len(syms_with_data)} / {n_syms}")
print(f"  Symbols with ZERO earnings:     {n_syms - len(syms_with_data)}")
print(f"  Total blackout-date entries:     {len(blackout_map)}")
print(f"  Total (date, symbol) pairs:      {total_blackout_entries}")
print(f"  Dates aligned to trading days:   {aligned}")
print(f"  Dates NOT on trading days:       {unaligned} (weekends/holidays, ignored)")

# Show sample blocked dates
if blackout_map:
    sample_dates = sorted(blackout_map.keys())[-10:]
    print(f"\n  Last 10 blackout dates:")
    for dt in sample_dates:
        syms = blackout_map[dt]
        sample = ', '.join(sorted(syms)[:8])
        suffix = '...' if len(syms) > 8 else ''
        print(f"    {dt.date()}: {sample}{suffix} ({len(syms)} symbols)")

# Convert to engine format and count coverage
sym_to_idx_map = {s: j for j, s in enumerate(sym_list)}
blackout_idx = {}
for dt, syms in blackout_map.items():
    if dt in price_df.index:
        idx = price_df.index.get_loc(dt)
        if isinstance(idx, (int, np.integer)):
            blackout_idx[int(idx)] = {sym_to_idx_map[s] for s in syms if s in sym_to_idx_map}

print(f"\n  Engine blackout_idx: {len(blackout_idx)} trading days with blackouts")
total_sym_blocks = sum(len(v) for v in blackout_idx.values())
print(f"  Total (day, symbol) blocks: {total_sym_blocks}")

# How many actual entry signals would be affected?
signal_arr = signal_df.values.astype(np.float64)
entry_threshold = config.get("backtest.entry_threshold", 1.0)
blocked_count = 0
total_entry_signals = 0
for i in range(1, n_days):
    for sym_idx in range(n_syms):
        sig = signal_arr[i, sym_idx]
        if np.isnan(sig) or abs(sig) <= entry_threshold:
            continue
        total_entry_signals += 1
        if i in blackout_idx and sym_idx in blackout_idx[i]:
            blocked_count += 1

print(f"\n  Entry signals that exceed threshold: {total_entry_signals}")
print(f"  Of those, in an earnings blackout:  {blocked_count}")
print(f"  Potential block rate:                {blocked_count / max(total_entry_signals, 1) * 100:.3f}%")


# -- 1b. Tier 1: Sentiment Proxy --
print("\n\n-- Tier 1: Sentiment Price-Action Proxy ------------------------------")
from data.news_sentiment import NewsSentiment

ns = NewsSentiment(
    project_root,
    penalty_floor=0.5,
    negative_threshold=-0.3,
)
t0 = time.perf_counter()
# IMPORTANT: pass price_df filtered to sym_list (mean-reverting subset)
# to match what the engine does (price_data[common_symbols])
sentiment_mult_df = ns.build_backtest_sentiment(
    price_df=price_df[sym_list],
    lookback_days=5,
    drop_threshold=-0.08,
)
t_ns = time.perf_counter() - t0
sentiment_arr = sentiment_mult_df.values.astype(np.float64)

print(f"  Build time: {t_ns:.2f}s")
print(f"  Shape: {sentiment_arr.shape} ({n_days} days x {n_syms} symbols)")
print(f"  Value range: [{sentiment_arr.min():.4f}, {sentiment_arr.max():.4f}]")
print(f"  Mean multiplier: {sentiment_arr.mean():.6f}")

# Distribution of penalties
total_cells = sentiment_arr.size
penalized = int((sentiment_arr < 1.0).sum())
heavily_penalized = int((sentiment_arr < 0.75).sum())
at_floor = int((sentiment_arr <= 0.501).sum())
print(f"\n  Multiplier distribution:")
print(f"    = 1.0 (no penalty):  {total_cells - penalized:>10d} ({(total_cells - penalized) / total_cells * 100:.2f}%)")
print(f"    < 1.0 (penalized):   {penalized:>10d} ({penalized / total_cells * 100:.4f}%)")
print(f"    < 0.75 (heavy):      {heavily_penalized:>10d} ({heavily_penalized / total_cells * 100:.4f}%)")
print(f"    <= 0.50 (at floor):  {at_floor:>10d} ({at_floor / total_cells * 100:.4f}%)")

# Per-symbol penalty frequency
penalty_per_sym = (sentiment_arr < 1.0).sum(axis=0)
sym_has_penalty = int((penalty_per_sym > 0).sum())
print(f"\n  Symbols ever penalized: {sym_has_penalty} / {n_syms}")

# Top 10 most penalized symbols
top_penalized_idx = np.argsort(penalty_per_sym)[::-1][:10]
if penalty_per_sym[top_penalized_idx[0]] > 0:
    print(f"  Top 10 most penalized symbols:")
    for idx in top_penalized_idx:
        if penalty_per_sym[idx] == 0:
            break
        sym = sym_list[idx]
        days_penalized = penalty_per_sym[idx]
        penalized_vals = sentiment_arr[:, idx][sentiment_arr[:, idx] < 1.0]
        avg_mult = penalized_vals.mean() if len(penalized_vals) > 0 else 1.0
        print(f"    {sym:8s}: {days_penalized:>4d} days penalized, avg mult={avg_mult:.3f}")

# Check: how many of those penalized cells overlap with actual entry signals?
penalized_at_entry = 0
for i in range(1, n_days):
    for sym_idx in range(n_syms):
        sig = signal_arr[i, sym_idx]
        if np.isnan(sig) or abs(sig) <= entry_threshold:
            continue
        if sentiment_arr[i, sym_idx] < 1.0:
            penalized_at_entry += 1

print(f"\n  Entry signals with a sentiment penalty: {penalized_at_entry} / {total_entry_signals}")
print(f"  Penalty rate at entry:                  {penalized_at_entry / max(total_entry_signals, 1) * 100:.3f}%")

# Validate the proxy: check rolling returns
price_returns = price_df.pct_change(periods=5)
deeply_negative = int((price_returns < -0.08).sum().sum())
print(f"\n  Underlying data check:")
print(f"    5-day returns < -8%: {deeply_negative} (symbol-day cells)")
print(f"    This should roughly correspond to penalized cells: {penalized}")


# ========================================================================================
#  PHASE 2: ENGINE A/B WITH DIAGNOSTICS
# ========================================================================================

print("\n\n")
print("=" * 80)
print("  PHASE 2: ENGINE A/B COMPARISON + DIAGNOSTICS")
print("=" * 80)

def run_variant(label, bt_cfg):
    print(f"\n{'_'*60}")
    print(f"  {label}")
    print(f"{'_'*60}")
    t0 = time.perf_counter()
    engine = BacktestEngine(bt_cfg)
    res = engine.run_backtest(price_df, signal_df, volume_df, exit_signal_data=zscore_df)
    elapsed = time.perf_counter() - t0
    print(f"  {res.total_trades} trades | Sharpe {res.sharpe_ratio:.3f} | {elapsed:.1f}s")
    return res

# -- Baseline --
base_cfg = config.to_backtest_config()
base_dict = asdict(base_cfg)
base_dict["earnings_blackout_enabled"] = False
base_dict["sentiment_penalty_enabled"] = False
baseline_cfg = BacktestConfig(**base_dict)
baseline = run_variant("BASELINE (all filters OFF)", baseline_cfg)

# -- Tier 2 Only --
t2_dict = asdict(base_cfg)
t2_dict["earnings_blackout_enabled"] = True
t2_dict["earnings_blackout_days"] = 2
t2_dict["sentiment_penalty_enabled"] = False
tier2_cfg = BacktestConfig(**t2_dict)
tier2_only = run_variant("TIER 2 ONLY (Earnings Blackout)", tier2_cfg)

# -- Tier 1 Only --
t1_dict = asdict(base_cfg)
t1_dict["earnings_blackout_enabled"] = False
t1_dict["sentiment_penalty_enabled"] = True
t1_dict["sentiment_penalty_floor"] = 0.5
t1_dict["sentiment_negative_threshold"] = -0.3
t1_dict["sentiment_proxy_lookback"] = 5
t1_dict["sentiment_proxy_drop_threshold"] = -0.08
tier1_cfg = BacktestConfig(**t1_dict)
tier1_only = run_variant("TIER 1 ONLY (Sentiment Penalty)", tier1_cfg)

# -- Both --
both_dict = asdict(base_cfg)
both_dict["earnings_blackout_enabled"] = True
both_dict["earnings_blackout_days"] = 2
both_dict["sentiment_penalty_enabled"] = True
both_dict["sentiment_penalty_floor"] = 0.5
both_dict["sentiment_negative_threshold"] = -0.3
both_dict["sentiment_proxy_lookback"] = 5
both_dict["sentiment_proxy_drop_threshold"] = -0.08
both_cfg = BacktestConfig(**both_dict)
both = run_variant("BOTH TIERS (Earnings + Sentiment)", both_cfg)


# -- Summary Table --
print("\n\n" + "=" * 100)
print("  4-WAY COMPARISON")
print("=" * 100)
header = f"{'Metric':25s} {'Baseline':>16s} {'T2 (Earn)':>16s} {'T1 (Sent)':>16s} {'Both':>16s}"
print(header)
print("-" * 100)

def row(label, vals, pct=False, plain=False):
    parts = []
    for v in vals:
        if plain:
            parts.append(f"{v}")
        elif pct:
            parts.append(f"{v * 100:.2f}%")
        else:
            parts.append(f"{v:.3f}")
    print(f"{label:25s} " + " ".join(f"{p:>16s}" for p in parts))

row("Total Trades", [baseline.total_trades, tier2_only.total_trades, tier1_only.total_trades, both.total_trades], plain=True)
row("Sharpe Ratio", [baseline.sharpe_ratio, tier2_only.sharpe_ratio, tier1_only.sharpe_ratio, both.sharpe_ratio])
row("Annualized Return", [baseline.annualized_return, tier2_only.annualized_return, tier1_only.annualized_return, both.annualized_return], pct=True)
row("Win Rate", [baseline.win_rate, tier2_only.win_rate, tier1_only.win_rate, both.win_rate], pct=True)
row("Max Drawdown", [baseline.max_drawdown, tier2_only.max_drawdown, tier1_only.max_drawdown, both.max_drawdown], pct=True)
row("Profit Factor", [baseline.profit_factor, tier2_only.profit_factor, tier1_only.profit_factor, both.profit_factor])
row("Avg Exposure", [baseline.avg_exposure, tier2_only.avg_exposure, tier1_only.avg_exposure, both.avg_exposure], pct=True)


# -- Diagnostics Dump --
print("\n\n" + "=" * 80)
print("  FILTER DIAGNOSTICS (from engine counters)")
print("=" * 80)

for label, res in [("T2 Only", tier2_only), ("T1 Only", tier1_only), ("Both", both)]:
    d = res.filter_diagnostics
    print(f"\n-- {label} --")

    if d.get("tier2_enabled"):
        print(f"  Tier 2 (Earnings Blackout):")
        print(f"    Symbols with earnings data:  {d['tier2_symbols_with_earnings']}")
        print(f"    Total (date,sym) blackouts:  {d['tier2_blackout_entries_total']}")
        print(f"    Trading days with blackouts: {d['tier2_blackout_day_count']}")
        print(f"    Entries actually BLOCKED:     {d['tier2_entries_blocked']}")
        if d["tier2_block_details"]:
            print(f"    Sample blocked entries (first 20):")
            for day_idx, sym, date_str in d["tier2_block_details"][:20]:
                print(f"      {date_str} | {sym:8s} (day {day_idx})")
    else:
        print(f"  Tier 2: DISABLED")

    if d.get("tier1_enabled"):
        print(f"  Tier 1 (Sentiment Penalty):")
        print(f"    Total cells:                 {d['tier1_cells_total']}")
        print(f"    Cells penalized (< 1.0):     {d['tier1_cells_penalized']} ({d['tier1_pct_penalized']:.4f}%)")
        print(f"    Min multiplier:              {d['tier1_min_multiplier']:.4f}")
        print(f"    Mean multiplier (all cells): {d['tier1_mean_multiplier']:.6f}")
        print(f"    Entries considered:           {d['tier1_total_entries']}")
        print(f"    Entries actually PENALIZED:   {d['tier1_entries_penalized']}")
        if d['tier1_total_entries'] > 0:
            penalty_rate = d['tier1_entries_penalized'] / d['tier1_total_entries'] * 100
            print(f"    Entry penalty rate:           {penalty_rate:.3f}%")
        if d["tier1_penalty_details"]:
            print(f"    Sample penalized entries (first 20):")
            for day_idx, sym, mult in d["tier1_penalty_details"][:20]:
                print(f"      Day {day_idx:>5d} | {sym:8s} | multiplier={mult}")
    else:
        print(f"  Tier 1: DISABLED")


# ========================================================================================
#  PHASE 3: TRADE-LEVEL DIFF (Baseline vs Both)
# ========================================================================================

print("\n\n" + "=" * 80)
print("  PHASE 3: TRADE-LEVEL DIFF (Baseline vs Both)")
print("=" * 80)

# Build trade keys for comparison: (entry_date, symbol, side) -> trade
base_trades = {(t.entry_date, t.symbol, t.side): t for t in baseline.trades}
both_trades = {(t.entry_date, t.symbol, t.side): t for t in both.trades}

only_in_baseline = set(base_trades.keys()) - set(both_trades.keys())
only_in_both = set(both_trades.keys()) - set(base_trades.keys())
in_both_keys = set(base_trades.keys()) & set(both_trades.keys())

print(f"\n  Trades only in Baseline:  {len(only_in_baseline)}")
print(f"  Trades only in Filtered:  {len(only_in_both)}")
print(f"  Trades in both (shared):  {len(in_both_keys)}")

# Show removed trades (likely blocked by earnings)
if only_in_baseline:
    print(f"\n  Trades REMOVED by filters (top 30):")
    removed = sorted(only_in_baseline, key=lambda k: k[0])[:30]
    for entry_dt, sym, side in removed:
        t = base_trades[(entry_dt, sym, side)]
        print(f"    {entry_dt.date()} | {sym:8s} {side:5s} | sig={t.entry_signal:+6.2f} | pnl={t.pnl_pct*100:+6.2f}% | exit={t.exit_reason}")

# Show added trades (position freed up room)
if only_in_both:
    print(f"\n  Trades ADDED by filters (top 30):")
    added = sorted(only_in_both, key=lambda k: k[0])[:30]
    for entry_dt, sym, side in added:
        t = both_trades[(entry_dt, sym, side)]
        print(f"    {entry_dt.date()} | {sym:8s} {side:5s} | sig={t.entry_signal:+6.2f} | pnl={t.pnl_pct*100:+6.2f}% | exit={t.exit_reason}")

# Show trades with different PnL (sentiment penalty -> different position size -> different timing)
size_changed = 0
for key in in_both_keys:
    bt = base_trades[key]
    ft = both_trades[key]
    if abs(bt.shares - ft.shares) > 0.01:
        size_changed += 1

print(f"\n  Shared trades with different position sizes: {size_changed}")
if size_changed > 0:
    print(f"  Sample size changes (first 20):")
    count = 0
    for key in sorted(in_both_keys, key=lambda k: k[0]):
        bt = base_trades[key]
        ft = both_trades[key]
        if abs(bt.shares - ft.shares) > 0.01:
            delta_pct = (ft.shares - bt.shares) / bt.shares * 100 if bt.shares > 0 else 0
            print(f"    {bt.entry_date.date()} | {bt.symbol:8s} | base={bt.shares:.1f} -> filt={ft.shares:.1f} ({delta_pct:+.1f}%)")
            count += 1
            if count >= 20:
                break


# ========================================================================================
#  PHASE 4: VERDICT
# ========================================================================================

print("\n\n" + "=" * 80)
print("  VERDICT")
print("=" * 80)

d_both = both.filter_diagnostics
t2_blocked = d_both.get("tier2_entries_blocked", 0)
t1_penalized = d_both.get("tier1_entries_penalized", 0)
t1_rate = d_both.get("tier1_pct_penalized", 0)

print(f"\n  Tier 2 entries blocked: {t2_blocked}")
print(f"  Tier 1 entries penalized: {t1_penalized}")
print(f"  Tier 1 cells with penalty: {t1_rate:.4f}% of all (day x symbol) cells")

if t2_blocked == 0 and t1_penalized == 0:
    print("\n  WARNING: BOTH FILTERS ARE INACTIVE -- zero impact on entries")
    print("     This means the pre-computed data has no overlap with actual entry signals.")
    print("     Check: are earnings dates covering the right time period?")
    print("     Check: is the -8% drop threshold too aggressive? Try -5%.")
elif t2_blocked < 5 and t1_penalized < 10:
    print("\n  WARNING: FILTERS HAVE MINIMAL IMPACT -- might need tuning")
    print("     Consider relaxing thresholds:")
    print("     - Tier 2: blackout_days=3 instead of 2")
    print("     - Tier 1: drop_threshold=-0.05 instead of -0.08")
else:
    print("\n  PASS: Filters are active and impacting entries as expected")

print("\nDone.")
