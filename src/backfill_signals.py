#!/usr/bin/env python3
"""
Backfill: Show what signals WOULD have been generated on missed days.

The stale cache bug (fixed in commit 679c1a3) meant signals were frozen on
Feb 27 data for Feb 28, Mar 2, Mar 3.  Now that we have fresh data through
Mar 4, the full signal_df covers all historical dates.  This script slices
the signal_df to each missed date and reports what the algorithm would have
produced.

Usage:
    python src/backfill_signals.py                   # Uses default missed dates
    python src/backfill_signals.py 2026-02-28 2026-03-02 2026-03-03
"""

import sys
import json
from pathlib import Path
from datetime import date

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "data" / "snapshots" / "signal_cache" / "live"


def load_signal_cache():
    """Load the current signal cache (parquet files)."""
    signal_df = pd.read_parquet(CACHE_DIR / "signal_df.parquet")
    zscore_df = pd.read_parquet(CACHE_DIR / "zscore_df.parquet")
    price_df = pd.read_parquet(CACHE_DIR / "price_df.parquet")

    with open(CACHE_DIR / "metadata.json") as f:
        metadata = json.load(f)

    print(f"Loaded signal cache: {metadata['n_symbols']} symbols, "
          f"{metadata['n_days']} days "
          f"({signal_df.index[0].date()} → {signal_df.index[-1].date()})")
    return signal_df, zscore_df, price_df, metadata


def backfill_date(target_date, signal_df, zscore_df, price_df, entry_threshold=1.5):
    """Show what signals would have been generated for a given date."""
    ts = pd.Timestamp(target_date)

    if ts not in signal_df.index:
        # Try nearest trading day
        available = signal_df.index[signal_df.index <= ts]
        if len(available) == 0:
            print(f"\n  {target_date}: No data available (before cache start)")
            return None
        ts = available[-1]
        if ts.date() != pd.Timestamp(target_date).date():
            print(f"\n  {target_date}: Not a trading day — nearest: {ts.date()}")
            return None

    today_signals = signal_df.loc[ts].dropna()
    today_zscores = zscore_df.loc[ts].dropna() if ts in zscore_df.index else pd.Series(dtype=float)
    today_prices = price_df.loc[ts].dropna() if ts in price_df.index else pd.Series(dtype=float)

    valid_count = len(today_signals)
    actionable = today_signals[today_signals.abs() > entry_threshold].sort_values()

    print(f"\n  {'='*60}")
    print(f"  BACKFILL: {ts.date()}  (what would have been generated)")
    print(f"  {'='*60}")
    print(f"  Valid signals: {valid_count} | Actionable: {len(actionable)} (threshold: {entry_threshold})")

    if len(actionable) > 0:
        longs = actionable[actionable < 0].sort_values()
        shorts = actionable[actionable > 0].sort_values(ascending=False)

        if len(longs) > 0:
            print(f"  LONG candidates ({len(longs)}):")
            for sym, sig in longs.items():
                z = today_zscores.get(sym, float('nan'))
                px = today_prices.get(sym, float('nan'))
                print(f"    BUY  {sym:<6s} | signal={sig:+.3f} | z-score={z:+.2f} | price=${px:,.2f}")

        if len(shorts) > 0:
            print(f"  SHORT candidates ({len(shorts)}):")
            for sym, sig in shorts.items():
                z = today_zscores.get(sym, float('nan'))
                px = today_prices.get(sym, float('nan'))
                print(f"    SELL {sym:<6s} | signal={sig:+.3f} | z-score={z:+.2f} | price=${px:,.2f}")
    else:
        print(f"  No actionable signals (all below threshold)")

    # Top 5 near threshold
    non_actionable = today_signals[
        (today_signals.abs() <= entry_threshold) & (~today_signals.isna())
    ]
    if len(non_actionable) > 0:
        ranked = non_actionable.reindex(
            non_actionable.abs().sort_values(ascending=False).index
        )
        top_n = ranked.head(5)
        print(f"  Top {len(top_n)} nearest to threshold:")
        for sym, sig in top_n.items():
            pct = abs(sig) / entry_threshold * 100
            direction = "BUY " if sig < 0 else "SELL"
            z = today_zscores.get(sym, float('nan'))
            px = today_prices.get(sym, float('nan'))
            print(f"    {direction} {sym:<6s} | signal={sig:+.3f} | z-score={z:+.2f} | price=${px:,.2f} | ({pct:.0f}%)")

    print(f"  {'='*60}")

    return {
        "date": str(ts.date()),
        "valid": valid_count,
        "actionable": len(actionable),
        "signals": {
            sym: {"signal": float(sig), "zscore": float(today_zscores.get(sym, 0))}
            for sym, sig in actionable.items()
        },
    }


def main():
    # Default: the 3 missed trading days
    if len(sys.argv) > 1:
        dates = sys.argv[1:]
    else:
        dates = ["2026-02-28", "2026-03-02", "2026-03-03"]

    signal_df, zscore_df, price_df, metadata = load_signal_cache()

    print(f"\nBackfilling signals for {len(dates)} missed day(s)...\n")

    results = []
    for d in dates:
        r = backfill_date(d, signal_df, zscore_df, price_df)
        if r:
            results.append(r)

    # Summary comparison
    print(f"\n  {'─'*60}")
    print(f"  SUMMARY")
    print(f"  {'─'*60}")
    for r in results:
        print(f"  {r['date']}: {r['actionable']} actionable out of {r['valid']} valid")
    print(f"  {'─'*60}")


if __name__ == "__main__":
    main()
