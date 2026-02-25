# Risk Analytics Review & Improvement Recommendations

**Generated from:** REPLAY mode analytics (3B.0 – 3B.9)
**Dataset:** 2,110 trades across 503 trading days (~Feb 2024 – Feb 2026)
**Starting Capital:** $100,000 → Final: $804,862 (706% return)

---

## CRITICAL ISSUES (Fix Before Live Trading)

### 1. Backtest Engine Capital-Compounding Bug
**Finding:** Backtest reports 9,935,413,689% return ($9.9 trillion final equity) vs Replay's 706%. Parity test fails 4/5 checks.

**Root Cause:** The `BacktestEngine` likely doesn't properly cap total exposure when compounding. As equity grows, position sizes grow unbounded (15% of $9.9T = $1.5T per position), which is physically impossible. The replay engine with fixed Alpaca-style ordering is more realistic.

**Action:**
- Add a `max_gross_notional` hard cap to `engine.py` (e.g., $500k max portfolio value)
- OR: Change position sizing to use `min(equity, initial_capital * N)` as the sizing base
- The backtest results should NOT be trusted for absolute return figures

### 2. Commission Not Applied in Replay
**Finding:** Turnover report shows "Total commission+slippage: $0" and "Cost drag: 0.00%".

**Impact:** At 0.15% per side and 2,110 round-trip trades with $226M total traded value:
- Commission cost ≈ `$226M × 0.15% × 2 = ~$678K` (nearly the entire P&L!)
- This would reduce net P&L from $819K to ~$141K

**Action:**
- Verify `simulation.py` is deducting commission on both entry and exit
- If using Alpaca replay, commission should be $0 (Alpaca is commission-free) — but then the config's `commission_pct: 0.001` is misleading in the analytics display
- Update the turnover report to clarify "Alpaca commission = $0" vs "Simulated commission"

### 3. Slippage Sensitivity Cliff
**Finding:** Strategy is extremely sensitive to execution costs:

| Slippage | Annual Return | Total Return |
|----------|--------------|-------------|
| 0.00% | ~165% | ~819% |
| 0.05% | 141% | 480% |
| 0.10% | 116% | 367% |
| 0.20% | 55% | 141% |
| 0.50% | -100% | -538% |

**Root Cause:** 313x annual turnover (1,057 trades/yr). Each trade sees slippage twice (entry + exit). At 0.20% per side, that's 0.40% round-trip × 1,057 = ~423% annual cost drag.

**Action:**
- **Increase slippage assumption to 0.10%** for conservative modeling (still shows 116% annual)
- Add a **turnover cost budget**: if estimated annual cost > 50% of gross P&L, raise `entry_threshold` to reduce trades
- Consider switching `execution_price: "vwap"` validation to confirm VWAP is actually being used (reduces real slippage)
- Target reducing turnover below 200x/year by raising `entry_threshold` from 1.0 to ~1.3

---

## HIGH PRIORITY (Improve Model Robustness)

### 4. Data Quality: 67% of Symbols Flagged
**Finding:**
- 194 symbols with **price jumps >20%** (879 total occurrences)
- 17 symbols with **zero-volume days** (390 total)
- 5 symbols with **sub-dollar prices** (4,686 total)
- NVDA: 2,579 sub-dollar prices (min $0.15) — this is pre-split data
- PATK: triple-flagged (price jumps + zero volume + sub-dollar)
- PECO: 7% of all trading days have zero volume

**Action:**
- **Add a minimum price filter** to `engine.py` / `simulation.py`: skip entries where `close < $5`
- **Add a minimum volume filter**: skip entries where `volume < 50,000` shares
- **PATK, PECO, IESC** should be on a watchlist or excluded entirely due to illiquidity
- The "sub-dollar prices" for NVDA/NFLX are historical pre-split values — this is normal for adjusted data but the audit should recognize split-adjusted prices and exclude them from sub-dollar flags
- Add a `data_quality_filters` section to `config.yaml`:
  ```yaml
  data_quality:
    min_price: 5.0
    min_avg_volume: 50000
    max_price_jump_pct: 0.50
    exclude_symbols: ["PATK", "PECO"]
  ```

### 5. Losing Symbols Destroying Alpha
**Finding:**
- LUMN: -$32,188 (29 trades, 48% WR) — persistent loser
- HELE: -$17,649 (6 trades, 33% WR) — very low WR
- CRS: -$11,880 (8 trades, 50% WR)
- SANM: -$11,245 (7 trades, 43% WR)
- ANDE: -$5,100 (22 trades, 59% WR — decent WR but still net negative)

**Impact:** These 5 symbols lost ~$78K, or 9.5% of total P&L

**Action:**
- Add a **per-symbol performance tracker** that pauses trading a symbol after N consecutive losses or after cumulative loss exceeds a threshold
- Consider a **rolling blacklist**: if a symbol's trailing-50-trade WR < 45%, skip it for 30 days
- Short-term: add `LUMN, HELE` to the `exclude_symbols` list since they're structurally problematic for mean reversion

### 6. Dow 30 Underperformance
**Finding:** Dow 30 stocks: 57.3% WR, +0.14% avg P&L (barely breakeven). Only $15K total P&L on 96 trades.

**Root Cause:** Mega-cap stocks are efficiently priced. Mean reversion signals in Dow 30 names are noise, not alpha.

**Action:**
- Consider reducing weight or excluding Dow 30 stocks from the universe
- OR: apply a stricter `entry_threshold` (e.g., 1.5) for Dow 30 symbols vs 1.0 for small/mid-caps
- The "Other" category (non-index stocks) generates the most alpha — lean into this

### 7. Fat Tails (Kurtosis = 15.2)
**Finding:** Return distribution has extreme kurtosis (normal = 3). Best day: +12.26%, worst day: -4.27% (4.3x daily std dev).

**Impact:** VaR/CVaR models assuming normality will underestimate tail risk. A 3-sigma event happens more frequently than expected.

**Action:**
- Use **CVaR (Expected Shortfall)** instead of VaR for risk budgeting — it's already computed (CVaR 95% = -2.09%)
- Set the **circuit breaker** based on CVaR 99% (-3.34% daily):
  - Current `circuit_breaker_pct: 0.08` is reasonable (~2.4× CVaR 99%)
- Consider adding a **daily P&L kill switch**: if intraday loss exceeds 2× CVaR 95% (-4.19%), close all positions and halt for the day
- The positive skewness (1.70) is good — gains are larger than losses on average

---

## MEDIUM PRIORITY (Optimization Opportunities)

### 8. Signal Saturation
**Finding:** 23 signals above threshold per day (max 94). Entry threshold = 1.0 in gated mode.

**Impact:** The model sees far more opportunities than it can take (max ~11 concurrent positions). It's ranking and selecting from these, but many marginal signals pass the bar.

**Action:**
- Raise `entry_threshold` from 1.0 to **1.2–1.5** to filter out weaker signals
- This will also reduce turnover (addressing Issue #3)
- Run a parameter sweep: plot trade count & Sharpe vs entry_threshold from 1.0 to 2.0

### 9. Holding Period May Be Too Short
**Finding:** Median holding = 1 day, mode = 1 day, max = 8 days (capped by `max_holding_days: 5`).

**Impact:** Many mean-reversion trades need 3-5 days to fully revert. Exiting at day 1 may leave money on the table. The PnL distribution shows larger gains at 2-3 day holdings.

**Action:**
- Experiment with `max_holding_days: 7` or `10` and compare Sharpe
- The current `exit_threshold: 0.50` may be exiting too early — try `0.30` to let winners run longer
- The time_decay_exit (`check_after_days: 10`) won't fire since max_holding is 5 — it's effectively disabled

### 10. Regime Edge Decay
**Finding:** Sharpe dropped from 6.42 (2024 Bull) to 4.63 (2025-present). Win rate dropped 64% → 61.7%.

**Impact:** 28% Sharpe degradation over ~1 year. Could indicate alpha decay, changing market microstructure, or increased competition.

**Action:**
- Monitor rolling 63-day Sharpe and alert when it drops below 2.0
- Consider reducing position sizing when the rolling Sharpe is declining (adaptive risk scaling)
- The strategy still has very strong absolute performance (4.63 Sharpe is excellent) — no urgent action needed

### 11. Capital Utilization Report Bug
**Finding:** Report shows "Max concurrent positions: 0" and "Avg exposure: 0.00%" but the distribution shows 10-11 positions most days.

**Action:** The replay analytics builds trades from `replay_results['trades_df']` which may not populate the same fields as the backtest's `BacktestResults`. The `capital_utilization()` method likely reads `self.results.trades` (a List[Trade]) which is empty for the replay-constructed analytics object. Fix the replay analytics construction to properly populate the `trades` field.

---

## SUMMARY: Priority Action Items

| # | Action | Impact | Files to Modify |
|---|--------|--------|----------------|
| 1 | Fix backtest capital bug | Trust backtest results | `engine.py` |
| 2 | Verify replay commission handling | Accurate P&L | `simulation.py` |
| 3 | Increase slippage to 0.10% | Conservative estimates | `config.yaml` |
| 4 | Add min_price ($5) & min_volume filters | Remove garbage data | `config.yaml`, `engine.py` |
| 5 | Exclude LUMN, HELE from universe | +$50K annual P&L | `config.yaml` |
| 6 | Raise entry_threshold to 1.2-1.5 | Reduce turnover 30-50% | `config.yaml` |
| 7 | Fix capital utilization for replay | Accurate diagnostics | `analytics.py` |
| 8 | Add daily CVaR kill switch | Tail risk protection | `config.yaml` |

### Suggested `config.yaml` Changes
```yaml
# CONSERVATIVE UPDATES (apply these):
backtest:
  slippage_pct: 0.0010      # was 0.0005 — more realistic
  entry_threshold: 1.3       # was 1.0 — reduce signal saturation
  max_holding_days: 7        # was 5 — let mean reversion complete

# NEW SECTION:
data_quality:
  min_price: 5.0
  min_avg_volume: 50000
  exclude_symbols: ["LUMN", "HELE", "PATK", "PECO"]
```
