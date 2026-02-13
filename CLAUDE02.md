# Quantitative Trading System - Codespaces Environment Context

**Environment**: GitHub Codespaces
**Claude Instance**: CLAUDE02 (Codespaces)
**Paired With**: CLAUDE01.md (Local Environment)

---

## Session Summary - 2026-02-13 (Phase 2: Mean Reversion Engine — COMPLETE)

### Overview

Phase 2 built a complete mean reversion trading system from scratch, then went through extensive debugging and optimization. The system is now functional with correct accounting, centralized YAML configuration, and data-backed optimal parameters.

### Critical Bugs Found & Fixed

During development, several serious accounting bugs were discovered and resolved. **If you're reviewing the code, these are already fixed in the current codebase.**

#### Bug 1: Inverted Short P&L (engine.py)
- **Problem**: Short trade PnL multiplied by negative shares, flipping profits into losses
- **Fix**: Uses `abs(shares)` for all PnL calculations
- **Impact**: Was the single biggest source of fake losses

#### Bug 2: Short Entry Cash Flow (engine.py)
- **Problem**: Short entries subtracted cash (like longs). Shorting should ADD cash (you receive money from selling borrowed shares).
- **Fix**: Longs: `cash -= entry_value`, Shorts: `cash += entry_value`

#### Bug 3: Exit Cash Double-Counting (engine.py)
- **Problem**: Exit added `exit_value + net_pnl`, counting the price change twice
- **Fix**: Separate exit logic for longs (receive exit price) and shorts (pay exit price)

#### Bug 4: Leverage Spiral (engine.py)
- **Problem**: Position sizing used raw CASH instead of EQUITY. Short sales inflate cash → bigger positions → more shorts → exponential exposure. Commission reached $1.88M on $100K capital.
- **Fix**: Position sizing now uses `current_equity = current_cash + (positions * prices).sum()`
- **Also added**: `max_total_exposure` enforcement before opening new positions

#### Bug 5: Signal Normalization Mismatch (mean_reversion.py)
- **Problem**: Composite signal was normalized to [-1, +1] but entry threshold was 2.0. Signal could NEVER cross the threshold → 0 trades every time.
- **Fix**: Raw z-score is now the primary signal (naturally ranges -3 to +6). Bollinger/RSI/divergence signals act as confirmation multipliers, not averaged components.

#### Bug 6: Signal Directional Bias (mean_reversion.py)
- **Problem**: Confirmation formula `zscore * (1 + confirmation)` amplified positive signals more than negative ones. In bull markets, created 3:1 short-to-long imbalance.
- **Fix**: `agreement = np.sign(zscore) * signal` — confirmations now boost magnitude symmetrically in both directions.

### Current Verified Performance (Config A — Optimal Parameters)

Tested on full 285-stock universe, 2 years of daily data:
```
Config: entry=3.0, exit=0.5, no SL, no TP, max_hold=10 days
Result: +27.2% return, Sharpe 0.82, 53% WR, 15.8% max DD, 698 trades
```

Long trades: +0.60% avg, 56.2% WR (stronger)
Short trades: -0.26% avg, 48.6% WR (weaker — bull market drag)

### Parameter Sweep Results (216 Combinations Tested)

Best found configurations ranked by Sharpe:
```
#1  entry=3.0 exit=0.5 SL=None TP=None hold=10  → +27.2% Sharpe=0.82 WR=53% DD=15.8%
#2  entry=3.0 exit=0.3 SL=None TP=0.10 hold=20  → +23.3% Sharpe=0.71 WR=54% DD=11.3%
#3  entry=3.0 exit=0.5 SL=None TP=None hold=15  → +22.5% Sharpe=0.69 WR=52% DD=14.8%
```

Parameter sensitivity (avg Sharpe by value):
```
entry_threshold: 3.0 (+0.13) >> 2.5 (-0.08) >> 2.0 (-0.24) >> 1.5 (-0.70)
exit_threshold:  0.5 slightly best, but low sensitivity
stop_loss:       None best. 5% SL hurts (stops out recoverable trades). 10% neutral.
take_profit:     None slightly better — let winners run
max_holding:     20 days best (-0.12), 15 days (-0.36), 10 days middle
```

**Key insight: Higher entry threshold = fewer, higher-quality trades = better performance.**

### Configuration System

All hardcoded parameters have been centralized into `config.yaml`:

```yaml
# Key parameters (Config A — currently active)
backtest:
  initial_capital: 1000000.0
  entry_threshold: 3.0
  exit_threshold: 0.5
  stop_loss_pct: null
  take_profit_pct: null
  max_holding_days: 10
  commission_pct: 0.001
  slippage_pct: 0.0005

signals:
  hurst:
    threshold: 0.5
  bollinger:
    std_multiplier: 4.0
    volume_multiplier: 1.5
  composite_weights:
    bollinger: 0.25
    rsi_divergence: 0.25
    rsi_level: 0.25

optimization:
  method: "grid"  # or "bayesian"
  objective_metric: "sharpe_ratio"
```

**Workflow**: Edit `config.yaml` → restart notebook kernel → re-run all cells. No code changes needed.

### Files Created/Modified in Phase 2

#### New Files
| File | Lines | Purpose |
|------|-------|---------|
| `src/strategies/mean_reversion.py` | ~470 | Signal generation: adaptive z-score, Hurst filter, Bollinger+volume, RSI divergence, cross-sectional ranking, regime detection |
| `src/backtest/engine.py` | ~430 | Vectorized backtesting: trade tracking, position management, performance metrics (Sharpe, Sortino, Calmar, etc.) |
| `src/backtest/optimizer.py` | ~400 | Walk-forward optimization: grid search, Bayesian (Optuna), parameter stability analysis |
| `src/strategy_config.py` | ~230 | YAML config loader: converts config.yaml to dataclasses, dot-path access (`config.get('backtest.entry_threshold')`) |
| `config.yaml` | ~130 | All strategy parameters: signals, backtest, optimization, data, visualization |
| `CONFIG_GUIDE.md` | ~150 | Documentation: how to use config, experiment templates, troubleshooting |
| `src/main_mean_reversion.ipynb` | 28 cells | Interactive Phase 2 workflow notebook |

#### Modified Files
| File | Change |
|------|--------|
| `requirements.txt` | Added scipy, statsmodels, optuna, scikit-learn, pyyaml |
| `src/main_data_collector.ipynb` | Renamed from main.ipynb (Phase 1 data collection) |

### Architecture: How the Signal System Works

```
Raw Price Data (Parquet files)
    │
    ▼
┌─────────────────────────────────────────────────┐
│  UniverseAnalyzer                                │
│  - Hurst exponent (H < 0.5 = mean-reverting)    │
│  - OU half-life estimation                       │
│  - ADF stationarity test                         │
│  → Filters 285/293 stocks (97% pass — too many) │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│  MeanReversionSignals.generate_composite_signal  │
│                                                  │
│  PRIMARY: Raw Z-Score (adaptive lookback)        │
│     lookback = 2 × half_life (from OU process)   │
│     range: typically -3 to +5                    │
│                                                  │
│  CONFIRMATIONS (multiply z-score):               │
│     × (1 + agreement)                            │
│     agreement = sign(zscore) × indicator_signal  │
│     - Bollinger + volume (weight: 0.25)          │
│     - RSI divergence (weight: 0.25)              │
│     - RSI level (weight: 0.25)                   │
│                                                  │
│  OUTPUT: composite signal in z-score units       │
│     signal < -3.0 → strong BUY (long)            │
│     signal > +3.0 → strong SELL (short)          │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│  BacktestEngine                                  │
│  - Entry when |signal| > entry_threshold (3.0)   │
│  - Exit when |signal| < exit_threshold (0.5)     │
│  - Max holding: 10 days                          │
│  - Position size: 10% of equity per trade        │
│  - Total exposure capped at 100%                 │
│  - Commission: 0.1% + 0.05% slippage             │
│                                                  │
│  Cash model:                                     │
│    Long entry: cash -= shares × price            │
│    Short entry: cash += shares × price           │
│    Long exit: cash += shares × exit_price        │
│    Short exit: cash -= shares × exit_price       │
│    Equity = cash + Σ(positions × prices)         │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│  ParameterOptimizer (walk-forward)               │
│  - Train: 252 days, Test: 126 days, Step: 63    │
│  - Grid or Bayesian (Optuna TPE)                 │
│  - Objective: Sharpe ratio                       │
│  - No look-ahead bias                            │
└─────────────────────────────────────────────────┘
```

### Project Structure (Current)

```
Quant/
├── CLAUDE01.md                    # Context for local Claude
├── CLAUDE02.md                    # Context for Codespaces Claude (this file)
├── config.yaml                    # ← ALL strategy parameters
├── CONFIG_GUIDE.md                # How to use config system
├── requirements.txt               # Updated with new deps
├── data/
│   ├── historical/daily/          # 293 parquet files (2 years daily OHLCV)
│   ├── snapshots/                 # Options data
│   └── universe/                  # Index composition JSONs
├── src/
│   ├── config/
│   │   └── config.py              # IB connection config
│   ├── connection/
│   │   └── ib_connection.py       # IB Gateway integration
│   ├── data/
│   │   ├── collector.py           # Historical data collector
│   │   ├── universe_builder.py    # Index universe builder
│   │   └── options.py             # Options data collector
│   ├── strategies/
│   │   └── mean_reversion.py      # ← Signal generation engine
│   ├── backtest/
│   │   ├── engine.py              # ← Backtesting engine
│   │   └── optimizer.py           # ← Walk-forward optimizer
│   ├── strategy_config.py         # ← YAML config loader
│   ├── main_data_collector.ipynb  # Phase 1 workflow
│   └── main_mean_reversion.ipynb  # ← Phase 2 workflow
```

### Usage Example (Current Working Code)

```python
from strategy_config import ConfigLoader
from strategies.mean_reversion import MeanReversionSignals, UniverseAnalyzer
from backtest.engine import BacktestEngine
from backtest.optimizer import ParameterOptimizer

# Load config
config = ConfigLoader(Path('config.yaml'))

# Build strategy components from config
signal_config = config.to_signal_config()
bt_config = config.to_backtest_config()
opt_config = config.to_optimization_config()
weights = config.get_composite_weights()

# Analyze universe
analyzer = UniverseAnalyzer(signal_config)
analysis = analyzer.analyze_universe(price_data)
mean_reverting = analysis[analysis['is_mean_reverting']]['symbol'].tolist()

# Generate signals
signal_gen = MeanReversionSignals(signal_config)
composite, individual = signal_gen.generate_composite_signal(prices, volumes, weights=weights)

# Backtest
engine = BacktestEngine(bt_config)
results = engine.run_backtest(price_df, signal_df, volume_df)
print(results.summary())

# Optimize
optimizer = ParameterOptimizer(opt_config)
opt_results = optimizer.walk_forward_optimization(price_df, signal_generator_fn, volume_df)
```

### Known Issues & Areas for Improvement

1. **97% of stocks pass Hurst filter** — threshold 0.5 is not selective enough. Consider 0.4 for stricter filtering.
2. **Shorts underperform longs** — expected in bull market data. Consider long-only mode or market regime awareness.
3. **Composite signal has slight positive bias** — z-score spends more time positive in uptrending markets. Inherent to the approach, not a bug.
4. **Walk-forward optimizer is slow** — grid search over 216 combos × multiple periods takes minutes. Bayesian is faster for large search spaces.

### Dependencies

```
# Core
ib_insync==0.9.86
pandas>=2.0.0
numpy>=1.24.0
pyarrow>=14.0.0
pyyaml>=6.0.0

# Visualization
matplotlib>=3.7.0
plotly>=5.14.0

# Configuration
python-dotenv>=1.0.0
colorlog>=6.7.0

# Development
jupyter>=1.0.0
ipykernel>=6.25.0

# Statistical Analysis & Optimization
scipy>=1.10.0
statsmodels>=0.14.0
optuna>=3.0.0
scikit-learn>=1.3.0
```

---

## Next Steps: Phase 3 — ML Signal Filter

**Goal**: Use machine learning to filter mean reversion signals and reduce false positives.

### Planned Approach
1. **Feature engineering**: Price momentum, volume profile, volatility, microstructure features
2. **Binary classifier**: "Will this signal be profitable?" (yes/no)
3. **Model comparison**: Logistic Regression, Random Forest, XGBoost, LightGBM
4. **Walk-forward training**: Same methodology as optimizer (no look-ahead)
5. **Performance comparison**: Filtered vs unfiltered signals
6. **Expected improvement**: Higher Sharpe, lower max DD, better win rate, fewer trades

### Other Potential Improvements
- Stricter Hurst filtering (0.4 instead of 0.5)
- Long-only mode for bull market periods
- Market regime overlay (VIX-based or volatility ratio)
- Pairs trading / cross-sectional signals
- Intraday data for tighter mean reversion

---

## Session Summary - 2026-02-12 (Phase 1: Data Infrastructure)

### What We Built

Created a complete OOP-based quantitative trading system with Interactive Brokers integration.

### Completed Work

#### 1. Project Structure
- Clean modular architecture: `src/config/`, `src/connection/`, `src/strategies/`, `src/data/`, `src/backtest/`, `src/execution/`, `src/utils/`

#### 2. Core Classes (OOP Design)
- **Config Class** (`src/config/config.py`): Loads `.env`, validates settings, masks credentials
- **IBConnection Class** (`src/connection/ib_connection.py`): IB Gateway lifecycle, event-driven, context manager support

#### 3. Data Infrastructure
- **DataCollector** (`src/data/collector.py`): Historical OHLCV data collection
- **UniverseBuilder** (`src/data/universe_builder.py`): Index composition management
- **OptionsCollector** (`src/data/options.py`): Options chain snapshots
- Collected: 293 stocks × 2 years daily data, 4 ETF options chains

#### 4. Notebook Workflow
- `src/main_data_collector.ipynb` — Phase 1 data collection interface
- Auto-reload enabled for rapid development

### Codespaces Limitation

**Cannot connect to IB Gateway from Codespaces** — gateway runs on local PC, code runs in cloud. Solution: develop in Codespaces, trade locally.

---

## Git Workflow

```bash
# Codespaces → Local
git add . && git commit -m "message" && git push origin main
# Then on local: git pull origin main

# Local → Codespaces
git add . && git commit -m "message" && git push origin main
# Then in Codespaces: git pull origin main
```

---

**Remember**:
- CLAUDE02 (Codespaces) = Development & Strategy Building
- CLAUDE01 (Local) = Live Trading & IB Connection
- All parameters in `config.yaml` — no hardcoded values in code
- Keep both .md files updated to maintain context sync!
