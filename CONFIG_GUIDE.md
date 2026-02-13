# Configuration Guide

All strategy parameters are centralized in `config.yaml` for easy experimentation.

## Quick Start

1. **Edit parameters** in `config.yaml`
2. **Restart kernel** in Jupyter notebook
3. **Re-run all cells** to apply changes

No code changes needed!

## Key Configuration Sections

### Data Loading
```yaml
data:
  max_symbols: null  # null = all symbols, or set to 50 for faster testing
  min_history: 100   # Minimum data points required
```

### Signal Generation
```yaml
signals:
  zscore:
    min_lookback: 10
    max_lookback: 252
    default_lookback: 20

  hurst:
    threshold: 0.5  # Only trade stocks with H < 0.5

  bollinger:
    std_multiplier: 2.0
    volume_multiplier: 1.5

  rsi:
    period: 14
    overbought: 70.0
    oversold: 30.0

  composite_weights:
    bollinger: 0.25
    rsi_divergence: 0.25
    rsi_level: 0.25
```

### Backtesting
```yaml
backtest:
  initial_capital: 1000000.0
  commission_pct: 0.001  # 0.1%
  entry_threshold: 2.0   # Z-score threshold
  exit_threshold: 0.5
  stop_loss_pct: 0.05    # null = disabled, 0.05 = 5%
  max_holding_days: null # null = hold until signal
```

### Optimization
```yaml
optimization:
  method: "grid"  # or "bayesian"
  objective_metric: "sharpe_ratio"

  param_ranges:
    entry_threshold: [1.5, 2.0, 2.5, 3.0]
    exit_threshold: [0.3, 0.5, 0.7]
```

## Common Experiments

### Experiment 1: Faster Mean Reversion
Make signals more sensitive to quick reversals:
```yaml
signals:
  zscore:
    default_lookback: 10  # Shorter window

backtest:
  entry_threshold: 1.5    # Enter earlier
  max_holding_days: 5     # Exit faster
```

### Experiment 2: Aggressive Risk Management
Add stop-loss and take-profit:
```yaml
backtest:
  stop_loss_pct: 0.05     # 5% stop loss
  take_profit_pct: 0.10   # 10% take profit
```

### Experiment 3: Only Trade Strong Signals
Filter to higher-quality setups:
```yaml
signals:
  hurst:
    threshold: 0.4        # Stricter mean reversion filter

  composite_weights:
    bollinger: 0.5        # Weight volume confirmation more
    rsi_divergence: 0.3
    rsi_level: 0.2

backtest:
  entry_threshold: 2.5    # Higher threshold
```

### Experiment 4: Test Subset
Quick iteration during development:
```yaml
data:
  max_symbols: 20         # Only load 20 stocks

optimization:
  train_period_days: 126  # Shorter training window
  test_period_days: 63    # Shorter test period
```

## What to Change for Better Performance

### If you see 0 trades:
- Lower `backtest.entry_threshold` (try 1.5)
- Check `signals.hurst.threshold` isn't filtering all stocks

### If Sharpe < 0.5:
- Increase `signals.composite_weights` for best-performing signal
- Add `stop_loss_pct` to cut losses
- Increase `entry_threshold` to trade less frequently but higher quality

### If win rate < 40%:
- Increase `entry_threshold` (stricter entry)
- Add confirmation: increase `signals.bollinger.volume_multiplier`
- Lower `signals.hurst.threshold` to 0.4 (only very mean-reverting stocks)

### If max drawdown > 30%:
- Add `stop_loss_pct: 0.10`
- Reduce `backtest.max_position_size` from 0.1 to 0.05
- Enable `use_regime_filter: true` to sit out volatile periods

## Advanced: Bayesian Optimization

For automatic parameter tuning:
```yaml
optimization:
  method: "bayesian"
  n_trials: 200  # More trials = better parameters

  bayesian_ranges:
    entry_threshold: [1.0, 4.0]
    exit_threshold: [0.1, 1.5]
```

This will search the continuous space instead of discrete grid points.

## Saving Experiments

Create experiment configs:
1. Copy `config.yaml` to `config_experiment1.yaml`
2. Modify parameters
3. Load in notebook: `config = ConfigLoader(Path('config_experiment1.yaml'))`

Compare results across experiments!
