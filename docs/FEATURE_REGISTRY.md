# Feature Registry

Cross-reference of trading features by execution mode.

| Feature | Historical (BacktestEngine) | Shadow (SimulationEngine) | Replay (AnimatedReplay) | Notes |
|---|:---:|:---:|:---:|---|
| **Entry Logic** |
| Composite signal gating | ✅ | ✅ | ✅ | Kalman + OU + RSI divergence |
| Entry threshold check | ✅ | ✅ | ✅ | `entry_threshold` from config |
| Short acceleration filter | ✅ | ✅ | ✅ | Blocks shorts after recent surge |
| **Exit Logic** |
| Signal mean-reversion exit | ✅ | ✅ | ✅ | Exit when z-score crosses `exit_threshold` |
| Stop loss | ✅ | ✅ | ✅ | Configurable via `stop_loss_pct` |
| Short stop loss (separate) | ✅ | ✅ | ✅ | `short_stop_loss_pct` |
| Trailing stop | ✅ | ✅ | ✅ | Activates after `trailing_stop_activation` profit |
| Time decay exit | ✅ | ✅ | ✅ | Exits flat trades after N days |
| Take profit | ✅ | ✅ | ✅ | Configurable via `take_profit_pct` |
| Max holding days | ✅ | ✅ | ✅ | Forced exit after N days |
| **Position Sizing** |
| Equal weight | ✅ | ✅ | ✅ | Fixed % per position |
| Signal-proportional | ✅ | ❌ | ❌ | Scales with signal strength |
| Volatility-scaled | ✅ | ❌ | ❌ | Targets per-position vol |
| Kelly criterion | ✅ | ❌ | ❌ | Fractional Kelly from rolling stats |
| **Execution Model** |
| Close price execution | ✅ | ✅ | ✅ | Default — uses daily close |
| VWAP execution | ✅ | ❌ | ❌ | `execution_price: vwap` in config |
| Option D order flow | ❌ | ✅ | ✅ | Market entry → poll fill → GTC stop |
| Alpaca paper trading | ❌ | ✅ | ✅ | Real API calls in paper mode |
| **Risk Filters** |
| Regime filter | ✅ | ✅ | ✅ | Blocks trades in adverse regimes |
| Earnings blackout (Tier 2) | ✅ | ❌ | ❌ | Blocks near earnings dates |
| Sentiment penalty (Tier 1) | ✅ | ❌ | ❌ | Price-drop proxy for sentiment |
| Max total exposure cap | ✅ | ✅ | ✅ | `max_total_exposure` config |
| **Transaction Costs** |
| Flat slippage model | ✅ | ✅ | ✅ | `slippage_pct` applied to each trade |
| Commission | ✅ | ✅ | ✅ | `commission_pct` per trade |
| **Analytics** |
| Equity curve | ✅ | ✅ | ✅ | Daily equity series |
| Trade log export | ✅ | ✅ | ✅ | CSV export of all trades |
| Performance metrics | ✅ | ✅ | ✅ | Sharpe, Sortino, Calmar, etc. |
| Capital utilization (3B.0) | ✅ | ❌ | ❌ | Post-backtest analytics |
| Risk metrics (3B.1) | ✅ | ❌ | ❌ | VaR, CVaR, tail ratio, etc. |
| Rolling analytics (3B.2) | ✅ | ❌ | ❌ | Edge decay detection |
| Trade analytics (3B.3) | ✅ | ❌ | ❌ | Trade clustering, heatmaps |
| Turnover & cost (3B.4) | ✅ | ❌ | ❌ | Break-even cost analysis |
| Regime analysis (3B.5) | ✅ | ❌ | ❌ | Per-regime Sharpe, drawdown |
| Sector/index analysis (3B.6) | ✅ | ❌ | ❌ | P&L by index membership |
| Slippage sensitivity (3B.7) | ✅ | ❌ | ❌ | Sharpe degradation chart |
| Trade validation (3B.8) | ✅ | ❌ | ❌ | Impossible trade detector |
| Data quality audit (3B.9) | ✅ | ❌ | ❌ | Price jump, zero-vol flags |
| **Visualization** |
| Animated replay charts | ❌ | ❌ | ✅ | Live Plotly equity + trade log |
| Backtest vs Replay overlay | ❌ | ❌ | ✅ | Normalized equity comparison |
| Fast mode (no animation) | ❌ | ❌ | ✅ | `fast_mode=True` skips delays |
| Shadow state snapshots | ❌ | ✅ | ❌ | CSV equity + positions snapshots |

## Execution Mode Summary

| Mode | Engine | Data Source | Use Case |
|---|---|---|---|
| **Historical** | `BacktestEngine` | Parquet files (daily OHLCV) | Full 20-year backtest with analytics |
| **Shadow** | `SimulationEngine` + `AlpacaExecutor` | Live Alpaca bars | Paper-trade alongside live market |
| **Replay** | `AnimatedReplay` + `SimulationEngine` | Parquet files | Visual replay of historical trades |

## Config Keys

| Key | Default | Description |
|---|---|---|
| `backtest.execution_price` | `"close"` | `"close"` or `"vwap"` — sets execution price model |
| `backtest.slippage_pct` | `0.0005` | Flat slippage per side (5 bps) |
| `backtest.commission_pct` | `0.001` | Commission per trade side (10 bps) |
| `backtest.position_size_method` | `"equal_weight"` | Sizing: equal_weight, signal_proportional, volatility_scaled, kelly |
| `backtest.stop_loss_pct` | `null` | Stop loss threshold (null = disabled) |
| `backtest.trailing_stop.enabled` | `false` | Trailing stop toggle |
| `backtest.time_decay_exit.enabled` | `false` | Time-decay exit toggle |
| `backtest.use_regime_filter` | `true` | Regime-aware trade gating |
