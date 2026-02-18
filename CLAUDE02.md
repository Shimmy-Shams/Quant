# CLAUDE02 — Codespaces Development Context

> **Last Updated**: 2026-02-17 (Phase 3D — Dashboard & Live Monitoring)
> **Purpose**: Context document for Claude sessions in GitHub Codespaces
> **Companion**: CLAUDE01.md (Local/VM — Live Trading & IB Connection)

---

## Quick Reference

### Infrastructure
| Component | Details |
|-----------|---------|
| **Dev Environment** | GitHub Codespaces |
| **Production VM** | Oracle Cloud Ubuntu @ `40.233.100.95` |
| **SSH Access** | `ssh -i ~/.ssh/id_rsa ubuntu@40.233.100.95` → then `sudo su - trader` |
| **Service** | `sudo systemctl {start,stop,restart,status} quant-trader` |
| **Broker** | Alpaca (paper trading), keys in `/home/trader/Quant/.env` |
| **Dashboard** | GitHub Pages on `dashboard-live` branch, served from `docs/` folder |
| **Dashboard URL** | `https://shimmy-shams.github.io/Quant/` |
| **Deploy Key** | `/home/trader/.ssh/id_ed25519_quant_dashboard` (SSH config maps to github.com) |

### Deployment Workflow (Codespaces → VM)
```bash
# 1. Commit & push from Codespaces
git add . && git commit -m "message" && git push origin main

# 2. SSH to VM and update
ssh -i ~/.ssh/id_rsa ubuntu@40.233.100.95
sudo su - trader
cd ~/Quant
sudo systemctl stop quant-trader
git reset --hard origin/main
sudo systemctl start quant-trader

# 3. Test dashboard generation
source venv/bin/activate
python src/main_trader.py --update-dashboard-only

# 4. Verify dashboard push
# Check: https://shimmy-shams.github.io/Quant/
```

### Dashboard Push Architecture
The dashboard pushes to `dashboard-live` branch (not `main`) to keep the main branch clean:
```
main_trader.py → _generate_and_push_dashboard()
  1. git stash (save any local changes)
  2. git checkout dashboard-live
  3. Generate docs/index.html
  4. git add + commit
  5. git push origin dashboard-live (force)
  6. git checkout main
  7. git stash pop
```

GitHub Pages is configured to serve from `dashboard-live` branch, `docs/` folder.

### VM SSH Config (`/home/trader/.ssh/config`)
```
Host github.com
    HostName github.com
    User git
    IdentityFile /home/trader/.ssh/id_ed25519_quant_dashboard
    IdentitiesOnly yes
```

---

## Current Codebase

### File Inventory (as of 2026-02-17)

| File | LOC | Purpose |
|------|-----|---------|
| `src/main_trader.py` | 810 | Headless 24/7 trader, dashboard push, equity tracking |
| `src/dashboard_generator.py` | 774 | HTML dashboard with Yahoo Finance live prices |
| `src/strategies/mean_reversion.py` | 861 | Signal generation (Kalman, OU, RSI divergence, dynamic short) |
| `src/backtest/engine.py` | 774 | Vectorized backtest (3s for 5000 days × 258 symbols) |
| `src/backtest/analytics.py` | 1055 | Performance analytics (risk, rolling, trade, cost, regime) |
| `src/backtest/optimizer.py` | 525 | Walk-forward optimization (Bayesian/Optuna) |
| `src/execution/alpaca_executor.py` | 524 | Signal-to-trade decisions with full backtest risk controls |
| `src/execution/simulation.py` | 729 | Day-by-day replay & shadow engine |
| `src/data/alpaca_data.py` | 589 | Cache-first data loading, concurrent fetch |
| `src/connection/alpaca_connection.py` | 462 | Alpaca API connection (stocks + crypto, LIVE/SHADOW/REPLAY) |
| `src/strategy_config.py` | 318 | YAML config loader |
| `config.yaml` | 275 | All strategy parameters |
| `src/main_alpaca_trader.ipynb` | 16 cells | Interactive Alpaca trading notebook |
| `deploy/setup.sh` | 145 | Oracle Cloud provisioning script |
| `deploy/quant-trader.service` | 34 | systemd service definition |

### Key Directories
```
data/
├── historical/daily/          # 258 parquet files (20 years daily OHLCV)
├── snapshots/
│   ├── alpaca_cache/          # Cached Alpaca data (per-symbol parquets)
│   ├── shadow_state.csv       # Shadow position persistence
│   └── trading_logs/          # Replay/shadow CSV logs
├── logs/                      # Application logs
└── universe/                  # Index composition JSONs

docs/
└── index.html                 # Generated dashboard (served by GitHub Pages)
```

---

## Strategy Overview

### Signal System (Phase B — Gated Mode)
```
RSI Divergence (gate signal) × (1 + 0.5 × |z-score|)
  ↓
Entry when |composite| > 1.43
Exit when |z-score| < 0.50   ← Separate exit signal
  ↓
Dynamic short filter (3-factor confidence: trend/momentum/vol)
Trailing stop (5% from peak after 2% profit)
Time decay exit (after 10 days if |PnL| < 1%)
```

### Current Config (Consensus Parameters)
```yaml
backtest:
  entry_threshold: 1.43
  exit_threshold: 0.50
  stop_loss_pct: 0.10
  short_stop_loss_pct: 0.10
  take_profit_pct: 0.15
  max_holding_days: 20
  position_size_method: volatility_scaled

signals:
  signal_mode: gated
  gate_signal: rsi_divergence
  zscore_boost_factor: 0.5
  gate_persistence_days: 7
  dynamic_short_filter:
    enabled: true
    min_confidence: 0.3
```

---

## Performance Summary — 20-Year Backtest

**Period**: 2006-02-14 to 2026-02-13 (5,032 days)
**Universe**: 216 mean-reverting stocks (Hurst < 0.5)

| Metric | Value | Context |
|--------|-------|---------|
| **Total Return** | 4,198.21% | 43x initial capital |
| **Annualized Return** | 20.72% | CAGR |
| **Sharpe Ratio** | 3.75 | Institutional-grade |
| **Sortino Ratio** | 6.41 | Exceptional downside protection |
| **Max Drawdown** | 1.25% | Minimal |
| **Win Rate** | 98.38% | 789 wins / 13 losses |
| **Profit Factor** | 125.74 | Wins/losses ratio |
| **Total Trades** | 802 | ~40/year |
| **Avg Exposure** | 3.89% | Low but profitable |
| **EV/Trade** | 4.74% | Long: 5.10%, Short: 4.06% |

### Regime Performance (Strategy thrives in crises)
| Regime | Trades | WR | AvgPnL | Return | Sharpe |
|--------|--------|------|--------|--------|--------|
| GFC | 17 | 100% | +5.02% | +8.87% | 2.96 |
| COVID Crash | 16 | 100% | +7.66% | +13.20% | 5.83 |
| 2022 Bear | 80 | 96.2% | +5.94% | +60.67% | 5.95 |
| 2024 Bull | 132 | 99.2% | +4.24% | +73.50% | 9.12 |

---

## Live Trading Status (Alpaca Paper)

### Account (as of 2026-02-17)
- **Equity**: ~$999,577
- **Cash**: ~$903,936
- **Mode**: Shadow (paper trading with real-time data)

### Current Positions
| Symbol | Side | Qty | Entry Price |
|--------|------|-----|-------------|
| CRWD | Long | 232 | ~$413.96 |
| ETHUSD | Short | — | — |

### VM Service
- **Service**: `quant-trader.service` (systemd)
- **User**: `trader` (dedicated, non-root)
- **Venv**: `/home/trader/Quant/venv/`
- **Execution Window**: 9:35 AM → 3:55 PM ET (matches backtest close prices)
- **Cron**: NONE (removed — was causing noisy commits every minute)

---

## Dashboard System (Phase 3D)

### Architecture: Hybrid Server-Side + Client-Side

**Server-side** (pushed once after trading):
- `main_trader.py` saves `live_state.json` (account, positions, trades)
- `dashboard_generator.py` reads state, generates `docs/index.html`
- HTML is pushed to `dashboard-live` branch via git

**Client-side** (runs in browser on every page load):
- Embedded JavaScript fetches live prices from Yahoo Finance via CORS proxies
- Updates positions with current market prices, P&L, day change
- Shows market status badge (Open / Pre-Market / After Hours / Closed)
- Auto-refreshes every 3 seconds

### Yahoo Finance Integration
- No API key needed — uses public Yahoo Finance v8 API
- CORS proxy chain: `corsproxy.io` (primary) → `api.allorigins.win` (fallback)
- Returns: current price, after-hours price, pre-market price, day change
- Works for stocks AND crypto (ETHUSD → ETH-USD conversion)

### Dashboard Features (Current + Planned)
| Feature | Status | Description |
|---------|--------|-------------|
| Account metrics | ✅ Done | Equity, cash, day P&L, total return |
| Market status badge | ✅ Done | Open/Pre-Market/After Hours/Closed with color |
| Position table | ✅ Done | Live prices, P&L from Yahoo Finance |
| Recent trades | ✅ Done | Last N trades with entry/exit details |
| Auto-refresh | ✅ Done | Configurable interval (currently 30s → changing to 3s) |
| Equity curve chart | 🔧 Fixing | Needs equity_history.json (no CSV files in LIVE mode) |
| Enhanced positions | 📋 Planned | Day change %, market value, % of portfolio, days held |
| Watchlist section | 📋 Planned | TradingView mini-charts + Yahoo Finance links for held stocks |

### Key Files
- `src/dashboard_generator.py` — `DashboardGenerator` class, `_CSS` constant, `_build_html()` method
- `src/main_trader.py` — `_save_live_state()` (~line 370-430), `_generate_and_push_dashboard()` (~line 540-600)
- `docs/index.html` — Generated output (never edit manually)

### Planned: Equity History Tracking
**Problem**: No `equity_*.csv` files exist on VM in LIVE mode. The equity graph shows nothing.
**Solution**: Accumulate equity snapshots in `equity_history.json`:
```python
# In _save_live_state() — append equity + timestamp each cycle
equity_history = []  # load existing
equity_history.append({"timestamp": now, "equity": equity_value})
# Save back to equity_history.json
```
Dashboard reads this file and renders Chart.js equity curve.

---

## Key Bugs Fixed (Historical Reference)

These are **already fixed** in current code:

1. **Inverted Short P&L** — Short trade PnL multiplied by negative shares. Fixed: uses `abs(shares)`.
2. **Short Entry Cash Flow** — Shorts subtracted cash instead of adding. Fixed: `cash += entry_value` for shorts.
3. **Exit Cash Double-Counting** — Exit added `exit_value + net_pnl` twice. Fixed: separate long/short exit logic.
4. **Leverage Spiral** — Position sizing used raw cash instead of equity. Fixed: `equity = cash + positions.sum()`.
5. **Signal Normalization Mismatch** — Composite normalized to [-1,+1], threshold was 2.0. Fixed: z-score primary signal.
6. **Signal Directional Bias** — Confirmation formula `z * (1 + c)` amplified positive signals more. Fixed: symmetric agreement.
7. **Git push to wrong branch** — Dashboard pushes went to `main`, GitHub Pages on `dashboard-live`. Fixed: rewrote push logic.
8. **Cron job noise** — Every-minute cron creating noisy git commits. Fixed: removed cron entirely.
9. **SSH deploy key not configured** — Git push failing on VM. Fixed: created `/home/trader/.ssh/config`.
10. **Dashboard not updating** — Multiple causes (git divergence, SSH failure, wrong branch). Fixed: hard-reset + SSH config + branch fix.

---

## Phase History

### Phase 1 (Feb 13) — Data Infrastructure
- Built IB Gateway connection, data collectors, universe builder
- Collected 293 stocks × 2 years daily data
- **Limitation**: Cannot connect to IB from Codespaces (local only)

### Phase 2 (Feb 13-14) — Mean Reversion Engine
- Built `MeanReversionSignals`, `BacktestEngine`, `ParameterOptimizer`
- Centralized config system (`config.yaml` + `ConfigLoader`)
- Fixed 6 critical accounting bugs
- **Result**: 27.2% return, 0.82 Sharpe (2 years)

### Phase 2.5 (Feb 15) — Model Optimization
- Added log returns, Kalman filter, OU prediction, volatility-scaled sizing
- **Result**: 13.58% return, 0.29 Sharpe (still weak)

### Phase A (Feb 15) — Signal Reweighting
- Boosted rsi_divergence to 0.75, disabled bollinger/rsi_level
- **Result**: 21.25% return, 0.45 Sharpe (better but not great)

### Phase B (Feb 16) — Gated Signal Architecture ⭐
- Inverted signal relationship: RSI divergence gates entries, z-score boosts conviction
- Added dynamic short filter, trailing stop, time decay exit
- **Breakthrough**: Architecture change > parameter tuning (186x improvement)
- **Result**: 3,965.96% return, 11.96 Sharpe (2 years)

### Phase 3A (Feb 16) — Vectorization
- Rewrote engine from DataFrame `.loc[]` per-iteration to NumPy arrays
- Pre-computed position sizes as matrix
- **Result**: 200-400x speedup (3.07s for 5000 days × 258 symbols)

### Phase 3B (Feb 16) — Analytics & 20-Year Validation
- Built comprehensive analytics (capital, risk, rolling, trade, cost, regime)
- Validated strategy across 20 years and all market regimes
- **Result**: 4,198% return, 3.75 Sharpe (20 years) — genuine alpha confirmed

### Phase 3C (Feb 16-17) — Alpaca Paper Trading + Deployment
- Complete Alpaca trading pipeline: connection, data, executor, simulation
- Cache-first data loading with concurrent fetch, incremental updates
- Gate persistence fix for SHADOW mode signal generation
- Crypto support (BTC/USD, ETH/USD via Alpaca)
- Headless `main_trader.py` for 24/7 autonomous operation
- Oracle Cloud deployment (setup.sh, systemd, deploy key)
- **Result**: Production-ready shadow trading system on Oracle Cloud

### Phase 3D (Feb 17) — Dashboard & Live Monitoring ← CURRENT
- Rebuilt dashboard with Yahoo Finance live prices (no API key needed)
- Client-side JavaScript fetches real-time prices via CORS proxies
- Market status badges (Open/Pre-Market/After Hours/Closed)
- Fixed git push workflow: `dashboard-live` branch for GitHub Pages
- Fixed VM: SSH deploy key, removed cron, hard-reset git
- **In Progress**: 4 dashboard enhancements (equity graph, 3s refresh, enhanced positions, watchlist)

---

## Pending Work (Phase 3D Enhancements)

### 1. Equity Graph Fix
- Add `equity_history.json` accumulation in `_save_live_state()`
- Dashboard reads it for Chart.js equity curve
- Each entry: `{"timestamp": "...", "equity": 999577.42}`

### 2. Real-Time 3-Second Refresh
- Change `REFRESH_INTERVAL` from 30s → 3s in dashboard_generator.py
- Smooth transitions, visual countdown timer optional

### 3. Enhanced Position Details
- Day change % (from Yahoo Finance `regularMarketChangePercent`)
- Market value (current price × qty)
- % of portfolio (market value / equity)
- Days held (from entry date)

### 4. Watchlist with TradingView Charts
- TradingView mini-chart widgets for each held stock
- Yahoo Finance links for quick external lookup
- Compact card layout with embedded charts

---

## Future Phases (Deferred)

### Phase 4: ML Signal Filter
- Feature engineering from price/volume/volatility/microstructure
- Binary classifier: "Will this signal be profitable?"
- Walk-forward training (no look-ahead bias)

### Phase 5: Multivariate / Cross-Sectional Strategies
- Pairs trading, factor-augmented signals, ensemble approach
- Compare univariate vs multivariate on 20-year dataset

### Phase 6: Live Deployment
- Switch shadow → live when validated
- Capital efficiency improvements (3.89% exposure → target 10-20%)
- Short squeeze protection (PECO-type catastrophe prevention)

---

## Usage (Current Working Code)

### Backtest
```python
from strategy_config import ConfigLoader
from strategies.mean_reversion import MeanReversionSignals, UniverseAnalyzer
from backtest.engine import BacktestEngine
from backtest.analytics import PerformanceAnalytics

config = ConfigLoader(Path('config.yaml'))
signal_config = config.to_signal_config()
bt_config = config.to_backtest_config()

analyzer = UniverseAnalyzer(signal_config)
analysis = analyzer.analyze_universe(price_data)
mean_reverting = analysis[analysis['is_mean_reverting']]['symbol'].tolist()

signal_gen = MeanReversionSignals(signal_config)
composite, individual = signal_gen.generate_composite_signal(prices, volumes, weights=config.get_composite_weights())

engine = BacktestEngine(bt_config)
results = engine.run_backtest(price_df, signal_df, volume_df, exit_signal_data=zscore_df)

analytics = PerformanceAnalytics(results, bt_config, signal_df, output_dir=Path('output'))
```

### Dashboard (Manual Push)
```bash
# On VM as trader user
source venv/bin/activate
python src/main_trader.py --update-dashboard-only
```

---

## Environment Variables (`.env`)
```bash
ALPACA_API_KEY=...
ALPACA_SECRET_KEY=...
ALPACA_BASE_URL=https://paper-api.alpaca.markets
ALPACA_DATA_URL=https://data.alpaca.markets
ALPACA_STREAM_URL=wss://stream.data.alpaca.markets
ALPACA_CRYPTO_STREAM_URL=wss://stream.data.alpaca.markets
```

---

**Remember**:
- CLAUDE02 (Codespaces) = Development & Strategy Building
- CLAUDE01 (Local/VM) = Live Trading & IB Connection
- All parameters in `config.yaml` — no hardcoded values
- Dashboard pushes to `dashboard-live` branch, NOT `main`
- Keep both .md files synced after major changes
