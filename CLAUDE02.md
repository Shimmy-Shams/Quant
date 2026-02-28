# CLAUDE02 — Codespaces Development Context

> **Last Updated**: 2026-02-28 (Signal Mode Analysis, Cache Staleness Fix, VM Stabilization)
> **Purpose**: Context document for Claude sessions in GitHub Codespaces
> **Companion**: CLAUDE01.md (Local/VM — Live Trading & IB Connection)
> **Latest Commit**: `30fcb20` — fix: fetch_data() cache shortcut validates data freshness

---

## Quick Reference

### Infrastructure
| Component | Details |
|-----------|---------|
| **Dev Environment** | GitHub Codespaces (venv at `/workspaces/Quant/venv/`) |
| **Production VM** | Oracle Cloud Ubuntu @ `40.233.100.95` |
| **SSH Access** | `ssh -i ~/.ssh/id_rsa ubuntu@40.233.100.95` then `sudo su - trader` |
| **Service** | `sudo systemctl {start,stop,restart,status} quant-trader` |
| **Broker** | Alpaca (paper trading), keys in `/home/trader/Quant/.env` |
| **Dashboard** | GitHub Pages on `dashboard-live` branch, served from `docs/` folder |
| **Dashboard URL** | `https://shimmy-shams.github.io/Quant/` |
| **Deploy Key** | `/home/trader/.ssh/id_ed25519_quant_dashboard` (SSH config maps to github.com) |
| **Git Remote** | Fetch: HTTPS, Push: SSH (`git@github.com:Shimmy-Shams/Quant.git`) |

### Deployment Workflow (Codespaces to VM)
```bash
# 1. Push from Codespaces
git add -A && git commit -m "message" && git push origin main

# 2. Deploy to VM (one-liner)
ssh ubuntu@40.233.100.95 "sudo systemctl stop quant-trader && sudo -u trader bash -c 'cd /home/trader/Quant && git pull origin main' && sudo systemctl start quant-trader && sleep 3 && sudo systemctl status quant-trader --no-pager -l | head -15"

# 3. Manual dashboard refresh
ssh ubuntu@40.233.100.95 "sudo -u trader bash -c 'cd /home/trader/Quant && source venv/bin/activate && python3 -c \"import sys; sys.path.insert(0, \\\"src\\\"); from main_trader import _generate_and_push_dashboard; _generate_and_push_dashboard(auto_push=True)\"'"
```

### Dashboard Push Architecture
Dashboard HTML is generated on `main` branch (where `live_state.json` exists), read into memory, then written to `dashboard-live` branch:
```
main_trader.py -> _generate_and_push_dashboard()
  1. Generate docs/index.html ON MAIN (reads live_state.json)
  2. Read HTML into memory variable (dashboard_html)
  3. Also read 5 JSON state files into memory (equity_history, etc.)
  4. git stash -> checkout dashboard-live
  5. Write saved HTML to docs/index.html
  6. Restore JSON files from memory (gitignored on main, tracked on dashboard-live)
  7. git add + commit + force push (SSH)
  8. FINALLY block: force-checkout main -> stash pop (always runs, even on error)
```
**Critical**: HTML and JSON files MUST be read into memory BEFORE branch switch. The `finally` block ensures we always return to `main` even if push fails. JSON files are gitignored on `main` but bundled into `dashboard-live` push.

---

## Current Codebase

### Architecture Overview
```
src/
+-- main_trader.py              # Headless 24/7 trader (VM entry point, 2-phase T+1)
+-- main_alpaca_trader.ipynb    # Interactive notebook (dev/analysis)
+-- dashboard_generator.py      # HTML dashboard generator
+-- strategy_config.py          # YAML config loader
+-- signal_mode_analysis.ipynb  # NEW: Gated vs Confirmation mode comparison
+-- trading/                    # Shared pipeline package
|   +-- pipeline.py             # Universe, data, signals, executor/sim factories
|   +-- analysis.py             # Plotting, metrics, comparison functions
|   +-- dashboard.py            # Position monitoring (notebook only)
+-- strategies/
|   +-- mean_reversion.py       # Signal generation (Kalman, OU, RSI, dynamic short)
+-- backtest/
|   +-- engine.py               # Vectorized backtest engine (run_backtest method)
|   +-- analytics.py            # Performance analytics
|   +-- optimizer.py            # Walk-forward optimization (Bayesian/Optuna)
+-- execution/
|   +-- alpaca_executor.py      # Signal-to-trade decisions
|   +-- simulation.py           # Day-by-day replay & shadow engine
|   +-- intraday_monitor.py     # Dynamic exit monitor (stop-loss, trailing, time-decay)
+-- connection/
|   +-- alpaca_connection.py    # Alpaca API (stocks + crypto, 3 modes)
+-- data/
|   +-- alpaca_data.py          # Cache-first data loading (per-symbol staleness fix)
|   +-- universe_builder.py     # SP500/NASDAQ/DOW/Russell symbol lists
|   +-- ...
+-- config/
|   +-- config.py               # Internal config module
+-- tests/
    +-- test_intraday_monitor.py # 30 unit tests for intraday exit monitor
```

### File Inventory (as of 2026-02-28)

| File | LOC | Purpose |
|------|-----|---------|
| `src/main_trader.py` | 1751 | Headless 24/7 trader: 2-phase T+1, dashboard push, equity tracking |
| `src/dashboard_generator.py` | 1715 | HTML dashboard: yfinance server quotes + client-side CORS refresh |
| `src/backtest/engine.py` | 1149 | Vectorized backtest (`run_backtest` method at line 379) |
| `src/backtest/analytics.py` | 1417 | Performance analytics (risk, rolling, trade, cost, regime) |
| `src/strategies/mean_reversion.py` | 861 | Signal generation (Kalman, OU, RSI divergence, dynamic short) |
| `src/execution/simulation.py` | 804 | Day-by-day replay & shadow engine |
| `src/connection/alpaca_connection.py` | 752 | Alpaca API connection (stocks + crypto, LIVE/SHADOW/REPLAY) |
| `src/execution/alpaca_executor.py` | 718 | Signal-to-trade decisions with full backtest risk controls |
| `src/execution/intraday_monitor.py` | 711 | Dynamic exit monitor (stop-loss, trailing stop, time-decay) |
| `src/data/alpaca_data.py` | 636 | Cache-first data loading, per-symbol staleness detection |
| `src/backtest/optimizer.py` | 551 | Walk-forward optimization (Bayesian/Optuna) |
| `src/trading/pipeline.py` | 386 | Universe select, data fetch, signal gen, executor/sim factories |
| `src/trading/analysis.py` | 375 | Shadow replay plots, metrics comparison, equity charts |
| `src/strategy_config.py` | 344 | YAML config loader |
| `src/trading/dashboard.py` | 112 | Position monitoring dashboard (explicit params, no globals) |

### Notebooks

| Notebook | Purpose | Status |
|----------|---------|--------|
| `src/main_mean_reversion.ipynb` | Core strategy dev, bridge data export | Complete |
| `src/main_alpaca_trader.ipynb` | Interactive trading (16 cells) | Complete |
| `src/replay_analytics.ipynb` | T+0/T+1 replay comparison | Complete |
| `src/optimization_comparison.ipynb` | Walk-forward optimization | Complete |
| `src/validation_tests.ipynb` | T+1, OOU, slippage stress tests | Complete |
| `src/signal_mode_analysis.ipynb` | **NEW**: Gated vs Confirmation mode analysis | Complete (14 cells) |

### Key Data Files
```
data/snapshots/
+-- notebook_bridge/
|   +-- core_results.pkl        # Bridge data for analysis notebooks (~203 MB)
+-- signal_history.json         # Daily signal snapshots (gitignored on main, pushed to dashboard-live)
+-- trade_history.json          # Daily trade results (same)
+-- live_state.json             # Current positions/equity (same)
+-- equity_history.json         # Equity curve snapshots (same)
+-- intraday_equity.json        # 1D chart data (same)
+-- signal_cache/               # T+1 signal cache (gitignored, env-specific)
|   +-- live/                   #   Live mode signals
|   +-- shadow/                 #   Shadow mode signals
+-- shadow_state.csv            # Shadow position persistence
+-- hurst_rankings.csv          # Daily Hurst rankings
+-- alpaca_cache/               # Cached Alpaca data (per-symbol parquets)
+-- trading_logs/               # Replay/shadow CSV logs

docs/
+-- index.html                  # Generated dashboard (NEVER edit manually)
```

**JSON State Files** -- These 5 files are gitignored on `main` but pushed to `dashboard-live`:
- `signal_history.json`, `trade_history.json`, `live_state.json`, `equity_history.json`, `intraday_equity.json`
- They exist on the VM at runtime and get bundled into each dashboard push

---

## 2-Phase T+1 Architecture (CRITICAL)

The live bot uses a two-phase execution model to eliminate T+0 look-ahead bias:

### Phase 1: Post-Close Signal Generation (~4:10 PM ET)
- Waits for market close, then fetches Day T closing prices
- Generates signals using full `mean_reversion.py` pipeline
- Caches signals to parquet: `data/snapshots/signal_cache/{live,shadow}/`
- Records signal snapshots to `signal_history.json`

### Phase 2: Morning Execution (9:35 AM T+1)
- Loads cached signals from Phase 1
- Overlays live T+1 opening prices for final validation
- Executes trades via `alpaca_executor.py`
- Records trade results to `trade_history.json`

### Intraday Monitor (09:45-15:50 ET)
- Polls every 5 minutes for held positions
- Checks: stop-loss, trailing stop, time-decay exit, circuit breaker
- 30 unit tests in `src/tests/test_intraday_monitor.py`

### Timing Validation
| Scenario | Sharpe | Notes |
|----------|--------|-------|
| T+0 Close (backtest ideal) | 6.99 | Look-ahead bias |
| T+1 Open (production) | 6.00 | Current live config |
| T+1 Close (conservative) | 3.48 | Possible alternative |

---

## Signal System Deep Dive

### Gated Mode (Production -- CONFIRMED OPTIMAL)

```
composite = gate * (1.0 + zscore_boost_factor * |z-score|)

Where gate = rsi_divergence signal (-1, 0, or +1)
```

**Triple-gate entry requirements:**
1. RSI divergence != 0 (momentum exhaustion detected)
2. |OU expected return| >= 0.5% hurdle (ou_hurdle_rate = 0.005)
3. |composite signal| > entry_threshold (1.5)

Additionally:
- Dynamic short filter zeros out low-confidence shorts (3-factor: trend/momentum/vol)
- OU hurdle rate at line 760 of `mean_reversion.py`

**Key behavior**: Gate = 0 means signal = 0 regardless of z-score. This is by design.
On any given day, most/all of the 100 universe symbols will have signal = 0.
The strategy fires only ~1-2% of symbol-days across the full dataset.
**This is a feature, not a bug.**

### Signal Mode Analysis Results (Feb 28, 2026)

Full comparison run in `signal_mode_analysis.ipynb` using 20-year bridge data:

| Metric | Gated (1.5) | Confirmation (best=3.0) |
|--------|-------------|-------------------------|
| Sharpe Ratio | **3.22** | 1.20 |
| Sortino Ratio | **3.80** | 1.58 |
| CAGR | **42.8%** | 28.5% |
| Max Drawdown | **12.4%** | 26.4% |
| Calmar Ratio | **3.45** | 1.08 |
| Profit Factor | **2.42** | 1.24 |
| Win/Loss Ratio | **1.35** | 0.72 |
| Total Trades | 16,804 | 8,302 |
| Gated wins per year | **20/20 years** | -- |

**Trade Quality (Gated)**: Avg win +0.03%, avg loss -0.02%, profit factor 2.42, avg hold 2.6 days, long WR 61.6%, short WR 71.4%.

**Conclusion**: Gated mode dominates on every risk-adjusted metric across every single year. The conservative approach (fewer but higher-conviction trades) is conclusively superior. Do NOT switch to confirmation mode.

### Key Configuration (config.yaml)
| Parameter | Value | Description |
|-----------|-------|-------------|
| `entry_threshold` | 1.5 | WF consensus median (range 1.1-4.0) |
| `exit_threshold` | 0.50 | WF consensus median |
| `stop_loss_pct` | 0.12 | Both long and short |
| `take_profit_pct` | 0.18 | WF consensus mode 0.20 |
| `max_holding_days` | 10 | WF consensus median |
| `max_universe_size` | 100 | Top mean-reverting symbols (Hurst filter) |
| `lookback_days` | 504 | Signal warmup window |
| `position_size_method` | `volatility_scaled` | Target 15% annualized vol |
| `signal_mode` | `gated` | RSI divergence gates entries |
| `gate_signal` | `rsi_divergence` | Which signal acts as entry gate |
| `zscore_boost_factor` | 0.5 | How much |zscore| boosts gated signal |
| `ou_hurdle_rate` | 0.005 | 0.5% minimum expected return (SignalConfig default) |
| `trailing_stop` | 5% trail, 2% activation | Locks in profits |
| `time_decay_exit` | 10 days, 1% threshold | Exits flat trades |
| `intraday_monitor` | 5-min polls, 09:45-15:50 ET | Exit monitoring |

---

## BacktestEngine API Reference

**IMPORTANT**: The correct method is `run_backtest`, NOT `run`.

```python
from backtest.engine import BacktestEngine, BacktestConfig

engine = BacktestEngine(bt_config)
results = engine.run_backtest(
    price_data=price_df,        # DataFrame: symbols as columns, dates as index
    signal_data=signal_df,      # DataFrame: same shape as price_data
    volume_data=volume_df,      # Optional: volume data
    regime_data=None,           # Optional: regime multipliers (0-1)
    exit_signal_data=zscore_df, # Optional: raw z-score for exit decisions
    vwap_data=None,             # Optional: VWAP data
)

# BacktestResults attributes:
results.sharpe_ratio        # float
results.total_return        # float (decimal, not %)
results.max_drawdown        # float (decimal)
results.win_rate            # float (decimal)
results.total_trades        # int
results.equity_curve        # pd.Series (dates -> equity)
results.returns             # pd.Series (daily returns)
results.trades              # list of Trade objects (.pnl, .pnl_pct, .holding_days, .side, .symbol, .entry_date)
```

---

## Recent Changes & Bug Fixes (Feb 26-28, 2026)

### Cache Staleness Fix -- TWO-LAYER BUG (commits `43fa866` + `30fcb20`)

**Problem**: Only 3/100 symbols had fresh data. The other 97 were stuck at Feb 23 prices.

**Layer 1** -- `fetch_pipeline_data()` in `alpaca_data.py`:
- Used `max()` of all cache end dates to check freshness
- 3 symbols happened to have data through Feb 26, so `max()` said "fresh"
- 97 symbols stuck at Feb 23 were invisible to the global check
- **Fix** (`43fa866`): After global check passes, iterate per-symbol. Any symbol whose end date < `latest_cached` gets an incremental fetch. Added `per_symbol_stale` list with logging.

**Layer 2** -- `fetch_data()` in `pipeline.py`:
- Cache shortcut checked file mtime (modification time) instead of data content
- Hurst refresh calls `save_cache()` which updates mtime without updating data
- **Fix** (`30fcb20`): Compute `data_freshness = fresh_count / total_symbols`. If < 80% of symbols share the latest end date, fall through to API fetch instead of trusting cache.

**Result**: Valid signals jumped from **3 to 90** after both fixes deployed on VM.

### VM Stabilization (commits `e5d88de` through `43fc7a8`)

| Fix | Commit | Detail |
|-----|--------|--------|
| Dashboard push robustness | `e5d88de` | Force-checkout to main in finally block, SSH push URL |
| Gitignore state JSONs | `371932a` | 5 JSON files gitignored on main, bundled into dashboard-live push |
| JSON restore after branch switch | `43fc7a8` | Read JSONs into memory before switch, restore after returning to main |
| Signal logging improvement | `371932a` | Always-visible summary: "X valid signals, Y with signal != 0" |

### Alpaca Data Feed Verification
- IEX daily bars use **regular session close (4 PM ET)** only
- No after-hours data is included in daily bars
- Confirmed via timestamp analysis of Alpaca API responses
- This means signal generation is safe from AH noise

### VM Health Check Results (Feb 27, 2026)
All 10 checks passed:
1. Service running (systemd active)
2. Python environment (venv, all packages)
3. Git status (clean, on main)
4. Config and .env present
5. Data directories exist and populated
6. Signal cache directories exist
7. Dashboard push works (SSH)
8. Alpaca API connection (account active)
9. Signal generation completes (90 valid signals)
10. File permissions (trader:trader ownership)

---

## Live Trading Status (Alpaca Paper)

### Account (as of Feb 27, 2026)
- **Portfolio Value**: ~$992,441
- **Cash**: ~$903,912
- **Mode**: Live (paper trading with real-time IEX data)
- **Feed**: IEX (regular session only, daily bars close at 4 PM ET -- no after-hours noise)

### Current Positions
| Symbol | Side | Qty | Notes |
|--------|------|-----|-------|
| ADBE | Long | 233 | With bracket (stop-loss) order |
| DY | Short | 230 | With bracket order |
| LOW | Short | 163 | With bracket order |
| RYAN | Long | 340 | With bracket order |

### Why Most Signals Are Zero
On any given day in live trading, you will see output like:
```
Valid signals: 90, non-zero: 0
```
This is **expected gated mode behavior**. The RSI divergence gate fires on only ~10-20% of symbol-days, and combined with the OU hurdle + z-score magnitude requirement, only ~1-2% of symbol-days produce actionable signals. The strategy is designed to wait for high-conviction setups.

### VM Service
- **Service**: `quant-trader.service` (systemd, enabled, auto-restart)
- **User**: `trader` (dedicated, non-root)
- **Venv**: `/home/trader/Quant/venv/`
- **Memory**: 2GB limit (`MemoryMax=2G`)
- **ExecStart**: `/home/trader/Quant/venv/bin/python src/main_trader.py --mode live --push-dashboard`
- **Execution**: Phase 1 ~4:10 PM ET, Phase 2 9:35 AM T+1

---

## Dashboard System

### Architecture: Two-Layer Price System

**Layer 1 -- Server-side (reliable, yfinance)**:
- `_fetch_server_quotes()` in `dashboard_generator.py` uses `yfinance`
- Returns: `price`, `marketState`, `prePrice`, `postPrice`, `previousClose`
- Falls back to bar data for pre/post prices when `t.info` fields are missing
- Embedded in HTML as `DATA.server_quotes` -- available instantly on page load

**Layer 2 -- Client-side (enhancement, CORS proxy)**:
- JavaScript fetches Yahoo v8/chart endpoint via CORS proxies every 5s
- Derives `postPrice`/`prePrice` from last bar timestamp when meta fields are null
- Merges with server quotes: inherits `prePrice`, `postPrice`, `name` from server
- `marketState` overridden by time-based `getCurrentMarketState()` (more reliable)

### Dashboard Features
| Feature | Description |
|---------|-------------|
| **Account metrics** | Portfolio value, cash, day P&L, total return |
| **Market status badge** | Pre-Market / Market Open / After Hours / Closed |
| **Position table** | Live prices, entry, P&L, day change, market value, % portfolio |
| **Pre/post market prices** | PM/AH badge with effective price (derived from bar data) |
| **Price flash animation** | Blue pulse on cells when prices change |
| **Watchlist dropdown** | TradingView chart + Yahoo Finance links |
| **Equity curve** | Chart.js with range toggles: All / 1Y / 3M / 1M / 1D |
| **Auto-refresh** | 5s interval with countdown timer |

---

## Notebook Bridge Data

Analysis notebooks load precomputed data from `data/snapshots/notebook_bridge/core_results.pkl` (~203 MB):
```python
import pickle
_bridge = pickle.load(open('data/snapshots/notebook_bridge/core_results.pkl', 'rb'))
# Contains:
_bridge['results']       # BacktestResults (gated mode)
_bridge['bt_config']     # BacktestConfig
_bridge['config_path']   # Path to config.yaml
_bridge['price_df']      # Close prices DataFrame
_bridge['signal_df']     # Gated signal DataFrame
_bridge['volume_df']     # Volume DataFrame
_bridge['zscore_df']     # Z-score DataFrame
_bridge['analysis_df']   # Analysis DataFrame
_bridge['all_data']      # Full data dict
```
This bridge was exported from `main_mean_reversion.ipynb` and is used by `signal_mode_analysis.ipynb`, `replay_analytics.ipynb`, and `validation_tests.ipynb`.

---

## Bugs Fixed (All Sessions)

| # | Bug | Root Cause | Fix | Commit |
|---|-----|-----------|-----|--------|
| 1 | Positions missing from dashboard | `git stash` hid `live_state.json` | Generate HTML on main before branch switch | -- |
| 2 | Yahoo prices not updating | CORS proxy caching | Cache-busting `&_=timestamp` + `no-store` headers | -- |
| 3 | Chart flickering on refresh | TradingView iframe recreated every 5s | Track `renderedChartSymbol`, skip if unchanged | -- |
| 4 | CRWD stuck at close price | v8/chart `marketState: null` overwrote server PRE | Merge client quotes with server pre/post data | -- |
| 5 | preChangePct showing -37% | yfinance raw decimal x100 twice | Sanity check, recompute if abs(%) > 10 | -- |
| 6 | Only 4/60 symbols loaded | Cache from old universe, 7% coverage | `min_coverage=80%` threshold in `fetch_data()` | `8761234` |
| 7 | AH prices show regular close | `meta.postMarketPrice` null during AH | Derive from last bar close when timestamp >= 16:00 ET | `1e5d478` |
| 8 | Dashboard push crash leaves VM on wrong branch | HTTPS push fails, no recovery | `finally` block force-checkouts to main, SSH push | `e5d88de` |
| 9 | JSON files deleted on branch switch | Gitignored on main but exist on disk | Read into memory before switch, restore after | `43fc7a8` |
| 10 | 97/100 symbols stale (per-symbol) | `max()` of end dates masked individual staleness | Per-symbol iteration, incremental fetch for stale symbols | `43fa866` |
| 11 | Cache shortcut trusts mtime not content | Hurst `save_cache()` updates mtime without refreshing data | Freshness ratio check: >=80% of symbols must share latest date | `30fcb20` |
| 12 | BacktestEngine.run() does not exist | Wrong method name in analysis notebook | Changed to `run_backtest()` with correct param names | local fix |

---

## Phase History

| Phase | Date | Key Achievement |
|-------|------|-----------------|
| 1 | Feb 13 | Data infrastructure, IB connection, universe builder |
| 2 | Feb 13-14 | Mean reversion engine, 6 accounting bugs fixed |
| 2.5 | Feb 15 | Kalman filter, OU prediction, vol-scaled sizing |
| 2B | Feb 15 | Yahoo Finance 20-year data collection (258 tickers) |
| A | Feb 15 | Signal reweighting (RSI divergence 0.75) |
| **B** | **Feb 16** | **Gated signal architecture -- 186x improvement** |
| 3A | Feb 16 | Vectorized backtest (200-400x speedup) |
| 3B | Feb 16 | 20-year validation (4,198% return, 3.75 Sharpe) |
| 3C | Feb 16-17 | Alpaca paper trading, Oracle Cloud deployment |
| 3D | Feb 17-18 | Dashboard complete: server-side yfinance + client CORS merge |
| 3E | Feb 18 | Code cleanup: trading/ package, cache coverage fix, AH price fix |
| -- | Feb 18-26 | 2-phase T+1 architecture, validation tests, replay analytics (CLAUDE01) |
| -- | **Feb 26-27** | **Cache staleness fix (2-layer), VM stabilization, dashboard robustness** |
| -- | **Feb 28** | **Signal mode analysis notebook -- gated confirmed optimal (20/20 years)** |

---

## Future Phases

### Phase 4: Algorithm Deep Dive & Optimization
- Capital efficiency improvement (currently ~1.8% avg exposure)
- ML signal filter (binary classifier for signal quality)
- Use `signal_history.json` + `trade_history.json` for training data

### Phase 5: Multivariate / Cross-Sectional
- Pairs trading, factor-augmented signals, ensemble

### Phase 6: Live Deployment
- Shadow to live migration when validated
- Position sizing scale-up

---

## Environment Variables (`.env`)
```bash
ALPACA_API_KEY=...
ALPACA_SECRET_KEY=...
ALPACA_BASE_URL=https://paper-api.alpaca.markets
ALPACA_DATA_URL=https://data.alpaca.markets
```

---

## VM Operations Quick Reference
```bash
# SSH to VM
ssh -i ~/.ssh/id_rsa ubuntu@40.233.100.95
sudo su - trader

# Service management
sudo systemctl status quant-trader
sudo systemctl restart quant-trader

# View logs
sudo journalctl -u quant-trader --since '10 min ago' --no-pager | tail -40
sudo journalctl -u quant-trader -f  # follow live

# Manual signal generation
sudo -u trader bash -c 'cd /home/trader/Quant/src && /home/trader/Quant/venv/bin/python main_trader.py --mode live --generate-signals'

# Pull latest code
sudo -u trader bash -c 'cd /home/trader/Quant && git pull origin main'

# Check signal cache
sudo -u trader ls -la /home/trader/Quant/data/snapshots/signal_cache/live/
```

### main_trader.py CLI Reference
```bash
python main_trader.py                              # Shadow mode (default), daily loop
python main_trader.py --mode live                   # Live paper trading, daily loop
python main_trader.py --mode live --push-dashboard  # Live + dashboard push (production)
python main_trader.py --mode live --generate-signals # Manual signal gen, exit after
python main_trader.py --mode live --once             # Single full cycle (legacy), exit
python main_trader.py --update-dashboard-only        # Refresh dashboard, exit
```

---

## Key Reminders for New Agents

1. **Dashboard pushes to `dashboard-live` branch**, NOT `main`. HTML generated on `main` first.
2. **BacktestEngine method is `run_backtest()`**, NOT `run()`. Parameters are `price_data`, `signal_data`, `volume_data`, `exit_signal_data` (NOT `signal_df`, `price_df`, `zscore_df`).
3. **All strategy params in `config.yaml`** -- no hardcoded values. `ou_hurdle_rate` defaults to 0.005 in `SignalConfig`.
4. **5 JSON state files are gitignored on main** -- they exist on VM and get bundled into dashboard-live push.
5. **`fetch_data()` has two-layer freshness checks** -- min_coverage=80% in pipeline.py + per-symbol staleness in alpaca_data.py.
6. **Signal = 0 for most symbols is EXPECTED** -- gated mode fires ~1-2% of symbol-days. Not a bug.
7. **IEX feed uses regular session close (4 PM ET)** -- no after-hours data in daily bars.
8. **VM git uses split remote**: HTTPS for fetch, SSH for push (deploy key).
9. **Codespace venv**: `source venv/bin/activate` (may need recreation after sessions).
10. **`src/trading/` package is shared** between notebook and main_trader.py -- changes affect both.
11. **Bridge data** at `data/snapshots/notebook_bridge/core_results.pkl` (~203 MB) -- used by analysis notebooks.
12. **No emojis in code** (project standard).
13. **Gated mode is confirmed optimal** -- 20/20 years, Sharpe 3.22 vs 1.20 confirmation. Do NOT switch signal modes.
14. **2-Phase T+1 is production architecture** -- Phase 1 at ~4:10 PM, Phase 2 at 9:35 AM. Do not revert to T+0.

---

**Repo**: https://github.com/Shimmy-Shams/Quant
