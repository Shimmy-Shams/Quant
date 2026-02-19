# CLAUDE02 — Codespaces Development Context

> **Last Updated**: 2026-02-18 (Phase 3E — Code Cleanup & Bug Fixes)
> **Purpose**: Context document for Claude sessions in GitHub Codespaces
> **Companion**: CLAUDE01.md (Local/VM — Live Trading & IB Connection)

---

## Quick Reference

### Infrastructure
| Component | Details |
|-----------|---------|
| **Dev Environment** | GitHub Codespaces (venv at `/workspaces/Quant/venv/`) |
| **Production VM** | Oracle Cloud Ubuntu @ `40.233.100.95` |
| **SSH Access** | `ssh -i ~/.ssh/id_rsa ubuntu@40.233.100.95` → then `sudo su - trader` |
| **Service** | `sudo systemctl {start,stop,restart,status} quant-trader` |
| **Broker** | Alpaca (paper trading), keys in `/home/trader/Quant/.env` |
| **Dashboard** | GitHub Pages on `dashboard-live` branch, served from `docs/` folder |
| **Dashboard URL** | `https://shimmy-shams.github.io/Quant/` |
| **Deploy Key** | `/home/trader/.ssh/id_ed25519_quant_dashboard` (SSH config maps to github.com) |

### Deployment Workflow (Codespaces → VM)
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
main_trader.py → _generate_and_push_dashboard()
  1. Generate docs/index.html ON MAIN (reads live_state.json)
  2. Read HTML into memory variable (dashboard_html)
  3. git stash → checkout dashboard-live
  4. Write saved HTML to docs/index.html
  5. git add + commit + force push
  6. checkout main → stash pop
```
**Critical**: HTML MUST be generated BEFORE branch switch. Previously generating on `dashboard-live` caused stale data because `git stash` hid `live_state.json`.

---

## Current Codebase

### Architecture Overview
```
src/
├── main_trader.py              # Headless 24/7 trader (VM entry point)
├── main_alpaca_trader.ipynb    # Interactive notebook (dev/analysis)
├── dashboard_generator.py      # HTML dashboard generator
├── strategy_config.py          # YAML config loader
├── trading/                    # ← NEW (Phase 3E): Shared pipeline package
│   ├── pipeline.py             # Universe, data, signals, executor/sim factories
│   ├── analysis.py             # Plotting, metrics, comparison functions
│   └── dashboard.py            # Position monitoring (notebook only)
├── strategies/
│   └── mean_reversion.py       # Signal generation (Kalman, OU, RSI, dynamic short)
├── backtest/
│   ├── engine.py               # Vectorized backtest engine
│   ├── analytics.py            # Performance analytics
│   └── optimizer.py            # Walk-forward optimization (Bayesian/Optuna)
├── execution/
│   ├── alpaca_executor.py      # Signal-to-trade decisions
│   └── simulation.py           # Day-by-day replay & shadow engine
├── connection/
│   └── alpaca_connection.py    # Alpaca API (stocks + crypto, 3 modes)
├── data/
│   ├── alpaca_data.py          # Cache-first data loading
│   ├── universe_builder.py     # SP500/NASDAQ/DOW/Russell symbol lists
│   └── ...
└── config/
    └── config.py               # Internal config module
```

### File Inventory (as of 2026-02-18)

| File | LOC | Purpose |
|------|-----|---------|
| `src/dashboard_generator.py` | ~1566 | HTML dashboard: yfinance server quotes + client-side CORS refresh |
| `src/main_trader.py` | ~751 | Headless 24/7 trader, dashboard push, equity tracking |
| `src/main_alpaca_trader.ipynb` | 16 cells | Interactive notebook (was 21 cells, refactored) |
| `src/trading/pipeline.py` | ~362 | **NEW**: Universe selection, data fetch, signal gen, executor/sim factories |
| `src/trading/analysis.py` | ~375 | **NEW**: Shadow replay plots, metrics comparison, equity charts |
| `src/trading/dashboard.py` | ~112 | **NEW**: Position monitoring dashboard (explicit params, no globals) |
| `src/strategies/mean_reversion.py` | 861 | Signal generation (Kalman, OU, RSI divergence, dynamic short) |
| `src/backtest/engine.py` | 774 | Vectorized backtest (3s for 5000 days × 258 symbols) |
| `src/backtest/analytics.py` | 1055 | Performance analytics (risk, rolling, trade, cost, regime) |
| `src/backtest/optimizer.py` | 525 | Walk-forward optimization (Bayesian/Optuna) |
| `src/execution/alpaca_executor.py` | 524 | Signal-to-trade decisions with full backtest risk controls |
| `src/execution/simulation.py` | 740 | Day-by-day replay & shadow engine |
| `src/data/alpaca_data.py` | 589 | Cache-first data loading, concurrent fetch |
| `src/connection/alpaca_connection.py` | 462 | Alpaca API connection (stocks + crypto, LIVE/SHADOW/REPLAY) |
| `src/strategy_config.py` | 318 | YAML config loader |
| `config.yaml` | 275 | All strategy parameters |

### Key Data Files
```
data/snapshots/
├── live_state.json         # Account, positions, trades (from Alpaca API)
├── equity_history.json     # Accumulated equity snapshots for chart (max 2000)
├── shadow_state.csv        # Shadow position persistence
├── hurst_rankings.csv      # Daily Hurst rankings (refreshed each cycle)
├── alpaca_cache/           # Cached Alpaca data (per-symbol parquets, label=latest)
└── trading_logs/           # Replay/shadow CSV logs

docs/
└── index.html              # Generated dashboard (NEVER edit manually)
```

---

## Recent Changes (Phase 3E — Feb 18, 2026)

### 1. Code Refactoring — `src/trading/` Package
Extracted shared code used by both `main_trader.py` and `main_alpaca_trader.ipynb`:
- **`pipeline.py`** (362 lines): `select_universe`, `refresh_universe_hurst`, `fetch_data`, `generate_signals`, `build_executor`, `build_simulation`
- **`analysis.py`** (375 lines): `plot_shadow_replay`, `print_monthly_returns`, `print_trade_summary`, `print_metrics_comparison`, `print_replay_trade_breakdown`, `plot_equity_comparison`, `plot_export_charts`
- **`dashboard.py`** (112 lines): `show_dashboard()` with explicit params (replaced `globals()` access)

**Impact**: Notebook 1577→969 lines (JSON), 21→16 cells. main_trader.py 960→751 lines.

### 2. Bug Fix — Cache Coverage Check (commit `8761234`)
**Problem**: `fetch_data()` loaded stale cache with only 4/60 symbols matching current universe.
- Cache had 60 symbols from a PREVIOUS Hurst universe (AAPL, TSLA, etc.)
- Current universe was completely different (TRV, SPGI, IBM, etc.)
- Only 4 overlapped: ADP, DY, GTES, KO → strategy ran on 4 stocks instead of 60
- Result: Sharpe 2.29, +3.4% (broken) vs Sharpe 3.30, +52.1% (fixed)

**Fix**: Added `min_coverage` param (default 80%) — rejects cache if <80% of requested symbols are found. Added `allow_stale_cache` param — REPLAY mode bypasses cache entirely.

### 3. Bug Fix — Dashboard After-Hours Prices (commit `1e5d478`)
**Problem**: Yahoo v8 chart API's `meta.postMarketPrice` is often null during the actual after-hours session. Dashboard was showing regular close price during AH.

**Fix**: Both server-side (yfinance) and client-side (JS v8 chart):
- If `postMarketPrice` is missing, check the last bar's timestamp
- If bar time >= 16:00 ET → use that bar's close as `postPrice`
- Same fallback for `preMarketPrice` using bars timestamped 4:00–9:29 AM ET
- Verified: CRWD shows `postPrice=417.22` during POST state (previously null)

---

## Dashboard System (Phase 3D+3E — Complete)

### Architecture: Two-Layer Price System

**Layer 1 — Server-side (reliable, yfinance)**:
- `_fetch_server_quotes()` in `dashboard_generator.py` uses `yfinance` to fetch prices
- Returns: `price`, `marketState`, `prePrice`, `postPrice`, `previousClose`, etc.
- **NEW**: Falls back to bar data for pre/post prices when `t.info` fields are missing
- Each symbol fetched independently with retry (not batched)
- Embedded in HTML as `DATA.server_quotes` — available instantly on page load

**Layer 2 — Client-side (enhancement, CORS proxy)**:
- JavaScript fetches Yahoo v8/chart endpoint via CORS proxies every 5s
- **NEW**: Derives `postPrice`/`prePrice` from last bar timestamp when meta fields are null
- Merges with server quotes: inherits `prePrice`, `postPrice`, `name` from server
- `marketState` overridden by time-based `getCurrentMarketState()` (more reliable)

### Dynamic Price Selection (`effectivePrice()`)
```javascript
function effectivePrice(symbol, quote) {
    const state = (quote.marketState || '').toUpperCase();
    if (['POST','POSTPOST','CLOSED'].includes(state) && quote.postPrice) return quote.postPrice;
    if (['PRE','PREPRE'].includes(state) && quote.prePrice) return quote.prePrice;
    return quote.price;  // regular market hours
}
```

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

## Strategy Overview

### Signal System (Phase B — Gated Mode)
```
RSI Divergence (gate signal) × (1 + 0.5 × |z-score|)
  ↓
Entry when |composite| > 1.43
Exit when |z-score| < 0.50
  ↓
Dynamic short filter (3-factor confidence: trend/momentum/vol)
Trailing stop (5% from peak after 2% profit)
Time decay exit (after 10 days if |PnL| < 1%)
```

### Key Configuration (config.yaml)
| Parameter | Value | Description |
|-----------|-------|-------------|
| `entry_threshold` | 1.43 | Enter when \|signal\| > threshold |
| `exit_threshold` | 0.50 | Exit when \|z-score\| < threshold |
| `stop_loss_pct` | 0.10 | 10% stop loss |
| `take_profit_pct` | 0.15 | 15% take profit |
| `max_holding_days` | 20 | Max holding period |
| `max_universe_size` | 60 | Top mean-reverting symbols |
| `lookback_days` | 500 | Signal warmup window |
| `position_size_method` | `volatility_scaled` | Target 15% annualized vol |
| `signal_mode` | `gated` | RSI divergence gates entries |
| `trailing_stop` | 5% trail, 2% activation | Locks in profits |
| `time_decay_exit` | 10 days, 1% threshold | Exits flat trades |
| `intraday_monitor` | 5-min polls, 09:45–15:50 ET | Exit monitoring between cycles |

### Execution Timing
- **Daily signal cycle**: 3:55 PM ET (last 5 min before close)
- **Intraday exit monitoring**: 09:45 AM – 3:50 PM ET, every 5 minutes
- **Rationale**: Execute near close so IEX daily bar is fully formed and signals match backtest data

### Latest Performance (Shadow Replay, Feb 2026)
| Metric | Shadow 1yr (252d) | Replay 90d | Backtest 90d |
|--------|-------------------|------------|--------------|
| **Total Return** | +52.07% | +15.69% | +14.94% |
| **Sharpe Ratio** | 3.30 | 5.73 | 5.37 |
| **Max Drawdown** | -6.01% | -1.89% | -1.90% |
| **Total Trades** | 473 | 199 | 198 |
| **Win Rate** | 63.0% | 63.3% | 62.1% |
| **Avg Trade P&L** | +1.01% | +0.85% | — |
| **Universe** | 60 symbols | 60 symbols | 60 symbols |
| **Replay-Backtest Correlation** | — | — | 0.9999 |

---

## Live Trading Status (Alpaca Paper)

### Account (as of 2026-02-18)
- **Portfolio Value**: $1,000,403.57
- **Cash**: $903,911.76
- **Mode**: Live (paper trading with real-time data)

### Current Positions
| Symbol | Side | Qty | Entry | Current | Unrealized P&L |
|--------|------|-----|-------|---------|----------------|
| CRWD | Long | 232 | $413.96 | $415.81 | +$427.87 (+0.45%) |
| ETHUSD | Long | 0.012 | $0.00 | $1,943.92 | +$23.89 |

### VM Service
- **Service**: `quant-trader.service` (systemd, enabled)
- **PID**: 44240 (as of last deploy)
- **User**: `trader` (dedicated, non-root)
- **Venv**: `/home/trader/Quant/venv/`
- **Execution Window**: 3:55 PM ET (daily signal cycle)
- **Intraday Monitor**: 09:45–15:50 ET (exit polling every 5 min)

---

## Phase History

| Phase | Date | Key Achievement |
|-------|------|-----------------|
| 1 | Feb 13 | Data infrastructure, IB connection, universe builder |
| 2 | Feb 13-14 | Mean reversion engine, 6 accounting bugs fixed |
| 2.5 | Feb 15 | Kalman filter, OU prediction, vol-scaled sizing |
| A | Feb 15 | Signal reweighting (RSI divergence 0.75) |
| **B** | **Feb 16** | **Gated signal architecture — 186x improvement** |
| 3A | Feb 16 | Vectorized backtest (200-400x speedup) |
| 3B | Feb 16 | 20-year validation (4,198% return, 3.75 Sharpe) |
| 3C | Feb 16-17 | Alpaca paper trading, Oracle Cloud deployment |
| 3D | Feb 17-18 | Dashboard complete: server-side yfinance + client CORS merge |
| **3E** | **Feb 18** | **Code cleanup: trading/ package, cache coverage fix, AH price fix** |

---

## Bugs Fixed (All Sessions)

| # | Bug | Root Cause | Fix |
|---|-----|-----------|-----|
| 1 | Positions missing from dashboard | `git stash` hid `live_state.json` | Generate HTML on main before branch switch |
| 2 | Yahoo prices not updating | CORS proxy caching | Cache-busting `&_=timestamp` + `no-store` headers |
| 3 | Chart flickering on refresh | TradingView iframe recreated every 5s | Track `renderedChartSymbol`, skip if unchanged |
| 4 | CRWD stuck at close price | v8/chart `marketState: null` overwrote server's PRE | Merge client quotes with server pre/post data |
| 5 | preChangePct showing -37% | yfinance raw decimal × 100 twice | Sanity check, recompute if \|%\| > 10 |
| 6 | **Only 4/60 symbols loaded** | **Cache from old universe, 7% coverage** | **`min_coverage=80%` threshold in `fetch_data()`** |
| 7 | **AH prices show regular close** | **`meta.postMarketPrice` null during AH** | **Derive from last bar close when timestamp >= 16:00 ET** |

---

## Future Phases

### Phase 4: Algorithm Deep Dive & Optimization
- Execution timing analysis (3:55 PM vs last hour vs intraday)
- Capital efficiency improvement (currently ~1.8% avg exposure)
- ML signal filter (binary classifier for signal quality)

### Phase 5: Multivariate / Cross-Sectional
- Pairs trading, factor-augmented signals, ensemble

### Phase 6: Live Deployment
- Shadow → live migration when validated
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

**Key Reminders**:
- Dashboard pushes to `dashboard-live` branch, NOT `main`
- HTML generated on `main` (where data lives), then written to `dashboard-live`
- All strategy params in `config.yaml` — no hardcoded values
- `fetch_data()` has `min_coverage=80%` check — prevents stale cache mismatches
- Server-side yfinance + bar-data fallback provides pre/post market prices; client CORS is enhancement only
- Codespace venv: `source venv/bin/activate` (may need recreation after sessions)
- `src/trading/` package is shared between notebook and main_trader.py — changes affect both
