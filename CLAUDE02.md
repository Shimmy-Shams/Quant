# CLAUDE02 — Codespaces Development Context

> **Last Updated**: 2026-02-18 (Phase 3D — Dashboard Complete)
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
ssh -i ~/.ssh/id_rsa ubuntu@40.233.100.95 "sudo systemctl stop quant-trader && sudo su - trader -c 'cd ~/Quant && git fetch origin && git reset --hard origin/main' && sudo su - trader -c 'cd ~/Quant && /home/trader/Quant/venv/bin/python src/main_trader.py --update-dashboard-only' && sudo systemctl start quant-trader"

# 3. Verify
ssh -i ~/.ssh/id_rsa ubuntu@40.233.100.95 "sudo systemctl status quant-trader | head -6"
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

### File Inventory (as of 2026-02-18)

| File | LOC | Purpose |
|------|-----|---------|
| `src/dashboard_generator.py` | ~1260 | HTML dashboard: yfinance server quotes + client-side CORS refresh |
| `src/main_trader.py` | ~840 | Headless 24/7 trader, dashboard push, equity tracking |
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

### Key Data Files
```
data/snapshots/
├── live_state.json         # Account, positions, trades (from Alpaca API)
├── equity_history.json     # Accumulated equity snapshots for chart (max 2000)
├── shadow_state.csv        # Shadow position persistence
├── alpaca_cache/           # Cached Alpaca data (per-symbol parquets)
└── trading_logs/           # Replay/shadow CSV logs

docs/
└── index.html              # Generated dashboard (NEVER edit manually)
```

---

## Dashboard System (Phase 3D — Complete)

### Architecture: Two-Layer Price System

**Layer 1 — Server-side (reliable, yfinance)**:
- `_fetch_server_quotes()` in `dashboard_generator.py` uses `yfinance` to fetch prices
- Returns: `price`, `marketState` (PRE/POST/REGULAR), `prePrice`, `postPrice`, `previousClose`, etc.
- Each symbol fetched independently with retry (not batched `Tickers()`)
- Embedded in HTML as `DATA.server_quotes` — available instantly on page load
- Sanity check on pre/post changePct (recomputes if |%| > 10, fixes yfinance decimal bug)

**Layer 2 — Client-side (enhancement, CORS proxy)**:
- JavaScript fetches Yahoo v8/chart endpoint via CORS proxies every 5s
- CORS proxies: `corsproxy.io` → `api.allorigins.win` (fallback)
- **Merges** with server quotes instead of replacing them:
  - v8/chart returns `marketState: null` and no pre/post prices
  - Client inherits `marketState`, `prePrice`, `postPrice` from server quotes
  - Client updates `price`, `change`, `volume` from CORS response
- If CORS fails entirely, server quotes still power the display

### Dynamic Price Selection (`effectivePrice()`)
```javascript
function effectivePrice(symbol, quote) {
    const state = (quote.marketState || '').toUpperCase();
    if (['POST','POSTPOST','CLOSED'].includes(state) && quote.postPrice) return quote.postPrice;
    if (['PRE','PREPRE'].includes(state) && quote.prePrice) return quote.prePrice;
    return quote.price;  // regular market hours
}
```
Automatically switches between pre-market ($412.72) → regular ($414.29) → after-hours based on `marketState`.

### Dashboard Features (All Complete)
| Feature | Description |
|---------|-------------|
| **Account metrics** | Portfolio value, cash, day P&L, total return, open positions, trades |
| **Market status badge** | Pre-Market / Market Open / After Hours / Closed with color |
| **Position table** | Live prices, entry, P&L, day change, market value, % portfolio |
| **Pre/post market prices** | Shows PM/AH badge with effective price under "Current" column |
| **Price flash animation** | Blue pulse on cells when prices change between refreshes |
| **Watchlist dropdown** | `<select>` dropdown to pick symbol for TradingView chart (scalable) |
| **TradingView charts** | 450px iframe, only re-renders on symbol change (no flicker) |
| **Equity curve** | Chart.js with time range toggles: All / 1Y / 3M / 1M / 1D |
| **Recent trades** | Last N trades with side, price, date, P&L |
| **Auto-refresh** | 5s interval with countdown timer and live status indicator |

### Key Dashboard Bugs Fixed (This Session)
1. **Positions not showing** — `git stash` during branch switch hid `live_state.json`. Fix: generate HTML on main first, save to memory.
2. **Yahoo prices not updating** — CORS proxies caching responses. Fix: cache-busting `&_=timestamp` + `no-store` headers.
3. **Chart flickering on refresh** — TradingView iframe re-created every 5s. Fix: track `renderedChartSymbol`, skip if unchanged.
4. **CRWD stuck at close price** — v8/chart returns `marketState: null`, overwrote server's `PRE` state. Fix: merge client quotes with server's pre/post data.
5. **preChangePct showing -37%** — yfinance returns raw decimal, our code multiplied by 100 again. Fix: sanity check, recompute if |%| > 10.
6. **Toggle buttons cluttering** — Replaced watchlist toggle buttons with `<select>` dropdown.

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

### 20-Year Backtest Results
| Metric | Value |
|--------|-------|
| **Total Return** | 4,198.21% |
| **Annualized Return** | 20.72% CAGR |
| **Sharpe Ratio** | 3.75 |
| **Sortino Ratio** | 6.41 |
| **Max Drawdown** | 1.25% |
| **Win Rate** | 98.38% (789W / 13L) |
| **Profit Factor** | 125.74 |

---

## Live Trading Status (Alpaca Paper)

### Account (as of 2026-02-18)
- **Portfolio Value**: ~$999,569
- **Cash**: ~$903,912
- **Mode**: Live (paper trading with real-time data)

### Current Positions
| Symbol | Side | Qty | Entry | Current |
|--------|------|-----|-------|---------|
| CRWD | Long | 232 | $413.96 | ~$412.72 (PM) |
| ETHUSD | Short | 0 | $1,989.40 | ~$2,018 |

### VM Service
- **Service**: `quant-trader.service` (systemd, enabled)
- **User**: `trader` (dedicated, non-root)
- **Venv**: `/home/trader/Quant/venv/`
- **Python**: `/home/trader/Quant/venv/bin/python`
- **Execution Window**: 9:35 AM → 3:55 PM ET

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
| **3D** | **Feb 17-18** | **Dashboard complete: server-side yfinance + client CORS merge** |

---

## Future Phases

### Phase 4: ML Signal Filter
- Feature engineering from price/volume/volatility
- Binary classifier: "Will this signal be profitable?"

### Phase 5: Multivariate / Cross-Sectional
- Pairs trading, factor-augmented signals, ensemble

### Phase 6: Live Deployment
- Shadow → live migration when validated
- Capital efficiency (3.89% exposure → 10-20% target)

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
- Server-side yfinance provides pre/post market prices; client CORS is enhancement only
- Codespace venv: `source venv/bin/activate` (may need recreation after sessions)
