# Quantitative Trading System - Local Environment Context

**Environment**: Local VS Code (Windows)  
**Claude Instance**: CLAUDE01 (Local)  
**Paired With**: CLAUDE02.md (Codespaces - for development)  
**Last Updated**: 2026-02-26 (2-phase T+1 architecture, VM deployment)

---

## Project Overview
Building a live algorithmic trading bot with Interactive Brokers paper trading integration. Focus on quantitative swing trading (days-weeks) on US equities + options, using mean reversion, ML filtering, and volatility strategies. Canadian regulatory requirements apply (Ontario).

**Team**: Developer (Computer Engineering background, Risk Analyst at RBC, Ontario, Canada)

## Current Status
- TWS connected and verified (Paper Trading, port 7497)
- Phase 1 COMPLETE -- IB data infrastructure built (2 years data)
- Phase 2 COMPLETE (Codespaces/CLAUDE02) -- Mean reversion engine built, optimized (+27.2% return, Sharpe 0.82)
- Phase 2B COMPLETE (Local) -- Yahoo Finance 20-year data collection for robust backtesting
- Phase 3 COMPLETE (Local/VM) -- Live paper trading on Oracle Cloud VM with Alpaca
- Account is in CAD -- code handles multi-currency
- VM is LIVE: Oracle Cloud (40.233.100.95), systemd service `quant-trader`, running 24/7

**Completed - Phase 3: Live Trading & VM Deployment** (2026-02-26)
- Oracle Cloud VM deployed (Ubuntu, 40.233.100.95, SSH as `ubuntu`)
- Alpaca paper trading integration (replacing IB for headless VM operation)
- `main_trader.py` -- headless 24/7 trader with 2-phase T+1 architecture
- **2-Phase T+1 Architecture** (critical fix for zero-signal bug):
  - Phase 1 (Post-Close ~4:10 PM ET): Generate signals from Day T close data, cache to parquet
  - Phase 2 (9:35 AM T+1): Load cached signals, overlay live prices, execute trades
  - Intraday Monitor (09:45-15:50): Watch held positions for dynamic exits
  - Eliminates T+0 look-ahead bias; validated Sharpe 6.00 at T+1 vs 6.99 at T+0
- **Mode-aware signal cache**: Separate directories for `live/` and `shadow/` modes
  - Path: `data/snapshots/signal_cache/{live,shadow}/` (gitignored, environment-specific)
- **Manual signal trigger**: `--generate-signals` CLI flag for cache priming
  - Usage: `python main_trader.py --mode live --generate-signals`
- **Signal & trade history** (git-tracked for model improvement):
  - `data/snapshots/signal_history.json` -- daily signal snapshots (symbol, direction, strength, z-score, price)
  - `data/snapshots/trade_history.json` -- daily execution results (decisions, fills, portfolio value)
  - Both accumulate up to 365 days, deduplicated by date+mode+phase
- **Detailed signal logging**: Per-stock breakdown in VM logs (BUY/SELL, signal, z-score, price)
  - Also logs near-threshold signals (within 80% of threshold) for context
- **Intraday monitor tests**: 30 unit tests covering stop-loss, trailing stop, time-decay, circuit breaker, edge cases
- **VM architecture**:
  - Users: `ubuntu` (SSH login), `trader` (service runner), `opc` (Oracle default)
  - Service: `quant-trader.service` (systemd, auto-restart, 2GB memory limit, 50% CPU cap)
  - Service config: `/etc/systemd/system/quant-trader.service`
  - Codebase: `/home/trader/Quant/` with `venv/` (not `.venv`)
  - Env vars: `/home/trader/Quant/.env` (Alpaca API keys)
  - Git remote: HTTPS (`https://github.com/Shimmy-Shams/Quant.git`)
  - Dashboard push currently broken (HTTPS needs PAT for push; SSH deploy key needed)
- **Bug fixed**: Main loop ordering caused zero signals -- intraday monitor blocked until 15:50, past execution window
- Current paper account: ~$991K portfolio value, 4 open positions (ADBE, DY, LOW, RYAN)
- `.venv` created locally for testing (Python 3.11.1, all packages from requirements.txt + pytest)

**Completed - Phase 2B** (2026-02-15)
- Yahoo Finance data collector built (20-year historical data capability)
- Config-driven architecture: centralized data collection configuration
- Dynamic CPU detection: auto-detects available threads for parallel downloads
- TWS-optional collection: Yahoo mode doesn't require TWS connection
- 258/273 tickers collected successfully (20 years daily OHLCV each)
- IB-compatible Parquet format: seamless integration with Phase 2 mean reversion code
- 12 tickers failed (delisted stocks: ANSS, SMAR, CSWI, GMS, PPBI, WBA, HES, SGEN, CEIX, ARCH, TCS, ATVI)
- Collection time: 26.6 seconds with 8 workers (0.1s per ticker)
- Total data: ~5,032 bars per ticker (20 years) vs previous 2-year limit

**Completed - Phase 2** (2026-02-13, CLAUDE02/Codespaces)
- Mean reversion engine: z-score signals with dynamic thresholds
- Backtesting framework: position tracking, PnL calculation, metrics
- Parameter optimization: Optuna-based grid search (100+ trials)
- Critical bug fixes: 6 major issues (position sizing, signal timing, exits, slippage)
- Optimized parameters: window=65, z_entry=2.5, z_exit=0.3, max_hold=15
- Performance: +27.2% return, Sharpe 0.82, Max DD -8.7%, Win Rate 57.3%

**Completed - Phase 1** (2026-02-12)
- Dynamic universe built: 312 stocks across 4 indices (100/index, 30 for Dow)
- Historical data collected: 293 tickers, 2 years daily OHLCV, 8.4 MB Parquet
- Options snapshots: 8 files for ETFs (SPY, QQQ, DIA, IWM)
- Universe builder refactored: deduplication across indices (single-pass filtering)
- Options collector refactored: batch qualify + batch market data + ATM strike filtering
- All data pushed to GitHub for Codespaces access

**Current Phase**: ML Filter development (Phase 4) and model improvement using signal/trade history data

## Strategy Direction (CONFIRMED)

**Style**: Swing Trading (days to weeks)  
**Assets**: US Equities + Options combo  
**Risk**: Moderate  
**Indices**: S&P 500, Nasdaq 100, Dow 30, Russell 2000  
**No emojis in code** (project standard)

### Three-Layer Architecture
1. **Mean Reversion** (Signal Generation) — Z-score deviation from fair value
2. **ML Model** (Signal Filtering) — Classify which setups actually revert vs trend
3. **Options Overlay** (Execution & Hedging) — Defined-risk trades, vol arbitrage

### Future Deep-Dives (Deferred to Dev Phase)
- ML model selection & feature engineering — full research session planned
- Market microstructure strategies — order flow, bid/ask imbalance research

## Execution Plan

### Phase 1: Data Infrastructure (COMPLETE)
- [x] Build data collector module (Parquet storage)
- [x] Build data streamer module (real-time quotes)
- [x] Build options collector module (chains, Greeks, IV)
- [x] Build dynamic universe builder (single source of truth)
- [x] Define stock universe (100 per index, 30 for Dow + 4 ETFs)
- [x] Implement smart caching (skip existing data for efficiency)
- [x] B+C performance optimization (local Parquet filtering, no slow IB calls)
- [x] Build dynamic universe (312 stocks, top 100 per index)
- [x] Collect historical data (293 tickers, 2 years daily OHLCV)
- [x] Run options chain collection (ETFs: SPY, QQQ, DIA, IWM)
- [x] Deduplication: single-pass filtering across indices (no redundant lookups)
- [x] Options batch optimization (batch qualify + batch data + ATM strike filter)
- [x] Push data to Git for Codespaces backtesting access

**Dynamic Universe System:**
Multi-criteria filtering with live constituent fetching:
1. Stock volume > 1M shares/day (liquidity requirement)
2. Listed options with OI > 1K (options strategies viability)
3. Price range: $10-$500 (accessibility)
4. Market cap > $1B (quality threshold — confirmed)
5. Options bid-ask spread < 10% (execution efficiency)
6. Top 100 per index (S&P, Nasdaq, Dow=30, Russell) = ~330 unique stocks + 4 ETFs

**Performance (B+C Approach):**
- B: Curated lists are pre-filtered -- skip slow IB API calls entirely
- C: When local Parquet data exists, filter from it (seconds vs minutes)
- Result: Universe build goes from 30-60 min to under 30 seconds

**Architecture:**
- Single file: `src/data/universe_builder.py` (lists + functions + class)
- No separate universe.py -- everything in one place
- All modules import from `data.universe_builder`

**Next Actions (Phase 2 -- Codespaces):**
1. Open project in Codespaces
2. Load historical Parquet data (already in repo)
3. Build mean reversion signals (z-score based)
4. Backtest on 2-year historical data
5. Parameter optimization for swing timeframe

### Phase 2: Mean Reversion Engine (COMPLETE -- Codespaces/CLAUDE02)
- [x] Statistical mean reversion signals (z-score based)
- [x] Backtest on 2-year historical data (293 tickers)
- [x] Parameter optimization for swing timeframe
- [x] Critical bug fixes (position sizing, signal timing, exits, slippage)
- [x] Performance metrics: +27.2% return, Sharpe 0.82, Max DD -8.7%

### Phase 2B: Extended Historical Data Collection (COMPLETE -- Local)
- [x] Yahoo Finance data collector module (20-year capability vs IB's 2-year limit)
- [x] Config-driven architecture (centralized data_collection_config.yaml)
- [x] Dynamic CPU detection (auto worker count optimization)
- [x] TWS-optional collection (Yahoo mode runs without IB connection)
- [x] IB-compatible Parquet format (seamless integration with existing code)
- [x] Smart caching (skips tickers with 10+ years existing data)
- [x] 258 tickers collected: 20 years daily OHLCV (~5,032 bars each)
- [x] Collection performance: 26.6s total (0.1s per ticker, 8 workers)

**Yahoo Finance Collector Architecture:**
- Module: `src/data/yahoo_collector.py` (336 lines)
- Config: `src/config/data_collection_config.yaml` (centralized parameters)
- Notebook: `src/main_data_collector.ipynb` (setup, test, full collection, status)
- Features:
  - Parallel downloads via ThreadPoolExecutor (dynamic worker count)
  - Identical Parquet schema to IB collector (OHLCV columns match)
  - Configurable source toggle: "yahoo" (20y) or "ib" (2y)
  - Universe selection: "dynamic" (273 tickers), "etf" (4), or "custom"
  - Auto-detects CPU cores: `max_workers: "auto"` -> `os.cpu_count()`
  - Smart caching: `min_years: 10` threshold (skips re-download)
- Data Quality:
  - 258/273 successful downloads (94.5% success rate)
  - Failed tickers: delisted stocks (ANSS, SMAR, CSWI, GMS, PPBI, WBA, HES, SGEN, CEIX, ARCH, TCS, ATVI)
  - Average: ~5,032 bars per ticker (20 years daily data)
  - Date range: 2006-02-15 to 2026-02-15 (for most tickers)

**Next Action:** Push 20-year data to GitHub, run extended backtest in Codespaces to validate mean reversion engine across 20-year period

### Phase 3: Live Trading & VM Deployment (COMPLETE -- Local/VM)
- [x] Oracle Cloud VM setup (Ubuntu, systemd service, auto-restart)
- [x] Alpaca paper trading integration (headless, no TWS dependency)
- [x] 2-phase T+1 architecture (post-close signal gen + morning execution)
- [x] Mode-aware signal cache (separate live/shadow parquet directories)
- [x] Manual signal trigger (`--generate-signals` CLI flag)
- [x] Persistent signal & trade history (git-tracked JSON for model improvement)
- [x] Detailed per-stock signal logging (direction, strength, z-score, price)
- [x] Intraday monitor (stop-loss, trailing stop, time-decay, circuit breaker)
- [x] 30 unit tests for intraday monitor (pytest)
- [x] Dashboard generator with GitHub Pages push
- [x] Equity history tracking (seeded from Alpaca portfolio history API)
- [x] Bracket order retrofit for unprotected positions

### Phase 4: ML Filter
- [ ] Feature engineering from market data
- [ ] Train classifier to filter mean reversion signals
- [ ] Compare filtered vs unfiltered performance
- [ ] Deep research on model selection
- [ ] Use signal_history.json + trade_history.json for training data

### Telegram Notifications (COMPLETE -- Codespaces, 2026-02-28)
- [x] Telegram Bot created (BotFather), token + chat ID stored in `.env`
- [x] `src/notifications/telegram_notifier.py` module (zero external dependencies, uses urllib)
- [x] Singleton pattern via `get_notifier()` -- lazily initializes from env vars, returns None if not configured
- [x] Integrated into `main_trader.py` at 7 hook points:
  - **Startup**: Service started notification (mode, PID, universe size)
  - **Phase 1 (Signal Generation)**: Signals generated summary (valid count, actionable count, per-symbol details)
  - **Phase 2 (Trade Execution)**: Trade results (symbol, action, qty, price, status, portfolio value)
  - **Daily Summary**: Post-close summary (portfolio value, cash, day P&L, open positions)
  - **Errors**: Signal generation failures + main loop exceptions
  - **Shutdown**: Service stopped notification (cycles completed)
- [x] Integrated into `src/execution/intraday_monitor.py`:
  - **Intraday Exits**: Per-exit notification (symbol, side, reason, entry/exit price, P&L %)
  - Covers: stop-loss, trailing stop, time-decay, circuit breaker exits
- [x] All sends are fire-and-forget (catch + log errors, never block trading logic)
- [x] Tested: 3 real messages sent successfully to Telegram
- **Env vars** (in `.env`, both Codespaces and VM):
  - `TELEGRAM_BOT_TOKEN` -- Bot API token from BotFather
  - `TELEGRAM_CHAT_ID` -- User's chat ID (2102346549)
- **VM deployment**: After `git push` + `git pull` on VM, add `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` to `/home/trader/Quant/.env`, then restart service

### Phase 5: Options Strategy Layer
- [ ] IV rank/percentile calculations
- [ ] Strategy selection logic (stock vs options execution)
- [ ] Options spread builder (verticals, iron condors)

### Phase 6: Risk & Integration
- [ ] Position sizing engine (Kelly criterion / fixed-fractional)
- [ ] Portfolio-level risk monitoring & correlation checks
- [ ] Full pipeline: signal -> filter -> execute -> monitor

## VM Operations Quick Reference
```bash
# SSH to VM
ssh -i ~/.ssh/id_rsa ubuntu@40.233.100.95

# Service management
sudo systemctl status quant-trader
sudo systemctl stop quant-trader
sudo systemctl start quant-trader
sudo systemctl restart quant-trader

# View logs
sudo journalctl -u quant-trader --since '10 min ago' --no-pager | tail -40
sudo journalctl -u quant-trader -f  # follow live

# Manual signal generation (run after market close to prime cache)
sudo -u trader bash -c 'cd /home/trader/Quant/src && /home/trader/Quant/venv/bin/python main_trader.py --mode live --generate-signals'

# Pull latest code to VM
sudo -u trader bash -c 'cd /home/trader/Quant && git pull origin main'

# Check signal cache
sudo -u trader ls -la /home/trader/Quant/data/snapshots/signal_cache/live/
sudo -u trader cat /home/trader/Quant/data/snapshots/signal_cache/live/metadata.json
```

## main_trader.py CLI Reference
```bash
python main_trader.py                              # Shadow mode (default), daily loop
python main_trader.py --mode live                   # Live paper trading, daily loop
python main_trader.py --mode live --generate-signals # Manual signal gen, exit after
python main_trader.py --mode live --once             # Single full cycle (legacy), exit
python main_trader.py --update-dashboard-only        # Refresh dashboard, exit
python main_trader.py --mode shadow --interval 300   # Shadow mode, 5-min cycle
```

## Hybrid Architecture (Data Flow)
```
LOCAL (this PC)                        CLOUD (Codespaces / GitHub)
+---------------------+               +----------------------+
| IB Gateway          |               | Strategy Dev         |
| DataCollector       |-- git push -> | Backtesting          |
|  -> Saves Parquet   |               | ML Training          |
| DataStreamer        |               | Reads Parquet files  |
+---------------------+               +----------------------+
         |
         | git push
         v
+---------------------+
| VM (Oracle Cloud)   |
| main_trader.py      |
|  Phase 1: Signal Gen|
|  Phase 2: Execution |
|  Intraday Monitor   |
| signal_history.json |-- git push -> GitHub (for model improvement)
| trade_history.json  |
+---------------------+
```

## Notebook Structure
src/main.ipynb — 11 main sections:
1. Setup & Config — Imports, config, connection object
2. Connect to TWS — Smart reconnect
3. Test Connection — Server verification
4. Account Overview — Summary + Positions + Portfolio
5. Data Infrastructure — 6 subsections:
   - Init modules (collector, streamer)
   - 5a. Build dynamic universe (multi-criteria filtering, top 100/index)
   - 5b. Collect daily historical data (3 options: ETFs, curated, or dynamic)
   - 5c. View stored data status
   - 5d. Real-time quotes (optional, requires market data subscription)
   - 5e. Options chain collection (Greeks, IV, expirations)
6. Disconnect — Cleanup streams & disconnect

**Data Collection Options:**
- Option A: ETFs only (4 tickers, ~1 min) -- Quick test
- Option B: Full universe (curated lists, ~25-30 min) -- Static lists
- Option C: Dynamic universe (~312 stocks, smart caching) -- Filtered quality stocks

**Data Collected:**
- 293 tickers with 2-year daily OHLCV (8.4 MB Parquet)
- 8 options snapshots for ETFs (SPY, QQQ, DIA, IWM)
- Universe snapshots for all 4 indices

## Project Structure
```
Quant/
├── CLAUDE01.md / CLAUDE02.md        # Context & planning docs
├── .env / .env.example              # Configuration (Alpaca API keys)
├── config.yaml                      # Strategy & trading configuration
├── requirements.txt                 # Python dependencies
├── .venv/                           # Local virtual environment (gitignored)
├── data/
│   ├── historical/daily/*.parquet   # OHLCV bars (20 years from Yahoo, 2 years from IB)
│   ├── logs/                        # Runtime logs (gitignored)
│   ├── snapshots/
│   │   ├── alpaca_cache/            # Alpaca data cache (parquet, git-tracked)
│   │   ├── signal_cache/            # T+1 signal cache (gitignored, env-specific)
│   │   │   ├── live/                #   Live mode signals
│   │   │   └── shadow/              #   Shadow mode signals
│   │   ├── signal_history.json      # Daily signal snapshots (git-tracked)
│   │   ├── trade_history.json       # Daily trade results (git-tracked)
│   │   ├── live_state.json          # Current positions/equity (git-tracked)
│   │   ├── equity_history.json      # Equity curve (git-tracked)
│   │   ├── intraday_equity.json     # 1D chart data (git-tracked)
│   │   ├── shadow_state.csv         # Shadow sim state (gitignored)
│   │   └── options/*.parquet        # Options chains
│   └── universe/*.parquet           # Dynamic universe snapshots
├── docs/
│   ├── index.html                   # Auto-generated dashboard
│   ├── IB_SETUP.md                  # IB Gateway/TWS setup guide
│   ├── TWS_SETUP.md                 # TWS-specific instructions
│   └── UNIVERSE_SYSTEM.md           # Universe system docs
├── src/
│   ├── main_trader.py               # Headless 24/7 trader (2-phase T+1 architecture)
│   ├── strategy_config.py           # ConfigLoader (config.yaml)
│   ├── dashboard_generator.py       # Static HTML dashboard generator
│   ├── test_connection.py           # Standalone connection test
│   ├── config/
│   │   ├── config.py                # Config class (.env loader)
│   │   └── data_collection_config.yaml
│   ├── connection/
│   │   ├── ib_connection.py         # IB TWS connection
│   │   └── alpaca_connection.py     # Alpaca connection (TradingMode: LIVE/SHADOW)
│   ├── data/
│   │   ├── universe_builder.py      # Universe: lists + functions + class
│   │   ├── collector.py             # IB historical data downloader
│   │   ├── yahoo_collector.py       # Yahoo Finance downloader (20 years)
│   │   ├── alpaca_data.py           # Alpaca data adapter (pipeline data)
│   │   ├── streamer.py              # Real-time quote streaming
│   │   └── options.py               # Options chain collector
│   ├── strategies/
│   │   └── mean_reversion.py        # Z-score signal generation (862 lines)
│   ├── backtest/
│   │   ├── engine.py                # BacktestConfig, backtesting framework
│   │   └── optimizer.py             # Optuna-based parameter optimization
│   ├── execution/
│   │   ├── alpaca_executor.py       # Live order execution
│   │   ├── simulation.py            # Shadow/paper simulation engine
│   │   └── intraday_monitor.py      # Dynamic exit monitor (712 lines)
│   ├── trading/
│   │   └── pipeline.py              # Universe select, data fetch, signal gen pipeline
│   ├── tests/
│   │   ├── __init__.py
│   │   └── test_intraday_monitor.py # 30 tests (stop-loss, trailing, time-decay, etc.)
│   ├── main.ipynb                   # TWS-connected workflow
│   ├── main_data_collector.ipynb    # Data collection workflow
│   └── main_mean_reversion.ipynb    # Strategy development notebook
```

## Key Technical Details
- Auto-Reload: Edit .py -> rerun cell (no kernel restart)
- **TWS Ports:** Paper=7497, Live=7496
- **IB Gateway Ports:** Paper=4002, Live=4001
- **Data Sources:**
  - IB TWS: 2 years daily historical max (11s rate limit between requests)
  - Yahoo Finance: 20+ years daily historical (parallel downloads, ~0.1s per ticker)
- Storage: Parquet via pyarrow (10-50x faster than CSV)
- **Data Collection Config:** src/config/data_collection_config.yaml
  - Source toggle: "yahoo" or "ib"
  - Period: "20y" (Yahoo) or "2 Y" (IB)
  - Workers: "auto" (detects CPU cores) or specific count
  - Universe: "dynamic" (273 tickers), "etf" (4), or "custom"
- Deps: ib_insync, yfinance, pandas, numpy, matplotlib, plotly, python-dotenv, pyarrow
- **Multi-currency:** Account values work with USD, CAD, EUR, GBP

## Important Reminders
- **No emojis** in any code output
- **Security**: Never commit credentials
- **Safety**: Paper trading first, small positions, kill switches
- **Canadian**: Comply with Ontario regulations, consult on tax implications
- **Workflow**: Research/Planning -> Confirmation -> Execution (always)

---
**Repo**: https://github.com/Shimmy-Shams/Quant

