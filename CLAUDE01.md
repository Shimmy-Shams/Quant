# Quantitative Trading System - Local Environment Context

**Environment**: Local VS Code (Windows)  
**Claude Instance**: CLAUDE01 (Local)  
**Paired With**: CLAUDE02.md (Codespaces - for development)  
**Last Updated**: 2026-02-15 (end of 20-year data collection)

---

## Project Overview
Building a live algorithmic trading bot with Interactive Brokers paper trading integration. Focus on quantitative swing trading (days-weeks) on US equities + options, using mean reversion, ML filtering, and volatility strategies. Canadian regulatory requirements apply (Ontario).

**Team**: Developer (Computer Engineering background, Risk Analyst at RBC, Ontario, Canada)

## Current Status
- TWS connected and verified (Paper Trading, port 7497)
- Phase 1 COMPLETE -- IB data infrastructure built (2 years data)
- Phase 2 COMPLETE (Codespaces/CLAUDE02) -- Mean reversion engine built, optimized (+27.2% return, Sharpe 0.82)
- Phase 2B COMPLETE (Local) -- Yahoo Finance 20-year data collection for robust backtesting
- Account is in CAD -- code handles multi-currency
- Ready for extended backtesting with 20 years of historical data

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

**Current Phase**: Extended Backtesting with 20-year data (next)

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

### Phase 3: ML Filter
- [ ] Feature engineering from market data
- [ ] Train classifier to filter mean reversion signals
- [ ] Compare filtered vs unfiltered performance
- [ ] Deep research on model selection

### Phase 4: Options Strategy Layer
- [ ] IV rank/percentile calculations
- [ ] Strategy selection logic (stock vs options execution)
- [ ] Options spread builder (verticals, iron condors)

### Phase 5: Risk & Integration
- [ ] Position sizing engine (Kelly criterion / fixed-fractional)
- [ ] Portfolio-level risk monitoring & correlation checks
- [ ] Full pipeline: signal -> filter -> execute -> monitor

## Hybrid Architecture (Data Flow)
```
LOCAL (this PC)                        CLOUD (Codespaces / GitHub)
+---------------------+               +----------------------+
| IB Gateway          |               | Strategy Dev         |
| DataCollector       |-- git push -> | Backtesting          |
|  -> Saves Parquet   |               | ML Training          |
| DataStreamer        |               | Reads Parquet files  |
+---------------------+               +----------------------+
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
├── .env / .env.example              # Configuration
├── requirements.txt                 # Python dependencies (includes yfinance>=0.2.32)
├── data/                            # Parquet storage (gitignored for large files)
│   ├── historical/daily/*.parquet   # OHLCV bars (20 years from Yahoo, 2 years from IB)
│   ├── snapshots/options/*.parquet  # Options chains
│   └── universe/*.parquet           # Dynamic universe snapshots
├── docs/
│   ├── IB_SETUP.md                  # IB Gateway/TWS setup guide
│   └── TWS_SETUP.md                 # TWS-specific instructions
├── src/
│   ├── config/
│   │   ├── config.py                      # Config class (.env loader)
│   │   └── data_collection_config.yaml    # Data collection parameters (NEW)
│   ├── connection/ib_connection.py  # Smart reconnect, asyncio patching
│   ├── data/
│   │   ├── universe_builder.py      # Single source: lists + functions + UniverseBuilder
│   │   ├── collector.py             # IB historical data downloader (2 years max)
│   │   ├── yahoo_collector.py       # Yahoo Finance downloader (20 years, NEW)
│   │   ├── streamer.py              # Real-time quote streaming
│   │   └── options.py               # Options chain collector
│   ├── strategies/                  # Future: Mean reversion, ML filter
│   ├── backtest/                    # Future: Backtesting engine
│   ├── execution/                   # Future: Order management
│   ├── main.ipynb                   # TWS-connected workflow
│   ├── main_data_collector.ipynb    # Data collection workflow (Yahoo/IB, NEW)
│   └── test_connection.py           # Standalone connection test
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

