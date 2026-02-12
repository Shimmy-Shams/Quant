# Quantitative Trading System - Local Environment Context

**Environment**: Local VS Code (Windows)  
**Claude Instance**: CLAUDE01 (Local)  
**Paired With**: CLAUDE02.md (Codespaces - for development)  
**Last Updated**: 2026-02-12 (end of Phase 1)

---

## Project Overview
Building a live algorithmic trading bot with Interactive Brokers paper trading integration. Focus on quantitative swing trading (days-weeks) on US equities + options, using mean reversion, ML filtering, and volatility strategies. Canadian regulatory requirements apply (Ontario).

**Team**: Developer (Computer Engineering background, Risk Analyst at RBC, Ontario, Canada)

## Current Status
- TWS connected and verified (Paper Trading, port 7497)
- Phase 1 COMPLETE -- all data infrastructure built, data collected, pushed to GitHub
- Account is in CAD -- code handles multi-currency
- Ready for Phase 2 (Mean Reversion Engine) on Codespaces

**Completed** (2026-02-12)
- Dynamic universe built: 312 stocks across 4 indices (100/index, 30 for Dow)
- Historical data collected: 293 tickers, 2 years daily OHLCV, 8.4 MB Parquet
- Options snapshots: 8 files for ETFs (SPY, QQQ, DIA, IWM)
- Universe builder refactored: deduplication across indices (single-pass filtering)
- Options collector refactored: batch qualify + batch market data + ATM strike filtering
- All data pushed to GitHub for Codespaces access

**Current Phase**: Phase 2 -- Mean Reversion Engine (next)

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

### Phase 2: Mean Reversion Engine (NEXT -- Codespaces)
- [ ] Statistical mean reversion signals (z-score based)
- [ ] Backtest on 2-year historical data (293 tickers available)
- [ ] Parameter optimization for swing timeframe

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
├── requirements.txt                 # Python dependencies
├── data/                            # Parquet storage (gitignored for large files)
│   ├── historical/daily/*.parquet   # OHLCV bars
│   ├── snapshots/options/*.parquet  # Options chains
│   └── universe/*.parquet           # Dynamic universe snapshots
├── docs/
│   ├── IB_SETUP.md                  # IB Gateway/TWS setup guide
│   └── TWS_SETUP.md                 # TWS-specific instructions
├── src/
│   ├── config/config.py             # Config class (.env loader)
│   ├── connection/ib_connection.py  # Smart reconnect, asyncio patching
│   ├── data/
│   │   ├── universe_builder.py      # Single source: lists + functions + UniverseBuilder
│   │   ├── collector.py             # Historical data downloader
│   │   ├── streamer.py              # Real-time quote streaming
│   │   └── options.py               # Options chain collector
│   ├── strategies/                  # Future: Mean reversion, ML filter
│   ├── backtest/                    # Future: Backtesting engine
│   ├── execution/                   # Future: Order management
│   ├── main.ipynb                   # Main workflow interface
│   └── test_connection.py           # Standalone connection test
```

## Key Technical Details
- Auto-Reload: Edit .py -> rerun cell (no kernel restart)
- **TWS Ports:** Paper=7497, Live=7496
- **IB Gateway Ports:** Paper=4002, Live=4001
- Storage: Parquet via pyarrow (10-50x faster than CSV)
- Rate Limit: 11 sec between IB historical requests
- Deps: ib_insync, pandas, numpy, matplotlib, plotly, python-dotenv, pyarrow
- **Multi-currency:** Account values work with USD, CAD, EUR, GBP

## Important Reminders
- **No emojis** in any code output
- **Security**: Never commit credentials
- **Safety**: Paper trading first, small positions, kill switches
- **Canadian**: Comply with Ontario regulations, consult on tax implications
- **Workflow**: Research/Planning -> Confirmation -> Execution (always)

---
**Repo**: https://github.com/Shimmy-Shams/Quant

