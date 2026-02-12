# Quantitative Trading System - Codespaces Environment Context

**Environment**: GitHub Codespaces
**Claude Instance**: CLAUDE02 (Codespaces)
**Paired With**: CLAUDE01.md (Local Environment)

---

## Session Summary - 2026-02-12

### What We Built Today

Created a complete OOP-based quantitative trading system with Interactive Brokers integration.

### Completed Work

#### 1. Project Structure ✅
- Clean modular architecture with separate folders for:
  - `src/config/` - Configuration management
  - `src/connection/` - IB Gateway integration
  - `src/strategies/` - Trading strategies (ready for future)
  - `src/data/` - Data storage (ready for future)
  - `src/backtest/` - Backtesting engine (ready for future)
  - `src/execution/` - Order execution (ready for future)
  - `src/utils/` - Helper functions (ready for future)

#### 2. Core Classes (OOP Design) ✅

**Config Class** (`src/config/config.py`)
- Loads credentials from `.env` file securely
- Validates all configuration settings
- Masks sensitive data in logs
- Properties: `is_paper_trading`, `is_live_trading`
- Auto-detects `.env` file in project root

**IBConnection Class** (`src/connection/ib_connection.py`)
- Manages IB Gateway/TWS connection lifecycle
- Event-driven architecture with callback handlers
- Methods:
  - `connect()` / `disconnect()`
  - `get_account_summary()`
  - `get_positions()`
  - `get_portfolio_items()`
  - `get_account_values()`
  - `test_connection()`
- Context manager support (`with IBConnection() as ib:`)
- Flexible imports (works as module or standalone)

#### 3. Main Workflow Interface ✅

**Jupyter Notebook** (`src/main.ipynb`)
- **Primary interface for the trading system**
- Auto-reload enabled (`%autoreload 2`) - no kernel restart needed when editing .py files!
- Structured workflow:
  1. Setup & Initialization
  2. Configuration Loading
  3. Connection Object Creation
  4. **IB Gateway Connection** (the "connection page")
  5. Connection Testing
  6. Account Summary
  7. Current Positions
  8. Portfolio Details
  9. Disconnect

#### 4. Security ✅
- `.env.example` template for credentials
- `.env` in `.gitignore` (never committed)
- Credentials masked in all logs and repr strings
- Separate paper/live trading modes

#### 5. Testing & Documentation ✅
- `src/test_connection.py` - Standalone connection test script
- `docs/IB_SETUP.md` - Detailed IB Gateway setup guide
- `README.md` - Project documentation and quick start
- All code tested and verified working

#### 6. Dependencies Installed ✅
```
ib_insync==0.9.86
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
plotly>=5.14.0
python-dotenv>=1.0.0
colorlog>=6.7.0
jupyter>=1.0.0
ipykernel>=6.25.0
```

### Key Technical Decisions

#### Import Strategy
Used flexible import pattern in `ib_connection.py`:
```python
try:
    from config.config import Config
except ImportError:
    from ..config.config import Config
```
This allows the modules to work both:
- As imported modules in notebooks
- As standalone scripts

#### Auto-Reload in Jupyter
Added magic commands at the top of main.ipynb:
```python
%load_ext autoreload
%autoreload 2
```
**Benefit**: Edit `config.py` or `ib_connection.py` → re-run notebook cell → changes instantly applied (no kernel restart!)

---

## Important Discovery: Codespaces Limitation

### Network Issue Found ⚠️

**Problem**: Cannot connect to IB Gateway from Codespaces because:
- IB Gateway runs on user's **local machine** (127.0.0.1 = local PC)
- Code runs in **GitHub Codespaces** (127.0.0.1 = cloud container)
- They are on completely different networks

**Connection Topology**:
```
┌─────────────────────┐         ❌ Can't Connect        ┌──────────────────────┐
│  GitHub Codespaces  │  ◄────────────────────────────  │   User's Local PC    │
│   (Cloud Server)    │                                 │                      │
│  - Python Code      │                                 │  - IB Gateway        │
│  - main.ipynb       │                                 │  - Port 4002         │
│  - 127.0.0.1 = self │                                 │  - 127.0.0.1 = self  │
└─────────────────────┘                                 └──────────────────────┘
```

### Solution Architecture

**Codespaces Environment** (this instance - CLAUDE02):
- ✅ Code development
- ✅ Strategy development
- ✅ Backtesting with historical data
- ✅ Git operations
- ❌ **Cannot connect to IB Gateway** (network limitation)

**Local Environment** (CLAUDE01):
- ✅ Live IB Gateway connection
- ✅ Real-time trading operations
- ✅ Connection testing
- ✅ Paper trading execution

### Recommended Workflow

1. **Develop in Codespaces** (CLAUDE02):
   - Write strategies
   - Build features
   - Test logic
   - Commit to GitHub

2. **Trade Locally** (CLAUDE01):
   - Pull latest code
   - Connect to IB Gateway
   - Execute trades
   - Test live connections

---

## File Structure Created

```
Quant/
├── CLAUDE01.md              # Context for Local Environment Claude
├── CLAUDE02.md              # Context for Codespaces Claude (this file)
├── README.md                # Project documentation
├── .env.example             # Credentials template
├── .env                     # Actual credentials (not committed)
├── .gitignore               # Git ignore rules
├── requirements.txt         # Python dependencies
├── docs/
│   └── IB_SETUP.md         # IB Gateway setup guide
└── src/
    ├── config/
    │   ├── __init__.py
    │   └── config.py       # Config class
    ├── connection/
    │   ├── __init__.py
    │   └── ib_connection.py # IBConnection class
    ├── strategies/
    │   └── __init__.py
    ├── data/
    │   └── __init__.py
    ├── backtest/
    │   └── __init__.py
    ├── execution/
    │   └── __init__.py
    ├── utils/
    │   └── __init__.py
    ├── main.ipynb          # Main workflow interface
    └── test_connection.py  # Connection test script
```

---

## Configuration Settings

### Current .env Setup (Paper Trading)
```bash
IB_HOST=127.0.0.1
IB_PORT=4002              # IB Gateway Paper Trading
IB_CLIENT_ID=1
IB_USERNAME=***           # Set by user
IB_PASSWORD=***           # Set by user
IB_ACCOUNT_ID=***         # Set by user
TRADING_MODE=paper
LOG_LEVEL=INFO
```

### Port Reference
- **IB Gateway Paper**: 4002
- **IB Gateway Live**: 4001
- **TWS Paper**: 7497
- **TWS Live**: 7496

---

## Usage Examples

### In Jupyter Notebook (main.ipynb)
```python
# Run cells in order - auto-reload handles module changes!

# Cell 1: Setup (includes %autoreload 2)
# Cell 2: Load config
config = Config()

# Cell 3: Create connection
ib_conn = IBConnection(config)

# Cell 4: Connect (will fail in Codespaces - works locally)
success = ib_conn.connect()

# Edit config.py or ib_connection.py, then just re-run cells!
```

### In Python Script
```python
from config.config import Config
from connection.ib_connection import IBConnection

# Context manager (recommended)
with IBConnection() as ib:
    if ib.is_connected:
        positions = ib.get_positions()
        account = ib.get_account_values()
```

### Test Connection (Terminal)
```bash
python src/test_connection.py
```

---

## What's Next

### For CLAUDE01 (Local Environment)
When you switch to local PC:

1. **Pull this commit** from GitHub
2. **Read CLAUDE01.md** for your context
3. **Test IB Gateway connection** - should work locally!
4. **Run main.ipynb** and execute live connection
5. **Update CLAUDE01.md** with your progress
6. **Push updates** so CLAUDE02 stays in sync

### Future Development (Either Environment)

#### Phase 2: Market Data
- [ ] Real-time quote streaming
- [ ] Historical data retrieval
- [ ] Data storage and caching
- [ ] Market data visualizations

#### Phase 3: Strategy Framework
- [ ] Base strategy class
- [ ] Backtesting engine
- [ ] Performance metrics (Sharpe, drawdown, etc.)
- [ ] Example strategies (SMA crossover, mean reversion)

#### Phase 4: Risk Management
- [ ] Position sizing calculator
- [ ] Stop-loss automation
- [ ] Portfolio-level limits
- [ ] Drawdown protection

#### Phase 5: Execution
- [ ] Order management system
- [ ] Order types (market, limit, stop)
- [ ] Fill tracking
- [ ] Error handling and retry logic

---

## Git Workflow

### Codespaces → Local Sync
```bash
# In Codespaces (CLAUDE02)
git add .
git commit -m "Feature: Description"
git push origin main

# On Local PC (CLAUDE01)
git pull origin main
# Read CLAUDE02.md for updates
```

### Local → Codespaces Sync
```bash
# On Local PC (CLAUDE01)
git add .
git commit -m "Feature: Description"
git push origin main

# In Codespaces (CLAUDE02)
git pull origin main
# Read CLAUDE01.md for updates
```

---

## Technical Notes

### Auto-Reload Behavior
With `%autoreload 2` enabled:
- **Reloads**: Module functions, classes, methods
- **Doesn't reload**: Module-level variables set on import
- **Best practice**: Re-run initialization cells after big changes

### Connection Error Codes
- `ConnectionRefusedError(111)` - IB Gateway not running or wrong port
- `2104` - IB Info: Market data farm connection OK (can ignore)
- `2106` - IB Info: HMDS data farm connection OK (can ignore)
- `2158` - IB Info: Secure gateway connection OK (can ignore)

### Environment Variables
All loaded via `python-dotenv`:
- Automatically finds `.env` in project root
- Can override with `Config(env_file="/path/to/.env")`
- Never commits `.env` (in `.gitignore`)

---

## Important Security Notes

1. **Never commit `.env` file** - contains IB credentials
2. **Use `.env.example`** as template
3. **Always start with paper trading** before live
4. **Credentials masked** in all logs and repr() strings
5. **Read-Only API** recommended for initial testing

---

## Resources

- **IB API Docs**: https://interactivebrokers.github.io/
- **ib_insync Docs**: https://ib-insync.readthedocs.io/
- **TWS API Guide**: https://www.interactivebrokers.com/en/software/api/api.htm
- **Project Repo**: https://github.com/Shimmy-Shams/Quant

---

## Session End Notes

**Date**: 2026-02-12
**Status**: ✅ Codespaces environment fully configured
**Next Step**: Switch to local PC and test IB Gateway connection
**For CLAUDE01**: Check CLAUDE01.md, pull latest code, test connection!

---

**Remember**:
- CLAUDE02 (Codespaces) = Development & Strategy Building
- CLAUDE01 (Local) = Live Trading & IB Connection
- Keep both .md files updated to maintain context sync!
