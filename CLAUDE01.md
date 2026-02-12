# Quantitative Trading System - Project Context

## Project Overview
Building a quantitative trading system with Interactive Brokers integration for algorithmic trading strategies.

## Current Status
- ✅ Git repository initialized
- ✅ Pushed to GitHub: https://github.com/Shimmy-Shams/Quant
- ✅ Organized temp files into `temp/` folder
- ✅ GitHub Codespaces environment set up
- ✅ Python dependencies installed (ib_insync, pandas, numpy, matplotlib, plotly)
- ✅ Project folder structure created with OOP design
- ✅ Secure credential management with .env configuration
- ✅ IBConnection class for IB Gateway integration
- ✅ Config class for managing settings
- ✅ Test connection script created
- ✅ Documentation and Jupyter notebook examples created
- ⏭️ Ready to connect to IB Gateway and test

## Development Environment
**Chosen Solution**: GitHub Codespaces
- **Free tier**: 60 hours/month on 2-core, 8GB RAM
- **Upgrade options**: 4-core ($0.36/hr) or 8-core ($0.72/hr) for intensive backtesting
- **Access**: VS Code in browser or desktop app
- **Auto-pause**: Saves hours when inactive

## Project Structure
```
Quant/
├── src/
│   ├── config/              # Configuration management
│   │   ├── __init__.py
│   │   └── config.py        # Config class for .env settings
│   ├── connection/          # IB Gateway connection
│   │   ├── __init__.py
│   │   └── ib_connection.py # IBConnection class
│   ├── strategies/          # Trading strategies (empty for now)
│   ├── data/               # Data storage (empty for now)
│   ├── backtest/           # Backtesting engine (empty for now)
│   ├── execution/          # Order execution (empty for now)
│   ├── utils/              # Helper functions (empty for now)
│   ├── main.ipynb          # Main Jupyter notebook with examples
│   └── test_connection.py  # Connection test script
├── docs/
│   └── IB_SETUP.md         # Detailed IB Gateway setup guide
├── requirements.txt         # Python dependencies
├── .env.example            # Template for credentials
├── .gitignore              # Git ignore patterns
└── CLAUDE.md               # This file - project context
```

## Original Project Plan

### Phase 1: Initial Setup & Interactive Brokers Integration
**Goal**: Establish connection to Interactive Brokers and verify data flow

1. **Environment Setup**
   - Install dependencies: `ib_insync`, `pandas`, `numpy`, `matplotlib`
   - Set up project structure with proper folders (data, strategies, backtest, etc.)

2. **Interactive Brokers Setup**
   - Install TWS (Trader Workstation) or IB Gateway
   - Configure API settings (enable API, set port 7497 for paper trading)
   - Create connection script using `ib_insync`
   - Test connection and verify account data retrieval

3. **Data Pipeline**
   - Implement real-time data fetching for stocks/options
   - Set up historical data retrieval functions
   - Create data storage structure (CSV or database)

### Phase 2: Strategy Development Framework
**Goal**: Build infrastructure for creating and testing trading strategies

1. **Strategy Base Class**
   - Create abstract strategy class with common methods
   - Implement signal generation interface
   - Add position sizing logic

2. **Backtesting Engine**
   - Build backtesting framework for historical data
   - Implement performance metrics (Sharpe ratio, max drawdown, etc.)
   - Create visualization tools for results

3. **Example Strategy Implementation**
   - Simple moving average crossover strategy
   - Mean reversion strategy
   - Momentum-based strategy

### Phase 3: Risk Management & Order Execution
**Goal**: Implement robust risk controls and order management

1. **Risk Management Module**
   - Position size calculator
   - Stop-loss and take-profit logic
   - Portfolio-level risk limits
   - Drawdown protection

2. **Order Management System**
   - Order placement functions (market, limit, stop orders)
   - Order tracking and status monitoring
   - Error handling and retry logic

3. **Paper Trading Integration**
   - Connect strategies to IB paper trading account
   - Real-time execution testing
   - Performance monitoring dashboard

### Phase 4: Live Trading & Monitoring
**Goal**: Deploy strategies to live trading with proper monitoring

1. **Live Trading Deployment**
   - Paper trading validation (minimum 2-4 weeks)
   - Gradual transition to live with small positions
   - Multi-strategy portfolio management

2. **Monitoring & Alerts**
   - Real-time P&L tracking
   - Email/SMS alerts for critical events
   - Daily performance reports
   - System health monitoring

3. **Logging & Analysis**
   - Comprehensive trade logging
   - Performance analytics
   - Strategy optimization based on live results

## Changes to Original Plan
- **Before IB setup**: Setting up cloud development environment (GitHub Codespaces)
- **Reason**: User's local machine is older/slower, needs more powerful development environment
- **Decision**: Use Codespaces for development + occasional bot testing

## Technology Stack
- **Language**: Python 3.x
- **IB Integration**: `ib_insync` library
- **Data Analysis**: `pandas`, `numpy`
- **Visualization**: `matplotlib`, `plotly`
- **Backtesting**: Custom framework (to be built)
- **Development**: Jupyter notebooks + Python scripts
- **Version Control**: Git + GitHub
- **Dev Environment**: GitHub Codespaces

## Next Steps (In Order)

### Immediate (In Codespaces)
1. Set up Python environment in Codespaces
2. Install required dependencies (`ib_insync`, `pandas`, `numpy`, etc.)
3. Create project folder structure:
   ```
   src/
   ├── strategies/      # Trading strategies
   ├── data/           # Data storage
   ├── backtest/       # Backtesting engine
   ├── execution/      # Order execution
   └── utils/          # Helper functions
   ```

### Interactive Brokers Setup
4. Download and install TWS or IB Gateway (on local machine or Windows VPS if needed)
5. Configure API settings in TWS
6. Create IB connection test script
7. Verify real-time and historical data retrieval

### Strategy Development
8. Build backtesting framework
9. Implement first simple strategy (moving average crossover)
10. Run backtests and analyze results

## Important Notes

### Security Considerations
- **Never commit API keys, credentials, or account numbers to Git**
- Use `.env` files for sensitive data (already in `.gitignore`)
- Keep trading configuration separate from code

### Trading Considerations
- **Always start with paper trading** before live money
- Test extensively with small positions
- Monitor performance continuously
- Have kill switches and risk limits in place

### GitHub Repository
- Repository URL: https://github.com/Shimmy-Shams/Quant
- Visibility: Private (recommended for trading code)
- Branch: `main`

## Questions to Address Later
1. What types of trading strategies are you most interested in? (day trading, swing trading, options, etc.)
2. What's your risk tolerance and position sizing approach?
3. Do you have Interactive Brokers account already set up?
4. What markets/instruments do you want to trade? (stocks, options, futures, forex)

## Resources
- Interactive Brokers API: https://interactivebrokers.github.io/
- ib_insync Documentation: https://ib-insync.readthedocs.io/
- TWS API Guide: https://www.interactivebrokers.com/en/software/api/api.htm

---

**Last Updated**: 2026-02-12
**Project Status**: Codespaces environment ready, IB connection framework built
**Next Action**: Set up IB Gateway locally and test connection using `python src/test_connection.py`

## Setup Completed

### Python Environment
- ✅ Python 3.12.1 with pip 25.3
- ✅ All dependencies installed from requirements.txt
- ✅ Jupyter notebooks enabled

### OOP Architecture
- ✅ **Config class** (`src/config/config.py`): Manages .env file loading and validation
- ✅ **IBConnection class** (`src/connection/ib_connection.py`): Handles IB Gateway connection lifecycle
- ✅ Modular structure with proper Python packages

### Security
- ✅ `.env.example` template for credentials
- ✅ `.env` in `.gitignore` (never committed)
- ✅ Credentials masked in logs and repr strings

### Documentation & Testing
- ✅ Comprehensive setup guide (`docs/IB_SETUP.md`)
- ✅ Connection test script (`src/test_connection.py`)
- ✅ Jupyter notebook with examples (`src/main.ipynb`)

## How to Use

### 1. First Time Setup
```bash
# Copy and configure credentials
cp .env.example .env
# Edit .env with your IB credentials
```

### 2. Test Connection
```bash
# Make sure IB Gateway/TWS is running first!
python src/test_connection.py
```

### 3. Use in Code
```python
from src.config.config import Config
from src.connection.ib_connection import IBConnection

# Simple usage
config = Config()
ib = IBConnection(config)
ib.connect()

# Or use context manager (recommended)
with IBConnection() as ib:
    if ib.is_connected:
        positions = ib.get_positions()
```

### 4. Use in Jupyter
Open `src/main.ipynb` for interactive examples.
