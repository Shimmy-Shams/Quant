# Quantitative Trading System

A Python-based quantitative trading system with Interactive Brokers integration for algorithmic trading strategies.

## Quick Start

### 1. Configure Credentials

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your IB credentials
# (use your preferred editor)
```

Configure the following in `.env`:
- `IB_HOST` - IB Gateway host (default: 127.0.0.1)
- `IB_PORT` - Port (7497 for paper TWS, 4002 for paper IB Gateway)
- `IB_USERNAME` - Your IB username
- `IB_PASSWORD` - Your IB password
- `IB_ACCOUNT_ID` - Your IB account number
- `TRADING_MODE` - Set to `paper` for testing, `live` for real trading

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up IB Gateway/TWS

1. Install and run IB Gateway or TWS
2. Login to your account
3. Enable API access:
   - Go to **Configure → Settings → API → Settings**
   - Enable **"Enable ActiveX and Socket Clients"**
   - Check **"Read-Only API"** for paper trading
   - Note the socket port number

See [docs/IB_SETUP.md](docs/IB_SETUP.md) for detailed instructions.

### 4. Test Connection

```bash
python src/test_connection.py
```

You should see:
```
✅ CONNECTION SUCCESSFUL
```

## Project Structure

```
Quant/
├── src/
│   ├── config/              # Configuration management
│   ├── connection/          # IB Gateway connection
│   ├── strategies/          # Trading strategies
│   ├── data/                # Data storage
│   ├── backtest/            # Backtesting engine
│   ├── execution/           # Order execution
│   ├── utils/               # Helper functions
│   ├── main.ipynb           # Jupyter notebook examples
│   └── test_connection.py   # Connection test script
├── docs/                    # Documentation
├── requirements.txt         # Python dependencies
└── .env.example             # Credentials template
```

## Usage Examples

### Python Script

```python
from src.config.config import Config
from src.connection.ib_connection import IBConnection

# Connect to IB using context manager (recommended)
with IBConnection() as ib:
    if ib.is_connected:
        # Get account summary
        account_values = ib.get_account_values()
        print(f"Net Liquidation: ${account_values['NetLiquidation_USD']}")

        # Get positions
        positions = ib.get_positions()
        for pos in positions:
            print(f"{pos.contract.symbol}: {pos.position} shares")
```

### Jupyter Notebook

Open `src/main.ipynb` for interactive examples with detailed documentation.

## Core Classes

### Config (`src/config/config.py`)
Manages configuration and credentials from `.env` file.

```python
config = Config()
print(config.ib_host)      # 127.0.0.1
print(config.ib_port)      # 7497
print(config.trading_mode) # paper
```

### IBConnection (`src/connection/ib_connection.py`)
Handles IB Gateway/TWS connection lifecycle.

```python
ib = IBConnection()
ib.connect()
ib.get_account_summary()
ib.get_positions()
ib.disconnect()
```

## Security

- **Never commit `.env` file** - it contains sensitive credentials
- `.env` is already in `.gitignore`
- Use `.env.example` as a template
- Always start with paper trading before going live

## Development Roadmap

### Phase 1: Connection & Data ✅
- [x] Environment setup
- [x] IB Gateway connection
- [ ] Real-time data feed
- [ ] Historical data retrieval

### Phase 2: Strategy Framework
- [ ] Base strategy class
- [ ] Backtesting engine
- [ ] Performance metrics
- [ ] Example strategies

### Phase 3: Execution & Risk
- [ ] Order management
- [ ] Risk controls
- [ ] Position sizing
- [ ] Paper trading integration

### Phase 4: Live Trading
- [ ] Live deployment
- [ ] Monitoring & alerts
- [ ] Performance tracking
- [ ] Multi-strategy management

## Documentation

- [IB Setup Guide](docs/IB_SETUP.md) - Detailed Interactive Brokers configuration
- [Project Context](CLAUDE.md) - Development notes and planning

## Resources

- [Interactive Brokers API](https://interactivebrokers.github.io/)
- [ib_insync Documentation](https://ib-insync.readthedocs.io/)
- [TWS API Guide](https://www.interactivebrokers.com/en/software/api/api.htm)

## Contributing

This is a personal trading project. If you're building your own system, feel free to use this as a reference.

## Disclaimer

This software is for educational purposes. Trading carries risk. Always test with paper trading before risking real capital.

## License

Private project - not for distribution.
