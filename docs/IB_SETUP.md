# Interactive Brokers Setup Guide

This guide will help you set up and test your connection to Interactive Brokers Gateway/TWS.

## Prerequisites

1. Interactive Brokers account (paper or live)
2. IB Gateway or Trader Workstation (TWS) installed
3. Python dependencies installed (done via `requirements.txt`)

## Step 1: Configure IB Gateway/TWS

### Enable API Access

1. Open IB Gateway or TWS
2. Go to **Configure → Settings → API → Settings**
3. Enable the following:
   - ✅ Enable ActiveX and Socket Clients
   - ✅ Allow connections from localhost only (recommended for security)
   - ✅ Read-Only API (recommended for paper trading)
4. Note the **Socket port**:
   - TWS Paper Trading: `7497`
   - TWS Live Trading: `7496`
   - IB Gateway Paper: `4002`
   - IB Gateway Live: `4001`

### Create Trusted IP

1. In the same API settings window
2. Add `127.0.0.1` to the trusted IPs list
3. Click **OK** to save

## Step 2: Configure Your Environment

### Create .env File

1. Copy the example file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` with your credentials:
   ```bash
   # IB Gateway/TWS Connection Settings
   IB_HOST=127.0.0.1
   IB_PORT=7497              # Use appropriate port from above
   IB_CLIENT_ID=1

   # IB Account Credentials
   IB_USERNAME=your_username_here
   IB_PASSWORD=your_password_here

   # Account Information
   IB_ACCOUNT_ID=your_account_id_here

   # Trading Mode
   TRADING_MODE=paper        # Options: paper, live

   # Logging Level
   LOG_LEVEL=INFO           # Options: DEBUG, INFO, WARNING, ERROR
   ```

### Important Security Notes

- **NEVER** commit your `.env` file to version control
- `.env` is already in `.gitignore`
- Keep your credentials secure
- Use paper trading for testing

## Step 3: Test Your Connection

### Run the Test Script

```bash
python src/test_connection.py
```

### Expected Output

If successful, you should see:

```
============================================================
  IB Gateway Connection Test
============================================================

1. Loading configuration...
   Config(host=127.0.0.1, port=7497, client_id=1, mode=paper, username=***)

2. Creating connection object...
   IBConnection(status=Disconnected, host=127.0.0.1:7497, mode=paper)

3. Connecting to IB Gateway/TWS...
   Host: 127.0.0.1
   Port: 7497
   Client ID: 1
   Trading Mode: PAPER

✅ CONNECTION SUCCESSFUL

4. Testing connection and retrieving information...
   Server Version: 176
   Connection Time: 2026-02-12 10:30:45.123456
   Managed Accounts: ['DU1234567']

============================================================
  Account Summary
============================================================

Metric                               Value
----------------------------------------------------
NetLiquidation                  $1,000,000.00
TotalCashValue                  $1,000,000.00
GrossPositionValue                       $0.00
BuyingPower                     $4,000,000.00
AvailableFunds                  $1,000,000.00

============================================================
  Current Positions
============================================================

No open positions

============================================================
  Disconnecting
============================================================

✅ Test completed successfully!
```

## Troubleshooting

### Connection Failed

If you see `❌ CONNECTION FAILED`, check:

1. **IB Gateway/TWS is running**
   - Make sure it's logged in and active

2. **API Settings are enabled**
   - Configure → Settings → API → Settings
   - Enable ActiveX and Socket Clients

3. **Port matches**
   - Check your `.env` file `IB_PORT` matches IB settings
   - Paper TWS: 7497
   - Live TWS: 7496

4. **Firewall**
   - Ensure firewall allows localhost connections

### Common Errors

| Error | Solution |
|-------|----------|
| `Connection refused` | IB Gateway/TWS not running or wrong port |
| `Already connected` | Another client connected with same ID, change `IB_CLIENT_ID` |
| `Invalid credentials` | Check username/password in `.env` |
| `API not enabled` | Enable API in IB settings |

## Using the Classes in Your Code

### Basic Usage

```python
from src.config.config import Config
from src.connection.ib_connection import IBConnection

# Load configuration
config = Config()

# Create connection
ib = IBConnection(config)

# Connect
if ib.connect():
    print("Connected!")

    # Get account summary
    summary = ib.get_account_summary()

    # Get positions
    positions = ib.get_positions()

    # Disconnect
    ib.disconnect()
```

### Using Context Manager

```python
from src.connection.ib_connection import IBConnection

# Automatically connects and disconnects
with IBConnection() as ib:
    if ib.is_connected:
        account_values = ib.get_account_values()
        positions = ib.get_positions()
```

## Next Steps

Once your connection is working:

1. ✅ Connection established
2. ⏭️ Implement data fetching (historical & real-time)
3. ⏭️ Build strategy framework
4. ⏭️ Create backtesting engine
5. ⏭️ Implement order execution

## Additional Resources

- [IB API Documentation](https://interactivebrokers.github.io/)
- [ib_insync Documentation](https://ib-insync.readthedocs.io/)
- [TWS API Guide](https://www.interactivebrokers.com/en/software/api/api.htm)
