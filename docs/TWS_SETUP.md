# Trader Workstation (TWS) Setup Guide

This guide covers setting up TWS for API access with this trading system.

---

## Prerequisites

1. **Interactive Brokers Account** (Paper or Live)
2. **Trader Workstation (TWS)** installed and running
3. **Python environment** set up with `ib_insync` installed

---

## TWS API Configuration

### Step 1: Enable API Access

1. **Open TWS** and log in to your account (Paper or Live)

2. **Navigate to API Settings:**
   - Click **File** → **Global Configuration** (or **Edit** → **Global Configuration** on some versions)
   - In the left panel, select **API** → **Settings**

3. **Configure API Settings:**
   - ✅ **Enable ActiveX and Socket Clients** — CHECK THIS BOX
   - ✅ **Allow connections from localhost only** — CHECK THIS for security (unless running remotely)
   - **Socket port:** Set based on your trading mode:
     - **Paper Trading:** `7497` (default)
     - **Live Trading:** `7496`
   - ✅ **Read-Only API** — UNCHECK (we need read-write for trading)
   - **Master API Client ID:** Leave as `0` (or note it if you change it)
   - ✅ **Download open orders on connection** — CHECK THIS
   - **Trusted IP Addresses:** Add `127.0.0.1` (localhost)

4. **Click OK** and restart TWS for changes to take effect

---

### Step 2: Configure Your `.env` File

Update your `.env` file with the correct port:

```bash
# For TWS Paper Trading
IB_PORT=7497

# For TWS Live Trading  
IB_PORT=7496

# For IB Gateway Paper Trading
# IB_PORT=4002

# For IB Gateway Live Trading
# IB_PORT=4001
```

---

## Common Issues & Troubleshooting

### Issue: "Connection failed - TimeoutError"

**Cause:** TWS is not running or API is not enabled

**Fix:**
1. Ensure TWS is open and logged in
2. Check **File → Global Configuration → API → Settings**
3. Verify **Enable ActiveX and Socket Clients** is checked
4. Verify the port matches your `.env` file (7497 for paper, 7496 for live)
5. Restart TWS after changing settings

---

### Issue: "Account values show N/A"

**Cause:** Account subscription not fully synced, or currency mismatch

**Fix:**
1. Wait 5-10 seconds after connecting for data to populate
2. Re-run the account overview cell
3. Check that your account currency (CAD/USD/etc) is handled in the code
4. Verify TWS shows account data in the main window

**Note:** The code now automatically handles multiple currencies (USD, CAD, EUR, GBP).

---

### Issue: "This event loop is already running"

**Cause:** Jupyter/IPython async event loop conflict with `ib_insync`

**Fix:** Already handled in the connection code via `util.patchAsyncio()`. If you still see this:
1. Restart the Jupyter kernel
2. Re-run Cell 1 (Setup)
3. Re-run Cell 2 (Connect)

---

### Issue: Client ID already in use

**Cause:** Previous connection not cleanly disconnected

**Fix:** The code automatically tries fallback client IDs (1, 2, 3). If all fail:
1. Disconnect in the notebook (run Cell 6)
2. Restart TWS
3. Reconnect

---

## Verifying Connection

After enabling API and restarting TWS, run these cells in order:

1. **Cell 1** — Setup (should show "Ready to connect to IB Gateway")
2. **Cell 2** — Connect (should show "[OK] Connected")
3. **Cell 3** — Test Connection (should show server version and accounts)
4. **Cell 4** — Account Overview (should show your account balance)

If Cell 3 shows server info and Cell 4 shows your account balance, you're all set!

---

## TWS vs IB Gateway

| Feature | TWS | IB Gateway |
|---|---|---|
| **UI** | Full trading interface | Minimal login window |
| **Resource Usage** | Higher | Lower |
| **Paper Port** | 7497 | 4002 |
| **Live Port** | 7496 | 4001 |
| **Best For** | Active monitoring | Automated trading |

**Recommendation:** Use **IB Gateway** for production automated trading (lighter, more stable). Use **TWS** for development and manual oversight.

---

## Security Notes

- **Never commit credentials** — `.env` file is gitignored
- **Use Paper Trading first** — Test all strategies thoroughly before going live
- **Localhost only** — Keep "Allow connections from localhost only" checked unless you need remote access
- **Firewall** — Ensure your firewall allows connections on the TWS port (7497/7496)

---

## Next Steps

Once connected successfully:
1. Run data collection (Cell 5a) to download historical OHLCV data
2. Verify data storage (Cell 5b)
3. Test real-time quotes during market hours (Cell 5c)
4. Begin strategy development (Phase 2)

---

**Last Updated:** 2026-02-12  
**Related:** [IB_SETUP.md](IB_SETUP.md) for IB Gateway setup
