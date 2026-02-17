# Oracle Cloud Free Tier — Deployment Guide

## Overview

Run the mean-reversion trader 24/7 on Oracle Cloud's **Always Free** ARM instance:
- **4 OCPU** (ARM Ampere A1) + **24 GB RAM**
- **$0/month** — truly free, no credit card charges
- Ubuntu 22.04 or 24.04

## Step 1: Create the Instance

1. Sign up at [cloud.oracle.com](https://cloud.oracle.com) (requires credit card for verification only)
2. Go to **Compute → Instances → Create Instance**
3. Settings:
   - **Image:** Ubuntu 22.04 or 24.04
   - **Shape:** VM.Standard.A1.Flex → **4 OCPU, 24 GB RAM**
   - **Boot volume:** 50 GB (free tier allows up to 200 GB)
   - **Networking:** Default VCN, assign public IP
   - **SSH key:** Upload your public key (`~/.ssh/id_rsa.pub`)
4. Click **Create** — wait 2-3 minutes for provisioning

## Step 2: SSH & Setup

```bash
# SSH into the instance
ssh ubuntu@<your-instance-ip>

# Clone the repo
git clone https://github.com/<your-username>/Quant.git ~/Quant

# Run the setup script
cd ~/Quant && bash deploy/setup.sh
```

The setup script:
- Installs Python 3, git, system deps
- Creates a `trader` user
- Sets up a virtual environment with all packages
- Installs the systemd service

## Step 3: Configure API Keys

```bash
sudo -u trader nano /home/trader/Quant/.env
```

Add your Alpaca paper trading keys:
```
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
```

## Step 4: Test

```bash
# Run a single shadow cycle to verify everything works
sudo -u trader /home/trader/Quant/venv/bin/python \
    /home/trader/Quant/src/main_trader.py --once
```

You should see the full cycle: data fetch → signals → shadow execution → shutdown.

## Step 5: Start the Service

```bash
# Start in shadow mode (default)
sudo systemctl start quant-trader

# Watch logs in real-time
sudo journalctl -u quant-trader -f
```

## Step 6: Go Live (when ready)

Edit the service file to switch from shadow to live:

```bash
sudo nano /etc/systemd/system/quant-trader.service
# Change: --mode shadow  →  --mode live
sudo systemctl daemon-reload
sudo systemctl restart quant-trader
```

## Monitoring

```bash
# Service status
sudo systemctl status quant-trader

# Recent logs
sudo journalctl -u quant-trader -n 100

# File-based logs
ls /home/trader/Quant/data/logs/
cat /home/trader/Quant/data/logs/trader_$(date +%Y%m%d).log

# Shadow state (current positions)
cat /home/trader/Quant/data/snapshots/shadow_state.csv
```

## Transfer Data Cache (Optional)

If you have cached data from your dev environment, transfer it to skip the initial API fetch:

```bash
# From your local machine
scp -r data/snapshots/alpaca_cache/ ubuntu@<server>:/home/trader/Quant/data/snapshots/
scp -r data/historical/daily/ ubuntu@<server>:/home/trader/Quant/data/historical/
sudo chown -R trader:trader /home/trader/Quant/data/
```

## Maintenance

```bash
# Update code
sudo -u trader git -C /home/trader/Quant pull
sudo systemctl restart quant-trader

# Update packages
sudo -u trader /home/trader/Quant/venv/bin/pip install -r /home/trader/Quant/requirements.txt
sudo systemctl restart quant-trader

# View resource usage
htop
```

## Architecture

```
┌──────────────────────────────────────┐
│         main_trader.py               │
│  ┌──────────────────────────────┐    │
│  │ startup                      │    │
│  │  ├─ load config.yaml         │    │
│  │  ├─ connect Alpaca           │    │
│  │  ├─ select universe (top 30) │    │
│  │  └─ restore shadow state     │    │
│  └──────────────────────────────┘    │
│                │                     │
│  ┌─────────────▼────────────────┐    │
│  │ daily loop                   │    │
│  │  ├─ wait for market open     │    │
│  │  ├─ fetch data (cached)      │    │
│  │  ├─ generate signals         │    │
│  │  ├─ execute (shadow/live)    │    │
│  │  ├─ persist state            │    │
│  │  └─ sleep until tomorrow     │    │
│  └──────────────────────────────┘    │
│                │                     │
│  ┌─────────────▼────────────────┐    │
│  │ graceful shutdown            │    │
│  │  ├─ SIGTERM / SIGINT         │    │
│  │  ├─ save shadow state        │    │
│  │  └─ log final equity         │    │
│  └──────────────────────────────┘    │
│                                      │
│  systemd: auto-restart on failure    │
│  logs: journald + data/logs/         │
└──────────────────────────────────────┘
```

## CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `--mode` | `shadow` or `live` | `shadow` |
| `--interval` | Seconds between cycles (0 = daily) | `0` |
| `--once` | Run single cycle and exit | off |
| `--log-level` | `DEBUG`, `INFO`, `WARNING` | `INFO` |
