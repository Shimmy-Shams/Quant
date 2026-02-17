#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# Oracle Cloud Free Tier — Quant Trader Setup Script
# ═══════════════════════════════════════════════════════════════════════════
#
# Run this on a fresh Ubuntu 22.04/24.04 ARM instance:
#   curl -sSL https://raw.githubusercontent.com/<user>/Quant/main/deploy/setup.sh | bash
#
# Or after cloning:
#   cd ~/Quant && bash deploy/setup.sh
#
# What this does:
#   1. Install system deps (Python 3.12, git)
#   2. Create dedicated 'trader' user
#   3. Clone repo & create venv
#   4. Install Python packages
#   5. Install systemd service
#   6. Print next steps (API keys, start service)
#
# Requirements:
#   - Oracle Cloud free-tier ARM instance (4 OCPU / 24 GB RAM)
#   - Ubuntu 22.04 or 24.04 image
#   - SSH access as opc or ubuntu user with sudo
# ═══════════════════════════════════════════════════════════════════════════

set -euo pipefail

REPO_DIR="/home/trader/Quant"
VENV_DIR="$REPO_DIR/venv"

echo "═══════════════════════════════════════════════════"
echo "  Oracle Cloud — Quant Trader Setup"
echo "═══════════════════════════════════════════════════"

# ── 1. System dependencies ──────────────────────────────────────────────
echo ""
echo "📦 Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3 python3-pip python3-venv \
    git curl htop tmux \
    > /dev/null 2>&1

PYTHON_VERSION=$(python3 --version 2>&1)
echo "  Python: $PYTHON_VERSION"

# ── 2. Create trader user ──────────────────────────────────────────────
if ! id -u trader > /dev/null 2>&1; then
    echo ""
    echo "👤 Creating 'trader' user..."
    sudo useradd -m -s /bin/bash trader
    sudo usermod -aG sudo trader
    echo "  User 'trader' created"
else
    echo "  User 'trader' already exists"
fi

# ── 3. Clone/update repo ──────────────────────────────────────────────
echo ""
if [ -d "$REPO_DIR/.git" ]; then
    echo "📁 Updating existing repo..."
    sudo -u trader git -C "$REPO_DIR" pull --ff-only
else
    echo "📁 Cloning repo..."
    # If running from the repo, copy files instead
    if [ -f "$(dirname "$0")/../requirements.txt" ]; then
        echo "  Copying from local checkout..."
        sudo mkdir -p "$REPO_DIR"
        sudo cp -r "$(dirname "$0")/.." "$REPO_DIR/"
        sudo chown -R trader:trader "$REPO_DIR"
    else
        echo "  ⚠️  Clone manually:"
        echo "  sudo -u trader git clone <your-repo-url> $REPO_DIR"
        sudo mkdir -p "$REPO_DIR"
        sudo chown -R trader:trader "$REPO_DIR"
    fi
fi

# ── 4. Python virtual environment ─────────────────────────────────────
echo ""
echo "🐍 Setting up Python virtual environment..."
sudo -u trader python3 -m venv "$VENV_DIR"
sudo -u trader "$VENV_DIR/bin/pip" install --upgrade pip -q
sudo -u trader "$VENV_DIR/bin/pip" install -r "$REPO_DIR/requirements.txt" -q
echo "  Packages installed"

# Verify key imports
sudo -u trader "$VENV_DIR/bin/python" -c "
import alpaca; import pandas; import numpy
print(f'  alpaca-py: {alpaca.__version__}')
print(f'  pandas: {pandas.__version__}')
print(f'  numpy: {numpy.__version__}')
"

# ── 5. Create data directories ─────────────────────────────────────────
echo ""
echo "📂 Creating data directories..."
sudo -u trader mkdir -p "$REPO_DIR/data/"{logs,snapshots/alpaca_cache,snapshots/trading_logs,historical/daily}
echo "  Done"

# ── 6. Install systemd service ─────────────────────────────────────────
echo ""
echo "🔧 Installing systemd service..."
sudo cp "$REPO_DIR/deploy/quant-trader.service" /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable quant-trader
echo "  Service installed and enabled (not started yet)"

# ── 7. Print next steps ───────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════"
echo "  ✅ SETUP COMPLETE"
echo "═══════════════════════════════════════════════════"
echo ""
echo "  Next steps:"
echo ""
echo "  1. Configure API keys:"
echo "     sudo -u trader nano $REPO_DIR/.env"
echo "     ──────────────────────────────────────"
echo "     ALPACA_API_KEY=your_key_here"
echo "     ALPACA_SECRET_KEY=your_secret_here"
echo "     ──────────────────────────────────────"
echo ""
echo "  2. Copy your data cache (optional, speeds up first run):"
echo "     scp -r data/snapshots/alpaca_cache/ trader@<server>:$REPO_DIR/data/snapshots/"
echo "     scp -r data/historical/daily/ trader@<server>:$REPO_DIR/data/historical/"
echo ""
echo "  3. Test with a single cycle:"
echo "     sudo -u trader $VENV_DIR/bin/python $REPO_DIR/src/main_trader.py --once"
echo ""
echo "  4. Start the service:"
echo "     sudo systemctl start quant-trader"
echo "     sudo journalctl -u quant-trader -f   # watch logs"
echo ""
echo "  5. Switch to LIVE mode when ready:"
echo "     Edit /etc/systemd/system/quant-trader.service"
echo "     Change --mode shadow → --mode live"
echo "     sudo systemctl daemon-reload && sudo systemctl restart quant-trader"
echo ""
echo "  Monitoring:"
echo "     sudo systemctl status quant-trader     # service status"
echo "     sudo journalctl -u quant-trader -n 50  # recent logs"
echo "     cat $REPO_DIR/data/logs/trader_*.log   # file logs"
echo ""
