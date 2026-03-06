#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# ops.sh — Operations toolkit for Quant Trader VM management
# ═══════════════════════════════════════════════════════════════════════════
#
# Usage:
#   ./ops.sh <command> [args]
#
# Commands:
#   deploy              Git push + pull on VM + restart service + verify
#   status              Service status + last 15 log lines
#   logs [N]            Tail N log lines (default: 50)
#   dashboard           Regenerate dashboard + push to GitHub Pages
#   positions           Show current open positions
#   trades [N]          Show recent trades (default: 20)
#   state               Dump live_state.json summary
#   equity              Show equity history (last 10 entries)
#   run <script.py>     Upload & run a Python script on VM (auto-cleanup)
#   ssh                 Open interactive SSH to VM as trader
#   restart             Restart the service (no code pull)
#   stop                Stop the service
#   start               Start the service
# ═══════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Configuration ───────────────────────────────────────────────────────
VM_HOST="40.233.100.95"
VM_USER="ubuntu"
TRADER_USER="trader"
REPO_DIR="/home/trader/Quant"
VENV_DIR="$REPO_DIR/venv"
SERVICE="quant-trader"

# ── Helpers ─────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

info()  { echo -e "${CYAN}▸${NC} $*"; }
ok()    { echo -e "${GREEN}✓${NC} $*"; }
warn()  { echo -e "${YELLOW}⚠${NC} $*"; }
fail()  { echo -e "${RED}✗${NC} $*"; exit 1; }
header() { echo -e "\n${BOLD}═══ $* ═══${NC}"; }

# Run command on VM as trader user
vm_trader() {
    ssh "$VM_USER@$VM_HOST" "sudo -u $TRADER_USER bash -c 'cd $REPO_DIR && source $VENV_DIR/bin/activate && set -a && source .env 2>/dev/null && set +a && $1'" 2>&1
}

# Run command on VM as ubuntu (for systemctl)
vm_sudo() {
    ssh "$VM_USER@$VM_HOST" "$1" 2>&1
}

# Run a Python script on VM via stdin (avoids shell escaping issues with $, f-strings)
vm_python() {
    ssh "$VM_USER@$VM_HOST" "sudo -u $TRADER_USER bash -c 'cd $REPO_DIR && source $VENV_DIR/bin/activate && set -a && source .env 2>/dev/null && set +a && python -'" 2>&1
}

# ── Commands ────────────────────────────────────────────────────────────

cmd_deploy() {
    header "DEPLOY TO VM"

    # Step 1: Push local changes
    info "Pushing to GitHub..."
    if git diff --quiet HEAD 2>/dev/null && git diff --cached --quiet 2>/dev/null; then
        ok "Working tree clean — skipping push"
    else
        warn "You have uncommitted changes. Commit first, then run deploy."
        exit 1
    fi
    git push origin main 2>&1 | tail -3
    ok "Pushed to GitHub"

    # Step 2: Pull on VM
    info "Pulling on VM..."
    vm_trader "git pull origin main 2>&1" | tail -5
    ok "Code updated on VM"

    # Step 3: Restart service
    info "Restarting service..."
    vm_sudo "sudo systemctl daemon-reload && sudo systemctl restart $SERVICE" 2>&1
    sleep 3
    ok "Service restarted"

    # Step 4: Verify
    info "Checking service status..."
    local status
    status=$(vm_sudo "sudo systemctl is-active $SERVICE" 2>&1)
    if [[ "$status" == "active" ]]; then
        ok "Service is ACTIVE"
    else
        fail "Service is $status — check logs with: ./ops.sh logs"
    fi

    # Step 5: Show recent logs
    info "Recent logs:"
    vm_sudo "sudo journalctl -u $SERVICE -n 10 --no-pager" 2>&1 | tail -10

    echo ""
    ok "Deploy complete"
}

cmd_status() {
    header "SERVICE STATUS"
    vm_sudo "sudo systemctl status $SERVICE --no-pager" 2>&1 | head -15
    echo ""
    header "RECENT LOGS"
    vm_sudo "sudo journalctl -u $SERVICE -n 15 --no-pager" 2>&1 | tail -15
}

cmd_logs() {
    local n="${1:-50}"
    vm_sudo "sudo journalctl -u $SERVICE -n $n --no-pager" 2>&1
}

cmd_dashboard() {
    header "REGENERATE DASHBOARD"

    info "Generating dashboard on VM..."
    vm_python <<'PYEOF'
import sys
from pathlib import Path
sys.path.insert(0, 'src')
from dashboard_generator import DashboardGenerator
gen = DashboardGenerator(Path('.'))
ok = gen.generate(Path('docs/index.html'))
print('Generated' if ok else 'FAILED')
PYEOF
    ok "Dashboard HTML generated"

    info "Pushing to GitHub Pages (orphan branch)..."
    # Push to main first
    vm_trader 'git add docs/index.html && git commit -m "Dashboard update $(date +%Y-%m-%d\ %H:%M)" 2>/dev/null && git push origin main 2>&1 || echo "No main changes"' | tail -3

    # Then update dashboard-live orphan branch with only docs + data/snapshots
    vm_trader 'bash -s' <<'DASHEOF'
set -e
cp docs/index.html /tmp/_dash_index.html
for f in signal_history.json trade_history.json live_state.json equity_history.json intraday_equity.json; do
    [ -f "data/snapshots/$f" ] && cp "data/snapshots/$f" "/tmp/_dash_$f"
done
git stash --include-untracked 2>/dev/null || true
git fetch origin dashboard-live 2>/dev/null && git checkout dashboard-live && git reset --hard origin/dashboard-live || {
    git checkout --orphan dashboard-live; git reset --hard;
}
mkdir -p docs data/snapshots
cp /tmp/_dash_index.html docs/index.html
touch docs/.nojekyll
for f in signal_history.json trade_history.json live_state.json equity_history.json intraday_equity.json; do
    [ -f "/tmp/_dash_$f" ] && cp "/tmp/_dash_$f" "data/snapshots/$f"
done
git add docs/ data/
git commit -m "Dashboard update $(date +%Y-%m-%d\ %H:%M)" 2>/dev/null || true
git push origin dashboard-live --force 2>&1 || echo "Push failed"
git checkout -f main
git stash pop 2>/dev/null || true
rm -f /tmp/_dash_*.html /tmp/_dash_*.json
DASHEOF
    ok "Dashboard deployed (clean orphan branch)"
    echo ""
    echo "  View: https://shimmy-shams.github.io/Quant/"
}

cmd_positions() {
    header "OPEN POSITIONS"
    vm_python <<'PYEOF'
import json, sys
from datetime import datetime
d = json.load(open('data/snapshots/live_state.json'))
pos = d.get('positions', [])
acct = d.get('account', {})
if not pos:
    print('No open positions')
    sys.exit(0)
print(f'Portfolio: ${acct.get("equity",0):,.2f}  |  Cash: ${acct.get("cash",0):,.2f}  |  Positions: {len(pos)}')
print()
hdr = f'{"Symbol":<8} {"Side":<6} {"Qty":>5} {"Entry":>10} {"Current":>10} {"P&L":>10} {"P&L%":>8} {"Entered":<12} {"Days":>4}'
print(hdr)
print('-' * len(hdr))
total_pnl = 0
for p in sorted(pos, key=lambda x: abs(x.get('unrealized_pl',0)), reverse=True):
    pnl = p.get('unrealized_pl', 0)
    pnl_pct = p.get('unrealized_plpc', 0)
    ed = p.get('entry_date', '')
    days = ''
    if ed:
        try: days = (datetime.now() - datetime.strptime(ed, '%Y-%m-%d')).days
        except: pass
    total_pnl += pnl
    print(f'{p["symbol"]:<8} {p["side"]:<6} {p["qty"]:>5.0f} ${p["entry_price"]:>9,.2f} ${p["current_price"]:>9,.2f} ${pnl:>9,.2f} {pnl_pct:>7.2f}% {ed:<12} {days:>4}')
print('-' * len(hdr))
print(f'{"TOTAL":<8} {"":6} {"":>5} {"":>10} {"":>10} ${total_pnl:>9,.2f}')
PYEOF
}

cmd_trades() {
    local n="${1:-20}"
    header "RECENT TRADES (last $n)"
    sed "s/__LIMIT__/$n/" <<'PYEOF' | vm_python
import json
d = json.load(open('data/snapshots/live_state.json'))
trades = d.get('recent_trades', [])[:__LIMIT__]
if not trades:
    print('No trades')
    exit(0)
hdr = f'{"Symbol":<8} {"Side":<6} {"Type":<7} {"Qty":>5} {"Price":>10} {"Entry":>10} {"P&L%":>8} {"Date":<12}'
print(hdr)
print('-' * len(hdr))
for t in trades:
    tt = t.get('trade_type','entry')
    ep = t.get('entry_price','')
    ep_str = f'${ep:>9,.2f}' if ep else '         -'
    pnl = t.get('pnl_pct','')
    pnl_str = f'{pnl:>7.2f}%' if pnl != '' and pnl is not None else '       -'
    date = (t.get('submitted_at','') or '')[:10]
    print(f'{t["symbol"]:<8} {t["side"]:<6} {tt:<7} {t.get("qty",0):>5.0f} ${t["filled_price"]:>9,.2f} {ep_str} {pnl_str} {date:<12}')
PYEOF
}

cmd_state() {
    header "LIVE STATE SUMMARY"
    vm_python <<'PYEOF'
import json
d = json.load(open('data/snapshots/live_state.json'))
ts = d.get('timestamp', 'unknown')
acct = d.get('account', {})
pos = d.get('positions', [])
trades = d.get('recent_trades', [])
orders = d.get('open_orders', [])
entries = [t for t in trades if t.get('trade_type') == 'entry']
exits = [t for t in trades if t.get('trade_type') == 'exit']
print(f'Snapshot:     {ts}')
print(f'Equity:       ${acct.get("equity",0):,.2f}')
print(f'Cash:         ${acct.get("cash",0):,.2f}')
print(f'Buying Power: ${acct.get("buying_power",0):,.2f}')
print(f'Positions:    {len(pos)}')
print(f'Open Orders:  {len(orders)}')
print(f'Trades:       {len(trades)} ({len(entries)} entries, {len(exits)} exits)')
if pos:
    syms = [p['symbol'] for p in pos]
    print(f'Holdings:     {" ".join(syms)}')
PYEOF
}

cmd_equity() {
    header "EQUITY HISTORY (last 10)"
    vm_python <<'PYEOF'
import json
h = json.load(open('data/snapshots/equity_history.json'))
print(f'Total points: {len(h)}')
print(f'{"Timestamp":<18} {"Equity":>14}')
print('-' * 34)
for e in h[-10:]:
    print(f'{e["timestamp"]:<18} ${e["equity"]:>13,.2f}')
PYEOF
}

cmd_run() {
    local script="$1"
    [[ -f "$script" ]] || fail "File not found: $script"

    header "RUN SCRIPT ON VM: $script"

    local remote_path="/tmp/ops_run_$(basename "$script")"

    info "Uploading $script..."
    cat "$script" | ssh "$VM_USER@$VM_HOST" "sudo -u $TRADER_USER tee $remote_path > /dev/null"
    ok "Uploaded to $remote_path"

    info "Running..."
    vm_trader "python $remote_path 2>&1"
    local rc=$?

    info "Cleaning up..."
    ssh "$VM_USER@$VM_HOST" "sudo -u $TRADER_USER rm -f $remote_path" 2>&1
    ok "Temp file removed"

    return $rc
}

cmd_ssh() {
    info "Opening SSH session as $TRADER_USER..."
    ssh -t "$VM_USER@$VM_HOST" "sudo -iu $TRADER_USER"
}

cmd_restart() {
    info "Restarting $SERVICE..."
    vm_sudo "sudo systemctl daemon-reload && sudo systemctl restart $SERVICE"
    sleep 2
    local status
    status=$(vm_sudo "sudo systemctl is-active $SERVICE")
    if [[ "$status" == "active" ]]; then
        ok "Service is ACTIVE"
    else
        fail "Service is $status"
    fi
}

cmd_stop() {
    info "Stopping $SERVICE..."
    vm_sudo "sudo systemctl stop $SERVICE"
    ok "Service stopped"
}

cmd_start() {
    info "Starting $SERVICE..."
    vm_sudo "sudo systemctl start $SERVICE"
    sleep 2
    local status
    status=$(vm_sudo "sudo systemctl is-active $SERVICE")
    if [[ "$status" == "active" ]]; then
        ok "Service is ACTIVE"
    else
        fail "Service is $status"
    fi
}

# ── Dispatch ────────────────────────────────────────────────────────────

cmd="${1:-help}"
shift 2>/dev/null || true

case "$cmd" in
    deploy)     cmd_deploy ;;
    status)     cmd_status ;;
    logs)       cmd_logs "$@" ;;
    dashboard)  cmd_dashboard ;;
    positions)  cmd_positions ;;
    trades)     cmd_trades "$@" ;;
    state)      cmd_state ;;
    equity)     cmd_equity ;;
    run)        [[ $# -ge 1 ]] || fail "Usage: ./ops.sh run <script.py>"; cmd_run "$1" ;;
    ssh)        cmd_ssh ;;
    restart)    cmd_restart ;;
    stop)       cmd_stop ;;
    start)      cmd_start ;;
    help|--help|-h)
        echo "Usage: ./ops.sh <command> [args]"
        echo ""
        echo "Commands:"
        echo "  deploy              Push code + pull on VM + restart + verify"
        echo "  status              Service status + recent logs"
        echo "  logs [N]            Tail N log lines (default: 50)"
        echo "  dashboard           Regenerate dashboard + push to GitHub Pages"
        echo "  positions           Show current open positions"
        echo "  trades [N]          Show recent trades (default: 20)"
        echo "  state               Live state summary (equity, positions, trades)"
        echo "  equity              Last 10 equity history entries"
        echo "  run <script.py>     Upload + run Python script on VM (auto-cleanup)"
        echo "  ssh                 Interactive SSH as trader"
        echo "  restart             Restart service (no code pull)"
        echo "  stop                Stop service"
        echo "  start               Start service"
        echo ""
        echo "Examples:"
        echo "  ./ops.sh deploy                   # Full deploy pipeline"
        echo "  ./ops.sh logs 100                 # Last 100 log lines"
        echo "  ./ops.sh run analysis.py          # Run script on VM"
        ;;
    *)
        fail "Unknown command: $cmd — run ./ops.sh help"
        ;;
esac
