#!/usr/bin/env python3
"""
Real-Time Trading Dashboard

FastAPI server that monitors the trading bot and displays:
- Current portfolio status
- Open positions
- Trade history
- Performance metrics
- Live logs

Access: http://your-ip:8080
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# Bootstrap path
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

app = FastAPI(title="Quant Trading Dashboard", version="1.0.0")

# Paths
DATA_DIR = PROJECT_ROOT / "data"
SHADOW_STATE = DATA_DIR / "snapshots" / "shadow_state.csv"
TRADING_LOGS_DIR = DATA_DIR / "snapshots" / "trading_logs"
LOG_DIR = DATA_DIR / "logs"


# ═══════════════════════════════════════════════════════════════════════════
# API ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/")
async def home():
    """Serve the main dashboard HTML"""
    html_file = SRC_DIR / "templates" / "dashboard.html"
    if html_file.exists():
        return FileResponse(html_file)
    return HTMLResponse(FALLBACK_HTML)


@app.get("/api/status")
async def get_status():
    """Current portfolio status and open positions"""
    try:
        positions = []
        total_value = 0.0
        total_pnl = 0.0
        
        if SHADOW_STATE.exists():
            df = pd.read_csv(SHADOW_STATE)
            for _, row in df.iterrows():
                pnl = (row['current_price'] - row['entry_price']) * row['qty']
                if row['side'] == 'short':
                    pnl = -pnl
                pnl_pct = pnl / (row['entry_price'] * abs(row['qty'])) * 100
                
                positions.append({
                    'symbol': row['symbol'],
                    'side': row['side'],
                    'qty': int(row['qty']),
                    'entry_price': float(row['entry_price']),
                    'current_price': float(row['current_price']),
                    'pnl': round(pnl, 2),
                    'pnl_pct': round(pnl_pct, 2),
                    'entry_date': str(row['entry_date'])[:10],
                    'signal_strength': round(float(row.get('signal_strength', 0)), 2)
                })
                
                total_value += abs(row['current_price'] * row['qty'])
                total_pnl += pnl
        
        return {
            'timestamp': datetime.now().isoformat(),
            'n_positions': len(positions),
            'positions': positions,
            'total_value': round(total_value, 2),
            'total_pnl': round(total_pnl, 2),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/performance")
async def get_performance():
    """Performance metrics and equity curve"""
    try:
        # Find all trading log files
        equity_data = []
        trade_data = []
        
        if TRADING_LOGS_DIR.exists():
            for log_file in sorted(TRADING_LOGS_DIR.glob("equity_*.csv")):
                try:
                    df = pd.read_csv(log_file)
                    if 'date' in df.columns and 'equity' in df.columns:
                        for _, row in df.iterrows():
                            equity_data.append({
                                'date': str(row['date'])[:10],
                                'equity': float(row['equity']),
                                'daily_return': float(row.get('daily_return', 0)) * 100
                            })
                except Exception:
                    continue
            
            for log_file in sorted(TRADING_LOGS_DIR.glob("trades_*.csv")):
                try:
                    df = pd.read_csv(log_file)
                    for _, row in df.iterrows():
                        trade_data.append({
                            'symbol': row['symbol'],
                            'side': row['side'],
                            'entry_date': str(row['entry_date'])[:10],
                            'exit_date': str(row.get('exit_date', ''))[:10],
                            'pnl_pct': round(float(row.get('pnl_pct', 0)) * 100, 2),
                            'exit_reason': row.get('exit_reason', 'Open')
                        })
                except Exception:
                    continue
        
        # Calculate metrics
        total_return = 0.0
        win_rate = 0.0
        total_trades = len(trade_data)
        
        if equity_data:
            initial = equity_data[0]['equity']
            final = equity_data[-1]['equity']
            total_return = ((final - initial) / initial) * 100
        
        if trade_data:
            wins = sum(1 for t in trade_data if t['pnl_pct'] > 0)
            win_rate = (wins / total_trades) * 100
        
        return {
            'equity_curve': equity_data[-100:],  # Last 100 days
            'recent_trades': trade_data[-50:],   # Last 50 trades
            'metrics': {
                'total_return': round(total_return, 2),
                'total_trades': total_trades,
                'win_rate': round(win_rate, 1),
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/logs")
async def get_logs(lines: int = 100):
    """Recent log entries"""
    try:
        today = datetime.now().strftime("%Y%m%d")
        log_file = LOG_DIR / f"trader_{today}.log"
        
        if not log_file.exists():
            # Try yesterday's log
            yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
            log_file = LOG_DIR / f"trader_{yesterday}.log"
        
        if log_file.exists():
            with open(log_file, 'r') as f:
                all_lines = f.readlines()
                recent = all_lines[-lines:] if len(all_lines) > lines else all_lines
                return {'logs': [line.strip() for line in recent]}
        
        return {'logs': ['No logs found']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


# ═══════════════════════════════════════════════════════════════════════════
# FALLBACK HTML (if template file missing)
# ═══════════════════════════════════════════════════════════════════════════

FALLBACK_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Quant Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
</head>
<body>
    <h1>Quant Trading Dashboard</h1>
    <p>Template file missing. Create: src/templates/dashboard.html</p>
    <p>API Endpoints:</p>
    <ul>
        <li><a href="/api/status">/api/status</a> - Portfolio status</li>
        <li><a href="/api/performance">/api/performance</a> - Performance metrics</li>
        <li><a href="/api/logs?lines=50">/api/logs</a> - Recent logs</li>
    </ul>
</body>
</html>
"""


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Trading Dashboard Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes")
    args = parser.parse_args()
    
    print(f"Starting dashboard on http://{args.host}:{args.port}")
    uvicorn.run(
        "dashboard:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )
