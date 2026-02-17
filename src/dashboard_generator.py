"""
Static Dashboard Generator

Generates a beautiful HTML dashboard from trading logs and shadow state.
Used after each trading cycle to create GitHub Pages content.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
import numpy as np


class DashboardGenerator:
    """Generate static HTML dashboard from trading data"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.data_dir = project_root / "data"
        self.shadow_state = self.data_dir / "snapshots" / "shadow_state.csv"
        self.live_state = self.data_dir / "snapshots" / "live_state.json"
        self.trading_logs_dir = self.data_dir / "snapshots" / "trading_logs"
    
    def generate(self, output_path: Path) -> bool:
        """Generate dashboard HTML file"""
        try:
            # Gather data
            positions = self._load_positions()
            equity_curve = self._load_equity_curve()
            trades = self._load_trades()
            metrics = self._calculate_metrics(equity_curve, trades)
            
            # Generate HTML
            html = self._build_html(positions, equity_curve, trades, metrics)
            
            # Write file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(html)
            
            return True
        except Exception as e:
            print(f"Dashboard generation error: {e}")
            return False
    
    def _load_positions(self) -> List[Dict]:
        """Load current open positions"""
        positions = []
        
        # Prefer live state if available
        if self.live_state.exists():
            try:
                with open(self.live_state, 'r') as f:
                    live_data = json.load(f)
                
                for pos in live_data.get('positions', []):
                    positions.append({
                        'symbol': pos['symbol'],
                        'side': pos['side'],
                        'qty': int(pos['qty']),
                        'entry_price': float(pos['entry_price']),
                        'current_price': float(pos['current_price']),
                        'pnl': float(pos['unrealized_pl']),
                        'pnl_pct': float(pos['unrealized_plpc']),
                        'entry_date': 'N/A',  # Not available from Alpaca position
                        'signal': 0.0,  # Not tracked in live mode
                    })
                return positions
            except Exception as e:
                print(f"Error loading live positions: {e}")
        
        # Fall back to shadow state
        if not self.shadow_state.exists():
            return positions
        
        try:
            df = pd.read_csv(self.shadow_state)
            for _, row in df.iterrows():
                pnl = (row['current_price'] - row['entry_price']) * row['qty']
                if row['side'] == 'short':
                    pnl = -pnl
                pnl_pct = (pnl / (row['entry_price'] * abs(row['qty']))) * 100
                
                positions.append({
                    'symbol': row['symbol'],
                    'side': row['side'],
                    'qty': int(row['qty']),
                    'entry_price': float(row['entry_price']),
                    'current_price': float(row['current_price']),
                    'pnl': round(pnl, 2),
                    'pnl_pct': round(pnl_pct, 2),
                    'entry_date': str(row['entry_date'])[:10],
                    'signal': round(float(row.get('signal_strength', 0)), 2)
                })
        except Exception as e:
            print(f"Error loading positions: {e}")
        
        return positions
    
    def _load_equity_curve(self) -> List[Dict]:
        """Load equity curve from trading logs"""
        equity_data = []
        if not self.trading_logs_dir.exists():
            return equity_data
        
        try:
            for log_file in sorted(self.trading_logs_dir.glob("equity_*.csv")):
                df = pd.read_csv(log_file)
                if 'date' in df.columns and 'equity' in df.columns:
                    for _, row in df.iterrows():
                        equity_data.append({
                            'date': str(row['date'])[:10],
                            'equity': float(row['equity']),
                            'daily_return': float(row.get('daily_return', 0)) * 100
                        })
        except Exception as e:
            print(f"Error loading equity curve: {e}")
        
        return sorted(equity_data, key=lambda x: x['date'])
    
    def _load_trades(self) -> List[Dict]:
        """Load completed trades"""
        trades = []
        
        # Prefer live trades if available
        if self.live_state.exists():
            try:
                with open(self.live_state, 'r') as f:
                    live_data = json.load(f)
                
                for trade in live_data.get('recent_trades', []):
                    # Group buy/sell pairs by symbol to calculate P&L
                    # For now, just display the orders
                    exec_date = trade['submitted_at'][:10] if trade.get('submitted_at') else 'N/A'
                    trades.append({
                        'symbol': trade['symbol'],
                        'side': trade['side'],
                        'entry_date': exec_date,
                        'exit_date': exec_date,
                        'entry_price': float(trade.get('filled_price', 0)),
                        'exit_price': float(trade.get('filled_price', 0)),
                        'pnl': 0.0,  # Can't calculate without matching pairs
                        'pnl_pct': 0.0,
                        'holding_days': 0,
                        'exit_reason': f"Signal exit (z={trade.get('signal_z', 'N/A')})" if 'signal_z' in trade else "Order filled"
                    })
                
                if trades:
                    return sorted(trades, key=lambda x: x['exit_date'], reverse=True)[:50]
            except Exception as e:
                print(f"Error loading live trades: {e}")
        
        # Fall back to shadow mode trade logs
        if not self.trading_logs_dir.exists():
            return trades
        
        try:
            for log_file in sorted(self.trading_logs_dir.glob("trades_*.csv")):
                df = pd.read_csv(log_file)
                for _, row in df.iterrows():
                    trades.append({
                        'symbol': row['symbol'],
                        'side': row['side'],
                        'entry_date': str(row['entry_date'])[:10],
                        'exit_date': str(row.get('exit_date', ''))[:10],
                        'entry_price': float(row.get('entry_price', 0)),
                        'exit_price': float(row.get('exit_price', 0)),
                        'pnl': float(row.get('pnl', 0)),
                        'pnl_pct': round(float(row.get('pnl_pct', 0)) * 100, 2),
                        'holding_days': int(row.get('holding_days', 0)),
                        'exit_reason': row.get('exit_reason', 'Open')
                    })
        except Exception as e:
            print(f"Error loading trades: {e}")
        
        return sorted(trades, key=lambda x: x['exit_date'], reverse=True)
    
    def _calculate_metrics(self, equity_curve: List[Dict], trades: List[Dict]) -> Dict:
        """Calculate performance metrics"""
        metrics = {
            'current_equity': 100000.0,
            'total_return': 0.0,
            'total_return_pct': 0.0,
            'total_trades': len(trades),
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
        }
        
        # Try to get current equity from live state first
        if self.live_state.exists():
            try:
                with open(self.live_state, 'r') as f:
                    live_data = json.load(f)
                metrics['current_equity'] = live_data['account']['portfolio_value']
                # Calculate return from $1M starting capital (Alpaca paper account)
                initial = 1000000.0
                current = metrics['current_equity']
                metrics['total_return'] = current - initial
                metrics['total_return_pct'] = ((current - initial) / initial) * 100
            except Exception as e:
                print(f"Error loading live equity: {e}")
        
        # Fall back to equity curve if no live state
        elif equity_curve:
            initial = equity_curve[0]['equity']
            current = equity_curve[-1]['equity']
            metrics['current_equity'] = current
            metrics['total_return'] = current - initial
            metrics['total_return_pct'] = ((current - initial) / initial) * 100
        
        # Trade statistics
        if trades:
            wins = [t for t in trades if t['pnl_pct'] > 0]
            losses = [t for t in trades if t['pnl_pct'] <= 0]
            
            metrics['win_rate'] = (len(wins) / len(trades)) * 100 if trades else 0
            metrics['avg_win'] = np.mean([t['pnl_pct'] for t in wins]) if wins else 0
            metrics['avg_loss'] = np.mean([t['pnl_pct'] for t in losses]) if losses else 0
        
        # Max drawdown and Sharpe
        if len(equity_curve) > 1:
            equity_series = pd.Series([e['equity'] for e in equity_curve])
            running_max = equity_series.cummax()
            drawdown = (equity_series - running_max) / running_max
            metrics['max_drawdown'] = abs(drawdown.min()) * 100
            
            returns = equity_series.pct_change().dropna()
            if len(returns) > 0 and returns.std() > 0:
                metrics['sharpe_ratio'] = (returns.mean() / returns.std()) * np.sqrt(252)
        
        return metrics
    
    def _build_html(
        self,
        positions: List[Dict],
        equity_curve: List[Dict],
        trades: List[Dict],
        metrics: Dict
    ) -> str:
        """Build the HTML dashboard"""
        
        # Encode data for JavaScript
        equity_json = json.dumps(equity_curve)
        positions_json = json.dumps(positions)
        trades_json = json.dumps(trades[:50])  # Last 50 trades
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quant Trading Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: #0d1117;
            color: #c9d1d9;
            padding: 20px;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{ 
            font-size: 2em; 
            margin-bottom: 10px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        .timestamp {{
            color: #8b949e;
            font-size: 0.9em;
            margin-bottom: 30px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 20px;
        }}
        .metric-label {{ color: #8b949e; font-size: 0.85em; margin-bottom: 8px; }}
        .metric-value {{ font-size: 1.8em; font-weight: 600; }}
        .positive {{ color: #3fb950; }}
        .negative {{ color: #f85149; }}
        .section {{
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 25px;
            margin-bottom: 25px;
        }}
        .section-title {{ font-size: 1.3em; margin-bottom: 20px; font-weight: 600; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th {{ 
            text-align: left; 
            padding: 12px; 
            border-bottom: 2px solid #30363d;
            color: #8b949e;
            font-weight: 500;
            font-size: 0.9em;
        }}
        td {{ padding: 12px; border-bottom: 1px solid #21262d; }}
        .badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.85em;
            font-weight: 500;
        }}
        .badge-long {{ background: #238636; color: #fff; }}
        .badge-short {{ background: #da3633; color: #fff; }}
        #equityChart {{ max-height: 400px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📈 Quant Trading Dashboard</h1>
        <div class="timestamp">Last Updated: {timestamp}</div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Portfolio Value</div>
                <div class="metric-value">${metrics['current_equity']:,.2f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Total Return</div>
                <div class="metric-value {'positive' if metrics['total_return'] > 0 else 'negative'}">
                    {metrics['total_return_pct']:+.2f}%
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Win Rate</div>
                <div class="metric-value">{metrics['win_rate']:.1f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Total Trades</div>
                <div class="metric-value">{metrics['total_trades']}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Sharpe Ratio</div>
                <div class="metric-value">{metrics['sharpe_ratio']:.2f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Max Drawdown</div>
                <div class="metric-value negative">{metrics['max_drawdown']:.2f}%</div>
            </div>
        </div>
        
        <div class="section">
            <div class="section-title">Open Positions ({len(positions)})</div>
            <div id="positions-container">
                {'<p style="color: #8b949e;">No open positions</p>' if not positions else ''}
            </div>
        </div>
        
        <div class="section">
            <div class="section-title">Equity Curve</div>
            <canvas id="equityChart"></canvas>
        </div>
        
        <div class="section">
            <div class="section-title">Recent Trades (Last 50)</div>
            <div id="trades-container" style="overflow-x: auto;">
                {'<p style="color: #8b949e;">No trades yet</p>' if not trades else ''}
            </div>
        </div>
    </div>
    
    <script>
        // Data
        const equityData = {equity_json};
        const positionsData = {positions_json};
        const tradesData = {trades_json};
        
        // Render positions table
        if (positionsData.length > 0) {{
            const posHtml = `
                <table>
                    <thead>
                        <tr>
                            <th>Symbol</th>
                            <th>Side</th>
                            <th>Qty</th>
                            <th>Entry</th>
                            <th>Current</th>
                            <th>P&L</th>
                            <th>P&L %</th>
                            <th>Entry Date</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${{positionsData.map(p => `
                            <tr>
                                <td><strong>${{p.symbol}}</strong></td>
                                <td><span class="badge badge-${{p.side}}">${{p.side.toUpperCase()}}</span></td>
                                <td>${{p.qty}}</td>
                                <td>$$${{p.entry_price.toFixed(2)}}</td>
                                <td>$$${{p.current_price.toFixed(2)}}</td>
                                <td class="${{p.pnl >= 0 ? 'positive' : 'negative'}}">$$${{p.pnl.toFixed(2)}}</td>
                                <td class="${{p.pnl_pct >= 0 ? 'positive' : 'negative'}}">${{p.pnl_pct >= 0 ? '+' : ''}}${{p.pnl_pct.toFixed(2)}}%</td>
                                <td>${{p.entry_date}}</td>
                            </tr>
                        `).join('')}}
                    </tbody>
                </table>
            `;
            document.getElementById('positions-container').innerHTML = posHtml;
        }}
        
        // Render trades table
        if (tradesData.length > 0) {{
            const tradesHtml = `
                <table>
                    <thead>
                        <tr>
                            <th>Symbol</th>
                            <th>Side</th>
                            <th>Entry Date</th>
                            <th>Exit Date</th>
                            <th>Days</th>
                            <th>P&L %</th>
                            <th>Exit Reason</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${{tradesData.map(t => `
                            <tr>
                                <td><strong>${{t.symbol}}</strong></td>
                                <td><span class="badge badge-${{t.side}}">${{t.side.toUpperCase()}}</span></td>
                                <td>${{t.entry_date}}</td>
                                <td>${{t.exit_date}}</td>
                                <td>${{t.holding_days}}</td>
                                <td class="${{t.pnl_pct >= 0 ? 'positive' : 'negative'}}">${{t.pnl_pct >= 0 ? '+' : ''}}${{t.pnl_pct.toFixed(2)}}%</td>
                                <td style="font-size: 0.9em; color: #8b949e;">${{t.exit_reason}}</td>
                            </tr>
                        `).join('')}}
                    </tbody>
                </table>
            `;
            document.getElementById('trades-container').innerHTML = tradesHtml;
        }}
        
        // Equity curve chart
        if (equityData.length > 0) {{
            const ctx = document.getElementById('equityChart').getContext('2d');
            new Chart(ctx, {{
                type: 'line',
                data: {{
                    labels: equityData.map(d => d.date),
                    datasets: [{{
                        label: 'Equity',
                        data: equityData.map(d => d.equity),
                        borderColor: '#667eea',
                        backgroundColor: 'rgba(102, 126, 234, 0.1)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.4
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: true,
                    plugins: {{
                        legend: {{ display: false }},
                        tooltip: {{
                            backgroundColor: '#161b22',
                            borderColor: '#30363d',
                            borderWidth: 1,
                            callbacks: {{
                                label: (context) => `Equity: $$${{context.parsed.y.toLocaleString(undefined, {{minimumFractionDigits: 2, maximumFractionDigits: 2}})}}`
                            }}
                        }}
                    }},
                    scales: {{
                        y: {{
                            grid: {{ color: '#21262d' }},
                            ticks: {{ 
                                color: '#8b949e',
                                callback: (value) => '$' + value.toLocaleString()
                            }}
                        }},
                        x: {{
                            grid: {{ color: '#21262d' }},
                            ticks: {{ 
                                color: '#8b949e',
                                maxRotation: 45,
                                minRotation: 45
                            }}
                        }}
                    }}
                }}
            }});
        }}
    </script>
</body>
</html>"""
        return html


if __name__ == "__main__":
    # Test generator
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    generator = DashboardGenerator(project_root)
    output = project_root / "docs" / "index.html"
    if generator.generate(output):
        print(f"✅ Dashboard generated: {output}")
    else:
        print("❌ Dashboard generation failed")
