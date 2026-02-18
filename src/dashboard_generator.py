"""
Live Dashboard Generator

Generates a self-updating HTML dashboard that:
1. Embeds position/account/trade data from the server
2. Fetches LIVE prices (including after-hours) via Yahoo Finance on page load
3. Auto-refreshes prices every 3 seconds
4. Equity curve from equity_history.json
5. Enhanced position details (day change, market value, % portfolio, days held)
6. Watchlist with TradingView mini-charts + Yahoo Finance links
7. Beautiful dark theme with modern financial dashboard UX

Architecture:
  - Server side (Python): Reads live_state.json + equity_history.json,
    embeds data as JSON in the HTML
  - Client side (JS): On page load, fetches current/after-hours quotes
    from Yahoo Finance via CORS proxy, updates P&L in real-time
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
import numpy as np


class DashboardGenerator:
    """Generate a live-updating HTML dashboard from trading data"""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.data_dir = project_root / "data"
        self.shadow_state = self.data_dir / "snapshots" / "shadow_state.csv"
        self.live_state = self.data_dir / "snapshots" / "live_state.json"
        self.equity_history = self.data_dir / "snapshots" / "equity_history.json"
        self.trading_logs_dir = self.data_dir / "snapshots" / "trading_logs"

    def generate(self, output_path: Path) -> bool:
        """Generate dashboard HTML file"""
        try:
            positions = self._load_positions()
            trades = self._load_trades()
            account = self._load_account()
            equity_curve = self._load_equity_curve()
            metrics = self._calculate_metrics(account, equity_curve, trades)

            html = self._build_html(positions, trades, account, equity_curve, metrics)

            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(html)
            return True
        except Exception as e:
            print(f"Dashboard generation error: {e}")
            import traceback
            traceback.print_exc()
            return False

    # ── Data Loading ──────────────────────────────────────────────────────

    def _load_account(self) -> Dict:
        """Load account summary from live_state.json"""
        default = {
            "equity": 0, "cash": 0, "portfolio_value": 0, "buying_power": 0
        }
        if self.live_state.exists():
            try:
                with open(self.live_state, "r") as f:
                    data = json.load(f)
                return data.get("account", default)
            except Exception:
                pass
        return default

    def _load_positions(self) -> List[Dict]:
        """Load current open positions (prefer live, fallback to shadow)"""
        positions = []

        if self.live_state.exists():
            try:
                with open(self.live_state, "r") as f:
                    live_data = json.load(f)
                for pos in live_data.get("positions", []):
                    positions.append({
                        "symbol": pos["symbol"],
                        "side": pos.get("side", "long"),
                        "qty": int(pos["qty"]),
                        "entry_price": float(pos["entry_price"]),
                        "current_price": float(pos["current_price"]),
                        "market_value": float(pos.get("market_value", 0)),
                        "pnl": float(pos.get("unrealized_pl", 0)),
                        "pnl_pct": float(pos.get("unrealized_plpc", 0)),
                    })
                return positions
            except Exception as e:
                print(f"Error loading live positions: {e}")

        # Fallback: shadow state
        if self.shadow_state.exists():
            try:
                df = pd.read_csv(self.shadow_state)
                for _, row in df.iterrows():
                    pnl = (row["current_price"] - row["entry_price"]) * row["qty"]
                    if row["side"] == "short":
                        pnl = -pnl
                    pnl_pct = (pnl / (row["entry_price"] * abs(row["qty"]))) * 100
                    positions.append({
                        "symbol": row["symbol"],
                        "side": row["side"],
                        "qty": int(row["qty"]),
                        "entry_price": float(row["entry_price"]),
                        "current_price": float(row["current_price"]),
                        "market_value": float(row["current_price"] * abs(row["qty"])),
                        "pnl": round(pnl, 2),
                        "pnl_pct": round(pnl_pct, 2),
                    })
            except Exception as e:
                print(f"Error loading shadow positions: {e}")

        return positions

    def _load_trades(self) -> List[Dict]:
        """Load recent trades"""
        trades = []

        if self.live_state.exists():
            try:
                with open(self.live_state, "r") as f:
                    live_data = json.load(f)
                for trade in live_data.get("recent_trades", []):
                    exec_date = trade.get("submitted_at", "")[:10] if trade.get("submitted_at") else "N/A"
                    trades.append({
                        "symbol": trade["symbol"],
                        "side": trade["side"],
                        "qty": float(trade.get("qty", 0)),
                        "price": float(trade.get("filled_price", 0)),
                        "date": exec_date,
                        "status": "filled",
                    })
                if trades:
                    return sorted(trades, key=lambda x: x["date"], reverse=True)[:50]
            except Exception as e:
                print(f"Error loading live trades: {e}")

        # Fallback: shadow trade logs
        if self.trading_logs_dir and self.trading_logs_dir.exists():
            try:
                for log_file in sorted(self.trading_logs_dir.glob("trades_*.csv")):
                    df = pd.read_csv(log_file)
                    for _, row in df.iterrows():
                        trades.append({
                            "symbol": row["symbol"],
                            "side": row["side"],
                            "qty": int(row.get("qty", 0)),
                            "price": float(row.get("exit_price", row.get("entry_price", 0))),
                            "date": str(row.get("exit_date", row.get("entry_date", "")))[:10],
                            "pnl_pct": round(float(row.get("pnl_pct", 0)) * 100, 2),
                            "status": "closed",
                        })
            except Exception as e:
                print(f"Error loading trade logs: {e}")

        return sorted(trades, key=lambda x: x["date"], reverse=True)[:50]

    def _load_equity_curve(self) -> List[Dict]:
        """Load equity curve from equity_history.json (primary) or trading logs (fallback)"""
        equity_data = []

        # Primary: equity_history.json (accumulated by main_trader.py)
        if self.equity_history.exists():
            try:
                with open(self.equity_history, "r") as f:
                    history = json.load(f)
                for entry in history:
                    equity_data.append({
                        "date": entry["timestamp"],
                        "equity": float(entry["equity"]),
                    })
                if equity_data:
                    return equity_data
            except Exception as e:
                print(f"Error loading equity history: {e}")

        # Fallback: equity_*.csv from trading logs
        if self.trading_logs_dir and self.trading_logs_dir.exists():
            try:
                for log_file in sorted(self.trading_logs_dir.glob("equity_*.csv")):
                    df = pd.read_csv(log_file)
                    if "date" in df.columns and "equity" in df.columns:
                        for _, row in df.iterrows():
                            equity_data.append({
                                "date": str(row["date"])[:10],
                                "equity": float(row["equity"]),
                            })
            except Exception as e:
                print(f"Error loading equity curve: {e}")

        return sorted(equity_data, key=lambda x: x["date"])

    def _calculate_metrics(self, account: Dict, equity_curve: List[Dict], trades: List[Dict]) -> Dict:
        """Calculate summary metrics"""
        initial_capital = 1_000_000.0  # Alpaca paper account
        current = float(account.get("portfolio_value", 0)) or initial_capital
        metrics = {
            "portfolio_value": current,
            "cash": float(account.get("cash", 0)),
            "buying_power": float(account.get("buying_power", 0)),
            "total_return_pct": ((current - initial_capital) / initial_capital) * 100,
            "total_trades": len(trades),
            "win_rate": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
        }
        # Win rate from trades with P&L data
        pnl_trades = [t for t in trades if "pnl_pct" in t and t.get("pnl_pct", 0) != 0]
        if pnl_trades:
            wins = sum(1 for t in pnl_trades if t["pnl_pct"] > 0)
            metrics["win_rate"] = (wins / len(pnl_trades)) * 100

        if len(equity_curve) > 1:
            eq = pd.Series([e["equity"] for e in equity_curve])
            dd = (eq - eq.cummax()) / eq.cummax()
            metrics["max_drawdown"] = abs(dd.min()) * 100
            rets = eq.pct_change().dropna()
            if len(rets) > 0 and rets.std() > 0:
                metrics["sharpe_ratio"] = (rets.mean() / rets.std()) * np.sqrt(252)
        return metrics

    # ── HTML Builder ──────────────────────────────────────────────────────

    def _build_html(
        self,
        positions: List[Dict],
        trades: List[Dict],
        account: Dict,
        equity_curve: List[Dict],
        metrics: Dict,
    ) -> str:
        """Build the self-updating HTML dashboard"""

        data_blob = json.dumps({
            "positions": positions,
            "trades": trades,
            "account": account,
            "equity_curve": equity_curve,
            "metrics": metrics,
            "generated_utc": datetime.now(tz=__import__('datetime').timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        })

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quant Trading Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
{_CSS}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <header class="header">
            <div class="header-left">
                <h1>Quant Trading Dashboard</h1>
                <div class="subtitle">Mean Reversion Strategy &middot; Alpaca Paper Trading</div>
            </div>
            <div class="header-right">
                <div id="market-status" class="market-badge">Loading...</div>
                <div id="last-update" class="timestamp"></div>
                <div id="refresh-countdown" class="timestamp"></div>
            </div>
        </header>

        <!-- Account Summary Cards -->
        <div class="metrics-grid" id="metrics-grid"></div>

        <!-- Positions -->
        <div class="section">
            <div class="section-header">
                <div class="section-title">Open Positions</div>
                <div id="price-status" class="price-status"></div>
            </div>
            <div id="positions-table" class="table-wrap"></div>
        </div>

        <!-- Watchlist / TradingView Charts -->
        <div class="section" id="watchlist-section" style="display:none;">
            <div class="section-title">Watchlist &amp; Charts</div>
            <div id="watchlist-grid" class="watchlist-grid"></div>
        </div>

        <!-- Equity Curve -->
        <div class="section">
            <div class="section-title">Equity Curve</div>
            <div class="chart-container">
                <canvas id="equityChart"></canvas>
            </div>
            <p id="no-equity" class="muted" style="display:none;">
                Equity curve data will appear after trading cycles accumulate snapshots.
            </p>
        </div>

        <!-- Recent Trades -->
        <div class="section">
            <div class="section-title">Recent Trades</div>
            <div id="trades-table" class="table-wrap"></div>
        </div>

        <footer class="footer">
            Dashboard data pushed after each trading cycle &middot;
            Live prices fetched from Yahoo Finance every 3s &middot;
            <a href="https://github.com/Shimmy-Shams/Quant" target="_blank">GitHub</a>
        </footer>
    </div>

    <script>
    // ── Embedded server-side data ─────────────────────────────────────────
    const DATA = {data_blob};

    // ── Configuration ─────────────────────────────────────────────────────
    const REFRESH_INTERVAL = 3;           // seconds between price refreshes
    const INITIAL_CAPITAL   = 1000000;
    const CORS_PROXIES = [
        url => 'https://corsproxy.io/?' + encodeURIComponent(url),
        url => 'https://api.allorigins.win/raw?url=' + encodeURIComponent(url),
    ];

    let countdown = REFRESH_INTERVAL;
    let liveQuotes = {{}};  // symbol -> quote data
    let equityChart = null;

    // ── Utilities ─────────────────────────────────────────────────────────
    const fmt  = (v, d=2) => v == null ? '—' : v.toLocaleString(undefined, {{minimumFractionDigits:d, maximumFractionDigits:d}});
    const fmtD = (v)      => v == null ? '—' : '$' + fmt(v);
    const fmtP = (v)      => v == null ? '—' : (v >= 0 ? '+' : '') + fmt(v) + '%';
    const cls  = (v)      => v >= 0 ? 'positive' : 'negative';

    // ── Yahoo Finance live price fetcher ──────────────────────────────────
    async function fetchLiveQuotes(symbols) {{
        if (!symbols.length) return {{}};
        // Convert crypto symbols: ETHUSD -> ETH-USD for Yahoo
        const yahooSymbols = symbols.map(s => {{
            if (s.endsWith('USD') && s.length <= 7 && !s.includes('-')) {{
                return s.slice(0, -3) + '-USD';
            }}
            return s;
        }});
        const symStr = yahooSymbols.join(',');
        const url = `https://query1.finance.yahoo.com/v7/finance/quote?symbols=${{symStr}}`;

        for (const makeProxy of CORS_PROXIES) {{
            try {{
                const resp = await fetch(makeProxy(url), {{ signal: AbortSignal.timeout(8000) }});
                if (!resp.ok) continue;
                const json = await resp.json();
                const results = json?.quoteResponse?.result;
                if (!results) continue;
                const out = {{}};
                for (const q of results) {{
                    // Map back: ETH-USD -> ETHUSD
                    let sym = q.symbol;
                    const origSym = symbols.find(s => {{
                        if (s.endsWith('USD') && s.length <= 7 && !s.includes('-')) {{
                            return s.slice(0,-3) + '-USD' === sym;
                        }}
                        return s === sym;
                    }});
                    const key = origSym || sym;
                    out[key] = {{
                        price:           q.regularMarketPrice,
                        change:          q.regularMarketChange,
                        changePct:       q.regularMarketChangePercent,
                        marketState:     q.marketState,
                        prePrice:        q.preMarketPrice,
                        preChange:       q.preMarketChange,
                        preChangePct:    q.preMarketChangePercent,
                        postPrice:       q.postMarketPrice,
                        postChange:      q.postMarketChange,
                        postChangePct:   q.postMarketChangePercent,
                        name:            q.shortName || q.symbol,
                        dayHigh:         q.regularMarketDayHigh,
                        dayLow:          q.regularMarketDayLow,
                        volume:          q.regularMarketVolume,
                        avgVolume:       q.averageDailyVolume3Month,
                        fiftyTwoHigh:    q.fiftyTwoWeekHigh,
                        fiftyTwoLow:     q.fiftyTwoWeekLow,
                        previousClose:   q.regularMarketPreviousClose,
                    }};
                }}
                return out;
            }} catch (e) {{
                console.warn('Proxy failed:', e.message);
            }}
        }}
        return {{}};
    }}

    // ── Render helpers ────────────────────────────────────────────────────

    function renderMetrics() {{
        const m = DATA.metrics;
        const a = DATA.account;
        const pv = livePortfolioValue();
        const dayPnl = calculateDayPnl();

        const cards = [
            {{ label: 'Portfolio Value', value: fmtD(pv), cls: '' }},
            {{ label: 'Cash', value: fmtD(a.cash), cls: '' }},
            {{ label: 'Day P&L', value: fmtD(dayPnl), cls: cls(dayPnl) }},
            {{ label: 'Total Return', value: fmtP(((pv - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100), cls: cls(pv - INITIAL_CAPITAL) }},
            {{ label: 'Open Positions', value: DATA.positions.length, cls: '' }},
            {{ label: 'Trades Executed', value: m.total_trades, cls: '' }},
        ];

        document.getElementById('metrics-grid').innerHTML = cards.map(c => `
            <div class="metric-card">
                <div class="metric-label">${{c.label}}</div>
                <div class="metric-value ${{c.cls}}">${{c.value}}</div>
            </div>
        `).join('');
    }}

    function calculateDayPnl() {{
        let total = 0;
        for (const p of DATA.positions) {{
            const q = liveQuotes[p.symbol];
            if (q && q.previousClose) {{
                const curPrice = effectivePrice(p.symbol, q) || p.current_price;
                const dayChange = curPrice - q.previousClose;
                total += dayChange * Math.abs(p.qty) * (p.side === 'short' ? -1 : 1);
            }}
        }}
        return total;
    }}

    function livePortfolioValue() {{
        let positionValue = 0;
        for (const p of DATA.positions) {{
            const q = liveQuotes[p.symbol];
            const curPrice = effectivePrice(p.symbol, q) || p.current_price;
            positionValue += curPrice * Math.abs(p.qty);
        }}
        if (Object.keys(liveQuotes).length > 0 && DATA.positions.length > 0) {{
            return DATA.account.cash + positionValue;
        }}
        return DATA.account.portfolio_value || INITIAL_CAPITAL;
    }}

    function effectivePrice(symbol, quote) {{
        if (!quote) return null;
        const state = quote.marketState;
        if (state === 'POST' && quote.postPrice) return quote.postPrice;
        if (state === 'PRE'  && quote.prePrice)  return quote.prePrice;
        return quote.price;
    }}

    function renderPositions() {{
        const positions = DATA.positions;
        if (!positions.length) {{
            document.getElementById('positions-table').innerHTML =
                '<p class="muted">No open positions</p>';
            return;
        }}

        const pv = livePortfolioValue();
        let rows = '';
        let totalPnl = 0;
        let totalMktVal = 0;

        for (const p of positions) {{
            const q = liveQuotes[p.symbol];
            const livePrice = effectivePrice(p.symbol, q);
            const curPrice = livePrice || p.current_price;
            const pnl = p.side === 'short'
                ? (p.entry_price - curPrice) * Math.abs(p.qty)
                : (curPrice - p.entry_price) * Math.abs(p.qty);
            const pnlPct = (pnl / (p.entry_price * Math.abs(p.qty))) * 100;
            const mktVal = curPrice * Math.abs(p.qty);
            const pctOfPortfolio = pv > 0 ? (mktVal / pv) * 100 : 0;
            totalPnl += pnl;
            totalMktVal += mktVal;

            // Day change from Yahoo
            let dayChangePct = null;
            let dayChangeAmt = null;
            if (q) {{
                dayChangePct = q.changePct;
                dayChangeAmt = q.change;
            }}

            // After-hours / pre-market badge
            let ahBadge = '';
            if (q) {{
                const state = q.marketState;
                if (state === 'POST' && q.postPrice) {{
                    ahBadge = `<div class="ah-badge">AH $$${{fmt(q.postPrice)}} <span class="${{cls(q.postChangePct)}}">${{fmtP(q.postChangePct)}}</span></div>`;
                }} else if (state === 'PRE' && q.prePrice) {{
                    ahBadge = `<div class="ah-badge">PM $$${{fmt(q.prePrice)}} <span class="${{cls(q.preChangePct)}}">${{fmtP(q.preChangePct)}}</span></div>`;
                }} else if (state === 'REGULAR') {{
                    ahBadge = `<div class="ah-badge live-dot">LIVE</div>`;
                }}
            }}

            rows += `<tr>
                <td>
                    <strong>${{p.symbol}}</strong>
                    ${{q ? `<div class="stock-name">${{q.name || ''}}</div>` : ''}}
                </td>
                <td><span class="badge badge-${{p.side}}">${{p.side.toUpperCase()}}</span></td>
                <td>${{Math.abs(p.qty)}}</td>
                <td>$$${{fmt(p.entry_price)}}</td>
                <td>
                    $$${{fmt(curPrice)}}
                    ${{ahBadge}}
                </td>
                <td class="${{dayChangePct != null ? cls(dayChangePct) : ''}}">${{dayChangePct != null ? fmtP(dayChangePct) : '—'}}</td>
                <td>$$${{fmt(mktVal)}}</td>
                <td>${{fmt(pctOfPortfolio, 1)}}%</td>
                <td class="${{cls(pnl)}}">$$${{fmt(pnl)}}</td>
                <td class="${{cls(pnlPct)}}">${{fmtP(pnlPct)}}</td>
            </tr>`;
        }}

        document.getElementById('positions-table').innerHTML = `
            <table>
                <thead><tr>
                    <th>Symbol</th><th>Side</th><th>Qty</th>
                    <th>Entry</th><th>Current</th><th>Day Chg</th>
                    <th>Mkt Value</th><th>% Port</th>
                    <th>P&amp;L</th><th>P&amp;L %</th>
                </tr></thead>
                <tbody>${{rows}}</tbody>
                <tfoot><tr>
                    <td colspan="6" style="text-align:right;font-weight:600;">Totals</td>
                    <td style="font-weight:600;">$$${{fmt(totalMktVal)}}</td>
                    <td style="font-weight:600;">${{fmt(pv > 0 ? (totalMktVal / pv) * 100 : 0, 1)}}%</td>
                    <td class="${{cls(totalPnl)}}" style="font-weight:600;">$$${{fmt(totalPnl)}}</td>
                    <td></td>
                </tr></tfoot>
            </table>`;
    }}

    function renderWatchlist() {{
        const positions = DATA.positions;
        if (!positions.length) {{
            document.getElementById('watchlist-section').style.display = 'none';
            return;
        }}

        document.getElementById('watchlist-section').style.display = 'block';

        const cards = positions.map(p => {{
            const q = liveQuotes[p.symbol];
            // TradingView symbol format
            const tvSymbol = p.symbol.endsWith('USD') && p.symbol.length <= 7 && !p.symbol.includes('-')
                ? 'COINBASE:' + p.symbol.slice(0, -3) + 'USD'
                : p.symbol;
            // Yahoo Finance symbol
            const yahooSym = p.symbol.endsWith('USD') && p.symbol.length <= 7 && !p.symbol.includes('-')
                ? p.symbol.slice(0, -3) + '-USD'
                : p.symbol;

            let priceInfo = '';
            if (q) {{
                priceInfo = `
                    <div class="watch-price">$$${{fmt(q.price)}}</div>
                    <div class="${{cls(q.changePct)}}" style="font-size:0.9em;">${{fmtP(q.changePct)}}</div>
                    <div class="watch-details">
                        <span>H: $$${{fmt(q.dayHigh)}}</span>
                        <span>L: $$${{fmt(q.dayLow)}}</span>
                    </div>
                    <div class="watch-details">
                        <span>Vol: ${{q.volume ? (q.volume / 1e6).toFixed(1) + 'M' : '—'}}</span>
                        <span>52H: $$${{fmt(q.fiftyTwoHigh)}}</span>
                    </div>
                `;
            }}

            return `
                <div class="watch-card">
                    <div class="watch-header">
                        <div>
                            <strong>${{p.symbol}}</strong>
                            ${{q ? `<span class="stock-name"> ${{q.name}}</span>` : ''}}
                        </div>
                        <div class="watch-links">
                            <a href="https://finance.yahoo.com/quote/${{yahooSym}}" target="_blank" title="Yahoo Finance">Y!</a>
                            <a href="https://www.tradingview.com/chart/?symbol=${{tvSymbol}}" target="_blank" title="TradingView">TV</a>
                        </div>
                    </div>
                    ${{priceInfo}}
                    <div class="tradingview-widget">
                        <iframe src="https://s.tradingview.com/widgetembed/?symbol=${{tvSymbol}}&interval=D&hidesidetoolbar=1&symboledit=0&saveimage=0&toolbarbg=0d1117&studies=[]&theme=dark&style=1&timezone=exchange&withdateranges=0&showpopupbutton=0&studies_overrides=%7B%7D&overrides=%7B%7D&enabled_features=[]&disabled_features=[]&locale=en&utm_source=&utm_medium=widget&utm_campaign=chart&hideideas=1&hidevolume=1&padding=0"
                            style="width:100%;height:200px;border:none;" allowtransparency="true"></iframe>
                    </div>
                </div>`;
        }}).join('');

        document.getElementById('watchlist-grid').innerHTML = cards;
    }}

    function renderTrades() {{
        const trades = DATA.trades;
        if (!trades.length) {{
            document.getElementById('trades-table').innerHTML =
                '<p class="muted">No trades yet</p>';
            return;
        }}
        let rows = trades.map(t => `
            <tr>
                <td><strong>${{t.symbol}}</strong></td>
                <td><span class="badge badge-${{t.side}}">${{t.side.toUpperCase()}}</span></td>
                <td>${{t.qty}}</td>
                <td>$$${{fmt(t.price)}}</td>
                <td>${{t.date}}</td>
                <td>${{t.pnl_pct != null ? `<span class="${{cls(t.pnl_pct)}}">${{fmtP(t.pnl_pct)}}</span>` : '—'}}</td>
                <td><span class="badge badge-filled">${{t.status}}</span></td>
            </tr>
        `).join('');

        document.getElementById('trades-table').innerHTML = `
            <table>
                <thead><tr>
                    <th>Symbol</th><th>Side</th><th>Qty</th>
                    <th>Price</th><th>Date</th><th>P&amp;L</th><th>Status</th>
                </tr></thead>
                <tbody>${{rows}}</tbody>
            </table>`;
    }}

    function renderEquityChart() {{
        const eq = DATA.equity_curve;
        if (!eq.length) {{
            document.getElementById('no-equity').style.display = 'block';
            return;
        }}
        document.getElementById('no-equity').style.display = 'none';

        // Thin out data if too many points for readability
        let chartData = eq;
        if (eq.length > 300) {{
            const step = Math.ceil(eq.length / 300);
            chartData = eq.filter((_, i) => i % step === 0 || i === eq.length - 1);
        }}

        const ctx = document.getElementById('equityChart').getContext('2d');
        if (equityChart) {{
            equityChart.data.labels = chartData.map(d => d.date);
            equityChart.data.datasets[0].data = chartData.map(d => d.equity);
            equityChart.update('none');
            return;
        }}

        equityChart = new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: chartData.map(d => d.date),
                datasets: [{{
                    label: 'Equity',
                    data: chartData.map(d => d.equity),
                    borderColor: '#667eea',
                    backgroundColor: 'rgba(102,126,234,0.08)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.3,
                    pointRadius: 0,
                    pointHitRadius: 6,
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                plugins: {{
                    legend: {{ display: false }},
                    tooltip: {{
                        backgroundColor: '#161b22',
                        borderColor: '#30363d',
                        borderWidth: 1,
                        callbacks: {{
                            label: ctx => 'Equity: $' + ctx.parsed.y.toLocaleString(undefined, {{minimumFractionDigits:2}})
                        }}
                    }}
                }},
                scales: {{
                    y: {{
                        grid: {{ color: '#21262d' }},
                        ticks: {{ color: '#8b949e', callback: v => '$' + v.toLocaleString() }}
                    }},
                    x: {{
                        grid: {{ color: '#21262d' }},
                        ticks: {{ color: '#8b949e', maxRotation: 45, autoSkipPadding: 20 }}
                    }}
                }}
            }}
        }});
    }}

    function renderMarketStatus() {{
        const anyQuote = Object.values(liveQuotes)[0];
        const el = document.getElementById('market-status');
        if (!anyQuote) {{
            el.textContent = 'Prices: cached';
            el.className = 'market-badge closed';
            return;
        }}
        const state = anyQuote.marketState;
        const labels = {{ REGULAR: 'Market Open', PRE: 'Pre-Market', POST: 'After Hours', CLOSED: 'Market Closed', PREPRE: 'Pre-Market', POSTPOST: 'After Hours' }};
        el.textContent = labels[state] || state;
        el.className = 'market-badge ' + (state === 'REGULAR' ? 'open' : state === 'CLOSED' ? 'closed' : 'extended');
    }}

    function updateTimestamps() {{
        document.getElementById('last-update').textContent =
            'Data snapshot: ' + DATA.generated_utc + ' UTC';
    }}

    // ── Main loop ─────────────────────────────────────────────────────────

    async function refreshPrices() {{
        const symbols = DATA.positions.map(p => p.symbol);
        if (symbols.length === 0) return;

        document.getElementById('price-status').textContent = 'Fetching live prices...';
        const quotes = await fetchLiveQuotes(symbols);
        if (Object.keys(quotes).length > 0) {{
            liveQuotes = quotes;
            document.getElementById('price-status').innerHTML =
                '<span class="live-dot">&#9679;</span> Live prices (auto-refresh every ' + REFRESH_INTERVAL + 's)';
        }} else {{
            document.getElementById('price-status').textContent = 'Using cached prices (live fetch unavailable)';
        }}

        renderPositions();
        renderMetrics();
        renderMarketStatus();
    }}

    function startCountdown() {{
        setInterval(() => {{
            countdown--;
            const el = document.getElementById('refresh-countdown');
            if (countdown <= 0) {{
                countdown = REFRESH_INTERVAL;
                el.textContent = 'Refreshing...';
                refreshPrices();
            }} else {{
                el.textContent = `Next refresh: ${{countdown}}s`;
            }}
        }}, 1000);
    }}

    // ── Init ──────────────────────────────────────────────────────────────
    (async function init() {{
        updateTimestamps();
        renderMetrics();
        renderPositions();
        renderTrades();
        renderEquityChart();
        renderWatchlist();

        await refreshPrices();
        startCountdown();
    }})();
    </script>
</body>
</html>"""


# ══════════════════════════════════════════════════════════════════════════
# CSS (kept separate for readability)
# ══════════════════════════════════════════════════════════════════════════

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #0d1117;
    color: #c9d1d9;
    padding: 24px;
    line-height: 1.5;
}
.container { max-width: 1400px; margin: 0 auto; }

/* Header */
.header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 32px;
    flex-wrap: wrap;
    gap: 16px;
}
.header h1 {
    font-size: 1.8em;
    background: linear-gradient(135deg, #667eea, #764ba2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.subtitle { color: #8b949e; font-size: 0.95em; margin-top: 4px; }
.header-right { text-align: right; }
.timestamp { color: #8b949e; font-size: 0.85em; margin-top: 4px; }

/* Market status badge */
.market-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.85em;
    font-weight: 600;
}
.market-badge.open     { background: #238636; color: #fff; }
.market-badge.closed   { background: #30363d; color: #8b949e; }
.market-badge.extended { background: #9e6a03; color: #fff; }

/* Metric cards */
.metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 16px;
    margin-bottom: 28px;
}
.metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 20px;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #667eea; }
.metric-label { color: #8b949e; font-size: 0.82em; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px; }
.metric-value { font-size: 1.6em; font-weight: 700; }

/* Positive / Negative */
.positive { color: #3fb950; }
.negative { color: #f85149; }

/* Sections */
.section {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 24px;
    margin-bottom: 24px;
}
.section-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; }
.section-title { font-size: 1.2em; font-weight: 600; margin-bottom: 16px; }
.section-header .section-title { margin-bottom: 0; }

/* Tables */
.table-wrap { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; }
th {
    text-align: left;
    padding: 10px 14px;
    border-bottom: 2px solid #30363d;
    color: #8b949e;
    font-weight: 500;
    font-size: 0.82em;
    text-transform: uppercase;
    letter-spacing: 0.3px;
    white-space: nowrap;
}
td { padding: 10px 14px; border-bottom: 1px solid #21262d; white-space: nowrap; }
tbody tr:hover { background: rgba(102,126,234,0.04); }
tfoot td { border-top: 2px solid #30363d; border-bottom: none; padding-top: 14px; }

/* Badges */
.badge {
    display: inline-block;
    padding: 3px 8px;
    border-radius: 4px;
    font-size: 0.8em;
    font-weight: 600;
    text-transform: uppercase;
}
.badge-long   { background: #238636; color: #fff; }
.badge-short  { background: #da3633; color: #fff; }
.badge-buy    { background: #238636; color: #fff; }
.badge-sell   { background: #da3633; color: #fff; }
.badge-filled { background: #1f6feb; color: #fff; }
.badge-closed { background: #30363d; color: #8b949e; }

/* After-hours info */
.ah-badge {
    font-size: 0.8em;
    color: #d29922;
    margin-top: 2px;
}
.stock-name { font-size: 0.78em; color: #8b949e; }

/* Live dot */
.live-dot { color: #3fb950; }
.live-dot::before { content: ''; display: inline-block; width: 8px; height: 8px; background: #3fb950; border-radius: 50%; margin-right: 6px; animation: pulse 2s infinite; }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }

/* Price status */
.price-status { font-size: 0.85em; color: #8b949e; }

/* Chart */
.chart-container { height: 350px; position: relative; }

/* Watchlist */
.watchlist-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
    gap: 16px;
}
.watch-card {
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 16px;
    transition: border-color 0.2s;
}
.watch-card:hover { border-color: #667eea; }
.watch-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
}
.watch-links a {
    color: #667eea;
    text-decoration: none;
    font-size: 0.85em;
    font-weight: 600;
    margin-left: 10px;
    padding: 2px 8px;
    border: 1px solid #30363d;
    border-radius: 4px;
    transition: all 0.2s;
}
.watch-links a:hover { background: #667eea; color: #fff; border-color: #667eea; }
.watch-price {
    font-size: 1.4em;
    font-weight: 700;
    margin-bottom: 4px;
}
.watch-details {
    display: flex;
    gap: 16px;
    font-size: 0.82em;
    color: #8b949e;
    margin-top: 4px;
}
.tradingview-widget {
    margin-top: 10px;
    border-radius: 6px;
    overflow: hidden;
    border: 1px solid #21262d;
}

/* Footer */
.footer {
    text-align: center;
    color: #484f58;
    font-size: 0.82em;
    margin-top: 20px;
    padding-top: 20px;
    border-top: 1px solid #21262d;
}
.footer a { color: #667eea; text-decoration: none; }
.footer a:hover { text-decoration: underline; }

.muted { color: #484f58; font-style: italic; }

/* Responsive */
@media (max-width: 768px) {
    body { padding: 12px; }
    .header { flex-direction: column; }
    .header-right { text-align: left; }
    .metrics-grid { grid-template-columns: repeat(2, 1fr); }
    .metric-value { font-size: 1.3em; }
    .watchlist-grid { grid-template-columns: 1fr; }
}
"""


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    generator = DashboardGenerator(project_root)
    output = project_root / "docs" / "index.html"
    if generator.generate(output):
        print(f"✅ Dashboard generated: {output}")
    else:
        print("❌ Dashboard generation failed")
