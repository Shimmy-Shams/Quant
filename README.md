# Quantitative Mean Reversion Trading System

An autonomous, headless trading system that identifies and trades statistically mean-reverting equities using Ornstein-Uhlenbeck modeling, Kalman filtering, and multi-signal gated entries — deployed on Alpaca with a live dashboard on GitHub Pages.

## Overview

This system trades a dynamically selected universe of up to 30 US equities, filtering for stocks that exhibit statistically significant mean-reverting behavior (Hurst exponent < 0.5). It generates composite trading signals from multiple quantitative indicators, executes near market close (3:55 PM ET) to match daily bar formation, and runs 24/7 as a systemd service on an Oracle Cloud VM.

**Live Dashboard**: [shimmy-shams.github.io/Quant](https://shimmy-shams.github.io/Quant/)

---

## How It Makes Trade Decisions

The system follows a structured pipeline from universe selection through signal generation to order execution. Every decision is data-driven with no discretionary overrides.

### 1. Universe Selection — Finding Mean-Reverting Stocks

Not all stocks mean-revert. The system filters the investable universe using three statistical tests:

| Test | What It Measures | Threshold |
|------|-----------------|-----------|
| **Hurst Exponent** | Long-memory behavior via R/S analysis. H < 0.5 = mean-reverting, H > 0.5 = trending | H < 0.5 |
| **OU Half-Life** | Speed of mean reversion via Ornstein-Uhlenbeck regression (dP = θ(μ - P)dt). Half-life = ln(2)/\|θ\| | 10–252 days |
| **ADF Test** | Augmented Dickey-Fuller stationarity test. Low p-value = stationary (mean-reverting) | p < 0.05 |

Stocks are ranked by Hurst exponent (ascending) and the top 30 most mean-reverting are selected. Rankings are cached in `data/snapshots/hurst_rankings.csv` and recomputed when stale.

### 2. Signal Generation — The Composite Signal Pipeline

Each stock passes through a multi-layer signal pipeline that produces a single composite score. The system operates in **gated mode** — a gate signal must fire before any trade is considered.

#### Primary Signal: Adaptive Z-Score

The z-score measures how far a stock's price has deviated from its estimated mean:

- **Kalman Filter Z-Score** (default): A state-space model that adaptively estimates the true mean price. The Kalman gain automatically adjusts to market conditions — responding quickly to regime changes while filtering noise in stable periods. Parameters: Q = 1e-5 (process noise), R = 2e-4 (observation noise).
- **Rolling Adaptive Z-Score** (fallback): Uses a lookback window sized to 2× the OU half-life, ensuring the window adapts to each stock's mean-reversion speed.

Both operate on log prices for improved stationarity.

#### Gate Signal: RSI Divergence

The gate signal controls *whether* a trade can be entered at all. The z-score then determines *conviction*.

**RSI Divergence** detects when price and momentum disagree — a leading indicator of reversals:
- **Bullish divergence** (long entry): Price makes a lower low, but RSI makes a higher low → selling momentum is exhausted
- **Bearish divergence** (short entry): Price makes a higher high, but RSI makes a lower high → buying momentum is exhausted

Detection uses vectorized rolling min/max to find local extrema, then checks divergence patterns across a 30-bar lookback. Gate signals persist for 7 days after firing to allow the setup to develop.

**Why gated?** Phase A optimization showed RSI divergence was the only alpha-generating signal (6,130% return, 13.46 Sharpe). Bollinger Bands and RSI levels were destroying returns. Gating eliminates noise entries from the z-score alone.

#### Composite Signal Formula (Gated Mode)

```
composite = gate × (1 + boost × |zscore|)
```

- `gate` = -1 (long), 0 (no trade), or +1 (short) from RSI divergence
- `boost` = 0.5 (z-score magnitude adds conviction but doesn't control direction)
- Result: When gate = 0, no trade. When gate fires, the z-score amplifies the signal strength.

#### OU Predicted Return Gate

Before entering, the system validates the trade has sufficient expected value using the Ornstein-Uhlenbeck model:

$$E[r] = (\mu - P)(1 - e^{-\theta \cdot h})$$

where μ is the estimated mean, P is current price, θ is the mean-reversion speed, and h is the prediction horizon (set to the half-life). Trades are rejected if |E[r]| < 0.5% (the hurdle rate).

#### Dynamic Short Confidence Filter

Short trades require additional conviction. The system computes a confidence score (0–1) from three components:

| Component | Weight | Logic |
|-----------|--------|-------|
| Trend Extension | 40% | How far price is above its 50-day MA (overextended = higher confidence) |
| Momentum Deceleration | 40% | Fast (5d) vs slow (20d) momentum divergence (decelerating = safer short) |
| Volatility Regime | 20% | Short-term/long-term vol ratio (elevated vol = stronger reversion signal) |

Shorts below 0.3 confidence are blocked entirely. Remaining shorts are scaled by their confidence score.

#### Short Acceleration Filter

An additional safety check: if a stock has surged more than 10% in the last 3 days, short entries are blocked regardless of signal strength. This prevents shorting into parabolic moves.

### 3. Position Sizing

Three methods are available, configured via `config.yaml`:

| Method | Logic |
|--------|-------|
| **Equal Weight** | Fixed fraction of equity per position (default: 10% max) |
| **Signal Proportional** | Base size (3%) + scale factor × signal excess above threshold |
| **Volatility Scaled** (active) | Target 15% annualized vol per position. Size = target_vol / realized_vol. Clamped to 2–10% of equity. Uses 60-day realized vol. |

Total portfolio exposure is capped at 100% of equity.

### 4. Entry & Exit Rules

| Rule | Condition |
|------|-----------|
| **Entry** | \|composite signal\| > 1.43 (optimized threshold) |
| **Signal Exit** | Z-score reverts past ±0.50 (long: z > -0.50, short: z < +0.50) |
| **Stop Loss** | P&L < -10% (both longs and shorts) |
| **Take Profit** | P&L > +15% |
| **Max Holding** | 20 days (optimizer consensus from walk-forward periods) |
| **Trailing Stop** | Activates at +2% profit, trails at 5% from peak |
| **Time Decay** | If P&L is within ±1% after 10 days, exit (setup failed) |

### 5. Execution

Trades execute at **3:55 PM ET** — 5 minutes before market close. This ensures:
- The daily price bar is fully formed
- Signals match the backtest engine's daily-close methodology
- Slippage is minimized in deep closing liquidity

**Execution order**: Exits are processed first (freeing capital), then entries. Each entry undergoes a pre-trade risk check:
- Single position ≤ 10% of equity
- Total exposure ≤ 100% of equity
- Market orders via Alpaca API

### 6. Volatility Regime Filter

The system detects three volatility regimes and adjusts position sizing:

| Regime | Condition | Position Multiplier |
|--------|-----------|-------------------|
| Normal | Short-term vol / long-term vol ≤ 1.5 | 1.0× |
| High Vol | Ratio 1.5–2.0 | 0.5× |
| Crisis | Ratio > 2.0 | 0.0× (no trading) |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    main_trader.py                        │
│            Headless 24/7 Trading Loop                    │
│                                                         │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │  Universe    │  │   Signal     │  │   Execution   │  │
│  │  Selection   │→ │   Pipeline   │→ │   Engine      │  │
│  │  (Hurst)     │  │  (Composite) │  │  (Alpaca)     │  │
│  └─────────────┘  └──────────────┘  └───────────────┘  │
│         │                │                  │            │
│         ▼                ▼                  ▼            │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │  Alpaca      │  │  Mean Rev    │  │  Alpaca       │  │
│  │  Data        │  │  Signals     │  │  Executor     │  │
│  │  Adapter     │  │  + Kalman    │  │  + Risk Mgmt  │  │
│  └─────────────┘  └──────────────┘  └───────────────┘  │
│                         │                               │
│                         ▼                               │
│              ┌──────────────────┐                       │
│              │    Dashboard     │                       │
│              │   Generator      │──→ GitHub Pages       │
│              └──────────────────┘                       │
└─────────────────────────────────────────────────────────┘
```

## Operating Modes

| Mode | Description |
|------|-------------|
| **Shadow** | Runs on live data, tracks hypothetical positions, no real orders. For building confidence before going live. |
| **Live** | Submits real orders to Alpaca. Requires `--mode live` flag. |
| **Replay** | Replays historical data day-by-day through the live pipeline. Validates that live execution matches backtest results. |
| **Dashboard-Only** | Refreshes account state and regenerates dashboard without trading (`--update-dashboard-only`). |

---

## Project Structure

```
Quant/
├── config.yaml                  # All strategy parameters (signals, sizing, thresholds)
├── requirements.txt             # Python dependencies
├── src/
│   ├── main_trader.py           # Headless 24/7 trader (entry point)
│   ├── strategy_config.py       # YAML config loader + validation
│   ├── strategies/
│   │   └── mean_reversion.py    # Signal pipeline (Z-score, Kalman, OU, RSI, BB)
│   ├── execution/
│   │   ├── alpaca_executor.py   # Trade decisions → Alpaca orders + risk checks
│   │   └── simulation.py        # Shadow/replay simulation engine
│   ├── backtest/
│   │   ├── engine.py            # Vectorized backtesting framework
│   │   ├── analytics.py         # Performance metrics (Sharpe, drawdown, etc.)
│   │   └── optimizer.py         # Bayesian + grid walk-forward optimization
│   ├── data/
│   │   ├── alpaca_data.py       # Alpaca market data adapter with caching
│   │   ├── collector.py         # Historical data collection
│   │   ├── universe_builder.py  # Stock universe definitions (S&P, NASDAQ, DOW)
│   │   └── yahoo_collector.py   # Yahoo Finance data collection
│   ├── connection/
│   │   ├── alpaca_connection.py # Alpaca API connection management
│   │   └── ib_connection.py     # Interactive Brokers connection (legacy)
│   └── config/
│       └── config.py            # Core configuration management
├── data/
│   ├── historical/daily/        # Cached daily price data (parquet)
│   ├── snapshots/               # Shadow state, equity history, hurst rankings
│   └── logs/                    # Trading logs
├── docs/                        # Dashboard output + documentation
│   └── index.html               # Generated live dashboard
└── deploy/
    ├── setup.sh                 # VM deployment script
    └── quant-trader.service     # systemd service definition
```

## Configuration

All strategy parameters are centralized in `config.yaml`. Key sections:

- **signals**: Z-score windows, Hurst thresholds, Kalman filter params, OU prediction, composite weights, gated mode settings, dynamic short filter
- **backtest**: Capital, transaction costs, position sizing method, entry/exit thresholds, risk management (stop loss, take profit, trailing stop, time decay)
- **optimization**: Walk-forward settings, Bayesian/grid search ranges, objective metric
- **universe**: Statistical test toggles (Hurst, half-life, ADF)
- **alpaca**: Data feed, universe size, execution mode

---

## Quick Start

### 1. Install Dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Alpaca Credentials

```bash
# Create .env file with your Alpaca API keys
cat > .env << EOF
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets
EOF
```

### 3. Run in Shadow Mode (Recommended First)

```bash
python src/main_trader.py --mode shadow --once
```

### 4. Run Continuous (Daily Execution)

```bash
python src/main_trader.py --mode shadow --push-dashboard
```

### 5. Deploy to VM

```bash
# See deploy/DEPLOY.md for full instructions
scp -r . trader@your-vm:/home/trader/Quant/
ssh trader@your-vm
cd /home/trader/Quant && bash deploy/setup.sh
sudo systemctl enable quant-trader
sudo systemctl start quant-trader
```

## CLI Options

```
python src/main_trader.py [OPTIONS]

--mode {shadow,live}        Trading mode (default: shadow)
--once                      Run single cycle and exit
--interval SECONDS          Seconds between cycles (0 = daily at 3:55 PM ET)
--push-dashboard            Auto-commit dashboard to GitHub Pages
--update-dashboard-only     Refresh dashboard without trading
--no-dashboard              Skip dashboard generation
--log-level LEVEL           Logging level (default: INFO)
```

## Live Dashboard

The dashboard is a self-updating static HTML page hosted on GitHub Pages (branch: `dashboard-live`). Features:

- **Broker-style equity curve** with P&L header, dynamic green/red coloring, gradient fill, and crosshair
- **Position table** with real-time prices via client-side Yahoo Finance polling
- **Trade history** with entry/exit details
- **Range selectors** (1D, 1W, 1M, 3M, YTD, 1Y, ALL) with dense intraday data for 1D view
- **Auto-refresh** every 3 seconds during market hours

## Optimization

The system uses **walk-forward Bayesian optimization** (Optuna TPE) to find robust parameters:

- 2-year training window, 1-year test window, 6-month steps
- Optimizes Sharpe ratio across multiple walk-forward periods
- 50 trials per period with parallel execution
- Current parameters are median values across 34 walk-forward periods

Run optimization:
```bash
# Ensure optimization.enabled is set to True in config.yaml
python src/main_trader.py --once
```

## Security

- **Never commit `.env` file** — it contains API credentials
- `.env` is in `.gitignore`
- Always start with shadow mode before live trading
- All risk checks are fail-safe (reject on error)

## Disclaimer

This software is for educational and personal use. Trading carries substantial risk of loss. Past performance (backtested or simulated) does not guarantee future results. Always use paper/shadow trading to validate before risking real capital.

## License

Private project — not for distribution.
