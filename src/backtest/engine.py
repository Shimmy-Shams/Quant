"""
Backtesting Engine

Vectorized backtesting framework for mean reversion strategies.
Supports multiple position sizing methods, transaction costs, and detailed performance metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


@dataclass
class BacktestConfig:
    """Configuration for backtesting"""
    # Capital
    initial_capital: float = 100000.0

    # Returns calculation
    use_log_returns: bool = True  # Log returns for metrics (Sharpe, Sortino, etc.)

    # Transaction costs
    commission_pct: float = 0.001  # 0.1% per trade
    slippage_pct: float = 0.0005   # 0.05% slippage

    # Position sizing method: equal_weight, signal_proportional, volatility_scaled, kelly
    position_size_method: str = 'equal_weight'
    max_position_size: float = 0.1  # Max 10% per position
    max_total_exposure: float = 1.0  # Max 100% total exposure

    # Signal-proportional sizing params
    signal_prop_base_size: float = 0.03    # Base position size (3%)
    signal_prop_scale_factor: float = 0.02  # Additional per unit above threshold

    # Volatility-scaled sizing params
    vol_target: float = 0.15      # Annualized target vol per position (15%)
    vol_lookback: int = 60        # Days for realized vol estimation
    vol_min_size: float = 0.02    # Min position size
    vol_max_size: float = 0.10    # Max position size

    # Kelly criterion params
    kelly_fraction: float = 0.25      # Fractional Kelly (quarter-Kelly)
    kelly_lookback_trades: int = 50   # Rolling window for win rate/payoff
    kelly_min_trades: int = 20        # Min trades before Kelly activates
    kelly_min_size: float = 0.02      # Floor
    kelly_max_size: float = 0.10      # Cap

    # Entry/Exit thresholds
    entry_threshold: float = 2.0  # Enter when |signal| > threshold
    exit_threshold: float = 0.5   # Exit when |signal| < threshold

    # Risk management
    stop_loss_pct: Optional[float] = None  # None = no stop loss
    take_profit_pct: Optional[float] = None  # None = no take profit
    max_holding_days: Optional[int] = None  # None = hold until signal

    # Trailing stop (Phase B.3 - locks in profits after activation)
    use_trailing_stop: bool = False
    trailing_stop_pct: float = 0.05       # Trail at 5% from peak profit
    trailing_stop_activation: float = 0.02  # Activate after 2% profit achieved

    # Time decay exit (Phase B.3 - exit flat trades where setup failed)
    use_time_decay_exit: bool = False
    time_decay_days: int = 10             # Check after N days
    time_decay_threshold: float = 0.01    # Exit if |pnl| < this (1%)

    # Regime filter
    use_regime_filter: bool = True
    min_regime_multiplier: float = 0.5  # Don't trade if regime < this


@dataclass
class Trade:
    """Individual trade record"""
    symbol: str
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    shares: float
    side: str  # 'long' or 'short'
    pnl: float
    pnl_pct: float
    commission: float
    holding_days: int
    entry_signal: float
    exit_signal: float
    exit_reason: str  # 'signal', 'stop_loss', 'take_profit', 'max_holding', 'trailing_stop', 'time_decay'


@dataclass
class BacktestResults:
    """Backtest results container"""
    # Equity curve
    equity_curve: pd.Series
    returns: pd.Series

    # Trades
    trades: List[Trade] = field(default_factory=list)

    # Performance metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0

    # Trade statistics
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_holding_days: float = 0.0

    # Expected Value metrics
    ev_per_trade: float = 0.0        # EV = P(win)*avg_win + P(loss)*avg_loss
    ev_per_trade_long: float = 0.0   # EV for long trades only
    ev_per_trade_short: float = 0.0  # EV for short trades only

    # Exposure
    avg_exposure: float = 0.0
    max_positions: int = 0

    # Transaction costs
    total_commission: float = 0.0

    def summary(self) -> Dict:
        """Return summary dict"""
        return {
            'Total Return': f"{self.total_return*100:.2f}%",
            'Annualized Return': f"{self.annualized_return*100:.2f}%",
            'Sharpe Ratio': f"{self.sharpe_ratio:.2f}",
            'Sortino Ratio': f"{self.sortino_ratio:.2f}",
            'Calmar Ratio': f"{self.calmar_ratio:.2f}",
            'Max Drawdown': f"{self.max_drawdown*100:.2f}%",
            'Max DD Duration': f"{self.max_drawdown_duration} days",
            'Total Trades': self.total_trades,
            'Win Rate': f"{self.win_rate*100:.2f}%",
            'Profit Factor': f"{self.profit_factor:.2f}",
            'EV Per Trade': f"{self.ev_per_trade*100:.3f}%",
            'EV Long': f"{self.ev_per_trade_long*100:.3f}%",
            'EV Short': f"{self.ev_per_trade_short*100:.3f}%",
            'Avg Win': f"{self.avg_win*100:.2f}%",
            'Avg Loss': f"{self.avg_loss*100:.2f}%",
            'Avg Holding': f"{self.avg_holding_days:.1f} days",
            'Avg Exposure': f"{self.avg_exposure*100:.2f}%",
            'Max Positions': self.max_positions,
            'Total Commission': f"${self.total_commission:,.2f}"
        }


class BacktestEngine:
    """
    Vectorized backtesting engine for mean reversion strategies.
    Supports multiple position sizing methods, log returns, and EV tracking.
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self._completed_trades: List[Trade] = []  # For Kelly rolling calculations

    def _calculate_position_size(
        self,
        symbol: str,
        signal_strength: float,
        current_equity: float,
        price_data: pd.DataFrame,
        date_idx: int,
        dates: pd.DatetimeIndex
    ) -> float:
        """
        Calculate position size as a fraction of equity based on the configured method.

        Args:
            symbol: Ticker symbol
            signal_strength: Absolute signal value (|z-score|)
            current_equity: Current portfolio equity
            price_data: Full price DataFrame (for vol lookback)
            date_idx: Current index into dates
            dates: DatetimeIndex

        Returns:
            Position size as fraction of equity (0.0 to max_position_size)
        """
        method = self.config.position_size_method

        if method == 'equal_weight':
            return self.config.max_position_size

        elif method == 'signal_proportional':
            # Piecewise linear: base_size + scale * (|signal| - threshold)
            excess = signal_strength - self.config.entry_threshold
            size = self.config.signal_prop_base_size + self.config.signal_prop_scale_factor * max(0, excess)
            return np.clip(size, 0.01, self.config.max_position_size)

        elif method == 'volatility_scaled':
            # Size inversely proportional to realized volatility
            lookback = self.config.vol_lookback
            start_idx = max(0, date_idx - lookback)
            date = dates[date_idx]

            if symbol in price_data.columns:
                hist_prices = price_data[symbol].iloc[start_idx:date_idx+1]
                if len(hist_prices) >= 10:
                    # Annualized realized volatility using log returns
                    log_rets = np.log(hist_prices / hist_prices.shift(1)).dropna()
                    if len(log_rets) >= 5:
                        realized_vol = log_rets.std() * np.sqrt(252)
                        if realized_vol > 0:
                            # Size = target_vol / realized_vol (risk-normalized)
                            size = self.config.vol_target / realized_vol
                            return np.clip(size, self.config.vol_min_size, self.config.vol_max_size)

            # Fallback to equal weight
            return self.config.max_position_size

        elif method == 'kelly':
            # Fractional Kelly criterion using rolling trade history
            completed = self._completed_trades
            lookback_n = self.config.kelly_lookback_trades

            if len(completed) < self.config.kelly_min_trades:
                # Not enough history, use equal weight
                return self.config.max_position_size

            # Use last N trades
            recent = completed[-lookback_n:]
            wins = [t for t in recent if t.pnl > 0]
            losses = [t for t in recent if t.pnl <= 0]

            if not wins or not losses:
                return self.config.max_position_size

            win_rate = len(wins) / len(recent)
            avg_win = np.mean([t.pnl_pct for t in wins])
            avg_loss = abs(np.mean([t.pnl_pct for t in losses]))

            if avg_loss == 0:
                return self.config.max_position_size

            # Kelly: f* = (p * b - q) / b  where b = avg_win/avg_loss, p = win_rate, q = 1-p
            b = avg_win / avg_loss
            kelly_full = (win_rate * b - (1 - win_rate)) / b
            kelly_size = self.config.kelly_fraction * max(0, kelly_full)

            return np.clip(kelly_size, self.config.kelly_min_size, self.config.kelly_max_size)

        else:
            return self.config.max_position_size

    def run_backtest(
        self,
        price_data: pd.DataFrame,
        signal_data: pd.DataFrame,
        volume_data: Optional[pd.DataFrame] = None,
        regime_data: Optional[pd.DataFrame] = None,
        exit_signal_data: Optional[pd.DataFrame] = None
    ) -> BacktestResults:
        """
        Run backtest on price/signal data

        Args:
            price_data: DataFrame with symbols as columns, dates as index
            signal_data: DataFrame with symbols as columns, dates as index (signals -1 to 1)
            volume_data: Optional volume data for position sizing
            regime_data: Optional regime multipliers (0 to 1)
            exit_signal_data: Optional separate signal for exit decisions
                              (e.g., raw z-score in gated mode for proper exit timing)

        Returns:
            BacktestResults object
        """
        # Store exit signal data for use in exit decisions
        self._exit_signal_data = exit_signal_data

        # Ensure data is aligned
        # Only use symbols that exist in both price_data AND signal_data
        common_symbols = price_data.columns.intersection(signal_data.columns)
        dates = price_data.index

        if len(common_symbols) == 0:
            raise ValueError("No common symbols between price_data and signal_data")

        # Filter data to common symbols
        price_data = price_data[common_symbols]
        signal_data = signal_data[common_symbols]
        if volume_data is not None:
            volume_data = volume_data[common_symbols.intersection(volume_data.columns)]
        if regime_data is not None:
            regime_data = regime_data[common_symbols.intersection(regime_data.columns)]

        symbols = common_symbols

        # Initialize tracking
        positions = pd.DataFrame(0.0, index=dates, columns=symbols)  # Shares held
        cash = pd.Series(float(self.config.initial_capital), index=dates, dtype=float)
        equity = pd.Series(float(self.config.initial_capital), index=dates, dtype=float)
        trades = []
        self._completed_trades = []  # Reset Kelly trade history

        # Track open positions
        open_positions = {}  # {symbol: {entry_date, entry_price, shares, side, entry_signal}}

        for i, date in enumerate(dates):
            if i == 0:
                continue

            prev_date = dates[i-1]
            current_cash = cash.iloc[i-1]
            current_positions = positions.loc[prev_date].copy()

            # Get current prices and signals
            current_prices = price_data.loc[date]
            current_signals = signal_data.loc[date]

            # Apply regime filter if enabled
            if self.config.use_regime_filter and regime_data is not None:
                regime_mult = regime_data.loc[date]
                # Don't open new positions if regime < threshold
                filtered_signals = current_signals.copy()
                filtered_signals[regime_mult < self.config.min_regime_multiplier] = 0.0
            else:
                filtered_signals = current_signals

            # Close positions if needed
            for symbol in list(open_positions.keys()):
                if symbol not in symbols:
                    continue

                # Skip if no price or signal data for this symbol
                if symbol not in current_prices.index or symbol not in filtered_signals.index:
                    continue

                if np.isnan(current_prices[symbol]):
                    continue

                pos = open_positions[symbol]

                # Update peak price for trailing stop tracking
                if self.config.use_trailing_stop:
                    curr_px = current_prices[symbol]
                    if pos['side'] == 'long':
                        pos['peak_price'] = max(pos.get('peak_price', pos['entry_price']), curr_px)
                    else:
                        pos['peak_price'] = min(pos.get('peak_price', pos['entry_price']), curr_px)

                current_signal = filtered_signals[symbol]

                # Use exit signal (z-score) for exit decisions in gated mode
                if self._exit_signal_data is not None and symbol in self._exit_signal_data.columns:
                    if date in self._exit_signal_data.index:
                        exit_signal = self._exit_signal_data.at[date, symbol]
                        if np.isnan(exit_signal):
                            exit_signal = current_signal
                    else:
                        exit_signal = current_signal
                else:
                    exit_signal = current_signal

                entry_date = pos['entry_date']
                holding_days = (date - entry_date).days

                should_exit = False
                exit_reason = None

                # Signal reversal or signal weakening (uses exit_signal for proper exit timing)
                if pos['side'] == 'long':
                    if exit_signal > -self.config.exit_threshold:
                        should_exit = True
                        exit_reason = 'signal'
                elif pos['side'] == 'short':
                    if exit_signal < self.config.exit_threshold:
                        should_exit = True
                        exit_reason = 'signal'

                # Stop loss
                if self.config.stop_loss_pct is not None:
                    pnl_pct = (current_prices[symbol] - pos['entry_price']) / pos['entry_price']
                    if pos['side'] == 'short':
                        pnl_pct = -pnl_pct

                    if pnl_pct < -self.config.stop_loss_pct:
                        should_exit = True
                        exit_reason = 'stop_loss'

                # Take profit
                if self.config.take_profit_pct is not None:
                    pnl_pct = (current_prices[symbol] - pos['entry_price']) / pos['entry_price']
                    if pos['side'] == 'short':
                        pnl_pct = -pnl_pct

                    if pnl_pct > self.config.take_profit_pct:
                        should_exit = True
                        exit_reason = 'take_profit'

                # Max holding period
                if self.config.max_holding_days is not None:
                    if holding_days >= self.config.max_holding_days:
                        should_exit = True
                        exit_reason = 'max_holding'

                # Trailing stop: lock in profits after activation threshold
                if not should_exit and self.config.use_trailing_stop:
                    curr_px = current_prices[symbol]
                    pnl_now = (curr_px - pos['entry_price']) / pos['entry_price']
                    if pos['side'] == 'short':
                        pnl_now = -pnl_now

                    peak_px = pos.get('peak_price', pos['entry_price'])
                    peak_pnl = (peak_px - pos['entry_price']) / pos['entry_price']
                    if pos['side'] == 'short':
                        peak_pnl = -peak_pnl

                    # Only activate trailing stop once profit exceeds activation threshold
                    if peak_pnl >= self.config.trailing_stop_activation:
                        drawdown_from_peak = peak_pnl - pnl_now
                        if drawdown_from_peak >= self.config.trailing_stop_pct:
                            should_exit = True
                            exit_reason = 'trailing_stop'

                # Time decay exit: close flat trades where setup hasn't played out
                if not should_exit and self.config.use_time_decay_exit:
                    if holding_days >= self.config.time_decay_days:
                        td_pnl = (current_prices[symbol] - pos['entry_price']) / pos['entry_price']
                        if pos['side'] == 'short':
                            td_pnl = -td_pnl
                        if abs(td_pnl) < self.config.time_decay_threshold:
                            should_exit = True
                            exit_reason = 'time_decay'

                # Execute exit
                if should_exit:
                    exit_price = current_prices[symbol]
                    shares_abs = abs(pos['shares'])

                    # Calculate P&L (always use absolute shares)
                    if pos['side'] == 'long':
                        gross_pnl = (exit_price - pos['entry_price']) * shares_abs
                    else:  # short
                        gross_pnl = (pos['entry_price'] - exit_price) * shares_abs

                    # Transaction costs (entry + exit)
                    entry_commission = shares_abs * pos['entry_price'] * (self.config.commission_pct + self.config.slippage_pct)
                    exit_commission = shares_abs * exit_price * (self.config.commission_pct + self.config.slippage_pct)
                    total_commission = entry_commission + exit_commission

                    net_pnl = gross_pnl - total_commission
                    pnl_pct = net_pnl / (shares_abs * pos['entry_price'])

                    # Record trade
                    trade = Trade(
                        symbol=symbol,
                        entry_date=pos['entry_date'],
                        exit_date=date,
                        entry_price=pos['entry_price'],
                        exit_price=exit_price,
                        shares=shares_abs,
                        side=pos['side'],
                        pnl=net_pnl,
                        pnl_pct=pnl_pct,
                        commission=total_commission,
                        holding_days=holding_days,
                        entry_signal=pos['entry_signal'],
                        exit_signal=current_signal,
                        exit_reason=exit_reason
                    )
                    trades.append(trade)
                    self._completed_trades.append(trade)  # For Kelly sizing

                    # Update cash: return entry capital + net P&L
                    if pos['side'] == 'long':
                        # Sell shares: receive exit value, minus exit commission
                        current_cash += shares_abs * exit_price - exit_commission
                    else:
                        # Buy back shares: the entry cash we received, plus/minus P&L
                        # We received entry_value at entry, now pay exit_value to close
                        current_cash -= shares_abs * exit_price + exit_commission

                    # Close position
                    current_positions[symbol] = 0
                    del open_positions[symbol]

            # Open new positions
            # Use EQUITY (not cash) for position sizing to prevent leverage spiral
            current_equity = current_cash + (current_positions * current_prices).fillna(0).sum()
            current_portfolio_value = max(current_equity, 0)  # Don't size off negative equity
            for symbol in symbols:
                # Skip if symbol not in position tracking or no current position
                if symbol not in current_positions.index:
                    continue

                if current_positions[symbol] == 0:  # No position
                    # Skip if no signal data for this symbol
                    if symbol not in filtered_signals.index or symbol not in current_prices.index:
                        continue

                    current_signal = filtered_signals[symbol]

                    if np.isnan(current_signal) or np.isnan(current_prices[symbol]):
                        continue

                    # Check entry threshold
                    if abs(current_signal) > self.config.entry_threshold:
                        # Check total exposure limit before opening new position
                        total_exposure = (current_positions.abs() * current_prices).fillna(0).sum()
                        max_exposure = current_portfolio_value * self.config.max_total_exposure
                        if total_exposure >= max_exposure:
                            continue

                        # Determine position size using configured method
                        size_fraction = self._calculate_position_size(
                            symbol=symbol,
                            signal_strength=abs(current_signal),
                            current_equity=current_equity,
                            price_data=price_data,
                            date_idx=i,
                            dates=dates
                        )
                        position_value = current_portfolio_value * size_fraction
                        price = current_prices[symbol]
                        shares = position_value / price

                        # Determine side
                        if current_signal < 0:
                            side = 'long'
                        else:
                            side = 'short'

                        # Entry commission
                        entry_commission = shares * price * (self.config.commission_pct + self.config.slippage_pct)

                        # Cash check: longs need cash, shorts need margin (use same check)
                        if position_value + entry_commission <= current_portfolio_value:
                            if side == 'long':
                                # Buy shares: cash goes out
                                current_cash -= shares * price + entry_commission
                                current_positions[symbol] = shares
                            else:
                                # Short sell: receive cash from selling borrowed shares
                                current_cash += shares * price - entry_commission
                                current_positions[symbol] = -shares

                            # Track position
                            open_positions[symbol] = {
                                'entry_date': date,
                                'entry_price': price,
                                'shares': shares if side == 'long' else -shares,
                                'side': side,
                                'entry_signal': current_signal,
                                'peak_price': price  # For trailing stop tracking
                            }

            # Update positions and cash
            positions.loc[date] = current_positions
            cash.iloc[i] = current_cash

            # Calculate equity
            position_values = (current_positions * current_prices).sum()
            equity.iloc[i] = current_cash + position_values

        # Calculate returns
        returns = equity.pct_change().fillna(0)

        # Calculate performance metrics
        results = BacktestResults(
            equity_curve=equity,
            returns=returns,
            trades=trades
        )

        self._calculate_metrics(results)

        return results

    def _calculate_metrics(self, results: BacktestResults):
        """Calculate performance metrics using log returns when configured"""
        equity = results.equity_curve
        trades = results.trades

        # --- Returns computation ---
        # Log returns: r = ln(E_t / E_{t-1}), symmetric and time-additive
        # Simple returns: r = (E_t - E_{t-1}) / E_{t-1}
        if self.config.use_log_returns:
            log_returns = np.log(equity / equity.shift(1)).fillna(0)
            # Replace infinities from zero/negative equity
            log_returns = log_returns.replace([np.inf, -np.inf], 0)
            results.returns = log_returns
        else:
            results.returns = equity.pct_change().fillna(0)

        returns = results.returns

        # Total return (always in simple return space for interpretability)
        results.total_return = (equity.iloc[-1] / equity.iloc[0]) - 1

        # Annualized return
        years = len(equity) / 252
        if self.config.use_log_returns and years > 0:
            # log returns are additive: annualized = mean_daily * 252, then convert
            mean_daily_log_return = returns.mean()
            results.annualized_return = np.exp(mean_daily_log_return * 252) - 1
        elif years > 0:
            results.annualized_return = (1 + results.total_return) ** (1 / years) - 1
        else:
            results.annualized_return = 0

        # Sharpe ratio (using whatever return type is configured)
        if returns.std() > 0:
            results.sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)

        # Sortino ratio (downside deviation only)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            results.sortino_ratio = (returns.mean() / downside_returns.std()) * np.sqrt(252)

        # Max drawdown (computed on equity, not returns)
        cumulative = (1 + equity.pct_change().fillna(0)).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        results.max_drawdown = abs(drawdown.min())

        # Max drawdown duration
        in_drawdown = drawdown < 0
        drawdown_periods = []
        current_dd_length = 0
        for is_dd in in_drawdown:
            if is_dd:
                current_dd_length += 1
            else:
                if current_dd_length > 0:
                    drawdown_periods.append(current_dd_length)
                current_dd_length = 0
        results.max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0

        # Calmar ratio
        if results.max_drawdown > 0:
            results.calmar_ratio = results.annualized_return / results.max_drawdown

        # --- Trade statistics ---
        results.total_trades = len(trades)

        if trades:
            winning_trades = [t for t in trades if t.pnl > 0]
            losing_trades = [t for t in trades if t.pnl <= 0]

            results.win_rate = len(winning_trades) / len(trades)

            if winning_trades:
                results.avg_win = np.mean([t.pnl_pct for t in winning_trades])
            if losing_trades:
                results.avg_loss = np.mean([t.pnl_pct for t in losing_trades])

            # Profit factor
            total_wins = sum(t.pnl for t in winning_trades)
            total_losses = abs(sum(t.pnl for t in losing_trades))
            if total_losses > 0:
                results.profit_factor = total_wins / total_losses

            # --- Expected Value (EV) per trade ---
            # EV = P(win) * avg_win + P(loss) * avg_loss
            results.ev_per_trade = (
                results.win_rate * results.avg_win +
                (1 - results.win_rate) * results.avg_loss  # avg_loss is negative
            )

            # EV by side
            long_trades = [t for t in trades if t.side == 'long']
            short_trades = [t for t in trades if t.side == 'short']

            if long_trades:
                long_wins = [t for t in long_trades if t.pnl > 0]
                long_wr = len(long_wins) / len(long_trades)
                long_avg_win = np.mean([t.pnl_pct for t in long_wins]) if long_wins else 0
                long_losses = [t for t in long_trades if t.pnl <= 0]
                long_avg_loss = np.mean([t.pnl_pct for t in long_losses]) if long_losses else 0
                results.ev_per_trade_long = long_wr * long_avg_win + (1 - long_wr) * long_avg_loss

            if short_trades:
                short_wins = [t for t in short_trades if t.pnl > 0]
                short_wr = len(short_wins) / len(short_trades)
                short_avg_win = np.mean([t.pnl_pct for t in short_wins]) if short_wins else 0
                short_losses = [t for t in short_trades if t.pnl <= 0]
                short_avg_loss = np.mean([t.pnl_pct for t in short_losses]) if short_losses else 0
                results.ev_per_trade_short = short_wr * short_avg_win + (1 - short_wr) * short_avg_loss

            # Avg holding days
            results.avg_holding_days = np.mean([t.holding_days for t in trades])

            # Total commission
            results.total_commission = sum(t.commission for t in trades)

        # Exposure tracking (simplified - based on equity curve vs initial capital)
        results.avg_exposure = 0.5  # Placeholder - would need position tracking
        results.max_positions = 10  # Placeholder


def calculate_rolling_sharpe(returns: pd.Series, window: int = 60) -> pd.Series:
    """
    Calculate rolling Sharpe ratio

    Args:
        returns: Daily returns
        window: Rolling window in days

    Returns:
        Rolling Sharpe ratio
    """
    rolling_mean = returns.rolling(window=window).mean()
    rolling_std = returns.rolling(window=window).std()

    rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(252)

    return rolling_sharpe


def calculate_underwater_curve(equity_curve: pd.Series) -> pd.Series:
    """
    Calculate underwater (drawdown) curve

    Args:
        equity_curve: Equity curve

    Returns:
        Drawdown series (negative values)
    """
    running_max = equity_curve.expanding().max()
    drawdown = (equity_curve - running_max) / running_max

    return drawdown
