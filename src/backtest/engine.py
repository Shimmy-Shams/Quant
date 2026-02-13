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

    # Transaction costs
    commission_pct: float = 0.001  # 0.1% per trade
    slippage_pct: float = 0.0005   # 0.05% slippage

    # Position sizing
    position_size_method: str = 'equal_weight'  # 'equal_weight', 'kelly', 'volatility_scaled'
    max_position_size: float = 0.1  # Max 10% per position
    max_total_exposure: float = 1.0  # Max 100% total exposure

    # Entry/Exit thresholds
    entry_threshold: float = 2.0  # Enter when |signal| > threshold
    exit_threshold: float = 0.5   # Exit when |signal| < threshold

    # Risk management
    stop_loss_pct: Optional[float] = None  # None = no stop loss
    take_profit_pct: Optional[float] = None  # None = no take profit
    max_holding_days: Optional[int] = None  # None = hold until signal

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
    exit_reason: str  # 'signal', 'stop_loss', 'take_profit', 'max_holding'


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
            'Avg Win': f"{self.avg_win*100:.2f}%",
            'Avg Loss': f"{self.avg_loss*100:.2f}%",
            'Avg Holding': f"{self.avg_holding_days:.1f} days",
            'Avg Exposure': f"{self.avg_exposure*100:.2f}%",
            'Max Positions': self.max_positions,
            'Total Commission': f"${self.total_commission:,.2f}"
        }


class BacktestEngine:
    """
    Vectorized backtesting engine for mean reversion strategies
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()

    def run_backtest(
        self,
        price_data: pd.DataFrame,
        signal_data: pd.DataFrame,
        volume_data: Optional[pd.DataFrame] = None,
        regime_data: Optional[pd.DataFrame] = None
    ) -> BacktestResults:
        """
        Run backtest on price/signal data

        Args:
            price_data: DataFrame with symbols as columns, dates as index
            signal_data: DataFrame with symbols as columns, dates as index (signals -1 to 1)
            volume_data: Optional volume data for position sizing
            regime_data: Optional regime multipliers (0 to 1)

        Returns:
            BacktestResults object
        """
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
        cash = pd.Series(self.config.initial_capital, index=dates)
        equity = pd.Series(self.config.initial_capital, index=dates)
        trades = []

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
                current_signal = filtered_signals[symbol]
                entry_date = pos['entry_date']
                holding_days = (date - entry_date).days

                should_exit = False
                exit_reason = None

                # Signal reversal or signal weakening
                if pos['side'] == 'long':
                    if current_signal > -self.config.exit_threshold:
                        should_exit = True
                        exit_reason = 'signal'
                elif pos['side'] == 'short':
                    if current_signal < self.config.exit_threshold:
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

                        # Determine position size
                        position_value = current_portfolio_value * self.config.max_position_size
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
                                'entry_signal': current_signal
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
        """Calculate performance metrics"""
        equity = results.equity_curve
        returns = results.returns
        trades = results.trades

        # Total return
        results.total_return = (equity.iloc[-1] / equity.iloc[0]) - 1

        # Annualized return
        years = len(equity) / 252
        results.annualized_return = (1 + results.total_return) ** (1 / years) - 1 if years > 0 else 0

        # Sharpe ratio
        if returns.std() > 0:
            results.sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)

        # Sortino ratio
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            results.sortino_ratio = (returns.mean() / downside_returns.std()) * np.sqrt(252)

        # Max drawdown
        cumulative = (1 + returns).cumprod()
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

        # Trade statistics
        results.total_trades = len(trades)

        if trades:
            winning_trades = [t for t in trades if t.pnl > 0]
            losing_trades = [t for t in trades if t.pnl < 0]

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
