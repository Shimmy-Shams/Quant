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

    Phase 3A: Core loop uses pre-computed NumPy arrays with integer indexing
    for 10-50x speedup over DataFrame .loc[] access. Sequential dependencies
    (cash, position limits, Kelly sizing) remain in a minimal Python loop.
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self._completed_trades: List[Trade] = []  # For Kelly rolling calculations

    def _precompute_vol_sizes(self, price_arr: np.ndarray) -> np.ndarray:
        """
        Pre-compute volatility-scaled position sizes for all symbols/dates.

        Uses pandas rolling for vectorized computation across all symbols at once.

        Args:
            price_arr: (n_days, n_symbols) price array

        Returns:
            (n_days, n_symbols) position size fraction array
        """
        cfg = self.config

        # Log returns via numpy (fast)
        with np.errstate(divide='ignore', invalid='ignore'):
            log_rets = np.log(price_arr[1:] / price_arr[:-1])
        log_rets = np.where(np.isfinite(log_rets), log_rets, np.nan)

        # Pad first row with NaN
        log_rets = np.vstack([np.full(price_arr.shape[1], np.nan), log_rets])

        # Use pandas rolling for vectorized rolling std (all columns at once)
        rets_df = pd.DataFrame(log_rets)
        rolling_std = rets_df.rolling(window=cfg.vol_lookback, min_periods=10).std()
        realized_vol = rolling_std.values * np.sqrt(252)

        # Compute position sizes
        with np.errstate(divide='ignore', invalid='ignore'):
            vol_sizes = cfg.vol_target / realized_vol
        vol_sizes = np.where(np.isfinite(vol_sizes), vol_sizes, cfg.max_position_size)
        vol_sizes = np.clip(vol_sizes, cfg.vol_min_size, cfg.vol_max_size)

        return vol_sizes

    def _calculate_position_size_fast(
        self,
        signal_strength: float,
        vol_size: float
    ) -> float:
        """
        Fast position size calculation using pre-computed data.

        Args:
            signal_strength: Absolute signal value
            vol_size: Pre-computed volatility-scaled size for this symbol/date

        Returns:
            Position size as fraction of equity
        """
        method = self.config.position_size_method

        if method == 'equal_weight':
            return self.config.max_position_size

        elif method == 'signal_proportional':
            excess = signal_strength - self.config.entry_threshold
            size = self.config.signal_prop_base_size + self.config.signal_prop_scale_factor * max(0, excess)
            return np.clip(size, 0.01, self.config.max_position_size)

        elif method == 'volatility_scaled':
            return vol_size  # Already pre-computed

        elif method == 'kelly':
            # Sequential dependency - uses trade history
            completed = self._completed_trades
            lookback_n = self.config.kelly_lookback_trades

            if len(completed) < self.config.kelly_min_trades:
                return self.config.max_position_size

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
        Run backtest on price/signal data.

        Phase 3A vectorized: Pre-computes NumPy arrays and uses integer indexing
        for 10-50x speedup over the original DataFrame .loc[] approach.

        Args:
            price_data: DataFrame with symbols as columns, dates as index
            signal_data: DataFrame with symbols as columns, dates as index
            volume_data: Optional volume data
            regime_data: Optional regime multipliers (0 to 1)
            exit_signal_data: Optional separate signal for exit decisions
                              (e.g., raw z-score in gated mode)

        Returns:
            BacktestResults object
        """
        cfg = self.config

        # --- Data alignment ---
        common_symbols = price_data.columns.intersection(signal_data.columns)
        if len(common_symbols) == 0:
            raise ValueError("No common symbols between price_data and signal_data")

        price_data = price_data[common_symbols]
        signal_data = signal_data[common_symbols]

        dates = price_data.index
        n_days = len(dates)
        n_syms = len(common_symbols)

        # --- Symbol mapping ---
        sym_list = list(common_symbols)
        sym_to_idx = {s: j for j, s in enumerate(sym_list)}

        # --- Convert to NumPy arrays (THE key optimization) ---
        price_arr = price_data.values.astype(np.float64)   # (n_days, n_syms)
        signal_arr = signal_data.values.astype(np.float64)  # (n_days, n_syms)

        # Exit signal array (z-score for gated mode exits)
        if exit_signal_data is not None:
            # Align to common_symbols
            exit_cols = [s for s in sym_list if s in exit_signal_data.columns]
            if exit_cols:
                exit_df = exit_signal_data.reindex(index=dates, columns=sym_list)
                exit_arr = exit_df.values.astype(np.float64)
            else:
                exit_arr = None
        else:
            exit_arr = None

        # Regime filter array
        if cfg.use_regime_filter and regime_data is not None:
            regime_cols = [s for s in sym_list if s in regime_data.columns]
            if regime_cols:
                regime_df = regime_data.reindex(index=dates, columns=sym_list)
                regime_arr = regime_df.values.astype(np.float64)
            else:
                regime_arr = None
        else:
            regime_arr = None

        # --- Pre-compute position sizes ---
        if cfg.position_size_method == 'volatility_scaled':
            vol_sizes = self._precompute_vol_sizes(price_arr)
        else:
            vol_sizes = np.full((n_days, n_syms), cfg.max_position_size)

        # --- Initialize tracking arrays ---
        positions_arr = np.zeros((n_days, n_syms), dtype=np.float64)  # Shares held
        cash_arr = np.zeros(n_days, dtype=np.float64)
        equity_arr = np.zeros(n_days, dtype=np.float64)
        cash_arr[0] = float(cfg.initial_capital)
        equity_arr[0] = float(cfg.initial_capital)

        # Exposure tracking (Phase 3A: real tracking instead of placeholders)
        daily_n_positions = np.zeros(n_days, dtype=np.int32)
        daily_exposure = np.zeros(n_days, dtype=np.float64)

        trades = []
        self._completed_trades = []

        # Track open positions: {sym_idx: dict}
        open_positions = {}
        txn_cost_rate = cfg.commission_pct + cfg.slippage_pct

        # --- Main loop (minimal Python, NumPy arrays only) ---
        for i in range(1, n_days):
            prices_today = price_arr[i]         # (n_syms,) - fast row slice
            signals_today = signal_arr[i]       # (n_syms,)
            current_cash = cash_arr[i - 1]
            prev_positions = positions_arr[i - 1].copy()  # (n_syms,)

            # Apply regime filter
            if regime_arr is not None:
                regime_today = regime_arr[i]
                filt_signals = signals_today.copy()
                mask = regime_today < cfg.min_regime_multiplier
                filt_signals[mask] = 0.0
            else:
                filt_signals = signals_today

            # --- EXIT LOOP ---
            for sym_idx in list(open_positions.keys()):
                px = prices_today[sym_idx]
                if np.isnan(px):
                    continue

                pos = open_positions[sym_idx]
                entry_px = pos['entry_price']
                side = pos['side']
                entry_date_idx = pos['entry_date_idx']
                try:
                    holding_days = (dates[i] - dates[entry_date_idx]).days
                except (AttributeError, TypeError):
                    holding_days = i - entry_date_idx

                # Direction-aware P&L
                if side == 'long':
                    pnl_pct_now = (px - entry_px) / entry_px
                else:
                    pnl_pct_now = (entry_px - px) / entry_px

                # Update peak price for trailing stop
                if cfg.use_trailing_stop:
                    if side == 'long':
                        pos['peak_price'] = max(pos['peak_price'], px)
                    else:
                        pos['peak_price'] = min(pos['peak_price'], px)

                # Get exit signal (z-score in gated mode, else composite)
                if exit_arr is not None:
                    es = exit_arr[i, sym_idx]
                    if np.isnan(es):
                        es = filt_signals[sym_idx]
                else:
                    es = filt_signals[sym_idx]

                # --- Check 6 exit conditions ---
                should_exit = False
                exit_reason = None

                # 1. Signal exit
                if side == 'long' and es > -cfg.exit_threshold:
                    should_exit = True
                    exit_reason = 'signal'
                elif side == 'short' and es < cfg.exit_threshold:
                    should_exit = True
                    exit_reason = 'signal'

                # 2. Stop loss
                if cfg.stop_loss_pct is not None and pnl_pct_now < -cfg.stop_loss_pct:
                    should_exit = True
                    exit_reason = 'stop_loss'

                # 3. Take profit
                if cfg.take_profit_pct is not None and pnl_pct_now > cfg.take_profit_pct:
                    should_exit = True
                    exit_reason = 'take_profit'

                # 4. Max holding
                if cfg.max_holding_days is not None and holding_days >= cfg.max_holding_days:
                    should_exit = True
                    exit_reason = 'max_holding'

                # 5. Trailing stop
                if not should_exit and cfg.use_trailing_stop:
                    peak_px = pos['peak_price']
                    if side == 'long':
                        peak_pnl = (peak_px - entry_px) / entry_px
                    else:
                        peak_pnl = (entry_px - peak_px) / entry_px
                    if peak_pnl >= cfg.trailing_stop_activation:
                        dd_from_peak = peak_pnl - pnl_pct_now
                        if dd_from_peak >= cfg.trailing_stop_pct:
                            should_exit = True
                            exit_reason = 'trailing_stop'

                # 6. Time decay
                if not should_exit and cfg.use_time_decay_exit:
                    if holding_days >= cfg.time_decay_days:
                        if abs(pnl_pct_now) < cfg.time_decay_threshold:
                            should_exit = True
                            exit_reason = 'time_decay'

                # --- Execute exit ---
                if should_exit:
                    shares_abs = abs(pos['shares'])
                    if side == 'long':
                        gross_pnl = (px - entry_px) * shares_abs
                    else:
                        gross_pnl = (entry_px - px) * shares_abs

                    entry_comm = shares_abs * entry_px * txn_cost_rate
                    exit_comm = shares_abs * px * txn_cost_rate
                    total_comm = entry_comm + exit_comm
                    net_pnl = gross_pnl - total_comm
                    pnl_pct_final = net_pnl / (shares_abs * entry_px)

                    trade = Trade(
                        symbol=sym_list[sym_idx],
                        entry_date=dates[entry_date_idx],
                        exit_date=dates[i],
                        entry_price=entry_px,
                        exit_price=px,
                        shares=shares_abs,
                        side=side,
                        pnl=net_pnl,
                        pnl_pct=pnl_pct_final,
                        commission=total_comm,
                        holding_days=holding_days,
                        entry_signal=pos['entry_signal'],
                        exit_signal=float(filt_signals[sym_idx]),
                        exit_reason=exit_reason
                    )
                    trades.append(trade)
                    self._completed_trades.append(trade)

                    if side == 'long':
                        current_cash += shares_abs * px - exit_comm
                    else:
                        current_cash -= shares_abs * px + exit_comm

                    prev_positions[sym_idx] = 0.0
                    del open_positions[sym_idx]

            # --- ENTRY LOOP ---
            # Compute equity for position sizing (prevents leverage spiral)
            position_values = np.nansum(prev_positions * prices_today)
            current_equity = current_cash + position_values
            portfolio_value = max(current_equity, 0.0)

            # Total current exposure
            total_exposure = np.nansum(np.abs(prev_positions) * prices_today)
            max_exposure = portfolio_value * cfg.max_total_exposure

            for sym_idx in range(n_syms):
                if prev_positions[sym_idx] != 0.0:
                    continue  # Already have a position

                sig = filt_signals[sym_idx]
                px = prices_today[sym_idx]

                if np.isnan(sig) or np.isnan(px) or px <= 0:
                    continue

                if abs(sig) <= cfg.entry_threshold:
                    continue

                # Exposure check
                if total_exposure >= max_exposure:
                    break  # No more room for any position

                # Position sizing
                size_frac = self._calculate_position_size_fast(
                    signal_strength=abs(sig),
                    vol_size=vol_sizes[i, sym_idx]
                )
                pos_value = portfolio_value * size_frac
                shares = pos_value / px
                side = 'long' if sig < 0 else 'short'

                # Entry commission
                entry_comm = shares * px * txn_cost_rate

                # Cash/margin check
                if pos_value + entry_comm > portfolio_value:
                    continue

                if side == 'long':
                    current_cash -= shares * px + entry_comm
                    prev_positions[sym_idx] = shares
                else:
                    current_cash += shares * px - entry_comm
                    prev_positions[sym_idx] = -shares

                open_positions[sym_idx] = {
                    'entry_date_idx': i,
                    'entry_price': px,
                    'shares': prev_positions[sym_idx],
                    'side': side,
                    'entry_signal': float(sig),
                    'peak_price': px
                }

                # Update exposure
                total_exposure += shares * px

            # --- Store day results ---
            positions_arr[i] = prev_positions
            cash_arr[i] = current_cash
            pos_vals = np.nansum(prev_positions * prices_today)
            equity_arr[i] = current_cash + pos_vals

            # Exposure tracking
            n_open = int(np.count_nonzero(prev_positions))
            daily_n_positions[i] = n_open
            if equity_arr[i] > 0:
                daily_exposure[i] = np.nansum(np.abs(prev_positions) * prices_today) / equity_arr[i]

        # --- Build results ---
        equity_series = pd.Series(equity_arr, index=dates)
        returns_series = equity_series.pct_change().fillna(0)

        results = BacktestResults(
            equity_curve=equity_series,
            returns=returns_series,
            trades=trades
        )

        # Real exposure/position tracking (Phase 3A: replaced placeholders)
        results.avg_exposure = float(np.mean(daily_exposure[1:]))  # skip day 0
        results.max_positions = int(np.max(daily_n_positions))

        self._calculate_metrics(results)

        return results

    def _calculate_metrics(self, results: BacktestResults):
        """Calculate performance metrics using log returns when configured"""
        equity = results.equity_curve
        trades = results.trades
        cfg = self.config

        # --- Returns computation ---
        if cfg.use_log_returns:
            with np.errstate(divide='ignore', invalid='ignore'):
                log_returns = np.log(equity / equity.shift(1)).fillna(0)
            log_returns = log_returns.replace([np.inf, -np.inf], 0)
            results.returns = log_returns
        else:
            results.returns = equity.pct_change().fillna(0)

        returns = results.returns

        # Total return (simple return space)
        results.total_return = (equity.iloc[-1] / equity.iloc[0]) - 1

        # Annualized return
        years = len(equity) / 252
        if cfg.use_log_returns and years > 0:
            mean_daily_log_return = returns.mean()
            results.annualized_return = np.exp(mean_daily_log_return * 252) - 1
        elif years > 0:
            results.annualized_return = (1 + results.total_return) ** (1 / years) - 1
        else:
            results.annualized_return = 0

        # Sharpe ratio
        ret_std = returns.std()
        if ret_std > 0:
            results.sharpe_ratio = (returns.mean() / ret_std) * np.sqrt(252)

        # Sortino ratio
        downside = returns[returns < 0]
        if len(downside) > 0:
            ds_std = downside.std()
            if ds_std > 0:
                results.sortino_ratio = (returns.mean() / ds_std) * np.sqrt(252)

        # Max drawdown (equity-based)
        equity_vals = equity.values
        running_max = np.maximum.accumulate(equity_vals)
        with np.errstate(divide='ignore', invalid='ignore'):
            drawdown = (equity_vals - running_max) / running_max
        drawdown = np.where(np.isfinite(drawdown), drawdown, 0.0)
        results.max_drawdown = abs(float(np.min(drawdown)))

        # Max drawdown duration (vectorized)
        in_dd = drawdown < 0
        if np.any(in_dd):
            # Find transitions: entering/leaving drawdown
            changes = np.diff(in_dd.astype(np.int8))
            # Starts of drawdown (0->1): changes == 1
            # Ends of drawdown (1->0): changes == -1
            starts = np.where(changes == 1)[0] + 1
            ends = np.where(changes == -1)[0] + 1

            # Handle edge cases
            if in_dd[0]:
                starts = np.insert(starts, 0, 0)
            if in_dd[-1]:
                ends = np.append(ends, len(in_dd))

            if len(starts) > 0 and len(ends) > 0:
                # Pair up starts and ends
                n_periods = min(len(starts), len(ends))
                durations = ends[:n_periods] - starts[:n_periods]
                results.max_drawdown_duration = int(np.max(durations)) if len(durations) > 0 else 0
            else:
                results.max_drawdown_duration = 0
        else:
            results.max_drawdown_duration = 0

        # Calmar ratio
        if results.max_drawdown > 0:
            results.calmar_ratio = results.annualized_return / results.max_drawdown

        # --- Trade statistics (vectorized where possible) ---
        results.total_trades = len(trades)

        if trades:
            # Convert to arrays for vectorized ops
            pnl_arr = np.array([t.pnl for t in trades])
            pnl_pct_arr = np.array([t.pnl_pct for t in trades])
            sides_arr = np.array([t.side for t in trades])
            hold_arr = np.array([t.holding_days for t in trades])
            comm_arr = np.array([t.commission for t in trades])

            win_mask = pnl_arr > 0
            lose_mask = ~win_mask
            n_wins = int(win_mask.sum())
            n_losses = int(lose_mask.sum())

            results.win_rate = n_wins / len(trades)

            if n_wins > 0:
                results.avg_win = float(pnl_pct_arr[win_mask].mean())
            if n_losses > 0:
                results.avg_loss = float(pnl_pct_arr[lose_mask].mean())

            # Profit factor
            total_wins = float(pnl_arr[win_mask].sum()) if n_wins > 0 else 0.0
            total_losses = abs(float(pnl_arr[lose_mask].sum())) if n_losses > 0 else 0.0
            if total_losses > 0:
                results.profit_factor = total_wins / total_losses

            # EV per trade
            results.ev_per_trade = (
                results.win_rate * results.avg_win +
                (1 - results.win_rate) * results.avg_loss
            )

            # EV by side (vectorized)
            for side_val, attr_name in [('long', 'ev_per_trade_long'), ('short', 'ev_per_trade_short')]:
                side_mask = sides_arr == side_val
                if side_mask.sum() > 0:
                    side_pnl = pnl_arr[side_mask]
                    side_pnl_pct = pnl_pct_arr[side_mask]
                    side_wins = side_pnl > 0
                    side_wr = side_wins.sum() / len(side_pnl)
                    side_avg_win = float(side_pnl_pct[side_wins].mean()) if side_wins.any() else 0.0
                    side_avg_loss = float(side_pnl_pct[~side_wins].mean()) if (~side_wins).any() else 0.0
                    setattr(results, attr_name, side_wr * side_avg_win + (1 - side_wr) * side_avg_loss)

            results.avg_holding_days = float(hold_arr.mean())
            results.total_commission = float(comm_arr.sum())


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
