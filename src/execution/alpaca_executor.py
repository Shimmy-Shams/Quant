"""
Alpaca Order Executor

Translates signal-based trade decisions into Alpaca API orders.
Handles position sizing, order submission, and fill tracking.
Respects trading mode (live/shadow/replay).
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

from connection.alpaca_connection import AlpacaConnection, TradingMode

logger = logging.getLogger(__name__)


@dataclass
class TradeDecision:
    """A single trade decision from the signal pipeline"""
    symbol: str
    action: str              # 'buy', 'sell', 'short', 'cover'
    target_qty: int          # Number of shares
    signal_strength: float   # Raw composite signal value
    signal_direction: int    # +1 (long) or -1 (short)
    reason: str              # Human-readable reason
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def side(self) -> str:
        """Alpaca order side"""
        if self.action in ('buy', 'cover'):
            return 'buy'
        return 'sell'


@dataclass
class TradeResult:
    """Result of executing a trade decision"""
    decision: TradeDecision
    order_id: str
    status: str              # 'filled', 'submitted', 'simulated', 'rejected', 'error'
    filled_price: Optional[float] = None
    filled_qty: Optional[int] = None
    commission: float = 0.0
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


class AlpacaExecutor:
    """
    Executes trading decisions via Alpaca API.
    
    Flow:
    1. Receive list of TradeDecisions from signal pipeline
    2. Validate against risk limits
    3. Submit orders (or simulate in shadow/replay mode)
    4. Track and return results
    """

    def __init__(
        self,
        connection: AlpacaConnection,
        commission_pct: float = 0.0,  # Alpaca is commission-free
        max_position_pct: float = 0.10,
        max_total_exposure: float = 1.0,
        stop_loss_pct: Optional[float] = None,
        take_profit_pct: Optional[float] = None,
    ):
        """
        Args:
            connection: AlpacaConnection instance
            commission_pct: Commission rate (0 for Alpaca)
            max_position_pct: Max % of portfolio per position
            max_total_exposure: Max % of portfolio in positions
            stop_loss_pct: If set, attach server-side stop-loss via bracket order
            take_profit_pct: If set, attach server-side take-profit via bracket order
        """
        self.connection = connection
        self.commission_pct = commission_pct
        self.max_position_pct = max_position_pct
        self.max_total_exposure = max_total_exposure
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

    def execute_decisions(
        self,
        decisions: List[TradeDecision],
        current_prices: pd.Series,
    ) -> List[TradeResult]:
        """
        Execute a batch of trade decisions.

        Processes exits first (free up capital), then entries.

        Args:
            decisions: List of TradeDecision objects
            current_prices: Series of current prices indexed by symbol

        Returns:
            List of TradeResult objects
        """
        # Sort: exits first, then entries
        exits = [d for d in decisions if d.action in ('sell', 'cover')]
        entries = [d for d in decisions if d.action in ('buy', 'short')]

        results = []

        # Execute exits
        for decision in exits:
            result = self._execute_single(decision, current_prices)
            results.append(result)

        # Execute entries (after exits free capital)
        for decision in entries:
            # Pre-trade risk check
            if not self._risk_check(decision, current_prices):
                results.append(TradeResult(
                    decision=decision,
                    order_id='',
                    status='rejected',
                    error_message='Failed risk check (exposure limit)',
                ))
                continue
            result = self._execute_single(decision, current_prices)
            results.append(result)

        # Log summary
        filled = sum(1 for r in results if r.status in ('filled', 'submitted', 'simulated'))
        rejected = sum(1 for r in results if r.status == 'rejected')
        errors = sum(1 for r in results if r.status == 'error')
        logger.info(
            f"Executed {len(results)} decisions: "
            f"{filled} filled, {rejected} rejected, {errors} errors"
        )

        return results

    def _execute_single(
        self,
        decision: TradeDecision,
        current_prices: pd.Series,
    ) -> TradeResult:
        """Execute a single trade decision, optionally with bracket (SL/TP) legs."""
        try:
            if decision.target_qty <= 0:
                return TradeResult(
                    decision=decision,
                    order_id='',
                    status='rejected',
                    error_message='Zero or negative quantity',
                )

            price = current_prices.get(decision.symbol, 0)

            # ── Use bracket order for entries when SL is configured ──
            is_entry = decision.action in ('buy', 'short')
            use_bracket = is_entry and self.stop_loss_pct is not None and price > 0

            if use_bracket:
                from execution.intraday_monitor import IntradayMonitor  # lazy to avoid circular import
                direction = 'long' if decision.action == 'buy' else 'short'
                bracket = IntradayMonitor.compute_bracket_prices(
                    entry_price=price,
                    side=direction,
                    stop_loss_pct=self.stop_loss_pct,
                    take_profit_pct=self.take_profit_pct,
                )
                sl_price = bracket['stop_loss_price']
                tp_price = bracket['take_profit_price']
                order = self.connection.submit_bracket_order(
                    symbol=decision.symbol,
                    qty=decision.target_qty,
                    side=decision.side,
                    stop_loss_price=sl_price,
                    take_profit_price=tp_price,
                )
                tp_str = f"TP=${tp_price:.2f}" if tp_price else "no TP"
                logger.info(f"Bracket order {decision.symbol}: SL=${sl_price:.2f} {tp_str}")
            else:
                order = self.connection.submit_market_order(
                    symbol=decision.symbol,
                    qty=decision.target_qty,
                    side=decision.side,
                )

            # Estimate commission
            commission = price * decision.target_qty * self.commission_pct

            return TradeResult(
                decision=decision,
                order_id=order['id'],
                status=order['status'],
                filled_price=order.get('filled_avg_price') or price,
                filled_qty=decision.target_qty,
                commission=commission,
            )

        except Exception as e:
            logger.error(f"Order error for {decision.symbol}: {e}")
            return TradeResult(
                decision=decision,
                order_id='',
                status='error',
                error_message=str(e),
            )

    def _risk_check(
        self,
        decision: TradeDecision,
        current_prices: pd.Series,
    ) -> bool:
        """
        Pre-trade risk validation.

        Checks:
        1. Single position size vs max_position_pct
        2. Total exposure vs max_total_exposure
        """
        try:
            if self.connection.config.trading_mode == TradingMode.REPLAY:
                return True  # Skip for replay (positions tracked internally)

            account = self.connection.get_account()
            equity = account['portfolio_value']

            if equity <= 0:
                return False

            # Check single position size
            price = current_prices.get(decision.symbol, 0)
            if price <= 0:
                return False

            position_value = price * decision.target_qty
            position_pct = position_value / equity

            if position_pct > self.max_position_pct:
                logger.warning(
                    f"Risk check failed for {decision.symbol}: "
                    f"position {position_pct:.1%} > limit {self.max_position_pct:.1%}"
                )
                return False

            # Check total exposure
            total_exposure = (
                abs(account['long_market_value']) +
                abs(account['short_market_value'])
            ) / equity

            if total_exposure + position_pct > self.max_total_exposure:
                logger.warning(
                    f"Risk check failed for {decision.symbol}: "
                    f"total exposure would be {total_exposure + position_pct:.1%}"
                )
                return False

            return True

        except Exception as e:
            logger.error(f"Risk check error: {e}")
            return False  # Fail-safe: reject on error

    def generate_decisions_from_signals(
        self,
        signal_df: pd.DataFrame,
        price_df: pd.DataFrame,
        volume_df: pd.DataFrame,
        exit_signal_df: Optional[pd.DataFrame],
        date: pd.Timestamp,
        current_positions: Dict[str, Dict],
        config,
        current_equity: Optional[float] = None,
    ) -> List[TradeDecision]:
        """
        Convert today's signals into trade decisions.

        Uses the same entry/exit logic as the backtest engine but for a single day.

        Args:
            signal_df: Full signal DataFrame
            price_df: Full price DataFrame
            volume_df: Full volume DataFrame
            exit_signal_df: Z-score DataFrame for exit signals (gated mode)
            date: Today's date
            current_positions: Dict of current positions {symbol: {qty, side, entry_price, entry_date}}
            config: BacktestConfig instance

        Returns:
            List of TradeDecision objects
        """
        decisions = []

        if date not in signal_df.index:
            logger.warning(f"No signals for {date.date()}")
            return decisions

        today_signals = signal_df.loc[date]
        today_prices = price_df.loc[date] if date in price_df.index else pd.Series(dtype=float)

        # Exit signals (z-score based in gated mode)
        today_exit = (
            exit_signal_df.loc[date]
            if exit_signal_df is not None and date in exit_signal_df.index
            else None
        )

        # ─── CHECK EXITS (matches backtest engine logic) ──────────────

        for symbol, pos in current_positions.items():
            if symbol not in today_prices.index:
                continue

            current_price = today_prices[symbol]
            entry_price = pos['entry_price']
            side = pos['side']  # 'long' or 'short'
            days_held = (date - pos['entry_date']).days

            # Direction-aware P&L
            if side == 'long':
                pnl_pct = (current_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - current_price) / entry_price

            should_exit = False
            exit_reason = ""

            # 1. Direction-aware signal exit (matches backtest engine)
            #    Long: exit when z-score reverts above -exit_threshold
            #    Short: exit when z-score reverts below +exit_threshold
            if today_exit is not None and symbol in today_exit.index:
                exit_val = today_exit[symbol]
                if not pd.isna(exit_val):
                    if side == 'long' and exit_val > -config.exit_threshold:
                        should_exit = True
                        exit_reason = f"Signal exit (z={exit_val:.2f} > -{config.exit_threshold})"
                    elif side == 'short' and exit_val < config.exit_threshold:
                        should_exit = True
                        exit_reason = f"Signal exit (z={exit_val:.2f} < {config.exit_threshold})"

            # 2. Stop loss
            if not should_exit and config.stop_loss_pct is not None:
                sl_pct = config.short_stop_loss_pct if (
                    side == 'short' and config.short_stop_loss_pct
                ) else config.stop_loss_pct

                if pnl_pct < -sl_pct:
                    should_exit = True
                    exit_reason = f"Stop loss ({pnl_pct:.2%} < -{sl_pct:.0%})"

            # 3. Take profit
            if not should_exit and config.take_profit_pct is not None:
                if pnl_pct > config.take_profit_pct:
                    should_exit = True
                    exit_reason = f"Take profit ({pnl_pct:.2%} > {config.take_profit_pct:.0%})"

            # 4. Max holding days
            if not should_exit and config.max_holding_days and days_held >= config.max_holding_days:
                should_exit = True
                exit_reason = f"Max holding ({days_held} days)"

            # 5. Trailing stop (matches backtest engine)
            if not should_exit and getattr(config, 'use_trailing_stop', False):
                peak_price = pos.get('peak_price', entry_price)
                if side == 'long':
                    peak_pnl = (peak_price - entry_price) / entry_price
                else:
                    peak_pnl = (entry_price - peak_price) / entry_price
                if peak_pnl >= config.trailing_stop_activation:
                    dd_from_peak = peak_pnl - pnl_pct
                    if dd_from_peak >= config.trailing_stop_pct:
                        should_exit = True
                        exit_reason = (f"Trailing stop (peak {peak_pnl:.2%}, "
                                       f"drawback {dd_from_peak:.2%})")

            # 6. Time decay exit (matches backtest engine)
            if not should_exit and getattr(config, 'use_time_decay_exit', False):
                decay_days = getattr(config, 'time_decay_days', 10)
                decay_thresh = getattr(config, 'time_decay_threshold', 0.01)
                if days_held >= decay_days:
                    if abs(pnl_pct) < decay_thresh:
                        should_exit = True
                        exit_reason = (f"Time decay (held {days_held}d, "
                                       f"PnL {pnl_pct:.2%} flat)")

            if should_exit:
                action = 'sell' if side == 'long' else 'cover'
                decisions.append(TradeDecision(
                    symbol=symbol,
                    action=action,
                    target_qty=abs(pos['qty']),
                    signal_strength=0.0,
                    signal_direction=0,
                    reason=exit_reason,
                ))

        # ─── CHECK ENTRIES ─────────────────────────────────────────────

        # Symbols that are being exited today
        exiting_symbols = {d.symbol for d in decisions}

        # Get portfolio equity for sizing and exposure limits
        if current_equity is not None:
            equity = current_equity
        else:
            try:
                if self.connection.config.trading_mode != TradingMode.REPLAY:
                    account = self.connection.get_account()
                    equity = account['portfolio_value']
                else:
                    equity = config.initial_capital
            except Exception:
                equity = config.initial_capital

        # Current exposure (value of all open positions not being exited)
        total_exposure_value = sum(
            abs(pos['qty']) * today_prices.get(sym, pos.get('current_price', pos['entry_price']))
            for sym, pos in current_positions.items()
            if sym not in exiting_symbols and sym in today_prices.index
        )
        current_exposure_pct = total_exposure_value / equity if equity > 0 else 1.0

        for symbol in today_signals.index:
            # Skip if already in position (unless exiting today)
            if symbol in current_positions and symbol not in exiting_symbols:
                continue

            signal_val = today_signals[symbol]
            if pd.isna(signal_val):
                continue

            # Check entry threshold
            if abs(signal_val) <= config.entry_threshold:
                continue

            # Determine direction
            if signal_val > 0:
                direction = -1  # Mean reversion: high signal → short
                action = 'short'
            else:
                direction = 1   # Mean reversion: low signal → long
                action = 'buy'

            price = today_prices.get(symbol, 0)
            if price <= 0:
                continue

            # Short acceleration filter (matches backtest engine)
            if action == 'short' and getattr(config, 'short_accel_filter', False):
                lookback = getattr(config, 'short_accel_lookback', 3)
                threshold = getattr(config, 'short_accel_threshold', 0.10)
                if date in price_df.index and symbol in price_df.columns:
                    date_idx = price_df.index.get_loc(date)
                    if date_idx >= lookback:
                        lookback_price = price_df.iloc[date_idx - lookback][symbol]
                        if lookback_price > 0 and (price - lookback_price) / lookback_price > threshold:
                            continue  # Stock surged — skip short

            # Exposure check (matches backtest: max_total_exposure cap)
            if current_exposure_pct >= config.max_total_exposure:
                break  # No more room for any position

            # Position sizing (matches backtest engine method)
            size_frac = self._compute_position_size(
                config=config,
                signal_strength=abs(signal_val),
                symbol=symbol,
                date=date,
                price_df=price_df,
            )
            position_value = equity * size_frac
            qty = int(position_value / price)

            if qty <= 0:
                continue

            # Check if this entry would exceed max exposure
            entry_exposure = (qty * price) / equity
            if current_exposure_pct + entry_exposure > config.max_total_exposure:
                continue

            current_exposure_pct += entry_exposure

            decisions.append(TradeDecision(
                symbol=symbol,
                action=action,
                target_qty=qty,
                signal_strength=float(signal_val),
                signal_direction=direction,
                reason=f"Entry signal ({signal_val:.3f})",
            ))

        return decisions

    def _compute_position_size(
        self,
        config,
        signal_strength: float,
        symbol: str,
        date: pd.Timestamp,
        price_df: pd.DataFrame,
    ) -> float:
        """
        Compute position size fraction matching backtest engine methods.

        Supports: equal_weight, signal_proportional, volatility_scaled.

        Args:
            config: BacktestConfig instance
            signal_strength: Absolute signal value
            symbol: Ticker symbol
            date: Current date
            price_df: Full price DataFrame (for vol computation)

        Returns:
            Position size as fraction of equity
        """
        method = getattr(config, 'position_size_method', 'equal_weight')

        if method == 'equal_weight':
            return config.max_position_size

        elif method == 'signal_proportional':
            excess = signal_strength - config.entry_threshold
            base = getattr(config, 'signal_prop_base_size', 0.03)
            scale = getattr(config, 'signal_prop_scale_factor', 0.02)
            size = base + scale * max(0, excess)
            return float(np.clip(size, 0.01, config.max_position_size))

        elif method == 'volatility_scaled':
            vol_lookback = getattr(config, 'vol_lookback', 60)
            vol_target = getattr(config, 'vol_target', 0.15)
            vol_min = getattr(config, 'vol_min_size', 0.02)
            vol_max = getattr(config, 'vol_max_size', 0.10)

            if symbol in price_df.columns and date in price_df.index:
                date_idx = price_df.index.get_loc(date)
                if date_idx >= vol_lookback:
                    prices = price_df[symbol].iloc[
                        date_idx - vol_lookback:date_idx + 1
                    ].dropna()
                    if len(prices) >= 10:
                        log_rets = np.log(prices / prices.shift(1)).dropna()
                        realized_vol = log_rets.std() * np.sqrt(252)
                        if realized_vol > 0:
                            size = vol_target / realized_vol
                            return float(np.clip(size, vol_min, vol_max))

            return config.max_position_size  # Fallback

        else:
            return config.max_position_size
