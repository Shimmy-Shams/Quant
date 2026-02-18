"""
Trading Simulation Engine

Provides two simulation modes that sit between backtest and live trading:

1. REPLAY MODE: Replays historical data day-by-day through the live pipeline.
   Validates that the live system produces the same results as the backtest.
   Uses actual historical prices, generates signals incrementally.

2. SHADOW MODE: Runs on live Alpaca data but doesn't submit orders.
   Tracks hypothetical positions and P&L. Builds confidence before going live.

Both modes produce detailed trade logs for comparison against backtest results.
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import numpy as np

from execution.alpaca_executor import AlpacaExecutor, TradeDecision, TradeResult

logger = logging.getLogger(__name__)


@dataclass
class SimulatedPosition:
    """A simulated open position"""
    symbol: str
    qty: int
    side: str               # 'long' or 'short'
    entry_price: float
    entry_date: pd.Timestamp
    signal_strength: float
    current_price: float = 0.0
    peak_price: float = 0.0      # For trailing stop tracking

    @property
    def market_value(self) -> float:
        return abs(self.qty) * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        if self.side == 'long':
            return (self.current_price - self.entry_price) * self.qty
        else:
            return (self.entry_price - self.current_price) * abs(self.qty)

    @property
    def unrealized_pnl_pct(self) -> float:
        if self.entry_price == 0:
            return 0.0
        if self.side == 'long':
            return (self.current_price - self.entry_price) / self.entry_price
        else:
            return (self.entry_price - self.current_price) / self.entry_price


@dataclass
class SimulatedTrade:
    """A completed simulated trade (entry + exit)"""
    symbol: str
    side: str
    qty: int
    entry_price: float
    entry_date: pd.Timestamp
    exit_price: float
    exit_date: pd.Timestamp
    entry_reason: str
    exit_reason: str
    pnl: float
    pnl_pct: float
    holding_days: int
    commission: float


@dataclass 
class DailySnapshot:
    """Daily portfolio state"""
    date: pd.Timestamp
    equity: float
    cash: float
    positions_value: float
    n_positions: int
    n_longs: int
    n_shorts: int
    daily_pnl: float
    daily_return_pct: float
    trades_entered: int
    trades_exited: int
    signals_generated: int
    entries_above_threshold: int


class SimulationEngine:
    """
    Day-by-day simulation engine for replay and shadow modes.
    
    Tracks:
    - Simulated positions (entries, current P&L, exits)  
    - Daily equity curve
    - Completed trade log
    - Signal-to-execution comparison
    """

    def __init__(
        self,
        executor: AlpacaExecutor,
        initial_capital: float = 100000.0,
        commission_pct: float = 0.001,
        slippage_pct: float = 0.0005,
    ):
        """
        Args:
            executor: AlpacaExecutor instance (handles order routing/simulation)
            initial_capital: Starting capital
            commission_pct: Commission per trade (0 for Alpaca, use for backtest parity)
            slippage_pct: Estimated slippage
        """
        self.executor = executor
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct

        # State
        self.cash = initial_capital
        self.positions: Dict[str, SimulatedPosition] = {}
        self.completed_trades: List[SimulatedTrade] = []
        self.daily_snapshots: List[DailySnapshot] = []
        self.all_decisions: List[TradeDecision] = []
        self.all_results: List[TradeResult] = []

    def reset(self):
        """Reset simulation state"""
        self.cash = self.initial_capital
        self.positions = {}
        self.completed_trades = []
        self.daily_snapshots = []
        self.all_decisions = []
        self.all_results = []

    @property
    def equity(self) -> float:
        """
        Current total equity.

        Longs are assets (add to equity), shorts are liabilities (subtract).
        When you short-sell, cash increases by the proceeds but you owe shares
        back — the current market value of those shares is a liability.
        """
        longs_value = sum(
            p.market_value for p in self.positions.values() if p.side == 'long'
        )
        shorts_liability = sum(
            p.market_value for p in self.positions.values() if p.side == 'short'
        )
        return self.cash + longs_value - shorts_liability

    @property
    def positions_as_dict(self) -> Dict[str, Dict]:
        """Current positions in dict format for signal generator"""
        return {
            symbol: {
                'qty': p.qty,
                'side': p.side,
                'entry_price': p.entry_price,
                'entry_date': p.entry_date,
                'current_price': p.current_price,
                'peak_price': p.peak_price if p.peak_price != 0 else p.entry_price,
            }
            for symbol, p in self.positions.items()
        }

    # ─── REPLAY MODE ───────────────────────────────────────────────────

    def run_replay(
        self,
        price_df: pd.DataFrame,
        signal_df: pd.DataFrame,
        volume_df: pd.DataFrame,
        exit_signal_df: Optional[pd.DataFrame],
        config,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
        verbose: bool = True,
    ) -> Dict:
        """
        Replay historical data through the live pipeline day-by-day.

        This validates that our live execution logic matches the backtest.
        Uses the same signals but processes them sequentially (like live would).

        Args:
            price_df: Historical prices (DatetimeIndex × symbols)
            signal_df: Pre-computed signals (same shape as price_df)
            volume_df: Volume data
            exit_signal_df: Exit signals (z-scores for gated mode)
            config: BacktestConfig instance
            start_date: Start replay from this date
            end_date: End replay at this date
            verbose: Print progress

        Returns:
            Dict with replay results and comparison data
        """
        self.reset()

        # Date range
        dates = signal_df.index.sort_values()
        if start_date:
            dates = dates[dates >= start_date]
        if end_date:
            dates = dates[dates <= end_date]

        if verbose:
            print(f"\n{'='*60}")
            print(f"  HISTORICAL REPLAY")
            print(f"  Period: {dates[0].date()} → {dates[-1].date()} ({len(dates)} days)")
            print(f"  Universe: {len(signal_df.columns)} symbols")
            print(f"  Initial Capital: ${self.initial_capital:,.0f}")
            print(f"{'='*60}\n")

        prev_equity = self.initial_capital

        for i, date in enumerate(dates):
            # Get today's prices
            if date not in price_df.index:
                continue
            today_prices = price_df.loc[date].dropna()

            # Update position prices and peak prices (for trailing stop)
            for symbol, pos in self.positions.items():
                if symbol in today_prices.index:
                    pos.current_price = today_prices[symbol]
                    if pos.peak_price == 0:
                        pos.peak_price = pos.entry_price
                    if pos.side == 'long':
                        pos.peak_price = max(pos.peak_price, pos.current_price)
                    else:
                        pos.peak_price = min(pos.peak_price, pos.current_price)

            # Generate decisions (pass current equity for proper sizing)
            decisions = self.executor.generate_decisions_from_signals(
                signal_df=signal_df,
                price_df=price_df,
                volume_df=volume_df,
                exit_signal_df=exit_signal_df,
                date=date,
                current_positions=self.positions_as_dict,
                config=config,
                current_equity=self.equity,
            )

            # Process decisions through simulation
            trades_entered = 0
            trades_exited = 0

            for decision in decisions:
                if decision.action in ('sell', 'cover'):
                    self._process_exit(decision, date, today_prices)
                    trades_exited += 1
                elif decision.action in ('buy', 'short'):
                    self._process_entry(decision, date, today_prices)
                    trades_entered += 1

            self.all_decisions.extend(decisions)

            # Count signals above threshold
            today_signals = signal_df.loc[date].dropna()
            entries_above = (today_signals.abs() > config.entry_threshold).sum()

            # Daily snapshot
            current_equity = self.equity
            daily_pnl = current_equity - prev_equity
            daily_return = daily_pnl / prev_equity if prev_equity > 0 else 0

            snapshot = DailySnapshot(
                date=date,
                equity=current_equity,
                cash=self.cash,
                positions_value=current_equity - self.cash,
                n_positions=len(self.positions),
                n_longs=sum(1 for p in self.positions.values() if p.side == 'long'),
                n_shorts=sum(1 for p in self.positions.values() if p.side == 'short'),
                daily_pnl=daily_pnl,
                daily_return_pct=daily_return,
                trades_entered=trades_entered,
                trades_exited=trades_exited,
                signals_generated=len(today_signals.dropna()),
                entries_above_threshold=int(entries_above),
            )
            self.daily_snapshots.append(snapshot)
            prev_equity = current_equity

            # Progress
            if verbose and (i + 1) % 250 == 0:
                total_return = (current_equity / self.initial_capital - 1) * 100
                print(
                    f"  Day {i+1}/{len(dates)} | {date.date()} | "
                    f"Equity: ${current_equity:,.0f} ({total_return:+.1f}%) | "
                    f"Positions: {len(self.positions)} | "
                    f"Trades: {len(self.completed_trades)}"
                )

        # Final summary
        results = self._compile_results()

        if verbose:
            self._print_summary(results)

        return results

    # ─── SHADOW MODE (Single Day) ──────────────────────────────────────

    def process_shadow_day(
        self,
        date: pd.Timestamp,
        signal_df: pd.DataFrame,
        price_df: pd.DataFrame,
        volume_df: pd.DataFrame,
        exit_signal_df: Optional[pd.DataFrame],
        config,
        verbose: bool = True,
    ) -> Dict:
        """
        Process a single day in shadow mode.
        
        Called daily with fresh data. Maintains state across calls.
        Does NOT submit real orders — tracks hypothetical positions.

        Args:
            date: Today's date
            signal_df: Signals up to today
            price_df: Prices up to today  
            volume_df: Volume up to today
            exit_signal_df: Exit signals up to today
            config: BacktestConfig instance
            verbose: Print trade log

        Returns:
            Dict with today's activity summary
        """
        prev_equity = self.equity

        # Get today's prices
        today_prices = price_df.loc[date].dropna() if date in price_df.index else pd.Series(dtype=float)

        # Update position prices and peak prices (for trailing stop)
        for symbol, pos in self.positions.items():
            if symbol in today_prices.index:
                pos.current_price = today_prices[symbol]
                if pos.peak_price == 0:
                    pos.peak_price = pos.entry_price
                if pos.side == 'long':
                    pos.peak_price = max(pos.peak_price, pos.current_price)
                else:
                    pos.peak_price = min(pos.peak_price, pos.current_price)

        # Generate decisions (pass current equity for proper sizing)
        decisions = self.executor.generate_decisions_from_signals(
            signal_df=signal_df,
            price_df=price_df,
            volume_df=volume_df,
            exit_signal_df=exit_signal_df,
            date=date,
            current_positions=self.positions_as_dict,
            config=config,
            current_equity=self.equity,
        )

        trades_entered = 0
        trades_exited = 0

        for decision in decisions:
            if decision.action in ('sell', 'cover'):
                self._process_exit(decision, date, today_prices)
                trades_exited += 1
            elif decision.action in ('buy', 'short'):
                self._process_entry(decision, date, today_prices)
                trades_entered += 1

        self.all_decisions.extend(decisions)

        current_equity = self.equity
        daily_pnl = current_equity - prev_equity
        daily_return = daily_pnl / prev_equity if prev_equity > 0 else 0

        # Count signals
        today_signals = signal_df.loc[date].dropna() if date in signal_df.index else pd.Series(dtype=float)
        entries_above = (today_signals.abs() > config.entry_threshold).sum()

        snapshot = DailySnapshot(
            date=date,
            equity=current_equity,
            cash=self.cash,
            positions_value=current_equity - self.cash,
            n_positions=len(self.positions),
            n_longs=sum(1 for p in self.positions.values() if p.side == 'long'),
            n_shorts=sum(1 for p in self.positions.values() if p.side == 'short'),
            daily_pnl=daily_pnl,
            daily_return_pct=daily_return,
            trades_entered=trades_entered,
            trades_exited=trades_exited,
            signals_generated=len(today_signals.dropna()),
            entries_above_threshold=int(entries_above),
        )
        self.daily_snapshots.append(snapshot)

        if verbose:
            print(f"\n── Shadow Day: {date.date()} ──")
            print(f"  Equity: ${current_equity:,.2f} (daily: {daily_return:+.3%})")
            print(f"  Positions: {len(self.positions)} ({snapshot.n_longs}L / {snapshot.n_shorts}S)")
            print(f"  Signals > threshold: {entries_above}")
            if trades_entered > 0:
                for d in decisions:
                    if d.action in ('buy', 'short'):
                        print(f"  → ENTRY: {d.action.upper()} {d.target_qty} {d.symbol} "
                              f"(signal={d.signal_strength:.3f})")
            if trades_exited > 0:
                for d in decisions:
                    if d.action in ('sell', 'cover'):
                        print(f"  ← EXIT:  {d.action.upper()} {d.target_qty} {d.symbol} "
                              f"({d.reason})")

        return {
            'date': date,
            'equity': current_equity,
            'daily_pnl': daily_pnl,
            'daily_return': daily_return,
            'trades_entered': trades_entered,
            'trades_exited': trades_exited,
            'n_positions': len(self.positions),
            'decisions': decisions,
        }

    # ─── Internal Position Tracking ────────────────────────────────────

    def _process_entry(
        self,
        decision: TradeDecision,
        date: pd.Timestamp,
        prices: pd.Series,
    ):
        """Process an entry decision"""
        price = prices.get(decision.symbol, 0)
        if price <= 0:
            return

        # Apply slippage
        if decision.action == 'buy':
            entry_price = price * (1 + self.slippage_pct)
            side = 'long'
        else:  # short
            entry_price = price * (1 - self.slippage_pct)
            side = 'short'

        # Commission
        trade_value = entry_price * decision.target_qty
        commission = trade_value * self.commission_pct

        # Update cash
        if side == 'long':
            self.cash -= (trade_value + commission)
        else:
            self.cash += (trade_value - commission)  # Short: receive cash

        self.positions[decision.symbol] = SimulatedPosition(
            symbol=decision.symbol,
            qty=decision.target_qty,
            side=side,
            entry_price=entry_price,
            entry_date=date,
            signal_strength=decision.signal_strength,
            current_price=price,
            peak_price=entry_price,
        )

    def _process_exit(
        self,
        decision: TradeDecision,
        date: pd.Timestamp,
        prices: pd.Series,
    ):
        """Process an exit decision"""
        symbol = decision.symbol
        if symbol not in self.positions:
            return

        pos = self.positions[symbol]
        price = prices.get(symbol, pos.current_price)

        # Apply slippage
        if pos.side == 'long':
            exit_price = price * (1 - self.slippage_pct)
        else:
            exit_price = price * (1 + self.slippage_pct)

        # Calculate P&L
        trade_value = exit_price * abs(pos.qty)
        commission = trade_value * self.commission_pct

        if pos.side == 'long':
            pnl = (exit_price - pos.entry_price) * pos.qty - commission
            self.cash += (trade_value - commission)
        else:
            pnl = (pos.entry_price - exit_price) * abs(pos.qty) - commission
            self.cash -= (trade_value + commission)

        pnl_pct = pnl / (pos.entry_price * abs(pos.qty)) if pos.entry_price > 0 else 0

        holding_days = (date - pos.entry_date).days

        self.completed_trades.append(SimulatedTrade(
            symbol=symbol,
            side=pos.side,
            qty=abs(pos.qty),
            entry_price=pos.entry_price,
            entry_date=pos.entry_date,
            exit_price=exit_price,
            exit_date=date,
            entry_reason=f"Signal: {pos.signal_strength:.3f}",
            exit_reason=decision.reason,
            pnl=pnl,
            pnl_pct=pnl_pct,
            holding_days=holding_days,
            commission=commission * 2,  # Entry + exit
        ))

        del self.positions[symbol]

    # ─── Results Compilation ───────────────────────────────────────────

    def _compile_results(self) -> Dict:
        """Compile simulation results into summary dict"""
        if not self.daily_snapshots:
            return {'error': 'No data processed'}

        equity_series = pd.Series(
            {s.date: s.equity for s in self.daily_snapshots}
        )
        returns = equity_series.pct_change().dropna()

        # Trade stats
        trades_df = pd.DataFrame([{
            'symbol': t.symbol,
            'side': t.side,
            'qty': t.qty,
            'entry_price': t.entry_price,
            'exit_price': t.exit_price,
            'entry_date': t.entry_date,
            'exit_date': t.exit_date,
            'entry_reason': t.entry_reason,
            'exit_reason': t.exit_reason,
            'pnl': t.pnl,
            'pnl_pct': t.pnl_pct,
            'holding_days': t.holding_days,
            'commission': t.commission,
        } for t in self.completed_trades]) if self.completed_trades else pd.DataFrame()

        total_return = (equity_series.iloc[-1] / equity_series.iloc[0] - 1) * 100

        # Sharpe
        if len(returns) > 0 and returns.std() > 0:
            sharpe = np.sqrt(252) * returns.mean() / returns.std()
        else:
            sharpe = 0.0

        # Max drawdown
        peak = equity_series.cummax()
        drawdown = (equity_series - peak) / peak
        max_dd = drawdown.min() * 100

        # Win rate
        if len(trades_df) > 0:
            win_rate = (trades_df['pnl'] > 0).mean() * 100
            avg_pnl = trades_df['pnl_pct'].mean() * 100
            avg_win = trades_df[trades_df['pnl'] > 0]['pnl_pct'].mean() * 100 if (trades_df['pnl'] > 0).any() else 0
            avg_loss = trades_df[trades_df['pnl'] <= 0]['pnl_pct'].mean() * 100 if (trades_df['pnl'] <= 0).any() else 0
        else:
            win_rate = avg_pnl = avg_win = avg_loss = 0.0

        return {
            'equity_curve': equity_series,
            'returns': returns,
            'trades_df': trades_df,
            'total_return_pct': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown_pct': max_dd,
            'total_trades': len(self.completed_trades),
            'win_rate': win_rate,
            'avg_pnl_pct': avg_pnl,
            'avg_win_pct': avg_win,
            'avg_loss_pct': avg_loss,
            'final_equity': equity_series.iloc[-1],
            'open_positions': len(self.positions),
            'daily_snapshots': self.daily_snapshots,
        }

    def _print_summary(self, results: Dict):
        """Print formatted simulation summary"""
        print(f"\n{'='*60}")
        print(f"  SIMULATION RESULTS")
        print(f"{'='*60}")
        print(f"  Final Equity:    ${results['final_equity']:,.2f}")
        print(f"  Total Return:    {results['total_return_pct']:+.2f}%")
        print(f"  Sharpe Ratio:    {results['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown:    {results['max_drawdown_pct']:.2f}%")
        print(f"  Total Trades:    {results['total_trades']}")
        print(f"  Win Rate:        {results['win_rate']:.1f}%")
        print(f"  Avg Trade P&L:   {results['avg_pnl_pct']:+.2f}%")
        print(f"  Avg Winner:      {results['avg_win_pct']:+.2f}%")
        print(f"  Avg Loser:       {results['avg_loss_pct']:+.2f}%")
        print(f"  Open Positions:  {results['open_positions']}")
        print(f"{'='*60}")

    # ─── Export ─────────────────────────────────────────────────────────

    def export_trade_log(self, path: Path):
        """Export completed trades to CSV"""
        if not self.completed_trades:
            print("No trades to export")
            return

        trades_df = pd.DataFrame([{
            'symbol': t.symbol,
            'side': t.side,
            'qty': t.qty,
            'entry_price': t.entry_price,
            'exit_price': t.exit_price,
            'entry_date': t.entry_date,
            'exit_date': t.exit_date,
            'entry_reason': t.entry_reason,
            'exit_reason': t.exit_reason,
            'pnl': t.pnl,
            'pnl_pct': t.pnl_pct,
            'holding_days': t.holding_days,
            'commission': t.commission,
        } for t in self.completed_trades])

        path.parent.mkdir(parents=True, exist_ok=True)
        trades_df.to_csv(path, index=False)
        print(f"Trade log exported: {path} ({len(trades_df)} trades)")

    def export_equity_curve(self, path: Path):
        """Export daily equity curve to CSV"""
        if not self.daily_snapshots:
            print("No snapshots to export")
            return

        df = pd.DataFrame([{
            'date': s.date,
            'equity': s.equity,
            'cash': s.cash,
            'positions_value': s.positions_value,
            'n_positions': s.n_positions,
            'daily_pnl': s.daily_pnl,
            'daily_return_pct': s.daily_return_pct,
            'trades_entered': s.trades_entered,
            'trades_exited': s.trades_exited,
        } for s in self.daily_snapshots])

        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        print(f"Equity curve exported: {path} ({len(df)} days)")

    def compare_with_backtest(
        self,
        backtest_equity: pd.Series,
        label_sim: str = 'Simulation',
        label_bt: str = 'Backtest',
    ) -> Dict:
        """
        Compare simulation results with backtest equity curve.

        Args:
            backtest_equity: Backtest equity series (from BacktestEngine results)
            label_sim: Label for simulation
            label_bt: Label for backtest

        Returns:
            Comparison metrics dict
        """
        import matplotlib.pyplot as plt

        sim_equity = pd.Series(
            {s.date: s.equity for s in self.daily_snapshots}
        )

        # Align dates
        common_dates = sim_equity.index.intersection(backtest_equity.index)
        if len(common_dates) == 0:
            print("No overlapping dates for comparison")
            return {}

        sim_aligned = sim_equity.loc[common_dates]
        bt_aligned = backtest_equity.loc[common_dates]

        # Normalize both to 100
        sim_norm = sim_aligned / sim_aligned.iloc[0] * 100
        bt_norm = bt_aligned / bt_aligned.iloc[0] * 100

        # Correlation
        correlation = sim_norm.corr(bt_norm)

        # Tracking error
        sim_returns = sim_norm.pct_change().dropna()
        bt_returns = bt_norm.pct_change().dropna()
        tracking_error = (sim_returns - bt_returns).std() * np.sqrt(252) * 100

        # Plot comparison
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        axes[0].plot(sim_norm.index, sim_norm.values, label=label_sim, alpha=0.8)
        axes[0].plot(bt_norm.index, bt_norm.values, label=label_bt, alpha=0.8, linestyle='--')
        axes[0].set_title('Equity Curve Comparison (Normalized to 100)')
        axes[0].set_ylabel('Normalized Equity')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        diff = sim_norm - bt_norm
        axes[1].bar(diff.index, diff.values, alpha=0.5, color='steelblue', width=2)
        axes[1].axhline(y=0, color='black', linewidth=0.5)
        axes[1].set_title(f'Deviation (Correlation: {correlation:.4f}, Tracking Error: {tracking_error:.2f}%)')
        axes[1].set_ylabel('Normalized Difference')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return {
            'correlation': correlation,
            'tracking_error_pct': tracking_error,
            'max_deviation': float(diff.abs().max()),
            'avg_deviation': float(diff.abs().mean()),
            'common_days': len(common_dates),
        }
