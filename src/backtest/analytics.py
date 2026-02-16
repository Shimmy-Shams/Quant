"""
Performance Analytics Module

Comprehensive post-backtest analytics for mean reversion strategies.
Includes capital utilization diagnostics, risk metrics, rolling analytics,
trade-level analysis, turnover & cost analysis, and regime-based analysis.
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from backtest.engine import BacktestConfig, BacktestResults, Trade


# ---------------------------------------------------------------------------
# Result Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CapitalUtilizationReport:
    """Results from capital utilization diagnostic."""
    total_trading_days: int = 0
    days_with_positions: int = 0
    days_idle: int = 0
    avg_exposure_pct: float = 0.0
    max_concurrent_positions: int = 0
    avg_position_value: float = 0.0
    avg_equity_at_entry: float = 0.0
    avg_position_pct_equity: float = 0.0
    avg_holding_days: float = 0.0
    days_any_signal_above_thresh: int = 0
    days_no_signal: int = 0
    avg_signals_per_active_day: float = 0.0
    max_signals_in_one_day: int = 0
    avg_idle_pct: float = 0.0
    avg_idle_dollar: float = 0.0
    opportunity_cost_5pct: float = 0.0
    return_on_deployed: float = 0.0
    return_on_total: float = 0.0
    position_count_distribution: Dict[int, int] = field(default_factory=dict)
    yearly_trade_stats: Dict[int, Dict] = field(default_factory=dict)


@dataclass
class RiskMetricsReport:
    """Results from risk metrics dashboard."""
    var_95: float = 0.0
    var_99: float = 0.0
    cvar_95: float = 0.0
    cvar_99: float = 0.0
    tail_ratio: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    jb_stat: float = 0.0
    jb_pval: float = 0.0
    omega_ratio: float = 0.0
    ulcer_index: float = 0.0
    ulcer_performance_index: float = 0.0
    gain_to_pain: float = 0.0
    daily_mean: float = 0.0
    daily_median: float = 0.0
    daily_std: float = 0.0
    best_day: float = 0.0
    worst_day: float = 0.0
    pct_positive: float = 0.0
    pct_negative: float = 0.0
    pct_flat: float = 0.0


@dataclass
class TurnoverReport:
    """Results from turnover & cost analysis."""
    gross_pnl: float = 0.0
    total_commission: float = 0.0
    net_pnl: float = 0.0
    cost_drag_pct: float = 0.0
    gross_return: float = 0.0
    net_return: float = 0.0
    total_traded_value: float = 0.0
    avg_portfolio_value: float = 0.0
    annual_turnover: float = 0.0
    avg_trades_per_year: float = 0.0
    commission_gross_ratio: float = 0.0
    avg_commission_per_trade: float = 0.0
    breakeven_one_way_cost: float = 0.0
    current_one_way_cost: float = 0.0
    safety_margin_pct: float = 0.0
    safety_margin_multiple: float = 0.0
    slippage_sensitivity: List[Dict] = field(default_factory=list)


@dataclass
class RegimeResult:
    """Metrics for a single market regime."""
    name: str = ""
    start: str = ""
    end: str = ""
    trades: int = 0
    win_rate: float = 0.0
    avg_pnl: float = 0.0
    period_return: float = 0.0
    sharpe: float = 0.0
    max_dd: float = 0.0
    var_95: float = 0.0
    exposure: float = 0.0


# ---------------------------------------------------------------------------
# Default Regime Definitions
# ---------------------------------------------------------------------------

DEFAULT_REGIMES = [
    ('Pre-GFC Bull', '2006-02-14', '2007-09-30'),
    ('GFC', '2007-10-01', '2009-03-31'),
    ('Recovery 2009-11', '2009-04-01', '2011-12-31'),
    ('Low Vol 2012-14', '2012-01-01', '2014-12-31'),
    ('Oil/China Crash', '2015-06-01', '2016-02-28'),
    ('Low Vol Bull 2016-17', '2016-03-01', '2017-12-31'),
    ('Volmageddon 2018', '2018-01-01', '2018-12-31'),
    ('Late Cycle 2019', '2019-01-01', '2019-12-31'),
    ('COVID Crash', '2020-02-01', '2020-04-30'),
    ('Post-COVID Bull', '2020-05-01', '2021-12-31'),
    ('2022 Bear Market', '2022-01-01', '2022-12-31'),
    ('2023 Recovery', '2023-01-01', '2023-12-31'),
    ('2024 Bull', '2024-01-01', '2024-12-31'),
    ('2025-present', '2025-01-01', '2026-02-28'),
]


# ---------------------------------------------------------------------------
# Main Analytics Class
# ---------------------------------------------------------------------------

class PerformanceAnalytics:
    """
    Comprehensive performance analytics for backtest results.

    Encapsulates all Phase 3B analytics: capital utilization diagnostics,
    risk metrics, rolling analytics, trade-level analysis, turnover & cost
    analysis, and regime-based analysis.

    Parameters
    ----------
    results : BacktestResults
        The completed backtest results.
    config : BacktestConfig
        The backtest configuration used.
    signal_df : pd.DataFrame
        DataFrame of composite signals (symbols as columns, dates as index).
    output_dir : Path, optional
        Directory for saving plots. Created if it does not exist.
    """

    def __init__(
        self,
        results: BacktestResults,
        config: BacktestConfig,
        signal_df: pd.DataFrame,
        output_dir: Optional[Path] = None,
    ):
        self.results = results
        self.config = config
        self.signal_df = signal_df
        self.output_dir = Path(output_dir) if output_dir else None

        # Derived series
        self.equity = results.equity_curve
        self.returns = results.returns

        # Build trades DataFrame once
        self.trades_df = self._build_trades_df()

        # Computed in capital utilization, reused in regime analysis
        self._avg_size_pct: Optional[float] = None

        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_trades_df(self) -> pd.DataFrame:
        """Convert list of Trade objects into a tidy DataFrame."""
        rows = [
            {
                'symbol': t.symbol,
                'entry_date': t.entry_date,
                'exit_date': t.exit_date,
                'side': t.side,
                'pnl': t.pnl,
                'pnl_pct': t.pnl_pct * 100,
                'holding_days': t.holding_days,
                'exit_reason': t.exit_reason,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'entry_signal': t.entry_signal,
                'shares': t.shares,
            }
            for t in self.results.trades
        ]
        df = pd.DataFrame(rows)
        df['entry_date'] = pd.to_datetime(df['entry_date'])
        df['exit_date'] = pd.to_datetime(df['exit_date'])
        df['year'] = df['entry_date'].dt.year
        df['entry_value'] = df['shares'] * df['entry_price']
        return df

    @staticmethod
    def _trailing_max_dd(eq_series: pd.Series, window: int = 252) -> pd.Series:
        """Compute trailing max drawdown over a rolling window."""
        result = pd.Series(index=eq_series.index, dtype=float)
        eq_vals = eq_series.values
        for i in range(window, len(eq_vals)):
            window_eq = eq_vals[i - window : i + 1]
            running_max = np.maximum.accumulate(window_eq)
            dd = (window_eq - running_max) / running_max
            result.iloc[i] = dd.min()
        return result

    # ==================================================================
    # 3B.0 — Capital Utilization Diagnostic
    # ==================================================================

    def capital_utilization(self, print_report: bool = True) -> CapitalUtilizationReport:
        """
        Diagnose capital utilization: market presence, position sizing,
        concurrent positions, signal gates, idle capital, and annual frequency.
        """
        equity = self.equity
        trades_df = self.trades_df
        config = self.config
        n_days = len(equity)

        # --- Market presence ---
        trade_day_ranges = []
        for _, t in trades_df.iterrows():
            trade_day_ranges.append(
                pd.date_range(t['entry_date'], t['exit_date'], freq='B')
            )

        if trade_day_ranges:
            all_trade_days = set()
            for dr in trade_day_ranges:
                all_trade_days.update(dr)
            days_in_market = len(all_trade_days.intersection(equity.index))
        else:
            days_in_market = 0

        # --- Position sizing when active ---
        avg_pos_value = trades_df['entry_value'].mean()
        avg_equity_at_entry = equity.loc[trades_df['entry_date']].mean()
        avg_size_pct = avg_pos_value / avg_equity_at_entry * 100
        self._avg_size_pct = avg_size_pct  # cache for regime analysis

        # --- Concurrent positions distribution ---
        pos_count_by_day = pd.Series(0, index=equity.index)
        for _, t in trades_df.iterrows():
            mask = (equity.index >= t['entry_date']) & (equity.index <= t['exit_date'])
            pos_count_by_day[mask] += 1

        pos_dist: Dict[int, int] = {}
        for n_pos in range(0, min(int(pos_count_by_day.max()) + 1, 12)):
            pos_dist[n_pos] = int((pos_count_by_day == n_pos).sum())

        # --- Signal gate analysis ---
        abs_signals = self.signal_df.abs()
        days_any_above = int((abs_signals > config.entry_threshold).any(axis=1).sum())
        max_daily_signals = (abs_signals > config.entry_threshold).sum(axis=1)
        avg_signals_active = (
            float(max_daily_signals[max_daily_signals > 0].mean())
            if (max_daily_signals > 0).any()
            else 0.0
        )

        # --- Idle capital ---
        avg_idle_pct = (1 - self.results.avg_exposure) * 100
        avg_idle_dollar = equity.mean() * (1 - self.results.avg_exposure)
        opp_cost = avg_idle_dollar * 0.05

        # --- Annual trade frequency ---
        yearly_trades = trades_df.groupby('year').size()
        yearly_avg_hold = trades_df.groupby('year')['holding_days'].mean()
        yearly_stats: Dict[int, Dict] = {}
        for yr in yearly_trades.index:
            cap_deployed_days = yearly_trades[yr] * yearly_avg_hold[yr]
            yearly_stats[int(yr)] = {
                'trades': int(yearly_trades[yr]),
                'avg_hold': float(yearly_avg_hold[yr]),
                'position_days': float(cap_deployed_days),
            }

        report = CapitalUtilizationReport(
            total_trading_days=n_days,
            days_with_positions=days_in_market,
            days_idle=n_days - days_in_market,
            avg_exposure_pct=self.results.avg_exposure * 100,
            max_concurrent_positions=self.results.max_positions,
            avg_position_value=avg_pos_value,
            avg_equity_at_entry=avg_equity_at_entry,
            avg_position_pct_equity=avg_size_pct,
            avg_holding_days=float(trades_df['holding_days'].mean()),
            days_any_signal_above_thresh=days_any_above,
            days_no_signal=n_days - days_any_above,
            avg_signals_per_active_day=avg_signals_active,
            max_signals_in_one_day=int(max_daily_signals.max()),
            avg_idle_pct=avg_idle_pct,
            avg_idle_dollar=avg_idle_dollar,
            opportunity_cost_5pct=opp_cost,
            return_on_deployed=float(
                trades_df['pnl'].sum() / trades_df['entry_value'].sum() * 100
            ),
            return_on_total=self.results.total_return * 100,
            position_count_distribution=pos_dist,
            yearly_trade_stats=yearly_stats,
        )

        if print_report:
            self._print_capital_utilization(report)

        return report

    def _print_capital_utilization(self, r: CapitalUtilizationReport) -> None:
        pnl_total = self.trades_df['pnl'].sum()
        print("=" * 80)
        print("CAPITAL UTILIZATION DIAGNOSTIC")
        print("=" * 80)

        print(f"\n--- Market Presence ---")
        print(f"  Total trading days:            {r.total_trading_days:,}")
        print(f"  Days with open positions:      {r.days_with_positions:,} ({r.days_with_positions / r.total_trading_days * 100:.1f}%)")
        print(f"  Days fully idle (0 positions): {r.days_idle:,} ({r.days_idle / r.total_trading_days * 100:.1f}%)")
        print(f"  Avg exposure:                  {r.avg_exposure_pct:.2f}%")
        print(f"  Max concurrent positions:      {r.max_concurrent_positions}")

        print(f"\n--- Position Sizing When Active ---")
        print(f"  Avg position value:    ${r.avg_position_value:,.0f}")
        print(f"  Avg equity at entry:   ${r.avg_equity_at_entry:,.0f}")
        print(f"  Avg position % equity: {r.avg_position_pct_equity:.2f}%")
        print(f"  Max position size cap: {self.config.max_position_size * 100:.0f}%")
        print(f"  Avg holding period:    {r.avg_holding_days:.1f} days")

        print(f"\n--- Concurrent Positions Distribution ---")
        for n_pos, count in sorted(r.position_count_distribution.items()):
            pct = count / r.total_trading_days * 100
            bar = "█" * int(pct / 2)
            print(f"  {n_pos:2d} positions: {count:5d} days ({pct:5.1f}%) {bar}")

        print(f"\n--- Signal Gate Analysis ---")
        print(f"  Entry threshold:                {self.config.entry_threshold}")
        print(f"  Days with ANY signal > thresh:  {r.days_any_signal_above_thresh} ({r.days_any_signal_above_thresh / r.total_trading_days * 100:.1f}%)")
        print(f"  Days with NO signal > thresh:   {r.days_no_signal} ({r.days_no_signal / r.total_trading_days * 100:.1f}%)")
        print(f"  Avg signals > thresh per day:   {r.avg_signals_per_active_day:.2f}")
        print(f"  Max signals > thresh in one day: {r.max_signals_in_one_day}")

        print(f"\n--- Idle Capital Analysis ---")
        print(f"  Avg idle capital:     {r.avg_idle_pct:.1f}% = ${r.avg_idle_dollar:,.0f}")
        print(f"  At 5% risk-free rate: ${r.opportunity_cost_5pct:,.0f}/yr opportunity cost")
        print(f"  Total P&L generated:  ${pnl_total:,.0f}")
        print(f"  Return on DEPLOYED capital: {r.return_on_deployed:.1f}%")
        print(f"  Return on TOTAL capital:    {r.return_on_total:.1f}%")

        print(f"\n--- Annual Trade Frequency ---")
        for yr, s in sorted(r.yearly_trade_stats.items()):
            print(f"  {yr}: {s['trades']:3d} trades × {s['avg_hold']:.1f}d avg = ~{s['position_days']:.0f} position-days")

    # ==================================================================
    # 3B.1 — Risk Metrics Dashboard
    # ==================================================================

    def risk_metrics(self, print_report: bool = True) -> RiskMetricsReport:
        """Compute VaR, CVaR, Tail Ratio, Omega, Skew, Kurtosis, Ulcer Index."""
        ret_vals = self.returns.values[1:]  # skip day 0

        # VaR
        var_95 = float(np.percentile(ret_vals, 5))
        var_99 = float(np.percentile(ret_vals, 1))

        # CVaR
        cvar_95 = float(ret_vals[ret_vals <= var_95].mean())
        cvar_99 = float(ret_vals[ret_vals <= var_99].mean())

        # Tail ratio
        p95_up = float(np.percentile(ret_vals, 95))
        tail_ratio = abs(p95_up / var_95) if var_95 != 0 else float('inf')

        # Distribution shape
        skewness = float(stats.skew(ret_vals))
        kurt = float(stats.kurtosis(ret_vals))
        jb_stat, jb_pval = stats.jarque_bera(ret_vals)

        # Omega ratio (threshold = 0)
        gains = ret_vals[ret_vals > 0]
        losses = -ret_vals[ret_vals <= 0]
        omega = float(gains.sum() / losses.sum()) if losses.sum() > 0 else float('inf')

        # Ulcer index
        equity_vals = self.equity.values
        running_max = np.maximum.accumulate(equity_vals)
        pct_dd = ((equity_vals - running_max) / running_max) * 100
        ulcer = float(np.sqrt(np.mean(pct_dd ** 2)))
        ulcer_perf = (
            self.results.annualized_return * 100 / ulcer if ulcer > 0 else float('inf')
        )

        # Gain-to-pain
        gp = (
            float(ret_vals.sum() / abs(ret_vals[ret_vals < 0].sum()))
            if (ret_vals < 0).any()
            else float('inf')
        )

        report = RiskMetricsReport(
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            tail_ratio=tail_ratio,
            skewness=skewness,
            kurtosis=kurt,
            jb_stat=float(jb_stat),
            jb_pval=float(jb_pval),
            omega_ratio=omega,
            ulcer_index=ulcer,
            ulcer_performance_index=ulcer_perf,
            gain_to_pain=gp,
            daily_mean=float(ret_vals.mean()),
            daily_median=float(np.median(ret_vals)),
            daily_std=float(ret_vals.std()),
            best_day=float(ret_vals.max()),
            worst_day=float(ret_vals.min()),
            pct_positive=float((ret_vals > 0).mean() * 100),
            pct_negative=float((ret_vals < 0).mean() * 100),
            pct_flat=float((ret_vals == 0).mean() * 100),
        )

        if print_report:
            self._print_risk_metrics(report)

        return report

    def _print_risk_metrics(self, r: RiskMetricsReport) -> None:
        print("=" * 80)
        print("3B.1 — RISK METRICS DASHBOARD")
        print("=" * 80)

        print(f"\n--- Value at Risk (Historical) ---")
        print(f"  VaR 95%: {r.var_95 * 100:.4f}%  (on 95% of days, daily loss < {abs(r.var_95) * 100:.4f}%)")
        print(f"  VaR 99%: {r.var_99 * 100:.4f}%  (on 99% of days, daily loss < {abs(r.var_99) * 100:.4f}%)")

        print(f"\n--- Conditional VaR (Expected Shortfall) ---")
        print(f"  CVaR 95%: {r.cvar_95 * 100:.4f}%  (avg loss on worst 5% of days)")
        print(f"  CVaR 99%: {r.cvar_99 * 100:.4f}%  (avg loss on worst 1% of days)")

        print(f"\n--- Tail Ratio ---")
        p95_up = abs(r.var_95) * r.tail_ratio  # reconstruct
        print(f"  95th percentile (gains): +{p95_up * 100:.4f}%")
        print(f"  5th percentile (losses): {r.var_95 * 100:.4f}%")
        print(f"  Tail Ratio: {r.tail_ratio:.2f}x  ({'✓ Positive skew' if r.tail_ratio > 1 else '✗ Fat left tail'})")

        print(f"\n--- Distribution Shape ---")
        print(f"  Skewness:  {r.skewness:.4f}  ({'Right-skewed (good)' if r.skewness > 0 else 'Left-skewed (crash risk)'})")
        print(f"  Kurtosis:  {r.kurtosis:.4f}  ({'Leptokurtic (fat tails)' if r.kurtosis > 0 else 'Platykurtic (thin tails)'})")
        print(f"  Jarque-Bera: stat={r.jb_stat:.1f}, p={r.jb_pval:.6f}  ({'Non-normal' if r.jb_pval < 0.05 else 'Normal'})")

        print(f"\n--- Omega Ratio (threshold=0%) ---")
        print(f"  Omega: {r.omega_ratio:.2f}  (sum of gains / sum of losses, >1 = profitable)")

        print(f"\n--- Ulcer Index ---")
        print(f"  Ulcer Index: {r.ulcer_index:.4f}%  (lower = smoother equity curve)")
        print(f"  Ulcer Performance Index: {r.ulcer_performance_index:.2f}  (return / ulcer, higher = better)")

        print(f"\n--- Gain-to-Pain Ratio ---")
        print(f"  G/P Ratio: {r.gain_to_pain:.2f}  (total gains / total losses, >1 = net profitable)")

        print(f"\n--- Daily Return Statistics ---")
        print(f"  Mean:      {r.daily_mean * 100:.4f}%")
        print(f"  Median:    {r.daily_median * 100:.4f}%")
        print(f"  Std Dev:   {r.daily_std * 100:.4f}%")
        print(f"  Best Day:  +{r.best_day * 100:.4f}%")
        print(f"  Worst Day: {r.worst_day * 100:.4f}%")
        print(f"  % Positive: {r.pct_positive:.1f}%")
        print(f"  % Negative: {r.pct_negative:.1f}%")
        print(f"  % Flat:     {r.pct_flat:.1f}%")

    # ==================================================================
    # 3B.2 — Rolling Analytics (Edge Decay Detection)
    # ==================================================================

    def rolling_analytics(self, save: bool = True, show: bool = True) -> None:
        """Plot rolling Sharpe, win rate, EV per trade, and trailing max drawdown."""
        returns = self.returns
        trades_df = self.trades_df
        equity = self.equity

        fig, axes = plt.subplots(4, 1, figsize=(16, 16), sharex=True)

        # --- Rolling Sharpe (63d, 126d, 252d) ---
        for window, color, label in [
            (63, 'steelblue', '63d (Quarterly)'),
            (126, 'darkorange', '126d (Semi-Annual)'),
            (252, 'darkred', '252d (Annual)'),
        ]:
            roll_mean = returns.rolling(window).mean()
            roll_std = returns.rolling(window).std()
            roll_sharpe = (roll_mean / roll_std) * np.sqrt(252)
            axes[0].plot(
                roll_sharpe.index, roll_sharpe, label=label,
                alpha=0.7, linewidth=1.2, color=color,
            )

        axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[0].axhline(y=2, color='green', linestyle='--', alpha=0.3, label='Sharpe=2 target')
        axes[0].set_ylabel('Sharpe Ratio')
        axes[0].set_title('Rolling Sharpe Ratio — Edge Decay Detection')
        axes[0].legend(loc='upper left', fontsize=9)
        axes[0].grid(alpha=0.3)
        axes[0].set_ylim(-5, 25)

        # --- Rolling Win Rate (trailing 50 trades) ---
        if len(trades_df) > 10:
            trades_sorted = trades_df.sort_values('exit_date')
            win_flags = (trades_sorted['pnl_pct'] > 0).astype(float)
            rolling_wr = win_flags.rolling(50, min_periods=10).mean() * 100
            axes[1].plot(
                trades_sorted['exit_date'].values, rolling_wr.values,
                color='teal', linewidth=1.5, label='Trailing 50 trades',
            )
            axes[1].axhline(y=95, color='green', linestyle='--', alpha=0.3, label='95% target')
            axes[1].axhline(y=90, color='orange', linestyle='--', alpha=0.3, label='90% warning')
        axes[1].set_ylabel('Win Rate (%)')
        axes[1].set_title('Rolling Win Rate (Trailing 50 Trades)')
        axes[1].legend(loc='lower left', fontsize=9)
        axes[1].grid(alpha=0.3)
        axes[1].set_ylim(70, 102)

        # --- Rolling EV per Trade ---
        if len(trades_df) > 10:
            trades_sorted = trades_df.sort_values('exit_date')
            rolling_ev = trades_sorted['pnl_pct'].rolling(50, min_periods=10).mean()
            axes[2].plot(
                trades_sorted['exit_date'].values, rolling_ev.values,
                color='purple', linewidth=1.5, label='Trailing 50 trades',
            )
            axes[2].axhline(y=0, color='red', linestyle='-', linewidth=0.5)
            axes[2].fill_between(
                trades_sorted['exit_date'].values, 0, rolling_ev.values,
                where=rolling_ev.values > 0, alpha=0.2, color='green',
            )
            axes[2].fill_between(
                trades_sorted['exit_date'].values, 0, rolling_ev.values,
                where=rolling_ev.values < 0, alpha=0.2, color='red',
            )
        axes[2].set_ylabel('Avg PnL per Trade (%)')
        axes[2].set_title('Rolling EV per Trade (Trailing 50 Trades)')
        axes[2].legend(loc='upper left', fontsize=9)
        axes[2].grid(alpha=0.3)

        # --- Rolling Max Drawdown (252d window) ---
        trailing_dd = self._trailing_max_dd(equity, 252)
        axes[3].fill_between(
            trailing_dd.index, 0, trailing_dd.values * 100, alpha=0.5, color='red',
        )
        axes[3].set_ylabel('Max Drawdown (%)')
        axes[3].set_title('Trailing 252-Day Maximum Drawdown')
        axes[3].set_xlabel('Date')
        axes[3].grid(alpha=0.3)

        for ax in axes:
            ax.xaxis.set_major_locator(mdates.YearLocator(2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        plt.tight_layout()
        if save and self.output_dir:
            path = self.output_dir / '3B2_rolling_analytics.png'
            plt.savefig(str(path), dpi=150, bbox_inches='tight')
            print(f"Saved: {path}")
        if show:
            plt.show()
        else:
            plt.close(fig)

    # ==================================================================
    # 3B.3 — Trade-Level Analytics
    # ==================================================================

    def trade_analytics(self, save: bool = True, show: bool = True) -> None:
        """Streaks, PnL distribution, holding period, exit reasons, monthly heatmap, trade clustering."""
        trades_df = self.trades_df
        equity = self.equity

        fig = plt.figure(figsize=(18, 20))
        trades_sorted = trades_df.sort_values('exit_date').reset_index(drop=True)
        wins = (trades_sorted['pnl_pct'] > 0).astype(int)

        # --- Win/Loss Streaks ---
        ax1 = fig.add_subplot(4, 2, 1)
        streaks = []
        current_streak = 1
        for i in range(1, len(wins)):
            if wins.iloc[i] == wins.iloc[i - 1]:
                current_streak += 1
            else:
                streaks.append((wins.iloc[i - 1], current_streak))
                current_streak = 1
        streaks.append((wins.iloc[-1], current_streak))

        win_streaks = [s[1] for s in streaks if s[0] == 1]
        loss_streaks = [s[1] for s in streaks if s[0] == 0]

        print("=" * 80)
        print("3B.3 — TRADE-LEVEL ANALYTICS")
        print("=" * 80)
        print(f"\n--- Win/Loss Streaks ---")
        print(f"  Max winning streak: {max(win_streaks)} trades")
        print(f"  Avg winning streak: {np.mean(win_streaks):.1f} trades")
        avg_ls = f"{np.mean(loss_streaks):.1f}" if loss_streaks else "0"
        print(f"  Max losing streak:  {max(loss_streaks) if loss_streaks else 0} trades")
        print(f"  Avg losing streak:  {avg_ls} trades")

        streak_vals = [s[1] * (1 if s[0] == 1 else -1) for s in streaks]
        colors_s = ['green' if v > 0 else 'red' for v in streak_vals]
        ax1.bar(range(len(streak_vals)), streak_vals, color=colors_s, alpha=0.7, width=1.0)
        ax1.set_ylabel('Streak Length')
        ax1.set_title('Win (+) / Loss (-) Streaks Over Time')
        ax1.axhline(y=0, color='black', linewidth=0.5)
        ax1.grid(alpha=0.3)

        # --- PnL Distribution ---
        ax2 = fig.add_subplot(4, 2, 2)
        ax2.hist(trades_sorted['pnl_pct'], bins=50, alpha=0.7, edgecolor='black', color='steelblue')
        ax2.axvline(x=0, color='red', linestyle='--', linewidth=1.5)
        ax2.axvline(
            x=trades_sorted['pnl_pct'].mean(), color='green', linestyle='--',
            label=f'Mean: {trades_sorted["pnl_pct"].mean():.2f}%',
        )
        ax2.axvline(
            x=trades_sorted['pnl_pct'].median(), color='orange', linestyle='--',
            label=f'Median: {trades_sorted["pnl_pct"].median():.2f}%',
        )
        ax2.set_xlabel('PnL (%)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Trade PnL Distribution')
        ax2.legend(fontsize=9)
        ax2.grid(alpha=0.3)

        # --- Holding Period Distribution ---
        ax3 = fig.add_subplot(4, 2, 3)
        ax3.hist(
            trades_sorted['holding_days'],
            bins=range(0, trades_sorted['holding_days'].max() + 2),
            alpha=0.7, edgecolor='black', color='teal',
        )
        ax3.set_xlabel('Holding Days')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Trade Duration Distribution')
        ax3.grid(alpha=0.3)
        print(f"\n--- Holding Period Stats ---")
        print(f"  Mean:   {trades_sorted['holding_days'].mean():.1f} days")
        print(f"  Median: {trades_sorted['holding_days'].median():.0f} days")
        print(f"  Mode:   {trades_sorted['holding_days'].mode().iloc[0]} days")
        print(f"  Max:    {trades_sorted['holding_days'].max()} days")

        # --- PnL by Exit Reason ---
        ax4 = fig.add_subplot(4, 2, 4)
        exit_reasons = sorted(trades_sorted['exit_reason'].unique())
        exit_data = [
            trades_sorted[trades_sorted['exit_reason'] == r]['pnl_pct'].values
            for r in exit_reasons
        ]
        bp = ax4.boxplot(exit_data, labels=[r[:12] for r in exit_reasons], patch_artist=True)
        colors_bp = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12', '#9b59b6']
        for patch, color in zip(bp['boxes'], colors_bp[: len(exit_reasons)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax4.set_ylabel('PnL (%)')
        ax4.set_title('PnL Distribution by Exit Reason')
        ax4.grid(alpha=0.3)
        ax4.axhline(y=0, color='red', linestyle='--', linewidth=0.5)

        # --- Monthly Returns Heatmap ---
        ax5 = fig.add_subplot(4, 1, 3)
        monthly_eq = equity.resample('ME').last()
        monthly_returns = monthly_eq.pct_change().dropna() * 100
        months_df_hm = pd.DataFrame({
            'year': monthly_returns.index.year,
            'month': monthly_returns.index.month,
            'return': monthly_returns.values,
        })
        heatmap_data = months_df_hm.pivot(index='year', columns='month', values='return')
        heatmap_data.columns = [
            'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
        ]

        im = ax5.imshow(heatmap_data.values, cmap='RdYlGn', aspect='auto', vmin=-5, vmax=15)
        ax5.set_yticks(range(len(heatmap_data.index)))
        ax5.set_yticklabels(heatmap_data.index)
        ax5.set_xticks(range(12))
        ax5.set_xticklabels(heatmap_data.columns, fontsize=9)
        ax5.set_title('Monthly Returns Heatmap (%)')
        for i in range(len(heatmap_data.index)):
            for j in range(12):
                val = heatmap_data.values[i, j]
                if not np.isnan(val):
                    color = 'white' if abs(val) > 8 else 'black'
                    ax5.text(j, i, f'{val:.1f}', ha='center', va='center', fontsize=7, color=color)
        plt.colorbar(im, ax=ax5, shrink=0.6, label='Return (%)')

        # --- Trade Clustering ---
        ax6 = fig.add_subplot(4, 1, 4)
        trades_sorted_c = trades_sorted.copy()
        trades_sorted_c['month_period'] = trades_sorted_c['entry_date'].dt.to_period('M')
        monthly_trades = trades_sorted_c.groupby('month_period').size()
        monthly_trades.index = monthly_trades.index.to_timestamp()
        ax6.bar(monthly_trades.index, monthly_trades.values, width=25, alpha=0.7, color='steelblue')
        ax6.set_ylabel('# Trades')
        ax6.set_title('Trade Clustering by Month')
        ax6.grid(alpha=0.3)
        if len(monthly_trades) > 12:
            rolling_avg = monthly_trades.rolling(12).mean()
            ax6.plot(rolling_avg.index, rolling_avg.values, color='red', linewidth=2, label='12-month MA')
            ax6.legend()

        plt.tight_layout()
        if save and self.output_dir:
            path = self.output_dir / '3B3_trade_analytics.png'
            plt.savefig(str(path), dpi=150, bbox_inches='tight')
            print(f"\nSaved: {path}")
        if show:
            plt.show()
        else:
            plt.close(fig)

    # ==================================================================
    # 3B.4 — Turnover & Cost Analysis
    # ==================================================================

    def turnover_analysis(self, print_report: bool = True) -> TurnoverReport:
        """Analyse gross vs net returns, turnover, break-even costs, slippage sensitivity."""
        trades_df = self.trades_df
        equity = self.equity
        config = self.config

        total_comm = self.results.total_commission
        total_gross_pnl = trades_df['pnl'].sum() + total_comm
        total_net_pnl = trades_df['pnl'].sum()

        total_traded_value = (trades_df['shares'] * trades_df['entry_price']).sum() * 2
        avg_portfolio_value = equity.mean()
        years = len(equity) / 252
        annual_turnover = (total_traded_value / avg_portfolio_value) / years

        # Break-even
        avg_gross_pnl_pct = (
            trades_df['pnl_pct'].mean() / 100
            + 2 * (config.commission_pct + config.slippage_pct)
        )
        breakeven_rate = avg_gross_pnl_pct / 2
        current_rate = config.commission_pct + config.slippage_pct
        safety_margin = breakeven_rate - current_rate
        safety_multiple = breakeven_rate / current_rate if current_rate > 0 else float('inf')

        # Slippage sensitivity
        slip_results: List[Dict] = []
        for slip in [0.0005, 0.001, 0.002, 0.005, 0.01]:
            total_cost_rate = config.commission_pct + slip
            adj_total_comm = 2 * total_cost_rate * trades_df['entry_value'].sum()
            adj_net_pnl = total_gross_pnl - adj_total_comm
            adj_total_return = adj_net_pnl / config.initial_capital
            adj_annual_return = (
                (1 + adj_total_return) ** (1 / years) - 1
                if adj_total_return > -1
                else -1
            )
            slip_results.append({
                'slippage_pct': slip,
                'annual_return': adj_annual_return,
                'total_return': adj_total_return,
                'net_pnl': adj_net_pnl,
            })

        report = TurnoverReport(
            gross_pnl=total_gross_pnl,
            total_commission=total_comm,
            net_pnl=total_net_pnl,
            cost_drag_pct=total_comm / total_gross_pnl * 100 if total_gross_pnl > 0 else 0,
            gross_return=total_gross_pnl / config.initial_capital * 100,
            net_return=total_net_pnl / config.initial_capital * 100,
            total_traded_value=total_traded_value,
            avg_portfolio_value=avg_portfolio_value,
            annual_turnover=annual_turnover,
            avg_trades_per_year=len(trades_df) / years,
            commission_gross_ratio=total_comm / total_gross_pnl * 100 if total_gross_pnl > 0 else 0,
            avg_commission_per_trade=total_comm / len(trades_df) if len(trades_df) > 0 else 0,
            breakeven_one_way_cost=breakeven_rate,
            current_one_way_cost=current_rate,
            safety_margin_pct=safety_margin * 100,
            safety_margin_multiple=safety_multiple,
            slippage_sensitivity=slip_results,
        )

        if print_report:
            self._print_turnover(report)

        return report

    def _print_turnover(self, r: TurnoverReport) -> None:
        config = self.config
        print("=" * 80)
        print("3B.4 — TURNOVER & COST ANALYSIS")
        print("=" * 80)

        print(f"\n--- Gross vs Net Returns ---")
        print(f"  Gross P&L (before costs):  ${r.gross_pnl:,.0f}")
        print(f"  Total commission+slippage: ${r.total_commission:,.0f}")
        print(f"  Net P&L (after costs):     ${r.net_pnl:,.0f}")
        print(f"  Cost drag:                 {r.cost_drag_pct:.2f}% of gross P&L")
        print(f"  Gross return:              {r.gross_return:.1f}%")
        print(f"  Net return:                {r.net_return:.1f}%")

        print(f"\n--- Turnover ---")
        print(f"  Total traded value (round-trip): ${r.total_traded_value:,.0f}")
        print(f"  Avg portfolio value:             ${r.avg_portfolio_value:,.0f}")
        print(f"  Annual turnover:                 {r.annual_turnover:.2f}x")
        print(f"  Avg trades per year:             {r.avg_trades_per_year:.0f}")

        print(f"\n--- Cost Efficiency ---")
        print(f"  Commission/Gross ratio:  {r.commission_gross_ratio:.2f}%")
        print(f"  Avg commission per trade: ${r.avg_commission_per_trade:,.0f}")
        print(f"  Commission rate:         {(config.commission_pct + config.slippage_pct) * 100:.2f}% per side")

        print(f"\n--- Break-Even Analysis ---")
        avg_gross_pnl_pct = r.breakeven_one_way_cost * 2
        print(f"  Avg gross P&L per trade:  {avg_gross_pnl_pct * 100:.2f}%")
        print(f"  Break-even one-way cost:  {r.breakeven_one_way_cost * 100:.2f}%")
        print(f"  Current one-way cost:     {r.current_one_way_cost * 100:.2f}%")
        print(f"  Safety margin:            {r.safety_margin_pct:.2f}%  ({r.safety_margin_multiple:.1f}x current)")

        print(f"\n--- Slippage Sensitivity ---")
        print(f"  {'Slippage':>10s} {'Annual Return':>15s} {'Total Return':>15s} {'Net P&L':>15s}")
        print(f"  {'-' * 10} {'-' * 15} {'-' * 15} {'-' * 15}")
        for s in r.slippage_sensitivity:
            print(
                f"  {s['slippage_pct'] * 100:>9.2f}% "
                f"{s['annual_return'] * 100:>14.1f}% "
                f"{s['total_return'] * 100:>14.1f}% "
                f"${s['net_pnl']:>13,.0f}"
            )
        print(f"\n  Current slippage: {config.slippage_pct * 100:.2f}% (first row above)")

    # ==================================================================
    # 3B.5 — Enhanced Regime Analysis
    # ==================================================================

    def regime_analysis(
        self,
        regimes: Optional[List[Tuple[str, str, str]]] = None,
        save: bool = True,
        show: bool = True,
        print_report: bool = True,
    ) -> List[RegimeResult]:
        """
        Per-regime Sharpe, drawdown, VaR, trade stats with visualization.

        Parameters
        ----------
        regimes : list of (name, start_date, end_date), optional
            Custom regime definitions. Defaults to ``DEFAULT_REGIMES``.
        """
        if regimes is None:
            regimes = DEFAULT_REGIMES

        equity = self.equity
        returns = self.returns
        trades_df = self.trades_df

        # Ensure avg_size_pct is available
        if self._avg_size_pct is None:
            avg_pos_value = trades_df['entry_value'].mean()
            avg_equity_at_entry = equity.loc[trades_df['entry_date']].mean()
            self._avg_size_pct = avg_pos_value / avg_equity_at_entry * 100

        if print_report:
            print("=" * 120)
            print("3B.5 — ENHANCED REGIME ANALYSIS")
            print("=" * 120)
            print(
                f"\n{'Regime':<25s} {'Period':>22s}  {'Trades':>6s} {'WR':>6s} "
                f"{'AvgPnL':>7s} {'Return':>8s} {'Sharpe':>7s} {'MaxDD':>7s} "
                f"{'VaR95':>8s} {'Exposure':>8s}"
            )
            print("-" * 120)

        regime_results: List[RegimeResult] = []

        for name, start, end in regimes:
            eq_slice = equity[(equity.index >= start) & (equity.index <= end)]
            ret_slice = returns[(returns.index >= start) & (returns.index <= end)]

            if len(eq_slice) < 5:
                continue

            period_trades = trades_df[
                (trades_df['entry_date'] >= start) & (trades_df['entry_date'] <= end)
            ]

            period_return = (
                (eq_slice.iloc[-1] / eq_slice.iloc[0] - 1) * 100
                if len(eq_slice) > 1
                else 0
            )

            ret_vals_regime = ret_slice.values
            ret_std = ret_vals_regime.std()
            sharpe = (
                (ret_vals_regime.mean() / ret_std) * np.sqrt(252) if ret_std > 0 else 0
            )

            eq_vals = eq_slice.values
            running_max = np.maximum.accumulate(eq_vals)
            dd = (eq_vals - running_max) / running_max
            max_dd = abs(dd.min()) * 100

            var_95_regime = float(np.percentile(ret_vals_regime, 5) * 100)

            n_trades = len(period_trades)
            wr = float((period_trades['pnl_pct'] > 0).mean() * 100) if n_trades > 0 else 0
            avg_pnl = float(period_trades['pnl_pct'].mean()) if n_trades > 0 else 0

            n_days_regime = len(eq_slice)
            pos_days = 0
            for _, t in period_trades.iterrows():
                d1 = max(pd.Timestamp(start), t['entry_date'])
                d2 = min(pd.Timestamp(end), t['exit_date'])
                pos_days += max(0, (d2 - d1).days)
            exposure = (
                (pos_days * self._avg_size_pct / 100) / n_days_regime * 100
                if n_days_regime > 0
                else 0
            )

            rr = RegimeResult(
                name=name, start=start, end=end,
                trades=n_trades, win_rate=wr, avg_pnl=avg_pnl,
                period_return=period_return, sharpe=sharpe,
                max_dd=max_dd, var_95=var_95_regime, exposure=exposure,
            )
            regime_results.append(rr)

            if print_report:
                print(
                    f"  {name:<23s} {start[:10]}→{end[:10]}  "
                    f"{n_trades:>6d} {wr:>5.1f}% {avg_pnl:>+6.2f}% "
                    f"{period_return:>+7.2f}% {sharpe:>6.2f} {max_dd:>6.2f}% "
                    f"{var_95_regime:>+7.4f}% {exposure:>7.2f}%"
                )

        # --- Visualization ---
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        regime_names = [r.name for r in regime_results]
        x = range(len(regime_names))

        # Sharpe by regime
        axes[0, 0].barh(x, [r.sharpe for r in regime_results], color='steelblue', alpha=0.7)
        axes[0, 0].set_yticks(list(x))
        axes[0, 0].set_yticklabels(regime_names, fontsize=8)
        axes[0, 0].axvline(x=0, color='red', linewidth=0.5)
        axes[0, 0].set_xlabel('Sharpe Ratio')
        axes[0, 0].set_title('Sharpe Ratio by Regime')
        axes[0, 0].grid(alpha=0.3)

        # Return by regime
        colors_ret = ['green' if r.period_return > 0 else 'red' for r in regime_results]
        axes[0, 1].barh(x, [r.period_return for r in regime_results], color=colors_ret, alpha=0.7)
        axes[0, 1].set_yticks(list(x))
        axes[0, 1].set_yticklabels(regime_names, fontsize=8)
        axes[0, 1].axvline(x=0, color='red', linewidth=0.5)
        axes[0, 1].set_xlabel('Period Return (%)')
        axes[0, 1].set_title('Return by Regime')
        axes[0, 1].grid(alpha=0.3)

        # Trades + Win Rate
        ax_trades = axes[1, 0]
        ax_trades.barh(x, [r.trades for r in regime_results], color='teal', alpha=0.7)
        ax_trades.set_yticks(list(x))
        ax_trades.set_yticklabels(regime_names, fontsize=8)
        ax_trades.set_xlabel('# Trades')
        ax_trades.set_title('Trade Count by Regime')
        ax_trades.grid(alpha=0.3)
        for i, r in enumerate(regime_results):
            if r.trades > 0:
                ax_trades.text(r.trades + 1, i, f"{r.win_rate:.0f}% WR", va='center', fontsize=8)

        # Max Drawdown
        axes[1, 1].barh(x, [-r.max_dd for r in regime_results], color='red', alpha=0.5)
        axes[1, 1].set_yticks(list(x))
        axes[1, 1].set_yticklabels(regime_names, fontsize=8)
        axes[1, 1].set_xlabel('Max Drawdown (%)')
        axes[1, 1].set_title('Max Drawdown by Regime')
        axes[1, 1].grid(alpha=0.3)

        plt.tight_layout()
        if save and self.output_dir:
            path = self.output_dir / '3B5_regime_analysis.png'
            plt.savefig(str(path), dpi=150, bbox_inches='tight')
            print(f"\nSaved: {path}")
        if show:
            plt.show()
        else:
            plt.close(fig)

        # Key insights
        if print_report and regime_results:
            best = max(regime_results, key=lambda r: r.sharpe)
            worst = min(regime_results, key=lambda r: r.sharpe)
            print(f"\n--- Key Regime Insights ---")
            print(f"  Best Sharpe:  {best.name} ({best.sharpe:.2f})")
            print(f"  Worst Sharpe: {worst.name} ({worst.sharpe:.2f})")
            print(f"  All regimes profitable: {'Yes' if all(r.period_return >= 0 for r in regime_results) else 'No'}")
            crisis = [
                r for r in regime_results
                if any(kw in r.name.lower() for kw in ('crash', 'bear', 'gfc'))
            ]
            if crisis:
                avg_crisis_sharpe = np.mean([r.sharpe for r in crisis])
                print(f"  Avg crisis Sharpe: {avg_crisis_sharpe:.2f} (strategy thrives in vol)")

        return regime_results

    # ==================================================================
    # Run All
    # ==================================================================

    def run_all(self, save: bool = True, show: bool = True) -> Dict:
        """Execute all analytics and return results dict."""
        cap = self.capital_utilization()
        risk = self.risk_metrics()
        self.rolling_analytics(save=save, show=show)
        self.trade_analytics(save=save, show=show)
        turnover = self.turnover_analysis()
        regimes = self.regime_analysis(save=save, show=show)
        return {
            'capital_utilization': cap,
            'risk_metrics': risk,
            'turnover': turnover,
            'regimes': regimes,
        }
