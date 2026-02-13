"""
Parameter Optimization Module

Implements walk-forward analysis and parameter optimization for mean reversion strategies.
Supports grid search and Bayesian optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field
from itertools import product
import warnings

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from .engine import BacktestEngine, BacktestConfig, BacktestResults

warnings.filterwarnings('ignore')


@dataclass
class OptimizationConfig:
    """Configuration for parameter optimization"""
    # Walk-forward settings
    train_period_days: int = 252  # 1 year training
    test_period_days: int = 126   # 6 months testing
    step_days: int = 63           # Step forward by 3 months

    # Optimization method
    method: str = 'grid'  # 'grid' or 'bayesian'
    n_trials: int = 100   # For Bayesian optimization

    # Objective metric
    objective_metric: str = 'sharpe_ratio'  # 'sharpe_ratio', 'total_return', 'calmar_ratio'

    # Grid search ranges
    entry_threshold_range: List[float] = field(default_factory=lambda: [1.5, 2.0, 2.5, 3.0])
    exit_threshold_range: List[float] = field(default_factory=lambda: [0.3, 0.5, 0.7])
    stop_loss_range: List[Optional[float]] = field(default_factory=lambda: [None, 0.05, 0.10])
    take_profit_range: List[Optional[float]] = field(default_factory=lambda: [None, 0.10, 0.15])
    max_holding_range: List[Optional[int]] = field(default_factory=lambda: [None, 10, 20])

    # Signal weight ranges (for Bayesian optimization)
    weight_ranges: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'zscore': (0.0, 1.0),
        'bollinger': (0.0, 1.0),
        'rsi_divergence': (0.0, 1.0),
        'rsi_level': (0.0, 1.0)
    })

    # Minimum trades required for valid result
    min_trades: int = 10


@dataclass
class WalkForwardResult:
    """Single walk-forward period result"""
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    best_params: Dict[str, Any]
    train_metric: float
    test_metric: float
    test_results: BacktestResults


@dataclass
class OptimizationResults:
    """Overall optimization results"""
    walk_forward_results: List[WalkForwardResult]
    combined_test_results: BacktestResults
    best_params_frequency: Dict[str, int] = field(default_factory=dict)
    avg_train_metric: float = 0.0
    avg_test_metric: float = 0.0
    stability_score: float = 0.0  # How stable are params across periods

    def summary(self) -> Dict:
        """Return summary dict"""
        return {
            'Num Periods': len(self.walk_forward_results),
            'Avg Train Metric': f"{self.avg_train_metric:.3f}",
            'Avg Test Metric': f"{self.avg_test_metric:.3f}",
            'Stability Score': f"{self.stability_score:.3f}",
            'Combined Test Return': f"{self.combined_test_results.total_return*100:.2f}%",
            'Combined Test Sharpe': f"{self.combined_test_results.sharpe_ratio:.2f}",
            'Combined Test Max DD': f"{self.combined_test_results.max_drawdown*100:.2f}%"
        }


class ParameterOptimizer:
    """
    Parameter optimizer with walk-forward analysis
    """

    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()

    def walk_forward_optimization(
        self,
        price_data: pd.DataFrame,
        signal_generator: Callable[[Dict], pd.DataFrame],
        volume_data: Optional[pd.DataFrame] = None,
        regime_data: Optional[pd.DataFrame] = None
    ) -> OptimizationResults:
        """
        Run walk-forward optimization

        Args:
            price_data: Price DataFrame
            signal_generator: Function that takes params and returns signal DataFrame
            volume_data: Optional volume data
            regime_data: Optional regime data

        Returns:
            OptimizationResults
        """
        dates = price_data.index
        wf_results = []

        # Generate walk-forward periods
        periods = self._generate_wf_periods(dates)

        print(f"Running walk-forward optimization: {len(periods)} periods")

        for i, (train_start, train_end, test_start, test_end) in enumerate(periods):
            print(f"\nPeriod {i+1}/{len(periods)}")
            train_start_str = train_start.date() if hasattr(train_start, 'date') else train_start
            train_end_str = train_end.date() if hasattr(train_end, 'date') else train_end
            test_start_str = test_start.date() if hasattr(test_start, 'date') else test_start
            test_end_str = test_end.date() if hasattr(test_end, 'date') else test_end
            print(f"  Train: {train_start_str} to {train_end_str}")
            print(f"  Test:  {test_start_str} to {test_end_str}")

            # Split data
            train_prices = price_data.loc[train_start:train_end]
            test_prices = price_data.loc[test_start:test_end]

            train_volume = volume_data.loc[train_start:train_end] if volume_data is not None else None
            test_volume = volume_data.loc[test_start:test_end] if volume_data is not None else None

            train_regime = regime_data.loc[train_start:train_end] if regime_data is not None else None
            test_regime = regime_data.loc[test_start:test_end] if regime_data is not None else None

            # Optimize on training period
            if self.config.method == 'grid':
                best_params, train_metric = self._grid_search(
                    train_prices, signal_generator, train_volume, train_regime
                )
            elif self.config.method == 'bayesian':
                if not OPTUNA_AVAILABLE:
                    raise ImportError("Optuna not installed. Use method='grid' or install optuna.")
                best_params, train_metric = self._bayesian_optimization(
                    train_prices, signal_generator, train_volume, train_regime
                )
            else:
                raise ValueError(f"Unknown optimization method: {self.config.method}")

            print(f"  Best params: {best_params}")
            print(f"  Train metric: {train_metric:.3f}")

            # Test on out-of-sample period
            test_results = self._backtest_with_params(
                test_prices, signal_generator, best_params, test_volume, test_regime
            )

            test_metric = self._get_metric(test_results, self.config.objective_metric)
            print(f"  Test metric: {test_metric:.3f}")

            # Store result
            wf_result = WalkForwardResult(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                best_params=best_params,
                train_metric=train_metric,
                test_metric=test_metric,
                test_results=test_results
            )
            wf_results.append(wf_result)

        # Combine test results
        combined_test_results = self._combine_test_results(wf_results)

        # Calculate stability metrics
        avg_train_metric = np.mean([r.train_metric for r in wf_results])
        avg_test_metric = np.mean([r.test_metric for r in wf_results])

        # Parameter stability: how often does each param value appear
        param_frequency = self._calculate_param_frequency(wf_results)

        # Stability score: coefficient of variation of test metrics (lower is more stable)
        test_metrics = [r.test_metric for r in wf_results]
        stability_score = 1.0 / (1.0 + np.std(test_metrics) / (np.abs(np.mean(test_metrics)) + 1e-6))

        results = OptimizationResults(
            walk_forward_results=wf_results,
            combined_test_results=combined_test_results,
            best_params_frequency=param_frequency,
            avg_train_metric=avg_train_metric,
            avg_test_metric=avg_test_metric,
            stability_score=stability_score
        )

        return results

    def _generate_wf_periods(self, dates: pd.DatetimeIndex) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """Generate walk-forward train/test periods"""
        periods = []

        train_days = self.config.train_period_days
        test_days = self.config.test_period_days
        step_days = self.config.step_days

        current_idx = 0
        while current_idx + train_days + test_days <= len(dates):
            train_start = dates[current_idx]
            train_end = dates[current_idx + train_days - 1]
            test_start = dates[current_idx + train_days]
            test_end_idx = min(current_idx + train_days + test_days - 1, len(dates) - 1)
            test_end = dates[test_end_idx]

            periods.append((train_start, train_end, test_start, test_end))

            current_idx += step_days

        return periods

    def _grid_search(
        self,
        price_data: pd.DataFrame,
        signal_generator: Callable,
        volume_data: Optional[pd.DataFrame],
        regime_data: Optional[pd.DataFrame]
    ) -> Tuple[Dict, float]:
        """Grid search optimization"""
        # Generate parameter grid
        param_grid = list(product(
            self.config.entry_threshold_range,
            self.config.exit_threshold_range,
            self.config.stop_loss_range,
            self.config.take_profit_range,
            self.config.max_holding_range
        ))

        best_metric = -np.inf
        best_params = None

        for entry_th, exit_th, stop_loss, take_profit, max_holding in param_grid:
            params = {
                'entry_threshold': entry_th,
                'exit_threshold': exit_th,
                'stop_loss_pct': stop_loss,
                'take_profit_pct': take_profit,
                'max_holding_days': max_holding
            }

            # Run backtest
            results = self._backtest_with_params(
                price_data, signal_generator, params, volume_data, regime_data
            )

            # Check minimum trades
            if results.total_trades < self.config.min_trades:
                continue

            # Get metric
            metric = self._get_metric(results, self.config.objective_metric)

            if metric > best_metric:
                best_metric = metric
                best_params = params

        if best_params is None:
            # Fallback to default params
            best_params = {
                'entry_threshold': 2.0,
                'exit_threshold': 0.5,
                'stop_loss_pct': None,
                'take_profit_pct': None,
                'max_holding_days': None
            }
            best_metric = 0.0

        return best_params, best_metric

    def _bayesian_optimization(
        self,
        price_data: pd.DataFrame,
        signal_generator: Callable,
        volume_data: Optional[pd.DataFrame],
        regime_data: Optional[pd.DataFrame]
    ) -> Tuple[Dict, float]:
        """Bayesian optimization using Optuna"""

        def objective(trial):
            # Sample parameters
            params = {
                'entry_threshold': trial.suggest_float('entry_threshold', 1.0, 4.0),
                'exit_threshold': trial.suggest_float('exit_threshold', 0.1, 1.5),
                'stop_loss_pct': trial.suggest_categorical('stop_loss_pct', [None, 0.05, 0.10, 0.15]),
                'take_profit_pct': trial.suggest_categorical('take_profit_pct', [None, 0.10, 0.15, 0.20]),
                'max_holding_days': trial.suggest_categorical('max_holding_days', [None, 10, 20, 30])
            }

            # Run backtest
            results = self._backtest_with_params(
                price_data, signal_generator, params, volume_data, regime_data
            )

            # Check minimum trades
            if results.total_trades < self.config.min_trades:
                return -np.inf

            # Return objective metric
            return self._get_metric(results, self.config.objective_metric)

        # Create study
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())

        # Optimize
        study.optimize(objective, n_trials=self.config.n_trials, show_progress_bar=False)

        # Get best params
        best_params = study.best_params
        best_metric = study.best_value

        return best_params, best_metric

    def _backtest_with_params(
        self,
        price_data: pd.DataFrame,
        signal_generator: Callable,
        params: Dict,
        volume_data: Optional[pd.DataFrame],
        regime_data: Optional[pd.DataFrame]
    ) -> BacktestResults:
        """Run backtest with specific parameters"""
        # Generate signals (signal generator should not depend on backtest params)
        signal_data = signal_generator(params)

        # Create backtest config
        bt_config = BacktestConfig(
            entry_threshold=params.get('entry_threshold', 2.0),
            exit_threshold=params.get('exit_threshold', 0.5),
            stop_loss_pct=params.get('stop_loss_pct'),
            take_profit_pct=params.get('take_profit_pct'),
            max_holding_days=params.get('max_holding_days')
        )

        # Run backtest
        engine = BacktestEngine(bt_config)
        results = engine.run_backtest(price_data, signal_data, volume_data, regime_data)

        return results

    def _get_metric(self, results: BacktestResults, metric_name: str) -> float:
        """Extract metric from backtest results"""
        if metric_name == 'sharpe_ratio':
            return results.sharpe_ratio
        elif metric_name == 'total_return':
            return results.total_return
        elif metric_name == 'calmar_ratio':
            return results.calmar_ratio
        elif metric_name == 'sortino_ratio':
            return results.sortino_ratio
        else:
            raise ValueError(f"Unknown metric: {metric_name}")

    def _combine_test_results(self, wf_results: List[WalkForwardResult]) -> BacktestResults:
        """Combine test results from all walk-forward periods"""
        # Concatenate equity curves
        equity_curves = []
        returns_list = []
        all_trades = []

        for wf_result in wf_results:
            equity_curves.append(wf_result.test_results.equity_curve)
            returns_list.append(wf_result.test_results.returns)
            all_trades.extend(wf_result.test_results.trades)

        # Combine equity curves (renormalize to start at initial capital)
        combined_equity = pd.Series(dtype=float)
        current_capital = 100000.0

        for i, equity in enumerate(equity_curves):
            if i > 0:
                # Scale to continue from previous equity level
                scaling_factor = current_capital / equity.iloc[0]
                equity = equity * scaling_factor

            combined_equity = pd.concat([combined_equity, equity])
            current_capital = equity.iloc[-1]

        # Combine returns
        combined_returns = pd.concat(returns_list)

        # Create combined results
        combined_results = BacktestResults(
            equity_curve=combined_equity,
            returns=combined_returns,
            trades=all_trades
        )

        # Calculate metrics
        engine = BacktestEngine()
        engine._calculate_metrics(combined_results)

        return combined_results

    def _calculate_param_frequency(self, wf_results: List[WalkForwardResult]) -> Dict[str, int]:
        """Calculate how often each parameter value appears"""
        frequency = {}

        for wf_result in wf_results:
            for param_name, param_value in wf_result.best_params.items():
                key = f"{param_name}={param_value}"
                frequency[key] = frequency.get(key, 0) + 1

        return frequency


def plot_wf_results(opt_results: OptimizationResults):
    """
    Plot walk-forward optimization results

    Args:
        opt_results: OptimizationResults object
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # 1. Train vs Test metrics
    periods = range(len(opt_results.walk_forward_results))
    train_metrics = [r.train_metric for r in opt_results.walk_forward_results]
    test_metrics = [r.test_metric for r in opt_results.walk_forward_results]

    axes[0].plot(periods, train_metrics, marker='o', label='Train', linewidth=2)
    axes[0].plot(periods, test_metrics, marker='s', label='Test', linewidth=2)
    axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('Period')
    axes[0].set_ylabel('Metric Value')
    axes[0].set_title('Train vs Test Performance')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # 2. Equity curve
    axes[1].plot(opt_results.combined_test_results.equity_curve, linewidth=2)
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Equity ($)')
    axes[1].set_title('Combined Test Equity Curve')
    axes[1].grid(alpha=0.3)

    # 3. Drawdown
    from .engine import calculate_underwater_curve
    drawdown = calculate_underwater_curve(opt_results.combined_test_results.equity_curve)
    axes[2].fill_between(drawdown.index, 0, drawdown * 100, alpha=0.5, color='red')
    axes[2].set_xlabel('Date')
    axes[2].set_ylabel('Drawdown (%)')
    axes[2].set_title('Underwater Curve')
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    return fig
