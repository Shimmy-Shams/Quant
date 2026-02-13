"""
Configuration Management Module

Loads and validates configuration from YAML file.
Provides easy access to all strategy parameters.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field


class ConfigLoader:
    """
    Configuration loader with validation and easy access
    """

    def __init__(self, config_path: Optional[Path] = None):
        """
        Load configuration from YAML file

        Args:
            config_path: Path to config.yaml (default: project_root/config.yaml)
        """
        if config_path is None:
            # Default: config.yaml in project root
            current_file = Path(__file__)
            project_root = current_file.parent.parent
            config_path = project_root / 'config.yaml'

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self._validate()

    def _validate(self):
        """Validate configuration"""
        required_sections = ['data', 'signals', 'backtest', 'optimization', 'universe']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required config section: {section}")

    def get(self, path: str, default: Any = None) -> Any:
        """
        Get config value by dot-separated path

        Args:
            path: Dot-separated path (e.g., 'signals.zscore.min_lookback')
            default: Default value if not found

        Returns:
            Config value
        """
        keys = path.split('.')
        value = self.config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    # Convenience methods for common config sections

    @property
    def data(self) -> Dict:
        """Data loading configuration"""
        return self.config['data']

    @property
    def signals(self) -> Dict:
        """Signal generation configuration"""
        return self.config['signals']

    @property
    def backtest(self) -> Dict:
        """Backtest configuration"""
        return self.config['backtest']

    @property
    def optimization(self) -> Dict:
        """Optimization configuration"""
        return self.config['optimization']

    @property
    def universe(self) -> Dict:
        """Universe filtering configuration"""
        return self.config['universe']

    @property
    def visualization(self) -> Dict:
        """Visualization configuration"""
        return self.config.get('visualization', {})

    @property
    def logging(self) -> Dict:
        """Logging configuration"""
        return self.config.get('logging', {})

    def to_signal_config(self):
        """
        Convert to SignalConfig dataclass

        Returns:
            SignalConfig instance
        """
        from strategies.mean_reversion import SignalConfig

        return SignalConfig(
            # Z-score
            min_lookback=self.get('signals.zscore.min_lookback', 10),
            max_lookback=self.get('signals.zscore.max_lookback', 252),
            default_lookback=self.get('signals.zscore.default_lookback', 20),

            # Hurst
            hurst_threshold=self.get('signals.hurst.threshold', 0.5),
            hurst_lags=self.get('signals.hurst.lags', 20),

            # Bollinger
            bb_std=self.get('signals.bollinger.std_multiplier', 2.0),
            bb_lookback=self.get('signals.bollinger.lookback', 20),
            volume_multiplier=self.get('signals.bollinger.volume_multiplier', 1.5),

            # RSI
            rsi_period=self.get('signals.rsi.period', 14),
            rsi_overbought=self.get('signals.rsi.overbought', 70.0),
            rsi_oversold=self.get('signals.rsi.oversold', 30.0),

            # Cross-sectional
            cs_lookback=self.get('signals.cross_sectional.lookback', 20),
            cs_percentiles=tuple(self.get('signals.cross_sectional.percentiles', [10, 90])),

            # Regime
            vol_lookback=self.get('signals.volatility_regime.short_lookback', 60),
            vol_long_lookback=self.get('signals.volatility_regime.long_lookback', 252)
        )

    def to_backtest_config(self):
        """
        Convert to BacktestConfig dataclass

        Returns:
            BacktestConfig instance
        """
        from backtest.engine import BacktestConfig

        return BacktestConfig(
            initial_capital=self.get('backtest.initial_capital', 100000.0),
            commission_pct=self.get('backtest.commission_pct', 0.001),
            slippage_pct=self.get('backtest.slippage_pct', 0.0005),
            position_size_method=self.get('backtest.position_size_method', 'equal_weight'),
            max_position_size=self.get('backtest.max_position_size', 0.1),
            max_total_exposure=self.get('backtest.max_total_exposure', 1.0),
            entry_threshold=self.get('backtest.entry_threshold', 2.0),
            exit_threshold=self.get('backtest.exit_threshold', 0.5),
            stop_loss_pct=self.get('backtest.stop_loss_pct'),
            take_profit_pct=self.get('backtest.take_profit_pct'),
            max_holding_days=self.get('backtest.max_holding_days'),
            use_regime_filter=self.get('backtest.use_regime_filter', True),
            min_regime_multiplier=self.get('backtest.min_regime_multiplier', 0.5)
        )

    def to_optimization_config(self):
        """
        Convert to OptimizationConfig dataclass

        Returns:
            OptimizationConfig instance
        """
        from backtest.optimizer import OptimizationConfig

        return OptimizationConfig(
            train_period_days=self.get('optimization.train_period_days', 252),
            test_period_days=self.get('optimization.test_period_days', 126),
            step_days=self.get('optimization.step_days', 63),
            method=self.get('optimization.method', 'grid'),
            n_trials=self.get('optimization.n_trials', 100),
            objective_metric=self.get('optimization.objective_metric', 'sharpe_ratio'),
            entry_threshold_range=self.get('optimization.param_ranges.entry_threshold', [1.5, 2.0, 2.5, 3.0]),
            exit_threshold_range=self.get('optimization.param_ranges.exit_threshold', [0.3, 0.5, 0.7]),
            stop_loss_range=self.get('optimization.param_ranges.stop_loss_pct', [None, 0.05, 0.10]),
            take_profit_range=self.get('optimization.param_ranges.take_profit_pct', [None, 0.10, 0.15]),
            max_holding_range=self.get('optimization.param_ranges.max_holding_days', [None, 10, 20]),
            min_trades=self.get('optimization.min_trades', 10)
        )

    def get_composite_weights(self) -> Dict[str, float]:
        """
        Get composite signal weights

        Returns:
            Dict of signal weights
        """
        return self.get('signals.composite_weights', {
            'bollinger': 0.25,
            'rsi_divergence': 0.25,
            'rsi_level': 0.25
        })


# Global config instance (lazy loaded)
_global_config = None


def get_config(config_path: Optional[Path] = None) -> ConfigLoader:
    """
    Get global config instance

    Args:
        config_path: Path to config file (only used on first call)

    Returns:
        ConfigLoader instance
    """
    global _global_config

    if _global_config is None:
        _global_config = ConfigLoader(config_path)

    return _global_config


def reload_config(config_path: Optional[Path] = None):
    """
    Reload configuration from file

    Args:
        config_path: Path to config file
    """
    global _global_config
    _global_config = ConfigLoader(config_path)
