"""
Mean Reversion Strategy Module

Implements adaptive mean reversion signals using statistical techniques:
- Adaptive Z-Score with Ornstein-Uhlenbeck half-life estimation
- Hurst Exponent filtering (only trade mean-reverting stocks)
- Bollinger Bands with volume confirmation
- RSI Divergence detection
- Cross-sectional relative value ranking

All signals are normalized to [-1, +1] for composite scoring.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from scipy import stats
from scipy.optimize import minimize
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.stattools import adfuller
import warnings

warnings.filterwarnings('ignore')


@dataclass
class SignalConfig:
    """Configuration for signal generation"""
    # Adaptive Z-Score
    min_lookback: int = 10
    max_lookback: int = 252
    default_lookback: int = 20

    # Hurst Exponent
    hurst_threshold: float = 0.5  # Only trade if H < 0.5 (mean reverting)
    hurst_lags: int = 20

    # Bollinger Bands
    bb_std: float = 2.0
    bb_lookback: int = 20
    volume_multiplier: float = 1.5  # Require 1.5x average volume

    # RSI
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0

    # Cross-sectional
    cs_lookback: int = 20
    cs_percentiles: Tuple[float, float] = (10, 90)  # Bottom 10% long, top 10% short

    # Regime detection
    vol_lookback: int = 60
    vol_long_lookback: int = 252


class MeanReversionSignals:
    """
    Generate mean reversion signals with adaptive parameters
    """

    def __init__(self, config: Optional[SignalConfig] = None):
        self.config = config or SignalConfig()

    def calculate_ou_half_life(self, prices: pd.Series) -> Optional[float]:
        """
        Estimate half-life of mean reversion using Ornstein-Uhlenbeck process

        dP = theta * (mu - P) * dt + sigma * dW
        where theta = mean reversion speed
        half_life = ln(2) / theta

        Args:
            prices: Price series

        Returns:
            Half-life in days, or None if not stationary
        """
        if len(prices) < 20:
            return None

        # Create lagged series
        prices_lag = prices.shift(1)
        delta_prices = prices - prices_lag

        # Remove NaN
        df = pd.DataFrame({
            'delta': delta_prices,
            'lag': prices_lag
        }).dropna()

        if len(df) < 10:
            return None

        # Regression: delta_P = alpha + theta * P_lag + error
        # theta should be negative for mean reversion
        try:
            X = add_constant(df['lag'])
            model = OLS(df['delta'], X).fit()
            theta = model.params.iloc[1]

            # Check if mean reverting (theta < 0)
            if theta >= 0:
                return None

            # Half-life = ln(2) / |theta|
            half_life = -np.log(2) / theta

            # Sanity check: half-life should be between 1 and max_lookback days
            if half_life < 1 or half_life > self.config.max_lookback:
                return None

            return half_life

        except Exception:
            return None

    def calculate_hurst_exponent(self, prices: pd.Series) -> Optional[float]:
        """
        Calculate Hurst exponent using R/S analysis

        H < 0.5: Mean reverting (good!)
        H = 0.5: Random walk
        H > 0.5: Trending

        Args:
            prices: Price series

        Returns:
            Hurst exponent or None
        """
        if len(prices) < self.config.hurst_lags * 2:
            return None

        lags = range(2, self.config.hurst_lags)
        tau = []

        for lag in lags:
            # Calculate returns
            returns = np.log(prices / prices.shift(lag)).dropna()

            if len(returns) < 2:
                continue

            # Calculate R/S statistic
            mean_return = returns.mean()
            deviations = returns - mean_return
            cum_deviations = deviations.cumsum()

            R = cum_deviations.max() - cum_deviations.min()
            S = returns.std()

            if S > 0:
                tau.append(R / S)
            else:
                tau.append(np.nan)

        # Remove NaN values
        valid_lags = [lag for lag, t in zip(lags, tau) if not np.isnan(t)]
        valid_tau = [t for t in tau if not np.isnan(t)]

        if len(valid_tau) < 3:
            return None

        # Hurst = slope of log(tau) vs log(lag)
        try:
            log_lags = np.log(valid_lags)
            log_tau = np.log(valid_tau)

            X = add_constant(log_lags)
            model = OLS(log_tau, X).fit()
            hurst = model.params[1]

            return hurst

        except Exception:
            return None

    def adaptive_zscore(self, prices: pd.Series, half_life: Optional[float] = None) -> pd.Series:
        """
        Calculate Z-score with adaptive lookback based on half-life

        Args:
            prices: Price series
            half_life: Estimated half-life (if None, use default lookback)

        Returns:
            Z-score series
        """
        if half_life is not None:
            lookback = int(np.clip(half_life * 2, self.config.min_lookback, self.config.max_lookback))
        else:
            lookback = self.config.default_lookback

        # Rolling mean and std
        rolling_mean = prices.rolling(window=lookback, min_periods=lookback//2).mean()
        rolling_std = prices.rolling(window=lookback, min_periods=lookback//2).std()

        # Z-score
        zscore = (prices - rolling_mean) / rolling_std
        zscore = zscore.fillna(0)

        return zscore

    def bollinger_bands_signal(self, prices: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Bollinger Bands with volume confirmation

        Signal = -1 when price breaks below lower band with high volume (oversold)
        Signal = +1 when price breaks above upper band with high volume (overbought)

        Args:
            prices: Price series
            volume: Volume series

        Returns:
            Signal series [-1, +1]
        """
        lookback = self.config.bb_lookback
        std_mult = self.config.bb_std

        # Calculate bands
        rolling_mean = prices.rolling(window=lookback).mean()
        rolling_std = prices.rolling(window=lookback).std()

        upper_band = rolling_mean + (rolling_std * std_mult)
        lower_band = rolling_mean - (rolling_std * std_mult)

        # Volume filter
        avg_volume = volume.rolling(window=lookback).mean()
        high_volume = volume > (avg_volume * self.config.volume_multiplier)

        # Generate signals
        signal = pd.Series(0.0, index=prices.index)

        # Oversold: price below lower band with high volume
        oversold = (prices < lower_band) & high_volume
        signal[oversold] = -1.0

        # Overbought: price above upper band with high volume
        overbought = (prices > upper_band) & high_volume
        signal[overbought] = 1.0

        return signal

    def rsi(self, prices: pd.Series, period: Optional[int] = None) -> pd.Series:
        """
        Calculate RSI (Relative Strength Index)

        Args:
            prices: Price series
            period: RSI period (default from config)

        Returns:
            RSI series [0, 100]
        """
        if period is None:
            period = self.config.rsi_period

        # Calculate price changes
        delta = prices.diff()

        # Separate gains and losses
        gains = delta.where(delta > 0, 0.0)
        losses = -delta.where(delta < 0, 0.0)

        # Calculate average gains and losses
        avg_gains = gains.rolling(window=period, min_periods=period).mean()
        avg_losses = losses.rolling(window=period, min_periods=period).mean()

        # Calculate RS and RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))

        return rsi.fillna(50)

    def rsi_divergence_signal(self, prices: pd.Series) -> pd.Series:
        """
        Detect RSI divergence

        Bullish divergence: Price makes lower low, RSI makes higher low
        Bearish divergence: Price makes higher high, RSI makes lower high

        Args:
            prices: Price series

        Returns:
            Signal series [-1, +1]
        """
        rsi_values = self.rsi(prices)
        signal = pd.Series(0.0, index=prices.index)

        # Find local minima and maxima (simple approach: 5-day window)
        window = 5

        for i in range(window, len(prices) - window):
            # Check for local minimum in prices
            if prices.iloc[i] == prices.iloc[i-window:i+window+1].min():
                # Look for previous local minimum
                for j in range(max(0, i - 30), i - window):
                    if prices.iloc[j] == prices.iloc[max(0, j-window):j+window+1].min():
                        # Bullish divergence: lower price low, higher RSI low
                        if prices.iloc[i] < prices.iloc[j] and rsi_values.iloc[i] > rsi_values.iloc[j]:
                            signal.iloc[i] = -1.0  # Buy signal
                        break

            # Check for local maximum in prices
            if prices.iloc[i] == prices.iloc[i-window:i+window+1].max():
                # Look for previous local maximum
                for j in range(max(0, i - 30), i - window):
                    if prices.iloc[j] == prices.iloc[max(0, j-window):j+window+1].max():
                        # Bearish divergence: higher price high, lower RSI high
                        if prices.iloc[i] > prices.iloc[j] and rsi_values.iloc[i] < rsi_values.iloc[j]:
                            signal.iloc[i] = 1.0  # Sell signal
                        break

        return signal

    def cross_sectional_signal(self, price_data: Dict[str, pd.Series], lookback: Optional[int] = None) -> Dict[str, pd.Series]:
        """
        Cross-sectional relative value ranking

        Ranks stocks by recent performance within the universe.
        Bottom performers get negative signal (long), top performers get positive signal (short).

        Args:
            price_data: Dict of {symbol: price_series}
            lookback: Lookback period for returns calculation

        Returns:
            Dict of {symbol: signal_series}
        """
        if lookback is None:
            lookback = self.config.cs_lookback

        # Calculate returns for all stocks
        returns_data = {}
        for symbol, prices in price_data.items():
            returns = prices.pct_change(lookback)
            returns_data[symbol] = returns

        # Create DataFrame for cross-sectional ranking
        returns_df = pd.DataFrame(returns_data)

        # Rank stocks at each time point
        ranks = returns_df.rank(axis=1, pct=True)  # Percentile ranks [0, 1]

        # Generate signals
        signals = {}
        low_pct, high_pct = self.config.cs_percentiles

        for symbol in price_data.keys():
            signal = pd.Series(0.0, index=price_data[symbol].index)

            # Bottom percentile: long signal
            signal[ranks[symbol] < low_pct / 100] = -1.0

            # Top percentile: short signal
            signal[ranks[symbol] > high_pct / 100] = 1.0

            signals[symbol] = signal

        return signals

    def volatility_regime(self, prices: pd.Series) -> pd.Series:
        """
        Classify volatility regime

        Returns multiplier for position sizing:
        - Low vol regime: 1.0 (normal size)
        - High vol regime: 0.5 (reduce size)
        - Crisis regime: 0.0 (sit out)

        Args:
            prices: Price series

        Returns:
            Regime multiplier series [0.0, 1.0]
        """
        # Calculate realized volatility
        returns = prices.pct_change()

        short_vol = returns.rolling(window=self.config.vol_lookback).std() * np.sqrt(252)
        long_vol = returns.rolling(window=self.config.vol_long_lookback).std() * np.sqrt(252)

        # Volatility ratio
        vol_ratio = short_vol / long_vol

        # Regime classification
        regime = pd.Series(1.0, index=prices.index)  # Default: normal
        regime[vol_ratio > 1.5] = 0.5  # High vol: reduce
        regime[vol_ratio > 2.0] = 0.0  # Crisis: sit out

        return regime.fillna(1.0)

    def generate_composite_signal(
        self,
        prices: pd.Series,
        volume: pd.Series,
        weights: Optional[Dict[str, float]] = None
    ) -> Tuple[pd.Series, Dict[str, pd.Series]]:
        """
        Generate composite signal from multiple indicators

        The raw z-score is the primary signal (naturally scaled for thresholds).
        Other indicators act as confirmation multipliers that boost or dampen
        the z-score. A z-score of 2.0 = 2 standard deviations from the mean.

        Confirmation logic:
        - Each confirming signal adds a boost (e.g., +0.25 per confirming indicator)
        - Each contradicting signal dampens (e.g., -0.25)
        - Net effect: z-score can be amplified ~1.75x or dampened ~0.25x

        Args:
            prices: Price series
            volume: Volume series
            weights: Dict of confirmation weights (default provided)

        Returns:
            (composite_signal, individual_signals)
        """
        # Confirmation weights (how much each confirming indicator boosts the z-score)
        if weights is None:
            weights = {
                'bollinger': 0.25,
                'rsi_divergence': 0.25,
                'rsi_level': 0.25
            }

        individual_signals = {}

        # 1. Adaptive Z-Score (PRIMARY signal - raw, unscaled)
        half_life = self.calculate_ou_half_life(prices)
        zscore = self.adaptive_zscore(prices, half_life)
        individual_signals['zscore'] = zscore

        # 2. Bollinger Bands (confirmation: -1, 0, or +1)
        bb_signal = self.bollinger_bands_signal(prices, volume)
        individual_signals['bollinger'] = bb_signal

        # 3. RSI Divergence (confirmation: -1, 0, or +1)
        rsi_div_signal = self.rsi_divergence_signal(prices)
        individual_signals['rsi_divergence'] = rsi_div_signal

        # 4. RSI Level (confirmation: -1, 0, or +1)
        rsi_values = self.rsi(prices)
        rsi_signal = pd.Series(0.0, index=prices.index)
        rsi_signal[rsi_values < self.config.rsi_oversold] = -1.0
        rsi_signal[rsi_values > self.config.rsi_overbought] = 1.0
        individual_signals['rsi_level'] = rsi_signal

        # Composite: z-score boosted by confirmation signals
        # Confirmation is symmetric: agreeing indicators boost magnitude in BOTH directions
        confirmation = pd.Series(0.0, index=prices.index)
        for name, weight in weights.items():
            signal = individual_signals[name]
            # Check agreement: signal and z-score have the same sign
            # agreement = +1 when they agree, -1 when they disagree, 0 when signal is 0
            agreement = np.sign(zscore) * signal
            confirmation += weight * agreement

        # Apply confirmation symmetrically: boost MAGNITUDE of z-score
        # confirmation > 0 means indicators agree with z-score direction
        # confirmation < 0 means indicators disagree
        composite = zscore * (1.0 + confirmation)

        return composite, individual_signals

    def is_mean_reverting(self, prices: pd.Series) -> Tuple[bool, Dict[str, float]]:
        """
        Check if a stock is mean reverting using multiple tests

        Args:
            prices: Price series

        Returns:
            (is_mean_reverting, metrics_dict)
        """
        metrics = {}

        # 1. Hurst Exponent
        hurst = self.calculate_hurst_exponent(prices)
        metrics['hurst'] = hurst

        # 2. Half-life
        half_life = self.calculate_ou_half_life(prices)
        metrics['half_life'] = half_life

        # 3. ADF test (Augmented Dickey-Fuller)
        try:
            adf_result = adfuller(prices.dropna(), maxlag=20)
            adf_pvalue = adf_result[1]
            metrics['adf_pvalue'] = adf_pvalue
        except Exception:
            metrics['adf_pvalue'] = None

        # Decision logic
        is_reverting = False

        # Hurst < 0.5 is strong evidence
        if hurst is not None and hurst < self.config.hurst_threshold:
            is_reverting = True

        # Half-life exists and is reasonable
        if half_life is not None and self.config.min_lookback < half_life < self.config.max_lookback:
            is_reverting = True

        # ADF test rejects null (stationary)
        if metrics['adf_pvalue'] is not None and metrics['adf_pvalue'] < 0.05:
            is_reverting = True

        return is_reverting, metrics


class UniverseAnalyzer:
    """
    Analyze entire universe to identify mean-reverting stocks
    """

    def __init__(self, config: Optional[SignalConfig] = None):
        self.config = config or SignalConfig()
        self.signal_gen = MeanReversionSignals(config)

    def analyze_universe(
        self,
        price_data: Dict[str, pd.Series],
        min_history: int = 100
    ) -> pd.DataFrame:
        """
        Analyze all stocks in universe for mean reversion characteristics

        Args:
            price_data: Dict of {symbol: price_series}
            min_history: Minimum data points required

        Returns:
            DataFrame with analysis results
        """
        results = []

        for symbol, prices in price_data.items():
            if len(prices.dropna()) < min_history:
                continue

            is_reverting, metrics = self.signal_gen.is_mean_reverting(prices)

            results.append({
                'symbol': symbol,
                'is_mean_reverting': is_reverting,
                'hurst': metrics.get('hurst'),
                'half_life': metrics.get('half_life'),
                'adf_pvalue': metrics.get('adf_pvalue'),
                'data_points': len(prices.dropna())
            })

        df = pd.DataFrame(results)

        # Sort by Hurst exponent (lower is more mean reverting)
        if not df.empty:
            df = df.sort_values('hurst', ascending=True)

        return df
