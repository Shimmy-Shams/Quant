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
    use_log_prices: bool = True  # Use log prices for z-score (more stationary)

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

    # Kalman Filter
    use_kalman: bool = True
    kalman_transition_cov: float = 1e-5      # Q: how fast the true mean drifts
    kalman_observation_cov: float = 2e-4     # R: daily log-return variance (~1.5% vol)
    kalman_initial_cov: float = 1e-3         # P0: initial uncertainty

    # OU Prediction
    use_predicted_return: bool = True
    ou_hurdle_rate: float = 0.005  # 0.5% minimum expected return
    ou_prediction_horizon: Optional[int] = None  # None = use half-life

    # Signal composition mode (Phase B.1)
    signal_mode: str = 'gated'           # 'confirmation' (legacy) or 'gated'
    gate_signal: str = 'rsi_divergence'  # Signal that controls entries in gated mode
    zscore_boost_factor: float = 0.5     # |zscore| conviction boost in gated mode

    # Dynamic short confidence filter (Phase B.2)
    use_dynamic_short_filter: bool = True
    short_trend_lookback: int = 50        # MA period for trend assessment
    short_momentum_fast: int = 5          # Fast momentum window (days)
    short_momentum_slow: int = 20         # Slow momentum window (days)
    short_min_confidence: float = 0.3     # Min confidence to allow shorts (0-1)


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
        Calculate Z-score with adaptive lookback based on half-life.
        Uses log prices when configured (more stationary for financial data).

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

        # Use log prices for stationarity if configured
        if self.config.use_log_prices:
            series = np.log(prices.clip(lower=1e-8))
        else:
            series = prices

        # Rolling mean and std
        rolling_mean = series.rolling(window=lookback, min_periods=lookback//2).mean()
        rolling_std = series.rolling(window=lookback, min_periods=lookback//2).std()

        # Z-score
        zscore = (series - rolling_mean) / rolling_std
        zscore = zscore.fillna(0)

        return zscore

    def kalman_zscore(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Kalman filter-based z-score estimation.

        Uses a simple state-space model where the hidden state is the
        'true mean' of the price process. The Kalman filter dynamically
        estimates this mean and the uncertainty around it, producing a
        z-score that adapts faster to regime changes than rolling windows.

        State model:
            mu_t = mu_{t-1} + w_t,   w_t ~ N(0, Q)   (mean drifts slowly)
            p_t  = mu_t + v_t,       v_t ~ N(0, R)   (price = mean + noise)

        Args:
            prices: Price series

        Returns:
            (zscore, estimated_mean, estimated_std) tuple of Series
        """
        # Work in log-price space for stationarity
        if self.config.use_log_prices:
            obs = np.log(prices.clip(lower=1e-8)).values
        else:
            obs = prices.values

        n = len(obs)
        Q = self.config.kalman_transition_cov   # Process noise
        R = self.config.kalman_observation_cov   # Observation noise

        # Initialize state
        mu = np.zeros(n)       # Estimated mean
        P = np.zeros(n)        # Estimated covariance (uncertainty)
        mu[0] = obs[0]
        P[0] = self.config.kalman_initial_cov

        # Kalman filter: predict-update cycle
        for t in range(1, n):
            # Predict
            mu_pred = mu[t-1]
            P_pred = P[t-1] + Q

            # Update (Kalman gain)
            K = P_pred / (P_pred + R)
            mu[t] = mu_pred + K * (obs[t] - mu_pred)
            P[t] = (1 - K) * P_pred

        # Z-score = (observation - estimated_mean) / sqrt(estimation_uncertainty + obs_noise)
        estimated_std = np.sqrt(P + R)
        zscore_values = (obs - mu) / np.where(estimated_std > 1e-10, estimated_std, 1e-10)

        zscore = pd.Series(zscore_values, index=prices.index, name='kalman_zscore')
        est_mean = pd.Series(mu, index=prices.index, name='kalman_mean')
        est_std = pd.Series(estimated_std, index=prices.index, name='kalman_std')

        return zscore, est_mean, est_std

    def ou_predicted_return(
        self,
        prices: pd.Series,
        half_life: Optional[float] = None
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Compute predicted return and expected time to reversion using
        the Ornstein-Uhlenbeck model.

        For the OU process: dP = theta*(mu - P)*dt + sigma*dW
        Expected return over horizon h: E[P_{t+h} - P_t] = (mu - P_t) * (1 - e^{-theta*h})
        Expected fractional return:     E[r] = ((mu - P_t) / P_t) * (1 - e^{-theta*h})

        Args:
            prices: Price series
            half_life: Estimated OU half-life (if None, estimates it)

        Returns:
            (expected_return_series, time_to_reversion_series)
        """
        if half_life is None:
            half_life = self.calculate_ou_half_life(prices)

        if half_life is None or half_life <= 0:
            # Cannot estimate OU parameters; return zeros
            zeros = pd.Series(0.0, index=prices.index)
            return zeros, zeros

        theta = np.log(2) / half_life  # Mean-reversion speed

        # Determine prediction horizon
        horizon = self.config.ou_prediction_horizon
        if horizon is None:
            horizon = max(1, int(half_life))

        # Use log prices for the OU model
        if self.config.use_log_prices:
            series = np.log(prices.clip(lower=1e-8))
        else:
            series = prices

        # Rolling estimate of the mean level (mu)
        lookback = int(np.clip(half_life * 2, self.config.min_lookback, self.config.max_lookback))
        mu = series.rolling(window=lookback, min_periods=lookback//2).mean()

        # Expected return: (mu - current) * (1 - exp(-theta * h))
        reversion_factor = 1.0 - np.exp(-theta * horizon)
        expected_return = (mu - series) * reversion_factor

        # Time to reversion (from half-life): how many days to cover X% of gap
        # For 90% reversion: t_90 = -ln(0.1) / theta = ln(10) / theta
        time_to_reversion = pd.Series(half_life, index=prices.index)

        return expected_return, time_to_reversion

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

    def _compute_short_confidence(self, prices: pd.Series) -> pd.Series:
        """
        Compute dynamic confidence score for short entries (0 to 1).

        Adapts short entry aggressiveness to market conditions using
        statistical measures rather than fixed multipliers. This eliminates
        the need for manual threshold adjustments across market regimes.

        Components:
        1. Trend Extension: How far price is from MA (overextended = higher confidence)
        2. Momentum Deceleration: Is the move losing steam? (decelerating = higher confidence)
        3. Volatility Regime: Elevated vol = stronger mean reversion signal

        Args:
            prices: Price series for a single stock

        Returns:
            Confidence series (0 to 1), per day
        """
        trend_lb = self.config.short_trend_lookback
        mom_fast = self.config.short_momentum_fast
        mom_slow = self.config.short_momentum_slow

        # --- 1. Trend Extension Score ---
        # For mean-reverting stocks: further above MA = more overextended
        # = more likely to revert = HIGHER confidence for shorting
        ma = prices.rolling(window=trend_lb, min_periods=trend_lb // 2).mean()
        extension = (prices - ma) / ma  # positive when above MA

        # Map extension to score: [-0.15, +0.15] -> [-1, +1]
        extension_score = extension.clip(-0.15, 0.15) / 0.15

        # --- 2. Momentum Deceleration Score ---
        # Compare fast vs slow momentum
        # If fast momentum < slow momentum -> move is decelerating -> shorts are safer
        roc_fast = prices.pct_change(mom_fast)
        roc_slow = prices.pct_change(mom_slow)

        # Positive deceleration = fast momentum weakening relative to slow
        deceleration = roc_slow - roc_fast
        decel_score = deceleration.clip(-0.05, 0.05) / 0.05  # [-1, +1]

        # --- 3. Volatility Regime Score ---
        # Elevated short-term vol vs long-term = mean reversion more powerful
        returns = prices.pct_change()
        vol_short = returns.rolling(window=20, min_periods=10).std() * np.sqrt(252)
        vol_long = returns.rolling(window=60, min_periods=30).std() * np.sqrt(252)
        vol_ratio = vol_short / vol_long.clip(lower=1e-6)

        # Higher ratio = more vol expansion = better for mean reversion
        vol_score = (vol_ratio.clip(0.5, 2.0) - 0.5) / 1.5  # [0, 1]
        vol_score = vol_score * 2 - 1  # Remap to [-1, +1]

        # --- Combine scores ---
        # Extension and deceleration are primary, vol is secondary
        raw_confidence = (
            extension_score * 0.4 +
            decel_score * 0.4 +
            vol_score * 0.2
        )

        # Map from [-1, 1] to [0, 1]
        confidence = (raw_confidence + 1) / 2

        return confidence.fillna(0.5)

    def generate_composite_signal(
        self,
        prices: pd.Series,
        volume: pd.Series,
        weights: Optional[Dict[str, float]] = None
    ) -> Tuple[pd.Series, Dict[str, pd.Series]]:
        """
        Generate composite signal from multiple indicators.

        The raw z-score is the primary signal (naturally scaled for thresholds).
        When Kalman filter is enabled, it replaces the rolling-window z-score.
        When OU prediction is enabled, signals are gated by expected return hurdle.
        Other indicators act as confirmation multipliers that boost or dampen
        the z-score.

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
        # Confirmation weights
        if weights is None:
            weights = {
                'bollinger': 0.25,
                'rsi_divergence': 0.25,
                'rsi_level': 0.25
            }

        individual_signals = {}

        # 1. Primary Z-Score: Kalman or rolling adaptive
        half_life = self.calculate_ou_half_life(prices)

        if self.config.use_kalman:
            zscore, kalman_mean, kalman_std = self.kalman_zscore(prices)
            individual_signals['kalman_mean'] = kalman_mean
            individual_signals['kalman_std'] = kalman_std
        else:
            zscore = self.adaptive_zscore(prices, half_life)

        individual_signals['zscore'] = zscore

        # 2. OU Predicted Return (makes strategy predictive)
        if self.config.use_predicted_return:
            expected_return, time_to_reversion = self.ou_predicted_return(prices, half_life)
            individual_signals['expected_return'] = expected_return
            individual_signals['time_to_reversion'] = time_to_reversion

        # 3. Bollinger Bands (confirmation: -1, 0, or +1)
        bb_signal = self.bollinger_bands_signal(prices, volume)
        individual_signals['bollinger'] = bb_signal

        # 4. RSI Divergence (confirmation: -1, 0, or +1)
        rsi_div_signal = self.rsi_divergence_signal(prices)
        individual_signals['rsi_divergence'] = rsi_div_signal

        # 5. RSI Level (confirmation: -1, 0, or +1)
        rsi_values = self.rsi(prices)
        rsi_signal = pd.Series(0.0, index=prices.index)
        rsi_signal[rsi_values < self.config.rsi_oversold] = -1.0
        rsi_signal[rsi_values > self.config.rsi_overbought] = 1.0
        individual_signals['rsi_level'] = rsi_signal

        # --- Signal Composition ---
        if self.config.signal_mode == 'gated':
            # GATED MODE (Phase B.1): Gate signal controls entries, z-score adds conviction
            # No gate signal = no trade (eliminates noise entries from z-score alone)
            gate_name = self.config.gate_signal
            if gate_name in individual_signals:
                gate = individual_signals[gate_name]  # -1, 0, or +1
            else:
                gate = pd.Series(0.0, index=prices.index)

            boost = self.config.zscore_boost_factor

            # Gate determines direction & entry; zscore magnitude adds conviction
            # composite = gate * (1 + boost * |zscore|)
            # When gate = 0:  composite = 0      -> no trade
            # When gate = -1: composite < 0       -> long entry
            # When gate = +1: composite > 0       -> short entry
            composite = gate * (1.0 + boost * zscore.abs())

            # Apply dynamic short confidence filter (Phase B.2)
            if self.config.use_dynamic_short_filter:
                short_confidence = self._compute_short_confidence(prices)
                # Dampen short signals (positive composite) when confidence is low
                short_mask = composite > 0
                # Zero out shorts below confidence threshold
                low_confidence = short_confidence < self.config.short_min_confidence
                composite[short_mask & low_confidence] = 0.0
                # Scale remaining shorts by confidence
                remaining_shorts = short_mask & ~low_confidence
                composite[remaining_shorts] = composite[remaining_shorts] * short_confidence[remaining_shorts]

        else:
            # CONFIRMATION MODE (legacy): z-score primary, others boost/dampen
            confirmation = pd.Series(0.0, index=prices.index)
            for name, weight in weights.items():
                signal = individual_signals[name]
                agreement = np.sign(zscore) * signal
                confirmation += weight * agreement
            composite = zscore * (1.0 + confirmation)

        # Apply OU hurdle gate: zero out signals where expected return is too small
        if self.config.use_predicted_return:
            hurdle = self.config.ou_hurdle_rate
            expected_return = individual_signals['expected_return']
            # Signal should agree with expected return direction AND exceed hurdle
            insufficient_ev = expected_return.abs() < hurdle
            composite[insufficient_ev] = 0.0

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
