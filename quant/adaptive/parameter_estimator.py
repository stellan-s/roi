"""
Parameter Estimator - Data-driven parameter learning system.

This module replaces hardcoded configuration values with parameters learned
from historical market data, providing evidence-based system calibration.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass
from pathlib import Path
import json
from datetime import datetime

@dataclass
class EstimatedParameter:
    """A parameter learned from historical data with uncertainty quantification."""
    value: float                               # The learned parameter value
    confidence_interval: Tuple[float, float]   # (lower, upper) confidence bounds
    estimation_method: str                     # How the parameter was estimated
    n_observations: int                       # Amount of data used
    r_squared: Optional[float] = None         # Goodness of fit (if applicable)
    p_value: Optional[float] = None           # Statistical significance
    default_value: Optional[float] = None     # Original hardcoded value

@dataclass
class ParameterEstimates:
    """Complete set of learned parameters for the adaptive system."""

    # Signal normalization parameters
    sentiment_scale_factor: EstimatedParameter
    momentum_scale_factor: EstimatedParameter

    # Bayesian engine parameters
    base_annual_return: EstimatedParameter
    sigmoid_scale_factor: EstimatedParameter
    signal_multiplier: EstimatedParameter

    # Risk parameters
    momentum_volatility_factor: EstimatedParameter

    # Regime adjustments (learned effectiveness multipliers)
    regime_adjustments: Dict[str, Dict[str, EstimatedParameter]]

    # Metadata
    estimation_date: datetime
    data_period_start: datetime
    data_period_end: datetime
    total_observations: int

class ParameterEstimator:
    """
    Data-driven parameter estimation system.

    Learns optimal system parameters from historical data instead of using
    hardcoded configuration values, improving adaptability and performance.
    """

    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.alpha = 1.0 - confidence_level  # For confidence intervals

    def estimate_all_parameters(self,
                               prices_df: pd.DataFrame,
                               sentiment_df: pd.DataFrame,
                               technical_df: pd.DataFrame,
                               regime_df: Optional[pd.DataFrame] = None,
                               returns_df: Optional[pd.DataFrame] = None) -> ParameterEstimates:
        """
        Estimate all parameters from historical data.

        Args:
            prices_df: Historical price data
            sentiment_df: Historical sentiment data
            technical_df: Technical indicator data
            regime_df: Optional regime classification data
            returns_df: Optional return data (will be calculated if not provided)

        Returns:
            ParameterEstimates with all learned parameters
        """

        print("Estimating parameters from historical data...")

        # Calculate returns if not provided
        if returns_df is None:
            returns_df = self._calculate_returns(prices_df)

        # Generate regime data if not provided
        if regime_df is None:
            regime_df = self._generate_regime_data(prices_df)

        # Estimate signal normalization parameters
        signal_params = self.estimate_signal_normalization_params(
            prices_df, sentiment_df, technical_df
        )

        # Estimate Bayesian engine parameters
        bayesian_params = self.estimate_bayesian_engine_params(
            technical_df, sentiment_df, returns_df
        )

        # Estimate risk parameters
        risk_params = self.estimate_risk_parameters(
            technical_df, returns_df
        )

        # Estimate regime adjustments
        regime_adjustments = self.estimate_regime_adjustments(
            prices_df, technical_df, regime_df, returns_df
        )

        # Compile all estimates
        estimates = ParameterEstimates(
            # Signal normalization
            sentiment_scale_factor=signal_params['sentiment_scale'],
            momentum_scale_factor=signal_params['momentum_scale'],

            # Bayesian engine
            base_annual_return=bayesian_params['base_annual_return'],
            sigmoid_scale_factor=bayesian_params['sigmoid_scale'],
            signal_multiplier=bayesian_params['signal_multiplier'],

            # Risk parameters
            momentum_volatility_factor=risk_params['momentum_volatility_factor'],

            # Regime adjustments
            regime_adjustments=regime_adjustments,

            # Metadata
            estimation_date=datetime.now(),
            data_period_start=prices_df['date'].min(),
            data_period_end=prices_df['date'].max(),
            total_observations=len(prices_df)
        )

        return estimates

    def estimate_signal_normalization_params(self,
                                           prices_df: pd.DataFrame,
                                           sentiment_df: pd.DataFrame,
                                           technical_df: pd.DataFrame) -> Dict[str, EstimatedParameter]:
        """
        Estimate signal normalization parameters based on empirical distributions.

        Replaces hardcoded values like:
        - sentiment_signal = np.clip(sent_score / 2.0, -1.0, 1.0)  # /2.0 → learned
        - momentum_signal = (mom_rank - 0.5) * 2.0                # *2.0 → learned

        Returns:
            Dict with 'sentiment_scale' and 'momentum_scale' parameters
        """

        print("  Estimating signal normalization parameters...")

        # Sentiment scale factor
        if not sentiment_df.empty and 'sent_score' in sentiment_df.columns:
            sent_scores = sentiment_df['sent_score'].dropna()
            if len(sent_scores) > 10:
                # Use 5th to 95th percentile range for robust scaling
                p5, p95 = np.percentile(sent_scores, [5, 95])
                # Scale to use full [-1, 1] range
                learned_scale = 2.0 / (p95 - p5) if (p95 - p5) > 0 else 0.5

                sentiment_scale = EstimatedParameter(
                    value=learned_scale,
                    confidence_interval=self._bootstrap_confidence_interval(
                        sent_scores, lambda x: 2.0 / (np.percentile(x, 95) - np.percentile(x, 5))
                    ),
                    estimation_method="percentile_based_scaling",
                    n_observations=len(sent_scores),
                    default_value=0.5  # Original hardcoded /2.0
                )
            else:
                sentiment_scale = self._default_parameter(0.5, "insufficient_sentiment_data")
        else:
            sentiment_scale = self._default_parameter(0.5, "no_sentiment_data")

        # Momentum scale factor
        if not technical_df.empty and 'mom_rank' in technical_df.columns:
            mom_ranks = technical_df['mom_rank'].dropna()
            if len(mom_ranks) > 10:
                # Momentum rank should be [0,1], optimal scaling factor
                actual_range = mom_ranks.max() - mom_ranks.min()
                learned_scale = 2.0 / actual_range if actual_range > 0 else 2.0

                momentum_scale = EstimatedParameter(
                    value=learned_scale,
                    confidence_interval=self._bootstrap_confidence_interval(
                        mom_ranks, lambda x: 2.0 / (x.max() - x.min()) if (x.max() - x.min()) > 0 else 2.0
                    ),
                    estimation_method="range_based_scaling",
                    n_observations=len(mom_ranks),
                    default_value=2.0  # Original hardcoded *2.0
                )
            else:
                momentum_scale = self._default_parameter(2.0, "insufficient_momentum_data")
        else:
            momentum_scale = self._default_parameter(2.0, "no_momentum_data")

        return {
            'sentiment_scale': sentiment_scale,
            'momentum_scale': momentum_scale
        }

    def estimate_bayesian_engine_params(self,
                                      technical_df: pd.DataFrame,
                                      sentiment_df: pd.DataFrame,
                                      returns_df: pd.DataFrame) -> Dict[str, EstimatedParameter]:
        """Estimate core Bayesian engine parameters."""

        print("  Estimating Bayesian engine parameters...")

        # Base annual return (market expectation)
        if not returns_df.empty:
            daily_returns = returns_df['return'].dropna()
            annual_return = daily_returns.mean() * 252

            base_annual_return = EstimatedParameter(
                value=annual_return,
                confidence_interval=self._bootstrap_confidence_interval(
                    daily_returns, lambda x: x.mean() * 252
                ),
                estimation_method="historical_mean_annualized",
                n_observations=len(daily_returns),
                default_value=0.07  # Typical 7% market return assumption
            )
        else:
            base_annual_return = self._default_parameter(0.07, "no_return_data")

        # Sigmoid scale factor (for probability transformation)
        sigmoid_scale = EstimatedParameter(
            value=3.0,  # Conservative scaling
            confidence_interval=(2.5, 3.5),
            estimation_method="theoretical_default",
            n_observations=0,
            default_value=3.0
        )

        # Signal multiplier (overall signal strength)
        signal_multiplier = EstimatedParameter(
            value=1.0,  # Start neutral
            confidence_interval=(0.8, 1.2),
            estimation_method="theoretical_default",
            n_observations=0,
            default_value=1.0
        )

        return {
            'base_annual_return': base_annual_return,
            'sigmoid_scale': sigmoid_scale,
            'signal_multiplier': signal_multiplier
        }

    def estimate_risk_parameters(self,
                               technical_df: pd.DataFrame,
                               returns_df: pd.DataFrame) -> Dict[str, EstimatedParameter]:
        """Estimate risk-related parameters."""

        print("  Estimating risk parameters...")

        # Momentum volatility factor
        if not technical_df.empty and not returns_df.empty:
            # Merge technical and returns data
            merged = technical_df.merge(returns_df, on=['ticker', 'date'], how='inner')

            if 'mom_rank' in merged.columns and len(merged) > 50:
                # Calculate correlation between momentum and subsequent volatility
                correlations = []
                for ticker in merged['ticker'].unique():
                    ticker_data = merged[merged['ticker'] == ticker].sort_values('date')
                    if len(ticker_data) > 20:
                        # Rolling volatility (10-day)
                        ticker_data['volatility'] = ticker_data['return'].rolling(10).std()
                        # Correlation between momentum and future volatility
                        corr = ticker_data['mom_rank'].corr(ticker_data['volatility'].shift(-5))
                        if not np.isnan(corr):
                            correlations.append(abs(corr))

                if correlations:
                    momentum_vol_factor = np.mean(correlations) * 0.5  # Scale to reasonable range

                    momentum_volatility_factor = EstimatedParameter(
                        value=momentum_vol_factor,
                        confidence_interval=self._bootstrap_confidence_interval(
                            np.array(correlations), lambda x: np.mean(x) * 0.5
                        ),
                        estimation_method="momentum_volatility_correlation",
                        n_observations=len(correlations),
                        default_value=0.3  # Original hardcoded factor
                    )
                else:
                    momentum_volatility_factor = self._default_parameter(0.3, "insufficient_correlation_data")
            else:
                momentum_volatility_factor = self._default_parameter(0.3, "insufficient_merged_data")
        else:
            momentum_volatility_factor = self._default_parameter(0.3, "no_data")

        return {
            'momentum_volatility_factor': momentum_volatility_factor
        }

    def estimate_regime_adjustments(self,
                                  prices_df: pd.DataFrame,
                                  signals_df: pd.DataFrame,
                                  regime_df: pd.DataFrame,
                                  returns_df: pd.DataFrame) -> Dict[str, Dict[str, EstimatedParameter]]:
        """
        Learn regime-conditional signal effectiveness multipliers.

        Replaces hardcoded regime adjustments like:
        - Bull: {"momentum": 1.3, "trend": 1.2, "sentiment": 0.8}
        - Bear: {"momentum": 0.7, "trend": 1.1, "sentiment": 1.4}

        Returns:
            Dict[regime][signal_type] = EstimatedParameter with learned multipliers
        """

        print("  Estimating regime-conditional signal adjustments...")

        # Default hardcoded values for fallback
        defaults = {
            'bull': {'momentum': 1.2, 'trend': 1.0, 'sentiment': 0.8},
            'bear': {'momentum': 0.7, 'trend': 1.1, 'sentiment': 1.4},
            'neutral': {'momentum': 1.0, 'trend': 1.0, 'sentiment': 1.0}
        }

        regime_adjustments = {}

        # Merge all data for analysis
        merged = signals_df.merge(returns_df, on=['ticker', 'date'], how='inner')
        merged = merged.merge(regime_df, on=['ticker', 'date'], how='inner')

        if len(merged) < 100:
            print("    Insufficient data for regime learning, using defaults")
            return self._default_regime_adjustments(defaults)

        # Calculate forward returns for signal effectiveness
        merged = merged.sort_values(['ticker', 'date'])
        merged['forward_return'] = merged.groupby('ticker')['return'].shift(-21)  # 21-day forward
        merged = merged.dropna()

        for regime in ['bull', 'bear', 'neutral']:
            regime_data = merged[merged['regime'] == regime]
            regime_adjustments[regime] = {}

            if len(regime_data) < 30:
                print(f"    Insufficient {regime} data, using defaults")
                for signal in ['momentum', 'trend', 'sentiment']:
                    regime_adjustments[regime][signal] = self._default_parameter(
                        defaults[regime][signal], f"insufficient_{regime}_data"
                    )
                continue

            # Calculate signal effectiveness in this regime
            for signal in ['momentum', 'trend', 'sentiment']:
                signal_col = f'{signal}_signal' if f'{signal}_signal' in regime_data.columns else signal

                if signal_col in regime_data.columns:
                    # Correlation in this regime
                    regime_corr = regime_data[signal_col].corr(regime_data['forward_return'])

                    # Overall correlation
                    overall_corr = merged[signal_col].corr(merged['forward_return'])

                    if not np.isnan(regime_corr) and not np.isnan(overall_corr) and abs(overall_corr) > 0.01:
                        # Learned multiplier = regime effectiveness / overall effectiveness
                        learned_multiplier = regime_corr / overall_corr
                        # Constrain to reasonable range
                        learned_multiplier = np.clip(learned_multiplier, 0.3, 3.0)

                        regime_adjustments[regime][signal] = EstimatedParameter(
                            value=learned_multiplier,
                            confidence_interval=self._regime_adjustment_confidence_interval(
                                regime_data, merged, signal_col
                            ),
                            estimation_method="regime_conditional_correlation",
                            n_observations=len(regime_data),
                            default_value=defaults[regime][signal]
                        )
                    else:
                        regime_adjustments[regime][signal] = self._default_parameter(
                            defaults[regime][signal], f"insufficient_{signal}_correlation"
                        )
                else:
                    regime_adjustments[regime][signal] = self._default_parameter(
                        defaults[regime][signal], f"missing_{signal}_column"
                    )

        return regime_adjustments

    def _calculate_returns(self, prices_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate daily returns from price data."""
        returns_df = prices_df.copy()
        returns_df = returns_df.sort_values(['ticker', 'date'])
        returns_df['return'] = returns_df.groupby('ticker')['close'].pct_change()
        return returns_df.dropna()

    def _generate_regime_data(self, prices_df: pd.DataFrame) -> pd.DataFrame:
        """Generate simple regime classification for learning."""
        # Simple regime classification based on 20-day returns
        regime_df = prices_df.copy()
        regime_df = regime_df.sort_values(['ticker', 'date'])

        # 20-day rolling return
        regime_df['return_20d'] = regime_df.groupby('ticker')['close'].pct_change(20)

        # Simple regime classification
        regime_df['regime'] = 'neutral'
        regime_df.loc[regime_df['return_20d'] > 0.05, 'regime'] = 'bull'    # >5% gain
        regime_df.loc[regime_df['return_20d'] < -0.05, 'regime'] = 'bear'   # >5% loss

        return regime_df[['ticker', 'date', 'regime']].dropna()

    def _bootstrap_confidence_interval(self, data: np.ndarray, estimator_func, n_bootstrap: int = 500) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval for an estimator."""
        if len(data) < 10:
            return (0.0, 0.0)

        bootstrap_estimates = []
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
            try:
                estimate = estimator_func(bootstrap_sample)
                if not np.isnan(estimate) and np.isfinite(estimate):
                    bootstrap_estimates.append(estimate)
            except:
                continue

        if len(bootstrap_estimates) < 10:
            return (0.0, 0.0)

        lower = np.percentile(bootstrap_estimates, 100 * self.alpha / 2)
        upper = np.percentile(bootstrap_estimates, 100 * (1 - self.alpha / 2))
        return (lower, upper)

    def _regime_adjustment_confidence_interval(self, regime_data: pd.DataFrame, all_data: pd.DataFrame, signal_col: str) -> Tuple[float, float]:
        """Calculate confidence interval for regime adjustment factors."""
        # Simplified confidence interval
        if len(regime_data) < 20:
            return (0.5, 2.0)

        # Use standard error approximation
        regime_corr = regime_data[signal_col].corr(regime_data['forward_return'])
        overall_corr = all_data[signal_col].corr(all_data['forward_return'])

        if abs(overall_corr) < 0.01:
            return (0.5, 2.0)

        multiplier = regime_corr / overall_corr
        # Rough confidence interval ±30%
        lower = max(0.3, multiplier * 0.7)
        upper = min(3.0, multiplier * 1.3)

        return (lower, upper)

    def _default_parameter(self, value: float, reason: str) -> EstimatedParameter:
        """Create a default parameter when learning fails."""
        return EstimatedParameter(
            value=value,
            confidence_interval=(value * 0.8, value * 1.2),
            estimation_method=f"default_fallback_{reason}",
            n_observations=0,
            default_value=value
        )

    def _default_regime_adjustments(self, defaults: Dict) -> Dict[str, Dict[str, EstimatedParameter]]:
        """Create default regime adjustments when learning fails."""
        result = {}
        for regime, signals in defaults.items():
            result[regime] = {}
            for signal, value in signals.items():
                result[regime][signal] = self._default_parameter(value, f"default_{regime}_{signal}")
        return result

    def save_parameters(self, estimates: ParameterEstimates, filepath: str):
        """Save parameter estimates to JSON file."""
        # Convert to serializable format
        data = {
            'estimation_metadata': {
                'estimation_date': estimates.estimation_date.isoformat(),
                'data_period_start': estimates.data_period_start.isoformat(),
                'data_period_end': estimates.data_period_end.isoformat(),
                'total_observations': estimates.total_observations
            },
            'parameters': {}
        }

        # Add all parameters
        for attr_name in ['sentiment_scale_factor', 'momentum_scale_factor',
                         'base_annual_return', 'sigmoid_scale_factor', 'signal_multiplier',
                         'momentum_volatility_factor']:
            param = getattr(estimates, attr_name)
            data['parameters'][attr_name] = {
                'value': param.value,
                'confidence_interval': param.confidence_interval,
                'estimation_method': param.estimation_method,
                'n_observations': param.n_observations,
                'default_value': param.default_value
            }

        # Add regime adjustments
        data['regime_adjustments'] = {}
        for regime, signals in estimates.regime_adjustments.items():
            data['regime_adjustments'][regime] = {}
            for signal, param in signals.items():
                data['regime_adjustments'][regime][signal] = {
                    'value': param.value,
                    'confidence_interval': param.confidence_interval,
                    'estimation_method': param.estimation_method,
                    'n_observations': param.n_observations,
                    'default_value': param.default_value
                }

        # Save to file
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Parameter estimates saved to {filepath}")

    def load_parameters(self, filepath: str) -> Optional[ParameterEstimates]:
        """Load parameter estimates from JSON file."""
        if not Path(filepath).exists():
            return None

        with open(filepath, 'r') as f:
            data = json.load(f)

        # Reconstruct ParameterEstimates object
        # This is a simplified reconstruction - full implementation would be more thorough
        print(f"Parameter estimates loaded from {filepath}")
        return None  # TODO: Implement full reconstruction