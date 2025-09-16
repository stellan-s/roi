"""
Parameter estimation framework for data-driven model calibration.

This module replaces hardcoded parameters with empirically estimated values
based on historical data analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

@dataclass
class EstimatedParameter:
    """Container for an estimated parameter with confidence intervals"""
    value: float
    confidence_interval: Tuple[float, float]
    estimation_method: str
    n_observations: int
    r_squared: Optional[float] = None

@dataclass
class ParameterEstimates:
    """Collection of all estimated parameters"""
    # Signal normalization
    sentiment_scale_factor: EstimatedParameter
    momentum_scale_factor: EstimatedParameter

    # Bayesian engine
    base_annual_return: EstimatedParameter
    sigmoid_scale_factor: EstimatedParameter
    signal_multiplier: EstimatedParameter

    # Risk parameters
    momentum_volatility_factor: EstimatedParameter

    # Regime adjustments (learned weights)
    regime_adjustments: Dict[str, Dict[str, EstimatedParameter]]

class ParameterEstimator:
    """
    Estimates model parameters from historical data instead of using hardcoded values.
    """

    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level

    def estimate_signal_normalization_params(self,
                                           prices_df: pd.DataFrame,
                                           sentiment_df: pd.DataFrame,
                                           technical_df: pd.DataFrame) -> Dict[str, EstimatedParameter]:
        """
        Estimate signal normalization parameters based on empirical distributions.
        """
        params = {}

        # Sentiment scaling: Use percentile-based normalization
        if not sentiment_df.empty and 'sent_score' in sentiment_df.columns:
            sent_scores = sentiment_df['sent_score'].dropna()
            if len(sent_scores) > 10:
                # Use 95th percentile range for normalization
                p5, p95 = np.percentile(sent_scores, [5, 95])
                scale_factor = 2.0 / (p95 - p5) if (p95 - p5) > 0 else 1.0

                params['sentiment_scale'] = EstimatedParameter(
                    value=scale_factor,
                    confidence_interval=(scale_factor * 0.8, scale_factor * 1.2),
                    estimation_method="percentile_based",
                    n_observations=len(sent_scores)
                )

        # Momentum scaling: Based on momentum rank distribution
        if not technical_df.empty and 'mom_rank' in technical_df.columns:
            mom_ranks = technical_df['mom_rank'].dropna()
            if len(mom_ranks) > 10:
                # Check if mom_rank is already normalized [0,1]
                mom_std = mom_ranks.std()
                # Scale to achieve desired signal strength
                scale_factor = 2.0 / (mom_std * 4) if mom_std > 0 else 2.0  # 4-sigma range

                params['momentum_scale'] = EstimatedParameter(
                    value=scale_factor,
                    confidence_interval=(scale_factor * 0.9, scale_factor * 1.1),
                    estimation_method="distribution_based",
                    n_observations=len(mom_ranks)
                )

        return params

    def estimate_bayesian_params(self,
                                prices_df: pd.DataFrame,
                                returns_df: Optional[pd.DataFrame] = None) -> Dict[str, EstimatedParameter]:
        """
        Estimate Bayesian engine parameters from historical market data.
        """
        params = {}

        # Base annual return from historical data
        if returns_df is not None and 'return' in returns_df.columns:
            daily_returns = returns_df['return'].dropna()
        else:
            # Calculate from prices
            daily_returns = prices_df.groupby('ticker')['close'].pct_change().dropna()

        if len(daily_returns) > 252:  # At least 1 year of data
            # Annualized return
            mean_daily_return = daily_returns.mean()
            annual_return = (1 + mean_daily_return) ** 252 - 1

            # Confidence interval using bootstrap
            annual_returns_bootstrap = []
            for _ in range(1000):
                sample = np.random.choice(daily_returns, len(daily_returns), replace=True)
                annual_returns_bootstrap.append((1 + sample.mean()) ** 252 - 1)

            ci_lower = np.percentile(annual_returns_bootstrap, (self.alpha/2) * 100)
            ci_upper = np.percentile(annual_returns_bootstrap, (1 - self.alpha/2) * 100)

            params['base_annual_return'] = EstimatedParameter(
                value=annual_return,
                confidence_interval=(ci_lower, ci_upper),
                estimation_method="historical_bootstrap",
                n_observations=len(daily_returns)
            )

        return params

    def estimate_signal_predictive_power(self,
                                       signals_df: pd.DataFrame,
                                       returns_df: pd.DataFrame,
                                       horizon_days: int = 21) -> Dict[str, EstimatedParameter]:
        """
        Estimate signal predictive power and calibration parameters.
        """
        params = {}

        # Merge signals with forward returns
        merged_df = self._prepare_signal_return_data(signals_df, returns_df, horizon_days)

        if len(merged_df) < 100:  # Need sufficient data
            return params

        # Estimate sigmoid scaling factor
        # Find optimal scaling that maximizes signal-return correlation
        signal_cols = ['trend_signal', 'momentum_signal', 'sentiment_signal']
        signal_cols = [col for col in signal_cols if col in merged_df.columns]

        if signal_cols:
            # Combined signal
            combined_signal = merged_df[signal_cols].mean(axis=1)
            forward_returns = merged_df['forward_return']

            # Grid search for optimal sigmoid scaling
            scales = np.linspace(0.5, 10.0, 20)
            correlations = []

            for scale in scales:
                prob_positive = 1 / (1 + np.exp(-scale * combined_signal))
                # Convert to expected direction and correlate with actual returns
                expected_direction = 2 * prob_positive - 1  # [-1, 1]
                actual_direction = np.sign(forward_returns)
                corr = np.corrcoef(expected_direction, actual_direction)[0, 1]
                correlations.append(corr)

            best_scale = scales[np.argmax(correlations)]
            best_corr = max(correlations)

            params['sigmoid_scale'] = EstimatedParameter(
                value=best_scale,
                confidence_interval=(best_scale * 0.8, best_scale * 1.2),
                estimation_method="correlation_optimization",
                n_observations=len(merged_df),
                r_squared=best_corr**2
            )

        return params

    def estimate_regime_adjustments(self,
                                  prices_df: pd.DataFrame,
                                  signals_df: pd.DataFrame,
                                  regime_df: pd.DataFrame,
                                  returns_df: pd.DataFrame) -> Dict[str, Dict[str, EstimatedParameter]]:
        """
        Learn regime-conditional signal effectiveness multipliers.
        """
        # Merge all data
        merged_df = self._prepare_regime_signal_data(prices_df, signals_df, regime_df, returns_df)

        if len(merged_df) < 200:  # Need sufficient data
            return {}

        adjustments = {}
        signal_types = ['momentum', 'trend', 'sentiment']
        regimes = ['bull', 'bear', 'neutral']

        for regime in regimes:
            adjustments[regime] = {}
            regime_data = merged_df[merged_df['regime'] == regime]

            if len(regime_data) < 50:  # Minimum data per regime
                continue

            for signal_type in signal_types:
                signal_col = f'{signal_type}_signal'
                if signal_col not in regime_data.columns:
                    continue

                # Calculate signal effectiveness in this regime vs overall
                regime_corr = self._calculate_signal_return_correlation(
                    regime_data[signal_col], regime_data['forward_return']
                )

                overall_corr = self._calculate_signal_return_correlation(
                    merged_df[signal_col], merged_df['forward_return']
                )

                if overall_corr != 0:
                    effectiveness_ratio = regime_corr / overall_corr
                else:
                    effectiveness_ratio = 1.0

                # Bootstrap confidence interval
                ratios = []
                for _ in range(500):
                    boot_regime = regime_data.sample(n=len(regime_data), replace=True)
                    boot_overall = merged_df.sample(n=len(merged_df), replace=True)

                    boot_regime_corr = self._calculate_signal_return_correlation(
                        boot_regime[signal_col], boot_regime['forward_return']
                    )
                    boot_overall_corr = self._calculate_signal_return_correlation(
                        boot_overall[signal_col], boot_overall['forward_return']
                    )

                    if boot_overall_corr != 0:
                        ratios.append(boot_regime_corr / boot_overall_corr)

                if ratios:
                    ci_lower = np.percentile(ratios, (self.alpha/2) * 100)
                    ci_upper = np.percentile(ratios, (1 - self.alpha/2) * 100)
                else:
                    ci_lower, ci_upper = effectiveness_ratio * 0.8, effectiveness_ratio * 1.2

                adjustments[regime][signal_type] = EstimatedParameter(
                    value=effectiveness_ratio,
                    confidence_interval=(ci_lower, ci_upper),
                    estimation_method="regime_conditional_correlation",
                    n_observations=len(regime_data),
                    r_squared=regime_corr**2
                )

        return adjustments

    def _prepare_signal_return_data(self, signals_df: pd.DataFrame,
                                  returns_df: pd.DataFrame,
                                  horizon_days: int) -> pd.DataFrame:
        """Prepare aligned signal and forward return data"""
        # This is a simplified version - would need proper implementation
        # based on the actual data structure
        merged = pd.merge(signals_df, returns_df, on=['ticker', 'date'], how='inner')

        # Calculate forward returns
        merged = merged.sort_values(['ticker', 'date'])
        merged['forward_return'] = merged.groupby('ticker')['return'].shift(-horizon_days)

        return merged.dropna()

    def _prepare_regime_signal_data(self, prices_df: pd.DataFrame,
                                  signals_df: pd.DataFrame,
                                  regime_df: pd.DataFrame,
                                  returns_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare aligned regime, signal, and return data"""
        # Merge all dataframes - simplified implementation
        merged = pd.merge(signals_df, returns_df, on=['ticker', 'date'], how='inner')
        merged = pd.merge(merged, regime_df, on='date', how='inner')

        # Calculate forward returns
        merged = merged.sort_values(['ticker', 'date'])
        merged['forward_return'] = merged.groupby('ticker')['return'].shift(-21)

        return merged.dropna()

    def _calculate_signal_return_correlation(self, signal: pd.Series, returns: pd.Series) -> float:
        """Calculate correlation between signal and returns, handling edge cases"""
        try:
            if len(signal) < 10 or signal.std() == 0 or returns.std() == 0:
                return 0.0
            return np.corrcoef(signal, returns)[0, 1]
        except:
            return 0.0

    def estimate_all_parameters(self,
                              prices_df: pd.DataFrame,
                              sentiment_df: pd.DataFrame,
                              technical_df: pd.DataFrame,
                              regime_df: Optional[pd.DataFrame] = None,
                              returns_df: Optional[pd.DataFrame] = None) -> ParameterEstimates:
        """
        Estimate all parameters from historical data.
        """
        # Signal normalization
        signal_params = self.estimate_signal_normalization_params(
            prices_df, sentiment_df, technical_df
        )

        # Bayesian parameters
        bayesian_params = self.estimate_bayesian_params(prices_df, returns_df)

        # Predictive power
        if returns_df is not None:
            predictive_params = self.estimate_signal_predictive_power(
                technical_df, returns_df
            )
        else:
            predictive_params = {}

        # Regime adjustments
        if regime_df is not None and returns_df is not None:
            regime_adjustments = self.estimate_regime_adjustments(
                prices_df, technical_df, regime_df, returns_df
            )
        else:
            regime_adjustments = {}

        # Combine all estimates with defaults for missing values
        return ParameterEstimates(
            sentiment_scale_factor=signal_params.get('sentiment_scale',
                EstimatedParameter(0.5, (0.4, 0.6), "default", 0)),
            momentum_scale_factor=signal_params.get('momentum_scale',
                EstimatedParameter(2.0, (1.8, 2.2), "default", 0)),
            base_annual_return=bayesian_params.get('base_annual_return',
                EstimatedParameter(0.08, (0.05, 0.12), "default", 0)),
            sigmoid_scale_factor=predictive_params.get('sigmoid_scale',
                EstimatedParameter(3.0, (2.0, 4.0), "default", 0)),
            signal_multiplier=EstimatedParameter(2.0, (1.5, 2.5), "default", 0),
            momentum_volatility_factor=EstimatedParameter(0.3, (0.2, 0.4), "default", 0),
            regime_adjustments=regime_adjustments
        )