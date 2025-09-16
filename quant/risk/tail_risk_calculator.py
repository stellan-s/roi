"""
Proper tail risk calculation based on statistical definitions.

Implements:
- Tail Risk (huvudmått): P[return < -2σ] (nedsidesrisk)
- Extreme Move Probability (sekundmått): P[|return| > 2σ] (tvåsidig)
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
from ..bayesian.signal_engine import SignalType

@dataclass
class TailRiskMetrics:
    """Comprehensive tail risk metrics for a position"""
    # Primary measure
    downside_tail_risk: float  # P[return < -2σ]

    # Secondary measure
    extreme_move_prob: float   # P[|return| > 2σ]

    # Additional metrics
    expected_shortfall: float  # E[return | return < -2σ]
    volatility_annual: float   # σ (annualized)

    # Distribution parameters
    distribution_type: str     # "normal", "student_t", "empirical"
    degrees_of_freedom: Optional[float]  # For Student-t
    skewness: float
    kurtosis: float

class TailRiskCalculator:
    """
    Statistical tail risk calculator using proper probability definitions.
    """

    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.min_observations = 100  # Minimum data for reliable estimates

    def calculate_tail_risk(self,
                          ticker: str,
                          historical_returns: pd.Series,
                          signals: Dict[SignalType, float],
                          regime: Optional[str] = None) -> TailRiskMetrics:
        """
        Calculate proper tail risk metrics for a ticker.

        Args:
            ticker: Stock ticker
            historical_returns: Daily returns series
            signals: Current signal values
            regime: Current market regime

        Returns:
            TailRiskMetrics with P[return < -2σ] and P[|return| > 2σ]
        """

        if len(historical_returns) < self.min_observations:
            return self._default_tail_risk()

        # Clean data
        returns = historical_returns.dropna()

        # Estimate volatility
        volatility_daily = returns.std()
        volatility_annual = volatility_daily * np.sqrt(252)

        # Distribution analysis
        distribution_type, params = self._fit_distribution(returns)

        # Calculate primary measure: P[return < -2σ]
        downside_tail_risk = self._calculate_downside_tail_risk(
            returns, volatility_daily, distribution_type, params
        )

        # Calculate secondary measure: P[|return| > 2σ]
        extreme_move_prob = self._calculate_extreme_move_prob(
            returns, volatility_daily, distribution_type, params
        )

        # Expected shortfall: E[return | return < -2σ]
        expected_shortfall = self._calculate_expected_shortfall(
            returns, volatility_daily, distribution_type, params
        )

        # Adjust for current signals and regime
        downside_tail_risk = self._adjust_for_signals_and_regime(
            downside_tail_risk, signals, regime
        )
        extreme_move_prob = self._adjust_for_signals_and_regime(
            extreme_move_prob, signals, regime, is_two_sided=True
        )

        return TailRiskMetrics(
            downside_tail_risk=downside_tail_risk,
            extreme_move_prob=extreme_move_prob,
            expected_shortfall=expected_shortfall,
            volatility_annual=volatility_annual,
            distribution_type=distribution_type,
            degrees_of_freedom=params.get('df') if distribution_type == 'student_t' else None,
            skewness=stats.skew(returns),
            kurtosis=stats.kurtosis(returns)
        )

    def _fit_distribution(self, returns: pd.Series) -> Tuple[str, Dict]:
        """
        Fit best distribution to returns data.

        Returns:
            tuple: (distribution_type, parameters)
        """

        # Test for normality
        _, p_value = stats.jarque_bera(returns)

        if p_value > 0.05:  # Accept normality
            return "normal", {
                'mean': returns.mean(),
                'std': returns.std()
            }

        # Try Student-t distribution for heavy tails
        try:
            df, loc, scale = stats.t.fit(returns)
            if df > 2.5:  # Reasonable degrees of freedom
                return "student_t", {
                    'df': df,
                    'loc': loc,
                    'scale': scale
                }
        except:
            pass

        # Fall back to empirical distribution
        return "empirical", {
            'returns': returns.values
        }

    def _calculate_downside_tail_risk(self,
                                    returns: pd.Series,
                                    volatility: float,
                                    distribution_type: str,
                                    params: Dict) -> float:
        """
        Calculate P[return < -2σ] using appropriate distribution.
        """

        threshold = -2 * volatility  # -2σ threshold

        if distribution_type == "normal":
            # For normal distribution: P[X < μ - 2σ]
            mean = params['mean']
            std = params['std']
            z_score = (threshold - mean) / std
            return stats.norm.cdf(z_score)

        elif distribution_type == "student_t":
            # For Student-t distribution
            df = params['df']
            loc = params['loc']
            scale = params['scale']
            return stats.t.cdf(threshold, df, loc, scale)

        else:  # empirical
            # Empirical probability
            return (returns < threshold).mean()

    def _calculate_extreme_move_prob(self,
                                   returns: pd.Series,
                                   volatility: float,
                                   distribution_type: str,
                                   params: Dict) -> float:
        """
        Calculate P[|return| > 2σ] using appropriate distribution.
        """

        upper_threshold = 2 * volatility   # +2σ
        lower_threshold = -2 * volatility  # -2σ

        if distribution_type == "normal":
            mean = params['mean']
            std = params['std']

            # P[X > μ + 2σ] + P[X < μ - 2σ]
            z_upper = (upper_threshold - mean) / std
            z_lower = (lower_threshold - mean) / std

            prob_upper = 1 - stats.norm.cdf(z_upper)
            prob_lower = stats.norm.cdf(z_lower)

            return prob_upper + prob_lower

        elif distribution_type == "student_t":
            df = params['df']
            loc = params['loc']
            scale = params['scale']

            prob_upper = 1 - stats.t.cdf(upper_threshold, df, loc, scale)
            prob_lower = stats.t.cdf(lower_threshold, df, loc, scale)

            return prob_upper + prob_lower

        else:  # empirical
            return ((returns > upper_threshold) | (returns < lower_threshold)).mean()

    def _calculate_expected_shortfall(self,
                                    returns: pd.Series,
                                    volatility: float,
                                    distribution_type: str,
                                    params: Dict) -> float:
        """
        Calculate E[return | return < -2σ] (expected shortfall).
        """

        threshold = -2 * volatility
        tail_returns = returns[returns < threshold]

        if len(tail_returns) == 0:
            # No observations in tail - estimate from distribution
            if distribution_type == "normal":
                mean = params['mean']
                std = params['std']
                z = (threshold - mean) / std
                # Conditional expectation for normal distribution
                phi_z = stats.norm.pdf(z)
                Phi_z = stats.norm.cdf(z)
                return mean - std * (phi_z / Phi_z) if Phi_z > 1e-10 else threshold
            else:
                return threshold * 1.5  # Conservative estimate

        return tail_returns.mean()

    def _adjust_for_signals_and_regime(self,
                                     base_risk: float,
                                     signals: Dict[SignalType, float],
                                     regime: Optional[str],
                                     is_two_sided: bool = False) -> float:
        """
        Adjust tail risk based on current signals and market regime.
        """

        adjustment_factor = 1.0

        # Signal-based adjustments
        if SignalType.MOMENTUM in signals:
            momentum = signals[SignalType.MOMENTUM]
            # Higher momentum = potentially higher volatility = higher tail risk
            momentum_adjustment = 1.0 + abs(momentum) * 0.2
            adjustment_factor *= momentum_adjustment

        if SignalType.SENTIMENT in signals:
            sentiment = signals[SignalType.SENTIMENT]
            if not is_two_sided:
                # For downside risk, negative sentiment increases risk
                if sentiment < 0:
                    sentiment_adjustment = 1.0 + abs(sentiment) * 0.15
                    adjustment_factor *= sentiment_adjustment
            else:
                # For two-sided risk, extreme sentiment (positive or negative) increases risk
                sentiment_adjustment = 1.0 + abs(sentiment) * 0.1
                adjustment_factor *= sentiment_adjustment

        # Regime-based adjustments
        if regime:
            regime_multipliers = {
                'bull': 0.8,    # Lower tail risk in bull markets
                'bear': 1.4,    # Higher tail risk in bear markets
                'neutral': 1.0  # Baseline tail risk
            }
            regime_multiplier = regime_multipliers.get(regime, 1.0)
            adjustment_factor *= regime_multiplier

        # Apply adjustment and ensure reasonable bounds
        adjusted_risk = base_risk * adjustment_factor
        return np.clip(adjusted_risk, 0.001, 0.5)  # Between 0.1% and 50%

    def _default_tail_risk(self) -> TailRiskMetrics:
        """Default tail risk when insufficient data."""
        return TailRiskMetrics(
            downside_tail_risk=0.025,  # ~2.5% default (slightly higher than normal 2.3%)
            extreme_move_prob=0.05,    # ~5% default (slightly higher than normal 4.6%)
            expected_shortfall=-0.03,  # -3% default shortfall
            volatility_annual=0.25,    # 25% default volatility
            distribution_type="default",
            degrees_of_freedom=None,
            skewness=0.0,
            kurtosis=0.0
        )

    def calculate_portfolio_tail_risk(self,
                                    positions: Dict[str, float],
                                    individual_risks: Dict[str, TailRiskMetrics],
                                    correlation_matrix: Optional[pd.DataFrame] = None) -> TailRiskMetrics:
        """
        Calculate portfolio-level tail risk from individual position risks.

        Args:
            positions: {ticker: weight} portfolio positions
            individual_risks: {ticker: TailRiskMetrics} for each position
            correlation_matrix: Asset correlation matrix

        Returns:
            Portfolio-level TailRiskMetrics
        """

        if not positions or not individual_risks:
            return self._default_tail_risk()

        # Weighted average of individual tail risks
        total_weight = sum(abs(w) for w in positions.values())
        if total_weight == 0:
            return self._default_tail_risk()

        # Portfolio downside tail risk (conservative: average of weighted individual risks)
        portfolio_downside_risk = sum(
            abs(weight) * individual_risks[ticker].downside_tail_risk
            for ticker, weight in positions.items()
            if ticker in individual_risks
        ) / total_weight

        # Portfolio extreme move probability
        portfolio_extreme_prob = sum(
            abs(weight) * individual_risks[ticker].extreme_move_prob
            for ticker, weight in positions.items()
            if ticker in individual_risks
        ) / total_weight

        # Portfolio volatility (simplified - could use correlation matrix)
        portfolio_vol = sum(
            abs(weight) * individual_risks[ticker].volatility_annual
            for ticker, weight in positions.items()
            if ticker in individual_risks
        ) / total_weight

        # Portfolio expected shortfall
        portfolio_shortfall = sum(
            abs(weight) * individual_risks[ticker].expected_shortfall
            for ticker, weight in positions.items()
            if ticker in individual_risks
        ) / total_weight

        return TailRiskMetrics(
            downside_tail_risk=portfolio_downside_risk,
            extreme_move_prob=portfolio_extreme_prob,
            expected_shortfall=portfolio_shortfall,
            volatility_annual=portfolio_vol,
            distribution_type="portfolio",
            degrees_of_freedom=None,
            skewness=0.0,  # Portfolio-level skewness calculation would be complex
            kurtosis=0.0   # Portfolio-level kurtosis calculation would be complex
        )