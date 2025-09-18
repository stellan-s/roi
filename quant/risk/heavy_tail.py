import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

class TailRiskMeasure(Enum):
    """Different tail risk measures."""
    VALUE_AT_RISK = "var"           # VaR at the given confidence level
    CONDITIONAL_VAR = "cvar"        # Expected Shortfall / CVaR
    EXTREME_DRAWDOWN = "max_dd"     # Maximum expected drawdown
    TAIL_EXPECTATION = "tail_exp"   # Expected return in the tail

@dataclass
class TailRiskMetrics:
    """Tail risk metrics for a given time horizon."""
    confidence_level: float         # 95%, 99%, 99.9%
    time_horizon_days: int         # 21, 63, 252 days

    # Basic tail measures
    var_normal: float              # VaR assuming normal distribution
    var_student_t: float           # VaR using Student-t (heavy tails)
    cvar_student_t: float          # Conditional VaR (expected shortfall)

    # Extreme value measures
    evt_var: float                 # VaR from EVT (GPD tail fitting)
    evt_return_level: float        # Expected extreme return level

    # Distribution parameters
    degrees_of_freedom: float      # Estimated DoF for Student-t
    tail_index: float              # Heavy-tail index from EVT

    # Interpretations
    tail_risk_multiplier: float    # Heavy-tail risk vs normal (t_var/normal_var)
    extreme_event_probability: float  # P(extreme event i tidshorisont)

@dataclass
class MonteCarloResults:
    """Monte Carlo simulation results for multiple outcome scenarios."""
    n_simulations: int
    time_horizon_months: int

    # Target probability levels
    prob_positive: float           # P(return > 0%)
    prob_plus_10: float           # P(return > +10%)
    prob_plus_20: float           # P(return > +20%)
    prob_plus_30: float           # P(return > +30%)

    # Downside probabilities
    prob_minus_10: float          # P(return < -10%)
    prob_minus_20: float          # P(return < -20%)
    prob_minus_30: float          # P(return < -30%)

    # Distribution statistics
    mean_return: float
    median_return: float
    std_return: float
    skewness: float
    kurtosis: float

    # Extreme scenarios
    percentile_1: float           # 1st percentile (very bad)
    percentile_5: float           # 5th percentile (bad)
    percentile_95: float          # 95th percentile (good)
    percentile_99: float          # 99th percentile (very good)

class HeavyTailRiskModel:
    """
    Heavy-tail risk modelling with Student-t and EVT.

    Implements:
    1. Student-t distribution fitting for heavy-tail modelling
    2. Extreme Value Theory (EVT) for tail risk estimation
    3. Monte Carlo simulation with heavy-tail properties
    4. Realistic risk measures versus naive normal assumptions
    """

    def __init__(self, config: Optional[Dict] = None):
        # Configurable parameters
        if config and 'risk_modeling' in config:
            risk_config = config['risk_modeling']
            self.confidence_levels = risk_config.get('confidence_levels', [0.95, 0.99, 0.999])
            self.time_horizons = risk_config.get('time_horizons_days', [21, 63, 252])
            self.mc_simulations = risk_config.get('monte_carlo_simulations', 10000)
            self.evt_threshold_percentile = risk_config.get('evt_threshold_percentile', 0.95)
        else:
            # Defaults
            self.confidence_levels = [0.95, 0.99, 0.999]
            self.time_horizons = [21, 63, 252]  # 1m, 3m, 1y
            self.mc_simulations = 10000
            self.evt_threshold_percentile = 0.95

        # Cached fitted parameters
        self.fitted_parameters = {}

    def fit_heavy_tail_distribution(self, returns: pd.Series) -> Dict:
        """
        Fit a Student-t distribution to historical returns.

        Args:
            returns: Daily returns series

        Returns:
            Dictionary with fitted parameters
        """

        # Clean returns (remove extreme outliers for robust fitting)
        clean_returns = returns.dropna()

        if len(clean_returns) < 30:
            raise ValueError("Too few observations for heavy-tail fitting (need ≥30)")

        # Remove extreme outliers (>6 sigma) that are likely data errors
        mean_ret = clean_returns.mean()
        std_ret = clean_returns.std()
        clean_returns = clean_returns[
            (clean_returns >= mean_ret - 6*std_ret) &
            (clean_returns <= mean_ret + 6*std_ret)
        ]

        # Method of moments estimation for Student-t
        # Uses robust estimators
        sample_mean = clean_returns.mean()
        sample_var = clean_returns.var()
        sample_kurtosis = clean_returns.kurtosis()

        # Estimate degrees of freedom from excess kurtosis
        # For Student-t: excess_kurtosis = 6/(ν-4) where ν = degrees of freedom
        if sample_kurtosis > 0.5:  # Significant heavy tails
            # Solve: sample_kurtosis = 6/(ν-4) for ν
            estimated_dof = 4 + 6/sample_kurtosis
            estimated_dof = max(4.1, min(estimated_dof, 30))  # Bound between 4 and 30
        else:
            estimated_dof = 30  # Close to normal

        # Scale parameter for Student-t
        # Var = σ²ν/(ν-2) so σ² = Var(ν-2)/ν
        if estimated_dof > 2:
            scale_squared = sample_var * (estimated_dof - 2) / estimated_dof
            scale_param = np.sqrt(max(scale_squared, 1e-8))
        else:
            scale_param = np.sqrt(sample_var)

        fitted_params = {
            'distribution': 'student_t',
            'location': sample_mean,          # μ parameter
            'scale': scale_param,             # σ parameter
            'degrees_of_freedom': estimated_dof,  # ν parameter
            'sample_size': len(clean_returns),
            'raw_kurtosis': sample_kurtosis,
            'raw_skewness': clean_returns.skew()
        }

        return fitted_params

    def fit_extreme_value_theory(self, returns: pd.Series, threshold_percentile: float = None) -> Dict:
        """
        Fit a Generalized Pareto Distribution (GPD) to tail extremes
        using a peaks-over-threshold approach.
        """

        if threshold_percentile is None:
            threshold_percentile = self.evt_threshold_percentile

        clean_returns = returns.dropna()

        # Define thresholds for extremes (both positive and negative tails)
        threshold_high = clean_returns.quantile(threshold_percentile)
        threshold_low = clean_returns.quantile(1 - threshold_percentile)

        # Extract exceedances over the thresholds
        high_exceedances = clean_returns[clean_returns > threshold_high] - threshold_high
        low_exceedances = threshold_low - clean_returns[clean_returns < threshold_low]

        # Fit GPD to both high and low tails
        evt_params = {}

        for tail_name, exceedances, threshold in [
            ('upper_tail', high_exceedances, threshold_high),
            ('lower_tail', low_exceedances, threshold_low)
        ]:

            if len(exceedances) < 10:  # Too few extreme events
                evt_params[tail_name] = {
                    'shape': 0.1,  # Mild heavy tail default
                    'scale': exceedances.std() if len(exceedances) > 0 else 0.01,
                    'threshold': threshold,
                    'n_exceedances': len(exceedances),
                    'exceedance_rate': len(exceedances) / len(clean_returns)
                }
                continue

            # Method of moments for GPD estimation
            sample_mean = exceedances.mean()
            sample_var = exceedances.var()

            # GPD: E[X] = β/(1-ξ), Var[X] = β²/((1-ξ)²(1-2ξ))
            # Solve for ξ (shape) and β (scale)
            if sample_var > 0 and sample_mean > 0:
                # Moment ratio method
                gamma = sample_mean**2 / sample_var

                if gamma < 1:  # Heavy tail case
                    shape_xi = (gamma - 1) / 2
                    scale_beta = sample_mean * (1 - shape_xi)
                else:  # Light tail, use different estimator
                    shape_xi = 0.1  # Mild heavy tail
                    scale_beta = sample_mean * 0.9

                # Bound parameters for stability
                shape_xi = max(-0.5, min(shape_xi, 0.5))
                scale_beta = max(scale_beta, 1e-6)
            else:
                shape_xi = 0.1
                scale_beta = 0.01

            evt_params[tail_name] = {
                'shape': shape_xi,
                'scale': scale_beta,
                'threshold': threshold,
                'n_exceedances': len(exceedances),
                'exceedance_rate': len(exceedances) / len(clean_returns),
                'sample_mean': sample_mean,
                'sample_var': sample_var
            }

        return evt_params

    def calculate_tail_risk_metrics(self,
                                   returns: pd.Series,
                                   confidence_level: float = 0.95,
                                   time_horizon_days: int = 21) -> TailRiskMetrics:
        """
        Calculate comprehensive tail risk metrics.
        """

        # Fit distributions
        student_t_params = self.fit_heavy_tail_distribution(returns)
        evt_params = self.fit_extreme_value_theory(returns)

        # Parameters
        mu = student_t_params['location']
        sigma = student_t_params['scale']
        dof = student_t_params['degrees_of_freedom']

        # Scale for the time horizon (assumes i.i.d. returns)
        horizon_mu = mu * time_horizon_days
        horizon_sigma = sigma * np.sqrt(time_horizon_days)

        # VaR calculations
        alpha = 1 - confidence_level

        # Normal VaR (for comparison)
        from math import sqrt
        var_normal = horizon_mu + horizon_sigma * self._normal_ppf(alpha)

        # Student-t VaR
        var_student_t = horizon_mu + horizon_sigma * self._student_t_ppf(alpha, dof)

        # CVaR (Expected Shortfall) for Student-t
        # CVaR = E[X | X ≤ VaR] for the lower tail
        cvar_student_t = self._calculate_cvar_student_t(horizon_mu, horizon_sigma, dof, alpha)

        # EVT-based VaR (focus on the lower tail for losses)
        evt_var = self._calculate_evt_var(returns, evt_params, confidence_level, time_horizon_days)

        # EVT return level (expected extreme event)
        evt_return_level = self._calculate_evt_return_level(evt_params, time_horizon_days)

        # Tail risk multiplier
        tail_risk_multiplier = abs(var_student_t / var_normal) if var_normal != 0 else 1.0

        # Extreme event probability (beyond 3-sigma equivalent)
        extreme_threshold = -3 * horizon_sigma  # 3-sigma loss event
        extreme_prob = self._student_t_cdf(extreme_threshold, horizon_mu, horizon_sigma, dof)

        return TailRiskMetrics(
            confidence_level=confidence_level,
            time_horizon_days=time_horizon_days,
            var_normal=var_normal,
            var_student_t=var_student_t,
            cvar_student_t=cvar_student_t,
            evt_var=evt_var,
            evt_return_level=evt_return_level,
            degrees_of_freedom=dof,
            tail_index=evt_params.get('lower_tail', {}).get('shape', 0.1),
            tail_risk_multiplier=tail_risk_multiplier,
            extreme_event_probability=extreme_prob
        )

    def monte_carlo_simulation(self,
                              expected_return: float,
                              volatility: float,
                              tail_parameters: Dict,
                              time_horizon_months: int = 12,
                              n_simulations: int = None) -> MonteCarloResults:
        """
        Monte Carlo simulation using a heavy-tail Student-t distribution.

        Args:
            expected_return: Annual expected return
            volatility: Annual volatility
            tail_parameters: Student-t parameters from fitting
            time_horizon_months: Simulation horizon
            n_simulations: Number of paths
        """

        if n_simulations is None:
            n_simulations = self.mc_simulations

        # Convert to monthly parameters
        monthly_return = expected_return / 12
        monthly_vol = volatility / np.sqrt(12)
        dof = tail_parameters.get('degrees_of_freedom', 10)

        # Generate heavy-tail random returns
        np.random.seed(42)  # For reproducibility

        # Student-t innovations
        t_innovations = np.random.standard_t(dof, size=(n_simulations, time_horizon_months))

        # Scale and centre
        monthly_returns = monthly_return + monthly_vol * t_innovations

        # Compound returns over the horizon
        cumulative_returns = np.prod(1 + monthly_returns, axis=1) - 1

        # Calculate probability metrics
        prob_positive = np.mean(cumulative_returns > 0)
        prob_plus_10 = np.mean(cumulative_returns > 0.10)
        prob_plus_20 = np.mean(cumulative_returns > 0.20)
        prob_plus_30 = np.mean(cumulative_returns > 0.30)

        prob_minus_10 = np.mean(cumulative_returns < -0.10)
        prob_minus_20 = np.mean(cumulative_returns < -0.20)
        prob_minus_30 = np.mean(cumulative_returns < -0.30)

        # Distribution statistics
        mean_return = np.mean(cumulative_returns)
        median_return = np.median(cumulative_returns)
        std_return = np.std(cumulative_returns)

        # Calculate skewness and kurtosis manually (avoid SciPy dependency)
        skewness = np.mean(((cumulative_returns - mean_return) / std_return)**3)
        kurtosis = np.mean(((cumulative_returns - mean_return) / std_return)**4) - 3  # Excess kurtosis

        # Percentiles
        percentiles = np.percentile(cumulative_returns, [1, 5, 95, 99])

        return MonteCarloResults(
            n_simulations=n_simulations,
            time_horizon_months=time_horizon_months,
            prob_positive=prob_positive,
            prob_plus_10=prob_plus_10,
            prob_plus_20=prob_plus_20,
            prob_plus_30=prob_plus_30,
            prob_minus_10=prob_minus_10,
            prob_minus_20=prob_minus_20,
            prob_minus_30=prob_minus_30,
            mean_return=mean_return,
            median_return=median_return,
            std_return=std_return,
            skewness=skewness,
            kurtosis=kurtosis,
            percentile_1=percentiles[0],
            percentile_5=percentiles[1],
            percentile_95=percentiles[2],
            percentile_99=percentiles[3]
        )

    # Helper methods for statistical functions (avoid SciPy dependency)
    def _normal_ppf(self, p: float) -> float:
        """Approximate the normal percent point function (quantile)."""
        # Beasley-Springer-Moro approximation
        if p <= 0 or p >= 1:
            return 0.0

        if p < 0.5:
            sign = -1
            p = 1 - p
        else:
            sign = 1

        if p == 0.5:
            return 0.0

        # Rational approximation
        t = np.sqrt(-2 * np.log(1 - p))

        # Coefficients
        c = [2.515517, 0.802853, 0.010328]
        d = [1.432788, 0.189269, 0.001308]

        numerator = c[0] + c[1]*t + c[2]*t**2
        denominator = 1 + d[0]*t + d[1]*t**2 + d[2]*t**3

        return sign * (t - numerator/denominator)

    def _student_t_ppf(self, p: float, dof: float) -> float:
        """Approximate the Student-t percent point function."""
        # For high DoF, approach normal
        if dof >= 30:
            return self._normal_ppf(p)

        # Simple approximation for Student-t quantiles
        z = self._normal_ppf(p)

        # Cornish-Fisher expansion approximation
        correction = (z**3 + z) / (4 * dof) + (5*z**5 + 16*z**3 + 3*z) / (96 * dof**2)

        return z + correction

    def _student_t_cdf(self, x: float, mu: float, sigma: float, dof: float) -> float:
        """Approximate the Student-t cumulative distribution function."""
        standardized = (x - mu) / sigma

        # For high DoF, use the normal approximation
        if dof >= 30:
            return self._normal_cdf(standardized)

        # Simplified approximation for the Student-t CDF
        # Use the normal CDF with a correction
        normal_cdf = self._normal_cdf(standardized)

        # Apply correction based on degrees of freedom
        if abs(standardized) < 3:  # Normal range
            return normal_cdf
        else:  # Tail region - heavier tails
            if standardized < 0:
                return normal_cdf * (1 + 1/dof)  # Heavier left tail
            else:
                return 1 - (1 - normal_cdf) * (1 + 1/dof)  # Heavier right tail

    def _normal_cdf(self, x: float) -> float:
        """Approximate the normal cumulative distribution function."""
        # Abramowitz and Stegun approximation
        sign = 1 if x >= 0 else -1
        x = abs(x)

        # Constants
        a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
        p = 0.3275911

        t = 1.0 / (1.0 + p * x)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)

        return 0.5 * (1.0 + sign * y)

    def _calculate_cvar_student_t(self, mu: float, sigma: float, dof: float, alpha: float) -> float:
        """Calculate Conditional VaR for a Student-t distribution."""
        var = mu + sigma * self._student_t_ppf(alpha, dof)

        # CVaR approximation for Student-t
        # Simplified formula based on VaR
        tail_expectation_ratio = (dof + var**2/sigma**2) / ((dof - 1) * (1 - alpha))
        cvar = var * tail_expectation_ratio

        return cvar

    def _calculate_evt_var(self, returns: pd.Series, evt_params: Dict, confidence: float, horizon: int) -> float:
        """Calculate VaR using EVT (GPD tail estimation)"""

        lower_tail = evt_params.get('lower_tail', {})
        if not lower_tail:
            return 0.0

        threshold = lower_tail['threshold']
        shape = lower_tail['shape']
        scale = lower_tail['scale']
        exceedance_rate = lower_tail['exceedance_rate']

        # EVT VaR formula
        p = (1 - confidence) / exceedance_rate

        if shape != 0:
            evt_quantile = scale / shape * ((p)**(-shape) - 1)
        else:
            evt_quantile = scale * np.log(p)

        evt_var = threshold - evt_quantile

        # Scale for the time horizon
        return evt_var * np.sqrt(horizon)

    def _calculate_evt_return_level(self, evt_params: Dict, horizon: int) -> float:
        """Calculate expected return level from EVT."""

        upper_tail = evt_params.get('upper_tail', {})
        if not upper_tail:
            return 0.0

        threshold = upper_tail['threshold']
        shape = upper_tail['shape']
        scale = upper_tail['scale']

        # Expected return level for extreme positive events
        if shape < 1:
            expected_excess = scale / (1 - shape)
            return_level = threshold + expected_excess
        else:
            return_level = threshold + scale  # Bounded case

        return return_level * np.sqrt(horizon)
