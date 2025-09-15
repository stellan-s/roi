import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum

class SignalType(Enum):
    TREND = "trend"
    MOMENTUM = "momentum"
    SENTIMENT = "sentiment"

@dataclass
class SignalPrior:
    """Bayesian prior for a signal"""
    mean_effectiveness: float  # Prior belief på signal effectiveness (0-1)
    confidence: float         # Prior precision (higher = more confident)
    predictive_power: float   # Prior för predictive power

@dataclass
class SignalPosterior:
    """Updated beliefs efter observationer"""
    mean_effectiveness: float
    std_effectiveness: float
    predictive_power: float
    confidence_interval: Tuple[float, float]
    n_observations: int

@dataclass
class SignalOutput:
    """Output från Bayesian engine"""
    expected_return: float     # E[r]
    prob_positive: float       # Pr(↑)
    confidence_lower: float    # Lower CI bound
    confidence_upper: float    # Upper CI bound
    signal_weights: Dict[SignalType, float]
    uncertainty: float         # Overall uncertainty measure

class BayesianSignalEngine:
    """
    Bayesiansk signalviktning som kombinerar trend/momentum/sentiment
    för att producera E[r] och Pr(↑) med uncertainty quantification
    """

    def __init__(self, config: Optional[Dict] = None):
        # Configurable priors eller defaults
        if config and 'bayesian' in config and 'priors' in config['bayesian']:
            prior_config = config['bayesian']['priors']
            trend_eff = prior_config.get('trend_effectiveness', 0.55)
            momentum_eff = prior_config.get('momentum_effectiveness', 0.58)
            sentiment_eff = prior_config.get('sentiment_effectiveness', 0.52)
        else:
            # Default values
            trend_eff = 0.55
            momentum_eff = 0.58
            sentiment_eff = 0.52

        # Informativa priors baserat på finansteori och config
        self.priors = {
            SignalType.TREND: SignalPrior(
                mean_effectiveness=trend_eff,   # Configurable
                confidence=10.0,                # Relativt säker på detta
                predictive_power=0.4            # Moderat predictive power
            ),
            SignalType.MOMENTUM: SignalPrior(
                mean_effectiveness=momentum_eff, # Configurable
                confidence=15.0,                 # Högre confidence
                predictive_power=0.6             # Stark predictive power
            ),
            SignalType.SENTIMENT: SignalPrior(
                mean_effectiveness=sentiment_eff, # Configurable
                confidence=5.0,                   # Lägre confidence
                predictive_power=0.3              # Svagare predictive power
            )
        }

        # Posteriors uppdateras över tid
        self.posteriors: Dict[SignalType, SignalPosterior] = {}

        # Performance tracking för Bayesian updates
        self.signal_history: List[Dict] = []

    def update_beliefs(self,
                      signal_values: Dict[SignalType, float],
                      actual_return: float,
                      time_horizon_days: int = 21) -> None:
        """
        Bayesiansk uppdatering av beliefs baserat på faktisk performance
        """
        # Lagra observation för framtida analysis
        self.signal_history.append({
            'signals': signal_values.copy(),
            'actual_return': actual_return,
            'horizon_days': time_horizon_days,
            'timestamp': pd.Timestamp.now()
        })

        # Uppdatera posteriors för varje signal
        for signal_type, signal_value in signal_values.items():
            self._update_signal_posterior(signal_type, signal_value, actual_return)

    def _update_signal_posterior(self,
                               signal_type: SignalType,
                               signal_value: float,
                               actual_return: float) -> None:
        """Uppdatera posterior för en specifik signal"""
        prior = self.priors[signal_type]

        # Likelihood: hur väl signalen predicerade return
        # Signal "strength" * actual return correlation
        signal_strength = abs(signal_value)  # 0-1 för normaliserade signals
        prediction_accuracy = 1.0 if (signal_value > 0) == (actual_return > 0) else 0.0

        # Bayesian update med Beta-Binomial conjugate prior
        if signal_type not in self.posteriors:
            # Initialize posterior från prior
            alpha_prior = prior.mean_effectiveness * prior.confidence
            beta_prior = (1 - prior.mean_effectiveness) * prior.confidence

            self.posteriors[signal_type] = SignalPosterior(
                mean_effectiveness=prior.mean_effectiveness,
                std_effectiveness=np.sqrt(alpha_prior * beta_prior /
                                       ((alpha_prior + beta_prior)**2 * (alpha_prior + beta_prior + 1))),
                predictive_power=prior.predictive_power,
                confidence_interval=(prior.mean_effectiveness - 0.1, prior.mean_effectiveness + 0.1),
                n_observations=0
            )

        posterior = self.posteriors[signal_type]

        # Update med ny observation (weighted by signal strength)
        effective_obs = signal_strength  # Stronger signals get more weight
        alpha_post = prior.mean_effectiveness * prior.confidence + prediction_accuracy * effective_obs
        beta_post = (1 - prior.mean_effectiveness) * prior.confidence + (1 - prediction_accuracy) * effective_obs

        # Uppdaterade posterior parameters
        new_mean = alpha_post / (alpha_post + beta_post)
        new_variance = (alpha_post * beta_post) / ((alpha_post + beta_post)**2 * (alpha_post + beta_post + 1))
        new_std = np.sqrt(new_variance)

        # 95% credible interval (approximation utan scipy)
        # Använd normal approximation för beta distribution
        ci_lower = max(0.0, new_mean - 1.96 * new_std)
        ci_upper = min(1.0, new_mean + 1.96 * new_std)

        self.posteriors[signal_type] = SignalPosterior(
            mean_effectiveness=new_mean,
            std_effectiveness=new_std,
            predictive_power=prior.predictive_power,  # Could also be updated
            confidence_interval=(ci_lower, ci_upper),
            n_observations=posterior.n_observations + 1
        )

    def combine_signals(self,
                       signal_values: Dict[SignalType, float],
                       regime_adjustment: float = 1.0) -> SignalOutput:
        """
        Kombinera signals med Bayesian weighting för att få E[r] och Pr(↑)
        """
        # Get current posterior weights (eller fall back till priors)
        weights = {}
        total_confidence = 0

        for signal_type in signal_values.keys():
            if signal_type in self.posteriors:
                posterior = self.posteriors[signal_type]
                # Weight = effectiveness * predictive_power * (1 / uncertainty)
                uncertainty = posterior.std_effectiveness
                weight = (posterior.mean_effectiveness *
                         posterior.predictive_power *
                         (1 / (1 + uncertainty)) *
                         regime_adjustment)
            else:
                # Använd prior om ingen posterior finns än
                prior = self.priors[signal_type]
                weight = (prior.mean_effectiveness *
                         prior.predictive_power *
                         regime_adjustment)

            weights[signal_type] = weight
            total_confidence += weight

        # Normalisera weights
        if total_confidence > 0:
            weights = {k: v/total_confidence for k, v in weights.items()}

        # Weighted combination av signals
        combined_signal = sum(weights[st] * signal_values[st] for st in signal_values.keys())

        # Convert till expected return och probability
        # Använd sigmoid-transformation för Pr(↑)
        prob_positive = 1 / (1 + np.exp(-3 * combined_signal))  # Sigmoid with scaling

        # Expected return baserat på signal strength och historical relationships
        base_return_annual = 0.08  # 8% baseline annual return assumption
        signal_multiplier = combined_signal * 2  # Signal kan ge +/- 200% av base
        expected_return = base_return_annual * (1 + signal_multiplier) / 252  # Daily return

        # Uncertainty quantification
        weight_uncertainty = np.std(list(weights.values())) if len(weights) > 1 else 0.1
        signal_uncertainty = np.std(list(signal_values.values())) if len(signal_values) > 1 else 0.1
        overall_uncertainty = (weight_uncertainty + signal_uncertainty) / 2

        # Confidence intervals
        ci_width = overall_uncertainty * 2  # 2-sigma approximation
        confidence_lower = expected_return - ci_width
        confidence_upper = expected_return + ci_width

        return SignalOutput(
            expected_return=expected_return,
            prob_positive=prob_positive,
            confidence_lower=confidence_lower,
            confidence_upper=confidence_upper,
            signal_weights=weights,
            uncertainty=overall_uncertainty
        )

    def get_signal_diagnostics(self) -> pd.DataFrame:
        """Diagnostics för signal performance och beliefs"""
        diagnostics = []

        for signal_type, posterior in self.posteriors.items():
            prior = self.priors[signal_type]

            diagnostics.append({
                'signal_type': signal_type.value,
                'prior_effectiveness': prior.mean_effectiveness,
                'posterior_effectiveness': posterior.mean_effectiveness,
                'effectiveness_std': posterior.std_effectiveness,
                'ci_lower': posterior.confidence_interval[0],
                'ci_upper': posterior.confidence_interval[1],
                'n_observations': posterior.n_observations,
                'prior_shift': posterior.mean_effectiveness - prior.mean_effectiveness
            })

        return pd.DataFrame(diagnostics)