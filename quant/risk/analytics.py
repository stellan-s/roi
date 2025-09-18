import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from .heavy_tail import HeavyTailRiskModel, TailRiskMetrics, MonteCarloResults

@dataclass
class PortfolioRiskProfile:
    """Comprehensive risk profile for a portfolio position."""
    ticker: str
    expected_return_annual: float
    volatility_annual: float

    # Heavy-tail characteristics
    tail_risk_metrics: TailRiskMetrics
    monte_carlo_12m: MonteCarloResults

    # Risk-adjusted metrics
    sharpe_ratio: float
    tail_risk_adjusted_return: float  # E[r] adjusted for tail risk
    risk_contribution: float          # Contribution to portfolio risk

    # Scenario probabilities (12 months)
    prob_loss_10_percent: float
    prob_loss_20_percent: float
    prob_gain_10_percent: float
    prob_gain_20_percent: float
    prob_gain_30_percent: float

    # Extreme scenarios
    worst_case_1_percent: float       # 1st percentile return
    best_case_99_percent: float       # 99th percentile return

@dataclass
class MarketStressScenario:
    """Stress test scenario definition."""
    name: str
    description: str
    market_shock_size: float          # Size of market shock (e.g., -0.20 for -20%)
    volatility_multiplier: float      # How much vol increases (e.g., 2.0)
    correlation_increase: float       # How much correlations increase (0.2 → 0.8)
    duration_days: int                # How long shock lasts

# Predefined stress scenarios
STRESS_SCENARIOS = {
    "black_monday": MarketStressScenario(
        name="Black Monday",
        description="Severe market crash with a -20% drop and volatility spike",
        market_shock_size=-0.20,
        volatility_multiplier=3.0,
        correlation_increase=0.6,
        duration_days=5
    ),
    "covid_crash": MarketStressScenario(
        name="COVID-19 Crash",
        description="Pandemic-style crash with extended volatility",
        market_shock_size=-0.35,
        volatility_multiplier=2.5,
        correlation_increase=0.5,
        duration_days=30
    ),
    "dot_com_burst": MarketStressScenario(
        name="Dot-Com Burst",
        description="Tech bubble burst with a slow recovery",
        market_shock_size=-0.50,
        volatility_multiplier=2.0,
        correlation_increase=0.4,
        duration_days=252  # 1 year
    ),
    "regime_shift": MarketStressScenario(
        name="Regime Shift",
        description="Structural change in market regime",
        market_shock_size=-0.15,
        volatility_multiplier=1.8,
        correlation_increase=0.3,
        duration_days=126  # 6 months
    )
}

class RiskAnalytics:
    """
    Advanced risk analytics incorporating heavy-tail modelling.

    Capabilities:
    1. Portfolio-level tail risk assessment
    2. Stress testing against historical scenarios
    3. Risk budgeting and contribution analysis
    4. Monte Carlo scenario analysis
    5. Risk-adjusted position sizing
    """

    def __init__(self, config: Optional[Dict] = None):
        self.heavy_tail_model = HeavyTailRiskModel(config)

        # Risk analytics configuration
        if config and 'risk_analytics' in config:
            risk_config = config['risk_analytics']
            self.stress_test_scenarios = risk_config.get('stress_scenarios', list(STRESS_SCENARIOS.keys()))
            self.risk_free_rate = risk_config.get('risk_free_rate', 0.02)  # 2% annual
            self.target_volatility = risk_config.get('target_portfolio_volatility', 0.15)  # 15% annual
        else:
            self.stress_test_scenarios = list(STRESS_SCENARIOS.keys())
            self.risk_free_rate = 0.02
            self.target_volatility = 0.15

    def analyze_position_risk(self,
                             ticker: str,
                             price_history: pd.Series,
                             expected_return: float,
                             time_horizon_months: int = 12) -> PortfolioRiskProfile:
        """
        Comprehensive risk analysis for a single position.

        Args:
            ticker: Stock ticker
            price_history: Historical price series
            expected_return: Annual expected return from the Bayesian model
            time_horizon_months: Analysis horizon

        Returns:
            Complete risk profile
        """

        # Calculate historical returns
        returns = price_history.pct_change().dropna()

        if len(returns) < 30:
            raise ValueError(f"Too few historical observations for {ticker} (need ≥30)")

        # Historical volatility
        annual_volatility = returns.std() * np.sqrt(252)

        # Fit heavy-tail distribution
        tail_params = self.heavy_tail_model.fit_heavy_tail_distribution(returns)

        # Calculate tail risk metrics (21-day horizon for comparability)
        tail_risk_21d = self.heavy_tail_model.calculate_tail_risk_metrics(
            returns, confidence_level=0.95, time_horizon_days=21
        )

        # Monte Carlo simulation for a 12-month horizon
        mc_results = self.heavy_tail_model.monte_carlo_simulation(
            expected_return=expected_return,
            volatility=annual_volatility,
            tail_parameters=tail_params,
            time_horizon_months=time_horizon_months
        )

        # Risk-adjusted metrics
        sharpe_ratio = (expected_return - self.risk_free_rate) / annual_volatility if annual_volatility > 0 else 0

        # Tail risk adjustment - penalise heavy-tail risk
        tail_risk_penalty = tail_risk_21d.tail_risk_multiplier - 1.0  # Additional risk from heavy tails
        tail_risk_adjusted_return = expected_return - (tail_risk_penalty * annual_volatility)

        # Risk contribution (approximation for the portfolio context)
        risk_contribution = annual_volatility  # Will be properly calculated at portfolio level

        return PortfolioRiskProfile(
            ticker=ticker,
            expected_return_annual=expected_return,
            volatility_annual=annual_volatility,
            tail_risk_metrics=tail_risk_21d,
            monte_carlo_12m=mc_results,
            sharpe_ratio=sharpe_ratio,
            tail_risk_adjusted_return=tail_risk_adjusted_return,
            risk_contribution=risk_contribution,
            prob_loss_10_percent=mc_results.prob_minus_10,
            prob_loss_20_percent=mc_results.prob_minus_20,
            prob_gain_10_percent=mc_results.prob_plus_10,
            prob_gain_20_percent=mc_results.prob_plus_20,
            prob_gain_30_percent=mc_results.prob_plus_30,
            worst_case_1_percent=mc_results.percentile_1,
            best_case_99_percent=mc_results.percentile_99
        )

    def stress_test_portfolio(self,
                             portfolio_weights: Dict[str, float],
                             risk_profiles: Dict[str, PortfolioRiskProfile],
                             scenarios: List[str] = None) -> Dict[str, Dict]:
        """
        Stress test portfolio mot predefined scenarios

        Args:
            portfolio_weights: Dict[ticker, weight]
            risk_profiles: Dict[ticker, PortfolioRiskProfile]
            scenarios: List av scenario names to test

        Returns:
            Dict med stress test results per scenario
        """

        if scenarios is None:
            scenarios = self.stress_test_scenarios

        stress_results = {}

        for scenario_name in scenarios:
            if scenario_name not in STRESS_SCENARIOS:
                continue

            scenario = STRESS_SCENARIOS[scenario_name]
            stress_results[scenario_name] = self._apply_stress_scenario(
                portfolio_weights, risk_profiles, scenario
            )

        return stress_results

    def _apply_stress_scenario(self,
                              portfolio_weights: Dict[str, float],
                              risk_profiles: Dict[str, PortfolioRiskProfile],
                              scenario: MarketStressScenario) -> Dict:
        """Apply a stress scenario to the portfolio."""

        portfolio_loss = 0.0
        position_impacts = {}

        for ticker, weight in portfolio_weights.items():
            if weight == 0 or ticker not in risk_profiles:
                continue

            risk_profile = risk_profiles[ticker]

            # Apply market shock
            base_loss = scenario.market_shock_size * weight

            # Apply volatility shock (affects tail risk)
            vol_shock = risk_profile.volatility_annual * scenario.volatility_multiplier
            tail_shock = vol_shock * np.sqrt(scenario.duration_days / 252)  # Scale for duration

            # Heavy-tail amplification under stress
            tail_multiplier = risk_profile.tail_risk_metrics.tail_risk_multiplier
            stress_amplified_loss = base_loss * (1 + tail_multiplier * 0.5)  # 50% amplification

            total_position_loss = stress_amplified_loss - tail_shock * 0.5  # Additional tail risk

            position_impacts[ticker] = {
                'base_loss': base_loss,
                'tail_amplified_loss': stress_amplified_loss,
                'total_loss': total_position_loss,
                'contribution_to_portfolio_loss': total_position_loss
            }

            portfolio_loss += total_position_loss

        return {
            'scenario': scenario.name,
            'total_portfolio_loss': portfolio_loss,
            'position_impacts': position_impacts,
            'loss_as_percentage': portfolio_loss,
            'description': scenario.description,
            'duration_days': scenario.duration_days
        }

    def calculate_risk_budgets(self,
                              risk_profiles: Dict[str, PortfolioRiskProfile],
                              target_portfolio_vol: float = None) -> Dict[str, Dict]:
        """
        Calculate optimal position sizes based on risk budgeting.

        Args:
            risk_profiles: Risk profiles for all candidates
            target_portfolio_vol: Target portfolio volatility

        Returns:
            Dict containing recommended weights and risk budgets
        """

        if target_portfolio_vol is None:
            target_portfolio_vol = self.target_volatility

        # Equal risk contribution approach (risk parity inspired)
        risk_budgets = {}

        total_inv_vol = 0.0
        for ticker, profile in risk_profiles.items():
            inv_vol = 1.0 / profile.volatility_annual if profile.volatility_annual > 0 else 0
            total_inv_vol += inv_vol

        for ticker, profile in risk_profiles.items():
            if profile.volatility_annual > 0:
                # Base weight from inverse volatility
                base_weight = (1.0 / profile.volatility_annual) / total_inv_vol

                # Tail risk adjustment
                tail_adjustment = 1.0 / profile.tail_risk_metrics.tail_risk_multiplier

                # Sharpe ratio adjustment
                sharpe_adjustment = max(0.1, profile.sharpe_ratio) / 1.0  # Normalize around 1.0

                # Combined adjustment
                adjusted_weight = base_weight * tail_adjustment * sharpe_adjustment

                risk_budgets[ticker] = {
                    'recommended_weight': adjusted_weight,
                    'risk_contribution': adjusted_weight * profile.volatility_annual,
                    'tail_risk_multiplier': profile.tail_risk_metrics.tail_risk_multiplier,
                    'sharpe_ratio': profile.sharpe_ratio,
                    'expected_return': profile.expected_return_annual
                }
            else:
                risk_budgets[ticker] = {
                    'recommended_weight': 0.0,
                    'risk_contribution': 0.0,
                    'tail_risk_multiplier': 1.0,
                    'sharpe_ratio': 0.0,
                    'expected_return': profile.expected_return_annual
                }

        # Normalize weights
        total_weight = sum(rb['recommended_weight'] for rb in risk_budgets.values())
        if total_weight > 0:
            for ticker in risk_budgets:
                risk_budgets[ticker]['recommended_weight'] /= total_weight

        return risk_budgets

    def generate_risk_summary(self,
                             portfolio_weights: Dict[str, float],
                             risk_profiles: Dict[str, PortfolioRiskProfile]) -> Dict:
        """Generate comprehensive portfolio risk summary"""

        if not portfolio_weights or not risk_profiles:
            return {'error': 'No portfolio data provided'}

        # Portfolio level metrics
        portfolio_expected_return = sum(
            weight * risk_profiles[ticker].expected_return_annual
            for ticker, weight in portfolio_weights.items()
            if ticker in risk_profiles and weight > 0
        )

        # Simple portfolio volatility (assumes zero correlation for simplicity)
        portfolio_variance = sum(
            (weight * risk_profiles[ticker].volatility_annual)**2
            for ticker, weight in portfolio_weights.items()
            if ticker in risk_profiles and weight > 0
        )
        portfolio_volatility = np.sqrt(portfolio_variance)

        # Weighted average tail risk
        weighted_tail_multiplier = sum(
            weight * risk_profiles[ticker].tail_risk_metrics.tail_risk_multiplier
            for ticker, weight in portfolio_weights.items()
            if ticker in risk_profiles and weight > 0
        )

        # Concentration risk
        max_position = max(portfolio_weights.values()) if portfolio_weights else 0
        num_positions = sum(1 for w in portfolio_weights.values() if w > 0.01)  # >1% positions

        # Monte Carlo aggregated probabilities (simplified)
        weighted_prob_loss_20 = sum(
            weight * risk_profiles[ticker].prob_loss_20_percent
            for ticker, weight in portfolio_weights.items()
            if ticker in risk_profiles and weight > 0
        )

        weighted_prob_gain_20 = sum(
            weight * risk_profiles[ticker].prob_gain_20_percent
            for ticker, weight in portfolio_weights.items()
            if ticker in risk_profiles and weight > 0
        )

        return {
            'portfolio_expected_return_annual': portfolio_expected_return,
            'portfolio_volatility_annual': portfolio_volatility,
            'portfolio_sharpe_ratio': (portfolio_expected_return - self.risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0,
            'weighted_tail_risk_multiplier': weighted_tail_multiplier,
            'concentration_risk': {
                'max_position_weight': max_position,
                'number_of_positions': num_positions,
                'concentration_score': max_position * (1 / max(1, num_positions))  # Higher = more concentrated
            },
            'scenario_probabilities_12m': {
                'prob_loss_20_percent': weighted_prob_loss_20,
                'prob_gain_20_percent': weighted_prob_gain_20
            },
            'risk_assessment': self._assess_risk_level(portfolio_volatility, weighted_tail_multiplier, max_position)
        }

    def _assess_risk_level(self, volatility: float, tail_multiplier: float, concentration: float) -> str:
        """Assess overall portfolio risk level"""

        risk_score = 0

        # Volatility contribution
        if volatility > 0.25:
            risk_score += 3
        elif volatility > 0.20:
            risk_score += 2
        elif volatility > 0.15:
            risk_score += 1

        # Tail risk contribution
        if tail_multiplier > 2.0:
            risk_score += 3
        elif tail_multiplier > 1.5:
            risk_score += 2
        elif tail_multiplier > 1.2:
            risk_score += 1

        # Concentration contribution
        if concentration > 0.3:
            risk_score += 2
        elif concentration > 0.2:
            risk_score += 1

        # Risk level mapping
        if risk_score >= 6:
            return "HIGH RISK"
        elif risk_score >= 4:
            return "MODERATE-HIGH RISK"
        elif risk_score >= 2:
            return "MODERATE RISK"
        else:
            return "LOW-MODERATE RISK"
