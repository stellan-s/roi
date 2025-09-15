# Heavy-tail Risk Modeling

## Overview

The Heavy-tail Risk Modeling system provides sophisticated risk quantification that goes beyond traditional normal distribution assumptions. It uses Student-t distributions, Extreme Value Theory (EVT), and Monte Carlo simulation to capture realistic tail risk and extreme event probabilities.

## Why Heavy-tail Risk Matters

Traditional risk models assume normal distributions, which severely underestimate the probability of extreme events. Real financial markets exhibit:

- **Fat Tails**: Extreme events occur more frequently than normal distributions predict
- **Skewness**: Asymmetric return distributions
- **Volatility Clustering**: High volatility periods persist
- **Regime Changes**: Market structure shifts over time

## Core Components

### 1. Student-t Distribution Modeling

#### Theoretical Foundation
Student-t distributions have an additional parameter (degrees of freedom, ν) that controls tail thickness:
- **ν → ∞**: Approaches normal distribution
- **ν < 10**: Heavy tails (realistic for most stocks)
- **ν ≈ 3-6**: Extremely heavy tails (high-risk assets)

#### Implementation
```python
def fit_heavy_tail_distribution(self, returns: pd.Series) -> Dict:
    # Fit Student-t distribution using maximum likelihood
    params = stats.t.fit(returns)
    degrees_of_freedom = params[0]
    location = params[1]  # Mean return
    scale = params[2]     # Volatility parameter

    return {
        'degrees_of_freedom': degrees_of_freedom,
        'location': location,
        'scale': scale,
        'sample_size': len(returns)
    }
```

### 2. Extreme Value Theory (EVT)

#### Peak Over Threshold Method
EVT focuses specifically on tail events by modeling exceedances over high thresholds:

```python
def fit_extreme_value_theory(self, returns: pd.Series) -> Dict:
    # Upper tail (positive extreme returns)
    upper_threshold = returns.quantile(0.95)
    upper_exceedances = returns[returns > upper_threshold] - upper_threshold

    # Lower tail (negative extreme returns)
    lower_threshold = returns.quantile(0.05)
    lower_exceedances = lower_threshold - returns[returns < lower_threshold]

    # Fit Generalized Pareto Distribution to exceedances
    upper_params = self._fit_gpd(upper_exceedances)
    lower_params = self._fit_gpd(lower_exceedances)
```

#### Generalized Pareto Distribution (GPD)
The GPD models the distribution of exceedances with two parameters:
- **Shape (ξ)**: Tail heaviness
  - ξ > 0: Heavy tails (power-law decay)
  - ξ = 0: Exponential tails
  - ξ < 0: Bounded tails
- **Scale (β)**: Spread of exceedances

### 3. Monte Carlo Simulation

#### 12-Month Probability Scenarios
The system simulates thousands of potential 12-month return paths:

```python
def monte_carlo_simulation(self,
                         expected_return: float,
                         volatility: float,
                         tail_parameters: Dict,
                         time_horizon_months: int = 12,
                         n_simulations: int = 10000) -> MonteCarloResults:
```

#### Simulation Process
1. **Draw Random Numbers**: From fitted Student-t distribution
2. **Apply Drift**: Add expected return component
3. **Scale for Horizon**: Adjust for time period
4. **Calculate Statistics**: Percentiles, probabilities, moments

#### Key Outputs
```python
class MonteCarloResults:
    # Probability targets
    prob_positive: float      # P(return > 0%)
    prob_plus_10: float      # P(return > +10%)
    prob_plus_20: float      # P(return > +20%)
    prob_plus_30: float      # P(return > +30%)
    prob_minus_10: float     # P(return < -10%)
    prob_minus_20: float     # P(return < -20%)
    prob_minus_30: float     # P(return < -30%)

    # Extreme scenarios
    percentile_1: float      # Worst 1% outcome
    percentile_5: float      # Worst 5% outcome
    percentile_95: float     # Best 5% outcome
    percentile_99: float     # Best 1% outcome
```

## Risk Metrics

### Value at Risk (VaR)

#### Traditional vs Heavy-tail VaR
```python
class TailRiskMetrics:
    var_normal: float         # Normal distribution VaR
    var_student_t: float      # Student-t VaR
    var_evt: float           # EVT-based VaR
    tail_risk_multiplier: float # Heavy-tail vs normal ratio
```

#### Interpretation
- **VaR Normal**: Traditional risk measure (often too optimistic)
- **VaR Student-t**: More realistic for heavy-tail assets
- **Tail Risk Multiplier**: How much worse heavy-tail risk is (typically 1.5-3x)

### Conditional Value at Risk (CVaR)

CVaR measures the expected loss beyond the VaR threshold:
```python
cvar_student_t: float     # Expected loss in worst α% of cases
```

### Extreme Event Probabilities
```python
extreme_event_probability: float  # P(|return| > 3σ equivalent)
```

## Configuration

### Risk Modeling Settings

```yaml
risk_modeling:
  enabled: true
  confidence_levels: [0.95, 0.99, 0.999]    # VaR confidence levels
  time_horizons_days: [21, 63, 252]         # Analysis periods
  monte_carlo_simulations: 10000            # MC sample size
  evt_threshold_percentile: 0.95            # EVT threshold
  min_observations_for_tail_fit: 30         # Minimum data requirement

  tail_risk_thresholds:
    low: 0.4        # Green indicator threshold
    high: 0.7       # Red indicator threshold
```

### Risk Analytics Settings

```yaml
risk_analytics:
  stress_scenarios: ["black_monday", "covid_crash", "regime_shift"]
  risk_free_rate: 0.02                      # 2% annual
  target_portfolio_volatility: 0.15         # 15% target
  tail_risk_penalty: 0.02                   # Extra penalty for heavy tails
```

## Stress Testing Framework

### Predefined Scenarios

#### Black Monday Scenario
```python
black_monday = MarketStressScenario(
    name="Black Monday",
    description="Severe market crash med -20% fall och vol spike",
    market_shock_size=-0.20,         # -20% market shock
    volatility_multiplier=3.0,       # 3x volatility increase
    correlation_increase=0.6,        # Correlations rise to 0.6
    duration_days=5                  # 5-day shock period
)
```

#### COVID-19 Crash
```python
covid_crash = MarketStressScenario(
    name="COVID-19 Crash",
    description="Pandemic-style crash med extended volatility",
    market_shock_size=-0.35,         # -35% market shock
    volatility_multiplier=2.5,       # 2.5x volatility
    correlation_increase=0.5,        # Higher correlations
    duration_days=30                 # Extended 30-day shock
)
```

### Stress Test Application

```python
def _apply_stress_scenario(self,
                          portfolio_weights: Dict[str, float],
                          risk_profiles: Dict[str, PortfolioRiskProfile],
                          scenario: MarketStressScenario) -> Dict:

    for ticker, weight in portfolio_weights.items():
        # Apply market shock
        base_loss = scenario.market_shock_size * weight

        # Apply volatility shock
        vol_shock = risk_profile.volatility_annual * scenario.volatility_multiplier

        # Heavy-tail amplification under stress
        tail_multiplier = risk_profile.tail_risk_metrics.tail_risk_multiplier
        stress_amplified_loss = base_loss * (1 + tail_multiplier * 0.5)
```

## Portfolio Integration

### Risk-Adjusted Position Sizing

Heavy-tail risk influences position sizing through penalty factors:

```python
# Heavy-tail risk adjustment in portfolio management
tail_risk_penalty = 1.0 - (tail_risk_score * 0.3)  # Up to 30% reduction
tail_risk_penalty = max(0.5, tail_risk_penalty)    # Minimum 50% of original

risk_adjusted_return = expected_return * confidence * regime_stability * tail_risk_penalty
```

### Tail Risk Score Calculation

The system generates tail risk scores (0-1) for each position:

```python
def _calculate_tail_risk_score(self, row: pd.Series, signals: Dict) -> float:
    # Base tail risk from volatility proxy
    momentum_volatility = abs(signals.get(SignalType.MOMENTUM, 0.0))
    base_tail_risk = momentum_volatility * 0.3

    # Regime adjustment
    if regime == "bear":
        regime_multiplier = 1.5  # Higher tail risk in bear markets
    elif regime == "bull":
        regime_multiplier = 0.8  # Lower tail risk in bull markets

    # Signal uncertainty contribution
    uncertainty_contribution = uncertainty * 0.2

    return np.clip((base_tail_risk + uncertainty_contribution) * regime_multiplier, 0.0, 1.0)
```

## Advanced Risk Analytics

### Portfolio Risk Profile

Complete risk characterization for each position:

```python
class PortfolioRiskProfile:
    ticker: str
    expected_return_annual: float
    volatility_annual: float

    # Heavy-tail characteristics
    tail_risk_metrics: TailRiskMetrics
    monte_carlo_12m: MonteCarloResults

    # Risk-adjusted metrics
    sharpe_ratio: float
    tail_risk_adjusted_return: float
    risk_contribution: float

    # Scenario probabilities
    prob_loss_10_percent: float
    prob_loss_20_percent: float
    prob_gain_10_percent: float
    prob_gain_20_percent: float
    prob_gain_30_percent: float

    # Extreme scenarios
    worst_case_1_percent: float
    best_case_99_percent: float
```

### Risk Budgeting

Optimal position sizing based on risk contribution:

```python
def calculate_risk_budgets(self,
                          risk_profiles: Dict[str, PortfolioRiskProfile],
                          target_portfolio_vol: float = None) -> Dict[str, Dict]:

    # Equal risk contribution approach
    for ticker, profile in risk_profiles.items():
        # Base weight from inverse volatility
        base_weight = (1.0 / profile.volatility_annual) / total_inv_vol

        # Tail risk adjustment
        tail_adjustment = 1.0 / profile.tail_risk_metrics.tail_risk_multiplier

        # Sharpe ratio adjustment
        sharpe_adjustment = max(0.1, profile.sharpe_ratio) / 1.0

        # Combined optimal weight
        optimal_weight = base_weight * tail_adjustment * sharpe_adjustment
```

## Validation and Testing

### Model Validation

#### Backtesting VaR Models
- **Coverage Tests**: Do 5% VaR violations occur ~5% of the time?
- **Independence Tests**: Are violations clustered (indicating model failure)?
- **Magnitude Tests**: How large are the violations when they occur?

#### Heavy-tail Detection
```python
# Example: TSLA analysis shows heavy tails
student_t_params = {
    'degrees_of_freedom': 5.4,  # Heavy tails (< 10)
    'location': 0.003,          # Daily mean return
    'scale': 0.043              # Daily volatility
}

# Tail risk multiplier
normal_var_21d = -18.64%
student_t_var_21d = -23.81%
tail_risk_multiplier = 1.28x  # 28% worse than normal
```

## Example Usage

### Complete Risk Analysis

```python
from quant.risk.analytics import RiskAnalytics
from quant.risk.heavy_tail import HeavyTailRiskModel

# Initialize risk analytics
risk_analytics = RiskAnalytics(config)

# Analyze individual position
risk_profile = risk_analytics.analyze_position_risk(
    ticker="TSLA",
    price_history=price_series,
    expected_return=0.08,  # 8% annual
    time_horizon_months=12
)

# Portfolio stress testing
stress_results = risk_analytics.stress_test_portfolio(
    portfolio_weights={"TSLA": 0.3, "AAPL": 0.4, "MSFT": 0.3},
    risk_profiles=risk_profiles,
    scenarios=["black_monday", "covid_crash"]
)
```

### Risk Metrics Interpretation

```python
# Heavy-tail analysis results
print(f"Student-t DoF: {tail_params['degrees_of_freedom']:.1f}")
if tail_params['degrees_of_freedom'] < 10:
    print("⚠️ Heavy tails detected - extreme events more likely")

print(f"21-day VaR (95%): Normal={var_normal:.1%}, Student-t={var_student_t:.1%}")
print(f"Tail risk multiplier: {tail_risk_multiplier:.2f}x")

# Monte Carlo probabilities
print(f"12-month scenarios:")
print(f"  P(return > +20%): {mc_results.prob_plus_20:.1%}")
print(f"  P(return < -20%): {mc_results.prob_minus_20:.1%}")
print(f"  Worst 1%: {mc_results.percentile_1:.1%}")
```

## Key Advantages

1. **Realistic Risk Assessment**: Captures fat tails and extreme events
2. **Multiple Methodologies**: Student-t, EVT, and Monte Carlo complement each other
3. **Actionable Insights**: Probability-based scenarios for decision making
4. **Portfolio Integration**: Risk-aware position sizing and allocation
5. **Stress Testing**: Quantifies portfolio resilience under extreme conditions

## Implementation Details

### Core Files
- `quant/risk/heavy_tail.py` - Student-t and EVT implementation
- `quant/risk/analytics.py` - Portfolio risk analytics and stress testing

### Performance Characteristics
- **Computation Time**: ~1-2 seconds for full analysis per asset
- **Memory Usage**: Minimal (rolling calculations)
- **Accuracy**: Validated against historical extreme events
- **Scalability**: Linear with portfolio size

## Future Enhancements

- **Dynamic Tail Parameters**: Time-varying degrees of freedom
- **Copula Models**: Multi-asset tail dependence
- **Machine Learning**: Neural network tail risk prediction
- **Alternative Data**: Options-implied tail risk measures