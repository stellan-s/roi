# Bayesian Signal Engine

## Overview

The Bayesian Signal Engine is the core decision-making component of the ROI system. It combines multiple market signals using Bayesian inference to generate probabilistic predictions and trading decisions.

## Core Concepts

### Bayesian Inference for Trading

The system treats each signal (trend, momentum, sentiment) as evidence that influences our belief about future price movements. Instead of simple weighted averages, we use Bayesian updating to:

1. **Start with Prior Beliefs**: Initial assumptions about signal effectiveness
2. **Update with Evidence**: Combine signals probabilistically
3. **Quantify Uncertainty**: Provide confidence intervals
4. **Adapt Over Time**: Learn from market performance

### Signal Types

#### 1. Trend Signal
- **Source**: 200-day Simple Moving Average (SMA)
- **Interpretation**: Long-term directional momentum
- **Bayesian Prior**: 62% effectiveness (configurable)
- **Regime Sensitivity**: Enhanced in bear markets

#### 2. Momentum Signal
- **Source**: 252-day momentum ranking
- **Interpretation**: Relative strength vs universe
- **Bayesian Prior**: 68% effectiveness (strongest documented edge)
- **Regime Sensitivity**: Reduced in bear markets (frequent reversals)

#### 3. Sentiment Signal
- **Source**: Swedish news sentiment analysis
- **Interpretation**: Market psychology proxy
- **Bayesian Prior**: 58% effectiveness (higher noise)
- **Regime Sensitivity**: Enhanced in bear markets (pessimism bias)

## Mathematical Framework

### Prior Beliefs

Each signal has a prior belief about its effectiveness:

```python
class SignalPrior:
    effectiveness: float    # Base probability (0.5 = no edge, >0.5 = positive edge)
    confidence: float      # How certain we are (higher = less learning)
    observations: int      # Pseudo-observations for Bayesian updating
```

### Signal Combination

The engine combines signals using a Beta-Binomial conjugate prior approach:

```
P(positive | signals) = Σ(weighted_signal_posteriors) / total_weight
```

Where each signal posterior is updated based on:
- Signal strength (-1 to +1)
- Historical effectiveness
- Regime-specific adjustments
- Uncertainty penalties

### Output Generation

The engine produces structured output:

```python
class SignalOutput:
    expected_return: float         # E[r] daily expected return
    prob_positive: float          # Pr(↑) probability of positive movement
    confidence_lower: float       # Lower confidence bound
    confidence_upper: float       # Upper confidence bound
    uncertainty: float            # Overall uncertainty (0-1)
    signal_weights: Dict         # Individual signal contributions
```

## Configuration

### Bayesian Settings (settings.yaml)

```yaml
bayesian:
  time_horizon_days: 21
  decision_thresholds:
    buy_probability: 0.58      # Pr(↑) threshold for buy
    sell_probability: 0.40     # Pr(↑) threshold for sell
    min_expected_return: 0.0005 # Minimum E[r] for buy
    max_uncertainty: 0.35      # Maximum uncertainty for action

  priors:
    trend_effectiveness: 0.62   # SMA trend-following edge
    momentum_effectiveness: 0.68 # Momentum edge (strongest)
    sentiment_effectiveness: 0.58 # Sentiment edge (noisiest)
```

## Decision Logic

### Buy Conditions
A stock receives a "Buy" recommendation when:
1. `Pr(↑) ≥ buy_probability + uncertainty_penalty`
2. `E[r] ≥ min_expected_return`
3. `uncertainty ≤ max_uncertainty`

### Sell Conditions
A stock receives a "Sell" recommendation when:
1. `Pr(↑) ≤ sell_probability - uncertainty_penalty`
2. OR `E[r] ≤ -min_expected_return` with low uncertainty

### Hold Conditions
All other cases, including high uncertainty situations.

## Regime Integration

The Bayesian engine adapts to market regimes:

### Bull Market Adjustments
- **Momentum**: Enhanced (trend persistence)
- **Trend**: Standard weighting
- **Sentiment**: Reduced (optimism bias)

### Bear Market Adjustments
- **Momentum**: Reduced (frequent reversals)
- **Trend**: Enhanced (trend following works)
- **Sentiment**: Enhanced (pessimism informative)

### Neutral Market
- **All Signals**: Standard weighting
- **Higher Uncertainty**: More conservative thresholds

## Learning and Adaptation

### Belief Updates
The system can update its beliefs based on actual performance:

```python
def update_beliefs(self,
                  signals: Dict[SignalType, float],
                  actual_return: float,
                  horizon_days: int) -> None:
    # Updates Bayesian priors based on prediction accuracy
```

### Performance Tracking
- Historical prediction accuracy
- Signal-specific performance metrics
- Regime-conditional effectiveness
- Uncertainty calibration

## Example Usage

### Basic Signal Combination
```python
from quant.bayesian.integration import BayesianPolicyEngine

# Initialize engine with configuration
engine = BayesianPolicyEngine(config)

# Process signals for decision making
decisions = engine.bayesian_score(
    tech=technical_features,
    senti=sentiment_data,
    prices=price_data  # For regime detection
)

# Extract key metrics
for _, row in decisions.iterrows():
    print(f"{row.ticker}: E[r]={row.expected_return:.3f}, "
          f"Pr(↑)={row.prob_positive:.2f}, "
          f"Decision={row.decision}")
```

### Diagnostics and Analysis
```python
# Get signal performance diagnostics
diagnostics = engine.get_diagnostics()

# Get regime information
regime_info = engine.get_regime_info()

# Historical signal data
history = engine.get_signal_history()
```

## Key Advantages

1. **Probabilistic Output**: Unlike binary signals, provides probability distributions
2. **Uncertainty Quantification**: Knows when it doesn't know
3. **Adaptive Learning**: Improves with market feedback
4. **Regime Awareness**: Adjusts to market context
5. **Principled Combination**: Bayesian inference vs ad-hoc weighting

## Implementation Details

### Core Files
- `quant/bayesian/signal_engine.py` - Core Bayesian logic
- `quant/bayesian/integration.py` - System integration layer
- `quant/config/settings.yaml` - Configuration parameters

### Dependencies
- NumPy for statistical calculations
- Pandas for data manipulation
- No external Bayesian libraries (pure implementation)

## Performance Considerations

- **Computational Complexity**: O(n) per signal combination
- **Memory Usage**: Minimal state storage
- **Update Frequency**: Real-time capable
- **Scalability**: Linear with number of assets

## Future Enhancements

- **Non-parametric Bayesian Methods**: Gaussian processes for signal modeling
- **Online Learning**: Continuous belief updates
- **Multi-asset Dependencies**: Cross-asset Bayesian networks
- **Alternative Priors**: Hierarchical and empirical Bayes approaches