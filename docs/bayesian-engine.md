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

### Adaptive Parameter Learning (NEW)

The system now learns key parameters from historical data instead of using hardcoded values:

1. **Signal Normalization**: Data-driven scaling replaces arbitrary constants
2. **Regime Adjustments**: Empirically estimated effectiveness multipliers
3. **Base Returns**: Market-derived expected returns
4. **Risk Parameters**: Statistically calibrated tail risk measures

**Benefits**:
- **Evidence-based**: Parameters derived from actual market data
- **Adaptive**: Automatically adjusts to changing market conditions
- **Transparent**: Clear confidence intervals and estimation methods
- **Robust**: Falls back to sensible defaults when data is insufficient

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

## Parameter Learning Process

### Calibration Phase

When using the `AdaptiveBayesianEngine`, the system performs parameter calibration before daily analysis:

```python
# 1. Load historical data (1000+ days)
prices, sentiment, technical, returns = load_historical_data()

# 2. Estimate parameters from data
parameter_estimator = ParameterEstimator()
estimated_params = parameter_estimator.estimate_all_parameters(
    prices, sentiment, technical, returns
)

# 3. Update engine with learned parameters
adaptive_engine.calibrate_parameters(estimated_params)
```

### What Gets Learned

#### 1. Signal Normalization Parameters
```python
# OLD: Hardcoded scaling
sentiment_signal = np.clip(sent_score / 2.0, -1.0, 1.0)  # Why /2.0?

# NEW: Data-driven scaling
p5, p95 = np.percentile(sentiment_scores, [5, 95])
scale_factor = 2.0 / (p95 - p5)  # Based on actual distribution
sentiment_signal = np.clip(sent_score * scale_factor, -1.0, 1.0)
```

#### 2. Stock-Specific Signal Effectiveness (NEW)
```python
# Per-stock Bayesian posteriors learning
for ticker in universe:
    for signal_type in [TREND, MOMENTUM, SENTIMENT]:
        # Update signal effectiveness based on historical performance
        prediction_accuracy = 1.0 if (signal > 0) == (return > 0) else 0.0
        posterior['alpha'] += prediction_accuracy * signal_strength
        posterior['beta'] += (1 - prediction_accuracy) * signal_strength

        # Calculate stock-specific effectiveness
        effectiveness = posterior['alpha'] / (posterior['alpha'] + posterior['beta'])
```

#### 3. Regime Effectiveness Multipliers
```python
# OLD: Assumed multipliers
bull_adjustments = {"momentum": 1.3, "trend": 1.2, "sentiment": 0.8}

# NEW: Empirically estimated with confidence intervals
bull_momentum_correlation = correlation(bull_periods, momentum_signal, returns)
overall_momentum_correlation = correlation(all_periods, momentum_signal, returns)
learned_multiplier = bull_momentum_correlation / overall_momentum_correlation

# Bootstrap confidence intervals
confidence_intervals = bootstrap_regime_effectiveness(regime_data, 500_iterations)
```

#### 4. Statistical Tail Risk Parameters
```python
# NEW: Proper statistical tail risk
downside_tail_risk = P(return < -2σ)  # Primary measure
extreme_move_prob = P(|return| > 2σ)  # Secondary measure

# Distribution fitting (Normal/Student-t/Empirical)
best_distribution = fit_optimal_distribution(historical_returns)
```

### Learning Results and Diagnostics

#### Parameter Estimation Summary
```python
learning_summary = engine.get_learning_summary()
# Output:
{
    'status': 'learning_complete',
    'estimation_date': '2025-09-16T10:30:00',
    'data_period': {
        'start': '2023-01-01T00:00:00',
        'end': '2025-09-15T23:59:59',
        'total_observations': 15847
    },
    'parameter_changes': {
        'total_parameters': 23,
        'significant_changes': 8,
        'avg_change_percent': 12.3,
        'max_change_percent': 34.4
    }
}
```

#### Detailed Parameter Diagnostics
```python
diagnostics = engine.get_parameter_diagnostics()
# Top parameter changes from defaults:
  sentiment_scale_factor: 0.500 → 0.672 (+34.4%) [CI: 0.61-0.73, N=1247]
  momentum_scale_factor: 2.000 → 1.847 (-7.7%) [CI: 1.79-1.90, N=2156]
  bear_sentiment_effectiveness: 1.400 → 1.520 (+8.6%) [CI: 1.38-1.66, N=287]
  base_annual_return: 0.080 → 0.073 (-8.8%) [CI: 0.065-0.081, N=1891]
```

#### Stock-Specific Learning Results
```python
# Bayesian Signal Learning Results:
TREND:
  Prior effectiveness: 0.620
  Learned effectiveness: 0.643 (Δ+0.023)
  Observations: 1456
  Confidence interval: (0.610, 0.676)

MOMENTUM:
  Prior effectiveness: 0.680
  Learned effectiveness: 0.695 (Δ+0.015)
  Observations: 1891
  Confidence interval: (0.671, 0.719)

SENTIMENT:
  Prior effectiveness: 0.580
  Learned effectiveness: 0.561 (Δ-0.019)
  Observations: 1247
  Confidence interval: (0.539, 0.583)
```

#### Data Quality Monitoring
```python
# Automatic data quality checks:
⚠️ Saknar prisdata för 3 tickers: ERIC-B, TELIA, VOLV-B
⚠️ Identiska signalvärden för tickers: ASSA-B, ATCO-A. Kontrollera mapping och källdata.
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

  # NEW: Parameter learning configuration
  parameter_learning:
    enabled: true                 # Enable adaptive parameter estimation
    calibration_lookback: 1000    # Days of data for calibration
    confidence_level: 0.95        # Confidence level for estimates
    min_observations: 100         # Minimum data for reliable estimates
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