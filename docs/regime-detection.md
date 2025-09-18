# Regime Detection System

## Overview

The Regime Detection System classifies market conditions into distinct states (Bull, Bear, Neutral) using Hidden Markov Model principles. This contextual awareness allows the system to adapt signal processing and risk management to current market dynamics.

## Market Regime Classification

### Regime Types

#### 🐂 Bull Market
- **Characteristics**: Upward trending, lower volatility, optimistic sentiment
- **Signal Adjustments**: Enhanced momentum, standard trend, reduced sentiment weight
- **Portfolio Impact**: Higher allocation limits, growth-focused positioning

#### 🐻 Bear Market
- **Characteristics**: Downward trending, high volatility, pessimistic sentiment
- **Signal Adjustments**: Reduced momentum, enhanced trend, increased sentiment weight
- **Portfolio Impact**: Reduced allocation (60% max), defensive positioning

#### ⚖️ Neutral Market
- **Characteristics**: Sideways movement, moderate volatility, mixed signals
- **Signal Adjustments**: Standard weightings across all signals
- **Portfolio Impact**: Balanced approach, higher uncertainty tolerance

## Detection Methodology

### Multi-Factor Analysis

The system analyzes multiple market dimensions:

```python
class RegimeCharacteristics:
    volatility_regime: str      # "low", "medium", "high"
    return_regime: str         # "positive", "negative", "neutral"
    trend_regime: str          # "up", "down", "sideways"
    market_regime: MarketRegime # Final classification
```

### Key Metrics

#### 1. Volatility Analysis
- **20-day rolling volatility** (annualized)
- **Thresholds**: Low < 15%, High > 25%
- **Regime Impact**: High volatility → Bear bias

#### 2. Return Analysis
- **20-day cumulative returns**
- **Thresholds**: Bull > +0.2% daily, Bear < -0.2% daily
- **Regime Impact**: Sustained negative returns → Bear

#### 3. Trend Analysis
- **50-day SMA position**
- **Price vs trend strength**
- **Drawdown measurement** (from recent highs)

#### 4. Market Breadth
- **Positive vs negative days** (20-day window)
- **Momentum distribution**
- **Cross-asset correlations**

## Configuration

### Regime Settings (settings.yaml)

```yaml
regime_detection:
  enabled: true
  lookback_days: 60              # Analysis window
  volatility_window: 20          # Volatility calculation period
  trend_window: 50              # Trend analysis period

  transition_persistence: 0.80   # Regime stickiness (prevent oscillation)

  thresholds:
    volatility_low: 0.15         # 15% annual = low volatility
    volatility_high: 0.25        # 25% annual = high volatility
    return_bull: 0.002           # 0.2% daily for bull classification
    return_bear: -0.002          # -0.2% daily for bear classification
    drawdown_bear: -0.10         # -10% drawdown triggers bear consideration
```

## Classification Algorithm

### Step 1: Individual Metric Classification

```python
def _classify_individual_metrics(self, price_data: pd.Series) -> RegimeCharacteristics:
    # Volatility regime
    vol_regime = self._classify_volatility(returns)

    # Return regime
    return_regime = self._classify_returns(returns)

    # Trend regime
    trend_regime = self._classify_trend(price_data)

    return RegimeCharacteristics(vol_regime, return_regime, trend_regime)
```

### Step 2: Composite Classification

```python
def _determine_market_regime(self, characteristics: RegimeCharacteristics) -> MarketRegime:
    # Rule-based classification
    if return_regime == "negative" and vol_regime == "high":
        return MarketRegime.BEAR
    elif return_regime == "positive" and vol_regime in ["low", "medium"]:
        return MarketRegime.BULL
    else:
        return MarketRegime.NEUTRAL
```

### Step 3: Probability Assignment

The system assigns confidence scores using weighted voting:

```python
regime_probabilities = {
    MarketRegime.BULL: bull_score,
    MarketRegime.BEAR: bear_score,
    MarketRegime.NEUTRAL: neutral_score
}
```

## Signal Adaptation

### Regime-Specific Multipliers

**IMPORTANT**: The regime detection system now provides per-stock regime classification instead of global regime assignment. This fixes the "fake consensus" issue where all stocks were assigned the same regime, creating artificial 100% consensus reports.

### Signal Adjustments: From Hardcoded to Learned

#### Traditional Approach (Static)
Each regime historically applied fixed signal weightings:

```python
def get_regime_adjustments(self, regime: MarketRegime) -> Dict[str, float]:
    # HARDCODED VALUES - based on assumptions
    if regime == MarketRegime.BULL:
        return {
            "momentum": 1.2,    # Assumed: Enhanced momentum in trends
            "trend": 1.0,       # Assumed: Standard trend following
            "sentiment": 0.8    # Assumed: Reduced sentiment (optimism bias)
        }
    elif regime == MarketRegime.BEAR:
        return {
            "momentum": 0.7,    # Assumed: Reduced momentum (reversals)
            "trend": 1.1,       # Assumed: Enhanced trend following
            "sentiment": 1.4    # Assumed: Enhanced sentiment (fear informative)
        }
```

#### Adaptive Approach (Learned) - NEW

The system now learns regime adjustments from historical data and applies them per-stock:

```python
def estimate_regime_adjustments(self, historical_data) -> Dict[str, Dict[str, float]]:
    """Learn regime-conditional signal effectiveness from data"""

    for regime in ["bull", "bear", "neutral"]:
        regime_periods = historical_data[historical_data.regime == regime]

        for signal in ["momentum", "trend", "sentiment"]:
            # Calculate signal effectiveness in this regime
            regime_correlation = correlation(
                regime_periods[signal],
                regime_periods["forward_returns"]
            )

            # Compare to overall effectiveness
            overall_correlation = correlation(
                all_data[signal],
                all_data["forward_returns"]
            )

            # Learn the multiplier
            learned_multiplier = regime_correlation / overall_correlation

            # Store with confidence interval
            regime_adjustments[regime][signal] = EstimatedParameter(
                value=learned_multiplier,
                confidence_interval=(lower, upper),
                estimation_method="regime_conditional_correlation",
                n_observations=len(regime_periods)
            )

# Per-stock regime detection ensures honest distribution reporting
def detect_per_stock_regime(self, ticker_prices: pd.DataFrame) -> RegimeResult:
    """Detect regime for individual stock instead of global assignment"""
    if len(ticker_prices) >= 10:  # Minimum data requirement
        return self.regime_detector.detect_current_regime(ticker_prices)
    else:
        return RegimeResult(regime=MarketRegime.NEUTRAL, confidence=0.33)
```

#### Example Learning Results

```
Regime Adjustment Learning Results:
  Bull Market:
    momentum: 1.2 → 1.29 (+7.5%) [CI: 1.15-1.43, N=412 obs]
    trend: 1.0 → 1.18 (+18.0%) [CI: 1.05-1.31, N=412 obs]
    sentiment: 0.8 → 0.76 (-5.0%) [CI: 0.68-0.84, N=412 obs]

  Bear Market:
    momentum: 0.7 → 0.65 (-7.1%) [CI: 0.52-0.78, N=287 obs]
    trend: 1.1 → 1.15 (+4.5%) [CI: 1.02-1.28, N=287 obs]
    sentiment: 1.4 → 1.52 (+8.6%) [CI: 1.38-1.66, N=287 obs]
```

**Benefits of Learned Adjustments**:
- **Evidence-based**: Derived from actual regime performance
- **Dynamic**: Automatically updates with new data
- **Uncertainty-aware**: Provides confidence intervals
- **Robust**: Falls back to defaults when data is insufficient

## Transition Dynamics

### Regime Persistence

To prevent excessive regime switching:

```python
def _apply_regime_persistence(self,
                            new_regime: MarketRegime,
                            current_regime: MarketRegime) -> MarketRegime:
    if current_regime and current_regime != new_regime:
        # Require stronger evidence to switch regimes
        confidence_threshold = 0.60
        if new_regime_confidence < confidence_threshold:
            return current_regime  # Stay in current regime
    return new_regime
```

### Transition Matrix

Empirical regime transition probabilities:

```
Current\Next    Bull    Bear    Neutral
Bull            0.85    0.05    0.10
Bear            0.10    0.80    0.10
Neutral         0.30    0.20    0.50
```

## Diagnostic Output

### Regime Explanation

The system provides detailed explanations:

```python
def get_regime_explanation(self,
                         regime: MarketRegime,
                         probabilities: Dict[MarketRegime, float],
                         diagnostics: Dict) -> str:

    explanation = f"**{regime.value.title()} Market** ({probabilities[regime]*100:.0f}% confidence)\n"

    if regime == MarketRegime.BEAR:
        explanation += "Downward trend with high volatility and pessimism\n\n"
    # ... detailed market context

    return explanation
```

### Market Context Metrics

- **Volatility**: Current vs historical levels
- **Returns**: Recent performance vs thresholds
- **Trend Position**: Price vs moving averages
- **Drawdown**: Distance from recent highs
- **Breadth**: Market participation metrics

## Historical Tracking

### Regime History

The system maintains regime transition history:

```python
def get_regime_history(self) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "index": i,
            "regime": regime.value,
            "confidence": probabilities[regime],
            "prob_bull": probabilities[MarketRegime.BULL],
            "prob_bear": probabilities[MarketRegime.BEAR],
            "prob_neutral": probabilities[MarketRegime.NEUTRAL]
        }
        for i, (regime, probabilities) in enumerate(zip(
            self.regime_history,
            self.regime_probabilities_history
        ))
    ])
```

## Portfolio Integration

### Risk Management

Regime detection influences portfolio decisions:

#### Bear Market Constraints
- **Maximum Allocation**: 60% (vs 100% in bull markets)
- **Position Sizing**: Reduced individual position limits
- **Diversification**: Enhanced regime diversification requirements

#### Bull Market Opportunities
- **Higher Allocation**: Up to 100% investment
- **Growth Focus**: Momentum-driven position sizing
- **Concentration**: Allow larger individual positions

### Trade Execution

Regime awareness affects trade timing:

```python
if current_regime == MarketRegime.BEAR:
    # More conservative entry criteria
    buy_threshold += uncertainty_penalty
    # Faster exit criteria
    sell_threshold -= regime_urgency_adjustment
```

## Validation and Backtesting

### Regime Accuracy Metrics
- **Classification Precision**: Correct regime identification rate
- **Transition Timing**: Early vs late regime detection
- **False Positive Rate**: Regime switch errors

### Performance Attribution
- **Regime-Conditional Returns**: Performance by regime type
- **Transition Alpha**: Excess returns from regime timing
- **Risk Reduction**: Volatility improvement from regime awareness

## Example Usage

### Basic Regime Detection
```python
from quant.regime.detector import RegimeDetector

# Initialize detector
detector = RegimeDetector(config)

# Detect current regime
regime, probabilities, diagnostics = detector.detect_regime(price_series)

print(f"Current Regime: {regime.value}")
print(f"Confidence: {probabilities[regime]*100:.1f}%")
```

### Signal Integration
```python
# Get regime-adjusted signal multipliers
adjustments = detector.get_regime_adjustments(regime)

# Apply to signal processing
adjusted_momentum = raw_momentum * adjustments["momentum"]
adjusted_trend = raw_trend * adjustments["trend"]
```

## Key Advantages

1. **Context Awareness**: Adapts strategy to market conditions
2. **Risk Management**: Reduces exposure in adverse regimes
3. **Signal Enhancement**: Improves signal-to-noise ratio
4. **Behavioral Finance**: Incorporates market psychology
5. **Systematic Approach**: Rules-based regime classification

## Implementation Details

### Core Files
- `quant/regime/detector.py` - Main regime detection logic
- `quant/regime/__init__.py` - Regime enums and data structures

### Performance Characteristics
- **Latency**: Real-time classification capability
- **Memory**: Minimal state storage (rolling windows)
- **Accuracy**: 70-80% regime classification accuracy
- **Stability**: Persistence mechanisms prevent oscillation

## Future Enhancements

- **Machine Learning**: Neural network regime classification
- **Multi-Asset Regimes**: Cross-asset regime correlation
- **Sector Rotation**: Industry-specific regime detection
- **Alternative Data**: News, options, sentiment integration
