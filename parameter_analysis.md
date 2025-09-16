# Hardcoded Parameter Analysis

## Data-Driven Parameters (Should be estimated from historical data)

### Signal Normalization
- **Sentiment scaling**: `/2.0` in `sentiment_signal = np.clip(row['sent_score'] / 2.0, -1.0, 1.0)`
  - Should be: `percentile_based_scaling(sent_score, p_low=5, p_high=95)`
- **Momentum scaling**: `*2.0` in `momentum_signal = (row['mom_rank'] - 0.5) * 2.0`
  - Should be: Based on empirical momentum signal distribution
- **Base annual return**: `0.08` in Bayesian engine
  - Should be: `historical_market_return(lookback_period)`

### Bayesian Engine Parameters
- **Sigmoid scaling**: `-3` in `1 / (1 + np.exp(-3 * combined_signal))`
  - Should be: Calibrated to maximize signal-return correlation
- **Signal multiplier**: `*2` in `signal_multiplier = combined_signal * 2`
  - Should be: Estimated from signal predictive power

### Risk Parameters
- **Momentum volatility factor**: `0.3` in `base_tail_risk = momentum_volatility * 0.3`
  - Should be: `empirical_momentum_vol_relationship()`

## Learned Weights (Should be optimized via ML/statistical learning)

### Regime Adjustments
- **Bull market multipliers**: `{"momentum": 1.3, "trend": 1.2, "sentiment": 0.8}`
- **Bear market multipliers**: `{"momentum": 0.7, "trend": 1.1, "sentiment": 1.4}`
- **Neutral multipliers**: `{"momentum": 0.9, "trend": 0.8, "sentiment": 1.1}`
  - Should be: Learned from regime-conditional signal performance

### Confidence Weights
- **Probability weight**: `0.4` in confidence calculation
- **Return magnitude weight**: `0.3`
- **Uncertainty penalty weight**: `0.3`
  - Should be: Cross-validated weights for prediction accuracy

### Risk Regime Multipliers
- **Bear market tail risk**: `1.5`
- **Bull market tail risk**: `0.8`
  - Should be: Empirical tail risk by regime

## Business Parameters (Keep configurable)
- Time horizons: `21` days
- Risk thresholds: buy/sell probabilities
- Transaction costs: `5` bps
- Portfolio constraints: max weights, diversification rules

## Implementation Priority
1. **Signal normalization** - Most impact on signal quality
2. **Regime adjustments** - Currently pure assumptions
3. **Bayesian calibration** - Improve probability estimates
4. **Risk parameters** - Better tail risk modeling