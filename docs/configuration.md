# Configuration Reference

## Overview

The ROI system is highly configurable through YAML files in the `quant/config/` directory. This document provides complete reference for all configuration options.

## Configuration Files

### settings.yaml
Main system configuration with all parameters.

### universe.yaml
Defines the stock universe for analysis.

## Complete Configuration Reference

### Run Configuration
```yaml
run:
  as_of: today         # Analysis date: "today" or "YYYY-MM-DD"
  outdir: "reports"    # Output directory for reports
```

### Data Configuration
```yaml
data:
  cache_dir: "data"    # Directory for cached price data
  lookback_days: 500   # Historical data period for analysis
```

### Signal Configuration
```yaml
signals:
  sma_long: 200        # Simple Moving Average period for trend signal
  momentum_window: 252 # Momentum calculation period (trading days)
  news_feed_urls:      # RSS feeds for sentiment analysis
    - "https://news.cision.com/se/rss/all"
```

### Bayesian Engine Configuration
```yaml
bayesian:
  time_horizon_days: 21          # Investment horizon for E[r] and Pr(↑)

  decision_thresholds:
    buy_probability: 0.58        # Buy when Pr(↑) ≥ 58%
    sell_probability: 0.40       # Sell when Pr(↑) ≤ 40%
    min_expected_return: 0.0005  # Minimum 0.05% daily E[r] for buy
    max_uncertainty: 0.35        # Maximum uncertainty for action

  # Prior beliefs for signals (adjustable based on backtesting)
  priors:
    trend_effectiveness: 0.62    # SMA trend-following edge
    momentum_effectiveness: 0.68 # Momentum edge (strongest documented)
    sentiment_effectiveness: 0.58 # Sentiment edge (noisiest)

  # Parameter Learning (NEW: replaces hardcoded values with data-driven estimates)
  parameter_learning:
    enabled: true                 # Enable adaptive parameter estimation
    calibration_lookback: 1000    # Days of data for parameter calibration
    confidence_level: 0.95        # Confidence level for parameter estimates
    min_observations: 100         # Minimum data points for reliable estimates

    # Signal normalization learning
    sentiment_normalization: "percentile_based"  # Use data distribution vs hardcoded /2.0
    momentum_scaling: "distribution_based"       # Scale based on actual momentum range

    # Regime adjustment learning
    regime_effectiveness_learning: true          # Learn regime multipliers from data
    bootstrap_iterations: 500                    # Bootstrap samples for confidence intervals
```

### Regime Detection Configuration
```yaml
regime_detection:
  enabled: true
  lookback_days: 60             # Regime detection window
  volatility_window: 20         # Volatility calculation period
  trend_window: 50             # Trend strength window

  # Regime transition probabilities (prevents oscillation)
  transition_persistence: 0.80  # Probability of staying in same regime

  # Regime classification thresholds
  thresholds:
    volatility_low: 0.15        # 15% annual volatility = low
    volatility_high: 0.25       # 25% annual volatility = high
    return_bull: 0.002          # 0.2% daily return for bull market
    return_bear: -0.002         # -0.2% daily return for bear market
    drawdown_bear: -0.10        # -10% drawdown triggers bear consideration
```

### Portfolio Policy Configuration
```yaml
policy:
  max_weight: 0.10             # Maximum position size per stock
  pre_earnings_freeze_days: 5  # No trading before earnings announcements
  trade_cost_bps: 5           # Transaction costs in basis points

  # Portfolio-level rules
  max_single_regime_exposure: 0.85  # Max 85% allocation in same regime
  regime_diversification: true      # Require regime diversification
  min_portfolio_positions: 3        # Minimum number of active positions
  bear_market_allocation: 0.60      # Maximum allocation in bear markets

  # Heavy-tail risk controls
  max_tail_risk_allocation: 0.30    # Max 30% in high-risk positions
  tail_risk_position_penalty: 0.30  # Max 30% position size reduction
  min_tail_risk_adjusted_weight: 0.5 # Minimum 50% of original weight
```

### Statistical Tail Risk Configuration (NEW)
```yaml
# Statistical Tail Risk Analysis
tail_risk:
  enabled: true
  calculation_method: "statistical"       # "statistical" vs "heuristic" (legacy)

  # Primary measure: P[return < -2σ] (nedsidesrisk)
  downside_tail_risk:
    enabled: true
    threshold_sigma: -2.0                 # -2σ threshold for downside risk
    display_thresholds:
      low: 0.025                          # 2.5% = green 🟢
      high: 0.05                          # 5.0% = red 🔴 (yellow between)

  # Secondary measure: P[|return| > 2σ] (extremrörelser)
  extreme_move_probability:
    enabled: true
    threshold_sigma: 2.0                  # ±2σ threshold for extreme moves
    display_thresholds:
      low: 0.046                          # 4.6% = normal distribution baseline
      high: 0.10                          # 10% = concerning level

  # Distribution fitting
  distribution_fitting:
    test_normality: true                  # Use Jarque-Bera test
    prefer_student_t: true                # Fit Student-t for heavy tails
    min_observations: 100                 # Minimum data for reliable fit
    fallback_distribution: "empirical"    # When parametric fitting fails

  # Signal and regime adjustments
  adjustments:
    momentum_factor: 0.2                  # How much momentum affects tail risk
    sentiment_factor: 0.15                # How much sentiment affects tail risk
    regime_multipliers:
      bull: 0.8                          # Lower tail risk in bull markets
      bear: 1.4                          # Higher tail risk in bear markets
      neutral: 1.0                       # Baseline tail risk
```

### Risk Modeling Configuration
```yaml
risk_modeling:
  enabled: true
  confidence_levels: [0.95, 0.99, 0.999]  # VaR confidence levels
  time_horizons_days: [21, 63, 252]       # Analysis periods (1m, 3m, 1y)
  monte_carlo_simulations: 10000          # Monte Carlo simulation size
  evt_threshold_percentile: 0.95          # EVT tail threshold
  min_observations_for_tail_fit: 30       # Minimum data for Student-t fitting

  # Tail risk score thresholds for visualization
  tail_risk_thresholds:
    low: 0.4        # Green 🟢 indicator below this
    high: 0.7       # Red 🔴 indicator above this (Yellow 🟡 between)
```

### Risk Analytics Configuration
```yaml
risk_analytics:
  stress_scenarios: ["black_monday", "covid_crash", "regime_shift"]
  risk_free_rate: 0.02                    # 2% annual risk-free rate
  target_portfolio_volatility: 0.15       # 15% target portfolio volatility
  tail_risk_penalty: 0.02                 # Extra penalty for heavy tails
```

## Universe Configuration

### universe.yaml
```yaml
tickers:
  # Swedish stocks
  - VOLV-B.ST   # Volvo B
  - INVE-B.ST   # Investor B
  - ERIC-B.ST   # Ericsson B
  - SAND.ST     # Sandvik
  - ATCO-A.ST   # Atlas Copco A
  - HM-B.ST     # Hennes & Mauritz B
  - SKF-B.ST    # SKF B
  - TEL2-B.ST   # Tele2 B
  - NIBE-B.ST   # NIBE
  - SWED-A.ST   # Swedbank A
  - ASSA-B.ST   # Assa Abloy B

  # US stocks
  - AAPL        # Apple
  - MSFT        # Microsoft
  - GOOGL       # Alphabet
  - AMZN        # Amazon
  - TSLA        # Tesla
  - META        # Meta Platforms
  - NVDA        # NVIDIA
```

## Configuration Guidelines

### Bayesian Decision Thresholds

#### Conservative Settings (Higher Quality Signals)
```yaml
decision_thresholds:
  buy_probability: 0.65        # Higher confidence required
  sell_probability: 0.35       # More decisive selling
  min_expected_return: 0.001   # Higher return threshold
  max_uncertainty: 0.25        # Lower uncertainty tolerance
```

#### Aggressive Settings (More Trading Activity)
```yaml
decision_thresholds:
  buy_probability: 0.55        # Lower confidence acceptable
  sell_probability: 0.45       # Less decisive selling
  min_expected_return: 0.0003  # Lower return threshold
  max_uncertainty: 0.40        # Higher uncertainty tolerance
```

### Risk Management Profiles

#### Conservative Risk Profile
```yaml
policy:
  max_weight: 0.05             # Smaller position sizes
  bear_market_allocation: 0.40  # Lower bear market exposure
  max_tail_risk_allocation: 0.20 # Less high-risk allocation

risk_modeling:
  confidence_levels: [0.99, 0.999] # Higher confidence VaR
  tail_risk_thresholds:
    low: 0.3                   # Stricter risk classification
    high: 0.6
```

#### Aggressive Risk Profile
```yaml
policy:
  max_weight: 0.15             # Larger position sizes
  bear_market_allocation: 0.80  # Higher bear market exposure
  max_tail_risk_allocation: 0.40 # More high-risk allocation

risk_modeling:
  confidence_levels: [0.90, 0.95] # Lower confidence VaR
  tail_risk_thresholds:
    low: 0.5                   # More lenient risk classification
    high: 0.8
```

## Adaptive System Usage

### Running with Parameter Learning

To use the adaptive system instead of hardcoded parameters:

```python
# Use adaptive_main.py instead of main.py
python -m quant.adaptive_main
```

This will:
1. **Calibrate parameters** from historical data (1000+ days)
2. **Train stock-specific Bayesian posteriors**
3. **Generate learning diagnostics** showing parameter changes
4. **Run daily analysis** with learned parameters

### Learning Summary Output
```
=== Parameter Estimation Results ===
Top parameter changes from defaults:
  sentiment_scale_factor: 0.500 → 0.672 (+34.4%)
  momentum_scale_factor: 2.000 → 1.847 (-7.7%)
  bear_sentiment_effectiveness: 1.400 → 1.520 (+8.6%)

Learning Summary:
  Total parameters estimated: 23
  Significant changes (>10%): 8
  Average change: 12.3%
```

### Configuration for Learning

The learning system respects configuration settings for thresholds and limits but learns the core signal processing parameters:

```yaml
# What stays configurable (business decisions)
bayesian:
  time_horizon_days: 21              # Investment horizon
  decision_thresholds:
    buy_probability: 0.58            # Risk tolerance
    sell_probability: 0.40
    min_expected_return: 0.0005

# What gets learned (signal processing)
# These sections still provide fallback defaults:
  priors:
    trend_effectiveness: 0.62        # → Learned from data
    momentum_effectiveness: 0.68     # → Learned from data
    sentiment_effectiveness: 0.58    # → Learned from data

  parameter_learning:
    enabled: true                    # Enable adaptive learning
    calibration_lookback: 1000       # Data period for learning
    min_observations: 100            # Minimum data for reliable estimates
```

## Environment-Specific Configuration

### Development Environment
```yaml
data:
  lookback_days: 100           # Shorter history for faster testing

# Reduce learning complexity for development
bayesian:
  parameter_learning:
    calibration_lookback: 500  # Less historical data
    bootstrap_iterations: 100  # Fewer bootstrap samples

risk_modeling:
  monte_carlo_simulations: 1000 # Fewer simulations for speed
```

### Production Environment
```yaml
data:
  lookback_days: 1000          # Longer history for stability

risk_modeling:
  monte_carlo_simulations: 50000 # More simulations for accuracy
```

## Configuration Validation

The system validates configuration on startup:

### Required Fields
- All threshold values must be between 0 and 1
- Time horizons must be positive integers
- Portfolio weights must sum to ≤ 1.0
- Risk-free rate must be reasonable (0-0.10)

### Logical Consistency
- `buy_probability` > `sell_probability`
- `max_uncertainty` < 1.0
- `bear_market_allocation` ≤ 1.0
- `tail_risk_thresholds.low` < `tail_risk_thresholds.high`

### Performance Considerations
- Higher `monte_carlo_simulations` = better accuracy, slower execution
- Longer `lookback_days` = more stable regime detection, slower startup
- More `confidence_levels` = comprehensive analysis, more computation

## Advanced Configuration

### Custom Signal Priors
Based on backtesting results, you might adjust:

```yaml
priors:
  trend_effectiveness: 0.72    # If trend following performs well
  momentum_effectiveness: 0.55  # If momentum shows mean reversion
  sentiment_effectiveness: 0.63 # If sentiment analysis improves
```

### Regime-Specific Tuning
```yaml
regime_detection:
  thresholds:
    volatility_low: 0.12       # Lower threshold for low-vol detection
    return_bull: 0.0015        # Tighter bull market criteria
    drawdown_bear: -0.08       # Earlier bear market detection
```

### Risk Model Customization
```yaml
risk_modeling:
  evt_threshold_percentile: 0.90 # Lower threshold = more tail data
  min_observations_for_tail_fit: 50 # Higher requirement for reliability
```

## Configuration Best Practices

1. **Start Conservative**: Begin with conservative thresholds and gradually adjust
2. **Backtest Changes**: Validate configuration changes with historical data
3. **Monitor Performance**: Track how configuration affects signal quality
4. **Document Rationale**: Keep notes on why specific values were chosen
5. **Version Control**: Track configuration changes over time

## Troubleshooting

### Common Issues

#### Too Few Buy Signals
- Lower `buy_probability` threshold
- Reduce `min_expected_return`
- Increase `max_uncertainty`

#### Too Many Trades
- Increase `trade_cost_bps`
- Raise decision thresholds
- Reduce regime switching sensitivity

#### Poor Risk Management
- Lower `max_weight` limits
- Increase `tail_risk_position_penalty`
- Use more conservative stress scenarios

### Performance Issues
- Reduce `monte_carlo_simulations`
- Shorten `lookback_days`
- Disable risk modeling temporarily (`enabled: false`)