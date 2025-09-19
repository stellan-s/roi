# Sharpe Ratio Configuration Quick Reference

## Quick Enable/Disable Toggles

### To Enable All Sharpe Ratio Improvements
```yaml
# In settings.yaml

# Enhanced diversification (HIGH IMPACT)
risk_budgeting:
  max_position_weight: 0.06          # ✅ Better diversification
  min_positions: 12                  # ✅ More positions
  correlation_controls:
    enabled: true                    # ✅ Enable correlation management

# Enhanced signal quality (MEDIUM IMPACT)
signals:
  trend_confirmation:
    enabled: true                    # ✅ Multi-timeframe confirmation
  sentiment_improvements:
    enabled: true                    # ✅ Better sentiment analysis

# Enhanced regime detection (MEDIUM IMPACT)
regime_detection:
  regime_transition_detection:
    enabled: true                    # ✅ Better regime timing

# Enhanced risk management (HIGH IMPACT)
policy:
  stop_loss_management:
    enabled: true                    # ✅ Stop loss protection
  transaction_optimization:
    enabled: true                    # ✅ Cost optimization
```

### Conservative Settings (Lower Risk, Moderate Returns)
```yaml
risk_budgeting:
  target_portfolio_volatility: 0.10  # Very low volatility target
  max_position_weight: 0.04          # Maximum diversification
  min_positions: 15                  # High diversification

  regime_allocation_multipliers:
    bull: 0.85                       # Conservative even in bull markets
    bear: 0.35                       # Very defensive in bear markets
    neutral: 0.60                    # Moderate exposure in neutral

policy:
  stop_loss_management:
    trailing_stop_loss: 0.06         # Tight stop losses (6%)
    portfolio_stop_loss: 0.08        # Portfolio protection (8%)
```

### Aggressive Settings (Higher Risk, Higher Returns)
```yaml
risk_budgeting:
  target_portfolio_volatility: 0.14  # Higher volatility tolerance
  max_position_weight: 0.08          # Allow larger positions
  min_positions: 8                   # Fewer positions, more concentration

  regime_allocation_multipliers:
    bull: 1.0                        # Full allocation in bull markets
    bear: 0.65                       # Less defensive in bear markets
    neutral: 0.80                    # Higher exposure in neutral

policy:
  stop_loss_management:
    trailing_stop_loss: 0.12         # Wider stop losses (12%)
    portfolio_stop_loss: 0.15        # Higher portfolio tolerance (15%)
```

## Performance Tuning by Priority

### Priority 1: Immediate High Impact (Enable First)
```yaml
risk_budgeting:
  max_position_weight: 0.06          # Diversification
  min_positions: 12                  # More positions
  target_portfolio_volatility: 0.12  # Lower vol target

policy:
  stop_loss_management:
    enabled: true                    # Risk protection
```

### Priority 2: Medium Impact (Enable Second)
```yaml
risk_budgeting:
  correlation_controls:
    enabled: true                    # Correlation management
  volatility_position_sizing:
    enabled: true                    # Vol-based sizing

regime_detection:
  regime_transition_detection:
    enabled: true                    # Better regime timing
```

### Priority 3: Fine-Tuning (Enable Last)
```yaml
signals:
  trend_confirmation:
    enabled: true                    # Signal quality
  sentiment_improvements:
    enabled: true                    # Better sentiment

policy:
  transaction_optimization:
    enabled: true                    # Cost optimization
  entry_exit_optimization:
    enabled: true                    # Entry/exit timing
```

## Common Configuration Patterns

### Pattern 1: "Maximum Sharpe" (Recommended)
- Maximize diversification (12+ positions, 6% max weight)
- Enable all correlation controls
- Conservative regime allocation
- Comprehensive stop loss management
- Enhanced signal quality

### Pattern 2: "Low Volatility Focus"
- Target 10% portfolio volatility
- Maximum diversification (15+ positions)
- Very tight stop losses (6%)
- Conservative regime allocation
- Strong volatility penalties

### Pattern 3: "Momentum Enhanced"
- Focus on signal quality improvements
- Multi-timeframe confirmation
- Enhanced regime detection
- Moderate diversification
- Balanced risk management

## Feature Dependencies

### Prerequisites
- **Correlation Controls** → Requires price data history
- **Volatility Sizing** → Requires volatility calculation
- **Regime Transitions** → Requires regime detection enabled
- **Stop Losses** → Requires position tracking

### Recommended Combinations
- **Diversification + Correlation Controls** (high synergy)
- **Volatility Sizing + Vol Targeting** (complementary)
- **Regime Transitions + Conservative Allocation** (risk management)
- **Signal Quality + Entry/Exit Optimization** (timing improvements)

## Monitoring and Validation

### Key Metrics to Monitor
```yaml
# Expected improvements after implementation:
sharpe_ratio: +0.85 to +2.05        # Primary target metric
portfolio_volatility: 15% → 12%      # Risk reduction
max_drawdown: -15% to -25% reduction # Tail risk improvement
win_rate: +5% to +15% improvement    # Consistency improvement
```

### Configuration Validation Checklist
- [ ] `max_position_weight` ≤ `1.0 / min_positions`
- [ ] `target_portfolio_volatility` > 0.05 and < 0.30
- [ ] `correlation_penalty_factor` > 1.0
- [ ] `stop_loss` values < `profit_taking` values
- [ ] Regime allocation multipliers sum reasonably

### Backtesting Validation
```bash
# Test configuration changes
python -m quant.backtest_runner --start-date 2023-01-01 --end-date 2024-12-31

# Compare before/after Sharpe ratios
# Monitor: Sharpe, Max DD, Win Rate, Volatility
```

## Troubleshooting

### Common Issues
1. **Too many positions** → Increase `max_positions` or reduce `min_positions`
2. **High turnover** → Increase `min_weight_drift` or `transaction_cost_threshold`
3. **Low allocation** → Check regime allocation multipliers
4. **Correlation warnings** → Adjust `max_pairwise_correlation`

### Performance Degradation
If Sharpe ratio doesn't improve:
1. Check if correlation controls are too restrictive
2. Verify stop losses aren't too tight
3. Ensure sufficient diversification (12+ positions)
4. Review regime allocation settings

### Emergency Disable
To quickly disable all enhancements:
```yaml
risk_budgeting:
  correlation_controls:
    enabled: false
  volatility_position_sizing:
    enabled: false

signals:
  trend_confirmation:
    enabled: false
  sentiment_improvements:
    enabled: false

regime_detection:
  regime_transition_detection:
    enabled: false

policy:
  stop_loss_management:
    enabled: false
  transaction_optimization:
    enabled: false
```