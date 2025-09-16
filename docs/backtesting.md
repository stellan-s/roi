# Backtesting Framework

## Overview

The ROI backtesting framework provides comprehensive historical performance analysis for both adaptive learning and static parameter configurations. It includes walk-forward validation, performance attribution analysis, and statistical significance testing to validate the effectiveness of the adaptive Bayesian engine.

## Architecture

### Core Components

```
Backtesting Framework
├── BacktestEngine (quant/backtesting/framework.py)
│   ├── Single Run Backtesting
│   ├── Walk-Forward Validation
│   └── Adaptive vs Static Comparison
├── Performance Attribution (quant/backtesting/attribution.py)
│   ├── Component-wise Analysis
│   ├── Statistical Significance
│   └── Contribution Decomposition
└── CLI Interface (quant/backtesting/cli.py)
    ├── Command-line Tools
    ├── Report Generation
    └── Comparison Analysis
```

## Core Features

### 1. Comprehensive Backtesting Engine

#### Single Run Backtesting
Execute a complete backtest over a specified historical period:

```python
from quant.backtesting.framework import BacktestEngine

engine = BacktestEngine(config)
results = engine.run_single_backtest(
    start_date="2023-01-01",
    end_date="2024-01-01",
    engine_factory=lambda cfg: AdaptiveBayesianEngine(cfg),
    engine_type="adaptive"
)

print(f"Total Return: {results.total_return:.1%}")
print(f"Sharpe Ratio: {results.sharpe_ratio:.3f}")
print(f"Max Drawdown: {results.max_drawdown:.1%}")
```

#### Walk-Forward Validation
Test the system with rolling training and testing periods:

```python
# Walk-forward backtest with periodic retraining
walk_forward_results = engine.run_walk_forward_backtest(
    start_date="2022-01-01",
    end_date="2024-01-01",
    train_days=500,      # 500 days for parameter learning
    test_days=21,        # 21 days out-of-sample testing
    rebalance_frequency=21  # Retrain every 21 days
)

# Analyze performance across all periods
total_periods = len(walk_forward_results)
avg_return = np.mean([r.annualized_return for r in walk_forward_results])
avg_sharpe = np.mean([r.sharpe_ratio for r in walk_forward_results])
```

#### Adaptive vs Static Comparison
Compare adaptive learning against static configuration:

```python
# Statistical comparison with significance testing
comparison = engine.compare_adaptive_vs_static(
    start_date="2023-01-01",
    end_date="2024-01-01"
)

print(f"Return Improvement: {comparison.return_improvement:.1%}")
print(f"Sharpe Improvement: {comparison.sharpe_improvement:.3f}")
print(f"Statistical Significance: {comparison.p_value_returns:.4f}")
```

### 2. Performance Attribution Analysis

#### Component-wise Performance Decomposition

The attribution framework analyzes which components of adaptive learning contribute most to performance improvements:

```python
from quant.backtesting.attribution import PerformanceAttributor

attributor = PerformanceAttributor(config)
attribution = attributor.full_attribution_analysis(
    start_date="2023-01-01",
    end_date="2024-01-01",
    baseline_engine="static"
)

# Results show contribution of each component
print("Component Contributions:")
print(f"Signal Normalization: {attribution.signal_normalization.return_contribution:.1%}")
print(f"Stock-Specific Learning: {attribution.stock_specific_learning.return_contribution:.1%}")
print(f"Regime Adjustments: {attribution.regime_adjustments.return_contribution:.1%}")
print(f"Tail Risk Modeling: {attribution.tail_risk_modeling.return_contribution:.1%}")
```

#### Statistical Significance Testing

Each component's contribution is tested for statistical significance:

```python
class AttributionResult:
    component_name: str
    return_contribution: float
    sharpe_contribution: float
    significance_p_value: float      # P-value for statistical test
    confidence_interval: Tuple[float, float]  # 95% confidence interval
```

### 3. CLI Interface for Easy Access

#### Command-line Backtesting
```bash
# Single backtest run
python -m quant.backtest_runner --mode single \
    --start-date 2023-01-01 --end-date 2024-01-01 \
    --engine adaptive

# Comparison analysis
python -m quant.backtest_runner --mode comparison \
    --start-date 2023-01-01 --end-date 2024-01-01

# Walk-forward validation
python -m quant.backtest_runner --mode walk-forward \
    --start-date 2022-01-01 --end-date 2024-01-01 \
    --train-days 500 --test-days 21
```

#### Report Generation
```bash
# Generate comprehensive report
python -m quant.backtest_runner --mode comparison \
    --start-date 2023-01-01 --end-date 2024-01-01 \
    --output-report results/backtest_analysis.md
```

## Backtest Results Structure

### BacktestResults Class
Complete performance analysis for a single backtest:

```python
@dataclass
class BacktestResults:
    # Performance metrics
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float

    # Trading metrics
    total_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float

    # Risk metrics
    var_95: float
    tail_risk_accuracy: float  # How well tail risk predictions performed

    # Daily data
    daily_returns: pd.Series
    daily_positions: pd.DataFrame
    daily_pnl: pd.Series

    # Metadata
    backtest_id: str
    engine_type: str  # "adaptive" or "static"
    config: Dict
```

### ComparisonResults Class
Statistical comparison between adaptive and static configurations:

```python
@dataclass
class ComparisonResults:
    adaptive_results: BacktestResults
    static_results: BacktestResults

    # Performance differences
    return_improvement: float
    sharpe_improvement: float
    drawdown_improvement: float
    tail_risk_improvement: float

    # Statistical significance
    p_value_returns: float
    confidence_interval: Tuple[float, float]
```

## Performance Metrics

### Financial Performance
- **Total Return**: Cumulative return over backtest period
- **Annualized Return**: Geometric mean return annualized
- **Sharpe Ratio**: Risk-adjusted return (excess return / volatility)
- **Maximum Drawdown**: Largest peak-to-trough decline

### Trading Performance
- **Total Trades**: Number of round-trip trades executed
- **Win Rate**: Percentage of profitable trades
- **Average Win/Loss**: Mean profit/loss per winning/losing trade

### Risk Performance
- **Volatility**: Standard deviation of daily returns (annualized)
- **VaR (95%)**: Value-at-Risk at 95% confidence level
- **Tail Risk Accuracy**: How well the system predicted extreme events

## Walk-Forward Validation

### Methodology
Walk-forward analysis simulates realistic trading by:

1. **Training Period**: Use historical data to calibrate parameters
2. **Testing Period**: Apply learned parameters to out-of-sample data
3. **Rolling Window**: Move forward in time and repeat

### Configuration
```python
# Walk-forward parameters
train_days = 500           # Training period length
test_days = 21            # Out-of-sample testing period
rebalance_frequency = 21  # How often to retrain
```

### Benefits
- **Realistic Performance**: Simulates actual parameter learning constraints
- **Overfitting Detection**: Poor out-of-sample performance indicates overfitting
- **Temporal Stability**: Shows how performance changes over different market periods

## Performance Attribution

### Component Analysis
The attribution framework decomposes adaptive learning performance into:

1. **Signal Normalization Learning**: Benefit from data-driven signal scaling vs hardcoded values
2. **Stock-Specific Learning**: Value of individual ticker effectiveness learning
3. **Regime Adjustments**: Contribution of regime-adaptive parameter adjustments
4. **Tail Risk Modeling**: Impact of statistical tail risk vs heuristic scores

### Implementation
```python
class PerformanceAttributor:
    def incremental_component_analysis(self, component: str) -> AttributionResult:
        """Test incremental value of adding a single component."""

        # Run baseline without component
        baseline_results = self._run_without_component(component)

        # Run with component added
        with_component_results = self._run_with_component(component)

        # Calculate incremental contribution
        contribution = (with_component_results.annualized_return -
                       baseline_results.annualized_return)

        # Statistical significance test
        p_value = self._statistical_test(baseline_results, with_component_results)

        return AttributionResult(
            component_name=component,
            return_contribution=contribution,
            significance_p_value=p_value,
            ...
        )
```

## Example Usage

### Complete Backtesting Workflow

```python
from quant.backtesting.framework import BacktestEngine
from quant.backtesting.attribution import PerformanceAttributor

# Initialize
config = load_configuration()
engine = BacktestEngine(config)

# 1. Single period comparison
comparison = engine.compare_adaptive_vs_static(
    start_date="2023-01-01",
    end_date="2024-01-01"
)

print(f"Performance Summary:")
print(f"Adaptive Return: {comparison.adaptive_results.annualized_return:.1%}")
print(f"Static Return: {comparison.static_results.annualized_return:.1%}")
print(f"Improvement: {comparison.return_improvement:.1%}")
print(f"Significant: {'Yes' if comparison.p_value_returns < 0.05 else 'No'}")

# 2. Walk-forward validation
walk_results = engine.run_walk_forward_backtest(
    start_date="2022-01-01",
    end_date="2024-01-01"
)

print(f"Walk-Forward Results:")
print(f"Periods Tested: {len(walk_results)}")
print(f"Average Return: {np.mean([r.annualized_return for r in walk_results]):.1%}")
print(f"Consistency: {np.std([r.annualized_return for r in walk_results]):.1%}")

# 3. Performance attribution
attributor = PerformanceAttributor(config)
attribution = attributor.full_attribution_analysis(
    start_date="2023-01-01",
    end_date="2024-01-01"
)

print(f"Component Rankings:")
for component, contribution in attribution.component_rankings:
    print(f"  {component}: {contribution:.1%}")
```

### CLI Quick Start

```bash
# Quick comparison analysis
python -m quant.backtest_runner --mode comparison \
    --start-date 2023-01-01 --end-date 2024-01-01

# Output:
# Comparison summary
# ------------------
# Adaptive annualized return: 12.3%
# Static annualized return: 8.7%
# Improvement: +3.6%
# Sharpe improvement: +0.45
# Statistical significance: p=0.023 (significant)
```

## Configuration

### Backtesting Settings
Add to `settings.yaml`:

```yaml
backtesting:
  enabled: true
  default_start_date: "2022-01-01"
  default_end_date: "2024-01-01"

  # Walk-forward parameters
  walk_forward:
    train_days: 500
    test_days: 21
    rebalance_frequency: 21
    min_trades_per_period: 5

  # Performance attribution
  attribution:
    enabled: true
    bootstrap_samples: 1000
    confidence_level: 0.95
    components: ["signal_normalization", "stock_specific", "regime_adjustments", "tail_risk"]

  # Reporting
  reports:
    output_dir: "results/backtesting"
    generate_plots: true
    detailed_attribution: true
```

## Advanced Features

### Custom Engine Comparison
Compare any two engine configurations:

```python
# Custom engine comparison
engine1 = lambda cfg: AdaptiveBayesianEngine(cfg)
engine2 = lambda cfg: BayesianPolicyEngine(cfg)

results1 = engine.run_single_backtest("2023-01-01", "2024-01-01", engine1, "adaptive")
results2 = engine.run_single_backtest("2023-01-01", "2024-01-01", engine2, "static")

# Compare results
improvement = results1.annualized_return - results2.annualized_return
```

### Sensitivity Analysis
Test parameter sensitivity:

```python
# Test different Bayesian thresholds
thresholds = [0.55, 0.58, 0.60, 0.62, 0.65]
results = []

for threshold in thresholds:
    config_copy = config.copy()
    config_copy['bayesian']['decision_thresholds']['buy_probability'] = threshold

    result = engine.run_single_backtest(
        "2023-01-01", "2024-01-01",
        lambda cfg: AdaptiveBayesianEngine(cfg),
        f"threshold_{threshold}"
    )
    results.append((threshold, result.annualized_return))

# Find optimal threshold
optimal = max(results, key=lambda x: x[1])
```

## Validation and Testing

### Model Validation Checks
The backtesting framework includes automatic validation:

```python
class BacktestValidation:
    def validate_results(self, results: BacktestResults) -> Dict[str, bool]:
        validation = {
            "sufficient_trades": results.total_trades >= 10,
            "reasonable_sharpe": -2.0 <= results.sharpe_ratio <= 5.0,
            "valid_drawdown": 0.0 <= abs(results.max_drawdown) <= 1.0,
            "non_zero_returns": not np.isclose(results.total_return, 0.0),
            "valid_win_rate": 0.0 <= results.win_rate <= 1.0
        }
        return validation
```

### Data Quality Checks
- **Sufficient History**: Minimum data requirements for reliable results
- **Missing Data**: Warnings for gaps in price/sentiment data
- **Outlier Detection**: Identification of extreme returns that may indicate data errors

## Performance Characteristics

### Computational Requirements
- **Single Backtest**: ~30-60 seconds for 1 year of daily data
- **Walk-Forward**: ~5-10 minutes for 2 years with 21-day rebalancing
- **Attribution Analysis**: ~2-5 minutes for comprehensive component analysis
- **Memory Usage**: ~100-500MB depending on universe size and period length

### Scalability
- **Linear with Universe Size**: Computation scales linearly with number of stocks
- **Linear with Time Period**: Memory and time scale with backtest length
- **Parallel Processing**: Multiple periods can be processed in parallel

## Key Benefits

1. **Robust Validation**: Walk-forward analysis provides realistic performance estimates
2. **Statistical Rigor**: Significance testing and confidence intervals for all improvements
3. **Component Attribution**: Understand which adaptive learning components add value
4. **Easy Integration**: CLI interface makes backtesting accessible for all users
5. **Comprehensive Metrics**: Full suite of performance, risk, and trading metrics
6. **Automated Reporting**: Generate publication-ready analysis reports

## Future Enhancements

- **Multi-asset Backtesting**: Support for portfolios with multiple asset classes
- **Transaction Cost Modeling**: More sophisticated cost models with market impact
- **Benchmark Comparison**: Compare against market indices and other strategies
- **Monte Carlo Analysis**: Scenario-based backtesting with random market conditions
- **Real-time Backtesting**: Live paper trading validation

## Files and Dependencies

### Core Implementation
- `quant/backtesting/framework.py` - Main backtesting engine
- `quant/backtesting/attribution.py` - Performance attribution analysis
- `quant/backtesting/cli.py` - Command-line interface
- `quant/backtest_runner.py` - Main execution script

### Dependencies
- **scipy**: Statistical testing and analysis
- **numpy**: Numerical computations
- **pandas**: Data manipulation and analysis
- **matplotlib** (optional): Plotting and visualization