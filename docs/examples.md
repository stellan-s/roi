# Examples and Tutorials

## Basic Usage Examples

### 1. Complete System Run
```bash
# Run the complete ROI analysis
python -m quant.main

# Output: Daily report in reports/daily_YYYY-MM-DD.md
# Example: reports/daily_2025-09-15.md
```

### 2. Custom Signal Analysis
```python
from quant.bayesian.integration import BayesianPolicyEngine
from quant.features.technical import compute_technical_features
from quant.data_layer.prices import fetch_prices

# Load configuration
config = load_yaml("settings.yaml")

# Get price data
tickers = ["TSLA", "AAPL", "MSFT"]
prices = fetch_prices(tickers, "data", 500)

# Compute technical features
tech_features = compute_technical_features(prices)

# Initialize Bayesian engine
engine = BayesianPolicyEngine(config)

# Generate decisions
decisions = engine.bayesian_score(tech_features, sentiment_data, prices)

# Analyze results
for _, row in decisions.iterrows():
    print(f"{row.ticker}: E[r]={row.expected_return*100:.2f}%, "
          f"Pr(↑)={row.prob_positive*100:.0f}%, "
          f"Decision={row.decision}")
```

### 3. Risk Analysis for Single Stock
```python
from quant.risk.analytics import RiskAnalytics
from quant.risk.heavy_tail import HeavyTailRiskModel

# Initialize risk analytics
risk_analytics = RiskAnalytics(config)

# Get TSLA price history
tsla_prices = prices[prices['ticker'] == 'TSLA']['close']

# Comprehensive risk analysis
risk_profile = risk_analytics.analyze_position_risk(
    ticker="TSLA",
    price_history=tsla_prices,
    expected_return=0.08,  # 8% annual expected return
    time_horizon_months=12
)

# Print results
print(f"TSLA Risk Analysis:")
print(f"  Annual Volatility: {risk_profile.volatility_annual*100:.1f}%")
print(f"  Sharpe Ratio: {risk_profile.sharpe_ratio:.2f}")
print(f"  Tail Risk Multiplier: {risk_profile.tail_risk_metrics.tail_risk_multiplier:.2f}x")
print(f"  12m P(loss > 20%): {risk_profile.prob_loss_20_percent*100:.1f}%")
print(f"  12m P(gain > 20%): {risk_profile.prob_gain_20_percent*100:.1f}%")
```

## Advanced Examples

### 4. Portfolio Stress Testing
```python
from quant.risk.analytics import RiskAnalytics, STRESS_SCENARIOS

# Define portfolio
portfolio_weights = {
    "TSLA": 0.30,
    "AAPL": 0.25,
    "GOOGL": 0.25,
    "MSFT": 0.20
}

# Analyze each position
risk_profiles = {}
for ticker, weight in portfolio_weights.items():
    ticker_prices = prices[prices['ticker'] == ticker]['close']
    expected_return = 0.08  # Simplified - should come from Bayesian engine

    risk_profiles[ticker] = risk_analytics.analyze_position_risk(
        ticker, ticker_prices, expected_return
    )

# Run stress tests
stress_results = risk_analytics.stress_test_portfolio(
    portfolio_weights,
    risk_profiles,
    scenarios=["black_monday", "covid_crash"]
)

# Print stress test results
for scenario, results in stress_results.items():
    print(f"\n{scenario.upper()}:")
    print(f"  Total Portfolio Loss: {results['total_portfolio_loss']*100:.1f}%")
    print(f"  Duration: {results['duration_days']} days")

    for ticker, impact in results['position_impacts'].items():
        print(f"  {ticker} Impact: {impact['total_loss']*100:.1f}%")
```

### 5. Custom Regime Detection
```python
from quant.regime.detector import RegimeDetector, MarketRegime

# Initialize regime detector
detector = RegimeDetector(config)

# Analyze market regime for specific period
market_data = prices[prices['ticker'] == 'TSLA']['close']

# Detect regime
current_regime, probabilities, diagnostics = detector.detect_regime(market_data)

print(f"Current Regime: {current_regime.value}")
print(f"Confidence: {probabilities[current_regime]*100:.1f}%")
print("\nAll Regime Probabilities:")
for regime, prob in probabilities.items():
    print(f"  {regime.value}: {prob*100:.1f}%")

# Get signal adjustments for current regime
adjustments = detector.get_regime_adjustments(current_regime)
print(f"\nSignal Adjustments:")
for signal, multiplier in adjustments.items():
    print(f"  {signal}: {multiplier:.1f}x")

# Get detailed explanation
explanation = detector.get_regime_explanation(current_regime, probabilities, diagnostics)
print(f"\nRegime Explanation:\n{explanation}")
```

### 6. Heavy-tail Risk Deep Dive
```python
from quant.risk.heavy_tail import HeavyTailRiskModel

# Initialize heavy-tail model
tail_model = HeavyTailRiskModel(config)

# Get return series
returns = prices[prices['ticker'] == 'TSLA']['close'].pct_change().dropna()

# Fit Student-t distribution
student_t_params = tail_model.fit_heavy_tail_distribution(returns)
print(f"Student-t Parameters:")
print(f"  Degrees of Freedom: {student_t_params['degrees_of_freedom']:.1f}")
print(f"  Location (μ): {student_t_params['location']*252*100:.1f}% annual")
print(f"  Scale (σ): {student_t_params['scale']*np.sqrt(252)*100:.1f}% annual")

# Heavy tails detected if DoF < 10
if student_t_params['degrees_of_freedom'] < 10:
    print("⚠️ HEAVY TAILS DETECTED - Higher extreme event risk")

# Fit Extreme Value Theory
evt_params = tail_model.fit_extreme_value_theory(returns)
print(f"\nExtreme Value Theory:")
for tail_name, params in evt_params.items():
    print(f"  {tail_name}:")
    print(f"    Shape (ξ): {params['shape']:.3f}")
    print(f"    Threshold: {params['threshold']*100:.2f}%")
    print(f"    Exceedances: {params['n_exceedances']}")

# Calculate tail risk metrics
tail_metrics = tail_model.calculate_tail_risk_metrics(returns, 0.95, 21)
print(f"\n21-day VaR (95%):")
print(f"  Normal Distribution: {tail_metrics.var_normal*100:.2f}%")
print(f"  Student-t: {tail_metrics.var_student_t*100:.2f}%")
print(f"  Tail Risk Multiplier: {tail_metrics.tail_risk_multiplier:.2f}x")

# Monte Carlo simulation
mc_results = tail_model.monte_carlo_simulation(
    expected_return=0.08,
    volatility=0.25,
    tail_parameters=student_t_params,
    time_horizon_months=12,
    n_simulations=10000
)

print(f"\n12-Month Monte Carlo Results:")
print(f"  P(return > 0%): {mc_results.prob_positive*100:.1f}%")
print(f"  P(return > +20%): {mc_results.prob_plus_20*100:.1f}%")
print(f"  P(return < -20%): {mc_results.prob_minus_20*100:.1f}%")
print(f"  Worst 1%: {mc_results.percentile_1*100:.1f}%")
print(f"  Best 1%: {mc_results.percentile_99*100:.1f}%")
```

### 7. Portfolio Construction Workflow
```python
from quant.portfolio.rules import PortfolioManager
from quant.portfolio.state import PortfolioState

# Initialize portfolio management
portfolio_manager = PortfolioManager(config)
portfolio_state = PortfolioState(initial_cash=100000)

# Get latest Bayesian decisions (from example 2)
latest_decisions = decisions[decisions['date'] == decisions['date'].max()]

# Apply portfolio rules
portfolio_decisions = portfolio_manager.apply_portfolio_rules(decisions)

# Generate trade recommendations
current_prices = {
    "TSLA": 410.04,
    "AAPL": 236.70,
    "GOOGL": 251.61,
    "MSFT": 515.36
}

# Create trade recommendations from portfolio decisions
trades = []
for _, row in portfolio_decisions.iterrows():
    if row['portfolio_weight'] > 0 and row['decision'] == 'Buy':
        trades.append(TradeRecommendation(
            ticker=row['ticker'],
            action="BUY",
            target_weight=row['portfolio_weight'],
            rationale=f"E[r]: {row['expected_return']*100:.2f}%, Pr(↑): {row['prob_positive']*100:.0f}%"
        ))

# Execute trades
if trades:
    transactions = portfolio_state.execute_trades(trades, current_prices)

    print("Executed Trades:")
    for transaction in transactions:
        print(f"  {transaction.action} {transaction.shares:.0f} shares of {transaction.ticker}")
        print(f"    Price: ${transaction.price:.2f}, Value: ${transaction.value:.0f}")

# Get portfolio summary
portfolio_summary = portfolio_state.get_portfolio_summary()
print(f"\nPortfolio Summary:")
print(f"  Total Value: ${portfolio_summary['total_value']:,.0f}")
print(f"  Cash: ${portfolio_summary['cash']:,.0f}")
print(f"  Invested: ${portfolio_summary['invested']:,.0f}")
print(f"  Positions: {portfolio_summary['positions']}")
```

## Configuration Examples

### 8. Conservative Configuration
```yaml
# settings_conservative.yaml
bayesian:
  decision_thresholds:
    buy_probability: 0.70      # Higher confidence required
    sell_probability: 0.30     # Faster selling
    min_expected_return: 0.001 # Higher return requirement
    max_uncertainty: 0.20      # Lower uncertainty tolerance

policy:
  max_weight: 0.05            # Smaller positions
  bear_market_allocation: 0.40 # Very defensive in bear markets
  max_tail_risk_allocation: 0.15 # Limited high-risk exposure

risk_modeling:
  confidence_levels: [0.99, 0.999] # Higher confidence VaR
```

### 9. Aggressive Configuration
```yaml
# settings_aggressive.yaml
bayesian:
  decision_thresholds:
    buy_probability: 0.52      # Lower confidence acceptable
    sell_probability: 0.48     # Hold positions longer
    min_expected_return: 0.0002 # Lower return threshold
    max_uncertainty: 0.45      # Higher uncertainty tolerance

policy:
  max_weight: 0.20            # Larger concentrated positions
  bear_market_allocation: 0.85 # Maintain exposure in bear markets
  max_tail_risk_allocation: 0.50 # Accept more tail risk

risk_modeling:
  confidence_levels: [0.90, 0.95] # Lower confidence VaR
```

## Debugging and Diagnostics

### 10. Signal Performance Analysis
```python
# Get signal diagnostics
diagnostics = engine.get_signal_diagnostics()
print("Signal Performance:")
print(diagnostics)

# Get signal history
history = engine.get_signal_history()
if not history.empty:
    print(f"\nSignal History ({len(history)} observations):")
    print(history.tail())

# Analyze signal weights over time
recent_decisions = decisions.tail(20)
avg_weights = {
    'trend': recent_decisions['trend_weight'].mean(),
    'momentum': recent_decisions['momentum_weight'].mean(),
    'sentiment': recent_decisions['sentiment_weight'].mean()
}

print(f"\nRecent Average Signal Weights:")
for signal, weight in avg_weights.items():
    print(f"  {signal}: {weight:.1%}")
```

### 11. Regime History Analysis
```python
# Get regime history
regime_history = engine.get_regime_history()

if not regime_history.empty:
    print("Recent Regime History:")
    print(regime_history.tail(10))

    # Analyze regime stability
    recent_regimes = regime_history['regime'].tail(20)
    regime_changes = (recent_regimes != recent_regimes.shift()).sum()
    print(f"\nRegime changes in last 20 periods: {regime_changes}")

    # Current regime distribution
    current_dist = regime_history.iloc[-1]
    print(f"\nCurrent Regime Probabilities:")
    print(f"  Bull: {current_dist['prob_bull']*100:.1f}%")
    print(f"  Bear: {current_dist['prob_bear']*100:.1f}%")
    print(f"  Neutral: {current_dist['prob_neutral']*100:.1f}%")
```

### 12. Performance Backtesting Framework
```python
def simple_backtest(decisions_df, prices_df, horizon_days=21):
    """
    Simple backtest to evaluate signal performance
    """
    results = []

    for _, decision in decisions_df.iterrows():
        ticker = decision['ticker']
        decision_date = decision['date']
        expected_return = decision['expected_return']
        prob_positive = decision['prob_positive']

        # Get actual return over horizon
        ticker_prices = prices_df[prices_df['ticker'] == ticker]
        start_idx = ticker_prices[ticker_prices['date'] == decision_date].index

        if len(start_idx) > 0:
            start_idx = start_idx[0]
            end_idx = min(start_idx + horizon_days, len(ticker_prices) - 1)

            start_price = ticker_prices.iloc[start_idx]['close']
            end_price = ticker_prices.iloc[end_idx]['close']
            actual_return = (end_price - start_price) / start_price

            results.append({
                'ticker': ticker,
                'date': decision_date,
                'expected_return': expected_return,
                'actual_return': actual_return,
                'prob_positive': prob_positive,
                'was_positive': actual_return > 0,
                'prediction_error': abs(expected_return - actual_return)
            })

    backtest_df = pd.DataFrame(results)

    # Calculate performance metrics
    if not backtest_df.empty:
        accuracy = (backtest_df['was_positive'] == (backtest_df['prob_positive'] > 0.5)).mean()
        mean_error = backtest_df['prediction_error'].mean()
        correlation = backtest_df['expected_return'].corr(backtest_df['actual_return'])

        print(f"Backtest Results:")
        print(f"  Directional Accuracy: {accuracy*100:.1f}%")
        print(f"  Mean Prediction Error: {mean_error*100:.2f}%")
        print(f"  Expected vs Actual Correlation: {correlation:.3f}")

        # Probability calibration
        high_confidence = backtest_df[backtest_df['prob_positive'] > 0.7]
        if not high_confidence.empty:
            actual_positive_rate = high_confidence['was_positive'].mean()
            print(f"  High Confidence (>70%) Actual Positive Rate: {actual_positive_rate*100:.1f}%")

    return backtest_df

# Run backtest
backtest_results = simple_backtest(decisions, prices)
```

## Integration Examples

### 13. Custom Signal Integration
```python
def add_rsi_signal(tech_features, window=14):
    """
    Add RSI signal to technical features
    """
    def calculate_rsi(prices, window):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    # Add RSI for each ticker
    for ticker in tech_features['ticker'].unique():
        ticker_data = tech_features[tech_features['ticker'] == ticker].copy()
        ticker_data['rsi'] = calculate_rsi(ticker_data['close'], window)

        # Normalize RSI to [-1, 1] range for signal combination
        ticker_data['rsi_signal'] = (ticker_data['rsi'] - 50) / 50

        # Update main dataframe
        tech_features.loc[tech_features['ticker'] == ticker, 'rsi_signal'] = ticker_data['rsi_signal']

    return tech_features

# Usage in signal combination
tech_features = add_rsi_signal(tech_features)

# Modify signal normalization in BayesianPolicyEngine to include RSI
def _normalize_signals_with_rsi(self, row):
    signals = self._normalize_signals(row)  # Get standard signals

    # Add RSI signal if available
    if 'rsi_signal' in row:
        signals[SignalType.RSI] = row['rsi_signal']

    return signals
```

## Production Deployment Examples

### 14. Automated Daily Execution
```bash
#!/bin/bash
# daily_roi_analysis.sh

cd /path/to/roi
source .venv/bin/activate

# Run daily analysis
python -m quant.main

# Check if report was generated
if [ -f "reports/daily_$(date +%Y-%m-%d).md" ]; then
    echo "✅ Daily analysis completed successfully"

    # Optional: Send report via email
    # mail -s "ROI Daily Analysis" user@example.com < reports/daily_$(date +%Y-%m-%d).md
else
    echo "❌ Daily analysis failed"
    exit 1
fi
```

### 15. Configuration Management
```python
import os
from pathlib import Path

def load_environment_config():
    """
    Load configuration based on environment
    """
    env = os.getenv('ROI_ENV', 'development')

    config_files = {
        'development': 'settings_dev.yaml',
        'staging': 'settings_staging.yaml',
        'production': 'settings_prod.yaml'
    }

    config_file = config_files.get(env, 'settings.yaml')
    config_path = Path('quant/config') / config_file

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    return load_yaml(config_file)

# Usage
config = load_environment_config()
```

These examples demonstrate the flexibility and power of the ROI system, from basic usage to advanced customization and production deployment scenarios.