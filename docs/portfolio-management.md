# Portfolio Management System

## Overview

The Portfolio Management System integrates Bayesian signals, regime detection, and heavy-tail risk modeling to make systematic portfolio construction and trade execution decisions. It provides position sizing, diversification controls, and state management for actionable trading recommendations.

## Architecture

### Core Components

```
Portfolio Management
├── Rules Engine (quant/portfolio/rules.py)
│   ├── Position Sizing
│   ├── Regime Diversification
│   └── Transaction Cost Filtering
├── State Management (quant/portfolio/state.py)
│   ├── Portfolio Tracking
│   ├── Trade Execution
│   └── Performance Attribution
└── Risk Integration
    ├── Tail Risk Penalties
    ├── Stress Testing
    └── Risk Budgeting
```

## Portfolio Rules Engine

### Position Sizing Framework

#### Kelly-Inspired Approach
The system uses a modified Kelly criterion that incorporates:

```python
def _apply_position_sizing(self, positions: List[PortfolioPosition]) -> List[PortfolioPosition]:
    # Risk-adjusted expected returns
    for pos in active_buys:
        confidence_adj = pos.decision_confidence      # Bayesian confidence
        regime_stability = 0.8 if pos.regime != 'neutral' else 0.6
        tail_risk_penalty = 1.0 - (pos.tail_risk_score * 0.3)  # Up to 30% reduction

        risk_adj_return = (pos.expected_return *
                          confidence_adj *
                          regime_stability *
                          tail_risk_penalty)
```

#### Position Weight Calculation
```python
# Proportional allocation based on risk-adjusted returns
total_risk_adj_return = sum(risk_adjusted_returns)
base_weight = (risk_adjusted_return / total_risk_adj_return) * total_weight_budget
final_weight = min(base_weight, max_weight_per_stock)
```

### Regime Diversification

#### Single Regime Exposure Limits
Prevents over-concentration in any single market regime:

```python
# Configuration
max_single_regime_exposure: 0.85  # Max 85% in same regime

# Implementation
if len(regime_counts) == 1 and len(active_positions) > 3:
    # Too much concentration - downgrade weakest signals
    max_positions = int(len(active_positions) * max_single_regime_exposure)
    # Downgrade excess positions to Hold
```

#### Bear Market Allocation
Special constraints during bear markets:

```python
if regime == 'bear':
    total_weight_budget = bear_market_allocation  # Default: 60%
    print(f"🐻 Bear market: Reducerar allokering till {total_weight_budget*100}%")
```

### Transaction Cost Filtering

#### Cost-Benefit Analysis
Filters out trades where expected return doesn't justify costs:

```python
def _apply_transaction_cost_filter(self, positions: List[PortfolioPosition]):
    cost_threshold = trade_cost_bps / 10000.0  # Convert bps to decimal

    for pos in positions:
        if abs(pos.expected_return) < cost_threshold * 2:  # 2x safety margin
            pos.decision = 'Hold'  # Not worth trading
            pos.weight = 0.0
```

## Portfolio State Management

### Portfolio Tracking

#### Core Data Structures
```python
class PortfolioHolding:
    ticker: str
    shares: float
    avg_cost: float
    current_price: float
    market_value: float
    weight: float
    return_pct: float
    unrealized_pnl: float

class PortfolioState:
    cash: float
    total_value: float
    holdings: Dict[str, PortfolioHolding]
    last_updated: datetime
    transaction_history: List[Transaction]
```

#### Persistence
Portfolio state is maintained in JSON format:

```python
def save_state(self, filepath: str = "data/portfolio_state.json"):
    state_dict = {
        'cash': self.cash,
        'total_value': self.total_value,
        'holdings': {ticker: asdict(holding) for ticker, holding in self.holdings.items()},
        'last_updated': self.last_updated.isoformat(),
        'transaction_history': [asdict(t) for t in self.transaction_history]
    }
```

### Evaluation & Logging

Daily runs now create a transparent audit trail for both the simulated portfolio and the raw recommendations:

- `data/portfolio/current_state.json` – latest paper-portfolio snapshot (cash, holdings, unrealized P&L)
- `data/portfolio/trade_history.json` – append-only log of simulated BUY trades with timestamp, size, and Bayesian metrics at execution
- `data/recommendation_logs/recommendations_YYYY-MM-DD.parquet` – portfolio-adjusted recommendations (E[r]_1d, E[R]_21d, Pr(↑), σ, tail metrics, decision confidence) for the run date plus a UTC `logged_at_utc`
- `data/recommendation_logs/simulated_trades_YYYY-MM-DD.json` – executed paper trades for the same session, stored alongside the recommendation file for quick comparison

These artifacts are produced by `quant/adaptive_main.py` after applying portfolio rules, making it easy to backtest hit rates, compute Brier scores, or reconcile portfolio P&L against the original recommendations.

### Trade Execution Simulation

#### Trade Processing
```python
def execute_trades(self,
                  trades: List[TradeRecommendation],
                  current_prices: Dict[str, float]) -> List[Transaction]:

    executed_transactions = []

    for trade in trades:
        if trade.action == "BUY":
            transaction = self._execute_buy(trade, current_prices[trade.ticker])
        elif trade.action == "SELL":
            transaction = self._execute_sell(trade, current_prices[trade.ticker])

        executed_transactions.append(transaction)
```

#### Buy Logic
```python
def _execute_buy(self, trade: TradeRecommendation, current_price: float) -> Transaction:
    target_value = trade.target_weight * self.total_value
    shares_to_buy = target_value / current_price
    cost = shares_to_buy * current_price

    if cost <= self.cash:
        # Execute the trade
        self.cash -= cost
        self._add_or_update_holding(trade.ticker, shares_to_buy, current_price)
```

#### Sell Logic
```python
def _execute_sell(self, trade: TradeRecommendation, current_price: float) -> Transaction:
    if trade.ticker in self.holdings:
        holding = self.holdings[trade.ticker]
        proceeds = holding.shares * current_price
        self.cash += proceeds
        realized_pnl = proceeds - (holding.shares * holding.avg_cost)
```

## Risk-Integrated Decision Making

### Tail Risk Position Sizing

#### Risk Score Integration
Heavy-tail risk scores directly influence position sizing:

```python
class PortfolioConstraints:
    max_tail_risk_allocation: float = 0.30     # Max 30% in high-risk positions
    tail_risk_position_penalty: float = 0.30  # Max 30% reduction for tail risk
    min_tail_risk_adjusted_weight: float = 0.5 # Minimum 50% after penalty
```

#### Implementation
```python
# In position sizing calculation
tail_risk_penalty = 1.0 - (getattr(pos, 'tail_risk_score', 0.0) * 0.3)
tail_risk_penalty = max(0.5, tail_risk_penalty)  # Minimum 50% allocation

risk_adjusted_return = (expected_return *
                       confidence *
                       regime_stability *
                       tail_risk_penalty)
```

### Stress Test Integration

#### Portfolio Stress Analysis
```python
def stress_test_portfolio(self,
                         portfolio_weights: Dict[str, float],
                         risk_profiles: Dict[str, PortfolioRiskProfile],
                         scenarios: List[str]) -> Dict[str, Dict]:

    stress_results = {}
    for scenario_name in scenarios:
        scenario = STRESS_SCENARIOS[scenario_name]
        results = self._apply_stress_scenario(portfolio_weights, risk_profiles, scenario)
        stress_results[scenario_name] = results

    return stress_results
```

## Configuration Management

### Portfolio Policy Settings

```yaml
policy:
  max_weight: 0.10                          # Max 10% per position
  pre_earnings_freeze_days: 5               # No trading before earnings
  trade_cost_bps: 5                        # 5 bps transaction costs

  # Portfolio-level rules
  max_single_regime_exposure: 0.85          # Max 85% same regime
  regime_diversification: true              # Require regime diversity
  min_portfolio_positions: 3                # Minimum 3 positions
  bear_market_allocation: 0.60              # Max 60% in bear markets

  # Heavy-tail risk controls
  max_tail_risk_allocation: 0.30            # Max 30% high-risk positions
  tail_risk_position_penalty: 0.30          # Max 30% penalty for tail risk
  min_tail_risk_adjusted_weight: 0.5        # Min 50% after tail adjustment
```

## Decision Flow

### Complete Portfolio Construction Process

1. **Signal Generation**: Bayesian engine produces expected returns and probabilities
2. **Regime Detection**: Current market regime identified
3. **Risk Assessment**: Tail risk scores calculated
4. **Position Sizing**: Risk-adjusted position weights determined
5. **Diversification Check**: Regime and concentration limits applied
6. **Transaction Filtering**: Cost-benefit analysis performed
7. **Trade Generation**: Final buy/sell/hold decisions
8. **Execution Simulation**: Portfolio state updated
9. **Reporting**: Comprehensive analysis generated

### Example Decision Logic

```python
def apply_portfolio_rules(self, decisions: pd.DataFrame) -> pd.DataFrame:
    # Get latest decisions
    latest_decisions = decisions[decisions['date'] == decisions['date'].max()]

    # Create position objects
    positions = self._create_positions(latest_decisions)

    # Apply portfolio rules in sequence
    positions = self._apply_regime_diversification(positions)
    positions = self._apply_position_sizing(positions)
    positions = self._apply_transaction_cost_filter(positions)

    # Convert back to DataFrame
    return self._positions_to_dataframe(positions, latest_decisions)
```

## Performance Analytics

### Portfolio Summary Generation

```python
def get_portfolio_summary(self, decisions: pd.DataFrame) -> Dict:
    active_positions = latest[latest['portfolio_weight'] > 0]

    return {
        'total_positions': len(active_positions),
        'total_weight': active_positions['portfolio_weight'].sum(),
        'avg_expected_return': active_positions['expected_return'].mean(),
        'avg_confidence': active_positions['decision_confidence'].mean(),
        'regime_distribution': active_positions['market_regime'].value_counts().to_dict(),
        'largest_position': active_positions['portfolio_weight'].max(),
        'decision_distribution': latest['decision'].value_counts().to_dict()
    }
```

### Risk Attribution

#### Position-Level Risk Contribution
```python
for ticker, weight in portfolio_weights.items():
    risk_profile = risk_profiles[ticker]
    risk_contribution = weight * risk_profile.volatility_annual
    tail_risk_contribution = weight * risk_profile.tail_risk_metrics.tail_risk_multiplier
```

#### Portfolio-Level Metrics
```python
# Portfolio volatility (simplified independence assumption)
portfolio_variance = sum((weight * volatility)**2 for weight, volatility in positions)
portfolio_volatility = sqrt(portfolio_variance)

# Weighted tail risk
weighted_tail_multiplier = sum(weight * tail_multiplier for weight, tail_multiplier in positions)
```

## Advanced Features

### Dynamic Rebalancing

#### Trigger Conditions
- Significant regime changes
- Position weights drift beyond tolerance
- New high-confidence signals
- Risk profile changes

#### Rebalancing Logic
```python
def should_rebalance(self, current_weights: Dict, target_weights: Dict) -> bool:
    max_drift = max(abs(current_weights.get(ticker, 0) - target_weight)
                   for ticker, target_weight in target_weights.items())

    return max_drift > self.rebalance_threshold  # e.g., 2%
```

### Tax Optimization

#### Harvest Loss Opportunities
```python
def identify_tax_loss_opportunities(self) -> List[str]:
    loss_candidates = []
    for ticker, holding in self.holdings.items():
        if holding.unrealized_pnl < -1000:  # Significant loss
            loss_candidates.append(ticker)
    return loss_candidates
```

## Example Usage

### Complete Portfolio Management Workflow

```python
from quant.portfolio.rules import PortfolioManager
from quant.portfolio.state import PortfolioState

# Initialize portfolio management
portfolio_manager = PortfolioManager(config)
portfolio_state = PortfolioState(initial_cash=100000)

# Get Bayesian decisions
decisions = bayesian_engine.bayesian_score(tech_data, sentiment_data, price_data)

# Apply portfolio rules
portfolio_decisions = portfolio_manager.apply_portfolio_rules(decisions)

# Generate trades
current_prices = get_current_prices(tickers)
trades = portfolio_state.generate_trades(portfolio_decisions, current_prices)

# Execute trades
transactions = portfolio_state.execute_trades(trades, current_prices)

# Get portfolio summary
summary = portfolio_manager.get_portfolio_summary(portfolio_decisions)
```

### Risk-Integrated Position Sizing

```python
# Example position with high tail risk
position = PortfolioPosition(
    ticker="TSLA",
    expected_return=0.0008,      # 0.08% daily
    decision_confidence=0.75,     # High confidence
    tail_risk_score=0.6          # Moderate-high tail risk
)

# Calculate risk-adjusted weight
confidence_adj = 0.75
regime_stability = 0.8           # Stable regime
tail_risk_penalty = 1.0 - (0.6 * 0.3) = 0.82  # 18% penalty

risk_adjusted_return = 0.0008 * 0.75 * 0.8 * 0.82 = 0.000394
# Reduces expected return by ~51% due to risk factors
```

## Key Advantages

1. **Systematic Decision Making**: Rules-based approach eliminates emotion
2. **Risk Awareness**: Integrates multiple risk dimensions
3. **Regime Adaptation**: Adjusts strategy to market conditions
4. **Cost Efficiency**: Filters unprofitable trades
5. **State Persistence**: Maintains portfolio history and performance
6. **Scalability**: Handles arbitrary number of positions

## Implementation Details

### Core Files
- `quant/portfolio/rules.py` - Portfolio construction rules
- `quant/portfolio/state.py` - State management and trade execution

### Performance Characteristics
- **Decision Speed**: Real-time portfolio construction
- **Memory Usage**: Minimal state storage
- **Scalability**: Linear with universe size
- **Reliability**: Persistent state with error handling

## Future Enhancements

- **Options Integration**: Hedging and income generation
- **Sector Constraints**: Industry diversification rules
- **ESG Integration**: Environmental and social criteria
- **Alternative Assets**: Crypto and commodities support
- **Real-time Execution**: Integration with brokers
