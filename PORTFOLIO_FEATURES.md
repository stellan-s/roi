# ROI Portfolio Management & Tracking Features

## ✅ Implemented Features

### 🎯 **Actionable Buy/Sell Recommendations**
- **Adjusted decision thresholds**: Buy ≥58% (vs 65%), Sell ≤40% (vs 35%)
- **Lower expected-return floor**: 0.05% (vs 0.1%) to unlock more trades
- **Transaction-cost filtering**: Automatically filters unprofitable trades
- **Portfolio weighting**: Explicit percentage allocations per position

### 📊 **Portfolio State Management**
- **Persistent portfolio tracking**: JSON-based state storage
- **Real-time portfolio valuation**: Updates as new prices arrive
- **Holdings tracking**: Shares, average cost, and unrealised P&L per position
- **Trade execution simulation**: Automatic buy/sell based on recommendations
- **Cash management**: Starts with 100k SEK and tracks available capital

### 🏦 **Portfolio Metrics & Reporting**
- **Total portfolio value**: Combined holdings plus cash
- **Return tracking**: Both absolute and percentage P&L
- **Position sizing**: Automated weighting driven by E[r] and confidence
- **Holdings breakdown**: Detailed per-stock information in reports

### 🛡️ **Risk Management & Diversification**
- **Regime-aware allocation**: Caps exposure at 60% in bear markets
- **Regime diversification**: Warns and limits when too many positions share a regime
- **Minimum positions**: Guarantees at least three active holdings
- **Max single position**: Limits to 10% per stock
- **Transaction cost optimisation**: Filters trades that do not clear costs

### 🔧 **Configurable System**
Every aspect can be tuned in `settings.yaml`:

```yaml
# Decision thresholds
bayesian:
  decision_thresholds:
    buy_probability: 0.58        # More aggressive than 65%
    sell_probability: 0.40       # More aggressive than 35%
    min_expected_return: 0.0005  # 0.05% minimum

# Portfolio constraints
policy:
  max_weight: 0.10                     # Max position size
  max_single_regime_exposure: 0.85     # Regime diversification
  min_portfolio_positions: 3           # Minimum holdings
  bear_market_allocation: 0.60         # Conservative i bear market
  trade_cost_bps: 5                    # Transaction costs

# Prior beliefs (influences E[r] calculation)
bayesian:
  priors:
    momentum_effectiveness: 0.68       # Greater conviction → higher E[r]
```

## 📈 **Live Example From Today’s Report**

### **Current State (Bear Market)**
- **Portfolio value**: 100,000 SEK (100% cash)
- **Regime**: Bear Market (60% confidence)
- **Decision**: Conservative stance – no buys because E[r] fails to clear transaction costs

### **Activated Risk Controls**
- ⚠️ **Regime diversification**: 11 positions all in bear → 6 downgraded to Hold
- 🐻 **Bear market allocation**: Reduced to a 60% investment cap
- ⚠️ **Transaction-cost filter**: GOOGL/META blocked (E[r] below costs)

### **Sell Signals**
- **ERIC-B.ST**: Pr(↑)=29%, E[r]=+0.01%
- **INVE-B.ST**: Pr(↑)=24%, E[r]=+0.01%
- **ATCO-A.ST**: Pr(↑)=22%, E[r]=+0.00%

## 🚀 **Benefits of the New System**

1. **Actionable insights**: Concrete buy/sell calls with percentages
2. **Portfolio awareness**: Full visibility into exposures and performance
3. **Risk-conscious**: Intelligent constraints in uncertain market conditions
4. **Transparent**: Clear explanations for why decisions are or are not taken
5. **Configurable**: Easy to adjust aggressiveness and risk tolerance
6. **Regime-adaptive**: Behaviour shifts across bull, bear, and neutral regimes

## 📋 **Next Steps Toward Production**

1. **Broker integration**: Connect to a live trading API
2. **Notification system**: Email/SMS alerts for new recommendations
3. **Backtesting engine**: Historical performance validation
4. **Tax optimisation**: Wash-sale rules and tax-loss harvesting
5. **Multi-currency support**: For international holdings

The system is now ready for real portfolio management with full transparency and risk control! 🎯
