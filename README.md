# ROI - Quantitative Trading System

A modular quantitative trading system for Swedish stocks that combines Bayesian signal processing, regime detection, and risk management to generate trading recommendations.

## Quick Start

### Prerequisites
- Python 3.12+ with virtual environment in `.venv`
- Internet connection for data fetching

### Available Commands

#### Main Trading Systems
```bash
# Standard trading pipeline
python -m quant.main

# Adaptive pipeline with parameter learning
python -m quant.adaptive_main
```

#### Backtesting
```bash
# Run backtest with default settings
python -m quant.backtest_runner

# Compare adaptive vs static configurations
python -m quant.backtest_runner --comparison
```

#### Testing
```bash
# Test individual modules
python test_sentiment_module.py
python test_regime_module.py
python test_risk_module.py
python test_portfolio_module.py

# Test full system
python test_full_modular_system.py
```

## Configuration

- `quant/config/settings.yaml` - Main system settings
- `quant/config/universe.yaml` - Stock tickers to analyze
- `quant/config/modules.yaml` - Module configurations

## Output

- **Daily Reports**: `reports/daily_YYYY-MM-DD.md` - Trading recommendations with analysis
- **Portfolio State**: `data/backtest_portfolio_adaptive/current_state.json` - Current holdings
- **Backtest Results**: `backtesting_results/` - Historical performance analysis

## Architecture

The system uses a modular pipeline:
1. **Data Layer** - Fetch prices, news, and macro data
2. **Feature Engineering** - Technical indicators and sentiment analysis
3. **Regime Detection** - Market state classification (Bull/Bear/Neutral)
4. **Bayesian Engine** - Signal fusion with probabilistic outputs
5. **Risk Management** - Tail risk and volatility assessment
6. **Portfolio Optimization** - Position sizing and allocation
7. **Decision Engine** - Buy/sell/hold recommendations

## Key Features

- **Bayesian Signal Processing** - Probabilistic market predictions
- **Regime-Aware Trading** - Adaptive strategies for different market conditions
- **Statistical Risk Management** - Heavy-tail modeling and VaR calculations
- **Portfolio Tracking** - Real-time position and P&L monitoring
- **Comprehensive Backtesting** - Historical performance with attribution analysis

## Trade Evaluation Process

The system uses a sophisticated multi-stage process to evaluate and execute trades. Each potential trade goes through the following evaluation pipeline:

### Stage 1: Signal Generation and Fusion
1. **Technical Analysis**
   - SMA trends (short, medium, long-term)
   - Momentum indicators (126-day window)
   - Volume confirmation
   - Breakout detection

2. **Sentiment Analysis**
   - News sentiment scoring from multiple RSS feeds
   - Sector-specific sentiment weighting
   - Volume-weighted sentiment impact
   - News decay factor (0.9 daily)

3. **Regime Detection**
   - Market classification: Bull/Bear/Neutral
   - VIX integration with dynamic thresholds
   - Regime persistence tracking (85% persistence threshold)
   - Uncertainty quantification

### Stage 2: Bayesian Signal Processing
1. **Signal Fusion**
   - Combines technical, sentiment, and regime signals
   - Uses learned parameters from historical calibration
   - Produces probabilistic outputs: P(price ↑) and E[return]

2. **Uncertainty Quantification**
   - Signal confidence measurement
   - Bayesian uncertainty estimation
   - Monte Carlo probability calculations

### Stage 3: Stock-Specific Factor Profiles
1. **Sector Classification**
   - Swedish Banks: Rate-sensitive (SEB-A.ST, SWED-A.ST, etc.)
   - Swedish Industrials: Cyclical exposure (SAND.ST, SKF-B.ST, ABB.ST, etc.)
   - US Tech Giants: Growth/momentum focus (GOOGL, AAPL, META, NVDA)
   - And 4+ additional sector categories

2. **Factor Weighting**
   - Momentum sensitivity: 0.3-1.5x depending on sector
   - Sentiment sensitivity: 0.4-1.2x depending on news relevance
   - Macro sensitivities: Interest rates, oil prices, currency effects

3. **Regime Multipliers**
   - Bull market: 0.8-2.0x position sizing based on sector
   - Bear market: 0.2-1.2x defensive adjustments
   - Neutral market: 0.5-1.1x baseline exposure

### Stage 4: Adaptive Thresholds
1. **Dynamic Entry Criteria**
   - Base minimum return: 0.0005 (0.05% daily)
   - Base minimum confidence: 0.40 (40%)
   - Volatility-based adjustments:
     - Low volatility: More aggressive thresholds (0.5x return, 0.8x confidence)
     - High volatility: More selective thresholds (1.5x return, 1.2x confidence)

2. **Alpha Requirements**
   - Minimum expected alpha: 3% per position
   - Minimum Sharpe contribution: 0.02 per position
   - Multiple signal confirmation required

### Stage 5: Risk Assessment
1. **Tail Risk Analysis**
   - Heavy-tail distribution modeling (Student-t)
   - P[return < -2σ] calculations
   - 1-day Value-at-Risk (VaR 95%)
   - Extreme movement probabilities P[|return| > 2σ]

2. **Position-Level Risk**
   - Maximum position risk: 6%
   - Liquidity requirements: Minimum 1000 MSEK market cap
   - Pre-earnings freeze: No trading 5 days before earnings

### Stage 6: Portfolio Rules and Optimization
1. **Regime Diversification**
   - Maximum 85% exposure to single regime
   - Minimum 2 regime representation required
   - Adaptive regime allocation based on uncertainty

2. **Position Sizing**
   - Initial position size: 2% of portfolio
   - Scaling strategy: Increase by 1% increments on 3%+ gains
   - Maximum scaled size: 6% per position
   - Maximum 4 scaling levels

3. **Risk Budgeting**
   - Portfolio VaR limit: 15%
   - Risk-parity adjustments across sectors
   - Correlation-aware position sizing

### Stage 7: Transaction Cost Optimization
1. **Cost-Benefit Analysis**
   - Minimum expected alpha: 0.1% to overcome costs
   - Cost-benefit ratio: 1.5x minimum
   - Trade cost modeling: 3 basis points

2. **Dynamic Filtering**
   - Focus portfolio: Maximum 5 positions
   - High conviction threshold: 55% minimum confidence
   - Cost threshold: 20 basis points minimum expected return

### Stage 8: Final Decision Generation
1. **Buy/Sell/Hold Logic**
   - Buy: P(↑) ≥ 55% AND E[return] ≥ threshold AND passes all filters
   - Sell: P(↑) ≤ 45% OR risk limits exceeded
   - Hold: Insufficient conviction or failed cost-benefit analysis

2. **Position Sizing Calculation**
   - Base weight from confidence and expected return
   - Sector multiplier application
   - Regime adjustment
   - Risk budget allocation
   - Final portfolio weight (0-6% per position)

### Stage 9: Execution and Monitoring
1. **Trade Execution**
   - Market order simulation
   - Portfolio state updates
   - Transaction cost accounting
   - P&L tracking (realized and unrealized)

2. **Performance Attribution**
   - Component contribution analysis
   - Adaptive vs. static parameter comparison
   - Signal effectiveness tracking
   - Risk-adjusted performance metrics

### Key Decision Criteria Summary
- **Entry**: P(↑) ≥ 40-55%, E[return] ≥ 0.05-0.075%, multiple signal confirmation
- **Position Size**: 2-6% based on confidence, sector, and regime
- **Exit**: P(↑) ≤ 45%, risk limits exceeded, or scaling targets hit
- **Portfolio**: Maximum 5 positions, regime diversified, 15% VaR limit

This comprehensive evaluation ensures that every trade decision is:
1. **Statistically Validated** - Based on probabilistic models and historical calibration
2. **Risk-Adjusted** - Accounts for tail risk, volatility, and correlation
3. **Regime-Aware** - Adapts to current market conditions
4. **Cost-Optimized** - Only executes when expected alpha exceeds transaction costs
5. **Sector-Intelligent** - Uses stock-specific factor sensitivities