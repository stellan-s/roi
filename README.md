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