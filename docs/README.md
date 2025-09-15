# ROI - Quantitative Trading System Documentation

## Overview

ROI is a sophisticated quantitative trading and investment analysis system written in Python. It combines Bayesian signal processing, regime detection, heavy-tail risk modeling, and portfolio management to generate actionable trading recommendations for Swedish and US stocks.

## System Architecture

The system follows a modular pipeline architecture with the following main components:

```
ROI System
├── Data Layer (quant/data_layer/)
│   ├── Price Data Management
│   └── News & Sentiment Data
├── Feature Engineering (quant/features/)
│   ├── Technical Indicators
│   └── Sentiment Analysis
├── Bayesian Engine (quant/bayesian/)
│   ├── Signal Processing
│   └── Market Integration
├── Regime Detection (quant/regime/)
│   └── HMM-based Market Classification
├── Risk Modeling (quant/risk/)
│   ├── Heavy-tail Distributions
│   └── Portfolio Risk Analytics
├── Portfolio Management (quant/portfolio/)
│   ├── Position Sizing
│   └── State Management
└── Reporting (quant/reports/)
    └── Daily Analysis Reports
```

## Key Features

### 🧠 Bayesian Signal Engine
- **Probabilistic Decision Making**: Combines multiple signals using Bayesian inference
- **Uncertainty Quantification**: Provides confidence intervals and uncertainty measures
- **Adaptive Learning**: Updates beliefs based on market performance
- **Multi-signal Integration**: Trend, momentum, and sentiment analysis

### 📊 Regime Detection
- **Market Context Awareness**: Classifies markets as Bull, Bear, or Neutral
- **HMM-based Classification**: Hidden Markov Models for regime transition
- **Signal Adaptation**: Adjusts signal weights based on market regime
- **Risk-aware Positioning**: Different allocation strategies per regime

### 📈 Heavy-tail Risk Modeling
- **Student-t Distributions**: Realistic modeling of fat-tail events
- **Extreme Value Theory (EVT)**: Tail risk quantification
- **Monte Carlo Simulation**: 12-month probability scenarios
- **Stress Testing**: Portfolio resilience under extreme conditions

### 💼 Portfolio Management
- **Risk Budgeting**: Position sizing based on risk-adjusted returns
- **Regime Diversification**: Prevents over-concentration in single regime
- **Transaction Cost Optimization**: Filters out unprofitable trades
- **Dynamic Rebalancing**: Continuous portfolio optimization

## Documentation Structure

- **[Getting Started](getting-started.md)** - Installation and quick start guide
- **[Configuration](configuration.md)** - Complete configuration reference
- **[Bayesian Engine](bayesian-engine.md)** - Signal processing and decision theory
- **[Regime Detection](regime-detection.md)** - Market classification system
- **[Risk Modeling](risk-modeling.md)** - Heavy-tail analysis and stress testing
- **[Portfolio Management](portfolio-management.md)** - Position sizing and optimization
- **[API Reference](api-reference.md)** - Complete function and class documentation
- **[Examples](examples.md)** - Usage examples and tutorials
- **[Glossary](glossary.md)** - Terms, formulas, and abbreviations explained

## Quick Start

```bash
# Install dependencies
python -m pip install -r requirements.txt

# Run complete analysis
python -m quant.main

# View report
cat reports/daily_$(date +%Y-%m-%d).md
```

## Sample Output

The system generates comprehensive daily reports with:

- **Bayesian Recommendations**: Buy/Sell/Hold decisions with probabilities
- **Risk Assessment**: Tail risk indicators and portfolio analytics
- **Market Context**: Current regime and signal weightings
- **Portfolio Status**: Holdings, performance, and recommended trades

## Development Status

✅ **Completed Features**:
- Bayesian signal combination with uncertainty quantification
- Regime detection using HMM models
- Student-t distribution fitting for heavy tails
- Extreme Value Theory implementation
- Monte Carlo risk simulation
- Portfolio management with regime diversification
- Comprehensive daily reporting

🔄 **In Development**:
- Backtesting framework
- Machine learning signal enhancement
- Real-time data integration
- Advanced portfolio optimization

## License

This is a proof-of-concept system for educational and research purposes.