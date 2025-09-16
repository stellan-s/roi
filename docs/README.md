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

### 🧠 Adaptive Bayesian Signal Engine
- **Probabilistic Decision Making**: Combines multiple signals using Bayesian inference
- **Stock-Specific Learning**: Individual effectiveness learning per ticker-signal combination
- **Data-Driven Parameters**: Replaces hardcoded values with empirically estimated parameters
- **Uncertainty Quantification**: Provides confidence intervals and uncertainty measures
- **Multi-signal Integration**: Trend, momentum, and sentiment analysis with adaptive scaling

### 📊 Regime Detection
- **Market Context Awareness**: Classifies markets as Bull, Bear, or Neutral
- **HMM-based Classification**: Hidden Markov Models for regime transition
- **Signal Adaptation**: Adjusts signal weights based on market regime
- **Risk-aware Positioning**: Different allocation strategies per regime

### 📈 Statistical Tail Risk Modeling
- **Proper Statistical Definitions**: P[return < -2σ] downside risk, P[|return| > 2σ] extreme moves
- **Distribution Fitting**: Automatic Normal/Student-t/Empirical distribution selection
- **Signal-Aware Risk**: Tail risk adjustments based on momentum, sentiment, and regime
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

# Run with adaptive parameter learning (RECOMMENDED)
python -m quant.adaptive_main

# Or run with static parameters (legacy)
python -m quant.main

# View report
cat reports/daily_$(date +%Y-%m-%d).md
```

### Adaptive vs Static Analysis

**Adaptive Learning** (`adaptive_main.py`):
- Parameters learned from historical data
- Stock-specific signal effectiveness
- Automatic data quality monitoring
- Enhanced diagnostics and learning summary

**Static Configuration** (`main.py`):
- Uses hardcoded parameters from settings.yaml
- Consistent behavior for validation
- Faster execution (no learning phase)

## Sample Output

### Adaptive Learning Summary
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

⚠️ Saknar prisdata för 3 tickers: ERIC-B, TELIA, VOLV-B
```

### Daily Report Format
```markdown
# Roi Daily Brief — 2025-09-16

## Köp-rekommendationer (3)

- **GOOGL** @ 251.61 — E[r]_1d: +0.06% | E[R]_21d: +1.28% | Pr(↑): 80% | Confidence: 0.50 | σ: 0.20% | Downside VaR_1d: 🟢 2.1%
  *Signalbidrag (normaliserade): Trend(0.30), Momentum(0.49), Sentiment(0.21)*
  *Extremrörelser P[|r| > 2σ]: 4.3%*

### 📊 Aktuell Marknadsregim: **Neutral** (67% säkerhet)
Sidledes rörelse med måttlig volatilitet och blandat sentiment

### Signalbidrag (Regime-justerad viktning)
- **Trend (SMA-200):** 30% — Långsiktig trendriktning
- **Momentum (252d):** 49% — Årsmomemtum-ranking
- **Sentiment:** 21% — Nyhetsanalys (svensk media)
```

The system generates comprehensive daily reports with:

- **Adaptive Recommendations**: Buy/Sell/Hold with learned parameters
- **Statistical Tail Risk**: P[return < -2σ] and P[|return| > 2σ] measures
- **Learning Diagnostics**: Parameter changes and confidence intervals
- **Market Context**: Current regime and adaptive signal weightings
- **Data Quality Monitoring**: Missing data and potential issues

## Development Status

✅ **Completed Features**:
- **Adaptive Bayesian Engine**: Stock-specific learning and parameter estimation
- **Statistical Tail Risk**: P[return < -2σ] and P[|return| > 2σ] proper definitions
- **Data-Driven Parameters**: Signal scaling, regime adjustments, risk parameters
- **Regime Detection**: HMM-based classification with learned effectiveness multipliers
- **Monte Carlo Risk Simulation**: 12-month probability scenarios
- **Portfolio Management**: Risk budgeting with regime diversification
- **Data Quality Monitoring**: Automatic detection of missing/inconsistent data
- Comprehensive daily reporting

🔄 **In Development**:
- Backtesting framework
- Machine learning signal enhancement
- Real-time data integration
- Advanced portfolio optimization

## License

This is a proof-of-concept system for educational and research purposes.