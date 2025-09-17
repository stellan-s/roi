# ROI (Return on Intelligence) - Advanced Quantitative Trading System

## 🏗️ System Architecture Overview

**ROI** is a sophisticated quantitative trading and investment analysis system that combines Bayesian signal processing, regime detection, adaptive parameter learning, and statistical risk modeling to generate actionable trading recommendations for Swedish and international stocks.

### 🎯 Core Philosophy
The system is built on **data-driven decision making** with proper **uncertainty quantification**. Rather than relying on hardcoded parameters, ROI learns from historical data to calibrate its models, providing probabilistic forecasts with confidence intervals and statistical risk measures.

### 🔬 Scientific Foundation
- **Bayesian Inference**: Multi-signal fusion with proper probability theory
- **Statistical Risk Theory**: Heavy-tail modeling with Student-t distributions
- **Regime Detection**: Hidden Markov Models for market state classification
- **Portfolio Theory**: Risk-adjusted optimization with regime diversification
- **Machine Learning**: Adaptive parameter estimation and backtesting validation

## 🏛️ Detailed System Architecture

The system follows a sophisticated **modular pipeline architecture** with clear separation of concerns:

```
ROI Advanced Trading System
├── 📊 Data Layer (quant/data_layer/)
│   ├── prices.py          - Price data via yfinance with parquet caching
│   └── news.py            - News feeds and sentiment analysis
├── 🔧 Feature Engineering (quant/features/)
│   ├── technical.py       - SMA-200, momentum ranking, volatility
│   └── sentiment.py       - News sentiment scoring and normalization
├── 🧠 Bayesian Engine (quant/bayesian/)
│   ├── signal_engine.py   - Core Bayesian signal processing
│   ├── integration.py     - Multi-signal fusion and policy engine
│   └── adaptive_integration.py - Self-learning parameter estimation
├── 📈 Regime Detection (quant/regime/)
│   └── detector.py        - HMM-based Bull/Bear/Neutral classification
├── ⚡ Risk Modeling (quant/risk/)
│   ├── heavy_tail.py      - Student-t distributions and EVT
│   ├── tail_risk_calculator.py - Statistical P[return < -2σ] measures
│   └── analytics.py       - Portfolio risk metrics and VaR
├── 💼 Portfolio Management (quant/portfolio/)
│   ├── rules.py           - Allocation rules and risk constraints
│   └── state.py           - Portfolio tracking and trade execution
├── 🎯 Policy Engine (quant/policy_engine/)
│   └── rules.py           - Buy/Sell/Hold decision logic
├── 📊 Backtesting Framework (quant/backtesting/)
│   ├── framework.py       - Walk-forward backtesting infrastructure
│   ├── attribution.py    - Performance decomposition analysis
│   └── cli.py             - Command-line backtesting interface
├── 🔬 Adaptive Learning (quant/adaptive/ & quant/calibration/)
│   └── parameter_estimator.py - Data-driven parameter calibration
└── 📋 Reporting (quant/reports/)
    └── daily_brief.py     - Comprehensive markdown reports
```

## 🚀 Advanced Features & Capabilities

### 🧠 Adaptive Bayesian Signal Engine
- **🎯 Probabilistic Decision Making**: Multi-signal fusion using proper Bayesian inference
- **📊 Stock-Specific Learning**: Individual effectiveness calibration per ticker-signal combination
- **🔬 Data-Driven Parameters**: Replaces ~23 hardcoded values with empirically estimated parameters
- **🎲 Uncertainty Quantification**: Confidence intervals and uncertainty thresholds for all decisions
- **⚖️ Multi-Signal Integration**: Trend (SMA-200), momentum (252d), sentiment with adaptive scaling
- **🔄 Real-Time Adaptation**: Parameters continuously updated based on rolling historical performance

### 📊 Advanced Regime Detection
- **🎭 Market Context Awareness**: Real-time Bull/Bear/Neutral classification
- **🔗 HMM-Based Classification**: Hidden Markov Models with regime persistence modeling
- **⚖️ Signal Adaptation**: Dynamic signal weights based on regime-specific effectiveness
- **🛡️ Risk-Aware Positioning**: Regime-specific allocation strategies (60% max in bear markets)
- **📈 Regime Diversification**: Portfolio constraints prevent over-concentration in single regime

### 📊 Statistical Tail Risk Modeling
- **📐 Proper Statistical Definitions**:
  - **P[return < -2σ]**: Downside tail risk (main risk measure)
  - **P[|return| > 2σ]**: Extreme movement probability (volatility risk)
- **📈 Distribution Fitting**: Automatic Normal/Student-t/Empirical distribution selection with Jarque-Bera normality testing
- **🎯 Signal-Aware Risk**: Tail risk adjustments based on momentum, sentiment, and current regime
- **🎰 Monte Carlo Simulation**: 10,000-iteration scenarios for 21-day, 3-month, and 1-year horizons
- **🔥 Stress Testing**: Portfolio resilience under Black Monday, COVID crash, and regime shift scenarios
- **🎨 Visual Risk Indicators**: Color-coded VaR 🟢🟡🔴 based on statistical thresholds

### 💼 Sophisticated Portfolio Management
- **💰 Risk Budgeting**: Position sizing based on expected returns, uncertainty, and tail risk
- **🎯 Regime Diversification**: Max 85% allocation in any single regime
- **💸 Transaction Cost Optimization**: 3bps cost modeling filters unprofitable trades
- **🔄 Dynamic Rebalancing**: Continuous optimization with regime-aware constraints
- **📊 Portfolio Tracking**: Real-time P&L, position sizing, and trade simulation
- **🛡️ Risk Limits**: Max 10% per position, 30% in high tail-risk assets

### 🔬 Comprehensive Backtesting Framework
- **📈 Walk-Forward Analysis**: Train/test splits with rolling windows
- **🔍 Performance Attribution**: Component-wise decomposition of adaptive learning contributions
- **📊 Statistical Significance**: Confidence intervals and significance testing for all metrics
- **⚖️ Adaptive vs Static Comparison**: Direct performance comparison between learned and hardcoded parameters
- **📋 CLI Interface**: Command-line tools for batch backtesting and analysis
- **📊 Rich Metrics**: Sharpe ratio, max drawdown, win rates, tail risk accuracy, regime diversification

## Documentation Structure

- **[Getting Started](getting-started.md)** - Installation and quick start guide
- **[Configuration](configuration.md)** - Complete configuration reference
- **[Bayesian Engine](bayesian-engine.md)** - Signal processing and decision theory
- **[Regime Detection](regime-detection.md)** - Market classification system
- **[Risk Modeling](risk-modeling.md)** - Heavy-tail analysis and stress testing
- **[Portfolio Management](portfolio-management.md)** - Position sizing and optimization
- **[Backtesting Framework](backtesting.md)** - Historical performance analysis and attribution
- **[API Reference](api-reference.md)** - Complete function and class documentation
- **[Examples](examples.md)** - Usage examples and tutorials
- **[Glossary](glossary.md)** - Terms, formulas, and abbreviations explained

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.12+ with virtual environment
- Dependencies: pandas, numpy, yfinance, yaml, beautifulsoup4, feedparser, duckdb

```bash
# Setup environment (if not already done)
source .venv/bin/activate  # or activate .venv on Windows

# 🎯 RECOMMENDED: Run adaptive pipeline with parameter learning
python -m quant.adaptive_main

# 📊 Alternative: Run static pipeline (legacy, faster)
python -m quant.main

# 🔬 Comprehensive backtesting and performance attribution
python -m quant.backtest_runner

# 📈 View latest report
cat reports/daily_$(date +%Y-%m-%d).md

# 📊 Monitor portfolio state
cat data/portfolio/portfolio_state.json
```

### ⚙️ Configuration Files
- **`quant/config/settings.yaml`**: Main system configuration (Bayesian parameters, risk thresholds)
- **`quant/config/universe.yaml`**: Stock universe definition (Swedish + international stocks)

### 📊 Output Locations
- **`reports/daily_YYYY-MM-DD.md`**: Daily analysis reports with recommendations
- **`data/`**: Cached price data (parquet files) and portfolio state
- **`backtesting_results/`**: Backtesting results and performance attribution

### 🔬 Execution Modes

#### 🧠 Adaptive Learning (`adaptive_main.py`) - **RECOMMENDED**
- **📊 Data-Driven Parameters**: 23+ parameters learned from 1000+ days of historical data
- **🎯 Stock-Specific Calibration**: Individual signal effectiveness per ticker
- **🔍 Quality Monitoring**: Automatic detection of missing data and signal anomalies
- **📈 Learning Diagnostics**: Parameter change summary with confidence intervals
- **⚡ Bootstrap Validation**: 500-iteration confidence intervals for parameter estimates
- **🎲 Uncertainty Quantification**: Proper uncertainty thresholds based on historical performance

#### ⚙️ Static Configuration (`main.py`) - **VALIDATION MODE**
- **🔧 Hardcoded Parameters**: Uses fixed values from `settings.yaml`
- **🔄 Consistent Behavior**: Reproducible results for testing and validation
- **⚡ Faster Execution**: No parameter learning phase (~2min vs ~8min)
- **📊 Baseline Comparison**: Reference implementation for measuring adaptive improvements

#### 🔬 Backtesting Framework (`backtest_runner.py`)
- **📈 Historical Validation**: Walk-forward backtesting with train/test splits
- **⚖️ Performance Attribution**: Quantifies contribution of each adaptive learning component
- **📊 Statistical Testing**: Significance tests for Sharpe ratio improvements
- **🎯 Risk Accuracy**: Measures how well tail risk predictions matched actual outcomes

## 📊 Sample Output & Analytics

### 🧠 Adaptive Learning Diagnostics
```bash
=== ROI Adaptive Trading System ===
Loaded configuration for 47 tickers
Preparing historical data for parameter estimation...
Prepared 23,500 price observations, 1,247 sentiment observations
Initializing adaptive Bayesian engine...

=== Parameter Estimation Results ===
Top parameter changes from defaults:
  sentiment_scale_factor: 0.500 → 0.672 (+34.4%)
  momentum_scale_factor: 2.000 → 1.847 (-7.7%)
  bear_sentiment_effectiveness: 1.400 → 1.520 (+8.6%)
  bull_momentum_multiplier: 1.200 → 1.089 (-9.2%)
  regime_transition_persistence: 0.800 → 0.893 (+11.6%)

Learning Summary:
  Total parameters estimated: 23
  Significant changes (>10%): 8
  Average change: 12.3%

⚠️ Saknar prisdata för 3 tickers: ERIC-B.ST, TELIA.ST, VOLV-B.ST
📝 Sparade dagens rekommendationer till data/recommendation_logs/recommendations_2025-09-16.parquet
🧾 Loggade 7 simulerade affärer till data/recommendation_logs/simulated_trades_2025-09-16.json

Report generated: reports/daily_2025-09-16.md
Generated 47 recommendations

Top buy recommendations:
  TSLA: 92% prob, 0.59 confidence
  SAAB-B.ST: 92% prob, 0.58 confidence
```

### 📋 Daily Report Analytics
```markdown
# Roi Daily Brief — 2025-09-16

## 🏗️ System Architecture Overview
**ROI (Return on Intelligence)** is an advanced quantitative trading system that combines
Bayesian signal processing, regime detection, and heavy-tail risk modeling...

## Köp-förslag

- **TSLA** @ 421.62 — E[r]_1d: +0.14% | E[R]_21d: +3.07% | Pr(↑): 92% | Confidence: 0.59 | σ: 31.03% | Downside VaR_1d: 🟡 2.8% | **Vikt: 0.2%**
  *Signalbidrag (normaliserade): Trend(0.25), Momentum(0.69), Sentiment(0.06)*
  *Extremrörelser P[|r| > 2σ]: 6.0%*

- **SAAB-B.ST** @ 517.70 — E[r]_1d: +0.14% | E[R]_21d: +3.05% | Pr(↑): 92% | Confidence: 0.58 | σ: 33.78% | Downside VaR_1d: 🟡 2.7% | **Vikt: 0.3%**
  *Signalbidrag (normaliserade): Trend(0.28), Momentum(0.66), Sentiment(0.06)*
  *Extremrörelser P[|r| > 2σ]: 6.2%*

### 📊 Aktuell Marknadsregim: **Unknown** (33% säkerhet)
Ingen regime detekterad

### 📊 Portfolio Status
**Portföljvärde:** 100,000 SEK
**Investerat kapital:** 49,075 SEK (49.1%)
**Kvarvarande cash:** 50,925 SEK (50.9%)
**Antal positioner:** 2

**Rekommenderade trades:**
- KÖP TSLA: 0.2% (E[r]: +0.14%, Pr(↑): 92%)
- KÖP SAAB-B.ST: 0.3% (E[r]: +0.14%, Pr(↑): 92%)
```

### 📊 Comprehensive Report Features
The system generates sophisticated daily reports with:

- **🎯 Adaptive Recommendations**: Buy/Sell/Hold decisions with learned parameters
- **📊 Statistical Risk Metrics**:
  - **E[r]_1d**: Daily expected return
  - **E[R]_21d**: 21-day aggregated expected return
  - **Pr(↑)**: Probability of positive performance
  - **Downside VaR_1d**: 1-day Value-at-Risk with color coding 🟢🟡🔴
  - **P[|r| > 2σ]**: Extreme movement probabilities
- **🧠 Learning Diagnostics**: Parameter changes and confidence intervals
- **🎭 Market Context**: Current regime classification and adaptive signal weightings
- **🔍 Data Quality Monitoring**: Missing data detection and signal anomaly alerts
- **💼 Portfolio Integration**: Current holdings, P&L tracking, and recommended position sizes
- **⚠️ Decision Rationale**: Explicit explanations for hold decisions (e.g., uncertainty thresholds)

## 🚀 Development Status & Roadmap

### ✅ **Production-Ready Features**

#### 🧠 **Advanced Analytics Engine**
- **🎯 Adaptive Bayesian Engine**: Stock-specific learning with 23+ calibrated parameters
- **📊 Statistical Tail Risk**: Proper P[return < -2σ] and P[|return| > 2σ] definitions with Student-t fitting
- **🔬 Data-Driven Parameters**: Signal scaling, regime adjustments, risk parameters learned from 1000+ days
- **🎭 Regime Detection**: HMM-based Bull/Bear/Neutral classification with learned effectiveness multipliers
- **🎰 Monte Carlo Risk Simulation**: 10,000-iteration scenarios for multiple time horizons
- **💼 Portfolio Management**: Risk budgeting with regime diversification and transaction cost modeling
- **🔍 Data Quality Monitoring**: Automatic detection of missing/inconsistent data with Swedish language alerts

#### 📊 **Backtesting & Validation Framework**
- **📈 Comprehensive Backtesting**: Walk-forward analysis with train/test splits and rolling windows
- **🔍 Performance Attribution**: Component-wise decomposition quantifying adaptive learning contributions
- **📊 Statistical Significance Testing**: Confidence intervals and hypothesis testing for performance metrics
- **⚖️ Adaptive vs Static Comparison**: Direct comparison between learned and hardcoded parameters
- **📋 CLI Interface**: Command-line tools for batch backtesting and analysis
- **🎯 Risk Accuracy Validation**: Measures how well tail risk predictions matched actual outcomes

#### 📋 **Reporting & Monitoring**
- **📊 Enhanced Daily Reports**: VaR indicators, signal attribution, confidence intervals
- **🎯 System Architecture Documentation**: Comprehensive system overview in reports
- **📈 Real-time Portfolio Tracking**: JSON-based state persistence with P&L tracking
- **⚠️ Decision Rationale**: Explicit explanations for all trading decisions
- **🔄 Trade Logging**: Comprehensive audit trail for recommendations and simulated executions

### 🔄 **In Active Development**
- **🤖 Enhanced ML Signal Processing**: Deep learning for sentiment and technical pattern recognition
- **📡 Real-time Data Integration**: Live market data feeds and low-latency execution
- **🌍 Multi-Asset Optimization**: Cross-asset portfolio optimization (equities, bonds, commodities)
- **📊 Advanced Risk Models**: Copula-based dependency modeling and regime-dependent correlations
- **🔗 API Infrastructure**: RESTful API for external system integration

### 🎯 **Future Roadmap**
- **🏦 Institutional Features**: Multi-account management and compliance reporting
- **📱 Web Interface**: Interactive dashboard for portfolio monitoring and manual overrides
- **🔄 Live Trading**: Paper trading and eventually live execution via broker APIs
- **🌐 International Expansion**: Support for additional markets (EU, APAC)
- **📊 ESG Integration**: Environmental, Social, and Governance factor integration

## 📄 Technical Specifications

### 🛠️ **Technology Stack**
- **Language**: Python 3.12+
- **Core Libraries**: pandas, numpy, scipy, scikit-learn
- **Data Sources**: yfinance (prices), RSS feeds (news), custom sentiment analysis
- **Storage**: Parquet files (price cache), JSON (portfolio state), YAML (configuration)
- **Risk Modeling**: Student-t distributions, Extreme Value Theory, Monte Carlo simulation
- **ML Framework**: Bootstrap validation, HMM regime detection, Bayesian parameter estimation

### ⚡ **Performance Characteristics**
- **Data Processing**: 23,500+ price observations in ~30 seconds
- **Parameter Learning**: 23 parameters calibrated in ~5 minutes
- **Daily Analysis**: 47 stocks analyzed in ~2 minutes
- **Backtesting**: 2+ years of data processed in ~10 minutes
- **Memory Usage**: ~500MB peak during parameter learning
- **Storage**: ~100MB for 2 years of cached price data

### 🔒 **Security & Compliance**
- **No Real Trading**: Paper trading only - no real money at risk
- **Data Privacy**: No personal data collection, public market data only
- **Audit Trail**: Complete logging of all decisions and parameter changes
- **Reproducibility**: Deterministic results with seed control for validation

## 📄 License & Disclaimer

**Educational & Research Use Only**

This is a proof-of-concept system developed for educational and research purposes. It is NOT intended for live trading or investment advice. The system:

- Provides educational examples of quantitative finance techniques
- Demonstrates modern risk management and portfolio theory
- Serves as a research platform for academic and professional development
- Should NOT be used for actual investment decisions without proper due diligence

**Risk Warning**: Past performance does not guarantee future results. All trading involves risk of loss.