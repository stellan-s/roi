# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
This is "ROI" - an advanced quantitative trading/investment analysis system written in Python. It's a sophisticated proof-of-concept that combines Bayesian signal processing, regime detection, portfolio management, and risk modeling to generate actionable trading recommendations for Swedish stocks.

## Architecture
The system follows a modular pipeline architecture with these main components:

### Core Data & Features
- **Data Layer** (`quant/data_layer/`):
  - `prices.py` - Fetches price data using yfinance, caches as parquet files
  - `news.py` - Fetches news feeds for sentiment analysis

- **Features** (`quant/features/`):
  - `technical.py` - Computes SMA, momentum, and ranking indicators
  - `sentiment.py` - Basic sentiment scoring from news feeds

### Advanced Analytics
- **Bayesian Engine** (`quant/bayesian/`):
  - `signal_engine.py` - Bayesian signal processing with probabilistic outputs
  - `integration.py` - Signal combination and probability calculations
  - `adaptive_integration.py` - Self-calibrating Bayesian engine with parameter learning

- **Regime Detection** (`quant/regime/`):
  - `detector.py` - Market regime classification (Bull/Bear/Neutral)

- **Advanced Risk Management** (`quant/risk/`):
  - `heavy_tail.py` - Heavy-tailed risk modeling with Student-t distributions
  - `tail_risk_calculator.py` - **NEW**: Statistical tail risk with P[return < -2σ] calculations
  - `analytics.py` - Risk analytics and metrics
  - **Statistical Tail Risk**: Proper probability-based risk measures (not heuristic scores)
  - **VaR & Expected Shortfall**: Value-at-Risk and conditional tail expectations
  - **Distribution Fitting**: Normal, Student-t, and empirical distribution modeling

### Portfolio & Trading
- **Portfolio Management** (`quant/portfolio/`):
  - `state.py` - Portfolio state tracking and persistence
  - `rules.py` - Portfolio allocation and risk management rules

- **Policy Engine** (`quant/policy_engine/`):
  - `rules.py` - Decision logic combining all signals into buy/sell/hold recommendations

### Analytics & Backtesting
- **Parameter Calibration** (`quant/calibration/` & `quant/adaptive/`):
  - `parameter_estimator.py` - Data-driven parameter estimation
  - Automatic calibration of model parameters from historical data

- **Comprehensive Backtesting Framework** (`quant/backtesting/`):
  - `framework.py` - Full backtesting engine with walk-forward analysis
  - `attribution.py` - Performance attribution analysis decomposing adaptive learning contributions
  - `cli.py` - Command-line interface for backtests with comparison modes
  - **Features**: Train/test splits, rolling windows, statistical significance testing
  - **Metrics**: Sharpe ratio, max drawdown, win rates, tail risk accuracy

- **Advanced Performance Attribution**:
  - Component-wise performance decomposition (signal normalization, regime adjustments, etc.)
  - Statistical significance testing for each component contribution
  - Adaptive vs. static configuration comparison with confidence intervals

- **Reports** (`quant/reports/`):
  - `daily_brief.py` - Generates comprehensive markdown reports with advanced analytics

## Configuration
The system uses YAML configuration files in `quant/config/`:
- `settings.yaml` - Main configuration (Bayesian parameters, risk thresholds, portfolio constraints)
- `universe.yaml` - Stock tickers to analyze (currently Swedish stocks)

## Running the System

### Main Execution Options
- **Standard Pipeline**: `python -m quant.main` (from project root)
- **Adaptive Pipeline**: `python -m quant.adaptive_main` (with parameter learning)
- **Backtesting Framework**: `python -m quant.backtest_runner` (comprehensive historical analysis)
  - **Single run**: Test adaptive vs. static configurations
  - **Comparison mode**: Full performance attribution analysis
  - **CLI options**: Custom date ranges, engine types, attribution depth

### Environment Setup
- **Python environment**: Uses `.venv` virtual environment with Python 3.12
- **Key dependencies**: yfinance, pandas, numpy, yaml, beautifulsoup4, feedparser, duckdb
- **No requirements.txt**: Dependencies managed in virtual environment

## Advanced Features

### Portfolio Management
- **Real-time portfolio tracking**: JSON-based state persistence
- **Actionable recommendations**: Concrete buy/sell decisions with position sizing
- **Risk-adjusted allocation**: Regime-aware position sizing and diversification
- **Transaction cost modeling**: Filters unprofitable trades

### Bayesian Signal Processing
- **Probabilistic outputs**: P(price_up) probabilities for each stock
- **Expected return estimation**: Data-driven return forecasts
- **Uncertainty quantification**: Confidence intervals and uncertainty measures
- **Multi-signal fusion**: Technical, sentiment, and regime signals

### Advanced Risk Management
- **Statistical Tail Risk**: Proper P[return < -2σ] calculations using fitted distributions
- **Heavy-tail modeling**: Student-t distributions with degrees of freedom estimation
- **VaR & Expected Shortfall**: 1-day Value-at-Risk with color-coded severity (🟢🟡🔴)
- **Extreme Movement Modeling**: P[|return| > 2σ] for two-sided tail risk
- **Regime-adaptive allocation**: Conservative positioning in bear markets
- **Distribution Fitting**: Automatic selection between Normal, Student-t, and empirical distributions
- **Diversification controls**: Limits single position and regime exposure

## Data Flow
1. **Configuration Loading**: YAML settings and universe definition
2. **Data Acquisition**: Price data (cached) + news feeds for sentiment
3. **Feature Engineering**: Technical indicators + sentiment scores
4. **Regime Detection**: Market state classification
5. **Bayesian Processing**: Signal fusion with probabilistic outputs
6. **Risk Assessment**: Tail risk and uncertainty quantification
7. **Portfolio Optimization**: Position sizing with risk constraints
8. **Decision Generation**: Buy/sell/hold recommendations with sizing
9. **Trade Execution**: Portfolio state updates and trade simulation
10. **Reporting**: Comprehensive markdown reports with analysis

## Key Entry Points
- `quant/main.py` - Standard pipeline with fixed parameters
- `quant/adaptive_main.py` - Adaptive pipeline with parameter learning
- `quant/backtest_runner.py` - Backtesting and historical analysis

## Output

### Enhanced Daily Reports (`reports/daily_YYYY-MM-DD.md`)
- **Advanced Analytics**: E[r]_1d (daily expected return), E[R]_21d (21-day aggregated return)
- **Probabilistic Forecasts**: Pr(↑) probability of positive returns with confidence intervals
- **Statistical Risk Metrics**:
  - Downside VaR_1d (1-day Value-at-Risk with color coding)
  - P[|r| > 2σ] (extreme movement probabilities)
  - Heavy-tail modeling with realistic volatility estimates
- **Signal Attribution**: Normalized signal contributions (Trend, Momentum, Sentiment weights)
- **Decision Rationale**: Explicit reasons for buy/sell/hold decisions with uncertainty thresholds
- **Portfolio Integration**: Current holdings, P&L tracking, and position sizing recommendations

### Portfolio & Backtest Outputs
- **Portfolio State**: JSON files tracking holdings, P&L, and trades with historical performance
- **Backtesting Results**:
  - Comprehensive performance metrics (Sharpe, max drawdown, win rates)
  - Performance attribution analysis with statistical significance
  - Component contribution decomposition (adaptive learning vs. static parameters)

## Development Notes
- **Parameter Calibration**: Many hardcoded parameters identified in `parameter_analysis.md`
- **Portfolio Features**: Full feature list documented in `PORTFOLIO_FEATURES.md`
- **Regime Detection**: Bull/Bear/Neutral classification influences all decisions
- **Risk-First Design**: All recommendations filtered through risk management layer