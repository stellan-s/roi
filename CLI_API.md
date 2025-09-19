# ROI Trading System - CLI API Documentation

## Overview

The ROI (Quantitative Trading/Investment Analysis) system provides multiple command-line interfaces for running different aspects of the trading pipeline. This document provides comprehensive documentation for all CLI entry points.

## Main CLI Entry Points

### 1. Standard Pipeline (`quant.main`)

**Command:** `python -m quant.main`

**Description:** Executes the standard trading pipeline with fixed configuration parameters.

**Features:**
- Fetches price and news data
- Computes technical indicators and sentiment scores
- Generates trading recommendations using static Bayesian engine
- Updates portfolio state and executes simulated trades
- Generates daily markdown reports

**Configuration:** Uses `quant/config/settings.yaml` and `quant/config/universe.yaml`

**Outputs:**
- Daily report: `reports/daily_YYYY-MM-DD.md`
- Portfolio state: `data/portfolio/portfolio_state.json`
- Trade logs: `data/recommendation_logs/`

### 2. Adaptive Pipeline (`quant.adaptive_main`)

**Command:** `python -m quant.adaptive_main`

**Description:** Executes the adaptive trading pipeline with parameter learning from historical data.

**Features:**
- **Parameter Calibration:** Learns optimal parameters from historical data
- **Adaptive Bayesian Engine:** Uses data-driven parameter estimates instead of hardcoded values
- **Enhanced Regime Detection:** Includes VIX and precious metals sentiment analysis
- **Factor Profile System:** Stock-specific factor weighting and regime adjustments
- **Risk Budgeting:** Advanced portfolio optimization with volatility targeting

**Key Differences from Standard Pipeline:**
- Longer initialization time due to parameter learning
- More sophisticated signal processing
- Better regime adaptation
- Enhanced risk management

**Outputs:** Same as standard pipeline plus:
- Parameter diagnostics in console output
- Learning summary statistics
- Factor profile diagnostics
- Risk budgeting optimization summary

### 3. Backtesting Runner (`quant.backtest_runner`)

**Command:** `python -m quant.backtest_runner`

**Description:** Comprehensive backtesting framework comparing adaptive vs. static configurations.

**Features:**
- Historical performance analysis
- Walk-forward backtesting
- Statistical significance testing
- Performance attribution analysis
- Trading activity analysis

**Process:**
1. Loads configuration and prepares historical data
2. Runs adaptive engine backtest for specified period
3. Runs static engine backtest for comparison
4. Generates comprehensive performance metrics
5. Saves results to JSON summary files

**Key Metrics:**
- Total return, annualized return, Sharpe ratio
- Maximum drawdown, win rate
- Trading frequency and efficiency
- Risk-adjusted performance

**Outputs:**
- Console performance summary
- JSON results: `backtesting_results/backtest_summary_YYYYMMDD_HHMMSS.json`

### 4. Advanced Backtesting CLI (`quant.backtesting.cli`)

**Command:** `python -m quant.backtesting.cli <subcommand> [options]`

**Description:** Command-line interface for advanced backtesting operations with argparse support.

#### Subcommands:

##### `single` - Run Single Engine Backtest
```bash
python -m quant.backtesting.cli single --start 2024-01-01 --end 2024-12-31 --engine adaptive --report results.md
```

**Options:**
- `--start` (required): Backtest start date (YYYY-MM-DD)
- `--end` (required): Backtest end date (YYYY-MM-DD)
- `--engine`: Engine type (`adaptive` or `static`, default: `adaptive`)
- `--report`: Optional path for Markdown report output

##### `compare` - Compare Adaptive vs Static
```bash
python -m quant.backtesting.cli compare --start 2024-01-01 --end 2024-12-31 --report comparison.md
```

**Options:**
- `--start` (required): Backtest start date (YYYY-MM-DD)
- `--end` (required): Backtest end date (YYYY-MM-DD)
- `--report`: Optional path for Markdown comparison report

##### `interactive` - Interactive Console
```bash
python -m quant.backtesting.cli interactive
```

**Features:**
- Menu-driven interface
- Date prompts for backtest periods
- Report generation prompts
- Error handling with user-friendly messages

### 5. Portfolio Dashboard (`quant.cli.dashboard`)

**Command:** `python -m quant.cli.dashboard`

**Description:** ANSI terminal dashboard summarising the current market regime, live portfolio snapshot, and latest buy/sell recommendations.

**Options:**
- `--state PATH` – Portfolio state JSON (default: `data/portfolio/current_state.json`)
- `--recommendations PATH` – Recommendations parquet (default: newest file in `data/recommendation_logs/`)
- `--limit INT` – Number of holdings to display (default: 8)
- `--bar-width INT` – Width of the position weight bars (default: 32)
- `--rec-limit INT` – Buy/Sell entries per category (default: 5)

## Configuration System

### Primary Configuration Files

1. **`quant/config/settings.yaml`** - Main system configuration
   - Bayesian engine parameters
   - Risk management settings
   - Regime detection thresholds
   - Portfolio optimization constraints

2. **`quant/config/universe.yaml`** - Stock universe definition
   - List of tickers to analyze
   - Market classifications

### Key Configuration Sections

#### Data Configuration
```yaml
data:
  cache_dir: "data"
  lookback_days: 750  # Historical data window
```

#### Bayesian Engine
```yaml
bayesian:
  time_horizon_days: 21
  decision_thresholds:
    buy_probability: 0.55
    sell_probability: 0.45
    min_expected_return: 0.0002
    max_uncertainty: 0.50
```

#### Risk Budgeting (Recent Updates)
```yaml
risk_budgeting:
  enabled: true
  max_position_weight: 0.10          # Max 10% per stock
  max_factor_concentration: 0.25     # Max 25% per factor category
  min_positions: 8                   # Force diversification
  uncertainty_penalty_factor: 3.0    # Increased penalty
  target_portfolio_volatility: 0.15  # 15% target volatility
  volatility_penalty_factor: 2.0     # Penalize high-vol positions
```

#### Signal Quality Filtering
```yaml
stock_factor_profiles:
  dynamic_filtering:
    cost_threshold: 0.002        # 20bps minimum expected return
    conviction_threshold: 0.6    # 60% minimum decision confidence
    max_positions: 20
    min_liquidity_mcap: 1000
```

## Environment Setup

### Python Environment
- **Version:** Python 3.12
- **Virtual Environment:** `.venv`
- **Dependencies:** Managed in virtual environment (no requirements.txt)

### Key Dependencies
- `yfinance` - Price data fetching
- `pandas`, `numpy` - Data manipulation
- `yaml` - Configuration management
- `beautifulsoup4`, `feedparser` - News processing
- `duckdb` - Data storage

### Directory Structure
```
roi/
├── quant/                    # Main package
│   ├── config/              # Configuration files
│   ├── data_layer/          # Data fetching modules
│   ├── features/            # Feature engineering
│   ├── bayesian/            # Bayesian engines
│   ├── regime/              # Regime detection
│   ├── risk/                # Risk modeling
│   ├── portfolio/           # Portfolio management
│   ├── backtesting/         # Backtesting framework
│   └── reports/             # Report generation
├── data/                    # Cached data and portfolio state
├── reports/                 # Generated daily reports
└── backtesting_results/     # Backtest results
```

## Usage Examples

### Daily Trading Analysis
```bash
# Standard pipeline
python -m quant.main

# Adaptive pipeline (recommended)
python -m quant.adaptive_main
```

### Historical Backtesting
```bash
# Quick backtest comparison
python -m quant.backtest_runner

# Specific period with advanced CLI
python -m quant.backtesting.cli compare --start 2024-01-01 --end 2024-06-30 --report Q1_Q2_analysis.md

# Single engine test
python -m quant.backtesting.cli single --start 2024-01-01 --end 2024-03-31 --engine adaptive --report adaptive_Q1.md

# Interactive mode
python -m quant.backtesting.cli interactive

# ANSI dashboard for portfolio snapshot, regime, and recommendations
python -m quant.cli.dashboard
python -m quant.cli.dashboard --state data/backtest_portfolio_adaptive/current_state.json --limit 5 --rec-limit 3
```

## Output Files

### Daily Reports (`reports/daily_YYYY-MM-DD.md`)
- **Market Analysis:** Regime detection and VIX analysis
- **Stock Recommendations:** Buy/sell/hold decisions with rationale
- **Risk Metrics:** Tail risk, VaR, statistical measures
- **Portfolio Summary:** Current holdings and P&L
- **Signal Attribution:** Technical, momentum, sentiment weights

### Portfolio State (`data/portfolio/portfolio_state.json`)
- Current holdings and cash position
- Historical P&L tracking
- Trade history and performance metrics

### Backtest Results (`backtesting_results/backtest_summary_*.json`)
- Performance metrics comparison (adaptive vs static)
- Trading activity analysis
- Statistical significance tests
- Component attribution analysis

## Error Handling and Debugging

### Common Issues
1. **Missing Data:** Ticker data unavailable or incomplete
2. **Configuration Errors:** Invalid YAML syntax or missing parameters
3. **Memory Issues:** Large datasets causing performance problems
4. **Network Issues:** Failed data fetching from external sources

### Debugging Features
- Data quality validation with health checks
- Duplicate recommendation detection
- Portfolio state consistency checks
- Parameter estimation diagnostics

### Logging
- Console output with progress indicators
- Trade and recommendation logging
- Error tracking with stack traces
- Data quality warnings

## Advanced Features

### Parameter Learning
- Historical data calibration
- Bootstrap confidence intervals
- Regime-specific parameter adaptation
- Signal effectiveness estimation

### Risk Management
- Statistical tail risk calculation
- Heavy-tail distribution modeling
- Volatility targeting
- Position concentration limits

### Regime Detection
- VIX integration for global market context
- Precious metals sentiment analysis
- Regime transition persistence modeling
- Multi-factor regime classification

### Portfolio Optimization
- Risk budgeting with factor constraints
- Transaction cost modeling
- Dynamic position sizing
- Regime-adaptive allocation

## Performance Considerations

### Data Caching
- Price data cached as parquet files
- News data cached with feed processing
- VIX and macro data cached separately

### Memory Management
- Latest data filtering to prevent explosion
- Efficient pandas operations
- Garbage collection for large datasets

### Processing Speed
- Parallel data fetching where possible
- Vectorized calculations
- Optimized technical indicator computation

## Best Practices

1. **Regular Execution:** Run adaptive pipeline daily for best results
2. **Backtesting:** Validate changes with historical analysis
3. **Configuration Management:** Version control configuration changes
4. **Data Quality:** Monitor data quality warnings
5. **Risk Monitoring:** Review risk metrics and portfolio exposure
6. **Performance Tracking:** Regularly analyze trading performance

## Future Enhancements

- Additional CLI parameters for runtime configuration
- More sophisticated backtesting strategies
- Enhanced reporting formats (HTML, PDF)
- Real-time execution capabilities
- Advanced performance attribution
- Multi-asset class support
