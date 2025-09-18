# Getting Started

## Installation

### Prerequisites
- Python 3.12 or higher
- Virtual environment (recommended)

### Setup
```bash
# Clone the repository (if applicable)
git clone <repository-url>
cd roi

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install pandas numpy yfinance pyyaml beautifulsoup4 feedparser duckdb
```

## Quick Start

### Basic Usage
```bash
# Run complete analysis with adaptive learning (RECOMMENDED)
python -m quant.adaptive_main

# Run with static parameters (legacy)
python -m quant.main

# Run comprehensive backtesting
python -m quant.backtest_runner

# View generated report
cat reports/daily_$(date +%Y-%m-%d).md
```

### Configuration
Edit `quant/config/settings.yaml` to customize:
- Bayesian decision thresholds
- Risk modeling parameters
- Portfolio management rules
- Data sources and lookback periods

Edit `quant/config/universe.yaml` to modify the stock universe.

## Sample Output

The system generates a comprehensive daily report including:

### Enhanced Bayesian Recommendations
```
## Buy Suggestions
- **TSLA** @ 418.93 — E[r]_1d: +0.14% | E[R]_21d: +3.07% | Pr(↑): 92% | Confidence: 0.59 | σ: 31.03% | Downside VaR_1d: 🟡 2.8%
  *Signal contributions (normalised): Trend(0.25), Momentum(0.69), Sentiment(0.06)*
  *Extreme move probability P[|r| > 2σ]: 6.0%*
```

### Market Analysis
```
### 📊 Current Market Regime: **Bear** (60% confidence)
**Average tail risk:** 0.25 (0=low, 1=high)
**Risk distribution:** 🟢17 / 🟡1 / 🔴0 stocks
```

### Portfolio Status
```
**Portfolio value:** 100,000 SEK
**Invested capital:** 0 SEK (0.0%)
**Number of positions:** 0
```

## Understanding the Output

### Enhanced Signal Interpretation
- **E[r]_1d**: Expected daily return from Bayesian analysis
- **E[R]_21d**: Expected 21-day aggregated return
- **Pr(↑)**: Probability of positive movement (21-day horizon)
- **Confidence**: Decision confidence (0-1, higher = more certain)
- **σ**: Daily volatility estimate (annualized %)
- **Downside VaR_1d**: 1-day Value-at-Risk with color coding (🟢 low, 🟡 medium, 🔴 high)
- **Signal Attribution**: Normalized contributions from Trend, Momentum, and Sentiment
- **Extreme Moves**: P[|return| > 2σ] probability of large movements

### Decision Criteria
- **Buy**: Pr(↑) ≥ 55% + E[r] ≥ 0.02% + uncertainty ≤ 0.50
- **Sell**: Pr(↑) ≤ 45% AND E[r] ≤ -0.02% + uncertainty ≤ 0.50
- **Hold**: All other cases or high uncertainty

**Note**: Decision thresholds have been calibrated from recent system fixes to ensure logical consistency and avoid contradictory recommendations.

## Next Steps

1. **Try Adaptive Analysis**: Run `python -m quant.adaptive_main` for parameter learning
2. **Review Configuration**: Customize settings in `settings.yaml`
3. **Run Backtesting**: Use `python -m quant.backtest_runner` for historical validation
4. **Explore Documentation**: Read detailed component documentation including:
   - [Backtesting Framework](backtesting.md) - Historical performance analysis
   - [Risk Modeling](risk-modeling.md) - Statistical tail risk calculations
   - [Bayesian Engine](bayesian-engine.md) - Adaptive signal processing
5. **Analyze Results**: Review daily reports with enhanced analytics and VaR indicators
6. **Performance Attribution**: Use backtesting to understand which components add value
7. **Customize**: Modify universe, add new signals, or adjust risk parameters
