# Getting Started

## Installation

### Prerequisites
- Python 3.8 or higher
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
pip install pandas numpy yfinance pyyaml
```

## Quick Start

### Basic Usage
```bash
# Run complete analysis
python -m quant.main

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

### Bayesian Recommendations
```
## Köp-förslag
- **TSLA** @ 410.04 — E[r]: +0.06% | Pr(↑): 82% | Confidence: 0.51 | σ: 0.21 | Tail Risk: 🟡 0.41
  *Signals: Trend(0.30), Momentum(0.49), Sentiment(0.21)*
```

### Market Analysis
```
### 📊 Aktuell Marknadsregim: **Bear** (60% säkerhet)
**Genomsnittlig tail risk:** 0.25 (0=låg, 1=hög)
**Risk-fördelning:** 🟢17 / 🟡1 / 🔴0 aktier
```

### Portfolio Status
```
**Portföljvärde:** 100,000 SEK
**Investerat kapital:** 0 SEK (0.0%)
**Antal positioner:** 0
```

## Understanding the Output

### Signal Interpretation
- **E[r]**: Expected daily return from Bayesian analysis
- **Pr(↑)**: Probability of positive movement (21-day horizon)
- **Confidence**: Decision confidence (0-1, higher = more certain)
- **σ**: Uncertainty measure (0-1, lower = more certain)
- **Tail Risk**: Heavy-tail risk score (🟢 low, 🟡 medium, 🔴 high)

### Decision Criteria
- **Buy**: Pr(↑) ≥ 58% + E[r] ≥ 0.05% + low uncertainty
- **Sell**: Pr(↑) ≤ 40% or E[r] ≤ -0.05% with low uncertainty
- **Hold**: All other cases or high uncertainty

## Next Steps

1. **Review Configuration**: Customize settings in `settings.yaml`
2. **Explore Documentation**: Read detailed component documentation
3. **Analyze Results**: Review daily reports and inspect the logged artifacts under `data/recommendation_logs/` and `data/portfolio/`
4. **Backtest**: Implement historical performance analysis
5. **Customize**: Modify universe, add new signals, or adjust rules
