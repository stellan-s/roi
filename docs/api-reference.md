# API Reference

## Overview

This document provides a comprehensive reference for all classes, functions, and interfaces in the ROI quantitative trading system.

## Core Data Structures

### Signal Processing

#### SignalType (Enum)
```python
class SignalType(Enum):
    TREND = "trend"
    MOMENTUM = "momentum"
    SENTIMENT = "sentiment"
```

#### SignalPrior
```python
@dataclass
class SignalPrior:
    effectiveness: float     # Base probability (>0.5 = positive edge)
    confidence: float       # Certainty about effectiveness (0-1)
    observations: int       # Pseudo-observations for Bayesian updating
```

#### SignalPosterior
```python
@dataclass
class SignalPosterior:
    alpha: float           # Beta distribution alpha parameter
    beta: float            # Beta distribution beta parameter
    mean: float            # Posterior mean effectiveness
    variance: float        # Posterior variance
    weight: float          # Signal weight in combination
```

#### SignalOutput
```python
@dataclass
class SignalOutput:
    expected_return: float         # E[r] daily expected return
    prob_positive: float          # Pr(↑) probability of positive movement
    confidence_lower: float       # Lower confidence bound
    confidence_upper: float       # Upper confidence bound
    uncertainty: float            # Overall uncertainty (0-1)
    signal_weights: Dict[SignalType, float]  # Individual signal contributions
```

### Regime Detection

#### MarketRegime (Enum)
```python
class MarketRegime(Enum):
    BULL = "bull"
    BEAR = "bear"
    NEUTRAL = "neutral"
```

#### RegimeCharacteristics
```python
@dataclass
class RegimeCharacteristics:
    volatility_regime: str     # "low", "medium", "high"
    return_regime: str        # "positive", "negative", "neutral"
    trend_regime: str         # "up", "down", "sideways"
    market_regime: MarketRegime
```

### Risk Modeling

#### TailRiskMetrics
```python
@dataclass
class TailRiskMetrics:
    confidence_level: float        # VaR confidence level (e.g., 0.95)
    time_horizon_days: int        # Analysis period
    var_normal: float             # Normal distribution VaR
    var_student_t: float          # Student-t VaR
    cvar_student_t: float         # Conditional VaR
    evt_var: float               # EVT-based VaR
    evt_return_level: float      # EVT return level
    degrees_of_freedom: float    # Student-t degrees of freedom
    tail_index: float            # EVT tail index
    tail_risk_multiplier: float  # Heavy-tail vs normal multiplier
    extreme_event_probability: float  # P(extreme event)
```

#### MonteCarloResults
```python
@dataclass
class MonteCarloResults:
    n_simulations: int           # Number of MC simulations
    time_horizon_months: int     # Simulation horizon

    # Probability targets
    prob_positive: float         # P(return > 0%)
    prob_plus_10: float         # P(return > +10%)
    prob_plus_20: float         # P(return > +20%)
    prob_plus_30: float         # P(return > +30%)
    prob_minus_10: float        # P(return < -10%)
    prob_minus_20: float        # P(return < -20%)
    prob_minus_30: float        # P(return < -30%)

    # Distribution moments
    mean_return: float          # Expected return
    median_return: float        # Median return
    std_return: float           # Return volatility
    skewness: float            # Distribution skewness
    kurtosis: float            # Excess kurtosis

    # Extreme scenarios
    percentile_1: float        # 1st percentile (worst case)
    percentile_5: float        # 5th percentile
    percentile_95: float       # 95th percentile
    percentile_99: float       # 99th percentile (best case)
```

### Portfolio Management

#### PortfolioPosition
```python
@dataclass
class PortfolioPosition:
    ticker: str
    weight: float              # Portfolio weight (0-1)
    decision: str             # Buy/Sell/Hold
    expected_return: float    # E[r] daily
    prob_positive: float      # Pr(↑)
    regime: str              # Market regime
    decision_confidence: float # Decision confidence (0-1)
```

#### PortfolioConstraints
```python
@dataclass
class PortfolioConstraints:
    max_weight_per_stock: float = 0.10
    max_single_regime_exposure: float = 0.85
    min_regime_diversification: bool = True
    pre_earnings_freeze_days: int = 5
    trade_cost_bps: int = 5
    min_portfolio_positions: int = 3
    bear_market_allocation: float = 0.60
```

## Bayesian Signal Engine

### BayesianSignalEngine

#### Class Definition
```python
class BayesianSignalEngine:
    def __init__(self, config: Optional[Dict] = None)
```

#### Core Methods

##### combine_signals
```python
def combine_signals(self,
                   signals: Dict[SignalType, float],
                   regime_adjustment: float = 1.0) -> SignalOutput:
    """
    Combine multiple signals using Bayesian inference.

    Args:
        signals: Dict mapping SignalType to signal strength (-1 to +1)
        regime_adjustment: Overall regime-based adjustment factor

    Returns:
        SignalOutput with E[r], Pr(↑), confidence bounds, and weights
    """
```

##### update_beliefs
```python
def update_beliefs(self,
                  signals: Dict[SignalType, float],
                  actual_return: float,
                  horizon_days: int) -> None:
    """
    Update Bayesian beliefs based on actual performance.

    Args:
        signals: Signal values when prediction was made
        actual_return: Realized return over horizon
        horizon_days: Investment horizon
    """
```

##### get_signal_diagnostics
```python
def get_signal_diagnostics(self) -> pd.DataFrame:
    """
    Get signal performance diagnostics.

    Returns:
        DataFrame with signal effectiveness metrics
    """
```

### BayesianPolicyEngine

#### Class Definition
```python
class BayesianPolicyEngine:
    def __init__(self, config: Optional[Dict] = None)
```

#### Core Methods

##### bayesian_score
```python
def bayesian_score(self,
                  tech: pd.DataFrame,
                  senti: pd.DataFrame,
                  prices: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Generate Bayesian scores and decisions for all stocks.

    Args:
        tech: Technical features (above_sma, mom_rank)
        senti: Sentiment scores
        prices: Price data for regime detection

    Returns:
        DataFrame with decisions, probabilities, and risk metrics
    """
```

## Regime Detection

### RegimeDetector

#### Class Definition
```python
class RegimeDetector:
    def __init__(self, config: Optional[Dict] = None)
```

#### Core Methods

##### detect_regime
```python
def detect_regime(self,
                 price_data: pd.Series) -> Tuple[MarketRegime, Dict[MarketRegime, float], Dict]:
    """
    Detect current market regime.

    Args:
        price_data: Historical price series

    Returns:
        Tuple of (regime, regime_probabilities, diagnostics)
    """
```

##### get_regime_adjustments
```python
def get_regime_adjustments(self, regime: MarketRegime) -> Dict[str, float]:
    """
    Get signal adjustment multipliers for given regime.

    Args:
        regime: Current market regime

    Returns:
        Dict mapping signal names to adjustment factors
    """
```

##### get_regime_explanation
```python
def get_regime_explanation(self,
                          regime: MarketRegime,
                          probabilities: Dict[MarketRegime, float],
                          diagnostics: Dict) -> str:
    """
    Generate human-readable regime explanation.

    Args:
        regime: Current regime
        probabilities: Regime probabilities
        diagnostics: Regime diagnostics

    Returns:
        Formatted explanation string
    """
```

## Risk Modeling

### HeavyTailRiskModel

#### Class Definition
```python
class HeavyTailRiskModel:
    def __init__(self, config: Optional[Dict] = None)
```

#### Core Methods

##### fit_heavy_tail_distribution
```python
def fit_heavy_tail_distribution(self, returns: pd.Series) -> Dict:
    """
    Fit Student-t distribution to return series.

    Args:
        returns: Historical return series

    Returns:
        Dict with degrees_of_freedom, location, scale, sample_size
    """
```

##### fit_extreme_value_theory
```python
def fit_extreme_value_theory(self, returns: pd.Series) -> Dict:
    """
    Fit EVT models to upper and lower tails.

    Args:
        returns: Historical return series

    Returns:
        Dict with upper_tail and lower_tail EVT parameters
    """
```

##### calculate_tail_risk_metrics
```python
def calculate_tail_risk_metrics(self,
                               returns: pd.Series,
                               confidence_level: float = 0.95,
                               time_horizon_days: int = 21) -> TailRiskMetrics:
    """
    Calculate comprehensive tail risk metrics.

    Args:
        returns: Historical returns
        confidence_level: VaR confidence level
        time_horizon_days: Risk horizon

    Returns:
        TailRiskMetrics object with all risk measures
    """
```

##### monte_carlo_simulation
```python
def monte_carlo_simulation(self,
                          expected_return: float,
                          volatility: float,
                          tail_parameters: Dict,
                          time_horizon_months: int = 12,
                          n_simulations: int = 10000) -> MonteCarloResults:
    """
    Run Monte Carlo simulation for probability scenarios.

    Args:
        expected_return: Annual expected return
        volatility: Annual volatility
        tail_parameters: Student-t parameters
        time_horizon_months: Simulation horizon
        n_simulations: Number of simulations

    Returns:
        MonteCarloResults with probability targets and percentiles
    """
```

### RiskAnalytics

#### Class Definition
```python
class RiskAnalytics:
    def __init__(self, config: Optional[Dict] = None)
```

#### Core Methods

##### analyze_position_risk
```python
def analyze_position_risk(self,
                         ticker: str,
                         price_history: pd.Series,
                         expected_return: float,
                         time_horizon_months: int = 12) -> PortfolioRiskProfile:
    """
    Complete risk analysis for individual position.

    Args:
        ticker: Stock ticker
        price_history: Historical prices
        expected_return: Annual expected return
        time_horizon_months: Analysis horizon

    Returns:
        Complete PortfolioRiskProfile
    """
```

##### stress_test_portfolio
```python
def stress_test_portfolio(self,
                         portfolio_weights: Dict[str, float],
                         risk_profiles: Dict[str, PortfolioRiskProfile],
                         scenarios: List[str] = None) -> Dict[str, Dict]:
    """
    Stress test portfolio against predefined scenarios.

    Args:
        portfolio_weights: Position weights
        risk_profiles: Risk profiles for each position
        scenarios: Stress scenarios to test

    Returns:
        Dict with stress test results per scenario
    """
```

##### generate_risk_summary
```python
def generate_risk_summary(self,
                         portfolio_weights: Dict[str, float],
                         risk_profiles: Dict[str, PortfolioRiskProfile]) -> Dict:
    """
    Generate portfolio-level risk summary.

    Args:
        portfolio_weights: Position weights
        risk_profiles: Individual position risk profiles

    Returns:
        Dict with portfolio risk metrics and assessment
    """
```

## Portfolio Management

### PortfolioManager

#### Class Definition
```python
class PortfolioManager:
    def __init__(self, config: Optional[Dict] = None)
```

#### Core Methods

##### apply_portfolio_rules
```python
def apply_portfolio_rules(self, decisions: pd.DataFrame) -> pd.DataFrame:
    """
    Apply portfolio-level rules to Bayesian decisions.

    Args:
        decisions: DataFrame with Bayesian decisions

    Returns:
        DataFrame with portfolio-adjusted decisions and weights
    """
```

##### get_portfolio_summary
```python
def get_portfolio_summary(self, decisions: pd.DataFrame) -> Dict:
    """
    Generate portfolio-level summary and diagnostics.

    Args:
        decisions: Portfolio decisions DataFrame

    Returns:
        Dict with portfolio metrics and distribution
    """
```

### PortfolioState

#### Class Definition
```python
class PortfolioState:
    def __init__(self, initial_cash: float = 100000)
```

#### Core Methods

##### execute_trades
```python
def execute_trades(self,
                  trades: List[TradeRecommendation],
                  current_prices: Dict[str, float]) -> List[Transaction]:
    """
    Execute portfolio trades and update state.

    Args:
        trades: List of trade recommendations
        current_prices: Current market prices

    Returns:
        List of executed transactions
    """
```

##### update_portfolio_values
```python
def update_portfolio_values(self, current_prices: Dict[str, float]) -> None:
    """
    Update portfolio values with current market prices.

    Args:
        current_prices: Dict mapping tickers to current prices
    """
```

##### get_portfolio_summary
```python
def get_portfolio_summary(self) -> Dict:
    """
    Get current portfolio summary.

    Returns:
        Dict with portfolio value, holdings, and performance metrics
    """
```

## Data Layer

### fetch_prices
```python
def fetch_prices(tickers: List[str],
                cache_dir: str,
                lookback_days: int) -> pd.DataFrame:
    """
    Fetch and cache price data for given tickers.

    Args:
        tickers: List of stock tickers
        cache_dir: Cache directory path
        lookback_days: Historical data period

    Returns:
        DataFrame with ticker, date, open, high, low, close, volume
    """
```

### fetch_news_sentiment
```python
def fetch_news_sentiment(feed_urls: List[str],
                        tickers: List[str]) -> pd.DataFrame:
    """
    Fetch news and compute sentiment scores.

    Args:
        feed_urls: List of RSS feed URLs
        tickers: List of tickers to analyze

    Returns:
        DataFrame with ticker, date, sent_score
    """
```

## Feature Engineering

### compute_technical_features
```python
def compute_technical_features(prices: pd.DataFrame,
                              sma_window: int = 200,
                              momentum_window: int = 252) -> pd.DataFrame:
    """
    Compute technical indicators from price data.

    Args:
        prices: Price DataFrame
        sma_window: Simple moving average window
        momentum_window: Momentum calculation window

    Returns:
        DataFrame with ticker, date, close, above_sma, mom_rank
    """
```

## Configuration

### load_yaml
```python
def load_yaml(filename: str) -> Dict:
    """
    Load YAML configuration file.

    Args:
        filename: Path to YAML file (relative to quant/config/)

    Returns:
        Dict with configuration parameters
    """
```

## Reporting

### save_daily_markdown
```python
def save_daily_markdown(decisions: pd.DataFrame,
                       outdir: str,
                       portfolio_summary: dict = None) -> str:
    """
    Generate daily markdown report.

    Args:
        decisions: Portfolio decisions DataFrame
        outdir: Output directory
        portfolio_summary: Optional portfolio summary

    Returns:
        Path to generated report file
    """
```

## Error Handling

### Common Exceptions

#### InsufficientDataError
```python
class InsufficientDataError(Exception):
    """Raised when insufficient historical data for analysis."""
    pass
```

#### RegimeDetectionError
```python
class RegimeDetectionError(Exception):
    """Raised when regime detection fails."""
    pass
```

#### RiskModelingError
```python
class RiskModelingError(Exception):
    """Raised when risk modeling encounters errors."""
    pass
```

## Usage Examples

### Complete Workflow
```python
from quant.main import main
from quant.bayesian.integration import BayesianPolicyEngine
from quant.portfolio.rules import PortfolioManager
from quant.risk.analytics import RiskAnalytics

# Load configuration
config = load_yaml("settings.yaml")

# Initialize components
bayesian_engine = BayesianPolicyEngine(config)
portfolio_manager = PortfolioManager(config)
risk_analytics = RiskAnalytics(config)

# Run complete analysis
decisions = main()  # Generates daily report
```

### Custom Analysis
```python
# Get price and feature data
prices = fetch_prices(tickers, cache_dir, lookback_days)
tech_features = compute_technical_features(prices)
sentiment_data = fetch_news_sentiment(feed_urls, tickers)

# Generate Bayesian decisions
decisions = bayesian_engine.bayesian_score(tech_features, sentiment_data, prices)

# Apply portfolio rules
portfolio_decisions = portfolio_manager.apply_portfolio_rules(decisions)

# Risk analysis
risk_profiles = {}
for ticker in tickers:
    ticker_prices = prices[prices['ticker'] == ticker]['close']
    expected_return = decisions[decisions['ticker'] == ticker]['expected_return'].iloc[-1]
    risk_profiles[ticker] = risk_analytics.analyze_position_risk(
        ticker, ticker_prices, expected_return
    )

# Generate reports
portfolio_summary = portfolio_manager.get_portfolio_summary(portfolio_decisions)
```

## Version Information

- **API Version**: 1.0.0
- **Python Version**: 3.8+
- **Dependencies**: pandas, numpy, yfinance, pyyaml
- **Last Updated**: 2025-09-15