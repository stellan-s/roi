"""
Adaptive main entry point that learns parameters from historical data.

This version of the main pipeline calibrates model parameters from historical data
before running the daily analysis, replacing hardcoded values with data-driven estimates.
"""

import yaml
import pandas as pd
from pathlib import Path
from typing import Dict, Optional

from quant.data_layer.prices import fetch_prices
from quant.data_layer.news import fetch_news
from quant.features.technical import compute_technical_features
from quant.features.sentiment import naive_sentiment
from quant.bayesian.adaptive_integration import AdaptiveBayesianEngine
from quant.reports.daily_brief import save_daily_markdown

def load_configuration() -> Dict:
    """Load configuration from YAML files."""
    config_dir = Path(__file__).parent / "config"

    # Main settings
    with open(config_dir / "settings.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # Stock universe
    with open(config_dir / "universe.yaml", 'r') as f:
        universe = yaml.safe_load(f)

    config['universe'] = universe
    return config

def prepare_historical_data(config: Dict) -> tuple:
    """
    Prepare historical data for parameter estimation.

    Returns:
        tuple: (prices_df, sentiment_df, technical_df, returns_df)
    """
    print("Preparing historical data for parameter estimation...")

    # Load universe
    universe = config['universe']['tickers']

    # Fetch extended historical data for calibration
    calibration_lookback = config['data'].get('lookback_days', 500) * 2  # Extra data for calibration

    prices = fetch_prices(
        tickers=universe,
        cache_dir=config['data']['cache_dir'],
        lookback_days=calibration_lookback
    )

    news = fetch_news(
        feed_urls=config['signals']['news_feed_urls'],
        cache_dir=config['data']['cache_dir']
    )

    # Compute features
    tech = compute_technical_features(
        prices,
        config["signals"]["sma_long"],
        config["signals"]["momentum_window"]
    )

    senti = naive_sentiment(news, universe)

    # Create returns dataframe
    returns_df = prices.copy()
    returns_df = returns_df.sort_values(['ticker', 'date'])
    returns_df['return'] = returns_df.groupby('ticker')['close'].pct_change()
    returns_df = returns_df.dropna()

    print(f"Prepared {len(prices)} price observations, {len(senti)} sentiment observations")
    return prices, senti, tech, returns_df

def calibrate_adaptive_engine(config: Dict,
                             prices_df: pd.DataFrame,
                             sentiment_df: pd.DataFrame,
                             technical_df: pd.DataFrame,
                             returns_df: pd.DataFrame) -> AdaptiveBayesianEngine:
    """
    Create and calibrate the adaptive Bayesian engine.
    """
    print("Initializing adaptive Bayesian engine...")

    # Create adaptive engine
    engine = AdaptiveBayesianEngine(config)

    # Calibrate parameters from historical data
    engine.calibrate_parameters(
        prices_df=prices_df,
        sentiment_df=sentiment_df,
        technical_df=technical_df,
        returns_df=returns_df
    )

    return engine

def run_daily_analysis(config: Dict, engine: AdaptiveBayesianEngine) -> pd.DataFrame:
    """
    Run the daily analysis using the Bayesian engine.
    """
    print("Running daily analysis...")

    universe = config['universe']['tickers']

    # Fetch recent data for analysis
    prices = fetch_prices(
        tickers=universe,
        cache_dir=config['data']['cache_dir'],
        lookback_days=config['data']['lookback_days']
    )

    news = fetch_news(
        feed_urls=config['signals']['news_feed_urls'],
        cache_dir=config['data']['cache_dir']
    )

    # Compute current features
    tech = compute_technical_features(
        prices,
        config["signals"]["sma_long"],
        config["signals"]["momentum_window"]
    )

    senti = naive_sentiment(news, universe)

    # Generate recommendations using adaptive Bayesian engine
    recommendations = engine.bayesian_score_adaptive(tech, senti, prices)

    return recommendations

def main():
    """
    Main execution with adaptive parameter learning.
    """
    print("=== ROI Adaptive Trading System ===")

    # Load configuration
    config = load_configuration()
    print(f"Loaded configuration for {len(config['universe']['tickers'])} tickers")

    # Prepare historical data for calibration
    prices_df, sentiment_df, technical_df, returns_df = prepare_historical_data(config)

    # Calibrate adaptive engine
    engine = calibrate_adaptive_engine(config, prices_df, sentiment_df, technical_df, returns_df)

    # Show parameter diagnostics
    print("\n=== Parameter Estimation Results ===")
    diagnostics = engine.get_parameter_diagnostics()
    if not diagnostics.empty:
        print("Top parameter changes from defaults:")
        for _, row in diagnostics.head(10).iterrows():
            default_val = row['default_value']
            estimated_val = row['estimated_value']
            if abs(estimated_val - default_val) > 0.01:  # Show changes > 1%
                change_pct = ((estimated_val - default_val) / default_val) * 100 if default_val != 0 else 0
                print(f"  {row['parameter_name']}: {default_val:.3f} → {estimated_val:.3f} ({change_pct:+.1f}%)")
    else:
        print("No parameter diagnostics available")

    # Show learning summary
    learning_summary = engine.get_learning_summary()
    if learning_summary.get('status') == 'learning_complete':
        print(f"\nLearning Summary:")
        print(f"  Total parameters estimated: {learning_summary['parameter_changes']['total_parameters']}")
        print(f"  Significant changes (>10%): {learning_summary['parameter_changes']['significant_changes']}")
        print(f"  Average change: {learning_summary['parameter_changes']['avg_change_percent']:.1f}%")

    # Run daily analysis
    recommendations = run_daily_analysis(config, engine)

    # Generate report
    output_dir = config['run']['outdir']

    report_path = save_daily_markdown(
        recommendations,
        output_dir
    )

    print(f"\nReport generated: {report_path}")
    print(f"Generated {len(recommendations)} recommendations")

    # Show top recommendations
    buy_recs = recommendations[recommendations['decision'] == 'Buy'].nlargest(3, 'decision_confidence')
    if not buy_recs.empty:
        print(f"\nTop buy recommendations:")
        for _, rec in buy_recs.iterrows():
            print(f"  {rec['ticker']}: {rec['prob_positive']:.1%} prob, {rec['decision_confidence']:.2f} confidence")

if __name__ == "__main__":
    main()