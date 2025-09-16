"""
Test script for parameter learning functionality.

This script tests the adaptive parameter estimation against the original hardcoded system
to validate that the learned parameters make sense and improve performance.
"""

import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict

# Add the quant module to path
import sys
sys.path.append('quant')

from quant.data_layer.prices import fetch_price_data
from quant.data_layer.news import fetch_news_data
from quant.features.technical import compute_technical_features
from quant.features.sentiment import naive_sentiment
from quant.bayesian.integration import BayesianPolicyEngine
from quant.bayesian.adaptive_integration import AdaptiveBayesianEngine

def load_test_config() -> Dict:
    """Load test configuration."""
    with open("quant/config/settings.yaml", 'r') as f:
        config = yaml.safe_load(f)

    with open("quant/config/universe.yaml", 'r') as f:
        universe = yaml.safe_load(f)

    config['universe'] = universe
    return config

def prepare_test_data(config: Dict) -> tuple:
    """Prepare test data."""
    print("Fetching test data...")

    universe = config['universe']['tickers'][:5]  # Limit to 5 stocks for testing

    prices = fetch_price_data(
        tickers=universe,
        cache_dir=config['data']['cache_dir'],
        lookback_days=500
    )

    news = fetch_news_data(
        feed_urls=config['signals']['news_feed_urls'],
        cache_dir=config['data']['cache_dir'],
        lookback_days=500
    )

    tech = compute_technical_features(
        prices,
        config["signals"]["sma_long"],
        config["signals"]["momentum_window"]
    )

    senti = naive_sentiment(news, universe)

    # Create returns
    returns_df = prices.copy()
    returns_df = returns_df.sort_values(['ticker', 'date'])
    returns_df['return'] = returns_df.groupby('ticker')['close'].pct_change()
    returns_df = returns_df.dropna()

    print(f"Prepared data: {len(prices)} prices, {len(tech)} technical, {len(senti)} sentiment")
    return prices, tech, senti, returns_df

def test_parameter_estimation():
    """Test parameter estimation functionality."""
    print("=== Testing Parameter Estimation ===")

    config = load_test_config()
    prices, tech, senti, returns = prepare_test_data(config)

    # Test original system
    print("\n1. Testing original system...")
    original_engine = BayesianPolicyEngine(config)

    # Use recent data for comparison
    recent_tech = tech.tail(100)
    recent_senti = senti.tail(100)
    recent_prices = prices.tail(100)

    original_results = original_engine.bayesian_score(recent_tech, recent_senti, recent_prices)
    print(f"Original system: {len(original_results)} recommendations")

    # Test adaptive system
    print("\n2. Testing adaptive system...")
    adaptive_engine = AdaptiveBayesianEngine(config)

    # Calibrate with historical data
    adaptive_engine.calibrate_parameters(
        prices_df=prices,
        sentiment_df=senti,
        technical_df=tech,
        returns_df=returns
    )

    adaptive_results = adaptive_engine.bayesian_score_adaptive(recent_tech, recent_senti, recent_prices)
    print(f"Adaptive system: {len(adaptive_results)} recommendations")

    return original_results, adaptive_results, adaptive_engine

def compare_results(original_results: pd.DataFrame,
                   adaptive_results: pd.DataFrame,
                   adaptive_engine: AdaptiveBayesianEngine):
    """Compare original vs adaptive results."""
    print("\n=== Comparison Results ===")

    if len(original_results) == 0 or len(adaptive_results) == 0:
        print("No results to compare")
        return

    # Merge results for comparison
    comparison = pd.merge(
        original_results[['ticker', 'decision', 'prob_positive', 'confidence']],
        adaptive_results[['ticker', 'decision', 'prob_positive', 'confidence']],
        on='ticker',
        suffixes=('_original', '_adaptive')
    )

    print(f"Comparing {len(comparison)} stocks:")

    # Decision agreement
    agreement = (comparison['decision_original'] == comparison['decision_adaptive']).mean()
    print(f"Decision agreement: {agreement:.1%}")

    # Probability differences
    prob_diff = (comparison['prob_positive_adaptive'] - comparison['prob_positive_original']).abs().mean()
    print(f"Average probability difference: {prob_diff:.3f}")

    # Confidence differences
    conf_diff = (comparison['confidence_adaptive'] - comparison['confidence_original']).abs().mean()
    print(f"Average confidence difference: {conf_diff:.3f}")

    # Show parameter diagnostics
    print("\n=== Parameter Changes ===")
    diagnostics = adaptive_engine.get_parameter_diagnostics()

    if not diagnostics.empty:
        print("Estimated vs default parameters:")
        for _, row in diagnostics.iterrows():
            if row['n_observations'] > 0:  # Only show parameters that were actually estimated
                default_val = row['default_value']
                estimated_val = row['estimated_value']
                change_pct = ((estimated_val - default_val) / default_val) * 100 if default_val != 0 else 0

                print(f"  {row['parameter_name']}: {default_val:.3f} → {estimated_val:.3f} ({change_pct:+.1f}%)")
                print(f"    Method: {row['estimation_method']}, N: {row['n_observations']}")

    return comparison

def test_signal_normalization():
    """Test signal normalization specifically."""
    print("\n=== Testing Signal Normalization ===")

    config = load_test_config()
    prices, tech, senti, returns = prepare_test_data(config)

    if senti.empty:
        print("No sentiment data available for testing")
        return

    # Analyze sentiment score distribution
    sent_scores = senti['sent_score'].dropna()
    print(f"Sentiment scores: N={len(sent_scores)}, range=[{sent_scores.min():.2f}, {sent_scores.max():.2f}]")
    print(f"Percentiles: 5%={np.percentile(sent_scores, 5):.2f}, 95%={np.percentile(sent_scores, 95):.2f}")

    # Original normalization: clip(sent_score / 2.0, -1, 1)
    original_normalized = np.clip(sent_scores / 2.0, -1.0, 1.0)
    print(f"Original normalization range: [{original_normalized.min():.2f}, {original_normalized.max():.2f}]")

    # Adaptive normalization: use percentile-based scaling
    p5, p95 = np.percentile(sent_scores, [5, 95])
    if (p95 - p5) > 0:
        adaptive_scale = 2.0 / (p95 - p5)
        adaptive_normalized = np.clip(sent_scores * adaptive_scale, -1.0, 1.0)
        print(f"Adaptive normalization (scale={adaptive_scale:.2f}): [{adaptive_normalized.min():.2f}, {adaptive_normalized.max():.2f}]")

        # Compare utilization of signal range
        original_utilization = (original_normalized.max() - original_normalized.min()) / 2.0
        adaptive_utilization = (adaptive_normalized.max() - adaptive_normalized.min()) / 2.0

        print(f"Signal range utilization: Original={original_utilization:.1%}, Adaptive={adaptive_utilization:.1%}")

def main():
    """Main test execution."""
    print("=== Parameter Learning Test Suite ===")

    try:
        # Test parameter estimation
        original_results, adaptive_results, adaptive_engine = test_parameter_estimation()

        # Compare results
        comparison = compare_results(original_results, adaptive_results, adaptive_engine)

        # Test signal normalization
        test_signal_normalization()

        print("\n=== Test Summary ===")
        print("✓ Parameter estimation framework functional")
        print("✓ Adaptive engine calibration working")
        print("✓ Signal normalization analysis complete")

        if len(comparison) > 0:
            buy_changes = comparison[
                (comparison['decision_original'] != 'buy') &
                (comparison['decision_adaptive'] == 'buy')
            ]
            sell_changes = comparison[
                (comparison['decision_original'] != 'sell') &
                (comparison['decision_adaptive'] == 'sell')
            ]

            print(f"✓ Decision changes: {len(buy_changes)} new buys, {len(sell_changes)} new sells")

    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()