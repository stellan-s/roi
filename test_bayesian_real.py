#!/usr/bin/env python3
"""
Test Bayesian engine med riktig ROI data
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from quant.main import load_yaml
from quant.data_layer.prices import fetch_prices
from quant.data_layer.news import fetch_news
from quant.features.technical import compute_technical_features
from quant.features.sentiment import naive_sentiment
from quant.policy_engine.rules import bayesian_score

def test_bayesian_with_real_data():
    """Test Bayesian engine med samma data som main.py använder"""
    print("=== TESTING BAYESIAN ENGINE MED REAL DATA ===")

    # Ladda samma config som main.py
    uni = load_yaml("universe.yaml")["tickers"]
    cfg = load_yaml("settings.yaml")
    cache = cfg["data"]["cache_dir"]

    print(f"Tickers: {uni}")
    print(f"Cache dir: {cache}")

    # Hämta samma data som main.py
    prices = fetch_prices(uni, cache, cfg["data"]["lookback_days"])
    news = fetch_news(cfg["signals"]["news_feed_urls"], cache)
    tech = compute_technical_features(prices, cfg["signals"]["sma_long"], cfg["signals"]["momentum_window"])
    senti = naive_sentiment(news, uni)

    print(f"\nData shapes:")
    print(f"Prices: {prices.shape}")
    print(f"Tech features: {tech.shape}")
    print(f"Sentiment: {senti.shape}")

    # Test Bayesian scoring
    bayesian_results = bayesian_score(tech, senti)

    print(f"\nBayesian results:")
    print(f"Shape: {bayesian_results.shape}")
    print(f"Columns: {list(bayesian_results.columns)}")

    # Senaste dagarna
    latest_date = bayesian_results['date'].max()
    latest_results = bayesian_results[bayesian_results['date'] == latest_date]

    print(f"\nSenaste dag ({latest_date}):")
    for _, row in latest_results.iterrows():
        print(f"{row['ticker']:>12} | E[r]: {row['expected_return']:>8.4f} | Pr(↑): {row['prob_positive']:>6.3f} | Decision: {row['decision']:>4} | Confidence: {row['decision_confidence']:>6.3f}")

    # Jämför med simple scoring
    from quant.policy_engine.rules import simple_score
    simple_results = simple_score(tech, senti)
    simple_latest = simple_results[simple_results['date'] == latest_date]

    print(f"\nJämförelse (Simple vs Bayesian) för {latest_date}:")
    merged = simple_latest.merge(latest_results, on=['ticker', 'date'], suffixes=('_simple', '_bayesian'))

    for _, row in merged.iterrows():
        print(f"{row['ticker']:>12} | Simple: {row['decision_simple']:>4} (score: {row['score']:>4.1f}) | Bayesian: {row['decision_bayesian']:>4} (E[r]: {row['expected_return']:>6.4f}, Pr(↑): {row['prob_positive']:>5.3f})")

    return bayesian_results

if __name__ == "__main__":
    try:
        results = test_bayesian_with_real_data()
        print(f"\n✅ SUCCESS: Bayesian engine fungerar med real data!")
        print(f"Generated {len(results)} predictions för {results['ticker'].nunique()} tickers")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)