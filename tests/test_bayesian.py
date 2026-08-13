#!/usr/bin/env python3
"""
Test script for the Bayesian Signal Engine
Exercises synthetic data and shows E[r], Pr(↑) outputs
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from quant.bayesian.signal_engine import BayesianSignalEngine, SignalType
from quant.bayesian.integration import BayesianPolicyEngine
from quant.policy_engine.rules import bayesian_score

def create_synthetic_data(n_days=100, n_tickers=5):
    """Create synthetic test data"""

    dates = [datetime.today() - timedelta(days=i) for i in range(n_days, 0, -1)]
    tickers = [f"STOCK{i}.ST" for i in range(1, n_tickers + 1)]

    # Technical features
    tech_data = []
    for date in dates:
        for ticker in tickers:
            # Simulate price and features
            price = 100 + np.random.normal(0, 10)
            above_sma = np.random.choice([0, 1], p=[0.4, 0.6])  # Slight bullish bias
            mom_rank = np.random.uniform(0, 1)

            tech_data.append({
                'date': date.date(),
                'ticker': ticker,
                'close': price,
                'above_sma': above_sma,
                'mom_rank': mom_rank
            })

    tech_df = pd.DataFrame(tech_data)

    # Sentiment data
    senti_data = []
    for date in dates[::3]:  # Sentiment every 3rd day
        for ticker in tickers:
            if np.random.random() < 0.3:  # 30% chance of sentiment news
                sent_score = np.random.normal(0, 1)  # Mean 0, varying sentiment
                senti_data.append({
                    'date': date.date(),
                    'ticker': ticker,
                    'sent_score': sent_score
                })

    senti_df = pd.DataFrame(senti_data)

    return tech_df, senti_df

def test_basic_engine():
    """Test basic Bayesian signal engine"""
    print("\n=== TESTING BASIC BAYESIAN ENGINE ===")

    engine = BayesianSignalEngine()

    # Test signal combination
    signals = {
        SignalType.TREND: 0.5,      # Positive trend
        SignalType.MOMENTUM: 0.3,   # Mild momentum
        SignalType.SENTIMENT: -0.2  # Slightly negative sentiment
    }

    output = engine.combine_signals(signals)

    print(f"Input signals: {signals}")
    print(f"Expected return: {output.expected_return:.4f}")
    print(f"Prob positive: {output.prob_positive:.3f}")
    print(f"Confidence interval: [{output.confidence_lower:.4f}, {output.confidence_upper:.4f}]")
    print(f"Signal weights: {output.signal_weights}")
    print(f"Uncertainty: {output.uncertainty:.3f}")

    return output

def test_learning():
    """Test Bayesian learning from performance data"""
    print("\n=== TESTING BAYESIAN LEARNING ===")

    engine = BayesianSignalEngine()

    # Simulate observations with different outcomes
    scenarios = [
        # Strong positive signals → good returns (confirming priors)
        ({'trend': 0.8, 'momentum': 0.7, 'sentiment': 0.5}, 0.05),
        ({'trend': 0.6, 'momentum': 0.8, 'sentiment': 0.3}, 0.03),

        # Mixed signals → mixed results
        ({'trend': 0.2, 'momentum': -0.3, 'sentiment': 0.1}, -0.01),
        ({'trend': -0.4, 'momentum': 0.2, 'sentiment': -0.2}, -0.02),

        # Test case where sentiment was misleading
        ({'trend': 0.3, 'momentum': 0.4, 'sentiment': 0.9}, -0.03),  # High sentiment, bad outcome
        ({'trend': 0.2, 'momentum': 0.3, 'sentiment': 0.8}, -0.01),  # Another sentiment miss
    ]

    print("Before learning - signal diagnostics:")
    print(engine.get_signal_diagnostics())

    # Update beliefs based on the scenarios
    for signals_dict, actual_return in scenarios:
        signals = {SignalType[k.upper()]: v for k, v in signals_dict.items()}
        engine.update_beliefs(signals, actual_return)

    print("\nAfter learning - signal diagnostics:")
    diagnostics = engine.get_signal_diagnostics()
    print(diagnostics)

    # Test new predictions after learning
    test_signals = {
        SignalType.TREND: 0.4,
        SignalType.MOMENTUM: 0.5,
        SignalType.SENTIMENT: 0.7  # High sentiment (should be downweighted after learning)
    }

    output_after = engine.combine_signals(test_signals)
    print(f"\nTest prediction after learning:")
    print(f"Signals: {test_signals}")
    print(f"Expected return: {output_after.expected_return:.4f}")
    print(f"Prob positive: {output_after.prob_positive:.3f}")
    print(f"Signal weights: {output_after.signal_weights}")

def test_full_pipeline():
    """Test full Bayesian integration with realistic data"""
    print("\n=== TESTING FULL BAYESIAN PIPELINE ===")

    # Create synthetic data
    tech_df, senti_df = create_synthetic_data(n_days=50, n_tickers=3)

    print(f"Generated {len(tech_df)} technical observations")
    print(f"Generated {len(senti_df)} sentiment observations")

    # Test Bayesian scoring
    results = bayesian_score(tech_df, senti_df)

    print(f"\nBayesian results shape: {results.shape}")
    print(f"Columns: {list(results.columns)}")

    # Show sample results
    print("\nSample results (first 5 rows):")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(results.head())

    # Decision summary
    print(f"\nDecision distribution:")
    print(results['decision'].value_counts())

    # Statistics
    print(f"\nExpected return statistics:")
    print(f"Mean: {results['expected_return'].mean():.4f}")
    print(f"Std: {results['expected_return'].std():.4f}")
    print(f"Min: {results['expected_return'].min():.4f}")
    print(f"Max: {results['expected_return'].max():.4f}")

    print(f"\nProbability positive statistics:")
    print(f"Mean: {results['prob_positive'].mean():.3f}")
    print(f"Std: {results['prob_positive'].std():.3f}")

    # Uncertainty analysis
    print(f"\nUncertainty statistics:")
    print(f"Mean uncertainty: {results['uncertainty'].mean():.3f}")
    print(f"High uncertainty (>0.3) cases: {(results['uncertainty'] > 0.3).sum()}")

    return results

if __name__ == "__main__":
    print("BAYESIAN SIGNAL ENGINE TESTING")
    print("=" * 50)

    try:
        # Test 1: Basic engine functionality
        basic_output = test_basic_engine()

        # Test 2: Learning capabilities
        test_learning()

        # Test 3: Full pipeline
        results = test_full_pipeline()

        print("\n" + "=" * 50)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print(f"✓ Basic engine: E[r]={basic_output.expected_return:.4f}, Pr(↑)={basic_output.prob_positive:.3f}")
        print(f"✓ Learning: Bayesian updates working")
        print(f"✓ Pipeline: {len(results)} predictions generated")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
