#!/usr/bin/env python3
"""
Test Script for Sentiment Analysis Module

Quick test to verify the sentiment module works correctly.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from quant.modules import ModuleRegistry
from quant.modules.technical import TechnicalIndicatorsModule
from quant.modules.sentiment import SentimentAnalysisModule
from quant.config.modules import modules as base_config

def test_sentiment_module():
    """Test the sentiment analysis module"""
    print("🔍 Testing Sentiment Analysis Module")
    print("=" * 40)

    # Create registry and register modules
    registry = ModuleRegistry()
    registry.register_module(TechnicalIndicatorsModule)
    registry.register_module(SentimentAnalysisModule)

    print(f"✅ Registered {len(registry)} modules")

    # Create module configuration
    module_config = {
        'sentiment_analysis': base_config['sentiment_analysis']
    }

    # Create and test individual module first
    sentiment_module = SentimentAnalysisModule(module_config['sentiment_analysis'])

    print("\n🧪 Running Module Health Check...")
    health = sentiment_module.health_check()
    print(f"   Status: {health['status']}")
    print(f"   Tests passed: {health['test_result']['tests_passed']}/{health['test_result']['total_tests']}")

    # Create pipeline with sentiment module
    pipeline = registry.create_pipeline(module_config)
    print(f"\n✅ Created pipeline with {len(pipeline)} modules")

    # Generate test data using the module's built-in generator
    test_data = sentiment_module._generate_test_data()
    news_df, tickers = test_data

    print(f"\n📊 Generated test data:")
    print(f"   News articles: {len(news_df)}")
    print(f"   Tickers: {tickers}")
    print(f"   Sample headlines:")
    for i, title in enumerate(news_df['title'].head(3)):
        print(f"     {i+1}. {title}")

    # Execute pipeline
    initial_inputs = {'news': news_df, 'tickers': tickers}
    results = pipeline.execute(initial_inputs)

    # Display results
    for module_name, result in results.items():
        print(f"\n📈 {module_name} Results:")
        print(f"   Confidence: {result.confidence:.2f}")
        print(f"   Execution time: {result.execution_time_ms:.1f}ms")

        if 'sentiment_signals' in result.data:
            signals = result.data['sentiment_signals']
            print(f"   Generated signals for {len(signals)} tickers")

            # Show signals
            for ticker, signal in signals.items():
                emoji = "📈" if signal > 0 else "📉" if signal < 0 else "➡️"
                print(f"     {emoji} {ticker}: {signal:.3f}")

        if 'sentiment_summary' in result.data:
            summary = result.data['sentiment_summary']
            print(f"   Articles processed: {summary.get('articles_with_sentiment', 0)}")
            print(f"   Average sentiment: {summary.get('avg_sentiment', 0):.3f}")

            dist = summary.get('sentiment_distribution', {})
            print(f"   Distribution: {dist.get('positive', 0)} pos, {dist.get('negative', 0)} neg, {dist.get('neutral', 0)} neutral")

    print("\n🎉 Sentiment module test completed successfully!")

if __name__ == "__main__":
    test_sentiment_module()