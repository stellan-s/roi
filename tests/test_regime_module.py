#!/usr/bin/env python3
"""
Test Script for Regime Detection Module

Quick test to verify the regime detection module works correctly.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from quant.modules import ModuleRegistry
from quant.modules.technical import TechnicalIndicatorsModule
from quant.modules.regime import RegimeDetectionModule
from quant.config.modules import modules as base_config

def test_regime_module():
    """Test the regime detection module"""
    print("🔍 Testing Regime Detection Module")
    print("=" * 40)

    # Create registry and register modules
    registry = ModuleRegistry()
    registry.register_module(TechnicalIndicatorsModule)
    registry.register_module(RegimeDetectionModule)

    print(f"✅ Registered {len(registry)} modules")

    # Create module configuration
    module_config = {
        'regime_detection': base_config['regime_detection']
    }

    # Create and test individual module first
    regime_module = RegimeDetectionModule(module_config['regime_detection'])

    print("\n🧪 Running Module Health Check...")
    health = regime_module.health_check()
    print(f"   Status: {health['status']}")
    print(f"   Tests passed: {health['test_result']['tests_passed']}/{health['test_result']['total_tests']}")
    print(f"   Detected regime: {health['test_result']['detected_regime']}")
    print(f"   Confidence: {health['test_result']['confidence']:.2f}")

    # Create pipeline with regime module
    pipeline = registry.create_pipeline(module_config)
    print(f"\n✅ Created pipeline with {len(pipeline)} modules")

    # Generate test data using the module's built-in generator
    test_prices = regime_module._generate_test_prices()

    print(f"\n📊 Generated test data:")
    print(f"   Price records: {len(test_prices)}")
    print(f"   Tickers: {test_prices['ticker'].unique().tolist()}")
    print(f"   Date range: {test_prices['date'].min()} to {test_prices['date'].max()}")
    print(f"   Sample prices:")
    for ticker in test_prices['ticker'].unique()[:2]:
        ticker_data = test_prices[test_prices['ticker'] == ticker]
        start_price = ticker_data.iloc[0]['close']
        end_price = ticker_data.iloc[-1]['close']
        pct_change = (end_price - start_price) / start_price * 100
        print(f"     {ticker}: ${start_price:.2f} → ${end_price:.2f} ({pct_change:+.1f}%)")

    # Execute pipeline
    initial_inputs = {'prices': test_prices}
    results = pipeline.execute(initial_inputs)

    # Display results
    for module_name, result in results.items():
        print(f"\n📈 {module_name} Results:")
        print(f"   Confidence: {result.confidence:.2f}")
        print(f"   Execution time: {result.execution_time_ms:.1f}ms")

        if 'current_regime' in result.data:
            current_regime = result.data['current_regime']
            regime_probs = result.data['regime_probabilities']
            regime_adjustments = result.data['regime_adjustments']

            # Show regime classification
            regime_emoji = {"bull": "🐂", "bear": "🐻", "neutral": "⚖️"}
            emoji = regime_emoji.get(current_regime, "❓")
            print(f"   Current regime: {emoji} {current_regime.upper()}")

            # Show probabilities
            print(f"   Regime probabilities:")
            for regime, prob in regime_probs.items():
                emoji = regime_emoji.get(regime, "❓")
                bar_length = int(prob * 20)  # 20-char bar
                bar = "█" * bar_length + "░" * (20 - bar_length)
                print(f"     {emoji} {regime.capitalize()}: {prob:.1%} {bar}")

            # Show adjustments
            print(f"   Signal adjustments:")
            for signal_type, adjustment in regime_adjustments.items():
                arrow = "📈" if adjustment > 1.0 else "📉" if adjustment < 1.0 else "➡️"
                print(f"     {arrow} {signal_type.capitalize()}: {adjustment:.1f}x")

        # Show metadata
        if result.metadata:
            print(f"   Metadata:")
            for key, value in result.metadata.items():
                if key not in ['regime_method']:  # Skip technical details
                    print(f"     {key}: {value}")

    print("\n🎉 Regime detection module test completed successfully!")

if __name__ == "__main__":
    test_regime_module()