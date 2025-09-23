#!/usr/bin/env python3
"""
Test script for fundamentals integration
Run: python test_fundamentals.py
"""

import sys
import os
sys.path.append('.')

from quant.modules.fundamentals_integration import FundamentalsIntegration

def test_fundamentals():
    """Test the fundamentals integration"""
    print("🔍 Testing Fundamentals Integration")
    print("=" * 50)

    # Initialize integration
    fundamentals = FundamentalsIntegration()

    # Check status
    status = fundamentals.get_status()
    print(f"📊 Status:")
    print(f"   Enabled: {status['enabled']}")
    print(f"   API Configured: {status['api_configured']}")
    print(f"   Data Available: {status['data_available']}")
    print()

    if not status['enabled']:
        print("⚠️  Fundamentals disabled in config")
        print("   To enable: set data.fundamentals_api.enabled = true in settings.yaml")
        return

    # Test with some Swedish stocks (including ALFA which has data)
    test_symbols = ['VOLV-B.ST', 'SEB-A.ST', 'ABB.ST', 'ALFA.ST']

    print(f"🧪 Testing with symbols: {test_symbols}")
    print()

    for symbol in test_symbols:
        print(f"📈 {symbol}:")

        # Get fundamental features
        features = fundamentals.get_fundamental_features(symbol)

        if features:
            print(f"   ✅ Found {len(features)} fundamental features")
            # Show a few key features
            for key in ['fundamental_score', 'roe', 'net_margin', 'debt_to_equity']:
                if key in features:
                    print(f"   {key}: {features[key]:.4f}")
        else:
            print("   ❌ No fundamental data available")
            print("   (This is expected until API securities are linked to companies)")

        # Test signal
        signal = fundamentals.get_fundamental_signal(symbol)
        print(f"   📊 Fundamental signal: {signal:.4f}")

        # Test attractiveness
        attractive = fundamentals.is_fundamentally_attractive(symbol)
        print(f"   🎯 Fundamentally attractive: {attractive}")
        print()

    print("✅ Test completed!")

if __name__ == "__main__":
    test_fundamentals()