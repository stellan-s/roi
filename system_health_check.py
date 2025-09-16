#!/usr/bin/env python3
"""
ROI System Health Check - Snabb diagnostik av systemets grundfunktioner

Kör denna för att verifitera att alla komponenter fungerar korrekt
innan du kör fullständiga backtests eller analyser.
"""

import sys
import time
from pathlib import Path
import yaml
import pandas as pd
import numpy as np

def print_status(test_name: str, status: str, details: str = ""):
    """Formaterad status-utskrift"""
    status_symbol = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
    print(f"{status_symbol} {test_name}: {status}")
    if details:
        print(f"   {details}")

def test_config_loading():
    """Test 1: Konfigurationsfiler laddas korrekt"""
    try:
        with open('quant/config/settings.yaml', 'r') as f:
            settings = yaml.safe_load(f)

        with open('quant/config/universe.yaml', 'r') as f:
            universe = yaml.safe_load(f)

        # Kontrollera nyckelkonfigurationer
        assert 'bayesian' in settings
        assert 'tickers' in universe
        assert len(universe['tickers']) > 0

        return "PASS", f"Loaded {len(universe['tickers'])} tickers"
    except Exception as e:
        return "FAIL", str(e)

def test_data_layer():
    """Test 2: Data layer kan hämta data"""
    try:
        from quant.data_layer.prices import fetch_prices

        # Testa med en enkel ticker och kort lookback
        test_tickers = ['AAPL', 'TSLA']
        prices = fetch_prices(test_tickers, 'data', 30)

        assert not prices.empty
        assert 'close' in prices.columns
        assert 'date' in prices.columns

        return "PASS", f"Fetched {len(prices)} price observations"
    except Exception as e:
        return "FAIL", str(e)

def test_regime_detection():
    """Test 3: Regime detection fungerar utan crash"""
    try:
        from quant.regime.detector import RegimeDetector
        from quant.data_layer.prices import fetch_prices

        # Minimal data
        prices = fetch_prices(['AAPL'], 'data', 50)

        with open('quant/config/settings.yaml', 'r') as f:
            cfg = yaml.safe_load(f)

        detector = RegimeDetector(cfg)
        regime, probabilities, diagnostics = detector.detect_regime(prices)

        # Kontrollera att vi får en regime tillbaka
        assert regime is not None
        assert isinstance(probabilities, dict)

        return "PASS", f"Detected regime: {regime.value}"
    except Exception as e:
        return "FAIL", str(e)

def test_bayesian_engine():
    """Test 4: Bayesian engine kan generera recommendations"""
    try:
        from quant.data_layer.prices import fetch_prices
        from quant.features.technical import compute_technical_features
        from quant.features.sentiment import naive_sentiment
        from quant.policy_engine.rules import bayesian_score

        with open('quant/config/settings.yaml', 'r') as f:
            cfg = yaml.safe_load(f)

        # Minimal test med 2 tickers
        test_tickers = ['AAPL', 'TSLA']
        prices = fetch_prices(test_tickers, 'data', 100)

        # Simplified features
        tech = compute_technical_features(prices, 20, 50)  # Kortare perioder

        # Empty sentiment for speed
        sentiment = pd.DataFrame(columns=['date', 'ticker', 'sent_score'])

        # Test Bayesian scoring
        decisions = bayesian_score(tech, sentiment, prices, cfg)

        assert not decisions.empty
        assert 'decision' in decisions.columns

        return "PASS", f"Generated {len(decisions)} recommendations"
    except Exception as e:
        return "FAIL", str(e)

def test_portfolio_rules():
    """Test 5: Portfolio rules appliceras utan infinite loop"""
    try:
        from quant.portfolio.rules import PortfolioManager

        with open('quant/config/settings.yaml', 'r') as f:
            cfg = yaml.safe_load(f)

        # Skapa dummy recommendations
        test_data = pd.DataFrame({
            'date': ['2024-01-01'] * 3,
            'ticker': ['AAPL', 'TSLA', 'MSFT'],
            'decision': ['Buy', 'Sell', 'Hold'],
            'expected_return': [0.02, -0.01, 0.005],
            'prob_positive': [0.7, 0.3, 0.6],
            'decision_confidence': [0.8, 0.7, 0.5],
            'market_regime': ['unknown', 'unknown', 'unknown']  # Test unknown regime
        })

        portfolio_mgr = PortfolioManager(cfg)
        result = portfolio_mgr.apply_portfolio_rules(test_data)

        assert not result.empty
        assert len(result) == len(test_data)

        return "PASS", "Portfolio rules applied successfully"
    except Exception as e:
        return "FAIL", str(e)

def test_system_integration():
    """Test 6: Enkel end-to-end test"""
    try:
        # Test att main.py kan importeras utan crash
        import quant.main

        # Test att adaptive_main.py kan importeras
        import quant.adaptive_main

        return "PASS", "Main modules import successfully"
    except Exception as e:
        return "FAIL", str(e)

def main():
    """Kör alla systemhälso-tester"""
    print("🔍 ROI System Health Check")
    print("=" * 50)

    start_time = time.time()

    # Lista av tester
    tests = [
        ("Configuration Loading", test_config_loading),
        ("Data Layer", test_data_layer),
        ("Regime Detection", test_regime_detection),
        ("Bayesian Engine", test_bayesian_engine),
        ("Portfolio Rules", test_portfolio_rules),
        ("System Integration", test_system_integration),
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\nRunning: {test_name}...")
        try:
            status, details = test_func()
            print_status(test_name, status, details)
            results.append((test_name, status, details))
        except Exception as e:
            print_status(test_name, "FAIL", f"Unexpected error: {e}")
            results.append((test_name, "FAIL", str(e)))

    # Sammanfattning
    elapsed = time.time() - start_time
    passed = sum(1 for _, status, _ in results if status == "PASS")
    total = len(results)

    print("\n" + "=" * 50)
    print(f"📊 RESULTS: {passed}/{total} tests passed ({elapsed:.1f}s)")

    if passed == total:
        print("🎉 All systems operational!")
        return 0
    else:
        print("🚨 Some tests failed. Check above for details.")

        # List failed tests
        print("\nFailed tests:")
        for test_name, status, details in results:
            if status == "FAIL":
                print(f"  - {test_name}: {details}")

        return 1

if __name__ == "__main__":
    sys.exit(main())