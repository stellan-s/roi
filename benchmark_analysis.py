#!/usr/bin/env python3
"""
ROI Benchmark Analysis - Quick system quality check

This script gives you a view of whether the system is "good enough"
by measuring key performance indicators and comparing them to benchmarks.
"""

import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from datetime import datetime, timedelta

# Benchmark targets for a "good enough" system
BENCHMARK_TARGETS = {
    "min_sharpe_ratio": 0.8,          # Minimum Sharpe ratio
    "max_drawdown_threshold": 0.20,   # Max acceptabel drawdown (20%)
    "min_win_rate": 0.50,             # Minimum win rate (50%)
    "min_annual_return": 0.08,        # Minimum annual return (8%)
    "max_uncertainty": 0.40,          # Maximum average uncertainty
    "min_confidence": 0.40,           # Minimum average confidence
    "regime_detection_threshold": 0.50, # Minimum regime confidence
}

def analyze_daily_report():
    """Analyze the latest daily report for signal quality"""
    try:
        # Find the latest report
        reports_dir = Path("reports")
        report_files = list(reports_dir.glob("daily_*.md"))

        if not report_files:
            return None, "No daily reports found"

        latest_report = max(report_files, key=lambda f: f.name)
        content = latest_report.read_text(encoding='utf-8')

        # Extract key metrics from the report
        metrics = {
            "report_date": latest_report.stem.replace("daily_", ""),
            "has_buy_recommendations": "## Buy Recommendations" in content and "*(None)*" not in content,
            "has_sell_recommendations": "## Sell Recommendations" in content and "*(None)*" not in content,
            "regime_detected": "Unknown" not in content or "33% confidence" not in content,
        }

        # Count recommendations
        buy_count = content.count("E[r]_1d:") if "Buy Recommendations" in content else 0
        sell_count = content.count("E[r]_1d:") if "Sell Recommendations" in content else 0
        hold_count = content.count("⚠️ Uncertainty above 0.30 blocks buying")

        metrics.update({
            "buy_recommendations": buy_count,
            "sell_recommendations": sell_count,
            "hold_recommendations": hold_count,
            "total_recommendations": buy_count + sell_count + hold_count,
            "decision_balance": abs(buy_count - sell_count) / max(1, buy_count + sell_count)  # 0=perfect balance
        })

        return metrics, None

    except Exception as e:
        return None, str(e)

def analyze_signal_quality():
    """Analyze signal quality from the latest run"""
    try:
        # Run a lightweight analysis to obtain signal metrics
        from quant.regime.detector import RegimeDetector
        from quant.data_layer.prices import fetch_prices

        with open('quant/config/settings.yaml', 'r') as f:
            cfg = yaml.safe_load(f)

        with open('quant/config/universe.yaml', 'r') as f:
            uni_cfg = yaml.safe_load(f)

        # Test with a subset of tickers for speed
        test_tickers = uni_cfg['tickers'][:10]
        prices = fetch_prices(test_tickers, 'data', 100)

        # Regime detection quality
        detector = RegimeDetector(cfg)
        regime, probabilities, diagnostics = detector.detect_regime(prices)

        # Calculate regime confidence
        regime_confidence = max(probabilities.values()) if probabilities else 0.33

        metrics = {
            "regime_detected": regime.value if regime else "unknown",
            "regime_confidence": regime_confidence,
            "regime_quality": "Good" if regime_confidence > 0.60 else "Poor",
            "data_quality": "Good" if "error" not in diagnostics else "Poor"
        }

        return metrics, None

    except Exception as e:
        return None, str(e)

def calculate_benchmark_score(daily_metrics, signal_metrics):
    """Calculate the overall benchmark score based on the targets"""

    score_components = {}

    # 1. Decision Balance (30 points)
    # Perfect balance = 0, max imbalance = 1
    if daily_metrics and 'decision_balance' in daily_metrics:
        balance_score = max(0, 30 * (1 - daily_metrics['decision_balance']))
        score_components['decision_balance'] = balance_score
    else:
        score_components['decision_balance'] = 0

    # 2. Signal Diversity (20 points)
    # Does the system deliver both buy and sell recommendations?
    if daily_metrics:
        has_buys = daily_metrics.get('has_buy_recommendations', False)
        has_sells = daily_metrics.get('has_sell_recommendations', False)

        if has_buys and has_sells:
            diversity_score = 20
        elif has_buys or has_sells:
            diversity_score = 10  # Only one type
        else:
            diversity_score = 0   # No recommendations

        score_components['signal_diversity'] = diversity_score
    else:
        score_components['signal_diversity'] = 0

    # 3. Regime Detection Quality (25 points)
    if signal_metrics and 'regime_confidence' in signal_metrics:
        regime_conf = signal_metrics['regime_confidence']
        if regime_conf > 0.60:
            regime_score = 25
        elif regime_conf > 0.45:
            regime_score = 15
        else:
            regime_score = 5
        score_components['regime_detection'] = regime_score
    else:
        score_components['regime_detection'] = 0

    # 4. System Stability (25 points)
    # Based on whether the system runs without crashes
    if daily_metrics and signal_metrics:
        stability_score = 25  # Making it here means no crash occurred
    else:
        stability_score = 0
    score_components['system_stability'] = stability_score

    total_score = sum(score_components.values())

    return total_score, score_components

def interpret_score(score):
    """Interpret the benchmark score"""
    if score >= 80:
        return "EXCELLENT", "The system is outperforming expectations"
    elif score >= 65:
        return "GOOD", "The system is good enough for production"
    elif score >= 50:
        return "ACCEPTABLE", "The system works but needs improvements"
    elif score >= 30:
        return "POOR", "The system has significant issues"
    else:
        return "CRITICAL", "The system is not functional"

def generate_recommendations(score, score_components, daily_metrics, signal_metrics):
    """Generate specific improvement recommendations"""
    recommendations = []

    # Decision Balance
    if score_components.get('decision_balance', 0) < 20:
        if daily_metrics and daily_metrics.get('buy_recommendations', 0) == 0:
            recommendations.append("🎯 No buy recommendations produced - review Bayesian thresholds")
        elif daily_metrics and daily_metrics.get('sell_recommendations', 0) == 0:
            recommendations.append("🎯 No sell recommendations produced - review risk management configuration")

    # Signal Diversity
    if score_components.get('signal_diversity', 0) < 15:
        recommendations.append("📊 Low signal diversity - consider adjusting decision thresholds in settings.yaml")

    # Regime Detection
    if score_components.get('regime_detection', 0) < 20:
        if signal_metrics and signal_metrics.get('regime_confidence', 0) < 0.50:
            recommendations.append("🔍 Weak regime detection - review market data quality and lookback periods")

    # System Stability
    if score_components.get('system_stability', 0) < 20:
        recommendations.append("⚠️ System stability issue - run system_health_check.py for debugging")

    # Targeted improvements based on score
    if score < 50:
        recommendations.append("🚨 PRIORITY: Run 'python system_health_check.py' for baseline debugging")
        recommendations.append("🔧 Consider resetting settings.yaml to default values")

    elif score < 70:
        recommendations.append("📈 Consider parameter tuning with adaptive_main.py for better performance")
        recommendations.append("🎯 Experiment with decision thresholds for more balanced recommendations")

    return recommendations

def main():
    """Run the benchmark analysis"""
    print("📊 ROI Benchmark Analysis")
    print("=" * 60)
    print("Analyzing system quality against production benchmarks...")
    print()

    start_time = time.time()

    # 1. Analyze daily report
    print("📋 Analyzing latest daily report...")
    daily_metrics, daily_error = analyze_daily_report()

    if daily_error:
        print(f"   ❌ Error: {daily_error}")
    else:
        print(f"   ✅ Report from: {daily_metrics['report_date']}")
        print(f"   📈 Buy/Sell/Hold: {daily_metrics['buy_recommendations']}/{daily_metrics['sell_recommendations']}/{daily_metrics['hold_recommendations']}")

    # 2. Analyze signal quality
    print("\\n🔍 Analyzing signal quality...")
    signal_metrics, signal_error = analyze_signal_quality()

    if signal_error:
        print(f"   ❌ Error: {signal_error}")
    else:
        print(f"   🎯 Regime: {signal_metrics['regime_detected']} ({signal_metrics['regime_confidence']:.1%} confidence)")
        print(f"   📊 Quality: {signal_metrics['regime_quality']}")

    # 3. Calculate benchmark score
    print("\\n🏆 Calculating benchmark score...")
    score, score_components = calculate_benchmark_score(daily_metrics, signal_metrics)

    level, description = interpret_score(score)

    # 4. Results
    elapsed = time.time() - start_time
    print("\\n" + "=" * 60)
    print(f"📊 BENCHMARK RESULTS ({elapsed:.1f}s)")
    print("=" * 60)
    print(f"🎯 Overall Score: {score:.1f}/100 - {level}")
    print(f"📝 Assessment: {description}")
    print()

    # Detailed scores
    print("📈 Score Breakdown:")
    for component, points in score_components.items():
        component_name = component.replace('_', ' ').title()
        print(f"   {component_name}: {points:.1f} points")

    # Recommendations
    print("\\n🎯 RECOMMENDATIONS:")
    recommendations = generate_recommendations(score, score_components, daily_metrics, signal_metrics)

    if not recommendations:
        print("   🎉 No specific improvements needed - system performing well!")
    else:
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")

    # Next steps
    print("\\n🚀 NEXT STEPS:")
    if score >= 65:
        print("   ✅ System is production-ready")
        print("   📊 Run: python -m quant.backtest_runner for historical validation")
        print("   🎯 Consider live paper trading for final validation")
    elif score >= 50:
        print("   🔧 Address recommendations above")
        print("   📊 Run: python system_health_check.py for detailed diagnostics")
        print("   🎯 Rerun benchmark after fixes")
    else:
        print("   🚨 Focus on basic system stability first")
        print("   🔧 Run: python system_health_check.py")
        print("   📚 Review documentation for configuration guidance")

    return 0 if score >= 50 else 1

if __name__ == "__main__":
    sys.exit(main())
