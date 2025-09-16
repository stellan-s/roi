#!/usr/bin/env python3
"""
ROI Benchmark Analysis - Snabb utvärdering av systemkvalitet

Detta script ger dig en översikt av om systemet är "tillräckligt bra"
genom att mäta key performance indicators och jämföra mot benchmarks.
"""

import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from datetime import datetime, timedelta

# Benchmark-målsättningar för "tillräckligt bra" system
BENCHMARK_TARGETS = {
    "min_sharpe_ratio": 0.8,          # Minimum Sharpe ratio
    "max_drawdown_threshold": 0.20,   # Max acceptabel drawdown (20%)
    "min_win_rate": 0.50,             # Minimum win rate (50%)
    "min_annual_return": 0.08,        # Minimum årlig avkastning (8%)
    "max_uncertainty": 0.40,          # Max genomsnittlig osäkerhet
    "min_confidence": 0.40,           # Min genomsnittlig confidence
    "regime_detection_threshold": 0.50, # Min regime confidence
}

def analyze_daily_report():
    """Analysera senaste daily report för signal quality"""
    try:
        # Hitta senaste rapport
        reports_dir = Path("reports")
        report_files = list(reports_dir.glob("daily_*.md"))

        if not report_files:
            return None, "Inga daily reports hittade"

        latest_report = max(report_files, key=lambda f: f.name)
        content = latest_report.read_text(encoding='utf-8')

        # Extrahera metrics från rapporten
        metrics = {
            "report_date": latest_report.stem.replace("daily_", ""),
            "has_buy_recommendations": "## Köp-förslag" in content and "*(Inget)*" not in content,
            "has_sell_recommendations": "## Sälj-förslag" in content and "*(Inget)*" not in content,
            "regime_detected": "Unknown" not in content or "33% säkerhet" not in content,
        }

        # Räkna rekommendationer
        buy_count = content.count("E[r]_1d:") if "Köp-förslag" in content else 0
        sell_count = content.count("E[r]_1d:") if "Sälj-förslag" in content else 0
        hold_count = content.count("⚠️ Osäkerhet över 0.30 hindrar köp")

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
    """Analysera signal-kvalitet från senaste körning"""
    try:
        # Kör en enkel analys för att få signal-metrics
        from quant.regime.detector import RegimeDetector
        from quant.data_layer.prices import fetch_prices

        with open('quant/config/settings.yaml', 'r') as f:
            cfg = yaml.safe_load(f)

        with open('quant/config/universe.yaml', 'r') as f:
            uni_cfg = yaml.safe_load(f)

        # Test med subset av tickers för hastighet
        test_tickers = uni_cfg['tickers'][:10]
        prices = fetch_prices(test_tickers, 'data', 100)

        # Regime detection quality
        detector = RegimeDetector(cfg)
        regime, probabilities, diagnostics = detector.detect_regime(prices)

        # Beräkna regime confidence
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
    """Beräkna overall benchmark score baserat på målsättningar"""

    score_components = {}

    # 1. Decision Balance (30 poäng)
    # Perfekt balans = 0, helt obalanserat = 1
    if daily_metrics and 'decision_balance' in daily_metrics:
        balance_score = max(0, 30 * (1 - daily_metrics['decision_balance']))
        score_components['decision_balance'] = balance_score
    else:
        score_components['decision_balance'] = 0

    # 2. Signal Diversity (20 poäng)
    # Har systemet både köp/sälj rekommendationer?
    if daily_metrics:
        has_buys = daily_metrics.get('has_buy_recommendations', False)
        has_sells = daily_metrics.get('has_sell_recommendations', False)

        if has_buys and has_sells:
            diversity_score = 20
        elif has_buys or has_sells:
            diversity_score = 10  # Endast en typ
        else:
            diversity_score = 0   # Inga rekommendationer

        score_components['signal_diversity'] = diversity_score
    else:
        score_components['signal_diversity'] = 0

    # 3. Regime Detection Quality (25 poäng)
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

    # 4. System Stability (25 poäng)
    # Baserat på om systemet kan köra utan crashes
    if daily_metrics and signal_metrics:
        stability_score = 25  # Om vi kommit hit utan crash
    else:
        stability_score = 0
    score_components['system_stability'] = stability_score

    total_score = sum(score_components.values())

    return total_score, score_components

def interpret_score(score):
    """Tolka benchmark score"""
    if score >= 80:
        return "EXCELLENT", "Systemet presterar över förväntningar"
    elif score >= 65:
        return "GOOD", "Systemet är tillräckligt bra för production"
    elif score >= 50:
        return "ACCEPTABLE", "Systemet fungerar men behöver förbättringar"
    elif score >= 30:
        return "POOR", "Systemet har betydande problem"
    else:
        return "CRITICAL", "Systemet är inte funktionsdugligt"

def generate_recommendations(score, score_components, daily_metrics, signal_metrics):
    """Generera specifika rekommendationer för förbättringar"""
    recommendations = []

    # Decision Balance
    if score_components.get('decision_balance', 0) < 20:
        if daily_metrics and daily_metrics.get('buy_recommendations', 0) == 0:
            recommendations.append("🎯 Systemet genererar inga köprekommendationer - kontrollera Bayesian thresholds")
        elif daily_metrics and daily_metrics.get('sell_recommendations', 0) == 0:
            recommendations.append("🎯 Systemet genererar inga säljrekommendationer - kontrollera risk management")

    # Signal Diversity
    if score_components.get('signal_diversity', 0) < 15:
        recommendations.append("📊 Låg signal-diversitet - överväg att justera decision thresholds i settings.yaml")

    # Regime Detection
    if score_components.get('regime_detection', 0) < 20:
        if signal_metrics and signal_metrics.get('regime_confidence', 0) < 0.50:
            recommendations.append("🔍 Svag regime detection - kontrollera market data quality och lookback periods")

    # System Stability
    if score_components.get('system_stability', 0) < 20:
        recommendations.append("⚠️ Systemstabilitet - kör system_health_check.py för debugging")

    # Specifika förbättringar baserat på score
    if score < 50:
        recommendations.append("🚨 PRIORITET: Kör 'python system_health_check.py' för grundläggande debugging")
        recommendations.append("🔧 Överväg att återställa settings.yaml till default-värden")

    elif score < 70:
        recommendations.append("📈 Överväg parameter-tuning med adaptive_main.py för bättre performance")
        recommendations.append("🎯 Testa olika decision thresholds för mer balanserade rekommendationer")

    return recommendations

def main():
    """Kör benchmark analysis"""
    print("📊 ROI Benchmark Analysis")
    print("=" * 60)
    print("Analyzing system quality against production benchmarks...")
    print()

    start_time = time.time()

    # 1. Analysera daily report
    print("📋 Analyzing latest daily report...")
    daily_metrics, daily_error = analyze_daily_report()

    if daily_error:
        print(f"   ❌ Error: {daily_error}")
    else:
        print(f"   ✅ Report from: {daily_metrics['report_date']}")
        print(f"   📈 Buy/Sell/Hold: {daily_metrics['buy_recommendations']}/{daily_metrics['sell_recommendations']}/{daily_metrics['hold_recommendations']}")

    # 2. Analysera signal quality
    print("\\n🔍 Analyzing signal quality...")
    signal_metrics, signal_error = analyze_signal_quality()

    if signal_error:
        print(f"   ❌ Error: {signal_error}")
    else:
        print(f"   🎯 Regime: {signal_metrics['regime_detected']} ({signal_metrics['regime_confidence']:.1%} confidence)")
        print(f"   📊 Quality: {signal_metrics['regime_quality']}")

    # 3. Beräkna benchmark score
    print("\\n🏆 Calculating benchmark score...")
    score, score_components = calculate_benchmark_score(daily_metrics, signal_metrics)

    level, description = interpret_score(score)

    # 4. Resultat
    elapsed = time.time() - start_time
    print("\\n" + "=" * 60)
    print(f"📊 BENCHMARK RESULTS ({elapsed:.1f}s)")
    print("=" * 60)
    print(f"🎯 Overall Score: {score:.1f}/100 - {level}")
    print(f"📝 Assessment: {description}")
    print()

    # Detaljerade scores
    print("📈 Score Breakdown:")
    for component, points in score_components.items():
        component_name = component.replace('_', ' ').title()
        print(f"   {component_name}: {points:.1f} points")

    # Rekommendationer
    print("\\n🎯 RECOMMENDATIONS:")
    recommendations = generate_recommendations(score, score_components, daily_metrics, signal_metrics)

    if not recommendations:
        print("   🎉 No specific improvements needed - system performing well!")
    else:
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")

    # Nästa steg
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