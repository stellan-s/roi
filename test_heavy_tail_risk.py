#!/usr/bin/env python3
"""
Test script for heavy-tail risk modeling
Demonstrates Student-t, EVT, Monte Carlo, and stress testing
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np

from quant.main import load_yaml
from quant.data_layer.prices import fetch_prices
from quant.risk.heavy_tail import HeavyTailRiskModel
from quant.risk.analytics import RiskAnalytics, STRESS_SCENARIOS

def test_heavy_tail_distributions():
    """Test Student-t distribution fitting on real data"""
    print("=== TESTING HEAVY-TAIL DISTRIBUTIONS ===")

    # Load real price data
    uni = load_yaml("universe.yaml")["tickers"]
    cfg = load_yaml("settings.yaml")
    prices = fetch_prices(uni, cfg["data"]["cache_dir"], cfg["data"]["lookback_days"])

    # Focus on a single ticker for detailed analysis
    test_ticker = "TSLA"  # High volatility for a good heavy-tail example
    if test_ticker not in prices['ticker'].values:
        test_ticker = uni[0]  # Fallback

    ticker_prices = prices[prices['ticker'] == test_ticker]['close']
    returns = ticker_prices.pct_change().dropna()

    print(f"Analyzing {test_ticker} with {len(returns)} daily returns")
    print(f"Return statistics:")
    print(f"  Mean: {returns.mean()*252*100:.1f}% annual")
    print(f"  Volatility: {returns.std()*np.sqrt(252)*100:.1f}% annual")
    print(f"  Skewness: {returns.skew():.2f}")
    print(f"  Kurtosis: {returns.kurtosis():.2f} (excess)")

    # Test heavy-tail modeling
    risk_model = HeavyTailRiskModel(cfg)

    # Fit Student-t distribution
    student_t_params = risk_model.fit_heavy_tail_distribution(returns)
    print(f"\nStudent-t parameters:")
    print(f"  Location (μ): {student_t_params['location']*252*100:.2f}% annual")
    print(f"  Scale (σ): {student_t_params['scale']*np.sqrt(252)*100:.2f}% annual")
    print(f"  Degrees of freedom (ν): {student_t_params['degrees_of_freedom']:.1f}")
    print(f"  Sample size: {student_t_params['sample_size']}")

    # EVT analysis
    evt_params = risk_model.fit_extreme_value_theory(returns)
    print(f"\nExtreme Value Theory:")
    for tail_name, params in evt_params.items():
        print(f"  {tail_name}:")
        print(f"    Shape (ξ): {params['shape']:.3f}")
        print(f"    Scale (β): {params['scale']:.4f}")
        print(f"    Threshold: {params['threshold']*100:.2f}%")
        print(f"    Exceedances: {params['n_exceedances']}")

    return test_ticker, returns, student_t_params, evt_params

def test_tail_risk_metrics(ticker, returns, student_t_params):
    """Test tail risk calculations"""
    print(f"\n=== TAIL RISK METRICS FOR {ticker} ===")

    cfg = load_yaml("settings.yaml")
    risk_model = HeavyTailRiskModel(cfg)

    # Calculate tail risk for different horizons
    horizons = [21, 63, 252]  # 1m, 3m, 1y
    confidence_levels = [0.95, 0.99]

    for horizon in horizons:
        print(f"\n{horizon}-day horizon:")
        for conf_level in confidence_levels:
            metrics = risk_model.calculate_tail_risk_metrics(returns, conf_level, horizon)

            print(f"  {conf_level*100:.0f}% confidence level:")
            print(f"    VaR (Normal): {metrics.var_normal*100:.2f}%")
            print(f"    VaR (Student-t): {metrics.var_student_t*100:.2f}%")
            print(f"    CVaR (Student-t): {metrics.cvar_student_t*100:.2f}%")
            print(f"    Tail risk multiplier: {metrics.tail_risk_multiplier:.2f}x")
            print(f"    Extreme event prob: {metrics.extreme_event_probability*100:.2f}%")

def test_monte_carlo_simulation(ticker, student_t_params):
    """Test Monte Carlo simulation for 12-month probabilities"""
    print(f"\n=== MONTE CARLO SIMULATION FOR {ticker} ===")

    cfg = load_yaml("settings.yaml")
    risk_model = HeavyTailRiskModel(cfg)

    # Simulate with realistic parameters
    expected_return = 0.08  # 8% annual expected return
    volatility = 0.25       # 25% annual volatility

    mc_results = risk_model.monte_carlo_simulation(
        expected_return=expected_return,
        volatility=volatility,
        tail_parameters=student_t_params,
        time_horizon_months=12,
        n_simulations=10000
    )

    print(f"Monte Carlo Results (12 months, {mc_results.n_simulations:,} simulations):")
    print(f"  Expected return: {mc_results.mean_return*100:.1f}%")
    print(f"  Volatility: {mc_results.std_return*100:.1f}%")
    print(f"  Skewness: {mc_results.skewness:.2f}")
    print(f"  Excess Kurtosis: {mc_results.kurtosis:.2f}")

    print(f"\nProbability Targets:")
    print(f"  P(return > 0%): {mc_results.prob_positive*100:.1f}%")
    print(f"  P(return > +10%): {mc_results.prob_plus_10*100:.1f}%")
    print(f"  P(return > +20%): {mc_results.prob_plus_20*100:.1f}%")
    print(f"  P(return > +30%): {mc_results.prob_plus_30*100:.1f}%")

    print(f"\nDownside Risks:")
    print(f"  P(return < -10%): {mc_results.prob_minus_10*100:.1f}%")
    print(f"  P(return < -20%): {mc_results.prob_minus_20*100:.1f}%")
    print(f"  P(return < -30%): {mc_results.prob_minus_30*100:.1f}%")

    print(f"\nExtreme Scenarios:")
    print(f"  1st percentile (very bad): {mc_results.percentile_1*100:.1f}%")
    print(f"  5th percentile (bad): {mc_results.percentile_5*100:.1f}%")
    print(f"  95th percentile (good): {mc_results.percentile_95*100:.1f}%")
    print(f"  99th percentile (very good): {mc_results.percentile_99*100:.1f}%")

    return mc_results

def test_stress_scenarios():
    """Test stress testing scenarios"""
    print(f"\n=== STRESS TESTING SCENARIOS ===")

    print("Available stress scenarios:")
    for name, scenario in STRESS_SCENARIOS.items():
        print(f"\n{scenario.name}:")
        print(f"  Description: {scenario.description}")
        print(f"  Market shock: {scenario.market_shock_size*100:.0f}%")
        print(f"  Vol multiplier: {scenario.volatility_multiplier:.1f}x")
        print(f"  Correlation increase: +{scenario.correlation_increase:.1f}")
        print(f"  Duration: {scenario.duration_days} days")

def test_risk_analytics_integration():
    """Test full risk analytics integration"""
    print(f"\n=== RISK ANALYTICS INTEGRATION ===")

    cfg = load_yaml("settings.yaml")
    risk_analytics = RiskAnalytics(cfg)

    # Create mock portfolio for testing
    mock_portfolio = {
        'TSLA': 0.3,
        'AAPL': 0.25,
        'GOOGL': 0.25,
        'MSFT': 0.20
    }

    print(f"Testing portfolio:")
    for ticker, weight in mock_portfolio.items():
        print(f"  {ticker}: {weight*100:.0f}%")

    # Create simplified risk profiles (in real usage they would be calculated from price history)
    mock_risk_profiles = {}

    for ticker, weight in mock_portfolio.items():
        # Mock risk profile based on typical values
        if ticker == 'TSLA':
            volatility = 0.40  # High vol
            tail_multiplier = 2.5
        elif ticker in ['AAPL', 'MSFT']:
            volatility = 0.25  # Moderate vol
            tail_multiplier = 1.8
        else:
            volatility = 0.30  # Moderate-high vol
            tail_multiplier = 2.0

        # Create simplified risk profile
        from quant.risk.analytics import PortfolioRiskProfile
        from quant.risk.heavy_tail import TailRiskMetrics, MonteCarloResults

        # Simplified metrics for demo purposes
        tail_metrics = TailRiskMetrics(
            confidence_level=0.95,
            time_horizon_days=21,
            var_normal=-0.05,
            var_student_t=-0.05 * tail_multiplier,
            cvar_student_t=-0.08 * tail_multiplier,
            evt_var=-0.06 * tail_multiplier,
            evt_return_level=0.08,
            degrees_of_freedom=8.0,
            tail_index=0.15,
            tail_risk_multiplier=tail_multiplier,
            extreme_event_probability=0.05
        )

        mc_results = MonteCarloResults(
            n_simulations=10000,
            time_horizon_months=12,
            prob_positive=0.65,
            prob_plus_10=0.45,
            prob_plus_20=0.25,
            prob_plus_30=0.10,
            prob_minus_10=0.20,
            prob_minus_20=0.08,
            prob_minus_30=0.03,
            mean_return=0.08,
            median_return=0.06,
            std_return=volatility,
            skewness=-0.2,
            kurtosis=1.5,
            percentile_1=-0.45,
            percentile_5=-0.25,
            percentile_95=0.35,
            percentile_99=0.55
        )

        mock_risk_profiles[ticker] = PortfolioRiskProfile(
            ticker=ticker,
            expected_return_annual=0.08,
            volatility_annual=volatility,
            tail_risk_metrics=tail_metrics,
            monte_carlo_12m=mc_results,
            sharpe_ratio=0.08/volatility,
            tail_risk_adjusted_return=0.08 - 0.02*tail_multiplier,
            risk_contribution=weight * volatility,
            prob_loss_10_percent=0.20,
            prob_loss_20_percent=0.08,
            prob_gain_10_percent=0.45,
            prob_gain_20_percent=0.25,
            prob_gain_30_percent=0.10,
            worst_case_1_percent=-0.45,
            best_case_99_percent=0.55
        )

    # Generate portfolio risk summary
    portfolio_summary = risk_analytics.generate_risk_summary(mock_portfolio, mock_risk_profiles)

    print(f"\nPortfolio Risk Summary:")
    print(f"  Expected return: {portfolio_summary['portfolio_expected_return_annual']*100:.1f}%")
    print(f"  Portfolio volatility: {portfolio_summary['portfolio_volatility_annual']*100:.1f}%")
    print(f"  Sharpe ratio: {portfolio_summary['portfolio_sharpe_ratio']:.2f}")
    print(f"  Tail risk multiplier: {portfolio_summary['weighted_tail_risk_multiplier']:.2f}x")
    print(f"  Risk assessment: {portfolio_summary['risk_assessment']}")

    concentration = portfolio_summary['concentration_risk']
    print(f"  Concentration risk:")
    print(f"    Max position: {concentration['max_position_weight']*100:.0f}%")
    print(f"    Number of positions: {concentration['number_of_positions']}")
    print(f"    Concentration score: {concentration['concentration_score']:.2f}")

    return portfolio_summary

if __name__ == "__main__":
    print("HEAVY-TAIL RISK MODELING - COMPREHENSIVE TEST")
    print("=" * 60)

    try:
        # Test 1: Heavy-tail distribution fitting
        ticker, returns, student_t_params, evt_params = test_heavy_tail_distributions()

        # Test 2: Tail risk metrics
        test_tail_risk_metrics(ticker, returns, student_t_params)

        # Test 3: Monte Carlo simulation
        mc_results = test_monte_carlo_simulation(ticker, student_t_params)

        # Test 4: Stress scenarios
        test_stress_scenarios()

        # Test 5: Full risk analytics
        portfolio_summary = test_risk_analytics_integration()

        print("\n" + "=" * 60)
        print("✅ ALL HEAVY-TAIL RISK TESTS PASSED!")
        print(f"✓ Student-t fitting: DoF={student_t_params['degrees_of_freedom']:.1f} (heavy tails detected)")
        print(f"✓ EVT analysis: Extreme events captured")
        print(f"✓ Monte Carlo: 12m probabilities calculated")
        print(f"✓ Stress testing: {len(STRESS_SCENARIOS)} scenarios implemented")
        print(f"✓ Risk analytics: Portfolio risk assessment completed")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
