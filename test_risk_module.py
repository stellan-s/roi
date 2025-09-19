#!/usr/bin/env python3
"""
Test Script for Risk Management Module

Quick test to verify the risk management module works correctly.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from quant.modules import ModuleRegistry
from quant.modules.technical import TechnicalIndicatorsModule
from quant.modules.risk import RiskManagementModule
from quant.config.modules import modules as base_config

def test_risk_module():
    """Test the risk management module"""
    print("🔍 Testing Risk Management Module")
    print("=" * 40)

    # Create registry and register modules
    registry = ModuleRegistry()
    registry.register_module(TechnicalIndicatorsModule)
    registry.register_module(RiskManagementModule)

    print(f"✅ Registered {len(registry)} modules")

    # Create module configuration
    module_config = {
        'risk_management': base_config['risk_management']
    }

    # Create and test individual module first
    risk_module = RiskManagementModule(module_config['risk_management'])

    print("\n🧪 Running Module Health Check...")
    health = risk_module.health_check()
    print(f"   Status: {health['status']}")
    print(f"   Tests passed: {health['test_result']['tests_passed']}/{health['test_result']['total_tests']}")
    print(f"   Portfolio volatility: {health['test_result']['portfolio_volatility']:.1%}")
    print(f"   Confidence: {health['test_result']['confidence']:.2f}")

    # Create pipeline with risk module
    pipeline = registry.create_pipeline(module_config)
    print(f"\n✅ Created pipeline with {len(pipeline)} modules")

    # Generate test data using the module's built-in generator
    test_inputs = risk_module._generate_test_inputs()
    test_prices = test_inputs['prices']
    test_weights = test_inputs['portfolio_weights']
    test_returns = test_inputs['expected_returns']

    print(f"\n📊 Generated test data:")
    print(f"   Price records: {len(test_prices)}")
    print(f"   Tickers: {list(test_weights.keys())}")
    print(f"   Portfolio weights:")
    for ticker, weight in test_weights.items():
        print(f"     {ticker}: {weight:.1%}")
    print(f"   Expected returns:")
    for ticker, ret in test_returns.items():
        print(f"     {ticker}: {ret:.1%}")

    # Calculate some basic stats from test data
    print(f"\n📈 Test Data Statistics:")
    for ticker in test_weights.keys():
        ticker_data = test_prices[test_prices['ticker'] == ticker]
        start_price = ticker_data.iloc[0]['close']
        end_price = ticker_data.iloc[-1]['close']
        total_return = (end_price - start_price) / start_price
        daily_returns = ticker_data['close'].pct_change().dropna()
        volatility = daily_returns.std() * (252**0.5)  # Annualized
        print(f"     {ticker}: Total return {total_return:+.1%}, Volatility {volatility:.1%}")

    # Execute pipeline
    initial_inputs = {
        'prices': test_prices,
        'portfolio_weights': test_weights,
        'expected_returns': test_returns
    }
    results = pipeline.execute(initial_inputs)

    # Display results
    for module_name, result in results.items():
        print(f"\n📊 {module_name} Results:")
        print(f"   Confidence: {result.confidence:.2f}")
        print(f"   Execution time: {result.execution_time_ms:.1f}ms")

        # Individual risk analysis
        if 'individual_risks' in result.data:
            individual_risks = result.data['individual_risks']
            print(f"\n   📋 Individual Risk Analysis:")
            print(f"   {'Ticker':<8} {'Vol':<8} {'Tail Risk':<10} {'Sharpe':<8} {'Quality':<8}")
            print(f"   {'-'*48}")

            for ticker, risk in individual_risks.items():
                vol = risk['volatility_annual']
                tail_risk = risk['downside_tail_risk']
                sharpe = risk['sharpe_ratio']
                quality = risk['data_quality_score']

                print(f"   {ticker:<8} {vol:<8.1%} {tail_risk:<10.2%} {sharpe:<8.2f} {quality:<8.2f}")

        # Portfolio risk analysis
        if 'portfolio_risk' in result.data and result.data['portfolio_risk']:
            portfolio_risk = result.data['portfolio_risk']
            print(f"\n   🎯 Portfolio Risk Summary:")
            print(f"     Portfolio Volatility: {portfolio_risk['volatility_annual']:.1%}")
            print(f"     Portfolio Tail Risk: {portfolio_risk['downside_tail_risk']:.2%}")
            print(f"     Portfolio Sharpe: {portfolio_risk['sharpe_ratio']:.2f}")
            print(f"     Expected Shortfall: {portfolio_risk['expected_shortfall']:.2%}")

        # Stress test results
        if 'stress_test_results' in result.data:
            stress_results = result.data['stress_test_results']
            print(f"\n   ⚡ Stress Test Results:")
            for scenario, results in stress_results.items():
                loss_pct = results['total_portfolio_loss_pct']
                duration = results['duration_days']
                emoji = "🔥" if loss_pct < -0.25 else "⚠️" if loss_pct < -0.15 else "📉"
                print(f"     {emoji} {scenario}: {loss_pct:+.1%} over {duration} days")

        # Risk summary
        if 'risk_summary' in result.data:
            summary = result.data['risk_summary']
            print(f"\n   📊 Risk Summary:")
            print(f"     Total positions: {summary.get('total_positions', 0)}")
            print(f"     Average volatility: {summary.get('avg_volatility', 0):.1%}")
            print(f"     High risk positions: {summary.get('high_risk_positions', 0)}")
            print(f"     Portfolio risk level: {summary.get('portfolio_risk_level', 'UNKNOWN')}")
            print(f"     Data quality: {summary.get('data_quality_avg', 0):.2f}")

        # Risk recommendations
        if 'risk_recommendations' in result.data:
            recommendations = result.data['risk_recommendations']
            print(f"\n   💡 Risk Recommendations:")
            for ticker, rec in recommendations.items():
                action = rec['recommended_action']
                risk_level = rec['risk_level']
                max_weight = rec['max_recommended_weight']
                current_weight = rec['current_weight']

                emoji = "🔴" if action == "REDUCE" else "🟡" if action == "HOLD" else "🟢"
                print(f"     {emoji} {ticker}: {action} (Risk: {risk_level}, Current: {current_weight:.1%}, Max: {max_weight:.1%})")

                if rec['reasoning']:
                    reasoning = "; ".join(rec['reasoning'][:2])  # Show top 2 reasons
                    print(f"        Reason: {reasoning}")

    print("\n🎉 Risk management module test completed successfully!")

if __name__ == "__main__":
    test_risk_module()