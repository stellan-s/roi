#!/usr/bin/env python3
"""
Test Script for Portfolio Management Module

Quick test to verify the portfolio management module works correctly.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from quant.modules import ModuleRegistry
from quant.modules.technical import TechnicalIndicatorsModule
from quant.modules.portfolio import PortfolioManagementModule
from quant.config.modules import modules as base_config

def test_portfolio_module():
    """Test the portfolio management module"""
    print("🔍 Testing Portfolio Management Module")
    print("=" * 40)

    # Create registry and register modules
    registry = ModuleRegistry()
    registry.register_module(TechnicalIndicatorsModule)
    registry.register_module(PortfolioManagementModule)

    print(f"✅ Registered {len(registry)} modules")

    # Create module configuration
    module_config = {
        'portfolio_management': base_config['portfolio_management']
    }

    # Create and test individual module first
    portfolio_module = PortfolioManagementModule(module_config['portfolio_management'])

    print("\n🧪 Running Module Health Check...")
    health = portfolio_module.health_check()
    print(f"   Status: {health['status']}")
    print(f"   Tests passed: {health['test_result']['tests_passed']}/{health['test_result']['total_tests']}")
    print(f"   Portfolio positions: {health['test_result']['portfolio_positions']}")
    print(f"   Total allocation: {health['test_result']['total_allocation']:.1%}")
    print(f"   Confidence: {health['test_result']['confidence']:.2f}")

    # Create pipeline with portfolio module
    pipeline = registry.create_pipeline(module_config)
    print(f"\n✅ Created pipeline with {len(pipeline)} modules")

    # Generate test data using the module's built-in generator
    test_inputs = portfolio_module._generate_test_inputs()
    test_candidates = test_inputs['candidate_positions']
    test_prices = test_inputs['current_prices']
    test_current = test_inputs['current_portfolio']

    print(f"\n📊 Generated test data:")
    print(f"   Candidate positions: {len(test_candidates)}")
    print(f"   Current prices: {len(test_prices)}")
    print(f"   Current portfolio positions: {len(test_current)}")

    print(f"\n📈 Candidate Positions:")
    print(f"   {'Ticker':<8} {'Return':<8} {'Conf':<6} {'Regime':<8} {'Risk':<6} {'Decision':<8}")
    print(f"   {'-'*54}")
    for _, row in test_candidates.head().iterrows():
        ticker = row['ticker']
        ret = row['expected_return']
        conf = row['confidence']
        regime = row['regime']
        risk = row['risk_score']
        decision = row['decision']
        print(f"   {ticker:<8} {ret:+.3%} {conf:.2f}   {regime:<8} {risk:.2f}   {decision:<8}")

    print(f"\n💰 Current Portfolio:")
    for ticker, weight in test_current.items():
        print(f"   {ticker}: {weight:.1%}")

    # Execute pipeline
    initial_inputs = {
        'candidate_positions': test_candidates,
        'current_prices': test_prices,
        'current_portfolio': test_current
    }
    results = pipeline.execute(initial_inputs)

    # Display results
    for module_name, result in results.items():
        print(f"\n📊 {module_name} Results:")
        print(f"   Confidence: {result.confidence:.2f}")
        print(f"   Execution time: {result.execution_time_ms:.1f}ms")

        # Portfolio allocation
        if 'portfolio_allocation' in result.data:
            allocation = result.data['portfolio_allocation']
            print(f"\n   🎯 Optimized Portfolio Allocation:")
            print(f"   {'Ticker':<8} {'Weight':<8} {'Value':<10}")
            print(f"   {'-'*26}")

            total_allocation = 0
            for ticker, weight in sorted(allocation.items(), key=lambda x: x[1], reverse=True):
                total_allocation += weight
                print(f"   {ticker:<8} {weight:<8.1%} {'█' * int(weight * 50)}")

            print(f"   {'-'*26}")
            print(f"   {'TOTAL':<8} {total_allocation:<8.1%}")

        # Portfolio metrics
        if 'portfolio_metrics' in result.data:
            metrics = result.data['portfolio_metrics']
            print(f"\n   📊 Portfolio Metrics:")
            print(f"     Total positions: {metrics.get('total_positions', 0)}")
            print(f"     Total allocation: {metrics.get('total_allocation', 0):.1%}")
            print(f"     Expected return (annual): {metrics.get('expected_return_annual', 0):+.1%}")
            print(f"     Risk-adjusted return: {metrics.get('risk_adjusted_return', 0):+.1%}")
            print(f"     Concentration risk: {metrics.get('concentration_risk', 0):.1%}")
            print(f"     Avg confidence: {metrics.get('avg_confidence', 0):.2f}")

            # Regime diversification
            regime_div = metrics.get('regime_diversification', {})
            if regime_div:
                print(f"     Regime diversification:")
                for regime, weight in regime_div.items():
                    emoji = "🐂" if regime == "bull" else "🐻" if regime == "bear" else "⚖️"
                    print(f"       {emoji} {regime.capitalize()}: {weight:.1%}")

        # Trade recommendations
        if 'trade_recommendations' in result.data:
            trades = result.data['trade_recommendations']
            print(f"\n   💼 Trade Recommendations ({len(trades)} trades):")
            if trades:
                print(f"   {'Ticker':<8} {'Action':<6} {'Current':<8} {'Target':<8} {'Change':<8} {'Priority':<8}")
                print(f"   {'-'*58}")

                for trade in trades[:8]:  # Show top 8 trades
                    ticker = trade['ticker']
                    action = trade['action']
                    current = trade['current_weight']
                    target = trade['target_weight']
                    change = trade['weight_change']
                    priority = trade['priority']

                    action_emoji = "🟢" if action == "BUY" else "🔴"
                    print(f"   {ticker:<8} {action_emoji}{action:<5} {current:<8.1%} {target:<8.1%} {change:+.1%}    {priority:.3f}")

        # Rebalancing summary
        if 'rebalancing_summary' in result.data:
            summary = result.data['rebalancing_summary']
            print(f"\n   🔄 Rebalancing Summary:")
            print(f"     Total trades: {summary.get('total_trades', 0)}")
            print(f"     New positions: {summary.get('new_positions', 0)}")
            print(f"     Closed positions: {summary.get('closed_positions', 0)}")
            print(f"     Adjusted positions: {summary.get('adjusted_positions', 0)}")
            print(f"     High priority trades: {summary.get('high_priority_trades', 0)}")
            print(f"     Net weight change: {summary.get('net_weight_change', 0):+.1%}")

        # Show metadata
        if result.metadata:
            print(f"\n   ℹ️ Metadata:")
            metadata = result.metadata
            print(f"     Optimization method: {metadata.get('optimization_method', 'unknown')}")
            print(f"     Positions considered: {metadata.get('positions_considered', 0)}")
            print(f"     Positions selected: {metadata.get('positions_selected', 0)}")
            print(f"     Regime diversification: {metadata.get('regime_diversification', 0)} regimes")

    print("\n🎉 Portfolio management module test completed successfully!")

if __name__ == "__main__":
    test_portfolio_module()