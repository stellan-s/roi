"""
Modular Main Entry Point - Fully Modular Trading System

This version uses the complete modular architecture with ModuleRegistry
and pipeline execution instead of legacy imports.
"""

import pandas as pd
from datetime import datetime
from typing import Dict, List

from quant.data_layer.prices import fetch_prices
from quant.data_layer.news import fetch_news
from quant.data_layer.macro import fetch_vix
from quant.modules import ModuleRegistry
from quant.modules.technical import TechnicalIndicatorsModule
from quant.modules.sentiment import SentimentAnalysisModule
from quant.modules.regime import RegimeDetectionModule
from quant.modules.risk import RiskManagementModule
from quant.modules.portfolio import PortfolioManagementModule
from quant.config.modules import modules as module_config
from quant.config.loader import load_configuration
from quant.reports.daily_brief import save_daily_markdown


def create_modular_pipeline():
    """Create and configure the modular pipeline."""
    print("🧩 Setting up modular pipeline...")

    # Create registry and register all modules
    registry = ModuleRegistry()
    registry.register_module(TechnicalIndicatorsModule)
    registry.register_module(SentimentAnalysisModule)
    registry.register_module(RegimeDetectionModule)
    registry.register_module(RiskManagementModule)
    registry.register_module(PortfolioManagementModule)

    print(f"✅ Registered {len(registry)} modules")

    # Create pipeline with all enabled modules
    pipeline = registry.create_pipeline(module_config)
    print(f"✅ Created pipeline with {len(pipeline)} active modules")

    return pipeline


def fetch_data(config: Dict) -> Dict:
    """Fetch all required data for the modular pipeline."""
    print("📊 Fetching market data...")

    # Fetch prices
    tickers = config['universe']['tickers']
    prices_df = fetch_prices(tickers, config['data']['cache_dir'])
    print(f"✅ Fetched price data for {len(prices_df['ticker'].unique())} tickers")

    # Fetch news
    news_feeds = config.get('data', {}).get('news_feeds', [
        "https://news.cision.com/se/rss/all",
        "https://www.di.se/rss/",
        "https://finance.yahoo.com/news/rssindex"
    ])
    sentiment_df = fetch_news(news_feeds, config['data']['cache_dir'])
    print(f"✅ Fetched {len(sentiment_df)} news articles")

    # Fetch VIX data
    vix_df = fetch_vix(config['data']['cache_dir'])
    print(f"✅ Fetched VIX data: {len(vix_df)} records")

    return {
        'prices': prices_df,
        'news': sentiment_df,
        'tickers': tickers,
        'vix_data': vix_df
    }


def execute_modular_pipeline(pipeline, data: Dict) -> Dict:
    """Execute the modular pipeline with market data."""
    print("\n🔄 Executing modular pipeline...")

    # Execute modules individually to handle dependencies
    results = {}

    # Step 1: Technical Analysis
    print("   🔧 Running technical indicators...")
    tech_module = TechnicalIndicatorsModule(module_config['technical_indicators'])
    tech_result = tech_module.process({'prices': data['prices']})
    results['technical_indicators'] = tech_result

    # Step 2: Sentiment Analysis
    print("   📰 Running sentiment analysis...")
    sentiment_module = SentimentAnalysisModule(module_config['sentiment_analysis'])
    sentiment_result = sentiment_module.process({
        'news': data['news'],
        'tickers': data['tickers']
    })
    results['sentiment_analysis'] = sentiment_result

    # Step 3: Regime Detection
    print("   ⚖️ Running regime detection...")
    regime_module = RegimeDetectionModule(module_config['regime_detection'])
    regime_result = regime_module.process({
        'prices': data['prices'],
        'vix_data': data.get('vix_data')
    })
    results['regime_detection'] = regime_result

    # Step 4: Risk Management
    print("   ⚠️ Running risk management...")
    risk_module = RiskManagementModule(module_config['risk_management'])

    # Create expected returns from technical and sentiment signals
    expected_returns = {}
    for ticker in data['tickers']:
        tech_signals = tech_result.data.get('signals', {}).get(ticker, {})
        sentiment_signals = sentiment_result.data.get('sentiment_signals', {})

        sma_signal = tech_signals.get('sma_signal', 0)
        momentum = tech_signals.get('momentum', 0)
        sentiment_signal = sentiment_signals.get(ticker, 0)

        expected_return = (sma_signal * 0.4 + momentum * 0.4 + sentiment_signal * 0.2)
        expected_returns[ticker] = expected_return

    risk_result = risk_module.process({
        'prices': data['prices'],
        'expected_returns': expected_returns
    })
    results['risk_management'] = risk_result

    # Step 5: Portfolio Management
    print("   💼 Running portfolio management...")

    # Create candidate positions from previous results
    candidate_positions = []
    regime = regime_result.data.get('current_regime', 'neutral')

    for ticker in data['tickers']:
        tech_signals = tech_result.data.get('signals', {}).get(ticker, {})
        sentiment_signals = sentiment_result.data.get('sentiment_signals', {})
        individual_risks = risk_result.data.get('individual_risks', {})

        expected_return = expected_returns.get(ticker, 0)
        confidence = min(0.9, 0.3 + abs(expected_return) * 5)
        risk_metrics = individual_risks.get(ticker, {})
        risk_score = 1.0 - confidence

        decision = 'Buy' if expected_return > 0.001 else 'Sell' if expected_return < -0.001 else 'Hold'

        candidate_positions.append({
            'ticker': ticker,
            'expected_return': expected_return,
            'confidence': confidence,
            'regime': regime,
            'risk_score': risk_score,
            'decision': decision,
            'prob_positive': confidence if expected_return > 0 else 1 - confidence
        })

    candidates_df = pd.DataFrame(candidate_positions)
    current_prices = data['prices'].groupby('ticker').last().reset_index()[['ticker', 'close']]

    portfolio_module = PortfolioManagementModule(module_config['portfolio_management'])
    portfolio_result = portfolio_module.process({
        'candidate_positions': candidates_df,
        'current_prices': current_prices,
        'current_portfolio': {}  # No existing portfolio for now
    })
    results['portfolio_management'] = portfolio_result

    # Display execution summary
    total_time = sum(getattr(result, 'execution_time_ms', 0) or 0 for result in results.values())
    avg_confidence = sum(result.confidence for result in results.values()) / len(results)

    print(f"✅ Pipeline executed successfully")
    print(f"   Total execution time: {total_time:.1f}ms")
    print(f"   Average confidence: {avg_confidence:.2f}")
    print(f"   Modules executed: {len(results)}")

    return results


def create_trading_recommendations(results: Dict, data: Dict) -> pd.DataFrame:
    """Create final trading recommendations from modular pipeline results."""
    print("\n💼 Creating trading recommendations...")

    # Get results from each module
    technical_result = results.get('technical_indicators')
    sentiment_result = results.get('sentiment_analysis')
    regime_result = results.get('regime_detection')
    risk_result = results.get('risk_management')
    portfolio_result = results.get('portfolio_management')

    # Extract portfolio allocation
    portfolio_allocation = portfolio_result.data.get('portfolio_allocation', {}) if portfolio_result else {}
    trade_recommendations = portfolio_result.data.get('trade_recommendations', []) if portfolio_result else []

    # Get current regime
    current_regime = regime_result.data.get('current_regime', 'neutral') if regime_result else 'neutral'
    regime_confidence = regime_result.confidence if regime_result else 0.5

    # Create recommendations DataFrame
    recommendations = []

    for ticker in data['tickers']:
        # Get technical signals
        tech_signals = technical_result.data.get('signals', {}).get(ticker, {}) if technical_result else {}
        sma_signal = tech_signals.get('sma_signal', 0)
        momentum = tech_signals.get('momentum', 0)

        # Get sentiment signal
        sentiment_signals = sentiment_result.data.get('sentiment_signals', {}) if sentiment_result else {}
        sentiment_signal = sentiment_signals.get(ticker, 0)

        # Get risk metrics
        individual_risks = risk_result.data.get('individual_risks', {}) if risk_result else {}
        risk_metrics = individual_risks.get(ticker, {})
        tail_risk = risk_metrics.get('downside_tail_risk', 0.02)
        volatility = risk_metrics.get('volatility_annual', 0.20)

        # Calculate combined expected return (simple combination)
        expected_return = (sma_signal * 0.4 + momentum * 0.3 + sentiment_signal * 0.3)

        # Get portfolio weight
        portfolio_weight = portfolio_allocation.get(ticker, 0.0)

        # Determine decision
        if portfolio_weight > 0.01:  # >1% allocation
            decision = 'Buy'
        elif expected_return < -0.005:  # <-0.5% expected return
            decision = 'Sell'
        else:
            decision = 'Hold'

        # Calculate probability of positive return
        confidence = min(0.9, 0.5 + abs(expected_return) * 10)
        prob_positive = confidence if expected_return > 0 else 1 - confidence

        # Get current price
        ticker_prices = data['prices'][data['prices']['ticker'] == ticker]
        current_price = ticker_prices.iloc[-1]['close'] if not ticker_prices.empty else 100.0

        recommendations.append({
            'date': pd.Timestamp.now().date(),
            'ticker': ticker,
            'close': current_price,
            'decision': decision,
            'expected_return': expected_return,
            'prob_positive': prob_positive,
            'decision_confidence': confidence,
            'uncertainty': 1 - confidence,
            'trend_weight': sma_signal,
            'momentum_weight': momentum,
            'sentiment_weight': sentiment_signal,
            'regime': current_regime,
            'regime_confidence': regime_confidence,
            'tail_risk': tail_risk,
            'extreme_move_prob': risk_metrics.get('extreme_move_prob', 0.05),
            'portfolio_weight': portfolio_weight,
            'portfolio_adjusted': True
        })

    recommendations_df = pd.DataFrame(recommendations)

    # Filter to only actionable recommendations
    actionable = recommendations_df[recommendations_df['decision'].isin(['Buy', 'Sell'])]
    print(f"✅ Generated {len(actionable)} actionable recommendations from {len(recommendations_df)} analyzed tickers")

    return recommendations_df


def execute_portfolio_management(recommendations: pd.DataFrame, config: Dict) -> List[Dict]:
    """Execute portfolio management and trade simulation."""
    print("\n💰 Portfolio management and trade simulation...")

    # Import portfolio management (keeping existing portfolio state functionality)
    from quant.portfolio.state import PortfolioTracker

    # Load current portfolio state
    portfolio_tracker = PortfolioTracker(config['data']['cache_dir'] + "/portfolio")

    # Update portfolio with latest prices
    current_prices = recommendations[['ticker', 'close']].drop_duplicates()
    current_state = portfolio_tracker.update_portfolio_state(current_prices)

    # Execute trades based on recommendations
    buy_recommendations = recommendations[
        (recommendations['decision'] == 'Buy') &
        (recommendations['portfolio_weight'] > 0)
    ]

    executed_trades = []
    if not buy_recommendations.empty:
        executed_trades = portfolio_tracker.execute_trades(
            buy_recommendations,
            current_prices,
            cash_per_position=10000.0
        )

    return executed_trades


def save_recommendations_log(recommendations: pd.DataFrame, executed_trades: List[Dict], cache_dir: str):
    """Save recommendations and trades for audit trail."""
    print("\n📝 Saving recommendation logs...")

    log_dir = Path(cache_dir) / "recommendation_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    run_date = pd.to_datetime(recommendations['date']).max()
    run_date_str = pd.to_datetime(run_date).date().isoformat()
    timestamp = datetime.utcnow().isoformat()

    recommendations_to_log = recommendations.copy()
    recommendations_to_log['logged_at_utc'] = timestamp

    rec_path = log_dir / f"recommendations_{run_date_str}.parquet"
    recommendations_to_log.to_parquet(rec_path, index=False)
    print(f"📝 Saved recommendations to {rec_path}")

    trades_path = log_dir / f"simulated_trades_{run_date_str}.json"
    with open(trades_path, 'w') as f:
        import json
        json.dump({
            'run_date': run_date_str,
            'logged_at_utc': timestamp,
            'trades': executed_trades
        }, f, indent=2)

    if executed_trades:
        print(f"🧾 Logged {len(executed_trades)} simulated trades to {trades_path}")
    else:
        print("ℹ️ No simulated trades executed today")


def main():
    """
    Main execution with fully modular pipeline.
    """
    print("=== ROI Modular Trading System ===")

    # Load configuration
    config = load_configuration()
    print(f"Loaded configuration for {len(config['universe']['tickers'])} tickers")

    # Create modular pipeline
    pipeline = create_modular_pipeline()

    # Fetch all required data
    data = fetch_data(config)

    # Execute modular pipeline
    results = execute_modular_pipeline(pipeline, data)

    # Create trading recommendations from modular results
    recommendations = create_trading_recommendations(results, data)

    # Execute portfolio management
    executed_trades = execute_portfolio_management(recommendations, config)

    # Save logs
    save_recommendations_log(recommendations, executed_trades, config['data']['cache_dir'])

    # Generate daily report with portfolio performance
    print("\n📊 Generating daily report...")
    report_dir = "reports"

    # Get current portfolio summary for the report
    portfolio_summary = portfolio_tracker.get_portfolio_summary() if portfolio_tracker else {}

    save_daily_markdown(recommendations, report_dir, portfolio_summary)
    print("✅ Daily report saved with portfolio performance")

    # Show summary
    buy_count = len(recommendations[recommendations['decision'] == 'Buy'])
    sell_count = len(recommendations[recommendations['decision'] == 'Sell'])

    print(f"\n📈 Summary:")
    print(f"   Buy recommendations: {buy_count}")
    print(f"   Sell recommendations: {sell_count}")
    print(f"   Executed trades: {len(executed_trades)}")
    print(f"   Total positions analyzed: {len(recommendations)}")

    print("\n🎉 Modular trading system execution completed!")


if __name__ == "__main__":
    main()