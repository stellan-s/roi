"""
Adaptive Modular Main Entry Point - Modular Trading System with Adaptive Learning

This version combines the complete modular architecture with the AdaptiveBayesianEngine
for data-driven parameter estimation and calibration.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

from quant.data_layer.prices import fetch_prices
from quant.data_layer.news import fetch_news
from quant.data_layer.macro import fetch_vix, fetch_precious_metals_sentiment, fetch_macro_indicators
from quant.features.technical import compute_technical_features
from quant.features.sentiment import naive_sentiment
from quant.modules import ModuleRegistry
from quant.modules.technical import TechnicalIndicatorsModule
from quant.modules.sentiment import SentimentAnalysisModule
from quant.modules.regime import RegimeDetectionModule
from quant.modules.fundamentals import FundamentalsModule
from quant.modules.risk import RiskManagementModule
from quant.modules.portfolio import PortfolioManagementModule
from quant.config.modules import modules as module_config
from quant.config.loader import load_configuration
from quant.bayesian.adaptive_integration import AdaptiveBayesianEngine
from quant.reports.daily_brief import save_daily_markdown
from quant.portfolio.state import PortfolioTracker


def prepare_historical_data(config: Dict) -> tuple:
    """
    Prepare historical data for parameter estimation.

    Returns:
        tuple: (prices_df, sentiment_df, technical_df, returns_df)
    """
    print("📊 Preparing historical data for parameter estimation...")

    # Load universe
    universe = config['universe']['tickers']

    # Fetch extended historical data for calibration
    calibration_lookback = config['data'].get('lookback_days', 500) * 2  # Extra data for calibration

    prices = fetch_prices(
        tickers=universe,
        cache_dir=config['data']['cache_dir'],
        lookback_days=calibration_lookback
    )
    print(f"✅ Fetched {len(prices)} historical price records")

    news = fetch_news(
        feed_urls=config['signals']['news_feed_urls'],
        cache_dir=config['data']['cache_dir']
    )
    print(f"✅ Fetched {len(news)} news articles for sentiment")

    # Compute features
    tech = compute_technical_features(
        prices,
        config["signals"]["sma_long"],
        config["signals"]["momentum_window"]
    )
    print(f"✅ Computed technical features for {len(tech['ticker'].unique())} tickers")

    senti = naive_sentiment(news, universe)
    print(f"✅ Computed sentiment features")

    # Create returns dataframe
    returns = prices.groupby('ticker').apply(
        lambda x: x.assign(**{'return': x['close'].pct_change()})
    ).reset_index(drop=True)

    return prices, senti, tech, returns


def calibrate_adaptive_engine(config: Dict, prices_df: pd.DataFrame, sentiment_df: pd.DataFrame,
                            technical_df: pd.DataFrame, returns_df: pd.DataFrame) -> AdaptiveBayesianEngine:
    """Calibrate the adaptive Bayesian engine with historical data."""
    print("\n🧠 Calibrating adaptive Bayesian engine...")

    engine = AdaptiveBayesianEngine(config)

    # Calibrate parameters
    engine.calibrate_parameters(
        prices_df=prices_df,
        technical_df=technical_df,
        sentiment_df=sentiment_df,
        returns_df=returns_df
    )

    print("✅ Adaptive engine calibration completed")
    return engine


def create_adaptive_modular_pipeline():
    """Create and configure the modular pipeline for adaptive execution."""
    print("\n🧩 Setting up adaptive modular pipeline...")

    # Create registry and register all modules
    registry = ModuleRegistry()
    registry.register_module(TechnicalIndicatorsModule)
    registry.register_module(SentimentAnalysisModule)
    registry.register_module(RegimeDetectionModule)
    registry.register_module(FundamentalsModule)
    registry.register_module(RiskManagementModule)
    registry.register_module(PortfolioManagementModule)

    print(f"✅ Registered {len(registry)} modules")

    # Create pipeline with all enabled modules
    pipeline = registry.create_pipeline(module_config)
    print(f"✅ Created adaptive pipeline with {len(pipeline)} active modules")

    return pipeline


def fetch_current_data(config: Dict) -> Dict:
    """Fetch current market data for adaptive pipeline."""
    print("\n📊 Fetching current market data...")

    # Fetch prices
    tickers = config['universe']['tickers']
    prices_df = fetch_prices(tickers, config['data']['cache_dir'])
    print(f"✅ Fetched current price data for {len(prices_df['ticker'].unique())} tickers")

    # Fetch news
    news_feeds = config.get('signals', {}).get('news_feed_urls', [
        "https://news.cision.com/se/rss/all",
        "https://www.di.se/rss/",
        "https://finance.yahoo.com/news/rssindex"
    ])
    sentiment_df = fetch_news(news_feeds, config['data']['cache_dir'])
    print(f"✅ Fetched {len(sentiment_df)} current news articles")

    # Fetch VIX data
    vix_df = fetch_vix(config['data']['cache_dir'])
    print(f"✅ Fetched VIX data: {len(vix_df)} records")

    # Fetch macro data
    macro_data = {}
    try:
        macro_data['precious_metals'] = fetch_precious_metals_sentiment(config['data']['cache_dir'])
        macro_data['indicators'] = fetch_macro_indicators(config['data']['cache_dir'])
        print(f"✅ Fetched macro indicators")
    except Exception as e:
        print(f"⚠️ Could not fetch macro data: {e}")

    return {
        'prices': prices_df,
        'news': sentiment_df,
        'tickers': tickers,
        'vix_data': vix_df,
        'macro_data': macro_data
    }


def execute_adaptive_modular_pipeline(pipeline, data: Dict, adaptive_engine: AdaptiveBayesianEngine, config: Dict) -> Tuple[Dict, PortfolioTracker]:
    """Execute the adaptive modular pipeline with calibrated engine."""
    print("\n🔄 Executing adaptive modular pipeline...")

    # Execute modules individually to handle dependencies and adaptive integration
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

    # Step 4: Fundamentals Analysis
    print("   📊 Running fundamentals analysis...")
    fundamentals_module = FundamentalsModule(module_config.get('fundamentals', {}))
    fundamentals_result = fundamentals_module.execute({
        'tickers': data['tickers']
    })
    results['fundamentals'] = fundamentals_result

    # Step 5: Adaptive Bayesian Integration
    print("   🧠 Running adaptive Bayesian integration...")

    # Transform modular data to legacy format expected by bayesian_score
    tech_signals_df = tech_result.data.get('technical_features', pd.DataFrame())
    sentiment_signals_df = sentiment_result.data.get('sentiment_features', pd.DataFrame())
    fundamentals_data = fundamentals_result.data.get('fundamentals', {})

    # Create combined dataframe with expected column names
    combined_data = []

    for ticker in data['tickers']:
        # Get technical data
        tech_data = tech_signals_df[tech_signals_df['ticker'] == ticker] if not tech_signals_df.empty else pd.DataFrame()
        senti_data = sentiment_signals_df[sentiment_signals_df['ticker'] == ticker] if not sentiment_signals_df.empty else pd.DataFrame()

        # Get latest price
        ticker_prices = data['prices'][data['prices']['ticker'] == ticker]
        if ticker_prices.empty:
            continue

        latest_price = ticker_prices.iloc[-1]

        # Get fundamentals data
        ticker_fundamentals = fundamentals_data.get(ticker, {})

        # Create row with expected column names
        row = {
            'ticker': ticker,
            'date': pd.to_datetime(latest_price['date'] if 'date' in latest_price else pd.Timestamp.now()),
            'close': latest_price['close'],
            # Technical indicators (convert from modular format)
            'above_sma': tech_data.iloc[0].get('sma_signal', 0) if not tech_data.empty else 0,
            'mom_rank': tech_data.iloc[0].get('momentum', 0) if not tech_data.empty else 0,
            # Sentiment (convert from modular format)
            'sent_score': senti_data.iloc[0].get('sentiment_score', 0) if not senti_data.empty else 0,
            # Fundamentals (from modular format)
            'fundamental_score': ticker_fundamentals.get('fundamental_score', 0.5)
        }
        combined_data.append(row)

    # Create combined DataFrame for bayesian_score
    combined_df = pd.DataFrame(combined_data)

    if combined_df.empty:
        print("⚠️ No data available for Bayesian integration")
        adaptive_recommendations = pd.DataFrame(columns=['ticker', 'expected_return', 'prob_positive', 'uncertainty'])
    else:
        # Ensure date columns are datetime type
        combined_df['date'] = pd.to_datetime(combined_df['date'])

        # Create separate DataFrames for tech and sentiment with consistent date types
        # Only include 'close' price in tech DataFrame to avoid merge conflicts
        tech_df = combined_df[['ticker', 'date', 'above_sma', 'mom_rank', 'close']].copy()
        senti_df = combined_df[['ticker', 'date', 'sent_score']].copy()

        # The bayesian_score method converts tech dates to .dt.date, so we need both to be datetime initially
        tech_df['date'] = pd.to_datetime(tech_df['date'])
        senti_df['date'] = pd.to_datetime(senti_df['date'])

        # Create fundamentals DataFrame if data available
        fundamentals_df = None
        if fundamentals_data:
            fund_rows = []
            for ticker, features in fundamentals_data.items():
                if features:  # Only include tickers with actual data
                    fund_rows.append({
                        'ticker': ticker,
                        'fundamental_score': features.get('fundamental_score', 0.5)
                    })
            if fund_rows:
                fundamentals_df = pd.DataFrame(fund_rows)
                fundamentals_df['ticker'] = fundamentals_df['ticker'].astype(str)

        # Ensure string columns are string type (to avoid object issues)
        tech_df['ticker'] = tech_df['ticker'].astype(str)
        senti_df['ticker'] = senti_df['ticker'].astype(str)

        # Run adaptive engine using bayesian_score_adaptive method with proper format
        adaptive_recommendations = adaptive_engine.bayesian_score_adaptive(
            tech=tech_df,
            senti=senti_df,
            fundamentals=fundamentals_df,
            prices=data['prices'],
            vix_data=data.get('vix_data')
        )

        if adaptive_recommendations is None or adaptive_recommendations.empty:
            adaptive_recommendations = pd.DataFrame(columns=['ticker', 'expected_return', 'prob_positive', 'uncertainty'])
        else:
            # Ensure ticker column exists for downstream logic
            if 'ticker' not in adaptive_recommendations.columns:
                if adaptive_recommendations.index.name == 'ticker' or 'ticker' in str(adaptive_recommendations.index):
                    adaptive_recommendations = adaptive_recommendations.reset_index()
                    if 'index' in adaptive_recommendations.columns:
                        adaptive_recommendations = adaptive_recommendations.rename(columns={'index': 'ticker'})
                else:
                    # If no ticker info, create empty DataFrame with proper structure
                    print("   ⚠️ No ticker information in adaptive recommendations")
                    adaptive_recommendations = pd.DataFrame(columns=['ticker', 'expected_return', 'prob_positive', 'uncertainty'])

    # Convert adaptive results to expected returns for risk module
    expected_returns = {}
    for ticker in data['tickers']:
        if 'ticker' in adaptive_recommendations.columns:
            ticker_rec = adaptive_recommendations[adaptive_recommendations['ticker'] == ticker]
            if not ticker_rec.empty:
                expected_returns[ticker] = ticker_rec.iloc[0].get('expected_return', 0.0)
                continue
        expected_returns[ticker] = 0.0

    results['adaptive_bayesian'] = adaptive_recommendations

    # Step 5: Risk Management (using adaptive expected returns)
    print("   ⚠️ Running risk management...")
    risk_module = RiskManagementModule(module_config['risk_management'])
    risk_result = risk_module.process({
        'prices': data['prices'],
        'expected_returns': expected_returns
    })
    results['risk_management'] = risk_result

    # Step 6: Portfolio Management (using adaptive recommendations)
    print("   💼 Running portfolio management...")

    # Create candidate positions from adaptive results
    candidate_positions = []
    regime = regime_result.data.get('current_regime', 'neutral')

    if adaptive_recommendations is None or adaptive_recommendations.empty:
        candidates_df = pd.DataFrame(columns=['ticker', 'expected_return', 'confidence', 'regime', 'risk_score', 'decision', 'prob_positive'])
    else:
        for _, row in adaptive_recommendations.iterrows():
            ticker = row['ticker']
            expected_return = row.get('expected_return', 0.0)
            prob_positive = row.get('prob_positive', 0.5)
            uncertainty = row.get('uncertainty', 0.5)

            individual_risks = risk_result.data.get('individual_risks', {})
            risk_metrics = individual_risks.get(ticker, {})
            risk_score = uncertainty  # Use uncertainty as risk score

            decision = 'Buy' if expected_return > 0.001 else 'Sell' if expected_return < -0.001 else 'Hold'

            candidate_positions.append({
                'ticker': ticker,
                'expected_return': expected_return,
                'confidence': 1 - uncertainty,
                'regime': regime,
                'risk_score': risk_score,
                'decision': decision,
                'prob_positive': prob_positive
            })

        candidates_df = pd.DataFrame(candidate_positions)
    current_prices = data['prices'].groupby('ticker').last().reset_index()[['ticker', 'close']]

    # Load current portfolio state to avoid duplicate buying
    portfolio_tracker = PortfolioTracker(config['data']['cache_dir'] + "/portfolio")
    current_state = portfolio_tracker.get_portfolio_summary()
    current_holdings = {
        holding.ticker: {
            'shares': holding.shares,
            'avg_cost': holding.avg_cost,
            'market_value': holding.market_value
        } for holding in (portfolio_tracker.current_state.holdings if portfolio_tracker.current_state else [])
    }

    portfolio_module = PortfolioManagementModule(module_config['portfolio_management'])
    portfolio_result = portfolio_module.process({
        'candidate_positions': candidates_df,
        'current_prices': current_prices,
        'current_portfolio': current_holdings
    })
    results['portfolio_management'] = portfolio_result

    # Display execution summary
    total_time = sum(getattr(result, 'execution_time_ms', 0) or 0 for result in results.values() if hasattr(result, 'execution_time_ms'))
    avg_confidence = sum(result.confidence for result in results.values() if hasattr(result, 'confidence')) / len(results)

    print(f"✅ Adaptive pipeline executed successfully")
    print(f"   Total execution time: {total_time:.1f}ms")
    print(f"   Average confidence: {avg_confidence:.2f}")
    print(f"   Modules executed: {len(results)}")

    return results, portfolio_tracker


def create_adaptive_recommendations(results: Dict, data: Dict, adaptive_engine: AdaptiveBayesianEngine) -> pd.DataFrame:
    """Create final trading recommendations from adaptive modular pipeline results."""
    print("\n💼 Creating adaptive trading recommendations...")

    # Get the adaptive recommendations directly
    adaptive_recommendations = results.get('adaptive_bayesian')
    if adaptive_recommendations is None or adaptive_recommendations.empty:
        print("⚠️ No adaptive recommendations available")
        return pd.DataFrame()

    # Get portfolio allocation
    portfolio_result = results.get('portfolio_management')
    portfolio_allocation = portfolio_result.data.get('portfolio_allocation', {}) if portfolio_result else {}

    # Get current regime
    regime_result = results.get('regime_detection')
    current_regime = regime_result.data.get('current_regime', 'neutral') if regime_result else 'neutral'
    regime_confidence = regime_result.confidence if regime_result else 0.5

    # Get risk metrics
    risk_result = results.get('risk_management')
    individual_risks = risk_result.data.get('individual_risks', {}) if risk_result else {}

    # Create final recommendations DataFrame
    recommendations = []

    for _, row in adaptive_recommendations.iterrows():
        ticker = row['ticker']

        # Get risk metrics
        risk_metrics = individual_risks.get(ticker, {})
        tail_risk = risk_metrics.get('downside_tail_risk', 0.02)
        extreme_move_prob = risk_metrics.get('extreme_move_prob', 0.05)

        # Get portfolio weight
        portfolio_weight = portfolio_allocation.get(ticker, 0.0)

        # Determine final decision based on portfolio weight and expected return
        expected_return = row.get('expected_return', 0.0)
        if portfolio_weight > 0.01:  # >1% allocation
            decision = 'Buy'
        elif expected_return < -0.005:  # <-0.5% expected return
            decision = 'Sell'
        else:
            decision = 'Hold'

        # Get current price
        ticker_prices = data['prices'][data['prices']['ticker'] == ticker]
        current_price = ticker_prices.iloc[-1]['close'] if not ticker_prices.empty else 100.0

        recommendations.append({
            'date': pd.Timestamp.now().date(),
            'ticker': ticker,
            'close': current_price,
            'decision': decision,
            'expected_return': expected_return,
            'prob_positive': row.get('prob_positive', 0.5),
            'decision_confidence': row.get('confidence', 0.5),
            'uncertainty': row.get('uncertainty', 0.5),
            'trend_weight': row.get('trend_weight', 0),
            'momentum_weight': row.get('momentum_weight', 0),
            'sentiment_weight': row.get('sentiment_weight', 0),
            'fundamentals_weight': row.get('fundamentals_weight', 0),
            'regime': current_regime,
            'regime_confidence': regime_confidence,
            'tail_risk': tail_risk,
            'extreme_move_prob': extreme_move_prob,
            'portfolio_weight': portfolio_weight,
            'portfolio_adjusted': True,
            'adaptive_calibrated': True  # Flag to indicate adaptive calibration
        })

    recommendations_df = pd.DataFrame(recommendations)

    # Filter to only actionable recommendations
    actionable = recommendations_df[recommendations_df['decision'].isin(['Buy', 'Sell'])]
    print(f"✅ Generated {len(actionable)} actionable adaptive recommendations from {len(recommendations_df)} analyzed tickers")

    return recommendations_df


def run_portfolio_management(recommendations: pd.DataFrame, config: Dict,
                            portfolio_tracker: PortfolioTracker = None) -> List[Dict]:
    """Execute portfolio management with adaptive recommendations."""
    print("\n💰 Portfolio management with adaptive recommendations...")

    # Use provided portfolio tracker or create new one
    if portfolio_tracker is None:
        from quant.portfolio.state import PortfolioTracker
        portfolio_tracker = PortfolioTracker(config['data']['cache_dir'] + "/portfolio")

    # Handle empty recommendations
    if recommendations.empty or 'ticker' not in recommendations.columns or 'close' not in recommendations.columns:
        print("⚠️ No valid recommendations for portfolio management")
        return []

    # Update portfolio with latest prices
    current_prices = recommendations[['ticker', 'close']].drop_duplicates()
    current_state = portfolio_tracker.update_portfolio_state(current_prices)

    # Execute trades based on adaptive recommendations
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


def save_adaptive_logs(recommendations: pd.DataFrame, executed_trades: List[Dict],
                      cache_dir: str, engine: AdaptiveBayesianEngine):
    """Save adaptive recommendations and logs for audit trail."""
    print("\n📝 Saving adaptive recommendation logs...")

    log_dir = Path(cache_dir) / "adaptive_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    run_date = pd.to_datetime(recommendations['date']).max()
    run_date_str = pd.to_datetime(run_date).date().isoformat()
    timestamp = datetime.utcnow().isoformat()

    # Save recommendations with adaptive metadata
    recommendations_to_log = recommendations.copy()
    recommendations_to_log['logged_at_utc'] = timestamp
    recommendations_to_log['engine_type'] = 'adaptive'

    rec_path = log_dir / f"adaptive_recommendations_{run_date_str}.parquet"
    recommendations_to_log.to_parquet(rec_path, index=False)
    print(f"📝 Saved adaptive recommendations to {rec_path}")

    # Save parameter diagnostics
    diagnostics = engine.get_parameter_diagnostics()
    if not diagnostics.empty:
        diag_path = log_dir / f"parameter_diagnostics_{run_date_str}.parquet"
        diagnostics.to_parquet(diag_path, index=False)
        print(f"📝 Saved parameter diagnostics to {diag_path}")

    # Save learning summary
    learning_summary = engine.get_learning_summary()
    summary_path = log_dir / f"learning_summary_{run_date_str}.json"
    with open(summary_path, 'w') as f:
        import json
        json.dump({
            'run_date': run_date_str,
            'logged_at_utc': timestamp,
            'learning_summary': learning_summary,
            'trades': executed_trades
        }, f, indent=2)

    if executed_trades:
        print(f"🧾 Logged {len(executed_trades)} adaptive trades")
    else:
        print("ℹ️ No adaptive trades executed today")


def main():
    """
    Main execution with adaptive modular pipeline.
    """
    print("=== ROI Adaptive Modular Trading System ===")

    # Load configuration
    config = load_configuration()
    print(f"Loaded configuration for {len(config['universe']['tickers'])} tickers")

    # Prepare historical data for calibration
    prices_df, sentiment_df, technical_df, returns_df = prepare_historical_data(config)

    # Calibrate adaptive engine
    engine = calibrate_adaptive_engine(config, prices_df, sentiment_df, technical_df, returns_df)

    # Show parameter diagnostics
    print("\n=== Parameter Estimation Results ===")
    diagnostics = engine.get_parameter_diagnostics()
    if not diagnostics.empty:
        print("Top parameter changes from defaults:")
        for _, row in diagnostics.head(10).iterrows():
            default_val = row['default_value']
            estimated_val = row['estimated_value']
            if abs(estimated_val - default_val) > 0.01:  # Show changes > 1%
                change_pct = ((estimated_val - default_val) / default_val) * 100 if default_val != 0 else 0
                print(f"  {row['parameter_name']}: {default_val:.3f} → {estimated_val:.3f} ({change_pct:+.1f}%)")
    else:
        print("No parameter diagnostics available")

    # Show learning summary
    learning_summary = engine.get_learning_summary()
    if learning_summary.get('status') == 'learning_complete':
        print(f"\nLearning Summary:")
        print(f"  Total parameters estimated: {learning_summary['parameter_changes']['total_parameters']}")
        print(f"  Significant changes (>10%): {learning_summary['parameter_changes']['significant_changes']}")
        print(f"  Average change: {learning_summary['parameter_changes']['avg_change_percent']:.1f}%")

    # Create adaptive modular pipeline
    pipeline = create_adaptive_modular_pipeline()

    # Fetch current market data
    data = fetch_current_data(config)

    # Execute adaptive modular pipeline
    results, portfolio_tracker = execute_adaptive_modular_pipeline(pipeline, data, engine, config)

    # Create adaptive trading recommendations
    recommendations = create_adaptive_recommendations(results, data, engine)

    # Execute portfolio management (using the same portfolio tracker to avoid duplicate buying)
    executed_trades = run_portfolio_management(recommendations, config, portfolio_tracker)

    # Save adaptive logs
    save_adaptive_logs(recommendations, executed_trades, config['data']['cache_dir'], engine)

    # Generate daily report with portfolio performance
    print("\n📊 Generating adaptive daily report...")
    report_dir = config.get('run', {}).get('outdir', 'reports')

    # Get current portfolio summary for the report
    portfolio_summary = portfolio_tracker.get_portfolio_summary() if portfolio_tracker else {}

    save_daily_markdown(recommendations, report_dir, portfolio_summary, engine, data.get('macro_data', {}))
    print("✅ Adaptive daily report saved with portfolio performance")

    # Show summary
    buy_count = len(recommendations[recommendations['decision'] == 'Buy'])
    sell_count = len(recommendations[recommendations['decision'] == 'Sell'])

    print(f"\n📈 Adaptive Summary:")
    print(f"   Buy recommendations: {buy_count}")
    print(f"   Sell recommendations: {sell_count}")
    print(f"   Executed trades: {len(executed_trades)}")
    print(f"   Total positions analyzed: {len(recommendations)}")
    print(f"   Adaptive engine status: {learning_summary.get('status', 'unknown')}")

    print("\n🎉 Adaptive modular trading system execution completed!")


if __name__ == "__main__":
    main()
