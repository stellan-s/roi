"""
Backtesting Runner - Execute historical performance analysis of the adaptive system.

This script runs backtests comparing the adaptive learning system against
static hardcoded parameters to validate the effectiveness of data-driven optimization.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import yaml

from quant.adaptive_main import load_configuration, prepare_historical_data, calibrate_adaptive_engine
from quant.bayesian.integration import BayesianPolicyEngine
from quant.backtesting.framework import BacktestEngine, BacktestPeriod, BacktestResults
from quant.backtesting.attribution import PerformanceAttributor
from quant.data_layer.prices import fetch_prices
from quant.data_layer.news import fetch_news
from quant.features.technical import compute_technical_features
from quant.features.sentiment import naive_sentiment
from quant.portfolio.rules import PortfolioManager
from quant.portfolio.state import PortfolioTracker

def create_adaptive_engine(config, prices_df, sentiment_df, technical_df, returns_df):
    """Factory function to create adaptive engine."""
    from quant.bayesian.adaptive_integration import AdaptiveBayesianEngine

    engine = AdaptiveBayesianEngine(config)
    engine.calibrate_parameters(prices_df, sentiment_df, technical_df, returns_df=returns_df)
    return engine

def create_static_engine(config):
    """Factory function to create static (non-adaptive) engine."""
    # Use the regular BayesianPolicyEngine with hardcoded parameters
    return BayesianPolicyEngine(config)

def run_backtest_period(engine, config, start_date, end_date):
    """Run backtest for a specific period with given engine."""

    # Get universe
    universe = config['universe']['tickers']

    # Fetch data for the period (add buffer for technical indicators)
    period_start = pd.Timestamp(start_date) - timedelta(days=300)  # Buffer for indicators

    prices = fetch_prices(
        tickers=universe,
        cache_dir=config['data']['cache_dir'],
        lookback_days=1000  # Enough for the full period
    )

    # Filter to backtest period
    prices_period = prices[
        (prices['date'] >= period_start) &
        (prices['date'] <= pd.Timestamp(end_date))
    ].copy()

    # Get news data
    news = fetch_news(
        feed_urls=config['signals']['news_feed_urls'],
        cache_dir=config['data']['cache_dir']
    )

    # Compute features
    tech = compute_technical_features(
        prices_period,
        config['signals']['sma_long'],
        config['signals']['momentum_window']
    )

    # Filter technical data to actual backtest period
    tech_backtest = tech[
        (tech['date'] >= pd.Timestamp(start_date)) &
        (tech['date'] <= pd.Timestamp(end_date))
    ].copy()

    if len(tech_backtest) == 0:
        raise ValueError(f"No technical data available for backtest period {start_date} to {end_date}")

    senti = naive_sentiment(news, universe)

    print(f"Backtest data: {len(tech_backtest)} tech records from {tech_backtest['date'].min()} to {tech_backtest['date'].max()}")

    # Generate recommendations for each day
    daily_results = []
    portfolio_tracker = PortfolioTracker(config['data']['cache_dir'] + "/backtest_portfolio")
    portfolio_mgr = PortfolioManager(config)

    # Get unique dates and sort them
    test_dates = sorted(tech_backtest['date'].unique())

    for i, current_date in enumerate(test_dates):
        if i % 10 == 0:
            print(f"Processing date {i+1}/{len(test_dates)}: {current_date.date()}")

        # Get data up to current date for decision making
        tech_current = tech[tech['date'] <= current_date]
        if len(tech_current) == 0:
            continue

        # Get unique latest data for each ticker for the current date
        # This fixes the duplicate recommendation issue where tail() was creating 235 duplicates
        latest_tech = tech_current.groupby('ticker').tail(1)

        # Generate recommendations using the engine
        if hasattr(engine, 'bayesian_score_adaptive'):
            recommendations = engine.bayesian_score_adaptive(
                latest_tech,  # Use deduplicated latest data per ticker
                senti,
                prices_period[prices_period['date'] <= current_date]
            )
        else:
            recommendations = engine.bayesian_score(
                latest_tech,  # Use deduplicated latest data per ticker
                senti,
                prices_period[prices_period['date'] <= current_date]
            )

        if len(recommendations) == 0:
            continue

        # Apply portfolio rules
        final_decisions = portfolio_mgr.apply_portfolio_rules(recommendations)

        # Get current prices
        current_prices = prices_period[prices_period['date'] == current_date][['ticker', 'close']]
        if len(current_prices) == 0:
            continue

        # Update portfolio
        portfolio_tracker.update_portfolio_state(current_prices, as_of_date=str(current_date.date()))
        executed_trades = portfolio_tracker.execute_trades(final_decisions, current_prices)

        # Calculate daily return
        portfolio_summary = portfolio_tracker.get_portfolio_summary()

        daily_results.append({
            'date': current_date,
            'portfolio_value': portfolio_summary.get('total_value', 100000),
            'daily_return': portfolio_summary.get('portfolio_return', 0.0),
            'positions': portfolio_summary.get('positions', 0),
            'trades': len(executed_trades)
        })

    if len(daily_results) == 0:
        raise ValueError("No backtest results generated")

    # Convert to DataFrame
    results_df = pd.DataFrame(daily_results)
    results_df['date'] = pd.to_datetime(results_df['date'])
    results_df = results_df.sort_values('date')

    # Calculate performance metrics
    daily_returns = results_df['daily_return'].fillna(0)
    portfolio_values = results_df['portfolio_value']

    total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
    annualized_return = (1 + total_return) ** (252 / len(daily_returns)) - 1
    volatility = daily_returns.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0

    # Calculate max drawdown
    running_max = portfolio_values.expanding().max()
    drawdowns = (portfolio_values - running_max) / running_max
    max_drawdown = drawdowns.min()

    # Trading metrics
    winning_days = (daily_returns > 0).sum()
    losing_days = (daily_returns < 0).sum()
    win_rate = winning_days / (winning_days + losing_days) if (winning_days + losing_days) > 0 else 0

    avg_win = daily_returns[daily_returns > 0].mean() if winning_days > 0 else 0
    avg_loss = daily_returns[daily_returns < 0].mean() if losing_days > 0 else 0

    # VaR 95%
    var_95 = daily_returns.quantile(0.05)

    return BacktestResults(
        total_return=total_return,
        annualized_return=annualized_return,
        volatility=volatility,
        sharpe_ratio=sharpe_ratio,
        max_drawdown=max_drawdown,
        total_trades=results_df['trades'].sum(),
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        var_95=var_95,
        tail_risk_accuracy=0.0,  # TODO: Implement tail risk accuracy calculation
        daily_returns=daily_returns,
        daily_positions=results_df[['date', 'positions']],
        daily_pnl=daily_returns,
        backtest_id=f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        engine_type="adaptive" if hasattr(engine, 'bayesian_score_adaptive') else "static",
        period=BacktestPeriod(
            start_date=pd.Timestamp(start_date),
            end_date=pd.Timestamp(end_date),
            train_start=pd.Timestamp(start_date),
            train_end=pd.Timestamp(end_date),
            test_start=pd.Timestamp(start_date),
            test_end=pd.Timestamp(end_date)
        ),
        config=config
    )

def main():
    """Run comprehensive backtesting analysis."""
    print("=== ROI Adaptive Trading System - Backtesting ===")

    # Load configuration
    config = load_configuration()
    print(f"Loaded configuration for {len(config['universe']['tickers'])} tickers")

    # Define backtest period (last 6 months for quick test)
    today = datetime.now().date()
    end_date = (today - timedelta(days=1)).isoformat()  # Yesterday to avoid incomplete data
    start_date = (today - timedelta(days=180)).isoformat()  # 6 months back

    print(f"Backtesting period: {start_date} to {end_date}")

    # Prepare historical data for adaptive engine calibration
    print("\nPreparing historical data for adaptive engine...")
    prices_hist, sentiment_hist, tech_hist, returns_hist = prepare_historical_data(config)

    print(f"Historical data: {len(prices_hist)} price records for calibration")

    # Test 1: Adaptive Engine Backtest
    print("\n=== Running Adaptive Engine Backtest ===")
    adaptive_engine = create_adaptive_engine(config, prices_hist, sentiment_hist, tech_hist, returns_hist)

    try:
        adaptive_results = run_backtest_period(adaptive_engine, config, start_date, end_date)

        print(f"\n=== Adaptive Engine Results ===")
        print(f"Total Return: {adaptive_results.total_return:.2%}")
        print(f"Annualized Return: {adaptive_results.annualized_return:.2%}")
        print(f"Volatility: {adaptive_results.volatility:.2%}")
        print(f"Sharpe Ratio: {adaptive_results.sharpe_ratio:.2f}")
        print(f"Max Drawdown: {adaptive_results.max_drawdown:.2%}")
        print(f"Win Rate: {adaptive_results.win_rate:.2%}")
        print(f"Total Trades: {adaptive_results.total_trades}")

    except Exception as e:
        print(f"Adaptive backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test 2: Static Engine Backtest (for comparison)
    print("\n=== Running Static Engine Backtest ===")
    static_engine = create_static_engine(config)

    try:
        static_results = run_backtest_period(static_engine, config, start_date, end_date)

        print(f"\n=== Static Engine Results ===")
        print(f"Total Return: {static_results.total_return:.2%}")
        print(f"Annualized Return: {static_results.annualized_return:.2%}")
        print(f"Volatility: {static_results.volatility:.2%}")
        print(f"Sharpe Ratio: {static_results.sharpe_ratio:.2f}")
        print(f"Max Drawdown: {static_results.max_drawdown:.2%}")
        print(f"Win Rate: {static_results.win_rate:.2%}")
        print(f"Total Trades: {static_results.total_trades}")

        # Compare results
        print(f"\n=== Comparison (Adaptive vs Static) ===")
        return_improvement = adaptive_results.total_return - static_results.total_return
        sharpe_improvement = adaptive_results.sharpe_ratio - static_results.sharpe_ratio
        drawdown_improvement = static_results.max_drawdown - adaptive_results.max_drawdown  # Positive = better

        print(f"Return Improvement: {return_improvement:.2%}")
        print(f"Sharpe Improvement: {sharpe_improvement:.2f}")
        print(f"Drawdown Improvement: {drawdown_improvement:.2%}")
        print(f"Win Rate Improvement: {(adaptive_results.win_rate - static_results.win_rate):.2%}")

        # Save results
        results_dir = Path("backtesting_results")
        results_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save summary
        summary = {
            'backtest_date': timestamp,
            'period': {'start': start_date, 'end': end_date},
            'adaptive_results': {
                'total_return': adaptive_results.total_return,
                'annualized_return': adaptive_results.annualized_return,
                'sharpe_ratio': adaptive_results.sharpe_ratio,
                'max_drawdown': adaptive_results.max_drawdown,
                'win_rate': adaptive_results.win_rate
            },
            'static_results': {
                'total_return': static_results.total_return,
                'annualized_return': static_results.annualized_return,
                'sharpe_ratio': static_results.sharpe_ratio,
                'max_drawdown': static_results.max_drawdown,
                'win_rate': static_results.win_rate
            },
            'improvements': {
                'return_improvement': return_improvement,
                'sharpe_improvement': sharpe_improvement,
                'drawdown_improvement': drawdown_improvement
            }
        }

        import json
        with open(results_dir / f"backtest_summary_{timestamp}.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"\nResults saved to: {results_dir}/backtest_summary_{timestamp}.json")

    except Exception as e:
        print(f"Static backtest failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()