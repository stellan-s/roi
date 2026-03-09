"""
Backtesting Runner - Execute historical performance analysis of the adaptive system.

This script runs backtests comparing the adaptive learning system against
static hardcoded parameters to validate the effectiveness of data-driven optimization.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import argparse

from quant.adaptive_main import load_configuration, prepare_historical_data, calibrate_adaptive_engine
from quant.bayesian.integration import BayesianPolicyEngine
from quant.backtesting.framework import BacktestPeriod, BacktestResults
from quant.data_layer.prices import fetch_prices
from quant.data_layer.news import fetch_news
from quant.data_layer.macro import fetch_vix
from quant.data_layer.validation import normalize_news_schema, normalize_prices_schema
from quant.engine import DayRunContext, run_engine_day
from quant.observability import get_logger, log_event, new_run_id
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

def run_backtest_period(engine, config, start_date, end_date, engine_type="unknown"):
    """Run backtest for a specific period with given engine."""
    logger = get_logger("quant.backtest")
    run_id = new_run_id("backtest")
    log_event(logger, "backtest_start", run_id=run_id, engine_type=engine_type, start_date=start_date, end_date=end_date)

    universe = config["universe"]["tickers"]
    period_start = pd.Timestamp(start_date) - timedelta(days=300)
    period_end = pd.Timestamp(end_date)

    prices = normalize_prices_schema(
        fetch_prices(
            tickers=universe,
            cache_dir=config["data"]["cache_dir"],
            lookback_days=1000,
        )
    )
    prices_period = prices[(prices["date"] >= period_start) & (prices["date"] <= period_end)].copy()
    if prices_period.empty:
        raise ValueError("No price data available for requested backtest period")

    news = normalize_news_schema(
        fetch_news(
            feed_urls=config["signals"]["news_feed_urls"],
            cache_dir=config["data"]["cache_dir"],
        )
    )

    vix_data = fetch_vix(
        cache_dir=config["data"]["cache_dir"],
        lookback_days=1000,
    )
    if vix_data is not None and not vix_data.empty and "date" in vix_data.columns:
        vix_data = vix_data.copy()
        vix_data["date"] = pd.to_datetime(vix_data["date"], errors="coerce").dt.normalize()
        vix_data = vix_data[vix_data["date"].notna()]

    # Backtest dates are derived from available close bars in requested period.
    raw_dates = prices_period[(prices_period["date"] >= pd.Timestamp(start_date)) & (prices_period["date"] <= period_end)]["date"]
    test_dates = sorted(pd.to_datetime(raw_dates, errors="coerce").dropna().dt.normalize().unique())
    if len(test_dates) < 2:
        raise ValueError("Need at least two trading dates to support next-bar execution")

    daily_results = []
    portfolio_dir = config["data"]["cache_dir"] + f"/backtest_portfolio_{engine_type}"
    from pathlib import Path
    import shutil

    portfolio_path = Path(portfolio_dir)
    if portfolio_path.exists():
        shutil.rmtree(portfolio_path)

    portfolio_tracker = PortfolioTracker(portfolio_dir)
    portfolio_mgr = PortfolioManager(config)

    exec_cfg = config.get("backtesting", {}).get("execution", {})
    slippage_bps = float(exec_cfg.get("slippage_bps", 3.0))
    fee_bps = float(exec_cfg.get("fee_bps", config.get("policy", {}).get("trade_cost_bps", 3)))
    max_drawdown_limit = float(config.get("risk_controls", {}).get("max_drawdown", 0.20))
    max_daily_loss_limit = float(config.get("risk_controls", {}).get("max_daily_loss", 0.05))

    peak_value = 100000.0
    prev_daily_return = 0.0

    # Use decision date -> execution date (next bar) to avoid look-ahead execution.
    for i in range(len(test_dates) - 1):
        decision_date = pd.Timestamp(test_dates[i]).normalize()
        execution_date = pd.Timestamp(test_dates[i + 1]).normalize()

        if i % 10 == 0:
            print(f"Processing day {i+1}/{len(test_dates)-1}: decision {decision_date.date()} -> execution {execution_date.date()}")

        context = DayRunContext(
            as_of=decision_date,
            tickers=universe,
            prices=prices_period,
            news=news,
            vix_data=vix_data,
        )
        day_result = run_engine_day(engine, context, config)
        recommendations = day_result.recommendations
        if recommendations.empty:
            continue

        # Apply portfolio rules with available history up to decision_date only.
        price_history = prices_period[prices_period["date"] <= decision_date].pivot(
            index="date", columns="ticker", values="close"
        ).dropna(axis=1, how="all")
        final_decisions = portfolio_mgr.apply_portfolio_rules(recommendations, price_history)

        # Guardrails: halt new risk if drawdown/daily loss limits are breached.
        current_summary = portfolio_tracker.get_portfolio_summary()
        current_value = float(current_summary.get("total_value", 100000.0))
        peak_value = max(peak_value, current_value)
        current_drawdown = 0.0 if peak_value <= 0 else (peak_value - current_value) / peak_value

        kill_switch = current_drawdown >= max_drawdown_limit or abs(prev_daily_return) >= max_daily_loss_limit
        if kill_switch:
            final_decisions = final_decisions.copy()
            final_decisions.loc[final_decisions["decision"] == "Buy", "decision"] = "Hold"
            final_decisions["portfolio_weight"] = 0.0

        # Execute on next bar prices.
        execution_prices = prices_period[prices_period["date"] == execution_date][["ticker", "close"]]
        if execution_prices.empty:
            continue

        portfolio_tracker.update_portfolio_state(execution_prices, as_of_date=str(execution_date.date()))
        executed_trades = portfolio_tracker.execute_trades(
            final_decisions,
            execution_prices,
            slippage_bps=slippage_bps,
            fee_bps=fee_bps,
        )

        portfolio_summary = portfolio_tracker.get_portfolio_summary()
        prev_daily_return = float(portfolio_summary.get("portfolio_return", 0.0))

        daily_results.append(
            {
                "date": execution_date,
                "decision_date": decision_date,
                "portfolio_value": portfolio_summary.get("total_value", 100000),
                "daily_return": portfolio_summary.get("portfolio_return", 0.0),
                "positions": portfolio_summary.get("positions", 0),
                "trades": len(executed_trades),
                "kill_switch": kill_switch,
            }
        )

    if len(daily_results) == 0:
        raise ValueError("No backtest results generated")

    # Convert to DataFrame
    results_df = pd.DataFrame(daily_results)
    results_df['date'] = pd.to_datetime(results_df['date'])
    results_df = results_df.sort_values('date')

    # Calculate performance metrics
    daily_returns = results_df['daily_return'].fillna(0)
    portfolio_values = results_df['portfolio_value']

    # DEBUG: Add validation checks for impossible results
    print(f"DEBUG: Portfolio values range: {portfolio_values.min():.2f} to {portfolio_values.max():.2f}")
    print(f"DEBUG: Daily returns range: {daily_returns.min():.6f} to {daily_returns.max():.6f}")
    print(f"DEBUG: Daily returns std: {daily_returns.std():.6f}")
    print(f"DEBUG: Non-zero daily returns: {(daily_returns != 0).sum()}/{len(daily_returns)}")

    total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1

    # Fix: Use actual trading days, not length of returns array
    start_date = pd.to_datetime(results_df['date'].iloc[0])
    end_date = pd.to_datetime(results_df['date'].iloc[-1])
    actual_days = (end_date - start_date).days

    annualized_return = (1 + total_return) ** (252 / actual_days) - 1
    volatility = daily_returns.std() * np.sqrt(252)
    risk_free_rate = 0.02  # 2% annual risk-free rate

    # DEBUG: Show annualization calculation
    print(f"DEBUG: Total return: {total_return:.6f} ({total_return*100:.3f}%)")
    print(f"DEBUG: Actual days: {actual_days}, Daily returns length: {len(daily_returns)}")
    print(f"DEBUG: Annualization factor: {252/actual_days:.3f}")
    print(f"DEBUG: Annualized return: {annualized_return:.6f} ({annualized_return*100:.2f}%)")

    # CRITICAL FIX: Add minimum volatility threshold to prevent division by near-zero
    min_volatility = 0.01  # 1% minimum annual volatility (extremely conservative)
    volatility = max(volatility, min_volatility)

    sharpe_ratio = (annualized_return - risk_free_rate) / volatility

    print(f"DEBUG: Final volatility: {volatility:.6f}, Sharpe ratio: {sharpe_ratio:.2f}")

    # VALIDATION: Flag suspicious results
    if sharpe_ratio > 3.0:
        print(f"⚠️  WARNING: Suspiciously high Sharpe ratio ({sharpe_ratio:.2f}) - possible calculation error")
    if volatility < 0.05:
        print(f"⚠️  WARNING: Suspiciously low volatility ({volatility:.2f}) - possible data issue")

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

    results = BacktestResults(
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
        engine_type=engine_type,
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
    log_event(
        logger,
        "backtest_complete",
        run_id=run_id,
        engine_type=engine_type,
        total_return=results.total_return,
        annualized_return=results.annualized_return,
        sharpe_ratio=results.sharpe_ratio,
        max_drawdown=results.max_drawdown,
        total_trades=results.total_trades,
    )
    return results

def main():
    """Run comprehensive backtesting analysis."""
    parser = argparse.ArgumentParser(description='ROI Backtesting Framework')
    parser.add_argument('--days', type=int, default=None,
                       help='Number of days to backtest (default: from config, usually 180 = 6 months)')
    parser.add_argument('--start-date', type=str,
                       help='Start date for backtest (YYYY-MM-DD format)')
    parser.add_argument('--end-date', type=str,
                       help='End date for backtest (YYYY-MM-DD format)')
    parser.add_argument('--comparison', action='store_true',
                       help='Run comparison between adaptive and static engines')

    args = parser.parse_args()

    print("=== ROI Adaptive Trading System - Backtesting ===")

    # Load configuration
    config = load_configuration()
    print(f"Loaded configuration for {len(config['universe']['tickers'])} tickers")

    # Define backtest period
    today = datetime.now().date()

    if args.start_date and args.end_date:
        # Use specified date range
        start_date = args.start_date
        end_date = args.end_date
        print(f"Using specified period: {start_date} to {end_date}")
    elif args.start_date:
        # Start date specified, use today as end
        start_date = args.start_date
        end_date = (today - timedelta(days=1)).isoformat()
        print(f"Using specified start date: {start_date} to {end_date}")
    else:
        # Use days parameter or config default to go back from today
        days = args.days if args.days is not None else config.get('backtesting', {}).get('default_period_days', 180)
        end_date = (today - timedelta(days=1)).isoformat()  # Yesterday to avoid incomplete data
        start_date = (today - timedelta(days=days)).isoformat()
        print(f"Using {days} day period: {start_date} to {end_date}")

    print(f"Backtesting period: {start_date} to {end_date}")

    # Prepare historical data for adaptive engine calibration
    print("\nPreparing historical data for adaptive engine...")
    prices_hist, sentiment_hist, tech_hist, returns_hist = prepare_historical_data(config)

    print(f"Historical data: {len(prices_hist)} price records for calibration")

    # Test 1: Adaptive Engine Backtest
    print("\n=== Running Adaptive Engine Backtest ===")
    adaptive_engine = create_adaptive_engine(config, prices_hist, sentiment_hist, tech_hist, returns_hist)

    try:
        adaptive_results = run_backtest_period(adaptive_engine, config, start_date, end_date, "adaptive")

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
        static_results = run_backtest_period(static_engine, config, start_date, end_date, "static")

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

        # CRITICAL ANALYSIS: Trade frequency comparison
        print(f"\n=== Trading Activity Analysis ===")
        print(f"Adaptive Total Trades: {adaptive_results.total_trades}")
        print(f"Static Total Trades: {static_results.total_trades}")
        print(f"Trade Frequency Difference: {adaptive_results.total_trades - static_results.total_trades} trades")

        # Calculate trade efficiency (return per trade)
        adaptive_return_per_trade = adaptive_results.total_return / adaptive_results.total_trades if adaptive_results.total_trades > 0 else 0
        static_return_per_trade = static_results.total_return / static_results.total_trades if static_results.total_trades > 0 else 0

        print(f"Adaptive Return per Trade: {adaptive_return_per_trade:.4f}")
        print(f"Static Return per Trade: {static_return_per_trade:.4f}")

        if adaptive_results.total_trades > static_results.total_trades:
            print("🔍 Adaptive trades more frequently - possibly overtrading")
        elif static_results.total_trades > adaptive_results.total_trades:
            print("🔍 Static trades more frequently - possibly more decisive")
        else:
            print("🔍 Similar trading frequency - difference is in trade quality")

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
                'win_rate': adaptive_results.win_rate,
                'total_trades': adaptive_results.total_trades
            },
            'static_results': {
                'total_return': static_results.total_return,
                'annualized_return': static_results.annualized_return,
                'sharpe_ratio': static_results.sharpe_ratio,
                'max_drawdown': static_results.max_drawdown,
                'win_rate': static_results.win_rate,
                'total_trades': static_results.total_trades
            },
            'improvements': {
                'return_improvement': return_improvement,
                'sharpe_improvement': sharpe_improvement,
                'drawdown_improvement': drawdown_improvement
            },
            'trading_analysis': {
                'adaptive_trades': adaptive_results.total_trades,
                'static_trades': static_results.total_trades,
                'trade_frequency_difference': adaptive_results.total_trades - static_results.total_trades,
                'adaptive_return_per_trade': adaptive_return_per_trade,
                'static_return_per_trade': static_return_per_trade
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
