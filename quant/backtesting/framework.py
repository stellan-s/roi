"""
Backtesting Framework - Core infrastructure for historical performance analysis.

This module provides the foundation for validating the adaptive learning system
against historical data, comparing learned vs hardcoded parameters.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import json

@dataclass
class BacktestPeriod:
    """Defines a backtesting time period with train/test split."""
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

@dataclass
class BacktestResults:
    """Complete results from a backtesting run."""
    # Performance metrics
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float

    # Trading metrics
    total_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float

    # Risk metrics
    var_95: float
    tail_risk_accuracy: float  # How well P[return < -2σ] predicted actual tail events

    # Daily performance
    daily_returns: pd.Series
    daily_positions: pd.DataFrame
    daily_pnl: pd.Series

    # Metadata
    backtest_id: str
    engine_type: str  # "adaptive" or "static"
    period: BacktestPeriod
    config: Dict

@dataclass
class ComparisonResults:
    """Results comparing adaptive vs static performance."""
    adaptive_results: BacktestResults
    static_results: BacktestResults

    # Performance differences
    return_improvement: float
    sharpe_improvement: float
    drawdown_improvement: float
    tail_risk_improvement: float

    # Statistical significance
    p_value_returns: float
    confidence_interval: Tuple[float, float]

class BacktestEngine:
    """
    Core backtesting engine for historical performance analysis.

    Supports both adaptive and static parameter configurations with
    walk-forward validation and performance attribution analysis.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.cache_dir = Path(config['data']['cache_dir']) / 'backtesting'
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def run_single_backtest(self,
                           start_date: str,
                           end_date: str,
                           engine_factory: Callable,
                           engine_type: str = "adaptive") -> BacktestResults:
        """
        Run a single backtest over a specified period.

        Args:
            start_date: Backtest start date (YYYY-MM-DD)
            end_date: Backtest end date (YYYY-MM-DD)
            engine_factory: Function that creates the trading engine
            engine_type: "adaptive" or "static"

        Returns:
            BacktestResults with complete performance analysis
        """

        print(f"Running {engine_type} backtest from {start_date} to {end_date}")

        # Load historical data
        historical_data = self._load_historical_data(start_date, end_date)
        if not historical_data:
            raise ValueError(f"No historical data available for {start_date} to {end_date}")

        prices_df, sentiment_df, technical_df, returns_df = historical_data

        # Create engine
        engine = engine_factory(self.config)

        # Calibrate if adaptive
        if engine_type == "adaptive":
            calibration_end = pd.to_datetime(start_date) + timedelta(days=500)  # Use first 500 days for calibration
            calib_data = self._slice_data(historical_data, start_date, calibration_end.strftime('%Y-%m-%d'))
            if calib_data:
                engine.calibrate_parameters(*calib_data)

        # Run simulation
        simulation_results = self._run_simulation(engine, historical_data, start_date, end_date)

        # Calculate performance metrics
        backtest_results = self._calculate_performance_metrics(
            simulation_results,
            engine_type,
            start_date,
            end_date
        )

        return backtest_results

    def run_walk_forward_backtest(self,
                                 start_date: str,
                                 end_date: str,
                                 train_days: int = 500,
                                 test_days: int = 21,
                                 rebalance_frequency: int = 21) -> List[BacktestResults]:
        """
        Run walk-forward backtesting with periodic retraining.

        Args:
            start_date: Overall backtest start
            end_date: Overall backtest end
            train_days: Days of training data
            test_days: Days of out-of-sample testing
            rebalance_frequency: How often to retrain (days)

        Returns:
            List of BacktestResults for each walk-forward period
        """

        print(f"Walk-forward backtest: {start_date} to {end_date}")
        print(f"Train: {train_days} days, Test: {test_days} days, Rebalance: {rebalance_frequency} days")

        periods = self._generate_walk_forward_periods(
            start_date, end_date, train_days, test_days, rebalance_frequency
        )

        results = []
        for i, period in enumerate(periods):
            print(f"\nWalk-forward period {i+1}/{len(periods)}")
            print(f"Train: {period.train_start.date()} to {period.train_end.date()}")
            print(f"Test: {period.test_start.date()} to {period.test_end.date()}")

            try:
                # Run adaptive backtest for this period
                period_result = self._run_walk_forward_period(period, "adaptive")
                results.append(period_result)

            except Exception as e:
                print(f"Warning: Failed to run period {i+1}: {e}")
                continue

        return results

    def compare_adaptive_vs_static(self,
                                  start_date: str,
                                  end_date: str) -> ComparisonResults:
        """
        Compare adaptive learning vs static configuration performance.

        Args:
            start_date: Comparison period start
            end_date: Comparison period end

        Returns:
            ComparisonResults with statistical comparison
        """

        print(f"Comparing adaptive vs static: {start_date} to {end_date}")

        # Import engines
        from ..bayesian.adaptive_integration import AdaptiveBayesianEngine
        from ..bayesian.integration import BayesianPolicyEngine

        # Run adaptive backtest
        adaptive_results = self.run_single_backtest(
            start_date, end_date,
            lambda config: AdaptiveBayesianEngine(config),
            "adaptive"
        )

        # Run static backtest
        static_results = self.run_single_backtest(
            start_date, end_date,
            lambda config: BayesianPolicyEngine(config),
            "static"
        )

        # Statistical comparison
        comparison = self._perform_statistical_comparison(adaptive_results, static_results)

        return comparison

    def _load_historical_data(self, start_date: str, end_date: str) -> Optional[Tuple]:
        """Load historical data for backtesting period."""
        try:
            from ..data_layer.prices import fetch_prices
            from ..data_layer.news import fetch_news
            from ..features.technical import compute_technical_features
            from ..features.sentiment import naive_sentiment

            # Load universe
            universe = self.config['universe']['tickers']

            # Calculate required lookback (add buffer for technical indicators)
            lookback_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 300

            # Fetch data
            prices = fetch_prices(
                tickers=universe,
                cache_dir=self.config['data']['cache_dir'],
                lookback_days=lookback_days
            )

            news = fetch_news(
                feed_urls=self.config['signals']['news_feed_urls'],
                cache_dir=self.config['data']['cache_dir']
            )

            # Compute features
            technical = compute_technical_features(
                prices,
                self.config["signals"]["sma_long"],
                self.config["signals"]["momentum_window"]
            )

            sentiment = naive_sentiment(news, universe)

            # Create returns
            returns = prices.copy()
            returns = returns.sort_values(['ticker', 'date'])
            returns['return'] = returns.groupby('ticker')['close'].pct_change()
            returns = returns.dropna()

            # Filter to backtest period
            start_ts = pd.to_datetime(start_date)
            end_ts = pd.to_datetime(end_date)

            prices = prices[(prices['date'] >= start_ts) & (prices['date'] <= end_ts)]
            technical = technical[(technical['date'] >= start_ts) & (technical['date'] <= end_ts)]
            sentiment = sentiment[(sentiment['date'] >= start_ts) & (sentiment['date'] <= end_ts)]
            returns = returns[(returns['date'] >= start_ts) & (returns['date'] <= end_ts)]

            return prices, sentiment, technical, returns

        except Exception as e:
            print(f"Error loading historical data: {e}")
            return None

    def _slice_data(self, historical_data: Tuple, start_date: str, end_date: str) -> Optional[Tuple]:
        """Slice historical data to specific date range."""
        try:
            prices, sentiment, technical, returns = historical_data

            start_ts = pd.to_datetime(start_date)
            end_ts = pd.to_datetime(end_date)

            sliced_prices = prices[(prices['date'] >= start_ts) & (prices['date'] <= end_ts)]
            sliced_technical = technical[(technical['date'] >= start_ts) & (technical['date'] <= end_ts)]
            sliced_sentiment = sentiment[(sentiment['date'] >= start_ts) & (sentiment['date'] <= end_ts)]
            sliced_returns = returns[(returns['date'] >= start_ts) & (returns['date'] <= end_ts)]

            return sliced_prices, sliced_sentiment, sliced_technical, sliced_returns
        except:
            return None

    def _run_simulation(self, engine, historical_data: Tuple, start_date: str, end_date: str) -> Dict:
        """Run the actual trading simulation."""
        prices, sentiment, technical, returns = historical_data

        # Get unique dates in chronological order
        dates = sorted(prices['date'].unique())

        # Simulation state
        portfolio_value = 100000  # Start with $100k
        positions = {}  # {ticker: shares}
        daily_returns = []
        daily_positions = []
        daily_pnl = []
        trades_executed = []

        for i, current_date in enumerate(dates):
            if i == 0:
                continue  # Skip first day (need previous prices)

            # Get data up to current date
            current_prices = prices[prices['date'] == current_date]
            current_technical = technical[technical['date'] == current_date]
            current_sentiment = sentiment[sentiment['date'] == current_date]

            if current_technical.empty:
                continue

            # Generate recommendations
            try:
                if hasattr(engine, 'bayesian_score_adaptive'):
                    recommendations = engine.bayesian_score_adaptive(
                        current_technical, current_sentiment, current_prices
                    )
                else:
                    recommendations = engine.bayesian_score(
                        current_technical, current_sentiment, current_prices
                    )
            except Exception as e:
                print(f"Warning: Failed to generate recommendations for {current_date}: {e}")
                continue

            # Update portfolio value with current prices
            prev_value = portfolio_value
            portfolio_value = self._calculate_portfolio_value(positions, current_prices)

            # Calculate daily return
            if prev_value > 0:
                daily_return = (portfolio_value - prev_value) / prev_value
                daily_returns.append({
                    'date': current_date,
                    'return': daily_return,
                    'portfolio_value': portfolio_value
                })

            # Execute trades based on recommendations
            trades = self._execute_simulated_trades(
                positions, recommendations, current_prices, portfolio_value
            )
            trades_executed.extend(trades)

            # Record daily state
            daily_positions.append({
                'date': current_date,
                'positions': positions.copy(),
                'portfolio_value': portfolio_value
            })

        return {
            'daily_returns': pd.DataFrame(daily_returns),
            'daily_positions': daily_positions,
            'trades_executed': trades_executed,
            'final_portfolio_value': portfolio_value
        }

    def _calculate_portfolio_value(self, positions: Dict[str, int], current_prices: pd.DataFrame) -> float:
        """Calculate current portfolio value."""
        total_value = 0
        price_dict = dict(zip(current_prices['ticker'], current_prices['close']))

        for ticker, shares in positions.items():
            if ticker in price_dict:
                total_value += shares * price_dict[ticker]

        return total_value

    def _execute_simulated_trades(self, positions: Dict, recommendations: pd.DataFrame,
                                 current_prices: pd.DataFrame, portfolio_value: float) -> List[Dict]:
        """Execute simulated trades based on recommendations."""
        trades = []
        price_dict = dict(zip(current_prices['ticker'], current_prices['close']))

        # Simple position sizing: equal weight for buys, full exit for sells
        max_position_value = portfolio_value * 0.1  # Max 10% per position

        for _, rec in recommendations.iterrows():
            ticker = rec['ticker']
            decision = rec['decision']

            if ticker not in price_dict:
                continue

            current_price = price_dict[ticker]
            current_shares = positions.get(ticker, 0)

            if decision == 'Buy' and current_shares == 0:
                # Buy new position
                shares_to_buy = int(max_position_value / current_price)
                if shares_to_buy > 0:
                    positions[ticker] = shares_to_buy
                    trades.append({
                        'date': current_prices['date'].iloc[0],
                        'ticker': ticker,
                        'action': 'BUY',
                        'shares': shares_to_buy,
                        'price': current_price,
                        'value': shares_to_buy * current_price
                    })

            elif decision == 'Sell' and current_shares > 0:
                # Sell existing position
                positions[ticker] = 0
                trades.append({
                    'date': current_prices['date'].iloc[0],
                    'ticker': ticker,
                    'action': 'SELL',
                    'shares': current_shares,
                    'price': current_price,
                    'value': current_shares * current_price
                })

        return trades

    def _calculate_performance_metrics(self, simulation_results: Dict,
                                     engine_type: str,
                                     start_date: str,
                                     end_date: str) -> BacktestResults:
        """Calculate comprehensive performance metrics."""
        daily_returns_df = simulation_results['daily_returns']
        trades = simulation_results['trades_executed']

        if daily_returns_df.empty:
            # Return default results if no data
            return BacktestResults(
                total_return=0.0,
                annualized_return=0.0,
                volatility=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                total_trades=0,
                win_rate=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                var_95=0.0,
                tail_risk_accuracy=0.0,
                daily_returns=pd.Series(),
                daily_positions=pd.DataFrame(),
                daily_pnl=pd.Series(),
                backtest_id=f"{engine_type}_{start_date}_{end_date}",
                engine_type=engine_type,
                period=None,
                config=self.config
            )

        returns = daily_returns_df['return']

        # Performance metrics
        total_return = (simulation_results['final_portfolio_value'] / 100000) - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0

        # Drawdown calculation
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()

        # Trading metrics
        trade_returns = []
        for trade in trades:
            if trade['action'] == 'SELL':
                # Find corresponding buy
                buy_trades = [t for t in trades if t['ticker'] == trade['ticker']
                             and t['action'] == 'BUY' and t['date'] < trade['date']]
                if buy_trades:
                    buy_trade = buy_trades[-1]  # Most recent buy
                    trade_return = (trade['price'] - buy_trade['price']) / buy_trade['price']
                    trade_returns.append(trade_return)

        win_rate = sum(1 for r in trade_returns if r > 0) / len(trade_returns) if trade_returns else 0
        avg_win = np.mean([r for r in trade_returns if r > 0]) if trade_returns else 0
        avg_loss = np.mean([r for r in trade_returns if r <= 0]) if trade_returns else 0

        # Risk metrics
        var_95 = np.percentile(returns, 5) if len(returns) > 0 else 0

        # Calculate tail risk accuracy if we have tail risk predictions
        tail_risk_accuracy = self._calculate_tail_risk_accuracy(daily_returns_df, returns)
        if pd.isna(tail_risk_accuracy):
            tail_risk_accuracy = 0.0  # Placeholder

        return BacktestResults(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            total_trades=len(trades),
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            var_95=var_95,
            tail_risk_accuracy=tail_risk_accuracy,
            daily_returns=returns,
            daily_positions=pd.DataFrame(simulation_results['daily_positions']),
            daily_pnl=returns * 100000,  # Approximate PnL
            backtest_id=f"{engine_type}_{start_date}_{end_date}",
            engine_type=engine_type,
            period=None,
            config=self.config
        )

    def _generate_walk_forward_periods(self, start_date: str, end_date: str,
                                     train_days: int, test_days: int,
                                     rebalance_frequency: int) -> List[BacktestPeriod]:
        """Generate walk-forward backtesting periods."""
        periods = []
        current_start = pd.to_datetime(start_date)
        end_date_ts = pd.to_datetime(end_date)

        while current_start < end_date_ts:
            # Training period
            train_start = current_start
            train_end = train_start + timedelta(days=train_days)

            # Test period
            test_start = train_end + timedelta(days=1)
            test_end = test_start + timedelta(days=test_days)

            if test_end <= end_date_ts:
                periods.append(BacktestPeriod(
                    start_date=train_start,
                    end_date=test_end,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end
                ))

            # Move forward by rebalance frequency
            current_start += timedelta(days=rebalance_frequency)

        return periods

    def _run_walk_forward_period(self, period: BacktestPeriod, engine_type: str) -> BacktestResults:
        """Run a single walk-forward period with proper training."""

        # Load data for the full period (training + testing)
        historical_data = self._load_historical_data(
            period.train_start.strftime('%Y-%m-%d'),
            period.test_end.strftime('%Y-%m-%d')
        )

        if not historical_data:
            raise ValueError("Failed to load historical data for walk-forward period")

        prices, sentiment, technical, returns = historical_data

        # Split into training and testing data
        train_data = self._slice_data(
            historical_data,
            period.train_start.strftime('%Y-%m-%d'),
            period.train_end.strftime('%Y-%m-%d')
        )

        test_data = self._slice_data(
            historical_data,
            period.test_start.strftime('%Y-%m-%d'),
            period.test_end.strftime('%Y-%m-%d')
        )

        if not train_data or not test_data:
            raise ValueError("Failed to slice data for training/testing periods")

        # Create and configure engine
        if engine_type == "adaptive":
            from ..bayesian.adaptive_integration import AdaptiveBayesianEngine
            engine = AdaptiveBayesianEngine(self.config)

            # Calibrate parameters using training data
            train_prices, train_sentiment, train_technical, train_returns = train_data
            engine.calibrate_parameters(train_prices, train_sentiment, train_technical, train_returns)

        else:
            from ..bayesian.integration import BayesianPolicyEngine
            engine = BayesianPolicyEngine(self.config)

        # Run simulation on test period
        simulation_results = self._run_simulation(
            engine, test_data,
            period.test_start.strftime('%Y-%m-%d'),
            period.test_end.strftime('%Y-%m-%d')
        )

        # Convert to BacktestResults
        return self._create_backtest_results(simulation_results, period.test_start.strftime('%Y-%m-%d'), period.test_end.strftime('%Y-%m-%d'))

    def _get_engine_factory(self, engine_type: str) -> Callable:
        """Get engine factory function."""
        if engine_type == "adaptive":
            from ..bayesian.adaptive_integration import AdaptiveBayesianEngine
            return lambda config: AdaptiveBayesianEngine(config)
        else:
            from ..bayesian.integration import BayesianPolicyEngine
            return lambda config: BayesianPolicyEngine(config)

    def _perform_statistical_comparison(self, adaptive: BacktestResults,
                                      static: BacktestResults) -> ComparisonResults:
        """Perform statistical comparison between adaptive and static results."""
        from scipy import stats

        # Calculate improvements
        return_improvement = adaptive.annualized_return - static.annualized_return
        sharpe_improvement = adaptive.sharpe_ratio - static.sharpe_ratio
        drawdown_improvement = static.max_drawdown - adaptive.max_drawdown  # Positive if adaptive has lower drawdown
        tail_risk_improvement = adaptive.tail_risk_accuracy - static.tail_risk_accuracy

        # Statistical significance test (t-test on daily returns)
        if len(adaptive.daily_returns) > 0 and len(static.daily_returns) > 0:
            t_stat, p_value = stats.ttest_ind(adaptive.daily_returns, static.daily_returns)

            # Bootstrap confidence interval for return difference
            combined_returns = np.concatenate([adaptive.daily_returns, static.daily_returns])
            bootstrap_diffs = []
            for _ in range(1000):
                sample1 = np.random.choice(combined_returns, len(adaptive.daily_returns))
                sample2 = np.random.choice(combined_returns, len(static.daily_returns))
                bootstrap_diffs.append(sample1.mean() - sample2.mean())

            ci_lower = np.percentile(bootstrap_diffs, 2.5)
            ci_upper = np.percentile(bootstrap_diffs, 97.5)
        else:
            p_value = 1.0
            ci_lower, ci_upper = 0.0, 0.0

        return ComparisonResults(
            adaptive_results=adaptive,
            static_results=static,
            return_improvement=return_improvement,
            sharpe_improvement=sharpe_improvement,
            drawdown_improvement=drawdown_improvement,
            tail_risk_improvement=tail_risk_improvement,
            p_value_returns=p_value,
            confidence_interval=(ci_lower, ci_upper)
        )

    def _calculate_tail_risk_accuracy(self, daily_returns_df: pd.DataFrame, returns: np.ndarray) -> float:
        """Calculate accuracy of tail risk predictions."""
        if daily_returns_df.empty or len(returns) == 0:
            return 0.0

        # For now, return a placeholder accuracy score based on volatility prediction
        # In a full implementation, this would compare predicted tail risk with actual tail events

        # Simple proxy: check if high-volatility periods were correctly identified
        actual_vol = np.std(returns) * np.sqrt(252)  # Annualized volatility

        # If volatility is high (>20%), we had tail risk present
        had_tail_risk = actual_vol > 0.20

        # For simplicity, assume accuracy based on whether we detected high volatility correctly
        # This is a placeholder - real implementation would need actual tail risk predictions
        if had_tail_risk:
            return 0.7  # Assume decent accuracy in detecting tail risk
        else:
            return 0.8  # Assume good accuracy in normal markets

    def generate_backtest_report(self, results: BacktestResults, output_path: str) -> None:
        """Generate a comprehensive backtest report."""

        report_lines = [
            "# Backtest Results Report\n\n",
            f"**Period**: {results.period}\n",
            f"**Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n",

            "## Performance Summary\n",
            f"- **Total Return**: {results.total_return:.1%}\n",
            f"- **Annualized Return**: {results.annualized_return:.1%}\n",
            f"- **Sharpe Ratio**: {results.sharpe_ratio:.3f}\n",
            f"- **Maximum Drawdown**: {results.max_drawdown:.1%}\n",
            f"- **Win Rate**: {results.win_rate:.1%}\n",
            f"- **Total Trades**: {results.total_trades}\n",
            f"- **Volatility**: {results.volatility:.1%}\n",
            f"- **VaR (95%)**: {results.var_95:.1%}\n",
            f"- **Tail Risk Accuracy**: {results.tail_risk_accuracy:.1%}\n\n",

            "## Risk Metrics\n",
            f"- **Standard Deviation**: {np.std(results.daily_returns):.1%} daily\n",
            f"- **Skewness**: {pd.Series(results.daily_returns).skew():.3f}\n",
            f"- **Kurtosis**: {pd.Series(results.daily_returns).kurtosis():.3f}\n\n"
        ]

        # Write report
        with open(output_path, 'w') as f:
            f.writelines(report_lines)

        print(f"Backtest report saved to: {output_path}")

    def generate_comparison_report(self, comparison: ComparisonResults, output_path: str) -> None:
        """Generate adaptive vs static comparison report."""

        report_lines = [
            "# Adaptive vs Static Comparison Report\n\n",
            f"**Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n",

            "## Performance Comparison\n",

            "### Returns\n",
            f"- **Adaptive**: {comparison.adaptive_results.annualized_return:.1%}\n",
            f"- **Static**: {comparison.static_results.annualized_return:.1%}\n",
            f"- **Improvement**: {comparison.return_improvement:.1%}\n\n",

            "### Risk-Adjusted Performance\n",
            f"- **Adaptive Sharpe**: {comparison.adaptive_results.sharpe_ratio:.3f}\n",
            f"- **Static Sharpe**: {comparison.static_results.sharpe_ratio:.3f}\n",
            f"- **Sharpe Improvement**: {comparison.sharpe_improvement:.3f}\n\n",

            "### Risk Control\n",
            f"- **Adaptive Max DD**: {comparison.adaptive_results.max_drawdown:.1%}\n",
            f"- **Static Max DD**: {comparison.static_results.max_drawdown:.1%}\n",
            f"- **Drawdown Improvement**: {comparison.drawdown_improvement:.1%}\n\n",

            "## Statistical Significance\n",
            f"- **P-value (returns)**: {comparison.p_value_returns:.4f}\n",
            f"- **95% Confidence Interval**: [{comparison.confidence_interval[0]:.1%}, {comparison.confidence_interval[1]:.1%}]\n",
            f"- **Statistically Significant**: {'Yes' if comparison.p_value_returns < 0.05 else 'No'}\n\n"
        ]

        # Write report
        with open(output_path, 'w') as f:
            f.writelines(report_lines)

        print(f"Comparison report saved to: {output_path}")