"""
Risk Management Module

Provides comprehensive risk analysis including:
- Statistical tail risk calculation with P[return < -2σ]
- Portfolio-level risk assessment
- Stress testing and scenario analysis
- Risk-adjusted position sizing
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass
from scipy import stats

from .base import BaseModule, ModuleOutput, ModuleContract

@dataclass
class RiskMetrics:
    """Comprehensive risk metrics for a position or portfolio"""
    # Primary tail risk measures
    downside_tail_risk: float      # P[return < -2σ]
    extreme_move_prob: float       # P[|return| > 2σ]
    expected_shortfall: float      # E[return | return < -2σ]

    # Portfolio metrics
    volatility_annual: float       # Annualized volatility
    sharpe_ratio: float           # Risk-adjusted return
    max_drawdown_1year: float     # Maximum drawdown over 1 year

    # Distribution characteristics
    distribution_type: str        # "normal", "student_t", "empirical"
    skewness: float              # Distribution skewness
    kurtosis: float              # Distribution kurtosis

    # Confidence and quality
    confidence_level: float       # Statistical confidence in estimates
    data_quality_score: float    # Quality of underlying data

@dataclass
class StressScenario:
    """Market stress scenario definition"""
    name: str
    description: str
    market_shock: float          # Market-wide shock (e.g., -0.20 for -20%)
    volatility_multiplier: float # Volatility increase factor
    duration_days: int           # Scenario duration

class RiskManagementModule(BaseModule):
    """Module for comprehensive risk management and assessment"""

    def define_contract(self) -> ModuleContract:
        return ModuleContract(
            name="risk_management",
            version="1.0.0",
            description="Comprehensive risk analysis and management",
            input_schema={
                "prices": "pd.DataFrame[date, ticker, close]",
                "portfolio_weights": "Dict[ticker, float]",
                "expected_returns": "Dict[ticker, float]"
            },
            output_schema={
                "individual_risks": "Dict[ticker, RiskMetrics]",
                "portfolio_risk": "RiskMetrics",
                "stress_test_results": "Dict[scenario, Dict]",
                "risk_recommendations": "Dict[ticker, Dict]",
                "risk_summary": "Dict[str, Any]"
            },
            performance_sla={
                "max_latency_ms": 500.0,
                "min_confidence": 0.6
            },
            dependencies=[],
            optional_inputs=["portfolio_weights", "expected_returns"]
        )

    def process(self, inputs: Dict[str, Any]) -> ModuleOutput:
        """Process risk analysis for prices and portfolio"""
        prices_df = inputs['prices']
        portfolio_weights = inputs.get('portfolio_weights', {})
        expected_returns = inputs.get('expected_returns', {})

        # Validate inputs
        if prices_df.empty:
            return self._default_risk_output("no_price_data")

        try:
            # Calculate individual risks for each ticker
            individual_risks = self._calculate_individual_risks(prices_df, expected_returns)

            if not individual_risks:
                return self._default_risk_output("no_valid_risks")

            # Calculate portfolio-level risk if weights provided
            portfolio_risk = None
            if portfolio_weights:
                portfolio_risk = self._calculate_portfolio_risk(
                    individual_risks, portfolio_weights, prices_df
                )

            # Run stress tests
            stress_results = self._run_stress_tests(individual_risks, portfolio_weights)

            # Generate risk recommendations
            recommendations = self._generate_recommendations(individual_risks, portfolio_weights)

            # Create risk summary
            risk_summary = self._create_risk_summary(individual_risks, portfolio_risk, stress_results)

            # Calculate overall confidence
            confidence = self._calculate_confidence(individual_risks, prices_df)

            metadata = {
                "risk_method": "statistical_tail_risk",
                "stress_scenarios_tested": len(stress_results),
                "positions_analyzed": len(individual_risks),
                "portfolio_analysis": portfolio_risk is not None
            }

            return ModuleOutput(
                data={
                    "individual_risks": {k: self._risk_metrics_to_dict(v) for k, v in individual_risks.items()},
                    "portfolio_risk": self._risk_metrics_to_dict(portfolio_risk) if portfolio_risk else None,
                    "stress_test_results": stress_results,
                    "risk_recommendations": recommendations,
                    "risk_summary": risk_summary
                },
                metadata=metadata,
                confidence=confidence
            )

        except Exception as e:
            return ModuleOutput(
                data={
                    "individual_risks": {},
                    "portfolio_risk": None,
                    "stress_test_results": {},
                    "risk_recommendations": {},
                    "risk_summary": {"error": str(e)}
                },
                metadata={"error": str(e)},
                confidence=0.1
            )

    def test_module(self) -> Dict[str, Any]:
        """Test the risk management module with synthetic data"""
        # Generate test data
        test_prices = self._generate_test_prices()
        test_weights = {"AAPL": 0.4, "MSFT": 0.3, "GOOGL": 0.3}
        test_returns = {"AAPL": 0.12, "MSFT": 0.10, "GOOGL": 0.15}

        # Test processing
        result = self.process({
            'prices': test_prices,
            'portfolio_weights': test_weights,
            'expected_returns': test_returns
        })

        # Validate outputs
        individual_risks = result.data['individual_risks']
        portfolio_risk = result.data['portfolio_risk']
        stress_results = result.data['stress_test_results']

        tests_passed = 0
        total_tests = 8

        # Test 1: Individual risks calculated
        if individual_risks and len(individual_risks) >= 3:
            tests_passed += 1

        # Test 2: Portfolio risk calculated
        if portfolio_risk is not None:
            tests_passed += 1

        # Test 3: Valid risk metrics ranges
        valid_metrics = True
        for ticker, risk in individual_risks.items():
            if not (0 <= risk['downside_tail_risk'] <= 1):
                valid_metrics = False
            if not (0 <= risk['volatility_annual'] <= 2):
                valid_metrics = False
        if valid_metrics:
            tests_passed += 1

        # Test 4: Stress tests completed
        if stress_results and len(stress_results) >= 2:
            tests_passed += 1

        # Test 5: Portfolio volatility reasonable
        if portfolio_risk and 0.05 <= portfolio_risk['volatility_annual'] <= 0.5:
            tests_passed += 1

        # Test 6: Tail risk probabilities reasonable
        valid_tail_risk = True
        for ticker, risk in individual_risks.items():
            # Normal distribution tail risk should be around 2.3%
            if not (0.001 <= risk['downside_tail_risk'] <= 0.15):
                valid_tail_risk = False
        if valid_tail_risk:
            tests_passed += 1

        # Test 7: Confidence reasonable
        if 0.5 <= result.confidence <= 1.0:
            tests_passed += 1

        # Test 8: Risk summary contains key metrics
        risk_summary = result.data['risk_summary']
        if risk_summary and 'total_positions' in risk_summary and 'avg_volatility' in risk_summary:
            tests_passed += 1

        return {
            "status": "PASS" if tests_passed >= 6 else "FAIL",
            "tests_passed": tests_passed,
            "total_tests": total_tests,
            "portfolio_volatility": portfolio_risk['volatility_annual'] if portfolio_risk else 0,
            "confidence": result.confidence
        }

    def _generate_test_inputs(self) -> Dict[str, Any]:
        """Generate synthetic test data for benchmarking"""
        return {
            "prices": self._generate_test_prices(),
            "portfolio_weights": {"AAPL": 0.4, "MSFT": 0.3, "GOOGL": 0.3},
            "expected_returns": {"AAPL": 0.12, "MSFT": 0.10, "GOOGL": 0.15}
        }

    def _generate_test_prices(self) -> pd.DataFrame:
        """Generate synthetic price data for testing"""
        dates = pd.date_range('2023-01-01', periods=252, freq='D')  # 1 year of data
        tickers = ['AAPL', 'MSFT', 'GOOGL']

        data = []
        np.random.seed(42)  # Reproducible results

        for ticker in tickers:
            base_price = 100

            # Different volatility regimes for testing
            if ticker == 'AAPL':
                daily_vol = 0.015  # 1.5% daily vol (moderate)
                drift = 0.0004     # 0.04% daily drift
            elif ticker == 'MSFT':
                daily_vol = 0.012  # 1.2% daily vol (low)
                drift = 0.0003     # 0.03% daily drift
            else:  # GOOGL
                daily_vol = 0.020  # 2.0% daily vol (high)
                drift = 0.0005     # 0.05% daily drift

            prices = [base_price]
            for i in range(1, len(dates)):
                # Add some regime changes and jumps for realistic testing
                if i == 100:  # Market shock
                    shock = -0.08  # -8% shock
                elif i == 150:  # Recovery
                    shock = 0.04   # +4% recovery
                else:
                    shock = 0

                change = np.random.normal(drift, daily_vol) + shock
                new_price = prices[-1] * (1 + change)
                prices.append(max(10, new_price))  # Floor at $10

            for i, date in enumerate(dates):
                data.append({
                    'date': date,
                    'ticker': ticker,
                    'close': prices[i]
                })

        return pd.DataFrame(data)

    def _calculate_individual_risks(self, prices_df: pd.DataFrame, expected_returns: Dict[str, float]) -> Dict[str, RiskMetrics]:
        """Calculate risk metrics for each individual ticker"""
        individual_risks = {}

        for ticker in prices_df['ticker'].unique():
            ticker_data = prices_df[prices_df['ticker'] == ticker].sort_values('date')

            if len(ticker_data) < 30:  # Need minimum data
                continue

            # Calculate returns
            ticker_data = ticker_data.copy()
            ticker_data['return'] = ticker_data['close'].pct_change()
            returns = ticker_data['return'].dropna()

            if len(returns) < 20:
                continue

            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(returns, expected_returns.get(ticker, 0.1))
            individual_risks[ticker] = risk_metrics

        return individual_risks

    def _calculate_risk_metrics(self, returns: pd.Series, expected_return: float) -> RiskMetrics:
        """Calculate comprehensive risk metrics for a return series"""

        # Basic statistics
        volatility_daily = returns.std()
        volatility_annual = volatility_daily * np.sqrt(252)
        mean_return = returns.mean()

        # Distribution analysis
        distribution_type, distribution_params = self._analyze_distribution(returns)

        # Tail risk calculations
        downside_tail_risk = self._calculate_downside_tail_risk(returns, volatility_daily, distribution_type, distribution_params)
        extreme_move_prob = self._calculate_extreme_move_prob(returns, volatility_daily, distribution_type, distribution_params)
        expected_shortfall = self._calculate_expected_shortfall(returns, volatility_daily)

        # Risk-adjusted metrics
        risk_free_rate = self.config.get('risk_free_rate', 0.02)
        sharpe_ratio = (expected_return - risk_free_rate) / volatility_annual if volatility_annual > 0 else 0

        # Maximum drawdown calculation
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown_1year = abs(drawdowns.min())

        # Data quality assessment
        data_quality_score = self._assess_data_quality(returns)

        return RiskMetrics(
            downside_tail_risk=downside_tail_risk,
            extreme_move_prob=extreme_move_prob,
            expected_shortfall=expected_shortfall,
            volatility_annual=volatility_annual,
            sharpe_ratio=sharpe_ratio,
            max_drawdown_1year=max_drawdown_1year,
            distribution_type=distribution_type,
            skewness=stats.skew(returns),
            kurtosis=stats.kurtosis(returns),
            confidence_level=0.95,
            data_quality_score=data_quality_score
        )

    def _analyze_distribution(self, returns: pd.Series) -> tuple:
        """Analyze return distribution and fit best model"""

        # Test for normality
        _, p_value = stats.jarque_bera(returns)

        if p_value > 0.05:  # Accept normality
            return "normal", {
                'mean': returns.mean(),
                'std': returns.std()
            }

        # Try Student-t distribution for heavy tails
        try:
            df, loc, scale = stats.t.fit(returns)
            if df > 2.5:  # Reasonable degrees of freedom
                return "student_t", {
                    'df': df,
                    'loc': loc,
                    'scale': scale
                }
        except:
            pass

        # Fall back to empirical distribution
        return "empirical", {
            'returns': returns.values
        }

    def _calculate_downside_tail_risk(self, returns: pd.Series, volatility: float,
                                    distribution_type: str, params: Dict) -> float:
        """Calculate P[return < -2σ]"""

        threshold = -2 * volatility

        if distribution_type == "normal":
            mean = params['mean']
            std = params['std']
            z_score = (threshold - mean) / std
            return stats.norm.cdf(z_score)

        elif distribution_type == "student_t":
            df = params['df']
            loc = params['loc']
            scale = params['scale']
            return stats.t.cdf(threshold, df, loc, scale)

        else:  # empirical
            return (returns < threshold).mean()

    def _calculate_extreme_move_prob(self, returns: pd.Series, volatility: float,
                                   distribution_type: str, params: Dict) -> float:
        """Calculate P[|return| > 2σ]"""

        upper_threshold = 2 * volatility
        lower_threshold = -2 * volatility

        if distribution_type == "normal":
            mean = params['mean']
            std = params['std']

            z_upper = (upper_threshold - mean) / std
            z_lower = (lower_threshold - mean) / std

            prob_upper = 1 - stats.norm.cdf(z_upper)
            prob_lower = stats.norm.cdf(z_lower)

            return prob_upper + prob_lower

        elif distribution_type == "student_t":
            df = params['df']
            loc = params['loc']
            scale = params['scale']

            prob_upper = 1 - stats.t.cdf(upper_threshold, df, loc, scale)
            prob_lower = stats.t.cdf(lower_threshold, df, loc, scale)

            return prob_upper + prob_lower

        else:  # empirical
            return ((returns > upper_threshold) | (returns < lower_threshold)).mean()

    def _calculate_expected_shortfall(self, returns: pd.Series, volatility: float) -> float:
        """Calculate E[return | return < -2σ]"""

        threshold = -2 * volatility
        tail_returns = returns[returns < threshold]

        if len(tail_returns) == 0:
            return threshold * 1.5  # Conservative estimate

        return tail_returns.mean()

    def _assess_data_quality(self, returns: pd.Series) -> float:
        """Assess quality of return data"""

        quality_score = 1.0

        # Penalize for insufficient data
        if len(returns) < 100:
            quality_score *= 0.7
        elif len(returns) < 50:
            quality_score *= 0.4

        # Penalize for extreme outliers
        outlier_threshold = 5 * returns.std()
        outlier_ratio = (abs(returns) > outlier_threshold).mean()
        if outlier_ratio > 0.05:  # More than 5% outliers
            quality_score *= 0.8

        # Penalize for missing data (gaps)
        # This would require date continuity analysis

        return max(0.1, quality_score)

    def _calculate_portfolio_risk(self, individual_risks: Dict[str, RiskMetrics],
                                portfolio_weights: Dict[str, float],
                                prices_df: pd.DataFrame) -> RiskMetrics:
        """Calculate portfolio-level risk metrics"""

        if not portfolio_weights or not individual_risks:
            return None

        # Portfolio expected return and volatility (simplified)
        portfolio_vol_squared = 0
        portfolio_expected_return = 0
        total_weight = sum(abs(w) for w in portfolio_weights.values())

        # Normalize weights
        normalized_weights = {k: v/total_weight for k, v in portfolio_weights.items()}

        for ticker, weight in normalized_weights.items():
            if ticker in individual_risks and weight != 0:
                risk = individual_risks[ticker]
                portfolio_vol_squared += (weight * risk.volatility_annual) ** 2
                # For expected return, we'd need expected returns input

        portfolio_volatility = np.sqrt(portfolio_vol_squared)

        # Aggregate tail risks (conservative approach)
        weighted_downside_risk = sum(
            abs(weight) * individual_risks[ticker].downside_tail_risk
            for ticker, weight in normalized_weights.items()
            if ticker in individual_risks
        )

        weighted_extreme_prob = sum(
            abs(weight) * individual_risks[ticker].extreme_move_prob
            for ticker, weight in normalized_weights.items()
            if ticker in individual_risks
        )

        weighted_shortfall = sum(
            abs(weight) * individual_risks[ticker].expected_shortfall
            for ticker, weight in normalized_weights.items()
            if ticker in individual_risks
        )

        # Portfolio metrics
        avg_sharpe = np.mean([risk.sharpe_ratio for risk in individual_risks.values()])
        avg_drawdown = np.mean([risk.max_drawdown_1year for risk in individual_risks.values()])

        return RiskMetrics(
            downside_tail_risk=weighted_downside_risk,
            extreme_move_prob=weighted_extreme_prob,
            expected_shortfall=weighted_shortfall,
            volatility_annual=portfolio_volatility,
            sharpe_ratio=avg_sharpe,  # Simplified
            max_drawdown_1year=avg_drawdown,  # Simplified
            distribution_type="portfolio",
            skewness=0.0,  # Would need complex calculation
            kurtosis=0.0,  # Would need complex calculation
            confidence_level=0.95,
            data_quality_score=np.mean([risk.data_quality_score for risk in individual_risks.values()])
        )

    def _run_stress_tests(self, individual_risks: Dict[str, RiskMetrics],
                        portfolio_weights: Dict[str, float]) -> Dict[str, Dict]:
        """Run stress test scenarios"""

        # Define stress scenarios
        scenarios = {
            "market_crash": StressScenario(
                name="Market Crash",
                description="Severe market decline with volatility spike",
                market_shock=-0.20,
                volatility_multiplier=2.5,
                duration_days=10
            ),
            "black_swan": StressScenario(
                name="Black Swan Event",
                description="Extreme tail event with massive volatility",
                market_shock=-0.35,
                volatility_multiplier=4.0,
                duration_days=5
            ),
            "recession": StressScenario(
                name="Economic Recession",
                description="Prolonged economic downturn",
                market_shock=-0.15,
                volatility_multiplier=1.8,
                duration_days=90
            )
        }

        stress_results = {}

        for scenario_name, scenario in scenarios.items():
            # Calculate scenario impact
            total_portfolio_loss = 0
            position_impacts = {}

            for ticker, risk in individual_risks.items():
                weight = portfolio_weights.get(ticker, 0)
                if weight == 0:
                    continue

                # Base loss from market shock
                base_loss = scenario.market_shock * weight

                # Amplify based on individual tail risk
                tail_amplification = 1 + (risk.downside_tail_risk * 2)  # 2x amplification factor

                # Volatility shock impact
                vol_impact = risk.volatility_annual * scenario.volatility_multiplier * 0.1

                total_loss = base_loss * tail_amplification - vol_impact

                position_impacts[ticker] = {
                    'base_loss': base_loss,
                    'tail_amplified_loss': total_loss,
                    'weight': weight
                }

                total_portfolio_loss += total_loss

            stress_results[scenario_name] = {
                'scenario_description': scenario.description,
                'total_portfolio_loss_pct': total_portfolio_loss,
                'position_impacts': position_impacts,
                'market_shock': scenario.market_shock,
                'duration_days': scenario.duration_days
            }

        return stress_results

    def _generate_recommendations(self, individual_risks: Dict[str, RiskMetrics],
                                portfolio_weights: Dict[str, float]) -> Dict[str, Dict]:
        """Generate risk-based recommendations"""

        recommendations = {}

        # Risk thresholds from config
        high_tail_risk_threshold = self.config.get('high_tail_risk_threshold', 0.05)
        high_volatility_threshold = self.config.get('high_volatility_threshold', 0.30)

        for ticker, risk in individual_risks.items():
            current_weight = portfolio_weights.get(ticker, 0)

            recommendation = {
                'current_weight': current_weight,
                'risk_level': 'LOW',
                'recommended_action': 'HOLD',
                'max_recommended_weight': 0.20,
                'reasoning': []
            }

            # Assess risk level
            if risk.downside_tail_risk > high_tail_risk_threshold:
                recommendation['risk_level'] = 'HIGH'
                recommendation['reasoning'].append(f"High tail risk: {risk.downside_tail_risk:.1%}")

            if risk.volatility_annual > high_volatility_threshold:
                recommendation['risk_level'] = 'HIGH'
                recommendation['reasoning'].append(f"High volatility: {risk.volatility_annual:.1%}")

            if risk.max_drawdown_1year > 0.25:
                recommendation['risk_level'] = 'MODERATE'
                recommendation['reasoning'].append(f"High drawdown: {risk.max_drawdown_1year:.1%}")

            # Generate action recommendation
            if recommendation['risk_level'] == 'HIGH':
                recommendation['recommended_action'] = 'REDUCE' if current_weight > 0.10 else 'AVOID'
                recommendation['max_recommended_weight'] = 0.05
            elif recommendation['risk_level'] == 'MODERATE':
                recommendation['max_recommended_weight'] = 0.15

            # Adjust based on Sharpe ratio
            if risk.sharpe_ratio > 1.0:
                recommendation['reasoning'].append(f"Good risk-adjusted return: {risk.sharpe_ratio:.2f}")
                recommendation['max_recommended_weight'] *= 1.2
            elif risk.sharpe_ratio < 0.5:
                recommendation['reasoning'].append(f"Poor risk-adjusted return: {risk.sharpe_ratio:.2f}")
                recommendation['max_recommended_weight'] *= 0.8

            recommendations[ticker] = recommendation

        return recommendations

    def _create_risk_summary(self, individual_risks: Dict[str, RiskMetrics],
                           portfolio_risk: Optional[RiskMetrics],
                           stress_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Create comprehensive risk summary"""

        if not individual_risks:
            return {"error": "No risk data available"}

        # Aggregate individual risk metrics
        avg_volatility = np.mean([r.volatility_annual for r in individual_risks.values()])
        avg_tail_risk = np.mean([r.downside_tail_risk for r in individual_risks.values()])
        avg_sharpe = np.mean([r.sharpe_ratio for r in individual_risks.values()])
        max_drawdown = max([r.max_drawdown_1year for r in individual_risks.values()])

        # Count risk levels
        high_risk_count = sum(1 for r in individual_risks.values()
                            if r.downside_tail_risk > 0.05 or r.volatility_annual > 0.30)

        # Stress test summary
        worst_stress_loss = min([s['total_portfolio_loss_pct'] for s in stress_results.values()]) if stress_results else 0

        summary = {
            'total_positions': len(individual_risks),
            'avg_volatility': avg_volatility,
            'avg_tail_risk': avg_tail_risk,
            'avg_sharpe_ratio': avg_sharpe,
            'max_portfolio_drawdown': max_drawdown,
            'high_risk_positions': high_risk_count,
            'portfolio_risk_level': self._assess_portfolio_risk_level(avg_volatility, avg_tail_risk, high_risk_count),
            'stress_test_worst_loss': worst_stress_loss,
            'data_quality_avg': np.mean([r.data_quality_score for r in individual_risks.values()])
        }

        # Add portfolio-specific metrics if available
        if portfolio_risk:
            summary.update({
                'portfolio_volatility': portfolio_risk.volatility_annual,
                'portfolio_tail_risk': portfolio_risk.downside_tail_risk,
                'portfolio_sharpe': portfolio_risk.sharpe_ratio
            })

        return summary

    def _assess_portfolio_risk_level(self, avg_vol: float, avg_tail_risk: float, high_risk_count: int) -> str:
        """Assess overall portfolio risk level"""

        risk_score = 0

        if avg_vol > 0.25:
            risk_score += 2
        elif avg_vol > 0.20:
            risk_score += 1

        if avg_tail_risk > 0.04:
            risk_score += 2
        elif avg_tail_risk > 0.03:
            risk_score += 1

        if high_risk_count > 2:
            risk_score += 2
        elif high_risk_count > 0:
            risk_score += 1

        if risk_score >= 5:
            return "HIGH RISK"
        elif risk_score >= 3:
            return "MODERATE RISK"
        else:
            return "LOW RISK"

    def _calculate_confidence(self, individual_risks: Dict[str, RiskMetrics], prices_df: pd.DataFrame) -> float:
        """Calculate overall confidence in risk assessment"""

        if not individual_risks:
            return 0.1

        # Data quality component
        avg_data_quality = np.mean([r.data_quality_score for r in individual_risks.values()])

        # Coverage component (how many tickers we could analyze)
        total_tickers = len(prices_df['ticker'].unique())
        analyzed_tickers = len(individual_risks)
        coverage_ratio = analyzed_tickers / total_tickers if total_tickers > 0 else 0

        # Model reliability component
        normal_dist_ratio = sum(1 for r in individual_risks.values()
                              if r.distribution_type == "normal") / len(individual_risks)
        model_reliability = 0.5 + (normal_dist_ratio * 0.3)  # Normal distributions are more reliable

        # Combined confidence
        confidence = (
            avg_data_quality * 0.4 +
            coverage_ratio * 0.3 +
            model_reliability * 0.3
        )

        return min(1.0, max(0.1, confidence))

    def _risk_metrics_to_dict(self, risk_metrics: Optional[RiskMetrics]) -> Optional[Dict]:
        """Convert RiskMetrics dataclass to dictionary"""
        if risk_metrics is None:
            return None

        return {
            'downside_tail_risk': risk_metrics.downside_tail_risk,
            'extreme_move_prob': risk_metrics.extreme_move_prob,
            'expected_shortfall': risk_metrics.expected_shortfall,
            'volatility_annual': risk_metrics.volatility_annual,
            'sharpe_ratio': risk_metrics.sharpe_ratio,
            'max_drawdown_1year': risk_metrics.max_drawdown_1year,
            'distribution_type': risk_metrics.distribution_type,
            'skewness': risk_metrics.skewness,
            'kurtosis': risk_metrics.kurtosis,
            'confidence_level': risk_metrics.confidence_level,
            'data_quality_score': risk_metrics.data_quality_score
        }

    def _default_risk_output(self, reason: str) -> ModuleOutput:
        """Return default risk output for error cases"""
        return ModuleOutput(
            data={
                "individual_risks": {},
                "portfolio_risk": None,
                "stress_test_results": {},
                "risk_recommendations": {},
                "risk_summary": {"error": reason}
            },
            metadata={"reason": reason},
            confidence=0.1
        )