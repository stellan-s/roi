"""
Performance Attribution Analysis - Decompose adaptive learning contributions.

This module analyzes which specific components of the adaptive system
contribute most to performance improvements over static configuration.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from .framework import BacktestEngine, BacktestResults

@dataclass
class AttributionResult:
    """Results from performance attribution analysis."""
    component_name: str
    return_contribution: float      # Annual return contribution
    sharpe_contribution: float      # Sharpe ratio contribution
    drawdown_contribution: float    # Max drawdown contribution
    hit_rate_contribution: float    # Win rate contribution
    significance_p_value: float     # Statistical significance
    confidence_interval: Tuple[float, float]

@dataclass
class FullAttributionAnalysis:
    """Complete attribution analysis results."""
    baseline_performance: BacktestResults
    full_adaptive_performance: BacktestResults

    # Individual component contributions
    signal_normalization: AttributionResult
    stock_specific_learning: AttributionResult
    regime_adjustments: AttributionResult
    tail_risk_modeling: AttributionResult

    # Summary
    total_explained_return: float
    unexplained_return: float
    component_rankings: List[Tuple[str, float]]

class PerformanceAttributor:
    """
    Decomposes adaptive learning performance into individual component contributions.

    Tests the incremental value of:
    1. Signal normalization learning vs hardcoded scaling
    2. Stock-specific effectiveness vs global priors
    3. Regime adjustment learning vs static multipliers
    4. Statistical tail risk vs heuristic measures
    """

    def __init__(self, backtest_engine: BacktestEngine):
        self.backtest_engine = backtest_engine

    def full_attribution_analysis(self,
                                 start_date: str,
                                 end_date: str) -> FullAttributionAnalysis:
        """
        Complete performance attribution analysis.

        Runs backtests with different combinations of adaptive components
        to isolate the contribution of each feature.
        """

        print(f"Running full attribution analysis: {start_date} to {end_date}")

        # Baseline: Pure static configuration
        baseline = self._run_component_backtest(
            start_date, end_date, "baseline", {}
        )

        # Full adaptive system
        full_adaptive = self._run_component_backtest(
            start_date, end_date, "full_adaptive", {
                'signal_normalization': True,
                'stock_specific_learning': True,
                'regime_adjustments': True,
                'tail_risk_modeling': True
            }
        )

        # Individual components (ablation study)
        signal_norm_only = self._run_component_backtest(
            start_date, end_date, "signal_norm_only", {
                'signal_normalization': True
            }
        )

        stock_learning_only = self._run_component_backtest(
            start_date, end_date, "stock_learning_only", {
                'stock_specific_learning': True
            }
        )

        regime_adj_only = self._run_component_backtest(
            start_date, end_date, "regime_adj_only", {
                'regime_adjustments': True
            }
        )

        tail_risk_only = self._run_component_backtest(
            start_date, end_date, "tail_risk_only", {
                'tail_risk_modeling': True
            }
        )

        # Calculate individual contributions
        signal_normalization = self._calculate_component_attribution(
            baseline, signal_norm_only, "Signal Normalization Learning"
        )

        stock_specific = self._calculate_component_attribution(
            baseline, stock_learning_only, "Stock-Specific Effectiveness Learning"
        )

        regime_adjustments = self._calculate_component_attribution(
            baseline, regime_adj_only, "Regime Adjustment Learning"
        )

        tail_risk = self._calculate_component_attribution(
            baseline, tail_risk_only, "Statistical Tail Risk Modeling"
        )

        # Calculate total and unexplained performance
        total_explained = (signal_normalization.return_contribution +
                         stock_specific.return_contribution +
                         regime_adjustments.return_contribution +
                         tail_risk.return_contribution)

        actual_improvement = full_adaptive.annualized_return - baseline.annualized_return
        unexplained = actual_improvement - total_explained

        # Component rankings
        components = [
            ("Signal Normalization", signal_normalization.return_contribution),
            ("Stock-Specific Learning", stock_specific.return_contribution),
            ("Regime Adjustments", regime_adjustments.return_contribution),
            ("Tail Risk Modeling", tail_risk.return_contribution)
        ]
        rankings = sorted(components, key=lambda x: abs(x[1]), reverse=True)

        return FullAttributionAnalysis(
            baseline_performance=baseline,
            full_adaptive_performance=full_adaptive,
            signal_normalization=signal_normalization,
            stock_specific_learning=stock_specific,
            regime_adjustments=regime_adjustments,
            tail_risk_modeling=tail_risk,
            total_explained_return=total_explained,
            unexplained_return=unexplained,
            component_rankings=rankings
        )

    def parameter_sensitivity_analysis(self,
                                     start_date: str,
                                     end_date: str,
                                     parameter_ranges: Dict[str, List[float]]) -> pd.DataFrame:
        """
        Analyze sensitivity to key learned parameters.

        Tests how performance changes when learned parameters are varied
        around their estimated values.
        """

        print("Running parameter sensitivity analysis...")

        results = []

        for param_name, param_values in parameter_ranges.items():
            print(f"Testing {param_name} sensitivity...")

            for value in param_values:
                # Create modified engine with specific parameter value
                modified_results = self._run_parameter_sensitivity_test(
                    start_date, end_date, param_name, value
                )

                results.append({
                    'parameter': param_name,
                    'value': value,
                    'annualized_return': modified_results.annualized_return,
                    'sharpe_ratio': modified_results.sharpe_ratio,
                    'max_drawdown': modified_results.max_drawdown,
                    'total_trades': modified_results.total_trades
                })

        return pd.DataFrame(results)

    def regime_effectiveness_analysis(self,
                                    start_date: str,
                                    end_date: str) -> pd.DataFrame:
        """
        Analyze learned regime adjustments vs hardcoded multipliers.

        Shows which regime-signal combinations learned the most
        significant improvements over static assumptions.
        """

        print("Analyzing regime effectiveness learning...")

        # Load historical data to analyze actual regime periods
        historical_data = self.backtest_engine._load_historical_data(start_date, end_date)
        if not historical_data:
            return pd.DataFrame()

        prices, sentiment, technical, returns = historical_data

        # Run adaptive backtest to get learned parameters
        from ..bayesian.adaptive_integration import AdaptiveBayesianEngine
        engine = AdaptiveBayesianEngine(self.backtest_engine.config)

        # Calibrate to get learned regime adjustments
        engine.calibrate_parameters(prices, sentiment, technical, returns)

        # Get learned vs default regime adjustments
        diagnostics = engine.get_parameter_diagnostics()
        regime_params = diagnostics[diagnostics['parameter_type'] == 'regime_adjustment']

        if regime_params.empty:
            return pd.DataFrame()

        # Add performance analysis for each regime period
        regime_analysis = []

        for _, param in regime_params.iterrows():
            regime_analysis.append({
                'parameter': param['parameter_name'],
                'learned_value': param['estimated_value'],
                'default_value': param['default_value'],
                'change_percent': param.get('change_percent', 0),
                'confidence_lower': param.get('confidence_lower', 0),
                'confidence_upper': param.get('confidence_upper', 0),
                'n_observations': param.get('n_observations', 0),
                'estimation_method': param.get('estimation_method', 'unknown')
            })

        return pd.DataFrame(regime_analysis)

    def _run_component_backtest(self,
                               start_date: str,
                               end_date: str,
                               config_name: str,
                               enabled_components: Dict[str, bool]) -> BacktestResults:
        """Run backtest with specific components enabled/disabled."""

        # Create engine factory based on enabled components
        def engine_factory(config):
            if not enabled_components:  # Baseline static
                from ..bayesian.integration import BayesianPolicyEngine
                return BayesianPolicyEngine(config)
            else:
                # Create selective adaptive configuration
                modified_config = config.copy()

                # Disable parameter learning components not requested
                if 'parameter_learning' not in modified_config:
                    modified_config['parameter_learning'] = {}

                # Configure based on enabled components
                if 'signal_normalization' not in enabled_components:
                    # Disable adaptive signal normalization - use hardcoded scaling
                    modified_config['parameter_learning']['sentiment_normalization'] = 'hardcoded'
                    modified_config['parameter_learning']['momentum_scaling'] = 'hardcoded'

                if 'regime_adjustments' not in enabled_components:
                    # Disable regime effectiveness learning
                    modified_config['parameter_learning']['regime_effectiveness_learning'] = False

                if 'tail_risk_modeling' not in enabled_components:
                    # Fall back to heuristic tail risk
                    if 'tail_risk' not in modified_config:
                        modified_config['tail_risk'] = {}
                    modified_config['tail_risk']['calculation_method'] = 'heuristic'

                from ..bayesian.adaptive_integration import AdaptiveBayesianEngine
                return AdaptiveBayesianEngine(modified_config)

        return self.backtest_engine.run_single_backtest(
            start_date, end_date, engine_factory,
            f"component_{config_name}"
        )

    def _calculate_component_attribution(self,
                                       baseline: BacktestResults,
                                       component: BacktestResults,
                                       component_name: str) -> AttributionResult:
        """Calculate attribution for a single component."""

        # Performance contributions
        return_contribution = component.annualized_return - baseline.annualized_return
        sharpe_contribution = component.sharpe_ratio - baseline.sharpe_ratio
        drawdown_contribution = baseline.max_drawdown - component.max_drawdown  # Positive if component reduces drawdown
        hit_rate_contribution = component.win_rate - baseline.win_rate

        # Statistical significance test
        if len(baseline.daily_returns) > 0 and len(component.daily_returns) > 0:
            from scipy import stats
            _, p_value = stats.ttest_ind(component.daily_returns, baseline.daily_returns)

            # Bootstrap confidence interval
            return_diffs = []
            for _ in range(1000):
                sample_baseline = np.random.choice(baseline.daily_returns, len(baseline.daily_returns))
                sample_component = np.random.choice(component.daily_returns, len(component.daily_returns))
                return_diffs.append(sample_component.mean() - sample_baseline.mean())

            ci_lower = np.percentile(return_diffs, 2.5)
            ci_upper = np.percentile(return_diffs, 97.5)
        else:
            p_value = 1.0
            ci_lower, ci_upper = 0.0, 0.0

        return AttributionResult(
            component_name=component_name,
            return_contribution=return_contribution,
            sharpe_contribution=sharpe_contribution,
            drawdown_contribution=drawdown_contribution,
            hit_rate_contribution=hit_rate_contribution,
            significance_p_value=p_value,
            confidence_interval=(ci_lower, ci_upper)
        )

    def _run_parameter_sensitivity_test(self,
                                      start_date: str,
                                      end_date: str,
                                      parameter_name: str,
                                      parameter_value: float) -> BacktestResults:
        """Run backtest with specific parameter value."""

        # Create configuration with parameter override
        def engine_factory(config):
            modified_config = config.copy()

            # Apply parameter override based on parameter name
            if parameter_name == 'momentum_effectiveness':
                if 'bayesian' not in modified_config:
                    modified_config['bayesian'] = {}
                if 'priors' not in modified_config['bayesian']:
                    modified_config['bayesian']['priors'] = {}
                modified_config['bayesian']['priors']['momentum_effectiveness'] = parameter_value

            elif parameter_name == 'sentiment_effectiveness':
                if 'bayesian' not in modified_config:
                    modified_config['bayesian'] = {}
                if 'priors' not in modified_config['bayesian']:
                    modified_config['bayesian']['priors'] = {}
                modified_config['bayesian']['priors']['sentiment_effectiveness'] = parameter_value

            elif parameter_name == 'trend_effectiveness':
                if 'bayesian' not in modified_config:
                    modified_config['bayesian'] = {}
                if 'priors' not in modified_config['bayesian']:
                    modified_config['bayesian']['priors'] = {}
                modified_config['bayesian']['priors']['trend_effectiveness'] = parameter_value

            elif parameter_name == 'buy_probability':
                if 'bayesian' not in modified_config:
                    modified_config['bayesian'] = {}
                if 'decision_thresholds' not in modified_config['bayesian']:
                    modified_config['bayesian']['decision_thresholds'] = {}
                modified_config['bayesian']['decision_thresholds']['buy_probability'] = parameter_value

            elif parameter_name == 'sell_probability':
                if 'bayesian' not in modified_config:
                    modified_config['bayesian'] = {}
                if 'decision_thresholds' not in modified_config['bayesian']:
                    modified_config['bayesian']['decision_thresholds'] = {}
                modified_config['bayesian']['decision_thresholds']['sell_probability'] = parameter_value

            # Use adaptive engine with modified configuration
            from ..bayesian.adaptive_integration import AdaptiveBayesianEngine
            return AdaptiveBayesianEngine(modified_config)

        return self.backtest_engine.run_single_backtest(
            start_date, end_date, engine_factory,
            f"sensitivity_{parameter_name}_{parameter_value}"
        )

    def generate_attribution_report(self,
                                   analysis: FullAttributionAnalysis,
                                   output_path: str) -> None:
        """Generate comprehensive attribution analysis report."""

        report_lines = [
            "# Performance Attribution Analysis Report\n",
            f"**Analysis Period**: {analysis.baseline_performance.period}\n",
            f"**Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n",

            "## Executive Summary\n",
            f"- **Baseline Performance**: {analysis.baseline_performance.annualized_return:.1%} annual return",
            f"- **Full Adaptive Performance**: {analysis.full_adaptive_performance.annualized_return:.1%} annual return",
            f"- **Total Improvement**: {(analysis.full_adaptive_performance.annualized_return - analysis.baseline_performance.annualized_return):.1%}\n",

            "## Component Contributions\n"
        ]

        # Component rankings
        for i, (component, contribution) in enumerate(analysis.component_rankings, 1):
            report_lines.append(f"{i}. **{component}**: {contribution:.1%} annual return contribution\n")

        report_lines.extend([
            f"\n**Total Explained**: {analysis.total_explained_return:.1%}",
            f"**Unexplained**: {analysis.unexplained_return:.1%}\n\n",

            "## Detailed Component Analysis\n"
        ])

        # Detailed analysis for each component
        components = [
            ("Signal Normalization Learning", analysis.signal_normalization),
            ("Stock-Specific Learning", analysis.stock_specific_learning),
            ("Regime Adjustments", analysis.regime_adjustments),
            ("Statistical Tail Risk", analysis.tail_risk_modeling)
        ]

        for name, result in components:
            report_lines.extend([
                f"### {name}\n",
                f"- **Return Contribution**: {result.return_contribution:.1%}",
                f"- **Sharpe Contribution**: {result.sharpe_contribution:.3f}",
                f"- **Drawdown Improvement**: {result.drawdown_contribution:.1%}",
                f"- **Hit Rate Improvement**: {result.hit_rate_contribution:.1%}",
                f"- **Statistical Significance**: p={result.significance_p_value:.3f}",
                f"- **95% Confidence Interval**: [{result.confidence_interval[0]:.1%}, {result.confidence_interval[1]:.1%}]\n\n"
            ])

        # Write report
        with open(output_path, 'w') as f:
            f.writelines(report_lines)

        print(f"Attribution report saved to: {output_path}")

    def quick_attribution_summary(self,
                                 start_date: str,
                                 end_date: str) -> Dict[str, float]:
        """
        Quick performance attribution summary for immediate insights.

        Returns simplified metrics showing which adaptive components
        contribute most to performance.
        """

        print(f"Quick attribution analysis: {start_date} to {end_date}")

        # Run baseline and full adaptive
        baseline = self._run_component_backtest(start_date, end_date, "baseline", {})
        full_adaptive = self._run_component_backtest(
            start_date, end_date, "full_adaptive", {
                'signal_normalization': True,
                'stock_specific_learning': True,
                'regime_adjustments': True,
                'tail_risk_modeling': True
            }
        )

        # Simple component tests (individual enables)
        components_to_test = [
            ('signal_normalization', 'Signal Normalization'),
            ('stock_specific_learning', 'Stock-Specific Learning'),
            ('regime_adjustments', 'Regime Adjustments'),
            ('tail_risk_modeling', 'Tail Risk Modeling')
        ]

        summary = {
            'baseline_return': baseline.annualized_return,
            'full_adaptive_return': full_adaptive.annualized_return,
            'total_improvement': full_adaptive.annualized_return - baseline.annualized_return
        }

        for component_key, component_name in components_to_test:
            component_result = self._run_component_backtest(
                start_date, end_date, component_key, {component_key: True}
            )
            contribution = component_result.annualized_return - baseline.annualized_return
            summary[f'{component_key}_contribution'] = contribution

        return summary