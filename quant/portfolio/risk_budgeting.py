"""
Risk Budgeting Portfolio Optimization

This module implements intelligent position sizing based on:
- Expected return vs uncertainty trade-offs
- Stock-specific factor profiles and tail risk
- Regime-aware allocation constraints
- Diversification across factor categories

Instead of equal-weight allocation, positions are sized based on:
1. Risk-adjusted expected returns (Sharpe-like ratios)
2. Factor profile diversification requirements
3. Tail risk penalties for high-risk positions
4. Regime-specific allocation constraints
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class RiskBudgetConfig:
    """Configuration for risk budgeting optimization"""
    max_position_weight: float = 0.15          # Max 15% per stock
    max_factor_concentration: float = 0.35     # Max 35% per factor category
    min_positions: int = 5                     # Minimum diversification
    max_positions: int = 20                    # Focus constraint
    tail_risk_penalty_factor: float = 0.5     # Penalty for high tail risk
    uncertainty_penalty_factor: float = 2.0   # Penalty for high uncertainty
    regime_allocation_multipliers: Dict[str, float] = None  # Regime-specific constraints


@dataclass
class PositionSizing:
    """Individual position sizing with risk metrics"""
    ticker: str
    target_weight: float
    expected_return: float
    risk_adjusted_return: float
    tail_risk: float
    uncertainty: float
    factor_category: str
    regime_suitability: float
    rationale: str


class RiskBudgetingEngine:
    """
    Portfolio optimization engine using risk budgeting principles
    """

    def __init__(self, config: Dict, factor_engine=None):
        """Initialize risk budgeting engine"""
        self.config = config
        self.factor_engine = factor_engine

        # Load risk budgeting configuration
        rb_config = config.get('risk_budgeting', {})
        self.risk_config = RiskBudgetConfig(
            max_position_weight=rb_config.get('max_position_weight', 0.15),
            max_factor_concentration=rb_config.get('max_factor_concentration', 0.35),
            min_positions=rb_config.get('min_positions', 5),
            max_positions=rb_config.get('max_positions', 20),
            tail_risk_penalty_factor=rb_config.get('tail_risk_penalty_factor', 0.5),
            uncertainty_penalty_factor=rb_config.get('uncertainty_penalty_factor', 2.0),
            regime_allocation_multipliers=rb_config.get('regime_allocation_multipliers', {
                'bull': 1.0,
                'bear': 0.7,
                'neutral': 0.85
            })
        )

        print(f"💼 Risk Budgeting Engine initialized:")
        print(f"   Max position: {self.risk_config.max_position_weight:.1%}")
        print(f"   Max factor concentration: {self.risk_config.max_factor_concentration:.1%}")
        print(f"   Portfolio size: {self.risk_config.min_positions}-{self.risk_config.max_positions} positions")

    def calculate_risk_adjusted_scores(self,
                                     recommendations: pd.DataFrame,
                                     current_regime: str = 'neutral') -> pd.DataFrame:
        """
        Calculate risk-adjusted scores for position sizing

        Risk-adjusted score = Expected Return / (1 + Uncertainty + Tail Risk Penalty)
        """

        if recommendations.empty:
            return recommendations

        scored_recs = recommendations.copy()

        # Calculate base risk-adjusted return (Sharpe-like ratio)
        scored_recs['base_score'] = scored_recs['expected_return'] / (
            1 + scored_recs['uncertainty'] * self.risk_config.uncertainty_penalty_factor
        )

        # Apply tail risk penalties
        if 'tail_risk' in scored_recs.columns:
            tail_penalty = scored_recs['tail_risk'] * self.risk_config.tail_risk_penalty_factor
            scored_recs['tail_risk_penalty'] = tail_penalty
        else:
            scored_recs['tail_risk_penalty'] = 0

        # Apply factor profile regime suitability
        scored_recs['regime_multiplier'] = 1.0
        if self.factor_engine and self.factor_engine.enabled:
            for idx, row in scored_recs.iterrows():
                profile = self.factor_engine.get_stock_profile(row['ticker'])
                if profile:
                    regime_multipliers = profile.regime_multipliers
                    multiplier = regime_multipliers.get(current_regime, 1.0)
                    scored_recs.loc[idx, 'regime_multiplier'] = multiplier

        # Final risk-adjusted score
        scored_recs['risk_adjusted_score'] = (
            scored_recs['base_score'] *
            scored_recs['regime_multiplier'] -
            scored_recs['tail_risk_penalty']
        )

        # Ensure scores are positive for ranking
        scored_recs['risk_adjusted_score'] = scored_recs['risk_adjusted_score'].clip(lower=0.001)

        return scored_recs.sort_values('risk_adjusted_score', ascending=False)

    def apply_diversification_constraints(self,
                                        ranked_recommendations: pd.DataFrame) -> pd.DataFrame:
        """
        Apply factor diversification constraints to prevent over-concentration
        """

        if ranked_recommendations.empty:
            return ranked_recommendations

        # Add factor categories if factor engine is available
        if self.factor_engine and self.factor_engine.enabled:
            ranked_recommendations['factor_category'] = ranked_recommendations['ticker'].apply(
                lambda t: self._get_factor_category(t)
            )
        else:
            ranked_recommendations['factor_category'] = 'unclassified'

        # Track factor allocation
        factor_allocations = {}
        selected_positions = []

        for idx, row in ranked_recommendations.iterrows():
            ticker = row['ticker']
            factor_category = row['factor_category']

            # Check if we've hit position limits
            if len(selected_positions) >= self.risk_config.max_positions:
                break

            # Check factor concentration constraint
            current_factor_allocation = factor_allocations.get(factor_category, 0.0)

            if current_factor_allocation < self.risk_config.max_factor_concentration:
                selected_positions.append(idx)
                # Estimate allocation (will be refined in weight calculation)
                estimated_weight = min(
                    self.risk_config.max_position_weight,
                    row['risk_adjusted_score'] * 0.1  # Rough estimate
                )
                factor_allocations[factor_category] = current_factor_allocation + estimated_weight

        # Ensure minimum diversification
        if len(selected_positions) < self.risk_config.min_positions:
            # Add more positions even if factor concentration is exceeded
            remaining_positions = ranked_recommendations.index.difference(selected_positions)
            needed = self.risk_config.min_positions - len(selected_positions)
            additional_positions = remaining_positions[:needed]
            selected_positions.extend(additional_positions)

        return ranked_recommendations.loc[selected_positions]

    def calculate_optimal_weights(self,
                                diversified_recommendations: pd.DataFrame,
                                current_regime: str = 'neutral') -> pd.DataFrame:
        """
        Calculate optimal position weights using risk budgeting principles
        """

        if diversified_recommendations.empty:
            return diversified_recommendations

        # Apply regime allocation constraint
        regime_multiplier = self.risk_config.regime_allocation_multipliers.get(current_regime, 1.0)
        total_allocation_budget = regime_multiplier

        # Calculate raw weights proportional to risk-adjusted scores
        total_score = diversified_recommendations['risk_adjusted_score'].sum()
        raw_weights = diversified_recommendations['risk_adjusted_score'] / total_score * total_allocation_budget

        # Apply position limits
        capped_weights = raw_weights.clip(upper=self.risk_config.max_position_weight)

        # Normalize to fit budget after capping
        weight_sum = capped_weights.sum()
        if weight_sum > total_allocation_budget:
            # Scale down proportionally
            final_weights = capped_weights * (total_allocation_budget / weight_sum)
        else:
            final_weights = capped_weights

        # Update the dataframe
        result = diversified_recommendations.copy()
        result['portfolio_weight'] = final_weights

        # Generate position sizing rationale
        result['sizing_rationale'] = result.apply(
            lambda row: self._generate_sizing_rationale(row, current_regime), axis=1
        )

        return result

    def optimize_portfolio(self,
                          recommendations: pd.DataFrame,
                          current_regime: str = 'neutral') -> Tuple[pd.DataFrame, Dict]:
        """
        Main portfolio optimization function

        Returns:
            - Optimized recommendations with position weights
            - Optimization diagnostics
        """

        if recommendations.empty:
            return recommendations, {"error": "No recommendations to optimize"}

        print(f"\n💼 Risk Budgeting Optimization (Regime: {current_regime.title()})")
        print(f"   Input: {len(recommendations)} recommendations")

        # Step 1: Calculate risk-adjusted scores
        scored_recs = self.calculate_risk_adjusted_scores(recommendations, current_regime)
        print(f"   Risk-adjusted scoring complete")

        # Step 2: Apply diversification constraints
        diversified_recs = self.apply_diversification_constraints(scored_recs)
        print(f"   Selected {len(diversified_recs)} positions after diversification constraints")

        # Step 3: Calculate optimal weights
        optimized_recs = self.calculate_optimal_weights(diversified_recs, current_regime)

        # Generate diagnostics
        diagnostics = self._generate_optimization_diagnostics(
            recommendations, optimized_recs, current_regime
        )

        print(f"   Final allocation: {optimized_recs['portfolio_weight'].sum():.1%} of capital")
        print(f"   Position count: {len(optimized_recs)}")

        # Show top allocations
        top_positions = optimized_recs.nlargest(3, 'portfolio_weight')
        for _, pos in top_positions.iterrows():
            print(f"   {pos['ticker']}: {pos['portfolio_weight']:.1%} ({pos['factor_category']})")

        return optimized_recs, diagnostics

    def _get_factor_category(self, ticker: str) -> str:
        """Get factor category for a ticker"""
        if not self.factor_engine or not self.factor_engine.enabled:
            return 'unclassified'

        category = self.factor_engine.stock_to_profile.get(ticker, 'unclassified')
        return category

    def _generate_sizing_rationale(self, row: pd.Series, current_regime: str) -> str:
        """Generate human-readable rationale for position sizing"""
        weight = row['portfolio_weight']
        score = row['risk_adjusted_score']
        expected_return = row['expected_return']
        uncertainty = row['uncertainty']

        rationale = f"{weight:.1%} weight based on "

        if score > 0.1:
            rationale += "high risk-adjusted return"
        elif score > 0.05:
            rationale += "moderate risk-adjusted return"
        else:
            rationale += "low risk-adjusted return"

        if uncertainty > 0.3:
            rationale += ", high uncertainty penalty"
        elif uncertainty < 0.1:
            rationale += ", low uncertainty"

        if 'tail_risk_penalty' in row and row['tail_risk_penalty'] > 0.01:
            rationale += ", tail risk penalty applied"

        rationale += f" in {current_regime} regime"

        return rationale

    def _generate_optimization_diagnostics(self,
                                         original_recs: pd.DataFrame,
                                         optimized_recs: pd.DataFrame,
                                         current_regime: str) -> Dict:
        """Generate comprehensive optimization diagnostics"""

        diagnostics = {
            'regime': current_regime,
            'input_count': len(original_recs),
            'output_count': len(optimized_recs),
            'total_allocation': optimized_recs['portfolio_weight'].sum() if not optimized_recs.empty else 0,
            'max_position': optimized_recs['portfolio_weight'].max() if not optimized_recs.empty else 0,
            'min_position': optimized_recs['portfolio_weight'].min() if not optimized_recs.empty else 0,
        }

        # Factor diversification analysis
        if not optimized_recs.empty and 'factor_category' in optimized_recs.columns:
            factor_allocation = optimized_recs.groupby('factor_category')['portfolio_weight'].sum()
            diagnostics['factor_diversification'] = factor_allocation.to_dict()
            diagnostics['max_factor_concentration'] = factor_allocation.max()
            diagnostics['factor_count'] = len(factor_allocation)

        # Risk metrics
        if not optimized_recs.empty:
            diagnostics['portfolio_expected_return'] = (
                optimized_recs['expected_return'] * optimized_recs['portfolio_weight']
            ).sum()

            diagnostics['portfolio_uncertainty'] = np.sqrt((
                (optimized_recs['uncertainty'] * optimized_recs['portfolio_weight']) ** 2
            ).sum())

        return diagnostics

    def get_optimization_summary(self, optimized_recommendations: pd.DataFrame,
                               diagnostics: Dict) -> str:
        """Generate human-readable optimization summary"""

        if optimized_recommendations.empty:
            return "🚫 No positions selected by risk budgeting optimization"

        summary = f"💼 **Risk Budgeting Summary ({diagnostics['regime'].title()} Regime)**\n\n"

        summary += f"**Portfolio Allocation:**\n"
        summary += f"- Total capital allocated: {diagnostics['total_allocation']:.1%}\n"
        summary += f"- Number of positions: {diagnostics['output_count']}\n"
        summary += f"- Largest position: {diagnostics['max_position']:.1%}\n"
        summary += f"- Portfolio expected return: {diagnostics.get('portfolio_expected_return', 0):.2%}\n\n"

        # Factor diversification
        if 'factor_diversification' in diagnostics:
            summary += f"**Factor Diversification:**\n"
            for factor, allocation in diagnostics['factor_diversification'].items():
                summary += f"- {factor.replace('_', ' ').title()}: {allocation:.1%}\n"
            summary += f"- Max factor concentration: {diagnostics['max_factor_concentration']:.1%}\n\n"

        # Top positions
        summary += f"**Top Positions:**\n"
        top_positions = optimized_recommendations.nlargest(5, 'portfolio_weight')
        for _, pos in top_positions.iterrows():
            summary += f"- **{pos['ticker']}**: {pos['portfolio_weight']:.1%} "
            summary += f"(E[r]: {pos['expected_return']:.2%}, Score: {pos['risk_adjusted_score']:.3f})\n"
            if 'sizing_rationale' in pos:
                summary += f"  *{pos['sizing_rationale']}*\n"

        return summary