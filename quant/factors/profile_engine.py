"""
Stock-Specific Factor Profile Engine

This module implements stock-specific factor weighting based on industry,
geography, and fundamental characteristics. Instead of applying universal
signal weights to all stocks, this system recognizes that different stocks
are driven by different fundamental factors.

Key Features:
- Stock categorization by industry/theme
- Factor-specific weight adjustments per category
- Regime-sensitive multipliers per stock type
- Dynamic universe filtering based on conviction thresholds
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class FactorProfile:
    """Profile defining factor sensitivities for a stock category"""
    category: str
    description: str
    stocks: List[str]
    factor_weights: Dict[str, float]
    regime_multipliers: Dict[str, float]


class StockFactorProfileEngine:
    """
    Engine for applying stock-specific factor profiles to Bayesian signals
    """

    def __init__(self, config: Dict):
        """Initialize with factor profile configuration"""
        self.config = config
        self.profiles_config = config.get('stock_factor_profiles', {})
        self.enabled = self.profiles_config.get('enabled', False)

        if not self.enabled:
            print("📊 Factor profiles disabled - using universal weights")
            return

        # Parse factor profiles from config
        self.profiles = self._load_profiles()
        self.stock_to_profile = self._build_stock_mapping()

        # Dynamic filtering settings
        filtering_config = self.profiles_config.get('dynamic_filtering', {})
        self.cost_threshold = filtering_config.get('cost_threshold', 0.003)
        self.conviction_threshold = filtering_config.get('conviction_threshold', 0.6)
        self.max_positions = filtering_config.get('max_positions', 20)
        self.min_liquidity_mcap = filtering_config.get('min_liquidity_mcap', 1000)

        print(f"📊 Loaded {len(self.profiles)} factor profiles covering {len(self.stock_to_profile)} stocks")
        print(f"🎯 Dynamic filtering: {self.cost_threshold:.1%} min return, {self.conviction_threshold:.1%} min conviction")

    def _load_profiles(self) -> Dict[str, FactorProfile]:
        """Load factor profiles from configuration"""
        profiles = {}
        categories_config = self.profiles_config.get('categories', {})

        for category_name, category_config in categories_config.items():
            profile = FactorProfile(
                category=category_name,
                description=category_config.get('description', ''),
                stocks=category_config.get('stocks', []),
                factor_weights=category_config.get('factor_weights', {}),
                regime_multipliers=category_config.get('regime_multipliers', {})
            )
            profiles[category_name] = profile

        return profiles

    def _build_stock_mapping(self) -> Dict[str, str]:
        """Build mapping from stock ticker to profile category"""
        stock_to_profile = {}

        for category_name, profile in self.profiles.items():
            for stock in profile.stocks:
                if stock in stock_to_profile:
                    print(f"⚠️ Stock {stock} appears in multiple profiles: {stock_to_profile[stock]} and {category_name}")
                stock_to_profile[stock] = category_name

        return stock_to_profile

    def get_stock_profile(self, ticker: str) -> Optional[FactorProfile]:
        """Get the factor profile for a specific stock"""
        if not self.enabled:
            return None

        category = self.stock_to_profile.get(ticker)
        if category:
            return self.profiles[category]
        return None

    def apply_factor_adjustments(self,
                                recommendations: pd.DataFrame,
                                current_regime: str = 'neutral') -> pd.DataFrame:
        """
        Apply stock-specific factor adjustments to recommendations

        Args:
            recommendations: DataFrame with Bayesian scores
            current_regime: Current market regime (bull/bear/neutral)

        Returns:
            DataFrame with factor-adjusted signals
        """

        if not self.enabled or recommendations.empty:
            return recommendations

        adjusted_recs = recommendations.copy()

        # Track adjustments for diagnostics
        adjustments_applied = []

        for idx, row in adjusted_recs.iterrows():
            ticker = row['ticker']
            profile = self.get_stock_profile(ticker)

            if profile is None:
                # No specific profile - use universal weights
                continue

            # Apply factor weight adjustments
            factor_weights = profile.factor_weights

            # Adjust momentum weight
            if 'momentum' in factor_weights:
                momentum_adjustment = factor_weights['momentum']
                adjusted_recs.loc[idx, 'momentum_weight'] *= momentum_adjustment

            # Adjust sentiment weight
            if 'sentiment' in factor_weights:
                sentiment_adjustment = factor_weights['sentiment']
                adjusted_recs.loc[idx, 'sentiment_weight'] *= sentiment_adjustment

            # Apply regime-specific multipliers
            regime_multipliers = profile.regime_multipliers
            if current_regime in regime_multipliers:
                regime_multiplier = regime_multipliers[current_regime]

                # Apply to expected return and probability
                adjusted_recs.loc[idx, 'expected_return'] *= regime_multiplier

                # Adjust probability towards regime expectation
                current_prob = adjusted_recs.loc[idx, 'prob_positive']
                if regime_multiplier > 1.0:
                    # Bull regime - boost probability
                    adjusted_recs.loc[idx, 'prob_positive'] = min(0.95, current_prob * regime_multiplier)
                elif regime_multiplier < 1.0:
                    # Bear regime - reduce probability
                    adjusted_recs.loc[idx, 'prob_positive'] = max(0.05, current_prob * regime_multiplier)

            adjustments_applied.append({
                'ticker': ticker,
                'profile': profile.category,
                'momentum_weight': factor_weights.get('momentum', 1.0),
                'sentiment_weight': factor_weights.get('sentiment', 1.0),
                'regime_multiplier': regime_multipliers.get(current_regime, 1.0)
            })

        # Recalculate decision confidence after adjustments
        adjusted_recs['decision_confidence'] = self._recalculate_confidence(adjusted_recs)

        print(f"📊 Applied factor adjustments to {len(adjustments_applied)} stocks in {current_regime} regime")

        return adjusted_recs

    def _recalculate_confidence(self, recommendations: pd.DataFrame) -> pd.Series:
        """Recalculate decision confidence after factor adjustments"""
        # Simple confidence based on probability distance from 0.5 and expected return magnitude
        prob_confidence = abs(recommendations['prob_positive'] - 0.5) * 2
        return_confidence = abs(recommendations['expected_return']) * 100

        # Combine both measures
        combined_confidence = (prob_confidence + return_confidence) / 2
        return combined_confidence.clip(0, 1)

    def filter_investable_universe(self, recommendations: pd.DataFrame) -> pd.DataFrame:
        """
        Apply dynamic filtering to focus on highest conviction opportunities

        Filters based on:
        1. Cost threshold (expected return > transaction costs)
        2. Conviction threshold (decision confidence)
        3. Position limit (max portfolio positions)
        """

        if not self.enabled or recommendations.empty:
            return recommendations

        # Apply basic filters
        filtered = recommendations[
            (recommendations['expected_return'] > self.cost_threshold) &
            (recommendations['decision_confidence'] > self.conviction_threshold) &
            (recommendations['prob_positive'] > 0.55)  # Basic Bayesian threshold
        ].copy()

        if len(filtered) == 0:
            print(f"⚠️ No stocks passed filtering criteria (cost: {self.cost_threshold:.1%}, conviction: {self.conviction_threshold:.1%})")
            return pd.DataFrame()

        # Focus on top opportunities by conviction
        if len(filtered) > self.max_positions:
            filtered = filtered.nlargest(self.max_positions, 'decision_confidence')

        print(f"🎯 Filtered to {len(filtered)} investable opportunities from {len(recommendations)} total")

        # Show filtering breakdown by profile
        if 'profile_category' not in filtered.columns:
            # Add profile category for diagnostics
            filtered['profile_category'] = filtered['ticker'].apply(
                lambda t: self.stock_to_profile.get(t, 'unclassified')
            )

        profile_breakdown = filtered['profile_category'].value_counts()
        print("📈 Investable universe by category:")
        for category, count in profile_breakdown.items():
            print(f"   {category}: {count} stocks")

        return filtered.drop('profile_category', axis=1, errors='ignore')

    def get_profile_diagnostics(self) -> pd.DataFrame:
        """Return diagnostics about factor profiles and stock coverage"""
        if not self.enabled:
            return pd.DataFrame()

        diagnostics = []

        for category_name, profile in self.profiles.items():
            diagnostics.append({
                'category': category_name,
                'description': profile.description,
                'stock_count': len(profile.stocks),
                'momentum_weight': profile.factor_weights.get('momentum', 1.0),
                'sentiment_weight': profile.factor_weights.get('sentiment', 1.0),
                'bull_multiplier': profile.regime_multipliers.get('bull', 1.0),
                'bear_multiplier': profile.regime_multipliers.get('bear', 1.0),
                'stocks': ', '.join(profile.stocks[:3]) + ('...' if len(profile.stocks) > 3 else '')
            })

        return pd.DataFrame(diagnostics)

    def explain_stock_factors(self, ticker: str) -> str:
        """Generate explanation of factor profile for a specific stock"""
        profile = self.get_stock_profile(ticker)

        if profile is None:
            return f"📊 {ticker}: Using universal factor weights (no specific profile)"

        explanation = f"📊 **{ticker}** - {profile.category.replace('_', ' ').title()}\n"
        explanation += f"   {profile.description}\n\n"

        explanation += "**Factor Weights:**\n"
        for factor, weight in profile.factor_weights.items():
            direction = "stronger" if weight > 1.0 else "weaker" if weight < 1.0 else "normal"
            explanation += f"   • {factor.replace('_', ' ').title()}: {weight:.1f}x ({direction})\n"

        explanation += "\n**Regime Sensitivity:**\n"
        for regime, multiplier in profile.regime_multipliers.items():
            direction = "amplified" if multiplier > 1.0 else "dampened" if multiplier < 1.0 else "neutral"
            explanation += f"   • {regime.title()}: {multiplier:.1f}x ({direction})\n"

        return explanation