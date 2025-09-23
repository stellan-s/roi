import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

class MarketRegime(Enum):
    """Market regimes with clear economic definitions."""
    BULL = "bull"       # Upward trend, low volatility, positive sentiment
    BEAR = "bear"       # Downward trend, high volatility, negative sentiment
    NEUTRAL = "neutral" # Sideways trend, moderate volatility, mixed sentiment

@dataclass
class RegimeCharacteristics:
    """Characteristics for each market regime."""
    name: str
    description: str
    typical_return_range: Tuple[float, float]  # Daily return range
    volatility_level: str                      # Low/Moderate/High
    momentum_behavior: str                     # Description of momentum behaviour
    sentiment_bias: str                        # Sentiment archetype
    signal_adjustments: Dict[str, float]       # Multipliers for signals

# Regime definitions based on empirical finance research
REGIME_DEFINITIONS = {
    MarketRegime.BULL: RegimeCharacteristics(
        name="Bull Market",
        description="Strong upward trend with low volatility and optimism",
        typical_return_range=(0.001, 0.008),  # 0.1% - 0.8% daily
        volatility_level="Low",
        momentum_behavior="High persistence - momentum trends run longer",
        sentiment_bias="Positive bias - news is interpreted optimistically",
        signal_adjustments={
            "momentum": 1.3,    # Momentum works better in bull markets
            "trend": 1.2,       # Trend-following is strong
            "sentiment": 0.8    # Sentiment less important (baseline optimism)
        }
    ),

    MarketRegime.BEAR: RegimeCharacteristics(
        name="Bear Market",
        description="Downward trend with high volatility and pessimism",
        typical_return_range=(-0.008, -0.001), # -0.8% to -0.1% daily
        volatility_level="High",
        momentum_behavior="Fast reversals - momentum breaks more often",
        sentiment_bias="Negative bias - news is interpreted pessimistically",
        signal_adjustments={
            "momentum": 0.7,    # Momentum less reliable
            "trend": 1.1,       # Trend still matters
            "sentiment": 1.4    # Sentiment very important (fear/panic)
        }
    ),

    MarketRegime.NEUTRAL: RegimeCharacteristics(
        name="Neutral Market",
        description="Sideways movement with moderate volatility and mixed sentiment",
        typical_return_range=(-0.002, 0.002), # -0.2% to +0.2% daily
        volatility_level="Moderate",
        momentum_behavior="Weak persistence - mean reversion dominates",
        sentiment_bias="Mixed - sentiment remains balanced",
        signal_adjustments={
            "momentum": 0.9,    # Softer momentum
            "trend": 0.8,       # Weaker trend-following
            "sentiment": 1.1    # Sentiment slightly more useful for timing
        }
    )
}

class RegimeDetector:
    """
    Market regime detector combining HMM-style transitions and heuristics.

    Combines:
    1. Hidden Markov Model intuition for latent regime states
    2. Heuristic rules based on volatility, returns, and sentiment
    3. Adaptive regime transitions with hysteresis
    """

    def __init__(self,
                 config: Optional[Dict] = None,
                 lookback_days: int = 60,
                 volatility_window: int = 20,
                 trend_window: int = 50):
        # Use configuration overrides when available
        if config and 'regime_detection' in config:
            regime_config = config['regime_detection']
            self.lookback_days = regime_config.get('lookback_days', lookback_days)
            self.volatility_window = regime_config.get('volatility_window', volatility_window)
            self.trend_window = regime_config.get('trend_window', trend_window)
            self.transition_persistence = regime_config.get('transition_persistence', 0.80)

            # Thresholds from the configuration
            thresholds = regime_config.get('thresholds', {})
            self.vol_low_threshold = thresholds.get('volatility_low', 0.15)
            self.vol_high_threshold = thresholds.get('volatility_high', 0.25)
            self.return_bull_threshold = thresholds.get('return_bull', 0.002)
            self.return_bear_threshold = thresholds.get('return_bear', -0.002)
            self.drawdown_bear_threshold = thresholds.get('drawdown_bear', -0.10)

            # VIX integration configuration
            self.vix_config = regime_config.get('vix_integration', {})
        else:
            # Fallback to default values
            self.lookback_days = lookback_days
            self.volatility_window = volatility_window
            self.trend_window = trend_window
            self.transition_persistence = 0.80
            self.vol_low_threshold = 0.15
            self.vol_high_threshold = 0.25
            self.return_bull_threshold = 0.002
            self.return_bear_threshold = -0.002
            self.drawdown_bear_threshold = -0.10

            # Default VIX configuration
            self.vix_config = {}

        # HMM-style transition probabilities (simplified)
        # Regimes tend to persist and should not flip too often
        persist = self.transition_persistence
        switch_prob = (1.0 - persist) / 2  # Distribute remaining probability across other states
        direct_switch = switch_prob * 0.3   # Lower likelihood for a direct bull->bear jump

        self.transition_matrix = {
            MarketRegime.BULL: {
                MarketRegime.BULL: persist,
                MarketRegime.NEUTRAL: switch_prob - direct_switch,
                MarketRegime.BEAR: direct_switch
            },
            MarketRegime.BEAR: {
                MarketRegime.BEAR: persist,
                MarketRegime.NEUTRAL: switch_prob - direct_switch,
                MarketRegime.BULL: direct_switch
            },
            MarketRegime.NEUTRAL: {
                MarketRegime.NEUTRAL: persist * 0.9,  # Neutral is slightly less persistent
                MarketRegime.BULL: (1.0 - persist * 0.9) * 0.6,
                MarketRegime.BEAR: (1.0 - persist * 0.9) * 0.4
            }
        }

        # Regime history for smoothing
        self.regime_history: List[MarketRegime] = []
        self.regime_probabilities_history: List[Dict[MarketRegime, float]] = []

    def compute_market_features(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Compute market features for regime classification.

        Features:
        - Returns (multiple horizons)
        - Volatility (realised volatility)
        - Trend strength (SMA crossovers)
        - Drawdown measures
        """

        # Assume prices has columns: date, ticker, close
        # Focus on index-like behaviour by averaging across tickers

        # Debug: check what columns are actually present
        if 'close' not in prices.columns:
            print(f"⚠️ Debug: prices columns are {list(prices.columns)}")
            # Try to find a close price column with different casing
            close_col = None
            for col in prices.columns:
                if col.lower() in ['close', 'close_price', 'price']:
                    close_col = col
                    break
            if close_col is None:
                raise ValueError(f"No close price column found in prices. Available columns: {list(prices.columns)}")
            print(f"✅ Using column '{close_col}' as close price")
        else:
            close_col = 'close'

        market_data = prices.groupby('date')[close_col].mean().sort_index()

        features = pd.DataFrame(index=market_data.index)
        features['price'] = market_data

        # Returns across multiple horizons
        features['ret_1d'] = market_data.pct_change()
        features['ret_5d'] = market_data.pct_change(5)
        features['ret_20d'] = market_data.pct_change(20)

        # Volatility (annualised realised volatility)
        features['vol_20d'] = features['ret_1d'].rolling(self.volatility_window).std() * np.sqrt(252)

        # Trend measures - dynamic windows based on available data
        available_days = len(market_data)
        sma_short_window = min(10, max(5, available_days // 6))
        sma_long_window = min(self.trend_window, max(10, available_days // 3))

        features['sma_short'] = market_data.rolling(sma_short_window).mean()
        features['sma_long'] = market_data.rolling(sma_long_window).mean()
        features['price_vs_sma_short'] = (market_data - features['sma_short']) / features['sma_short']
        features['price_vs_sma_long'] = (market_data - features['sma_long']) / features['sma_long']
        features['sma_slope'] = features['sma_short'].pct_change(min(5, available_days // 10))  # Dynamic slope window

        # Drawdown from recent highs (using a shorter window)
        rolling_max_window = min(60, max(20, available_days // 2))
        features['rolling_max'] = market_data.rolling(rolling_max_window).max()
        features['drawdown'] = (market_data - features['rolling_max']) / features['rolling_max']

        # Trend consistency (% positive days within the window)
        positive_days_window = min(20, max(5, available_days // 4))
        features['positive_days_pct'] = (features['ret_1d'] > 0).rolling(positive_days_window).mean()

        # Drop NaNs while keeping enough data
        features_clean = features.dropna()

        # If too little data remains, forward-fill specific columns
        if len(features_clean) < max(10, available_days // 3):
            print(f"⚠️ Few regime features ({len(features_clean)} rows), applying forward fill")
            features = features.fillna(method='ffill').dropna()
        else:
            features = features_clean

        return features

    def classify_regime_heuristic(self, features: pd.Series, vix_data: Optional[pd.DataFrame] = None) -> Dict[MarketRegime, float]:
        """
        Heuristic regime classification based on market features plus VIX.
        Returns probabilities for each regime.
        """

        # Feature thresholds (configurable)
        vol_low_threshold = self.vol_low_threshold
        vol_high_threshold = self.vol_high_threshold
        return_bull_threshold = self.return_bull_threshold
        return_bear_threshold = self.return_bear_threshold
        drawdown_bear_threshold = self.drawdown_bear_threshold

        scores = {regime: 0.0 for regime in MarketRegime}

        # VIX-based regime scoring (if available and enabled)
        if (vix_data is not None and not vix_data.empty and
            self.vix_config.get('enabled', False)):
            try:
                # Get latest VIX regime
                latest_vix = vix_data.iloc[-1]
                vix_regime = latest_vix['vix_regime']
                vix_level = latest_vix['vix_close']

                # Get VIX configuration parameters
                influence_weight = self.vix_config.get('influence_weight', 0.4)
                override_threshold = self.vix_config.get('override_threshold', 18.0)
                override_strength = self.vix_config.get('override_strength', 0.8)
                regime_multipliers = self.vix_config.get('regime_multipliers', {})
                momentum_adjustments = self.vix_config.get('momentum_adjustments', {})

                # Apply base influence weight to all VIX contributions
                base_influence = influence_weight

                # VIX regime mapping to market regimes using configurable multipliers
                if vix_regime == 'low_fear':  # VIX < 20
                    low_fear_config = regime_multipliers.get('low_fear', {})
                    scores[MarketRegime.BULL] += base_influence * low_fear_config.get('bull_boost', 0.6)
                    scores[MarketRegime.BEAR] += base_influence * low_fear_config.get('bear_penalty', -0.3)
                    scores[MarketRegime.NEUTRAL] += base_influence * low_fear_config.get('neutral_boost', 0.1)
                elif vix_regime == 'moderate_fear':  # VIX 20-30
                    moderate_fear_config = regime_multipliers.get('moderate_fear', {})
                    scores[MarketRegime.BULL] += base_influence * moderate_fear_config.get('bull_boost', 0.1)
                    scores[MarketRegime.BEAR] += base_influence * moderate_fear_config.get('bear_penalty', 0.0)
                    scores[MarketRegime.NEUTRAL] += base_influence * moderate_fear_config.get('neutral_boost', 0.4)
                elif vix_regime == 'high_fear':  # VIX 30-40
                    high_fear_config = regime_multipliers.get('high_fear', {})
                    scores[MarketRegime.BULL] += base_influence * high_fear_config.get('bull_boost', -0.2)
                    scores[MarketRegime.BEAR] += base_influence * high_fear_config.get('bear_penalty', 0.0)
                    scores[MarketRegime.NEUTRAL] += base_influence * high_fear_config.get('neutral_boost', 0.2)
                elif vix_regime == 'extreme_fear':  # VIX > 40
                    extreme_fear_config = regime_multipliers.get('extreme_fear', {})
                    scores[MarketRegime.BULL] += base_influence * extreme_fear_config.get('bull_boost', -0.4)
                    scores[MarketRegime.BEAR] += base_influence * extreme_fear_config.get('bear_penalty', 0.0)
                    scores[MarketRegime.NEUTRAL] += base_influence * extreme_fear_config.get('neutral_boost', 0.1)

                # VIX momentum adjustment using configurable thresholds
                if 'vix_momentum_5d' in latest_vix:
                    vix_momentum = latest_vix['vix_momentum_5d']
                    spike_threshold = momentum_adjustments.get('spike_threshold', 0.2)
                    drop_threshold = momentum_adjustments.get('drop_threshold', -0.2)
                    spike_bear_boost = momentum_adjustments.get('spike_bear_boost', 0.2)
                    drop_bull_boost = momentum_adjustments.get('drop_bull_boost', 0.2)

                    if vix_momentum > spike_threshold:  # VIX spiking = fear increasing
                        scores[MarketRegime.BEAR] += base_influence * spike_bear_boost
                        scores[MarketRegime.BULL] = max(0, scores[MarketRegime.BULL] - base_influence * spike_bear_boost)
                    elif vix_momentum < drop_threshold:  # VIX falling = fear decreasing
                        scores[MarketRegime.BULL] += base_influence * drop_bull_boost
                        scores[MarketRegime.BEAR] = max(0, scores[MarketRegime.BEAR] - base_influence * drop_bull_boost)

                # VIX override logic for very low VIX levels
                if vix_level < override_threshold:
                    override_boost = override_strength * base_influence
                    scores[MarketRegime.BULL] += override_boost
                    scores[MarketRegime.BEAR] = max(0, scores[MarketRegime.BEAR] - override_boost * 0.5)

                print(f"🔍 VIX regime: {vix_regime} (level: {vix_level:.1f}, influence: {base_influence:.1f}) - Bull: +{scores[MarketRegime.BULL]:.2f}, Bear: +{scores[MarketRegime.BEAR]:.2f}")

            except Exception as e:
                print(f"⚠️ Error processing VIX data in regime classification: {e}")
                # Continue without VIX

        # Volatility-based classification
        vol = features['vol_20d']
        if vol < vol_low_threshold:
            scores[MarketRegime.BULL] += 0.3
            scores[MarketRegime.NEUTRAL] += 0.1
        elif vol > vol_high_threshold:
            scores[MarketRegime.BEAR] += 0.4
            scores[MarketRegime.NEUTRAL] += 0.1
        else:
            scores[MarketRegime.NEUTRAL] += 0.3

        # Return-based classification
        ret_20d = features['ret_20d']
        if ret_20d > return_bull_threshold * 20:  # 20-day cumulative
            scores[MarketRegime.BULL] += 0.4
        elif ret_20d < return_bear_threshold * 20:
            scores[MarketRegime.BEAR] += 0.4
        else:
            scores[MarketRegime.NEUTRAL] += 0.3

        # Trend-based classification (uses updated column names)
        price_vs_sma = features['price_vs_sma_long']
        sma_slope = features['sma_slope']
        if price_vs_sma > 0.02 and sma_slope > 0.001:  # Above SMA + rising
            scores[MarketRegime.BULL] += 0.3
        elif price_vs_sma < -0.02 and sma_slope < -0.001:  # Below SMA + falling
            scores[MarketRegime.BEAR] += 0.3
        else:
            scores[MarketRegime.NEUTRAL] += 0.2

        # Drawdown-based (bear market signal)
        drawdown = features['drawdown']
        if drawdown < drawdown_bear_threshold:
            scores[MarketRegime.BEAR] += 0.5
            scores[MarketRegime.BULL] = max(0, scores[MarketRegime.BULL] - 0.3)

        # Consistency check (trend persistence)
        positive_days = features['positive_days_pct']
        if positive_days > 0.65:  # 65%+ positive days
            scores[MarketRegime.BULL] += 0.2
        elif positive_days < 0.35:  # 35%- positive days
            scores[MarketRegime.BEAR] += 0.2
        else:
            scores[MarketRegime.NEUTRAL] += 0.2

        # Normalise to probabilities
        total_score = sum(scores.values())
        if total_score > 0:
            probabilities = {regime: score/total_score for regime, score in scores.items()}
        else:
            # Default to a uniform distribution if scores are zero
            probabilities = {regime: 1/3 for regime in MarketRegime}

        return probabilities

    def apply_transition_smoothing(self,
                                  current_probs: Dict[MarketRegime, float],
                                  previous_regime: Optional[MarketRegime] = None) -> Dict[MarketRegime, float]:
        """
        Apply the transition matrix to smooth regime changes.
        Regimes should be sticky and not flip too quickly.
        """

        if previous_regime is None or len(self.regime_history) == 0:
            return current_probs

        # Combine evidence with transition probabilities
        smoothed_probs = {}

        for regime in MarketRegime:
            # Bayes update: P(regime|data) ∝ P(data|regime) * P(regime|previous)
            evidence = current_probs[regime]
            prior = self.transition_matrix[previous_regime][regime]

            smoothed_probs[regime] = evidence * prior

        # Normalise
        total = sum(smoothed_probs.values())
        if total > 0:
            smoothed_probs = {k: v/total for k, v in smoothed_probs.items()}
        else:
            smoothed_probs = current_probs

        return smoothed_probs

    def detect_regime(self, prices: pd.DataFrame, vix_data: Optional[pd.DataFrame] = None) -> Tuple[MarketRegime, Dict[MarketRegime, float], Dict]:
        """
        Primary regime-detection function with optional VIX integration.

        Args:
            prices: Price data for regime detection
            vix_data: Optional VIX data for enhanced regime classification

        Returns:
        - Most likely regime
        - Probabilities for all regimes
        - Diagnostic information
        """

        # Compute market features
        features = self.compute_market_features(prices)

        if features.empty:
            # Default to neutral if no data is available
            default_probs = {regime: 1/3 for regime in MarketRegime}
            return MarketRegime.NEUTRAL, default_probs, {"error": "No market data"}

        # Use the latest observation
        latest_features = features.iloc[-1]

        # Heuristic classification with VIX integration
        raw_probabilities = self.classify_regime_heuristic(latest_features, vix_data)

        # Apply transition smoothing
        previous_regime = self.regime_history[-1] if self.regime_history else None
        smoothed_probabilities = self.apply_transition_smoothing(raw_probabilities, previous_regime)

        # Select the most likely regime
        most_likely_regime = max(smoothed_probabilities, key=smoothed_probabilities.get)

        # Diagnostics
        diagnostics = {
            "market_features": {
                "volatility_20d": latest_features['vol_20d'],
                "return_20d": latest_features['ret_20d'],
                "price_vs_sma_long": latest_features['price_vs_sma_long'],
                "drawdown": latest_features['drawdown'],
                "positive_days_pct": latest_features['positive_days_pct']
            },
            "raw_probabilities": raw_probabilities,
            "smoothed_probabilities": smoothed_probabilities,
            "previous_regime": previous_regime.value if previous_regime else None,
            "regime_persistence": len([r for r in self.regime_history[-10:] if r == most_likely_regime]) / min(10, len(self.regime_history)) if self.regime_history else 0
        }

        # Update history
        self.regime_history.append(most_likely_regime)
        self.regime_probabilities_history.append(smoothed_probabilities)

        # Limit history size
        if len(self.regime_history) > 100:
            self.regime_history = self.regime_history[-100:]
            self.regime_probabilities_history = self.regime_probabilities_history[-100:]

        return most_likely_regime, smoothed_probabilities, diagnostics

    def detect_current_regime(self, ticker_prices: pd.DataFrame, vix_data: Optional[pd.DataFrame] = None):
        """
        Convenience method for current regime detection with VIX support.

        Returns RegimeResult-like object for compatibility.
        """
        regime, probabilities, diagnostics = self.detect_regime(ticker_prices, vix_data)

        # Create a simple result object with required attributes
        class RegimeResult:
            def __init__(self, regime, confidence):
                self.regime = regime
                self.confidence = confidence

        confidence = probabilities[regime] if regime in probabilities else 0.33
        return RegimeResult(regime, confidence)

    def get_regime_adjustments(self, regime: MarketRegime) -> Dict[str, float]:
        """Return signal adjustments for the given regime."""
        return REGIME_DEFINITIONS[regime].signal_adjustments

    def get_regime_explanation(self,
                             regime: MarketRegime,
                             probabilities: Dict[MarketRegime, float],
                             diagnostics: Dict,
                             vix_data: Optional[pd.DataFrame] = None,
                             metals_data: Optional[pd.DataFrame] = None) -> str:
        """Generate a user-facing explanation of the regime classification."""

        regime_def = REGIME_DEFINITIONS[regime]
        prob_pct = probabilities[regime] * 100

        explanation = f"**{regime_def.name}** ({prob_pct:.0f}% confidence)\n"
        explanation += f"{regime_def.description}\n\n"

        # Market context (guard against missing diagnostics)
        features = diagnostics.get("market_features") if diagnostics else None
        if features is not None:
            # Handle both DataFrame and dict formats
            if hasattr(features, 'empty') and not features.empty:
                explanation += "**Market Context:**\n"
                explanation += f"- Volatility: {features.get('volatility_20d', 0.0)*100:.1f}% ({regime_def.volatility_level})\n"
                explanation += f"- 20-day return: {features.get('ret_20d', 0.0)*100:+.1f}%\n"
                explanation += f"- Position vs SMA-long: {features.get('price_vs_sma_long', 0.0)*100:+.1f}%\n"
            elif isinstance(features, dict) and features:
                explanation += "**Market Context:**\n"
                explanation += f"- Volatility: {features.get('volatility_20d', 0.0)*100:.1f}% ({regime_def.volatility_level})\n"
                explanation += f"- 20-day return: {features.get('ret_20d', 0.0)*100:+.1f}%\n"
                explanation += f"- Position vs SMA-long: {features.get('price_vs_sma_long', 0.0)*100:+.1f}%\n"
                explanation += f"- Drawdown from high: {features.get('drawdown', 0.0)*100:.1f}%\n"
                explanation += f"- Positive days (20d): {features.get('positive_days_pct', 0.0)*100:.0f}%\n\n"

            # Add remaining context for DataFrame format
            if hasattr(features, 'empty') and not features.empty:
                explanation += f"- Drawdown from high: {features.get('drawdown', 0.0)*100:.1f}%\n"
                explanation += f"- Positive days (20d): {features.get('positive_days_pct', 0.0)*100:.0f}%\n\n"
        else:
            explanation += "**Market Context:**\nNo market features available (insufficient data).\n\n"

        # VIX Analysis (if available)
        if vix_data is not None and not vix_data.empty:
            try:
                latest_vix = vix_data.iloc[-1]
                vix_regime = latest_vix['vix_regime']
                vix_level = latest_vix['vix_close']

                explanation += "**VIX Macro Analysis:**\n"
                explanation += f"- VIX Level: {vix_level:.1f} ({vix_regime.replace('_', ' ').title()})\n"

                # VIX regime interpretation
                vix_interp = {
                    'low_fear': 'Low fear – investors are confident',
                    'moderate_fear': 'Moderate concern – cautious optimism',
                    'high_fear': 'High concern – investors are nervous',
                    'extreme_fear': 'Extreme fear – market panic'
                }
                explanation += f"- Interpretation: {vix_interp.get(vix_regime, 'Unknown VIX regime')}\n"

                # VIX momentum if available
                if 'vix_momentum_5d' in latest_vix:
                    vix_momentum = latest_vix['vix_momentum_5d']
                    momentum_dir = "rising" if vix_momentum > 0.05 else "falling" if vix_momentum < -0.05 else "stable"
                    explanation += f"- VIX Trend (5d): {vix_momentum*100:+.1f}% ({momentum_dir})\n"

                # VIX influence on regime decision
                if self.vix_config.get('enabled', False):
                    influence = self.vix_config.get('influence_weight', 0.4)
                    explanation += f"- VIX Influence: {influence:.0%} of regime decision\n"

                explanation += "\n"

            except Exception as e:
                explanation += f"**VIX Analysis Error:** {e}\n\n"
        else:
            explanation += "**VIX Analysis:** Not available\n\n"

        # Precious Metals Sentiment Analysis (if available)
        if metals_data is not None and not metals_data.empty:
            try:
                latest_metals = metals_data.iloc[-1]
                gold_level = latest_metals['gold_close']
                silver_level = latest_metals['silver_close']
                gold_silver_ratio = latest_metals['gold_silver_ratio']
                metals_sentiment = latest_metals['metals_sentiment']

                explanation += "**Precious Metals Sentiment Analysis:**\n"
                explanation += f"- Gold (GLD): ${gold_level:.1f}\n"
                explanation += f"- Silver (SLV): ${silver_level:.1f}\n"
                explanation += f"- Gold/Silver Ratio: {gold_silver_ratio:.1f}\n"

                # Precious metals sentiment interpretation
                metals_interp = {
                    'risk_off_strong': 'Strong flight to gold – defensive positioning favored',
                    'risk_off_mild': 'Mild safe-haven demand – cautious sentiment',
                    'neutral': 'Balanced precious metals sentiment – no clear directional bias',
                    'risk_on_mild': 'Mild risk appetite – modest growth sentiment',
                    'risk_on_strong': 'Strong risk appetite – growth assets favored'
                }
                explanation += f"- Sentiment: {metals_interp.get(metals_sentiment, 'Unknown metals sentiment')}\n"

                # Gold momentum if available
                if 'gold_return_20d' in latest_metals:
                    gold_momentum = latest_metals['gold_return_20d']
                    momentum_dir = "rallying" if gold_momentum > 0.02 else "declining" if gold_momentum < -0.02 else "stable"
                    explanation += f"- Gold Trend (20d): {gold_momentum*100:+.1f}% ({momentum_dir})\n"

                explanation += "\n"

            except Exception as e:
                explanation += f"**Precious Metals Analysis Error:** {e}\n\n"

        # Factor analysis explaining regime detection logic
        explanation += "**Factor Analysis - How Each Component Affects Regime Detection:**\n\n"

        # VIX factor analysis
        if vix_data is not None and not vix_data.empty:
            try:
                latest_vix = vix_data.iloc[-1]
                vix_level = latest_vix['vix_close']
                vix_influence_pct = 60  # From the VIX influence weight

                if vix_level < 15:
                    vix_effect = "🟢 **Low VIX** → Strong Bull bias (+0.3), Bear penalty (-0.2)"
                elif vix_level < 20:
                    vix_effect = "🟡 **Moderate VIX** → Neutral environment, balanced probabilities"
                elif vix_level < 30:
                    vix_effect = "🟠 **Elevated VIX** → Reduces Bull confidence, favors Neutral"
                elif vix_level < 40:
                    vix_effect = "🔴 **High VIX** → Bear bias (+0.2), Bull penalty (-0.2)"
                else:
                    vix_effect = "🚨 **Extreme VIX** → Strong Bear bias (+0.4), Major Bull penalty (-0.4)"

                explanation += f"- **VIX Impact ({vix_influence_pct}% weight):** {vix_effect}\n"

                if 'vix_momentum_5d' in latest_vix:
                    vix_momentum = latest_vix['vix_momentum_5d']
                    if abs(vix_momentum) > 0.2:
                        momentum_effect = "Rising fear → Bear boost (+0.2)" if vix_momentum > 0.2 else "Falling fear → Bull boost (+0.2)"
                        explanation += f"  - VIX Momentum: {momentum_effect}\n"

            except Exception:
                pass

        # Market technical factors (computed from diagnostics if available)
        explanation += "\n- **Technical Factors:**\n"
        if diagnostics:
            # Volatility analysis
            volatility = diagnostics.get('volatility', 0.3)
            if volatility < 0.2:
                vol_effect = "Low volatility → Bull boost (+0.3), some Neutral (+0.1)"
            elif volatility > 0.4:
                vol_effect = "High volatility → Bear boost (+0.4), some Neutral (+0.1)"
            else:
                vol_effect = "Moderate volatility → Neutral bias (+0.3)"
            explanation += f"  - Volatility: {vol_effect}\n"

            # Return analysis
            recent_return = diagnostics.get('ret_20d', 0.0)
            if recent_return > 0.04:  # >4% in 20 days
                return_effect = "Strong recent gains → Bull boost (+0.4)"
            elif recent_return < -0.04:  # <-4% in 20 days
                return_effect = "Recent losses → Bear boost (+0.4)"
            else:
                return_effect = "Sideways movement → Neutral bias (+0.3)"
            explanation += f"  - Recent Returns: {return_effect}\n"

            # Trend analysis
            position_vs_sma = diagnostics.get('price_vs_sma_long', 0.0)
            sma_slope = diagnostics.get('sma_slope', 0.0)
            if position_vs_sma > 0.02 and sma_slope > 0.001:
                trend_effect = "Above rising SMA → Bull boost (+0.3)"
            elif position_vs_sma < -0.02 and sma_slope < -0.001:
                trend_effect = "Below falling SMA → Bear boost (+0.3)"
            else:
                trend_effect = "Mixed trend signals → Neutral bias (+0.2)"
            explanation += f"  - Trend Position: {trend_effect}\n"

            # Drawdown analysis
            drawdown = diagnostics.get('drawdown', 0.0)
            if drawdown < -0.1:  # >10% drawdown
                dd_effect = "Significant drawdown → Bear boost (+0.5), Bull penalty (-0.3)"
                explanation += f"  - Drawdown: {dd_effect}\n"

            # Positive days consistency
            positive_days = diagnostics.get('positive_days_pct', 0.5)
            if positive_days > 0.65:
                consistency_effect = "High win rate (>65%) → Bull boost (+0.2)"
            elif positive_days < 0.35:
                consistency_effect = "Low win rate (<35%) → Bear boost (+0.2)"
            else:
                consistency_effect = "Mixed daily performance → Neutral bias (+0.2)"
            explanation += f"  - Daily Consistency: {consistency_effect}\n"

        # Precious metals factor analysis
        if metals_data is not None and not metals_data.empty:
            try:
                latest_metals = metals_data.iloc[-1]
                metals_sentiment = latest_metals['metals_sentiment']

                explanation += "\n- **Precious Metals Sentiment Impact:**\n"
                if metals_sentiment in ['risk_off_strong', 'risk_off_mild']:
                    metals_effect = "Flight to safety → Reduces risk appetite, favors defensive positioning"
                elif metals_sentiment in ['risk_on_strong', 'risk_on_mild']:
                    metals_effect = "Risk appetite strong → Growth assets favored, reduces safe-haven demand"
                else:
                    metals_effect = "Neutral metals sentiment → No clear directional bias"
                explanation += f"  - Gold/Silver Analysis: {metals_effect}\n"

            except Exception:
                pass

        explanation += "\n**Final Regime Decision Logic:**\n"
        explanation += "- Scores from all factors are weighted and combined\n"
        explanation += "- VIX provides the strongest signal (60% influence)\n"
        explanation += "- Technical factors validate or contradict VIX signals\n"
        explanation += "- Precious metals provide sentiment confirmation\n"
        explanation += "- Transition smoothing prevents regime oscillation\n\n"

        # Regime probability breakdown
        explanation += "**Regime Probabilities:**\n"
        for reg, prob in probabilities.items():
            explanation += f"- {reg.value.title()}: {prob*100:.0f}%\n"
        explanation += "\n"

        # Signal implications
        explanation += "**Signal Adjustments:**\n"
        for signal, multiplier in regime_def.signal_adjustments.items():
            direction = "amplified" if multiplier > 1.0 else "dampened" if multiplier < 1.0 else "unchanged"
            explanation += f"- {signal.title()}: {multiplier:.1f}x ({direction})\n"

        explanation += f"\n**Regime Characteristics:**\n"
        explanation += f"- *{regime_def.momentum_behavior}*\n"
        explanation += f"- *{regime_def.sentiment_bias}*"

        return explanation
