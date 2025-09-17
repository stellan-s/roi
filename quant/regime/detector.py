import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

class MarketRegime(Enum):
    """Marknadsregimer med tydliga definitioner"""
    BULL = "bull"       # Uppåtgående trend, låg volatilitet, positiv sentiment
    BEAR = "bear"       # Nedåtgående trend, hög volatilitet, negativ sentiment
    NEUTRAL = "neutral" # Sidledes trend, måttlig volatilitet, blandad sentiment

@dataclass
class RegimeCharacteristics:
    """Karakteristika för varje marknadsregim"""
    name: str
    description: str
    typical_return_range: Tuple[float, float]  # Daglig return range
    volatility_level: str                      # Låg/Måttlig/Hög
    momentum_behavior: str                     # Beskrivning av momentum
    sentiment_bias: str                        # Sentiment-karakteristik
    signal_adjustments: Dict[str, float]       # Multipliers för signaler

# Regime definitioner baserat på empirisk finansforskning
REGIME_DEFINITIONS = {
    MarketRegime.BULL: RegimeCharacteristics(
        name="Bull Market",
        description="Stark uppåtgående trend med låg volatilitet och optimism",
        typical_return_range=(0.001, 0.008),  # 0.1% - 0.8% daglig
        volatility_level="Låg",
        momentum_behavior="Stark persistens - momentum fortsätter längre",
        sentiment_bias="Positivt bias - nyheter tolkas optimistiskt",
        signal_adjustments={
            "momentum": 1.3,    # Momentum fungerar bättre i bull markets
            "trend": 1.2,       # Trend-following stark
            "sentiment": 0.8    # Sentiment mindre viktigt (redan optimistiskt)
        }
    ),

    MarketRegime.BEAR: RegimeCharacteristics(
        name="Bear Market",
        description="Nedåtgående trend med hög volatilitet och pessimism",
        typical_return_range=(-0.008, -0.001), # -0.8% - -0.1% daglig
        volatility_level="Hög",
        momentum_behavior="Snabba reversal - momentum bryter oftare",
        sentiment_bias="Negativt bias - nyheter tolkas pessimistiskt",
        signal_adjustments={
            "momentum": 0.7,    # Momentum mindre tillförlitligt
            "trend": 1.1,       # Trend fortfarande viktig
            "sentiment": 1.4    # Sentiment mycket viktigt (fear/panic)
        }
    ),

    MarketRegime.NEUTRAL: RegimeCharacteristics(
        name="Neutral Market",
        description="Sidledes rörelse med måttlig volatilitet och blandat sentiment",
        typical_return_range=(-0.002, 0.002), # -0.2% - +0.2% daglig
        volatility_level="Måttlig",
        momentum_behavior="Svag persistens - mean reversion dominerar",
        sentiment_bias="Blandat - sentiment mer balanserat",
        signal_adjustments={
            "momentum": 0.9,    # Svagare momentum
            "trend": 0.8,       # Svagare trend-following
            "sentiment": 1.1    # Sentiment något viktigare för timing
        }
    )
}

class RegimeDetector:
    """
    Marknadsregim-detektor baserad på HMM och heuristisk analys

    Kombinerar:
    1. Hidden Markov Model för latenta regim-states
    2. Heuristiska regler baserat på volatilitet, returns, sentiment
    3. Adaptive regime transitions med hysteresis
    """

    def __init__(self,
                 config: Optional[Dict] = None,
                 lookback_days: int = 60,
                 volatility_window: int = 20,
                 trend_window: int = 50):
        # Använd config värden om tillgängliga
        if config and 'regime_detection' in config:
            regime_config = config['regime_detection']
            self.lookback_days = regime_config.get('lookback_days', lookback_days)
            self.volatility_window = regime_config.get('volatility_window', volatility_window)
            self.trend_window = regime_config.get('trend_window', trend_window)
            self.transition_persistence = regime_config.get('transition_persistence', 0.80)

            # Thresholds från config
            thresholds = regime_config.get('thresholds', {})
            self.vol_low_threshold = thresholds.get('volatility_low', 0.15)
            self.vol_high_threshold = thresholds.get('volatility_high', 0.25)
            self.return_bull_threshold = thresholds.get('return_bull', 0.002)
            self.return_bear_threshold = thresholds.get('return_bear', -0.002)
            self.drawdown_bear_threshold = thresholds.get('drawdown_bear', -0.10)

            # VIX integration configuration
            self.vix_config = regime_config.get('vix_integration', {})
        else:
            # Fallback till default värden
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

        # HMM-liknande transition probabilities (simplified)
        # Regimer tenderar att persista - ändras inte för ofta
        persist = self.transition_persistence
        switch_prob = (1.0 - persist) / 2  # Fördela resten mellan andra states
        direct_switch = switch_prob * 0.3   # Mindre sannolikhet för direkt bull->bear

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
                MarketRegime.NEUTRAL: persist * 0.9,  # Neutral något mindre persistent
                MarketRegime.BULL: (1.0 - persist * 0.9) * 0.6,
                MarketRegime.BEAR: (1.0 - persist * 0.9) * 0.4
            }
        }

        # Regime history för smoothing
        self.regime_history: List[MarketRegime] = []
        self.regime_probabilities_history: List[Dict[MarketRegime, float]] = []

    def compute_market_features(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Beräkna marknadsfeatures för regime-klassificering

        Features:
        - Returns (olika tidshorisonter)
        - Volatilitet (realized vol)
        - Trend strength (olika SMA-crossovers)
        - Drawdown measures
        """

        # Antag prices har columns: date, ticker, close
        # Vi fokuserar på index-liknande behavior - tar medelvärde över tickers
        market_data = prices.groupby('date')['close'].mean().sort_index()

        features = pd.DataFrame(index=market_data.index)
        features['price'] = market_data

        # Returns på olika tidshorisonter
        features['ret_1d'] = market_data.pct_change()
        features['ret_5d'] = market_data.pct_change(5)
        features['ret_20d'] = market_data.pct_change(20)

        # Volatilitet (realized vol)
        features['vol_20d'] = features['ret_1d'].rolling(self.volatility_window).std() * np.sqrt(252)

        # Trend measures - dynamic window based on available data
        available_days = len(market_data)
        sma_short_window = min(10, max(5, available_days // 6))
        sma_long_window = min(self.trend_window, max(10, available_days // 3))

        features['sma_short'] = market_data.rolling(sma_short_window).mean()
        features['sma_long'] = market_data.rolling(sma_long_window).mean()
        features['price_vs_sma_short'] = (market_data - features['sma_short']) / features['sma_short']
        features['price_vs_sma_long'] = (market_data - features['sma_long']) / features['sma_long']
        features['sma_slope'] = features['sma_short'].pct_change(min(5, available_days // 10))  # Dynamic slope window

        # Drawdown från recent high (använd mindre window)
        rolling_max_window = min(60, max(20, available_days // 2))
        features['rolling_max'] = market_data.rolling(rolling_max_window).max()
        features['drawdown'] = (market_data - features['rolling_max']) / features['rolling_max']

        # Trend consistency (% positive days i period)
        positive_days_window = min(20, max(5, available_days // 4))
        features['positive_days_pct'] = (features['ret_1d'] > 0).rolling(positive_days_window).mean()

        # Drop NaN men behåll tillräckligt data
        features_clean = features.dropna()

        # Om för lite data kvar efter dropna, använd forward fill för vissa kolumner
        if len(features_clean) < max(10, available_days // 3):
            print(f"⚠️ Få regime features ({len(features_clean)} rows), använder forward fill")
            features = features.fillna(method='ffill').dropna()
        else:
            features = features_clean

        return features

    def classify_regime_heuristic(self, features: pd.Series, vix_data: Optional[pd.DataFrame] = None) -> Dict[MarketRegime, float]:
        """
        Heuristisk regime-klassificering baserat på market features + VIX
        Returnerar sannolikheter för varje regim
        """

        # Feature thresholds (konfigurerbara)
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

        # Trend-based classification (använd nya kolumnnamn)
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

        # Normalisera till sannolikheter
        total_score = sum(scores.values())
        if total_score > 0:
            probabilities = {regime: score/total_score for regime, score in scores.items()}
        else:
            # Default uniform distribution om inga scores
            probabilities = {regime: 1/3 for regime in MarketRegime}

        return probabilities

    def apply_transition_smoothing(self,
                                  current_probs: Dict[MarketRegime, float],
                                  previous_regime: Optional[MarketRegime] = None) -> Dict[MarketRegime, float]:
        """
        Applicera transition matrix för att smooth regime changes
        Regimer bör vara sticky - ändras inte för snabbt
        """

        if previous_regime is None or len(self.regime_history) == 0:
            return current_probs

        # Kombinera current evidence med transition probabilities
        smoothed_probs = {}

        for regime in MarketRegime:
            # Bayes update: P(regime|data) ∝ P(data|regime) * P(regime|previous)
            evidence = current_probs[regime]
            prior = self.transition_matrix[previous_regime][regime]

            smoothed_probs[regime] = evidence * prior

        # Normalisera
        total = sum(smoothed_probs.values())
        if total > 0:
            smoothed_probs = {k: v/total for k, v in smoothed_probs.items()}
        else:
            smoothed_probs = current_probs

        return smoothed_probs

    def detect_regime(self, prices: pd.DataFrame, vix_data: Optional[pd.DataFrame] = None) -> Tuple[MarketRegime, Dict[MarketRegime, float], Dict]:
        """
        Huvudfunktion för regime-detektion med VIX integration

        Args:
            prices: Price data for regime detection
            vix_data: Optional VIX data for enhanced regime classification

        Returns:
        - Most likely regime
        - Probabilities för alla regimer
        - Diagnostic information
        """

        # Beräkna market features
        features = self.compute_market_features(prices)

        if features.empty:
            # Fallback till neutral om ingen data
            default_probs = {regime: 1/3 for regime in MarketRegime}
            return MarketRegime.NEUTRAL, default_probs, {"error": "No market data"}

        # Ta senaste observation
        latest_features = features.iloc[-1]

        # Heuristisk klassificering med VIX
        raw_probabilities = self.classify_regime_heuristic(latest_features, vix_data)

        # Applicera transition smoothing
        previous_regime = self.regime_history[-1] if self.regime_history else None
        smoothed_probabilities = self.apply_transition_smoothing(raw_probabilities, previous_regime)

        # Välj mest troliga regim
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

        # Uppdatera history
        self.regime_history.append(most_likely_regime)
        self.regime_probabilities_history.append(smoothed_probabilities)

        # Begränsa history size
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
        """Hämta signal adjustments för given regim"""
        return REGIME_DEFINITIONS[regime].signal_adjustments

    def get_regime_explanation(self,
                             regime: MarketRegime,
                             probabilities: Dict[MarketRegime, float],
                             diagnostics: Dict,
                             vix_data: Optional[pd.DataFrame] = None) -> str:
        """
        Generera förklaring av regime-klassificering för användaren
        """

        regime_def = REGIME_DEFINITIONS[regime]
        prob_pct = probabilities[regime] * 100

        explanation = f"**{regime_def.name}** ({prob_pct:.0f}% säkerhet)\n"
        explanation += f"{regime_def.description}\n\n"

        # Market context
        features = diagnostics["market_features"]
        explanation += "**Marknadskontext:**\n"
        explanation += f"- Volatilitet: {features['volatility_20d']*100:.1f}% ({regime_def.volatility_level})\n"
        explanation += f"- 20-dagars avkastning: {features['return_20d']*100:+.1f}%\n"
        explanation += f"- Position vs SMA-long: {features['price_vs_sma_long']*100:+.1f}%\n"
        explanation += f"- Drawdown från high: {features['drawdown']*100:.1f}%\n"
        explanation += f"- Positiva dagar (20d): {features['positive_days_pct']*100:.0f}%\n\n"

        # VIX Analysis (if available)
        if vix_data is not None and not vix_data.empty:
            try:
                latest_vix = vix_data.iloc[-1]
                vix_regime = latest_vix['vix_regime']
                vix_level = latest_vix['vix_close']

                explanation += "**VIX Makroanalys:**\n"
                explanation += f"- VIX Nivå: {vix_level:.1f} ({vix_regime.replace('_', ' ').title()})\n"

                # VIX regime interpretation
                vix_interp = {
                    'low_fear': 'Låg rädsla - Investerare är självförtroende',
                    'moderate_fear': 'Måttlig oro - Försiktig optimism',
                    'high_fear': 'Hög oro - Investerare är nervösa',
                    'extreme_fear': 'Extrem rädsla - Panik i marknaden'
                }
                explanation += f"- Tolkning: {vix_interp.get(vix_regime, 'Okänd VIX regim')}\n"

                # VIX momentum if available
                if 'vix_momentum_5d' in latest_vix:
                    vix_momentum = latest_vix['vix_momentum_5d']
                    momentum_dir = "ökar" if vix_momentum > 0.05 else "minskar" if vix_momentum < -0.05 else "stabil"
                    explanation += f"- VIX Trend (5d): {vix_momentum*100:+.1f}% ({momentum_dir})\n"

                # VIX influence on regime decision
                if self.vix_config.get('enabled', False):
                    influence = self.vix_config.get('influence_weight', 0.4)
                    explanation += f"- VIX Påverkan: {influence:.0%} av regimbeslut\n"

                explanation += "\n"

            except Exception as e:
                explanation += "**VIX Analys:** Ej tillgänglig\n\n"

        # Regime probability breakdown
        explanation += "**Regim Sannolikheter:**\n"
        for reg, prob in probabilities.items():
            explanation += f"- {reg.value.title()}: {prob*100:.0f}%\n"
        explanation += "\n"

        # Signal implications
        explanation += "**Signal-justeringar:**\n"
        for signal, multiplier in regime_def.signal_adjustments.items():
            direction = "förstärks" if multiplier > 1.0 else "dämpas" if multiplier < 1.0 else "oförändrad"
            explanation += f"- {signal.title()}: {multiplier:.1f}x ({direction})\n"

        explanation += f"\n**Regime Karakteristik:**\n"
        explanation += f"- *{regime_def.momentum_behavior}*\n"
        explanation += f"- *{regime_def.sentiment_bias}*"

        return explanation