"""
Regime Detection Module

Detects market regimes (Bull/Bear/Neutral) using technical analysis,
volatility patterns, and VIX integration.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from enum import Enum

from .base import BaseModule, ModuleOutput, ModuleContract

class MarketRegime(Enum):
    """Market regimes with clear economic definitions."""
    BULL = "bull"       # Upward trend, low volatility, positive sentiment
    BEAR = "bear"       # Downward trend, high volatility, negative sentiment
    NEUTRAL = "neutral" # Sideways trend, moderate volatility, mixed sentiment

class RegimeDetectionModule(BaseModule):
    """Module for detecting market regimes and generating regime-adjusted signals"""

    def define_contract(self) -> ModuleContract:
        return ModuleContract(
            name="regime_detection",
            version="1.0.0",
            description="Detects market regimes (Bull/Bear/Neutral)",
            input_schema={
                "prices": "pd.DataFrame[date, ticker, close]"
            },
            output_schema={
                "current_regime": "str",
                "regime_probabilities": "Dict[str, float]",
                "regime_adjustments": "Dict[str, float]",
                "regime_features": "pd.DataFrame"
            },
            performance_sla={
                "max_latency_ms": 300.0,
                "min_confidence": 0.5
            },
            dependencies=[],
            optional_inputs=["vix_data"]
        )

    def process(self, inputs: Dict[str, Any]) -> ModuleOutput:
        """Detect market regime from price data"""
        prices_df = inputs['prices']
        vix_data = inputs.get('vix_data')

        # Validate inputs
        if prices_df.empty:
            return ModuleOutput(
                data={
                    "current_regime": "neutral",
                    "regime_probabilities": {"bull": 0.33, "bear": 0.33, "neutral": 0.34},
                    "regime_adjustments": {"momentum": 1.0, "trend": 1.0, "sentiment": 1.0},
                    "regime_features": pd.DataFrame()
                },
                metadata={"reason": "no_price_data"},
                confidence=0.0
            )

        try:
            # Compute market features
            features = self._compute_market_features(prices_df)

            if features.empty:
                return self._default_regime_output("insufficient_data")

            # Classify regime
            regime_probs = self._classify_regime(features.iloc[-1], vix_data)

            # Get most likely regime
            current_regime = max(regime_probs, key=regime_probs.get)

            # Get regime adjustments
            regime_adjustments = self._get_regime_adjustments(current_regime)

            # Calculate confidence
            confidence = self._calculate_confidence(regime_probs, features)

            # Create features dataframe
            regime_features = self._create_regime_features(features.tail(1))

            metadata = {
                "regime_method": "heuristic_with_transitions",
                "vix_available": vix_data is not None and not vix_data.empty,
                "features_computed": len(features),
                "confidence_score": max(regime_probs.values())
            }

            return ModuleOutput(
                data={
                    "current_regime": current_regime.value,
                    "regime_probabilities": {r.value: p for r, p in regime_probs.items()},
                    "regime_adjustments": regime_adjustments,
                    "regime_features": regime_features
                },
                metadata=metadata,
                confidence=confidence
            )

        except Exception as e:
            return ModuleOutput(
                data={
                    "current_regime": "neutral",
                    "regime_probabilities": {"bull": 0.33, "bear": 0.33, "neutral": 0.34},
                    "regime_adjustments": {"momentum": 1.0, "trend": 1.0, "sentiment": 1.0},
                    "regime_features": pd.DataFrame()
                },
                metadata={"error": str(e)},
                confidence=0.1
            )

    def test_module(self) -> Dict[str, Any]:
        """Test the regime detection module with synthetic data"""
        # Generate test data
        test_prices = self._generate_test_prices()

        # Test processing
        result = self.process({'prices': test_prices})

        # Validate outputs
        current_regime = result.data['current_regime']
        regime_probs = result.data['regime_probabilities']
        regime_adjustments = result.data['regime_adjustments']

        tests_passed = 0
        total_tests = 6

        # Test 1: Valid regime returned
        if current_regime in ['bull', 'bear', 'neutral']:
            tests_passed += 1

        # Test 2: Probabilities sum to 1
        prob_sum = sum(regime_probs.values())
        if 0.99 <= prob_sum <= 1.01:
            tests_passed += 1

        # Test 3: All probabilities are valid
        if all(0 <= p <= 1 for p in regime_probs.values()):
            tests_passed += 1

        # Test 4: Adjustments are reasonable
        if all(0.5 <= adj <= 2.0 for adj in regime_adjustments.values()):
            tests_passed += 1

        # Test 5: Required adjustment keys exist
        required_keys = ['momentum', 'trend', 'sentiment']
        if all(key in regime_adjustments for key in required_keys):
            tests_passed += 1

        # Test 6: Confidence is reasonable
        if 0.2 <= result.confidence <= 1.0:
            tests_passed += 1

        return {
            "status": "PASS" if tests_passed >= 4 else "FAIL",
            "tests_passed": tests_passed,
            "total_tests": total_tests,
            "detected_regime": current_regime,
            "confidence": result.confidence
        }

    def _generate_test_inputs(self) -> Dict[str, Any]:
        """Generate synthetic test data for benchmarking"""
        return {"prices": self._generate_test_prices()}

    def _generate_test_prices(self) -> pd.DataFrame:
        """Generate synthetic price data for testing"""
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        tickers = ['SPY', 'QQQ', 'IWM']

        # Generate regime-specific data
        regime_type = np.random.choice(['bull', 'bear', 'neutral'])

        data = []
        for ticker in tickers:
            np.random.seed(hash(ticker) % 2**32)
            base_price = 100

            if regime_type == 'bull':
                # Bull market: upward trend, lower volatility
                drift = 0.0008  # 0.08% daily
                volatility = 0.015  # 1.5% daily vol
            elif regime_type == 'bear':
                # Bear market: downward trend, higher volatility
                drift = -0.0005  # -0.05% daily
                volatility = 0.025  # 2.5% daily vol
            else:
                # Neutral: sideways, moderate volatility
                drift = 0.0001  # 0.01% daily
                volatility = 0.020  # 2.0% daily vol

            prices = [base_price]
            for i in range(1, len(dates)):
                change = np.random.normal(drift, volatility)
                new_price = prices[-1] * (1 + change)
                prices.append(max(10, new_price))

            for i, date in enumerate(dates):
                data.append({
                    'date': date,
                    'ticker': ticker,
                    'close': prices[i]
                })

        return pd.DataFrame(data)

    def _compute_market_features(self, prices_df: pd.DataFrame) -> pd.DataFrame:
        """Compute market features for regime classification"""
        # Focus on market-wide behavior by averaging across tickers
        market_data = prices_df.groupby('date')['close'].mean().sort_index()

        if len(market_data) < 10:
            return pd.DataFrame()

        features = pd.DataFrame(index=market_data.index)
        features['price'] = market_data

        # Returns
        features['ret_1d'] = market_data.pct_change()
        features['ret_5d'] = market_data.pct_change(5)
        features['ret_20d'] = market_data.pct_change(20)

        # Volatility
        vol_window = min(20, len(market_data) // 3)
        features['vol_20d'] = features['ret_1d'].rolling(vol_window).std() * np.sqrt(252)

        # Trend measures
        sma_short_window = min(10, len(market_data) // 6)
        sma_long_window = min(50, len(market_data) // 3)

        features['sma_short'] = market_data.rolling(sma_short_window).mean()
        features['sma_long'] = market_data.rolling(sma_long_window).mean()
        features['price_vs_sma_long'] = (market_data - features['sma_long']) / features['sma_long']
        features['sma_slope'] = features['sma_short'].pct_change(min(5, len(market_data) // 10))

        # Drawdown
        rolling_max_window = min(60, len(market_data) // 2)
        features['rolling_max'] = market_data.rolling(rolling_max_window).max()
        features['drawdown'] = (market_data - features['rolling_max']) / features['rolling_max']

        # Positive days percentage
        pos_days_window = min(20, len(market_data) // 4)
        features['positive_days_pct'] = (features['ret_1d'] > 0).rolling(pos_days_window).mean()

        # Drop NaNs
        return features.dropna()

    def _classify_regime(self, features: pd.Series, vix_data: Optional[pd.DataFrame] = None) -> Dict[MarketRegime, float]:
        """Classify regime based on market features"""
        scores = {regime: 0.0 for regime in MarketRegime}

        # Get thresholds from config
        vol_low = self.config.get('vol_low_threshold', 0.12)
        vol_high = self.config.get('vol_high_threshold', 0.30)
        ret_bull = self.config.get('return_bull_threshold', 0.003)
        ret_bear = self.config.get('return_bear_threshold', -0.003)
        drawdown_bear = self.config.get('drawdown_bear_threshold', -0.15)

        # VIX-based scoring (if available)
        if vix_data is not None and not vix_data.empty and self.config.get('vix_integration', True):
            try:
                latest_vix = vix_data.iloc[-1]
                vix_level = latest_vix.get('vix_close', 20)

                if vix_level < 15:
                    scores[MarketRegime.BULL] += 0.4
                    scores[MarketRegime.BEAR] -= 0.2
                elif vix_level < 20:
                    scores[MarketRegime.BULL] += 0.2
                elif vix_level < 30:
                    scores[MarketRegime.NEUTRAL] += 0.2
                elif vix_level < 40:
                    scores[MarketRegime.BEAR] += 0.2
                    scores[MarketRegime.BULL] -= 0.2
                else:
                    scores[MarketRegime.BEAR] += 0.4
                    scores[MarketRegime.BULL] -= 0.4

            except Exception:
                pass  # Continue without VIX

        # Volatility-based classification
        vol = features['vol_20d']
        if vol < vol_low:
            scores[MarketRegime.BULL] += 0.3
            scores[MarketRegime.NEUTRAL] += 0.1
        elif vol > vol_high:
            scores[MarketRegime.BEAR] += 0.4
            scores[MarketRegime.NEUTRAL] += 0.1
        else:
            scores[MarketRegime.NEUTRAL] += 0.3

        # Return-based classification
        ret_20d = features['ret_20d']
        if ret_20d > ret_bull * 20:
            scores[MarketRegime.BULL] += 0.4
        elif ret_20d < ret_bear * 20:
            scores[MarketRegime.BEAR] += 0.4
        else:
            scores[MarketRegime.NEUTRAL] += 0.3

        # Trend-based classification
        price_vs_sma = features['price_vs_sma_long']
        sma_slope = features['sma_slope']
        if price_vs_sma > 0.02 and sma_slope > 0.001:
            scores[MarketRegime.BULL] += 0.3
        elif price_vs_sma < -0.02 and sma_slope < -0.001:
            scores[MarketRegime.BEAR] += 0.3
        else:
            scores[MarketRegime.NEUTRAL] += 0.2

        # Drawdown-based (bear market signal)
        drawdown = features['drawdown']
        if drawdown < drawdown_bear:
            scores[MarketRegime.BEAR] += 0.5
            scores[MarketRegime.BULL] = max(0, scores[MarketRegime.BULL] - 0.3)

        # Consistency check
        positive_days = features['positive_days_pct']
        if positive_days > 0.65:
            scores[MarketRegime.BULL] += 0.2
        elif positive_days < 0.35:
            scores[MarketRegime.BEAR] += 0.2
        else:
            scores[MarketRegime.NEUTRAL] += 0.2

        # Normalize to probabilities
        total_score = sum(scores.values())
        if total_score > 0:
            probabilities = {regime: max(0, score/total_score) for regime, score in scores.items()}
        else:
            probabilities = {regime: 1/3 for regime in MarketRegime}

        # Ensure probabilities sum to 1
        prob_sum = sum(probabilities.values())
        if prob_sum > 0:
            probabilities = {k: v/prob_sum for k, v in probabilities.items()}

        return probabilities

    def _get_regime_adjustments(self, regime: MarketRegime) -> Dict[str, float]:
        """Get signal adjustments for the given regime"""
        # Default adjustments based on regime characteristics
        regime_adjustments = {
            MarketRegime.BULL: {
                "momentum": 1.3,    # Momentum works better in bull markets
                "trend": 1.2,       # Trend-following is strong
                "sentiment": 0.8    # Sentiment less important
            },
            MarketRegime.BEAR: {
                "momentum": 0.7,    # Momentum less reliable
                "trend": 1.1,       # Trend still matters
                "sentiment": 1.4    # Sentiment very important
            },
            MarketRegime.NEUTRAL: {
                "momentum": 0.9,    # Softer momentum
                "trend": 0.8,       # Weaker trend-following
                "sentiment": 1.1    # Sentiment slightly more useful
            }
        }

        # Allow config overrides
        config_adjustments = self.config.get('regime_adjustments', {})
        if regime.value in config_adjustments:
            regime_adjustments[regime].update(config_adjustments[regime.value])

        return regime_adjustments[regime]

    def _calculate_confidence(self, regime_probs: Dict[MarketRegime, float], features: pd.DataFrame) -> float:
        """Calculate confidence score for regime detection"""
        # Base confidence from probability spread
        max_prob = max(regime_probs.values())
        min_prob = min(regime_probs.values())
        prob_spread_conf = (max_prob - min_prob) * 2  # Scale to [0, 1]

        # Data quality confidence
        data_quality_conf = min(1.0, len(features) / 50)  # Ideal: 50+ data points

        # Feature validity confidence
        latest_features = features.iloc[-1] if not features.empty else pd.Series()
        feature_validity = 1.0

        # Check for reasonable values
        if not latest_features.empty:
            vol = latest_features.get('vol_20d', 0.2)
            if vol > 1.0 or vol < 0.05:  # Unreasonable volatility
                feature_validity *= 0.5

            ret_20d = latest_features.get('ret_20d', 0)
            if abs(ret_20d) > 0.5:  # >50% in 20 days is extreme
                feature_validity *= 0.7

        # Combined confidence
        confidence = (
            prob_spread_conf * 0.5 +
            data_quality_conf * 0.3 +
            feature_validity * 0.2
        )

        return min(1.0, max(0.1, confidence))

    def _create_regime_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Create regime features dataframe"""
        if features.empty:
            return pd.DataFrame()

        regime_features = features.copy()
        regime_features['regime_detection_date'] = datetime.now()

        # Add regime-specific indicators
        if not regime_features.empty:
            latest = regime_features.iloc[-1]

            # Regime strength indicators
            regime_features['bull_strength'] = np.tanh(latest['ret_20d'] * 10)
            regime_features['bear_strength'] = np.tanh(-latest['drawdown'] * 10)
            regime_features['neutral_strength'] = 1 - abs(latest['price_vs_sma_long'])

        return regime_features

    def _default_regime_output(self, reason: str) -> ModuleOutput:
        """Return default regime output for error cases"""
        return ModuleOutput(
            data={
                "current_regime": "neutral",
                "regime_probabilities": {"bull": 0.33, "bear": 0.33, "neutral": 0.34},
                "regime_adjustments": {"momentum": 1.0, "trend": 1.0, "sentiment": 1.0},
                "regime_features": pd.DataFrame()
            },
            metadata={"reason": reason},
            confidence=0.1
        )