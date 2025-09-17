import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from .signal_engine import BayesianSignalEngine, SignalType, SignalOutput
from ..regime.detector import RegimeDetector, MarketRegime
from ..risk.analytics import RiskAnalytics, PortfolioRiskProfile

class BayesianPolicyEngine:
    """
    Integration layer som ersätter simple_score med Bayesian approach
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config
        self.engine = BayesianSignalEngine(config)
        self.regime_detector = RegimeDetector(config)
        self.risk_analytics = RiskAnalytics(config)

        # Cache för regime information
        self.current_regime: Optional[MarketRegime] = None
        self.regime_probabilities: Optional[Dict[MarketRegime, float]] = None
        self.regime_diagnostics: Optional[Dict] = None

        # Decision thresholds från config
        if config and 'bayesian' in config and 'decision_thresholds' in config['bayesian']:
            thresholds = config['bayesian']['decision_thresholds']
            self.buy_probability = thresholds.get('buy_probability', 0.65)
            self.sell_probability = thresholds.get('sell_probability', 0.35)
            self.min_expected_return = thresholds.get('min_expected_return', 0.001)
            self.max_uncertainty = thresholds.get('max_uncertainty', 0.30)
        else:
            # Default thresholds
            self.buy_probability = 0.65
            self.sell_probability = 0.35
            self.min_expected_return = 0.001
            self.max_uncertainty = 0.30

    def bayesian_score(self,
                      tech: pd.DataFrame,
                      senti: pd.DataFrame,
                      prices: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Ersättning för simple_score med Bayesian signal combination

        Input: tech features (above_sma, mom_rank) + sentiment + prices for regime detection
        Output: E[r], Pr(↑), decisions med uncertainty och regime-justering
        """

        # Initialize default values for when regime detection is not available
        self.current_regime = None
        self.regime_probabilities = None
        self.regime_diagnostics = None

        # Merge som tidigare
        t = tech.copy()
        t["date"] = pd.to_datetime(t["date"]).dt.date
        s = senti.rename(columns={"date": "date"})
        df = t.merge(s, on=["date", "ticker"], how="left")
        df["sent_score"] = df["sent_score"].fillna(0).infer_objects()

        # Process varje rad genom Bayesian engine
        results = []

        for _, row in df.iterrows():
            # Normalisera signals till standardiserade ranges
            signals = self._normalize_signals(row)

            # Per-stock regime detection
            stock_regime = None
            stock_regime_confidence = 0.33
            regime_adjustment = 1.0

            if prices is not None:
                try:
                    # Get price data for this specific stock
                    ticker_prices = prices[prices['ticker'] == row['ticker']].tail(60)  # Last 60 days
                    if len(ticker_prices) >= 10:  # Need minimum data
                        stock_regime_result = self.regime_detector.detect_current_regime(ticker_prices)
                        stock_regime = stock_regime_result.regime
                        stock_regime_confidence = stock_regime_result.confidence

                        # Get regime-specific adjustments for this stock
                        regime_adjustments = self.regime_detector.get_regime_adjustments(stock_regime)
                        regime_adjustment = np.mean(list(regime_adjustments.values()))

                        # Apply regime-specific signal adjustments
                        adjusted_signals = {}
                        for signal_type, value in signals.items():
                            signal_name = signal_type.value
                            multiplier = regime_adjustments.get(signal_name, 1.0)
                            adjusted_signals[signal_type] = value * multiplier
                        signals = adjusted_signals
                except Exception as e:
                    # Fallback to neutral if detection fails for this stock
                    from ..regime.detector import MarketRegime
                    stock_regime = MarketRegime.NEUTRAL
                    stock_regime_confidence = 0.33

            # Bayesian combination with stock-specific regime adjustment
            output = self.engine.combine_signals(signals, regime_adjustment)

            # Konvertera till decisions
            decision = self._output_to_decision(output)

            results.append({
                'date': row['date'],
                'ticker': row['ticker'],
                'close': row['close'],
                'above_sma': row['above_sma'],
                'mom_rank': row['mom_rank'],
                'sent_score': row['sent_score'],
                'expected_return': output.expected_return,
                'prob_positive': output.prob_positive,
                'confidence_lower': output.confidence_lower,
                'confidence_upper': output.confidence_upper,
                'uncertainty': output.uncertainty,
                'trend_weight': output.signal_weights.get(SignalType.TREND, 0),
                'momentum_weight': output.signal_weights.get(SignalType.MOMENTUM, 0),
                'sentiment_weight': output.signal_weights.get(SignalType.SENTIMENT, 0),
                'decision': decision,
                'decision_confidence': self._decision_confidence(output),
                # Per-stock regime information
                'market_regime': stock_regime.value if stock_regime else 'unknown',
                'regime_confidence': stock_regime_confidence,
                # Heavy-tail risk metrics (will be calculated if price data available)
                'tail_risk_score': self._calculate_tail_risk_score(row, signals),
                'monte_carlo_prob_gain_20': 0.0,  # Will be calculated in risk analytics
                'monte_carlo_prob_loss_20': 0.0   # Will be calculated in risk analytics
            })

        return pd.DataFrame(results)

    def _normalize_signals(self, row: pd.Series) -> Dict[SignalType, float]:
        """Normalisera signals till [-1, 1] range för Bayesian engine"""

        # Trend signal: above_sma (0/1) -> (-0.5, +0.5)
        trend_signal = (row['above_sma'] - 0.5) * 1.0

        # Momentum signal: mom_rank (0-1) -> (-1, +1)
        momentum_signal = (row['mom_rank'] - 0.5) * 2.0

        # Sentiment signal: sent_score (typically -2 to +2) -> (-1, +1)
        sentiment_signal = np.clip(row['sent_score'] / 2.0, -1.0, 1.0)

        return {
            SignalType.TREND: trend_signal,
            SignalType.MOMENTUM: momentum_signal,
            SignalType.SENTIMENT: sentiment_signal
        }

    def _output_to_decision(self, output: SignalOutput) -> str:
        """
        Konvertera Bayesian output till Buy/Sell/Hold decisions
        Använder både prob_positive och expected_return med uncertainty
        """

        # Beslutströsklar med uncertainty-justering (från config)
        high_confidence_threshold = self.buy_probability
        low_confidence_threshold = self.sell_probability
        min_expected_return = self.min_expected_return
        max_uncertainty = self.max_uncertainty

        # Adjustera thresholds baserat på uncertainty (reduced penalty)
        uncertainty_penalty = output.uncertainty * 0.1  # Smaller penalty for uncertainty
        buy_threshold = high_confidence_threshold + uncertainty_penalty
        sell_threshold = low_confidence_threshold - uncertainty_penalty

        # Buy conditions: High probability AND positive expected return AND low uncertainty
        if (output.prob_positive >= buy_threshold and
            output.expected_return >= min_expected_return and
            output.uncertainty <= max_uncertainty):
            return "Buy"

        # Sell conditions: Low probability AND negative expected return with confidence
        elif (output.prob_positive <= sell_threshold and
              output.expected_return <= -min_expected_return and
              output.uncertainty <= max_uncertainty):
            return "Sell"

        else:
            return "Hold"

    def _decision_confidence(self, output: SignalOutput) -> float:
        """
        Beräkna confidence score för decision (0-1)
        Högre värde = mer säker på beslut
        """

        # Distance från neutralitet (0.5 prob)
        prob_distance = abs(output.prob_positive - 0.5) * 2  # 0-1 scale

        # Expected return magnitude (normalized)
        return_magnitude = min(abs(output.expected_return) * 100, 1.0)  # Cap at 1.0

        # Uncertainty penalty
        uncertainty_penalty = 1.0 - output.uncertainty

        # Combined confidence
        confidence = (prob_distance * 0.4 +
                     return_magnitude * 0.3 +
                     uncertainty_penalty * 0.3)

        return np.clip(confidence, 0.0, 1.0)

    def update_with_performance(self,
                              historical_predictions: pd.DataFrame,
                              actual_returns: pd.DataFrame,
                              horizon_days: int = 21) -> None:
        """
        Uppdatera Bayesian beliefs baserat på faktisk performance

        historical_predictions: Earlier output från bayesian_score
        actual_returns: Faktiska returns för samma period
        """

        # Merge predictions med actual returns
        merged = historical_predictions.merge(
            actual_returns,
            on=['date', 'ticker'],
            how='inner'
        )

        # Uppdatera för varje observation
        for _, row in merged.iterrows():
            signals = self._normalize_signals(row)
            actual_return = row['actual_return']  # From actual_returns df

            self.engine.update_beliefs(signals, actual_return, horizon_days)

    def get_diagnostics(self) -> pd.DataFrame:
        """Hämta diagnostics om signal performance"""
        return self.engine.get_signal_diagnostics()

    def get_signal_history(self) -> pd.DataFrame:
        """Hämta historik av signal observations för analysis"""
        if not self.engine.signal_history:
            return pd.DataFrame()

        history_records = []
        for record in self.engine.signal_history:
            flat_record = {
                'timestamp': record['timestamp'],
                'actual_return': record['actual_return'],
                'horizon_days': record['horizon_days']
            }

            # Flatten signals dict
            for signal_type, value in record['signals'].items():
                flat_record[f'{signal_type.value}_signal'] = value

            history_records.append(flat_record)

        return pd.DataFrame(history_records)

    def get_regime_info(self) -> Dict:
        """Hämta aktuell regime information"""
        if not self.current_regime:
            return {"regime": "unknown", "confidence": 0.33, "explanation": "Ingen regime detekterad"}

        explanation = self.regime_detector.get_regime_explanation(
            self.current_regime,
            self.regime_probabilities,
            self.regime_diagnostics
        )

        return {
            "regime": self.current_regime.value,
            "confidence": self.regime_probabilities[self.current_regime],
            "probabilities": {r.value: p for r, p in self.regime_probabilities.items()},
            "explanation": explanation,
            "diagnostics": self.regime_diagnostics
        }

    def get_regime_history(self) -> pd.DataFrame:
        """Hämta historik av regime detections"""
        if not self.regime_detector.regime_history:
            return pd.DataFrame()

        history_data = []
        for i, (regime, probs) in enumerate(zip(
            self.regime_detector.regime_history,
            self.regime_detector.regime_probabilities_history
        )):
            record = {
                "index": i,
                "regime": regime.value,
                "confidence": probs[regime]
            }
            # Add probability för varje regim
            for r, p in probs.items():
                record[f"prob_{r.value}"] = p

            history_data.append(record)

        return pd.DataFrame(history_data)

    def _calculate_tail_risk_score(self, row: pd.Series, signals: Dict) -> float:
        """
        Calculate simplified tail risk score based på signal characteristics

        Returns score 0-1 where higher = more tail risk
        Detta är en approximation - full calculation kräver price history
        """

        # Base tail risk från volatility proxy (momentum volatility)
        momentum_volatility = abs(signals.get(SignalType.MOMENTUM, 0.0))  # Higher momentum = potentially higher vol
        base_tail_risk = momentum_volatility * 0.3  # Scale to reasonable range

        # Regime adjustment
        regime_multiplier = 1.0
        if self.current_regime:
            if self.current_regime.value == 'bear':
                regime_multiplier = 1.5  # Bear markets have higher tail risk
            elif self.current_regime.value == 'bull':
                regime_multiplier = 0.8  # Bull markets somewhat lower tail risk

        # Signal uncertainty contribution
        uncertainty_contribution = getattr(row, 'uncertainty', 0.3) * 0.2

        # Combined tail risk score
        tail_risk_score = (base_tail_risk + uncertainty_contribution) * regime_multiplier

        return np.clip(tail_risk_score, 0.0, 1.0)