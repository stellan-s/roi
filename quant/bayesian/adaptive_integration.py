"""
Adaptive Bayesian integration with learned parameters instead of hardcoded values.

This module extends the existing BayesianPolicyEngine to use empirically estimated
parameters from the calibration framework.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from .signal_engine import BayesianSignalEngine, SignalType, SignalOutput
from .integration import BayesianPolicyEngine
from ..regime.detector import RegimeDetector, MarketRegime
from ..risk.analytics import RiskAnalytics, PortfolioRiskProfile
from ..adaptive.parameter_estimator import ParameterEstimator, ParameterEstimates

class AdaptiveBayesianEngine(BayesianPolicyEngine):
    """
    Adaptive Bayesian engine that learns parameters from data instead of using hardcoded values.
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.parameter_estimator = ParameterEstimator()
        # Initialize tail risk calculator if available
        try:
            from ..risk.tail_risk_calculator import TailRiskCalculator
            self.tail_risk_calculator = TailRiskCalculator()
        except ImportError:
            self.tail_risk_calculator = None
        self.estimated_params: Optional[ParameterEstimates] = None
        self.is_calibrated = False
        self.historical_returns_cache: Dict[str, pd.Series] = {}

        # Stock-specific signal effectiveness tracking
        self.stock_signal_posteriors: Dict[str, Dict[SignalType, any]] = {}

    def calibrate_parameters(self,
                           prices_df: pd.DataFrame,
                           sentiment_df: pd.DataFrame,
                           technical_df: pd.DataFrame,
                           regime_df: Optional[pd.DataFrame] = None,
                           returns_df: Optional[pd.DataFrame] = None) -> None:
        """
        Calibrate all model parameters from historical data.
        """
        print("Calibrating parameters from historical data...")

        self.estimated_params = self.parameter_estimator.estimate_all_parameters(
            prices_df=prices_df,
            sentiment_df=sentiment_df,
            technical_df=technical_df,
            regime_df=regime_df,
            returns_df=returns_df
        )

        self.is_calibrated = True
        self._update_engine_parameters()

        print(f"Calibration complete. Updated {self._count_estimated_params()} parameters.")

        # Train Bayesian signal posteriors from historical performance
        self._train_bayesian_posteriors(prices_df, sentiment_df, technical_df, returns_df)

        # Cache historical returns for tail risk calculation
        self._cache_historical_returns(prices_df)

    def _update_engine_parameters(self) -> None:
        """Update the Bayesian engine with estimated parameters."""
        if not self.estimated_params:
            return

        # Update base annual return in signal engine
        if hasattr(self.engine, '_base_annual_return'):
            self.engine._base_annual_return = self.estimated_params.base_annual_return.value

        # Update sigmoid scaling
        if hasattr(self.engine, '_sigmoid_scale'):
            self.engine._sigmoid_scale = self.estimated_params.sigmoid_scale_factor.value

    def _count_estimated_params(self) -> int:
        """Count how many parameters were successfully estimated."""
        if not self.estimated_params:
            return 0

        count = 0
        if self.estimated_params.sentiment_scale_factor.n_observations > 0:
            count += 1
        if self.estimated_params.momentum_scale_factor.n_observations > 0:
            count += 1
        if self.estimated_params.base_annual_return.n_observations > 0:
            count += 1
        if self.estimated_params.sigmoid_scale_factor.n_observations > 0:
            count += 1

        # Count regime adjustments
        for regime_adj in self.estimated_params.regime_adjustments.values():
            count += len(regime_adj)

        return count

    def bayesian_score_adaptive(self,
                               tech: pd.DataFrame,
                               senti: pd.DataFrame,
                               prices: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Enhanced Bayesian scoring with adaptive parameters.
        Uses learned parameters instead of hardcoded config values.
        """
        if not self.is_calibrated or not self.estimated_params:
            print("⚠️ Adaptive engine not calibrated, falling back to static parameters")
            return self.bayesian_score(tech, senti, prices)

        # Apply adaptive signal normalization
        adapted_tech = self._apply_adaptive_technical_scaling(tech)
        adapted_senti = self._apply_adaptive_sentiment_scaling(senti)

        # Get base recommendations using adapted signals
        recommendations = self.bayesian_score(adapted_tech, adapted_senti, prices)

        # Apply adaptive regime adjustments
        final_recommendations = self._apply_adaptive_regime_adjustments(recommendations, prices)

        print(f"✅ Applied adaptive parameters to {len(final_recommendations)} recommendations")
        return final_recommendations

    def get_parameter_diagnostics(self) -> pd.DataFrame:
        """Get parameter estimation diagnostics."""
        if not self.estimated_params:
            return pd.DataFrame()

        diagnostics_data = []

        # Individual parameters
        params = [
            ('sentiment_scale_factor', self.estimated_params.sentiment_scale_factor),
            ('momentum_scale_factor', self.estimated_params.momentum_scale_factor),
            ('base_annual_return', self.estimated_params.base_annual_return),
            ('sigmoid_scale_factor', self.estimated_params.sigmoid_scale_factor),
            ('signal_multiplier', self.estimated_params.signal_multiplier),
            ('momentum_volatility_factor', self.estimated_params.momentum_volatility_factor)
        ]

        for param_name, param in params:
            diagnostics_data.append({
                'parameter_name': param_name,
                'estimated_value': param.value,
                'default_value': param.default_value or 0.0,
                'confidence_lower': param.confidence_interval[0],
                'confidence_upper': param.confidence_interval[1],
                'estimation_method': param.estimation_method,
                'n_observations': param.n_observations,
                'parameter_type': 'signal_processing'
            })

        # Regime adjustments
        for regime, signals in self.estimated_params.regime_adjustments.items():
            for signal, param in signals.items():
                diagnostics_data.append({
                    'parameter_name': f'{regime}_{signal}_effectiveness',
                    'estimated_value': param.value,
                    'default_value': param.default_value or 1.0,
                    'confidence_lower': param.confidence_interval[0],
                    'confidence_upper': param.confidence_interval[1],
                    'estimation_method': param.estimation_method,
                    'n_observations': param.n_observations,
                    'parameter_type': 'regime_adjustment'
                })

        df = pd.DataFrame(diagnostics_data)

        if not df.empty:
            # Calculate change percentages
            df['change_percent'] = ((df['estimated_value'] - df['default_value']) /
                                   df['default_value'].replace(0, np.nan)) * 100

            # Sort by significance of change
            df['abs_change_percent'] = df['change_percent'].abs()
            df = df.sort_values('abs_change_percent', ascending=False)

        return df

    def _apply_adaptive_technical_scaling(self, tech: pd.DataFrame) -> pd.DataFrame:
        """Apply learned technical signal scaling."""
        adapted = tech.copy()

        if self.estimated_params and hasattr(self.estimated_params, 'momentum_scale_factor'):
            momentum_scale = self.estimated_params.momentum_scale_factor.value
            if 'mom_rank' in adapted.columns:
                # Adaptive momentum scaling: (rank - 0.5) * learned_scale
                adapted['mom_rank'] = (adapted['mom_rank'] - 0.5) * momentum_scale
                print(f"📊 Applied adaptive momentum scaling: {momentum_scale:.3f}")

        return adapted

    def _apply_adaptive_sentiment_scaling(self, senti: pd.DataFrame) -> pd.DataFrame:
        """Apply learned sentiment signal scaling."""
        adapted = senti.copy()

        if self.estimated_params and hasattr(self.estimated_params, 'sentiment_scale_factor'):
            sentiment_scale = self.estimated_params.sentiment_scale_factor.value
            if 'sent_score' in adapted.columns:
                # Adaptive sentiment scaling: score / learned_scale
                adapted['sent_score'] = np.clip(adapted['sent_score'] / sentiment_scale, -1.0, 1.0)
                print(f"📊 Applied adaptive sentiment scaling: {sentiment_scale:.3f}")

        return adapted

    def _apply_adaptive_regime_adjustments(self, recommendations: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
        """Apply learned regime-specific adjustments."""
        if not self.estimated_params or recommendations.empty:
            return recommendations

        adapted = recommendations.copy()

        # Detect current regime for each stock
        from ..regime.detector import RegimeDetector
        regime_detector = RegimeDetector(self.config)

        for _, row in recommendations.iterrows():
            ticker = row['ticker']

            # Get ticker's recent price data
            ticker_prices = prices[prices['ticker'] == ticker].tail(60)  # Last 60 days
            if len(ticker_prices) < 10:
                continue

            # Detect regime
            current_regime = regime_detector.detect_current_regime(ticker_prices)

            # Apply regime-specific adjustments
            regime_name = current_regime.regime.value.lower()
            if regime_name in self.estimated_params.regime_adjustments:
                regime_adj = self.estimated_params.regime_adjustments[regime_name]

                # Adjust signal strengths based on learned regime effectiveness
                if 'trend' in regime_adj:
                    trend_adj = regime_adj['trend'].value
                    if 'trend_weight' in adapted.columns:
                        mask = adapted['ticker'] == ticker
                        adapted.loc[mask, 'trend_weight'] *= trend_adj

                if 'momentum' in regime_adj:
                    momentum_adj = regime_adj['momentum'].value
                    if 'momentum_weight' in adapted.columns:
                        mask = adapted['ticker'] == ticker
                        adapted.loc[mask, 'momentum_weight'] *= momentum_adj

        print(f"🎯 Applied adaptive regime adjustments")
        return adapted

    def get_learning_summary(self) -> Dict:
        """Get summary of parameter learning results."""
        if not self.estimated_params:
            return {'status': 'no_learning_performed'}

        diagnostics = self.get_parameter_diagnostics()

        if diagnostics.empty:
            return {'status': 'no_diagnostics_available'}

        # Significant changes (>10%)
        significant_changes = diagnostics[diagnostics['abs_change_percent'] > 10]

        summary = {
            'status': 'learning_complete',
            'estimation_date': self.estimated_params.estimation_date.isoformat(),
            'data_period': {
                'start': self.estimated_params.data_period_start.isoformat(),
                'end': self.estimated_params.data_period_end.isoformat(),
                'total_observations': self.estimated_params.total_observations
            },
            'parameter_changes': {
                'total_parameters': len(diagnostics),
                'significant_changes': len(significant_changes),
                'avg_change_percent': diagnostics['abs_change_percent'].mean(),
                'max_change_percent': diagnostics['abs_change_percent'].max()
            },
            'top_changes': []
        }

        # Add top parameter changes
        for _, row in significant_changes.head(5).iterrows():
            summary['top_changes'].append({
                'parameter': row['parameter_name'],
                'old_value': row['default_value'],
                'new_value': row['estimated_value'],
                'change_percent': row['change_percent'],
                'method': row['estimation_method']
            })

        return summary

    def _normalize_signals_adaptive(self, row: pd.Series) -> Dict[SignalType, float]:
        """
        Adaptive signal normalization using estimated parameters instead of hardcoded values.
        """
        if not self.estimated_params:
            # Fall back to original method
            return self._normalize_signals(row)

        # Trend signal (unchanged - already well normalized)
        trend_signal = (row['above_sma'] - 0.5) * 1.0

        # Momentum signal with estimated scaling
        momentum_scale = self.estimated_params.momentum_scale_factor.value
        momentum_signal = (row['mom_rank'] - 0.5) * momentum_scale

        # Sentiment signal with estimated scaling
        sentiment_scale = self.estimated_params.sentiment_scale_factor.value
        sentiment_signal = np.clip(row['sent_score'] * sentiment_scale, -1.0, 1.0)

        return {
            SignalType.TREND: trend_signal,
            SignalType.MOMENTUM: momentum_signal,
            SignalType.SENTIMENT: sentiment_signal
        }

    def _get_adaptive_regime_adjustments(self, regime: MarketRegime) -> Dict[str, float]:
        """
        Get regime adjustments from estimated parameters instead of hardcoded values.
        """
        if not self.estimated_params or not self.estimated_params.regime_adjustments:
            # Fall back to hardcoded values
            return self._get_hardcoded_regime_adjustments(regime)

        regime_name = regime.value if hasattr(regime, 'value') else str(regime)

        if regime_name not in self.estimated_params.regime_adjustments:
            return {"momentum": 1.0, "trend": 1.0, "sentiment": 1.0}

        adjustments = {}
        regime_params = self.estimated_params.regime_adjustments[regime_name]

        for signal_type in ["momentum", "trend", "sentiment"]:
            if signal_type in regime_params:
                adjustments[signal_type] = regime_params[signal_type].value
            else:
                adjustments[signal_type] = 1.0

        return adjustments

    def _get_hardcoded_regime_adjustments(self, regime: MarketRegime) -> Dict[str, float]:
        """Fallback to original hardcoded regime adjustments."""
        regime_definitions = {
            MarketRegime.BULL: {
                "momentum": 1.3,
                "trend": 1.2,
                "sentiment": 0.8
            },
            MarketRegime.BEAR: {
                "momentum": 0.7,
                "trend": 1.1,
                "sentiment": 1.4
            },
            MarketRegime.NEUTRAL: {
                "momentum": 0.9,
                "trend": 0.8,
                "sentiment": 1.1
            }
        }
        return regime_definitions.get(regime, {"momentum": 1.0, "trend": 1.0, "sentiment": 1.0})

    def bayesian_score_adaptive(self,
                               tech: pd.DataFrame,
                               senti: pd.DataFrame,
                               prices: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Adaptive version of bayesian_score that uses learned parameters.
        """
        if not self.is_calibrated:
            print("Warning: Engine not calibrated. Using default parameters.")
            return self.bayesian_score(tech, senti, prices)

        # Regime detection
        regime_adjustment = 1.0
        if prices is not None:
            regime_info = self.regime_detector.detect_regime(prices)
            # detect_regime returns (regime, probabilities, diagnostics)
            self.current_regime = regime_info[0]
            self.regime_probabilities = regime_info[1]
            self.regime_diagnostics = regime_info[2]

            if self.current_regime:
                regime_adjustments = self._get_adaptive_regime_adjustments(self.current_regime)
            else:
                regime_adjustments = {"momentum": 1.0, "trend": 1.0, "sentiment": 1.0}
        else:
            # Equal regime probabilities if no price data
            self.regime_probabilities = {regime: 1/3 for regime in MarketRegime}
            regime_adjustments = {"momentum": 1.0, "trend": 1.0, "sentiment": 1.0}

        # Merge technical and sentiment data
        merged = pd.merge(tech, senti, on='ticker', how='left')
        merged["sent_score"] = merged["sent_score"].fillna(0).infer_objects()

        # Add current prices if available (tech already has close prices, but might be outdated)
        if prices is not None:
            current_prices = prices.groupby('ticker')['close'].last().reset_index()
            current_prices = current_prices.rename(columns={'close': 'current_close'})
            merged = pd.merge(merged, current_prices, on='ticker', how='left')
            # Use the most recent price data
            merged['close'] = merged['current_close'].fillna(merged['close'])
            merged = merged.drop(columns=['current_close'])

        results = []
        for _, row in merged.iterrows():
            ticker = row['ticker']

            # Adaptive signal normalization
            signals = self._normalize_signals_adaptive(row)

            # Apply regime adjustments
            for signal_name in ["momentum", "trend", "sentiment"]:
                signal_type = getattr(SignalType, signal_name.upper())
                if signal_type in signals:
                    multiplier = regime_adjustments.get(signal_name, 1.0)
                    signals[signal_type] *= multiplier

            # Calculate stock-specific weights
            stock_weights = self._get_stock_specific_weights(ticker, signals)

            # Manual signal combination using stock-specific weights
            combined_signal = sum(stock_weights[st] * signals[st] for st in signals.keys())
            combined_signal = float(np.clip(combined_signal, -3.0, 3.0))

            # Convert to probability and expected return using calibrated parameters
            if self.estimated_params and self.estimated_params.sigmoid_scale_factor.n_observations >= 0:
                sigmoid_scale = self.estimated_params.sigmoid_scale_factor.value
            else:
                sigmoid_scale = getattr(self.engine, '_sigmoid_scale', 3.0)

            prob_positive = 1 / (1 + np.exp(-sigmoid_scale * combined_signal))

            base_return = self.estimated_params.base_annual_return.value if self.estimated_params else 0.08
            signal_multiplier = self.estimated_params.signal_multiplier.value if self.estimated_params else 1.0
            expected_return = base_return * (1 + signal_multiplier * combined_signal) / 252

            # Calculate uncertainty (simplified)
            uncertainty = 1.0 - max(stock_weights.values())  # Higher when weights are more equal

            # Create custom output
            from quant.bayesian.signal_engine import SignalOutput
            output = SignalOutput(
                expected_return=expected_return,
                prob_positive=prob_positive,
                confidence_lower=expected_return - uncertainty * 0.01,
                confidence_upper=expected_return + uncertainty * 0.01,
                signal_weights=stock_weights,
                uncertainty=uncertainty
            )

            decision = self._output_to_decision(output)
            confidence = self._decision_confidence(output)

            results.append({
                'ticker': row['ticker'],
                'date': pd.Timestamp.now().date(),  # Add current date
                'close': row.get('close', 0.0),  # Add current price
                'decision': decision,
                'expected_return': output.expected_return,
                'prob_positive': output.prob_positive,
                'decision_confidence': confidence,
                'uncertainty': output.uncertainty,
                'trend_weight': output.signal_weights.get(SignalType.TREND, 0),
                'momentum_weight': output.signal_weights.get(SignalType.MOMENTUM, 0),
                'sentiment_weight': output.signal_weights.get(SignalType.SENTIMENT, 0),
                'regime': self.current_regime.value if self.current_regime else 'unknown',
                'regime_confidence': self.regime_probabilities[self.current_regime] if self.current_regime and self.regime_probabilities else 0.33,
                **self._get_tail_risk_metrics(row['ticker'], signals),
                'monte_carlo_prob_gain_20': 0.0,
                'monte_carlo_prob_loss_20': 0.0
            })

        return pd.DataFrame(results)

    def get_parameter_diagnostics(self) -> pd.DataFrame:
        """
        Get diagnostics on estimated vs default parameters.
        """
        if not self.estimated_params:
            return pd.DataFrame()

        diagnostics = []

        # Default values mapping
        defaults = {
            'sentiment_scale_factor': 1.0,
            'momentum_scale_factor': 2.0,
            'base_annual_return': 0.08,
            'annual_volatility': 0.25,
            'market_correlation': 0.7,
            'bull_momentum': 1.3,
            'bull_trend': 1.2,
            'bull_sentiment': 0.8,
            'bear_momentum': 0.7,
            'bear_trend': 1.1,
            'bear_sentiment': 1.4,
            'neutral_momentum': 0.9,
            'neutral_trend': 0.8,
            'neutral_sentiment': 1.1
        }

        # Signal normalization parameters
        default_val = defaults.get('sentiment_scale_factor', 1.0)
        estimated_val = self.estimated_params.sentiment_scale_factor.value
        change_pct = ((estimated_val - default_val) / default_val) * 100 if default_val != 0 else 0

        diagnostics.append({
            'parameter_type': 'signal_normalization',
            'parameter_name': 'sentiment_scale_factor',
            'estimated_value': estimated_val,
            'default_value': default_val,
            'change_percent': change_pct,
            'abs_change_percent': abs(change_pct),
            'confidence_interval': str(self.estimated_params.sentiment_scale_factor.confidence_interval),
            'estimation_method': self.estimated_params.sentiment_scale_factor.estimation_method,
            'n_observations': self.estimated_params.sentiment_scale_factor.n_observations
        })

        default_val = defaults.get('momentum_scale_factor', 2.0)
        estimated_val = self.estimated_params.momentum_scale_factor.value
        change_pct = ((estimated_val - default_val) / default_val) * 100 if default_val != 0 else 0

        diagnostics.append({
            'parameter_type': 'signal_normalization',
            'parameter_name': 'momentum_scale_factor',
            'estimated_value': estimated_val,
            'default_value': default_val,
            'change_percent': change_pct,
            'abs_change_percent': abs(change_pct),
            'confidence_interval': str(self.estimated_params.momentum_scale_factor.confidence_interval),
            'estimation_method': self.estimated_params.momentum_scale_factor.estimation_method,
            'n_observations': self.estimated_params.momentum_scale_factor.n_observations
        })

        # Bayesian parameters
        default_val = defaults.get('base_annual_return', 0.08)
        estimated_val = self.estimated_params.base_annual_return.value
        change_pct = ((estimated_val - default_val) / default_val) * 100 if default_val != 0 else 0

        diagnostics.append({
            'parameter_type': 'bayesian_engine',
            'parameter_name': 'base_annual_return',
            'estimated_value': estimated_val,
            'default_value': default_val,
            'change_percent': change_pct,
            'abs_change_percent': abs(change_pct),
            'confidence_interval': str(self.estimated_params.base_annual_return.confidence_interval),
            'estimation_method': self.estimated_params.base_annual_return.estimation_method,
            'n_observations': self.estimated_params.base_annual_return.n_observations
        })

        # Regime adjustments
        for regime_name, regime_params in self.estimated_params.regime_adjustments.items():
            for signal_type, param in regime_params.items():
                default_val = self._get_default_regime_value(regime_name, signal_type)
                estimated_val = param.value
                change_pct = ((estimated_val - default_val) / default_val) * 100 if default_val != 0 else 0

                diagnostics.append({
                    'parameter_type': 'regime_adjustment',
                    'parameter_name': f'{regime_name}_{signal_type}',
                    'estimated_value': estimated_val,
                    'default_value': default_val,
                    'change_percent': change_pct,
                    'abs_change_percent': abs(change_pct),
                    'confidence_interval': str(param.confidence_interval),
                    'estimation_method': param.estimation_method,
                    'n_observations': param.n_observations
                })

        return pd.DataFrame(diagnostics)

    def _get_default_regime_value(self, regime_name: str, signal_type: str) -> float:
        """Get the hardcoded default value for a regime adjustment."""
        defaults = {
            'bull': {"momentum": 1.3, "trend": 1.2, "sentiment": 0.8},
            'bear': {"momentum": 0.7, "trend": 1.1, "sentiment": 1.4},
            'neutral': {"momentum": 0.9, "trend": 0.8, "sentiment": 1.1}
        }
        return defaults.get(regime_name, {}).get(signal_type, 1.0)

    def _train_bayesian_posteriors(self,
                                  prices_df: pd.DataFrame,
                                  sentiment_df: pd.DataFrame,
                                  technical_df: pd.DataFrame,
                                  returns_df: Optional[pd.DataFrame]) -> None:
        """
        Train Bayesian signal posteriors from historical performance.
        This is where real adaptive learning happens.
        """
        if returns_df is None:
            print("No returns data provided for Bayesian training.")
            return

        print("Training Bayesian signal posteriors from historical performance...")

        # Prepare historical signal and return data
        historical_data = self._prepare_historical_signals(
            prices_df, sentiment_df, technical_df, returns_df
        )

        # Train stock-specific Bayesian posteriors
        training_samples = 0
        for _, row in historical_data.iterrows():
            ticker = row['ticker']

            # Extract signal values for this historical period
            signals = {
                SignalType.TREND: row['trend_signal'],
                SignalType.MOMENTUM: row['momentum_signal'],
                SignalType.SENTIMENT: row['sentiment_signal']
            }

            # Get actual forward return
            actual_return = row['forward_return']

            # Update stock-specific signal posteriors
            for signal_type, signal_value in signals.items():
                if not pd.isna(signal_value) and not pd.isna(actual_return):
                    self._update_stock_specific_posterior(ticker, signal_type, signal_value, actual_return)
                    training_samples += 1

        print(f"Trained Bayesian posteriors on {training_samples} signal-return observations")

        # Log the learning results
        self._log_posterior_learning()

    def _prepare_historical_signals(self,
                                   prices_df: pd.DataFrame,
                                   sentiment_df: pd.DataFrame,
                                   technical_df: pd.DataFrame,
                                   returns_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare historical signal values aligned with forward returns for training.
        """
        # Ensure all inputs use consistent datetime types before joining.
        technical_df = technical_df.copy()
        sentiment_df = sentiment_df.copy()
        returns_df = returns_df.copy()

        for frame in (technical_df, sentiment_df, returns_df):
            if 'date' in frame.columns and not np.issubdtype(frame['date'].dtype, np.datetime64):
                frame['date'] = pd.to_datetime(frame['date'], utc=False)

        # Merge all data sources
        historical = pd.merge(technical_df, sentiment_df, on=['ticker', 'date'], how='left')
        historical['sent_score'] = historical['sent_score'].fillna(0)

        # Add forward returns (shift returns backwards to align with signal dates)
        returns_pivot = returns_df.pivot(index='date', columns='ticker', values='return')

        # Calculate 21-day forward returns (our prediction horizon)
        forward_returns = returns_pivot.rolling(window=21).mean().shift(-21)

        # Reshape back to long format
        forward_returns_long = forward_returns.stack().reset_index()
        forward_returns_long.columns = ['date', 'ticker', 'forward_return']

        # Merge forward returns with signals
        historical = pd.merge(
            historical,
            forward_returns_long,
            on=['ticker', 'date'],
            how='left'
        )

        # Generate normalized signal values (same as real-time processing)
        historical['trend_signal'] = (historical['above_sma'] - 0.5) * 1.0
        historical['momentum_signal'] = (historical['mom_rank'] - 0.5) * 2.0  # Use current scale
        historical['sentiment_signal'] = np.clip(historical['sent_score'] * 0.5, -1.0, 1.0)  # Use current scale

        # Remove rows with missing forward returns
        historical = historical.dropna(subset=['forward_return'])

        return historical

    def _log_posterior_learning(self) -> None:
        """Log the results of Bayesian posterior learning."""
        print("\n=== Bayesian Signal Learning Results ===")

        for signal_type, posterior in self.engine.posteriors.items():
            prior = self.engine.priors[signal_type]
            effectiveness_change = posterior.mean_effectiveness - prior.mean_effectiveness

            print(f"{signal_type.name}:")
            print(f"  Prior effectiveness: {prior.mean_effectiveness:.3f}")
            print(f"  Learned effectiveness: {posterior.mean_effectiveness:.3f} (Δ{effectiveness_change:+.3f})")
            print(f"  Observations: {posterior.n_observations}")
            print(f"  Confidence interval: {posterior.confidence_interval}")

    def _update_stock_specific_posterior(self,
                                        ticker: str,
                                        signal_type: SignalType,
                                        signal_value: float,
                                        actual_return: float) -> None:
        """Update stock-specific signal effectiveness posteriors."""
        if ticker not in self.stock_signal_posteriors:
            self.stock_signal_posteriors[ticker] = {}

        # Initialize stock-specific posterior if not exists
        if signal_type not in self.stock_signal_posteriors[ticker]:
            # Start with global prior but track separately per stock
            prior = self.engine.priors[signal_type]
            self.stock_signal_posteriors[ticker][signal_type] = {
                'alpha': prior.mean_effectiveness * prior.confidence,
                'beta': (1 - prior.mean_effectiveness) * prior.confidence,
                'n_observations': 0,
                'effectiveness': prior.mean_effectiveness
            }

        # Get current posterior
        posterior = self.stock_signal_posteriors[ticker][signal_type]

        # Bayesian update: did signal predict direction correctly?
        signal_strength = abs(signal_value)
        prediction_accuracy = 1.0 if (signal_value > 0) == (actual_return > 0) else 0.0

        # Weight the update by signal strength
        effective_weight = signal_strength

        # Update Beta distribution parameters
        posterior['alpha'] += prediction_accuracy * effective_weight
        posterior['beta'] += (1 - prediction_accuracy) * effective_weight
        posterior['n_observations'] += 1

        # Calculate new effectiveness
        total = posterior['alpha'] + posterior['beta']
        posterior['effectiveness'] = posterior['alpha'] / total if total > 0 else 0.5

    def _get_stock_specific_weights(self, ticker: str, signals: Dict[SignalType, float]) -> Dict[SignalType, float]:
        """Calculate stock-specific signal weights based on learned effectiveness."""
        weights = {}
        total_weight = 0

        for signal_type, signal_value in signals.items():
            # Get stock-specific effectiveness if available
            if (ticker in self.stock_signal_posteriors and
                signal_type in self.stock_signal_posteriors[ticker]):
                effectiveness = self.stock_signal_posteriors[ticker][signal_type]['effectiveness']
                n_obs = self.stock_signal_posteriors[ticker][signal_type]['n_observations']

                # Confidence based on number of observations
                confidence_multiplier = min(1.0, n_obs / 100.0)  # Full confidence after 100 observations
            else:
                # Fall back to global prior
                effectiveness = self.engine.priors[signal_type].mean_effectiveness
                confidence_multiplier = 0.1  # Low confidence for unseen stock-signal combinations

            # Weight = effectiveness * signal_strength * confidence
            signal_strength = abs(signal_value) if signal_value != 0 else 0.1  # Minimum weight
            weight = effectiveness * signal_strength * confidence_multiplier

            weights[signal_type] = weight
            total_weight += weight

        # Normalize weights
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        else:
            # Equal weights as fallback
            n_signals = len(signals)
            weights = {k: 1.0 / n_signals for k in signals.keys()}

        return weights

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

    def update_parameters_online(self,
                               signal_values: Dict[SignalType, float],
                               actual_return: float,
                               time_horizon_days: int = 21) -> None:
        """
        Online parameter updates as new data becomes available.
        """
        # Update the Bayesian engine beliefs
        self.engine.update_beliefs(signal_values, actual_return, time_horizon_days)

        # Could also update other parameter estimates here
        # This would be part of a continuous learning system

    def _cache_historical_returns(self, prices_df: pd.DataFrame) -> None:
        """Cache historical returns for each ticker for tail risk calculation."""
        if prices_df.empty:
            return

        # Calculate returns for each ticker
        prices_sorted = prices_df.sort_values(['ticker', 'date'])
        prices_sorted['return'] = prices_sorted.groupby('ticker')['close'].pct_change()

        for ticker in prices_sorted['ticker'].unique():
            ticker_returns = prices_sorted[prices_sorted['ticker'] == ticker]['return'].dropna()
            self.historical_returns_cache[ticker] = ticker_returns

    def _calculate_statistical_tail_risk(self,
                                       ticker: str,
                                       signals: Dict[SignalType, float]) -> Tuple[float, float]:
        """
        Calculate proper statistical tail risk: P[return < -2σ] and P[|return| > 2σ]

        Returns:
            tuple: (downside_tail_risk, extreme_move_prob)
        """

        # Get historical returns for this ticker
        if ticker not in self.historical_returns_cache:
            # Default values if no historical data
            return 0.025, 0.05  # Roughly 2.5% and 5%

        historical_returns = self.historical_returns_cache[ticker]
        current_regime = self.current_regime.value if self.current_regime else None

        # Calculate proper tail risk metrics
        tail_metrics = self.tail_risk_calculator.calculate_tail_risk(
            ticker=ticker,
            historical_returns=historical_returns,
            signals=signals,
            regime=current_regime
        )

        return tail_metrics.downside_tail_risk, tail_metrics.extreme_move_prob

    def _get_tail_risk_metrics(self, ticker: str, signals: Dict[SignalType, float]) -> Dict[str, float]:
        """
        Get tail risk metrics as a dictionary for efficient single calculation.
        """
        tail_risk, extreme_move_prob = self._calculate_statistical_tail_risk(ticker, signals)
        return {
            'tail_risk': tail_risk,
            'extreme_move_prob': extreme_move_prob
        }

    def _calculate_tail_risk_legacy(self, row: pd.Series, signals: Dict[SignalType, float]) -> float:
        """
        Legacy tail risk calculation (kept for compatibility).

        This is the old heuristic-based approach.
        """
        # Base tail risk från volatility proxy (momentum volatility)
        momentum_volatility = abs(signals.get(SignalType.MOMENTUM, 0.0))
        base_tail_risk = momentum_volatility * 0.3

        # Regime adjustment
        regime_multiplier = 1.0
        if self.current_regime:
            if self.current_regime == MarketRegime.BEAR:
                regime_multiplier = 1.5  # Bear markets have higher tail risk
            elif self.current_regime == MarketRegime.BULL:
                regime_multiplier = 0.8  # Bull markets somewhat lower tail risk

        # Uncertainty contribution
        uncertainty_contribution = getattr(row, 'uncertainty', 0.3) * 0.2

        # Combined tail risk score
        tail_risk_score = (base_tail_risk + uncertainty_contribution) * regime_multiplier

        return np.clip(tail_risk_score, 0.0, 1.0)
