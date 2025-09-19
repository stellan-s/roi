"""
Technical Indicators Module

Extracts technical analysis signals from price data including SMA crossovers,
momentum indicators, and other technical patterns.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from datetime import datetime, timedelta

from .base import BaseModule, ModuleOutput, ModuleContract

class TechnicalIndicatorsModule(BaseModule):
    """Module for computing technical analysis indicators"""

    def define_contract(self) -> ModuleContract:
        return ModuleContract(
            name="technical_indicators",
            version="1.0.0",
            description="Computes technical analysis indicators from price data",
            input_schema={
                "prices": "pd.DataFrame[date, ticker, close, volume, high, low]"
            },
            output_schema={
                "technical_signals": "Dict[ticker, Dict[indicator, float]]",
                "technical_features": "pd.DataFrame"
            },
            performance_sla={
                "max_latency_ms": 200.0,
                "min_confidence": 0.7
            },
            dependencies=[],
            optional_inputs=["vix_data"]
        )

    def process(self, inputs: Dict[str, Any]) -> ModuleOutput:
        """Compute technical indicators from price data"""
        prices_df = inputs['prices']
        vix_data = inputs.get('vix_data')

        # Validate input data
        required_columns = ['date', 'ticker', 'close']
        if not all(col in prices_df.columns for col in required_columns):
            raise ValueError(f"Price data must contain columns: {required_columns}")

        if len(prices_df) == 0:
            return ModuleOutput(
                data={"technical_signals": {}, "technical_features": pd.DataFrame()},
                metadata={"reason": "no_price_data"},
                confidence=0.0
            )

        # Compute indicators for each ticker
        technical_signals = {}
        all_features = []

        tickers = prices_df['ticker'].unique()
        processed_tickers = 0

        for ticker in tickers:
            ticker_data = prices_df[prices_df['ticker'] == ticker].copy()

            if len(ticker_data) < self.config.get('min_data_points', 50):
                continue  # Skip tickers with insufficient data

            # Sort by date
            ticker_data = ticker_data.sort_values('date')

            # Compute indicators
            indicators = self._compute_indicators(ticker_data)

            if indicators:
                technical_signals[ticker] = indicators

                # Add to features dataframe
                latest_features = {
                    'ticker': ticker,
                    'date': ticker_data['date'].iloc[-1],
                    **indicators
                }
                all_features.append(latest_features)
                processed_tickers += 1

        # Create features DataFrame
        technical_features_df = pd.DataFrame(all_features) if all_features else pd.DataFrame()

        # Calculate confidence based on data quality and indicator strength
        confidence = self._calculate_confidence(
            processed_tickers,
            len(tickers),
            technical_signals
        )

        metadata = {
            "processed_tickers": processed_tickers,
            "total_tickers": len(tickers),
            "indicators_computed": list(self._get_enabled_indicators().keys()),
            "vix_available": vix_data is not None
        }

        return ModuleOutput(
            data={
                "technical_signals": technical_signals,
                "technical_features": technical_features_df
            },
            metadata=metadata,
            confidence=confidence
        )

    def test_module(self) -> Dict[str, Any]:
        """Test the technical indicators module with synthetic data"""
        # Generate synthetic price data
        test_data = self._generate_test_prices()

        # Test processing
        result = self.process({'prices': test_data})

        # Validate outputs
        signals = result.data['technical_signals']
        features_df = result.data['technical_features']

        tests_passed = 0
        total_tests = 5

        # Test 1: Signals generated
        if signals and len(signals) > 0:
            tests_passed += 1

        # Test 2: All signal values are valid (between -1 and 1)
        all_valid = True
        for ticker_signals in signals.values():
            for signal_value in ticker_signals.values():
                if not isinstance(signal_value, (int, float)) or not -1 <= signal_value <= 1:
                    all_valid = False
                    break
        if all_valid:
            tests_passed += 1

        # Test 3: Features dataframe has correct structure
        if not features_df.empty and 'ticker' in features_df.columns:
            tests_passed += 1

        # Test 4: SMA signal exists
        if signals and any('sma_signal' in s for s in signals.values()):
            tests_passed += 1

        # Test 5: Momentum exists
        if signals and any('momentum' in s for s in signals.values()):
            tests_passed += 1

        return {
            "status": "PASS" if tests_passed >= 4 else "FAIL",
            "tests_passed": tests_passed,
            "total_tests": total_tests,
            "signals_generated": len(signals),
            "confidence": result.confidence
        }

    def _generate_test_inputs(self) -> Dict[str, Any]:
        """Generate synthetic test data for benchmarking"""
        return {"prices": self._generate_test_prices()}

    def _generate_test_prices(self) -> pd.DataFrame:
        """Generate synthetic price data for testing"""
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        tickers = ['TEST1', 'TEST2', 'TEST3']

        data = []
        for ticker in tickers:
            # Generate random walk prices
            np.random.seed(hash(ticker) % 2**32)  # Deterministic per ticker
            prices = 100 + np.cumsum(np.random.normal(0, 1, len(dates)))
            volumes = np.random.randint(1000, 10000, len(dates))

            for i, date in enumerate(dates):
                data.append({
                    'date': date,
                    'ticker': ticker,
                    'close': prices[i],
                    'high': prices[i] * 1.02,
                    'low': prices[i] * 0.98,
                    'volume': volumes[i]
                })

        return pd.DataFrame(data)

    def _compute_indicators(self, ticker_data: pd.DataFrame) -> Dict[str, float]:
        """Compute technical indicators for a single ticker"""
        indicators = {}
        enabled_indicators = self._get_enabled_indicators()

        try:
            # Simple Moving Average Signal
            if 'sma' in enabled_indicators:
                sma_signal = self._compute_sma_signal(ticker_data)
                if sma_signal is not None:
                    indicators['sma_signal'] = sma_signal

            # Momentum
            if 'momentum' in enabled_indicators:
                momentum = self._compute_momentum(ticker_data)
                if momentum is not None:
                    indicators['momentum'] = momentum

            # RSI (if enabled)
            if 'rsi' in enabled_indicators:
                rsi = self._compute_rsi(ticker_data)
                if rsi is not None:
                    indicators['rsi'] = rsi

            # Volume trend
            if 'volume_trend' in enabled_indicators:
                volume_trend = self._compute_volume_trend(ticker_data)
                if volume_trend is not None:
                    indicators['volume_trend'] = volume_trend

            # Volatility
            if 'volatility' in enabled_indicators:
                volatility = self._compute_volatility(ticker_data)
                if volatility is not None:
                    indicators['volatility'] = volatility

        except Exception as e:
            # Log error but don't fail completely
            print(f"Warning: Error computing indicators for {ticker_data['ticker'].iloc[0]}: {e}")

        return indicators

    def _compute_sma_signal(self, data: pd.DataFrame) -> float:
        """Compute SMA crossover signal"""
        short_window = self.config.get('sma_short', 20)
        long_window = self.config.get('sma_long', 50)

        if len(data) < long_window:
            return None

        # Calculate SMAs
        data = data.copy()
        data['sma_short'] = data['close'].rolling(window=short_window).mean()
        data['sma_long'] = data['close'].rolling(window=long_window).mean()

        # Current values
        current_short = data['sma_short'].iloc[-1]
        current_long = data['sma_long'].iloc[-1]
        current_price = data['close'].iloc[-1]

        if pd.isna(current_short) or pd.isna(current_long):
            return None

        # Signal strength based on price relative to SMAs and crossover
        if current_short > current_long:
            # Bullish configuration
            signal = (current_price - current_long) / current_long
        else:
            # Bearish configuration
            signal = (current_price - current_long) / current_long

        # Normalize to [-1, 1]
        return np.tanh(signal * 10)

    def _compute_momentum(self, data: pd.DataFrame) -> float:
        """Compute momentum indicator"""
        momentum_window = self.config.get('momentum_window', 21)

        if len(data) < momentum_window:
            return None

        # Calculate momentum as percentage change
        current_price = data['close'].iloc[-1]
        past_price = data['close'].iloc[-momentum_window]

        if past_price == 0:
            return None

        momentum = (current_price - past_price) / past_price

        # Normalize to [-1, 1]
        return np.tanh(momentum * 5)

    def _compute_rsi(self, data: pd.DataFrame, window: int = 14) -> float:
        """Compute Relative Strength Index"""
        if len(data) < window + 1:
            return None

        # Calculate price changes
        data = data.copy()
        data['price_change'] = data['close'].diff()

        # Separate gains and losses
        data['gain'] = data['price_change'].where(data['price_change'] > 0, 0)
        data['loss'] = -data['price_change'].where(data['price_change'] < 0, 0)

        # Calculate average gains and losses
        avg_gain = data['gain'].rolling(window=window).mean().iloc[-1]
        avg_loss = data['loss'].rolling(window=window).mean().iloc[-1]

        if pd.isna(avg_gain) or pd.isna(avg_loss) or avg_loss == 0:
            return None

        # Calculate RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        # Convert to [-1, 1] scale (0.5 is neutral)
        return (rsi - 50) / 50

    def _compute_volume_trend(self, data: pd.DataFrame) -> float:
        """Compute volume trend indicator"""
        if len(data) < 20:
            return None

        # Calculate volume moving average
        volume_ma = data['volume'].rolling(window=20).mean()
        current_volume = data['volume'].iloc[-1]
        avg_volume = volume_ma.iloc[-1]

        if pd.isna(avg_volume) or avg_volume == 0:
            return None

        # Volume relative to average
        volume_ratio = current_volume / avg_volume

        # Convert to signal scale
        return np.tanh((volume_ratio - 1) * 2)

    def _compute_volatility(self, data: pd.DataFrame) -> float:
        """Compute volatility indicator"""
        if len(data) < 20:
            return None

        # Calculate daily returns
        returns = data['close'].pct_change().dropna()

        if len(returns) < 10:
            return None

        # Calculate rolling volatility
        volatility = returns.rolling(window=20).std().iloc[-1]

        if pd.isna(volatility):
            return None

        # Normalize (high volatility = negative signal for stability)
        return -np.tanh(volatility * 20)

    def _get_enabled_indicators(self) -> Dict[str, bool]:
        """Get which indicators are enabled in config"""
        return {
            'sma': self.config.get('sma_enabled', True),
            'momentum': self.config.get('momentum_enabled', True),
            'rsi': self.config.get('rsi_enabled', False),
            'volume_trend': self.config.get('volume_trend_enabled', True),
            'volatility': self.config.get('volatility_enabled', True)
        }

    def _calculate_confidence(self, processed_tickers: int, total_tickers: int, signals: Dict) -> float:
        """Calculate confidence score based on data quality and signal strength"""
        if total_tickers == 0:
            return 0.0

        # Base confidence from data coverage
        coverage_confidence = processed_tickers / total_tickers

        # Signal quality confidence
        signal_quality = 0.8  # Default
        if signals:
            # Check signal diversity and strength
            total_signals = sum(len(ticker_signals) for ticker_signals in signals.values())
            expected_signals = processed_tickers * len(self._get_enabled_indicators())

            if expected_signals > 0:
                signal_quality = min(1.0, total_signals / expected_signals)

        # Combined confidence
        confidence = (coverage_confidence * 0.6) + (signal_quality * 0.4)

        return min(1.0, confidence)