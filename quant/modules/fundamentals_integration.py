"""
Fundamentals integration module for ROI system
Provides optional fundamental analysis integration
"""

import logging
from typing import Dict, List, Optional
import yaml

from quant.data_layer.fundamentals import FundamentalsDataLayer
from quant.features.fundamentals import get_fundamental_features

logger = logging.getLogger(__name__)

class FundamentalsIntegration:
    """Main integration point for fundamentals in ROI system"""

    def __init__(self, config_path: str = "quant/config/settings.yaml"):
        """Initialize fundamentals integration"""
        self.config = self._load_config(config_path)
        self.enabled = self._is_enabled()

        if self.enabled:
            self.data_layer = FundamentalsDataLayer(
                config=self.config.get('data', {}),
                enabled=True
            )
            logger.info("Fundamentals integration enabled")
        else:
            self.data_layer = None
            logger.debug("Fundamentals integration disabled")

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            return {}

    def _is_enabled(self) -> bool:
        """Check if fundamentals are enabled in config"""
        return (self.config.get('data', {})
                .get('fundamentals_api', {})
                .get('enabled', False))

    def get_fundamental_features(self, symbol: str) -> Dict[str, float]:
        """
        Get fundamental features for a symbol

        Args:
            symbol: Stock symbol (e.g., 'VOLV-B.ST')

        Returns:
            Dictionary of fundamental features (empty if disabled or unavailable)
        """
        if not self.enabled or not self.data_layer:
            return {}

        try:
            # Get fundamentals data
            fundamentals_df = self.data_layer.get_fundamentals_for_symbol(symbol)

            # Calculate features
            features = get_fundamental_features(fundamentals_df)

            if features:
                logger.debug(f"Generated {len(features)} fundamental features for {symbol}")
            else:
                logger.debug(f"No fundamental features available for {symbol}")

            return features

        except Exception as e:
            logger.warning(f"Error getting fundamental features for {symbol}: {e}")
            return {}

    def get_fundamental_signal(self, symbol: str) -> float:
        """
        Get fundamental signal strength for a symbol

        Returns:
            Float between 0 and 1 representing fundamental strength
            0.5 if fundamentals disabled or unavailable
        """
        if not self.enabled:
            return 0.5

        features = self.get_fundamental_features(symbol)
        return features.get('fundamental_score', 0.5)

    def get_fundamentals_batch(self, symbols: List[str]) -> Dict[str, Dict[str, float]]:
        """Get fundamental features for multiple symbols"""
        if not self.enabled or not self.data_layer:
            return {symbol: {} for symbol in symbols}

        results = {}
        for symbol in symbols:
            results[symbol] = self.get_fundamental_features(symbol)

        return results

    def is_fundamentally_attractive(self, symbol: str, threshold: float = 0.6) -> bool:
        """
        Check if a stock is fundamentally attractive

        Args:
            symbol: Stock symbol
            threshold: Minimum fundamental score (0-1)

        Returns:
            True if fundamental score >= threshold, False otherwise
        """
        if not self.enabled:
            return True  # Don't filter if fundamentals disabled

        score = self.get_fundamental_signal(symbol)
        return score >= threshold

    def get_status(self) -> Dict[str, any]:
        """Get status of fundamentals integration"""
        status = {
            'enabled': self.enabled,
            'api_configured': False,
            'data_available': False
        }

        if self.enabled and self.data_layer:
            api_config = self.config.get('data', {}).get('fundamentals_api', {})
            status['api_configured'] = bool(api_config.get('api_key'))

            # Test with a sample symbol to check data availability
            try:
                test_features = self.get_fundamental_features('VOLV-B.ST')
                status['data_available'] = len(test_features) > 0
            except:
                pass

        return status

# Convenience functions for integration with existing code

def get_fundamentals_integration(config_path: str = "quant/config/settings.yaml") -> FundamentalsIntegration:
    """Get fundamentals integration instance"""
    return FundamentalsIntegration(config_path)

def add_fundamental_features(symbol: str, features_dict: Dict[str, float],
                           fundamentals: Optional[FundamentalsIntegration] = None) -> Dict[str, float]:
    """
    Add fundamental features to existing features dictionary

    Args:
        symbol: Stock symbol
        features_dict: Existing features dictionary to extend
        fundamentals: Optional fundamentals integration instance

    Returns:
        Extended features dictionary
    """
    if fundamentals is None:
        fundamentals = get_fundamentals_integration()

    fundamental_features = fundamentals.get_fundamental_features(symbol)

    # Add with prefix to avoid conflicts
    for key, value in fundamental_features.items():
        features_dict[f'fund_{key}'] = value

    return features_dict