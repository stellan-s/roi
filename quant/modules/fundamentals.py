"""
Fundamentals Analysis Module for ROI System
Provides fundamental analysis as a pipeline module
"""

import logging
from typing import Dict, List, Any
import pandas as pd

from quant.modules.fundamentals_integration import FundamentalsIntegration
from quant.modules.base import BaseModule, ModuleOutput, ModuleContract

logger = logging.getLogger(__name__)

class FundamentalsModule(BaseModule):
    """Pipeline module for fundamental analysis"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = "fundamentals_analysis"
        self.version = "1.0.0"
        fundamentals_config = config.get('data', {}).get('fundamentals_api', {})
        self.enabled = fundamentals_config.get('enabled', False)

        if self.enabled:
            self.fundamentals = FundamentalsIntegration()
            logger.info("Fundamentals module enabled")
        else:
            self.fundamentals = None
            logger.debug("Fundamentals module disabled")

    def define_contract(self) -> ModuleContract:
        """Define the module contract"""
        return ModuleContract(
            name="fundamentals_analysis",
            version="1.0.0",
            description="Fundamental analysis using company metrics API",
            input_schema={"tickers": "List[str]"},
            output_schema={"fundamentals": "Dict[str, Dict]"},
            performance_sla={"max_latency_ms": 5000.0, "min_confidence": 0.7}
        )

    def process(self, data: Dict[str, Any]) -> ModuleOutput:
        """
        Process fundamental data for all tickers

        Args:
            data: Pipeline data containing tickers and other modules' results

        Returns:
            Updated data with fundamental features and signals
        """
        if not self.enabled or not self.fundamentals:
            logger.debug("Fundamentals module disabled, skipping")
            return data

        tickers = data.get('tickers', [])
        if not tickers:
            logger.warning("No tickers provided to fundamentals module")
            return data

        logger.info(f"Processing fundamentals for {len(tickers)} tickers")

        # Initialize fundamentals data structures
        if 'fundamentals' not in data:
            data['fundamentals'] = {}

        # Process each ticker
        fundamentals_results = {}
        working_stocks = []

        for ticker in tickers:
            try:
                # Get fundamental features
                features = self.fundamentals.get_fundamental_features(ticker)

                if features:
                    fundamentals_results[ticker] = features
                    working_stocks.append(ticker)
                    logger.debug(f"Got {len(features)} fundamental features for {ticker}")
                else:
                    # Provide neutral defaults for stocks without fundamentals
                    fundamentals_results[ticker] = self._get_neutral_features()
                    logger.debug(f"No fundamentals for {ticker}, using neutral features")

            except Exception as e:
                logger.warning(f"Error processing fundamentals for {ticker}: {e}")
                fundamentals_results[ticker] = self._get_neutral_features()

        # Store results
        output_data = {
            'fundamentals': fundamentals_results,
            'fundamentals_working_stocks': working_stocks,
            'fundamentals_enabled': True
        }

        logger.info(f"Fundamentals processing complete: {len(working_stocks)}/{len(tickers)} stocks have data")

        confidence = len(working_stocks) / len(tickers) if tickers else 0.5

        return ModuleOutput(
            data=output_data,
            confidence=confidence,
            metadata={
                'total_tickers': len(tickers),
                'successful_tickers': len(working_stocks),
                'coverage_rate': confidence
            }
        )

    def _get_neutral_features(self) -> Dict[str, float]:
        """Get neutral fundamental features for stocks without data"""
        return {
            'fundamental_score': 0.5,
            'fundamental_signal': 0.5,
            'fundamental_quality': 0.5,
            'fundamental_growth': 0.5,
            'fundamental_profitability': 0.5,
            'fundamental_financial_health': 0.5
        }

    def get_signal_contribution(self, ticker: str, data: Dict[str, Any]) -> float:
        """
        Get fundamental signal contribution for a ticker

        Args:
            ticker: Stock ticker
            data: Pipeline data

        Returns:
            Fundamental signal strength (0-1)
        """
        if not self.enabled:
            return 0.5

        fundamentals = data.get('fundamentals', {})
        ticker_fundamentals = fundamentals.get(ticker, {})

        return ticker_fundamentals.get('fundamental_score', 0.5)

    def get_quality_filter(self, ticker: str, data: Dict[str, Any], threshold: float = 0.6) -> bool:
        """
        Check if stock passes fundamental quality filter

        Args:
            ticker: Stock ticker
            data: Pipeline data
            threshold: Minimum fundamental score

        Returns:
            True if stock passes quality filter
        """
        if not self.enabled:
            return True  # Don't filter if fundamentals disabled

        signal = self.get_signal_contribution(ticker, data)
        return signal >= threshold

    def get_feature_summary(self, ticker: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get summary of fundamental features for reporting

        Args:
            ticker: Stock ticker
            data: Pipeline data

        Returns:
            Dictionary with fundamental summary
        """
        if not self.enabled:
            return {'enabled': False}

        fundamentals = data.get('fundamentals', {})
        ticker_fundamentals = fundamentals.get(ticker, {})
        working_stocks = data.get('fundamentals_working_stocks', [])

        summary = {
            'enabled': True,
            'has_data': ticker in working_stocks,
            'fundamental_score': ticker_fundamentals.get('fundamental_score', 0.5),
            'features_count': len([k for k, v in ticker_fundamentals.items() if k.startswith('fund_') or k in ['roe', 'net_margin', 'debt_to_equity']])
        }

        # Add key metrics if available
        for key in ['roe', 'net_margin', 'debt_to_equity', 'revenue_growth_1y']:
            if key in ticker_fundamentals:
                summary[key] = ticker_fundamentals[key]

        return summary

    def get_status(self) -> Dict[str, Any]:
        """Get module status"""
        status = {
            'name': self.name,
            'version': self.version,
            'enabled': self.enabled
        }

        if self.enabled and self.fundamentals:
            fund_status = self.fundamentals.get_status()
            status.update({
                'api_configured': fund_status['api_configured'],
                'data_available': fund_status['data_available']
            })

        return status

    def test_module(self) -> Dict[str, Any]:
        """Built-in health check for the fundamentals module"""
        try:
            if not self.enabled:
                return {"status": "success", "message": "Module disabled"}

            if not self.fundamentals:
                return {"status": "error", "message": "Fundamentals integration not initialized"}

            # Test with a known ticker
            test_result = self.fundamentals.get_fundamental_features("ALFA")

            return {
                "status": "success",
                "message": "Test completed successfully",
                "test_data_available": test_result is not None
            }
        except Exception as e:
            return {"status": "error", "message": f"Test failed: {str(e)}"}

    def _generate_test_inputs(self) -> Dict[str, Any]:
        """Generate synthetic test data for benchmarking"""
        return {
            "tickers": ["ALFA", "SEB-A", "VOLV-B"]
        }