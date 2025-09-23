"""
Fundamentals feature module for ROI system
Provides optional fundamental analysis features
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

def get_fundamental_features(fundamentals_df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate fundamental features from API data

    Args:
        fundamentals_df: DataFrame from FundamentalsDataLayer

    Returns:
        Dictionary of fundamental features (empty if no data)
    """
    if fundamentals_df.empty:
        return {}

    try:
        processor = FundamentalsProcessor()
        return processor.create_fundamental_features(fundamentals_df)
    except Exception as e:
        logger.warning(f"Error processing fundamentals: {e}")
        return {}

class FundamentalsProcessor:
    """Process fundamental data into investment features"""

    def __init__(self):
        # Map API metric keys to our internal names
        self.metric_mapping = {
            'revenue': 'revenue',
            'ebit': 'ebit',
            'netIncome': 'net_income',
            'eps': 'eps',
            'equity': 'equity',
            'assets': 'assets',
            'operatingCF': 'operating_cf',
            'capex': 'capex',
            'shares': 'shares',
            'netDebt': 'net_debt'
        }

    def create_fundamental_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Create all fundamental features for a stock"""
        if df.empty:
            return {}

        # Get latest metrics
        latest_metrics = self._get_latest_metrics(df)

        if not latest_metrics:
            return {}

        features = {}

        # Calculate ratios
        features.update(self._calculate_profitability_ratios(latest_metrics))
        features.update(self._calculate_efficiency_ratios(latest_metrics))
        features.update(self._calculate_leverage_ratios(latest_metrics))
        features.update(self._calculate_growth_rates(df))

        # Calculate composite scores
        fundamental_score = self._calculate_fundamental_score(features)
        features['fundamental_score'] = fundamental_score

        return features

    def _get_latest_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Get most recent value for each metric"""
        if df.empty:
            return {}

        latest = df.loc[df.groupby('metric_key')['period_end'].idxmax()]
        metrics = {}

        for _, row in latest.iterrows():
            api_key = row['metric_key']
            internal_key = self.metric_mapping.get(api_key, api_key)
            metrics[internal_key] = row['value']

        return metrics

    def _calculate_profitability_ratios(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Calculate profitability ratios"""
        ratios = {}

        # Return on Equity
        if 'net_income' in metrics and 'equity' in metrics and metrics['equity'] > 0:
            ratios['roe'] = metrics['net_income'] / metrics['equity']

        # Return on Assets
        if 'net_income' in metrics and 'assets' in metrics and metrics['assets'] > 0:
            ratios['roa'] = metrics['net_income'] / metrics['assets']

        # Net Profit Margin
        if 'net_income' in metrics and 'revenue' in metrics and metrics['revenue'] > 0:
            ratios['net_margin'] = metrics['net_income'] / metrics['revenue']

        # EBIT Margin
        if 'ebit' in metrics and 'revenue' in metrics and metrics['revenue'] > 0:
            ratios['ebit_margin'] = metrics['ebit'] / metrics['revenue']

        # Operating Cash Flow Margin
        if 'operating_cf' in metrics and 'revenue' in metrics and metrics['revenue'] > 0:
            ratios['ocf_margin'] = metrics['operating_cf'] / metrics['revenue']

        return ratios

    def _calculate_efficiency_ratios(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Calculate efficiency ratios"""
        ratios = {}

        # Asset Turnover
        if 'revenue' in metrics and 'assets' in metrics and metrics['assets'] > 0:
            ratios['asset_turnover'] = metrics['revenue'] / metrics['assets']

        # Free Cash Flow (approx)
        if 'operating_cf' in metrics and 'capex' in metrics:
            ratios['free_cash_flow'] = metrics['operating_cf'] - metrics['capex']

            # FCF Margin
            if 'revenue' in metrics and metrics['revenue'] > 0:
                ratios['fcf_margin'] = ratios['free_cash_flow'] / metrics['revenue']

        return ratios

    def _calculate_leverage_ratios(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Calculate leverage ratios"""
        ratios = {}

        # Debt to Equity
        if 'net_debt' in metrics and 'equity' in metrics and metrics['equity'] > 0:
            ratios['debt_to_equity'] = metrics['net_debt'] / metrics['equity']

        # Debt to Assets
        if 'net_debt' in metrics and 'assets' in metrics and metrics['assets'] > 0:
            ratios['debt_to_assets'] = metrics['net_debt'] / metrics['assets']

        # Equity Ratio
        if 'equity' in metrics and 'assets' in metrics and metrics['assets'] > 0:
            ratios['equity_ratio'] = metrics['equity'] / metrics['assets']

        return ratios

    def _calculate_growth_rates(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate growth rates"""
        growth_rates = {}

        for metric in ['revenue', 'net_income', 'operating_cf']:
            api_metric = {v: k for k, v in self.metric_mapping.items()}.get(metric, metric)
            metric_data = df[df['metric_key'] == api_metric].copy()

            if len(metric_data) >= 2:
                metric_data = metric_data.sort_values('period_end')
                values = metric_data['value'].values

                # 1-year growth
                if len(values) >= 2:
                    latest, previous = values[-1], values[-2]
                    if previous != 0:
                        growth_rates[f'{metric}_growth_1y'] = (latest - previous) / abs(previous)

        return growth_rates

    def _calculate_fundamental_score(self, features: Dict[str, float]) -> float:
        """Calculate composite fundamental score (0-1)"""
        scores = []

        # Profitability component
        if 'roe' in features:
            roe = features['roe']
            if roe >= 0.15:
                scores.append(1.0)
            elif roe >= 0.10:
                scores.append(0.8)
            elif roe >= 0.05:
                scores.append(0.6)
            elif roe >= 0.0:
                scores.append(0.4)
            else:
                scores.append(0.2)

        # Efficiency component
        if 'roa' in features:
            roa = features['roa']
            if roa >= 0.10:
                scores.append(1.0)
            elif roa >= 0.05:
                scores.append(0.8)
            elif roa >= 0.02:
                scores.append(0.6)
            elif roa >= 0.0:
                scores.append(0.4)
            else:
                scores.append(0.2)

        # Financial health component
        if 'debt_to_equity' in features:
            dte = features['debt_to_equity']
            if dte <= 0.3:
                scores.append(1.0)
            elif dte <= 0.5:
                scores.append(0.8)
            elif dte <= 1.0:
                scores.append(0.6)
            elif dte <= 2.0:
                scores.append(0.4)
            else:
                scores.append(0.2)

        # Growth component
        if 'revenue_growth_1y' in features:
            growth = features['revenue_growth_1y']
            if growth >= 0.15:
                scores.append(1.0)
            elif growth >= 0.10:
                scores.append(0.8)
            elif growth >= 0.05:
                scores.append(0.6)
            elif growth >= 0.0:
                scores.append(0.4)
            else:
                scores.append(0.2)

        return np.mean(scores) if scores else 0.5