"""
Data Quality and Sanity Check Utilities for ROI Trading System

Comprehensive validation to catch data quality issues, stale data,
missing tickers, and other common problems that lead to poor trading decisions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import warnings

class DataQualityError(Exception):
    """Custom exception for data quality issues"""
    pass

class DataQualityChecker:
    """
    Comprehensive data quality validation for the ROI trading system.

    Catches common issues:
    - Stale price data
    - Missing tickers
    - Duplicate recommendations
    - Inconsistent dates
    - Empty datasets
    - Suspicious portfolio weights
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.universe_tickers = self._load_universe()

    def _load_universe(self) -> List[str]:
        """Load expected universe of tickers"""
        try:
            if self.config and 'universe' in self.config:
                return self.config['universe']['tickers']
            else:
                # Fallback - load from file
                import yaml
                with open('quant/config/universe.yaml', 'r') as f:
                    universe = yaml.safe_load(f)
                return universe['tickers']
        except Exception:
            return []

    def check_price_data_quality(self, prices_df: pd.DataFrame) -> Dict:
        """
        Validate price data quality and freshness.

        Returns:
            Dict with validation results and warnings
        """
        issues = []
        stats = {}

        if prices_df.empty:
            raise DataQualityError("Price data is completely empty")

        # Basic stats
        stats['total_rows'] = len(prices_df)
        stats['unique_tickers'] = prices_df['ticker'].nunique()
        stats['date_range'] = (prices_df['date'].min(), prices_df['date'].max())

        # Check for expected tickers
        available_tickers = set(prices_df['ticker'].unique())
        expected_tickers = set(self.universe_tickers)
        missing_tickers = expected_tickers - available_tickers

        if missing_tickers:
            issues.append(f"Missing {len(missing_tickers)} tickers from universe: {sorted(list(missing_tickers))[:10]}")

        # Check data freshness (latest date should be within 3 days)
        latest_date = pd.to_datetime(prices_df['date'].max())
        days_old = (pd.Timestamp.now() - latest_date).days

        if days_old > 3:
            issues.append(f"Price data is {days_old} days old (latest: {latest_date.date()})")

        # Check for missing US vs Swedish stocks specifically
        us_tickers = [t for t in self.universe_tickers if not t.endswith('.ST')]
        swedish_tickers = [t for t in self.universe_tickers if t.endswith('.ST')]

        available_us = [t for t in available_tickers if not t.endswith('.ST')]
        available_swedish = [t for t in available_tickers if t.endswith('.ST')]

        if len(available_us) == 0 and len(us_tickers) > 0:
            issues.append(f"No US stocks available (expected {len(us_tickers)})")
        elif len(available_us) < len(us_tickers) * 0.5:
            issues.append(f"Only {len(available_us)}/{len(us_tickers)} US stocks available")

        # Check latest date completeness
        latest_data = prices_df[prices_df['date'] == latest_date]
        latest_tickers = set(latest_data['ticker'].unique())
        missing_from_latest = expected_tickers - latest_tickers

        if missing_from_latest:
            issues.append(f"Latest date missing {len(missing_from_latest)} tickers: {sorted(list(missing_from_latest))[:10]}")

        return {
            'status': 'pass' if not issues else 'warning',
            'issues': issues,
            'stats': stats,
            'missing_tickers': sorted(list(missing_tickers)),
            'missing_from_latest': sorted(list(missing_from_latest))
        }

    def check_recommendations_quality(self, recommendations_df: pd.DataFrame) -> Dict:
        """
        Validate recommendation data quality.

        Checks for duplicates, missing data, suspicious values, stale data.
        """
        issues = []
        stats = {}

        if recommendations_df.empty:
            raise DataQualityError("Recommendations data is completely empty")

        # Basic stats
        stats['total_rows'] = len(recommendations_df)
        stats['unique_tickers'] = recommendations_df['ticker'].nunique()
        stats['unique_dates'] = recommendations_df['date'].nunique()

        # Check for suspicious duplicates
        latest_date = recommendations_df['date'].max()
        latest_recs = recommendations_df[recommendations_df['date'] == latest_date]

        # Check for exact duplicates
        duplicate_check_cols = ['ticker', 'date', 'decision', 'expected_return', 'prob_positive']
        available_cols = [col for col in duplicate_check_cols if col in recommendations_df.columns]

        if len(available_cols) >= 3:
            duplicates = latest_recs.duplicated(subset=available_cols, keep=False)
            if duplicates.any():
                dup_count = duplicates.sum()
                dup_tickers = latest_recs[duplicates]['ticker'].unique()
                issues.append(f"Found {dup_count} duplicate recommendation rows for tickers: {list(dup_tickers)[:5]}")

        # Check for stale data being processed
        if stats['unique_dates'] > 5:
            issues.append(f"Processing {stats['unique_dates']} different dates - should focus on latest date only")

        # Check decision distribution
        if 'decision' in recommendations_df.columns:
            decisions = latest_recs['decision'].value_counts()
            stats['decisions'] = decisions.to_dict()

            if len(decisions) == 1:
                issues.append(f"Only one decision type: {decisions.index[0]} (no diversity)")

            # Check for too many Buy decisions (could indicate overfitting)
            if 'Buy' in decisions and decisions['Buy'] > len(latest_recs) * 0.5:
                issues.append(f"Suspiciously high Buy ratio: {decisions['Buy']}/{len(latest_recs)} ({decisions['Buy']/len(latest_recs)*100:.1f}%)")

        # Check portfolio weight distribution
        if 'portfolio_weight' in recommendations_df.columns:
            weights = latest_recs['portfolio_weight']
            active_weights = weights[weights > 0]

            if len(active_weights) > 0:
                stats['weight_stats'] = {
                    'total_weight': weights.sum(),
                    'active_positions': len(active_weights),
                    'max_weight': weights.max(),
                    'min_active_weight': active_weights.min(),
                    'avg_weight': active_weights.mean()
                }

                # Check for suspiciously small weights
                if active_weights.min() < 0.005:  # Less than 0.5%
                    tiny_weights = active_weights[active_weights < 0.005]
                    issues.append(f"Found {len(tiny_weights)} positions with tiny weights < 0.5%: min={tiny_weights.min():.4f}")

                # Check for total weight issues
                if weights.sum() > 1.1:
                    issues.append(f"Total portfolio weight exceeds 100%: {weights.sum():.2%}")
                elif weights.sum() < 0.1:
                    issues.append(f"Total portfolio weight too low: {weights.sum():.2%}")

        return {
            'status': 'pass' if not issues else 'warning',
            'issues': issues,
            'stats': stats
        }

    def check_execution_sanity(self,
                              recommendations_df: pd.DataFrame,
                              executed_trades: List[Dict],
                              portfolio_state: Dict) -> Dict:
        """
        Validate trade execution logic and portfolio state.
        """
        issues = []
        stats = {}

        # Check if we have buy recommendations but no executions
        latest_date = recommendations_df['date'].max()
        latest_recs = recommendations_df[recommendations_df['date'] == latest_date]

        buy_recs = latest_recs[latest_recs['decision'] == 'Buy'] if 'decision' in latest_recs.columns else pd.DataFrame()

        stats['buy_recommendations'] = len(buy_recs)
        stats['executed_trades'] = len(executed_trades)
        stats['available_cash'] = portfolio_state.get('cash', 0)
        stats['portfolio_value'] = portfolio_state.get('total_value', 0)

        if len(buy_recs) > 0 and len(executed_trades) == 0:
            issues.append(f"Have {len(buy_recs)} buy recommendations but 0 executed trades")

            # Diagnose why trades weren't executed
            if 'portfolio_weight' in buy_recs.columns:
                active_weights = buy_recs[buy_recs['portfolio_weight'] > 0]
                if len(active_weights) == 0:
                    issues.append("All buy recommendations have 0 portfolio weight")
                else:
                    min_cost = (active_weights['portfolio_weight'].min() * stats['portfolio_value'])
                    issues.append(f"Min trade cost: {min_cost:.0f}, Available cash: {stats['available_cash']:.0f}")

        # Check for over-concentration
        if 'ticker' in buy_recs.columns and len(buy_recs) > 0:
            ticker_counts = buy_recs['ticker'].value_counts()
            if ticker_counts.max() > 1:
                issues.append(f"Duplicate tickers in buy recommendations: {dict(ticker_counts[ticker_counts > 1])}")

        return {
            'status': 'pass' if not issues else 'warning',
            'issues': issues,
            'stats': stats
        }

    def generate_data_quality_report(self,
                                   prices_df: pd.DataFrame,
                                   recommendations_df: pd.DataFrame,
                                   executed_trades: List[Dict],
                                   portfolio_state: Dict) -> str:
        """
        Generate a comprehensive data quality report.
        """
        report_lines = ["🔍 DATA QUALITY REPORT", "=" * 50]

        # Price data check
        price_check = self.check_price_data_quality(prices_df)
        report_lines.append(f"\n📊 PRICE DATA: {price_check['status'].upper()}")
        if price_check['issues']:
            for issue in price_check['issues']:
                report_lines.append(f"  ⚠️ {issue}")
        report_lines.append(f"  📈 {price_check['stats']['unique_tickers']} tickers, {price_check['stats']['total_rows']} rows")

        # Recommendations check
        rec_check = self.check_recommendations_quality(recommendations_df)
        report_lines.append(f"\n🎯 RECOMMENDATIONS: {rec_check['status'].upper()}")
        if rec_check['issues']:
            for issue in rec_check['issues']:
                report_lines.append(f"  ⚠️ {issue}")
        if 'decisions' in rec_check['stats']:
            decisions_str = ", ".join([f"{k}={v}" for k, v in rec_check['stats']['decisions'].items()])
            report_lines.append(f"  📋 Decisions: {decisions_str}")

        # Execution check
        exec_check = self.check_execution_sanity(recommendations_df, executed_trades, portfolio_state)
        report_lines.append(f"\n⚡ EXECUTION: {exec_check['status'].upper()}")
        if exec_check['issues']:
            for issue in exec_check['issues']:
                report_lines.append(f"  ⚠️ {issue}")
        report_lines.append(f"  💰 Cash: {exec_check['stats']['available_cash']:,.0f} / Portfolio: {exec_check['stats']['portfolio_value']:,.0f}")

        return "\n".join(report_lines)

def validate_system_health(prices_df: pd.DataFrame,
                          recommendations_df: pd.DataFrame,
                          executed_trades: List[Dict],
                          portfolio_state: Dict,
                          config: Optional[Dict] = None) -> Tuple[bool, str]:
    """
    Quick system health check with pass/fail result.

    Returns:
        (is_healthy, report_text)
    """
    checker = DataQualityChecker(config)

    try:
        price_check = checker.check_price_data_quality(prices_df)
        rec_check = checker.check_recommendations_quality(recommendations_df)
        exec_check = checker.check_execution_sanity(recommendations_df, executed_trades, portfolio_state)

        total_issues = len(price_check['issues']) + len(rec_check['issues']) + len(exec_check['issues'])

        # System is healthy if we have < 3 total issues and no critical failures
        is_healthy = total_issues < 3

        report = checker.generate_data_quality_report(prices_df, recommendations_df, executed_trades, portfolio_state)

        return is_healthy, report

    except Exception as e:
        return False, f"🚨 DATA QUALITY CHECK FAILED: {str(e)}"