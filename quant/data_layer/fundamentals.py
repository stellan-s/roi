"""
Fundamentals data layer for ROI system
Fetches fundamental metrics from the EquityAPI on localhost:8000
"""

import requests
import pandas as pd
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

class FundamentalsAPI:
    """Client for EquityAPI fundamentals data"""

    def __init__(self, base_url: str = "http://localhost:8000", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})

    def _request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make authenticated request to API"""
        url = f"{self.base_url}{endpoint}"
        try:
            response = self.session.get(url, params=params or {})
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {url} - {e}")
            raise

    def get_available_metrics(self) -> Dict[str, str]:
        """Get list of available fundamental metrics"""
        return self._request("/v1/fundamentals/metrics")

    def get_fundamentals(self,
                        symbol: Optional[str] = None,
                        isin: Optional[str] = None,
                        lei: Optional[str] = None,
                        metrics: List[str] = None,
                        years: int = 5,
                        period_type: str = "annual") -> Dict:
        """
        Get fundamental metrics for a company

        Args:
            symbol: Stock symbol (e.g., 'VOLV-B.ST')
            isin: ISIN code
            lei: LEI code
            metrics: List of metric keys to fetch
            years: Number of years of historical data
            period_type: 'annual', 'quarterly', or 'all'
        """
        if not any([symbol, isin, lei]):
            raise ValueError("Must provide symbol, isin, or lei")

        params = {
            "years": years,
            "period_type": period_type
        }

        if symbol:
            params["symbol"] = symbol
        if isin:
            params["isin"] = isin
        if lei:
            params["lei"] = lei
        if metrics:
            params["metrics"] = ",".join(metrics)

        return self._request("/v1/fundamentals", params)

    def get_security_by_symbol(self, symbol: str) -> Dict:
        """Get security information by symbol"""
        return self._request(f"/v1/securities/symbol/{symbol}")

class FundamentalsDataLayer:
    """Main fundamentals data layer with caching and processing"""

    def __init__(self, config: Dict, enabled: bool = True):
        self.enabled = enabled
        self.config = config

        if self.enabled and config:
            api_config = config.get('fundamentals_api', {})
            self.api = FundamentalsAPI(
                base_url=api_config.get('base_url', 'http://localhost:8000'),
                api_key=api_config.get('api_key')
            )
            self.cache_hours = api_config.get('cache_hours', 24)
            self.default_metrics = api_config.get('default_metrics', [])
        else:
            self.api = None

        self._cache = {}
        self._cache_timestamps = {}

    def _get_cache_key(self, symbol: str, metrics: List[str], years: int, period_type: str) -> str:
        """Generate cache key for fundamentals data"""
        metrics_str = ",".join(sorted(metrics)) if metrics else "default"
        return f"{symbol}_{metrics_str}_{years}_{period_type}"

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self._cache_timestamps:
            return False

        cache_time = self._cache_timestamps[cache_key]
        return datetime.now() - cache_time < timedelta(hours=self.cache_hours)

    def get_fundamentals_for_symbol(self,
                                   symbol: str,
                                   metrics: List[str] = None,
                                   years: int = 5,
                                   period_type: str = "annual",
                                   use_cache: bool = True) -> pd.DataFrame:
        """
        Get fundamentals data for a symbol as DataFrame

        Returns:
            DataFrame with columns: period_end, metric_key, value, year, quarter
            Empty DataFrame if fundamentals are disabled or unavailable
        """
        # Return empty DataFrame if fundamentals are disabled
        if not self.enabled or not self.api:
            logger.debug(f"Fundamentals disabled for {symbol}")
            return pd.DataFrame()

        cache_key = self._get_cache_key(symbol, metrics or [], years, period_type)

        if use_cache and self._is_cache_valid(cache_key):
            logger.info(f"Using cached fundamentals for {symbol}")
            return self._cache[cache_key]

        try:
            # Convert Swedish symbols to base format for API (VOLV-B.ST -> VOLV B)
            api_symbol = symbol.replace('.ST', '').replace('-', ' ')

            # Get fundamentals directly by symbol (no need for company_id mapping)
            data = self.api.get_fundamentals(
                symbol=api_symbol,
                metrics=metrics or self.default_metrics,
                years=years,
                period_type=period_type
            )

            # Convert to DataFrame
            df = self._process_fundamentals_response(data)

            if use_cache:
                self._cache[cache_key] = df
                self._cache_timestamps[cache_key] = datetime.now()

            logger.info(f"Fetched fundamentals for {symbol}: {len(df)} records")
            return df

        except Exception as e:
            logger.debug(f"Fundamentals not available for {symbol}: {e}")
            return pd.DataFrame()

    def _get_security_by_symbol(self, symbol: str) -> Optional[Dict]:
        """Get security information by symbol"""
        try:
            return self.api.get_security_by_symbol(symbol)
        except Exception as e:
            logger.warning(f"Could not find security for {symbol}: {e}")
            return None

    def _process_fundamentals_response(self, response: Dict) -> pd.DataFrame:
        """Process API response into standardized DataFrame"""
        if 'metrics' not in response:
            return pd.DataFrame()

        records = []

        # Handle the actual API response structure
        for metric_data in response.get('metrics', []):
            metric_key = metric_data.get('metric_key')

            for value_data in metric_data.get('values', []):
                if value_data.get('value'):
                    try:
                        value = float(value_data['value'])
                        period_end = value_data.get('period_end')
                        period_start = value_data.get('period_start')

                        # Extract year from period_end
                        year = None
                        quarter = None
                        if period_end:
                            year = pd.to_datetime(period_end).year

                        records.append({
                            'period_end': period_end,
                            'period_start': period_start,
                            'metric_key': metric_key,
                            'value': value,
                            'year': year,
                            'quarter': quarter,
                            'currency': value_data.get('currency'),
                            'period_type': value_data.get('period_type')
                        })
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Could not parse value {value_data.get('value')} for {metric_key}: {e}")

        df = pd.DataFrame(records)
        if not df.empty:
            df['period_end'] = pd.to_datetime(df['period_end'])
            df = df.sort_values(['period_end', 'metric_key'])

        return df

    def get_latest_metrics(self, symbol: str, metrics: List[str] = None) -> Dict[str, float]:
        """Get the most recent values for specified metrics"""
        df = self.get_fundamentals_for_symbol(symbol, metrics=metrics)

        if df.empty:
            return {}

        # Get latest period for each metric
        latest = df.loc[df.groupby('metric_key')['period_end'].idxmax()]
        return dict(zip(latest['metric_key'], latest['value']))

    def get_metric_history(self, symbol: str, metric_key: str, years: int = 5) -> pd.Series:
        """Get historical values for a specific metric"""
        df = self.get_fundamentals_for_symbol(symbol, metrics=[metric_key], years=years)

        metric_data = df[df['metric_key'] == metric_key].copy()
        if metric_data.empty:
            return pd.Series(dtype=float)

        metric_data = metric_data.set_index('period_end')['value']
        return metric_data.sort_index()

    def get_fundamentals_batch(self, symbols: List[str], metrics: List[str] = None) -> Dict[str, pd.DataFrame]:
        """Get fundamentals for multiple symbols"""
        results = {}
        for symbol in symbols:
            try:
                results[symbol] = self.get_fundamentals_for_symbol(symbol, metrics=metrics)
            except Exception as e:
                logger.error(f"Failed to fetch fundamentals for {symbol}: {e}")
                results[symbol] = pd.DataFrame()

        return results