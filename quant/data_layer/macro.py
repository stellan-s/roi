"""
Macro-economic data fetching for market indicators.

Handles fetching of:
- VIX (Volatility Index) - Market fear/greed gauge
- Currency data (USD/SEK)
- Bond yields
- Commodity prices

All data cached for performance and reliability.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
import concurrent.futures
from datetime import datetime, timedelta


def fetch_vix(cache_dir: str, lookback_days: int = 500) -> pd.DataFrame:
    """
    Fetch VIX (CBOE Volatility Index) data.

    VIX measures implied volatility of S&P 500 options and serves as
    a "fear gauge" for market sentiment.

    Args:
        cache_dir: Directory for caching data
        lookback_days: Number of days to fetch

    Returns:
        DataFrame with columns: date, vix_close, vix_change, vix_regime
    """

    cache_path = Path(cache_dir) / "macro" / "vix_data.parquet"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Check cache freshness
    if cache_path.exists():
        try:
            cached_data = pd.read_parquet(cache_path)
            cached_data['date'] = pd.to_datetime(cached_data['date'])

            # Check if cache is recent (within last day)
            latest_cached = cached_data['date'].max()
            today = pd.Timestamp.now().date()

            if latest_cached.date() >= today - timedelta(days=1):
                print(f"📈 Using cached VIX data (latest: {latest_cached.date()})")

                # Filter to requested lookback period
                cutoff_date = today - timedelta(days=lookback_days)
                recent_data = cached_data[cached_data['date'].dt.date >= cutoff_date]
                return recent_data

        except Exception as e:
            print(f"⚠️ Error reading VIX cache: {e}")

    # Fetch fresh VIX data
    print("📈 Fetching VIX data from Yahoo Finance...")

    try:
        # VIX symbol is ^VIX
        start_date = (datetime.now() - timedelta(days=lookback_days + 30)).strftime('%Y-%m-%d')

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(
                yf.download,
                "^VIX",
                start=start_date,
                auto_adjust=True,
                progress=False
            )
            vix_data = future.result(timeout=30)  # 30 second timeout

        if vix_data.empty:
            raise ValueError("No VIX data received")

        # Process VIX data - handle multi-level columns
        vix_df = vix_data.reset_index()

        # Handle multi-level columns from yfinance
        if isinstance(vix_df.columns, pd.MultiIndex):
            # Flatten multi-level columns and keep only the price type (first level)
            vix_df.columns = [col[0] if isinstance(col, tuple) else col for col in vix_df.columns]

        # Convert column names to lowercase
        vix_df.columns = [str(col).lower() for col in vix_df.columns]

        # Rename columns for consistency
        column_mapping = {
            'date': 'date',
            'close': 'vix_close'
        }

        vix_df = vix_df.rename(columns=column_mapping)

        # Ensure we have the required columns
        if 'vix_close' not in vix_df.columns:
            if 'close' in vix_df.columns:
                vix_df['vix_close'] = vix_df['close']
            else:
                raise ValueError(f"Could not find Close price in VIX data. Available columns: {vix_df.columns.tolist()}")

        # Keep only needed columns
        vix_df = vix_df[['date', 'vix_close']].copy()

        # Calculate additional VIX features
        vix_df['vix_change'] = vix_df['vix_close'].pct_change()
        vix_df['vix_sma_10'] = vix_df['vix_close'].rolling(10).mean()
        vix_df['vix_vs_sma'] = vix_df['vix_close'] / vix_df['vix_sma_10'] - 1

        # VIX regime classification
        # Based on historical VIX levels:
        # < 20: Low fear (bull market conditions)
        # 20-30: Moderate fear (neutral market)
        # > 30: High fear (bear market conditions)
        # > 40: Extreme fear (crisis conditions)

        def classify_vix_regime(vix_level):
            if vix_level < 20:
                return 'low_fear'  # Bull market indicator
            elif vix_level < 30:
                return 'moderate_fear'  # Neutral market
            elif vix_level < 40:
                return 'high_fear'  # Bear market indicator
            else:
                return 'extreme_fear'  # Crisis/panic

        vix_df['vix_regime'] = vix_df['vix_close'].apply(classify_vix_regime)

        # Add VIX momentum (5-day change)
        vix_df['vix_momentum_5d'] = vix_df['vix_close'].pct_change(5)

        # VIX mean reversion indicator (distance from 20-day mean)
        vix_df['vix_sma_20'] = vix_df['vix_close'].rolling(20).mean()
        vix_df['vix_mean_reversion'] = (vix_df['vix_close'] - vix_df['vix_sma_20']) / vix_df['vix_sma_20']

        # Clean up
        vix_df = vix_df.dropna()

        print(f"✅ Fetched {len(vix_df)} VIX data points from {vix_df['date'].min().date()} to {vix_df['date'].max().date()}")

        # Cache the data
        vix_df.to_parquet(cache_path, index=False)
        print(f"💾 Cached VIX data to {cache_path}")

        # Filter to requested lookback period
        cutoff_date = datetime.now().date() - timedelta(days=lookback_days)
        recent_data = vix_df[vix_df['date'].dt.date >= cutoff_date]

        return recent_data

    except Exception as e:
        print(f"❌ Failed to fetch VIX data: {e}")
        # Return empty DataFrame with correct structure
        return pd.DataFrame(columns=[
            'date', 'vix_close', 'vix_change', 'vix_sma_10', 'vix_vs_sma',
            'vix_regime', 'vix_momentum_5d', 'vix_sma_20', 'vix_mean_reversion'
        ])


def fetch_macro_indicators(cache_dir: str, lookback_days: int = 500) -> Dict[str, pd.DataFrame]:
    """
    Fetch multiple macro indicators in parallel.

    Returns:
        Dictionary with keys: 'vix', 'usd_sek', 'oil', etc.
    """

    indicators = {}

    # Start with VIX
    print("📊 Fetching macro indicators...")

    vix_data = fetch_vix(cache_dir, lookback_days)
    if not vix_data.empty:
        indicators['vix'] = vix_data
        print(f"✅ VIX: {len(vix_data)} data points")
    else:
        print("❌ VIX: No data available")

    # TODO: Add more indicators (USD/SEK, oil, etc.)
    # indicators['usd_sek'] = fetch_currency('USDSEK=X', cache_dir, lookback_days)
    # indicators['oil'] = fetch_commodity('CL=F', cache_dir, lookback_days)

    return indicators


def get_latest_vix_regime(vix_data: pd.DataFrame) -> Dict:
    """
    Get the latest VIX regime information for reporting.

    Returns:
        Dict with current VIX level, regime, and explanation
    """

    if vix_data.empty:
        return {
            'vix_level': None,
            'vix_regime': 'unknown',
            'explanation': 'VIX data not available'
        }

    latest = vix_data.iloc[-1]
    vix_level = latest['vix_close']
    vix_regime = latest['vix_regime']

    # Create explanation
    regime_explanations = {
        'low_fear': f"VIX {vix_level:.1f} signals low market fear - favorable for risk assets",
        'moderate_fear': f"VIX {vix_level:.1f} indicates moderate uncertainty - neutral market conditions",
        'high_fear': f"VIX {vix_level:.1f} shows elevated fear - defensive positioning recommended",
        'extreme_fear': f"VIX {vix_level:.1f} indicates extreme fear/panic - crisis conditions"
    }

    explanation = regime_explanations.get(vix_regime, f"VIX {vix_level:.1f} - regime unknown")

    # Add momentum context
    if 'vix_momentum_5d' in latest:
        momentum = latest['vix_momentum_5d']
        if momentum > 0.1:  # 10% increase
            explanation += " (rising fear)"
        elif momentum < -0.1:  # 10% decrease
            explanation += " (declining fear)"

    return {
        'vix_level': vix_level,
        'vix_regime': vix_regime,
        'explanation': explanation,
        'momentum_5d': latest.get('vix_momentum_5d', 0),
        'vs_sma_10': latest.get('vix_vs_sma', 0)
    }


if __name__ == "__main__":
    # Test VIX fetching
    test_data = fetch_vix("data/cache", lookback_days=100)
    print(f"\nTest results: {len(test_data)} VIX records")
    if not test_data.empty:
        print(f"Latest VIX: {test_data.iloc[-1]['vix_close']:.1f}")
        print(f"Latest regime: {test_data.iloc[-1]['vix_regime']}")

        regime_info = get_latest_vix_regime(test_data)
        print(f"Regime info: {regime_info}")