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


def fetch_precious_metals_sentiment(cache_dir: str, lookback_days: int = 500,
                                   gold_symbol: str = "GLD", silver_symbol: str = "SLV") -> pd.DataFrame:
    """
    Fetch gold and silver data for market sentiment analysis.

    Args:
        cache_dir: Directory for caching data
        lookback_days: Number of days to fetch
        gold_symbol: Gold ETF symbol (default: GLD)
        silver_symbol: Silver ETF symbol (default: SLV)

    Returns:
        DataFrame with precious metals sentiment indicators
    """

    cache_path = Path(cache_dir) / "macro" / "precious_metals_sentiment.parquet"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Check cache freshness
    if cache_path.exists():
        try:
            cached_data = pd.read_parquet(cache_path)
            cached_data['date'] = pd.to_datetime(cached_data['date'])

            latest_cached = cached_data['date'].max()
            today = pd.Timestamp.now().date()

            if latest_cached.date() >= today - timedelta(days=1):
                print(f"🥇 Using cached precious metals sentiment (latest: {latest_cached.date()})")
                cutoff_date = today - timedelta(days=lookback_days)
                recent_data = cached_data[cached_data['date'].dt.date >= cutoff_date]
                return recent_data

        except Exception as e:
            print(f"⚠️ Error reading precious metals cache: {e}")

    # Fetch fresh data
    print("🥇 Fetching gold and silver data for sentiment analysis...")

    try:
        start_date = (datetime.now() - timedelta(days=lookback_days + 30)).strftime('%Y-%m-%d')

        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Fetch both gold and silver in parallel
            gold_future = executor.submit(
                yf.download, gold_symbol, start=start_date, auto_adjust=True, progress=False
            )
            silver_future = executor.submit(
                yf.download, silver_symbol, start=start_date, auto_adjust=True, progress=False
            )

            gold_data = gold_future.result(timeout=30)
            silver_data = silver_future.result(timeout=30)

        if gold_data.empty or silver_data.empty:
            raise ValueError("No precious metals data received")

        # Process gold data - handle MultiIndex properly
        gold_df = gold_data.reset_index()
        if isinstance(gold_df.columns, pd.MultiIndex):
            # Extract the 'Close' column specifically for this symbol
            gold_df = pd.DataFrame({
                'date': gold_df.index if 'Date' not in gold_df.columns else gold_df['Date'],
                'gold_close': gold_df[('Close', gold_symbol)]
            })
        else:
            gold_df.columns = [str(col).lower() for col in gold_df.columns]
            gold_df = gold_df.rename(columns={'close': 'gold_close'})

        # Process silver data - handle MultiIndex properly
        silver_df = silver_data.reset_index()
        if isinstance(silver_df.columns, pd.MultiIndex):
            # Extract the 'Close' column specifically for this symbol
            silver_df = pd.DataFrame({
                'date': silver_df.index if 'Date' not in silver_df.columns else silver_df['Date'],
                'silver_close': silver_df[('Close', silver_symbol)]
            })
        else:
            silver_df.columns = [str(col).lower() for col in silver_df.columns]
            silver_df = silver_df.rename(columns={'close': 'silver_close'})

        # Merge gold and silver data
        metals_df = pd.merge(
            gold_df[['date', 'gold_close']],
            silver_df[['date', 'silver_close']],
            on='date', how='inner'
        )

        # Calculate sentiment indicators

        # 1. Gold momentum (risk-on/risk-off indicator)
        metals_df = metals_df.copy()  # Ensure we can modify the DataFrame
        metals_df['gold_return_1d'] = metals_df['gold_close'].pct_change()
        metals_df['gold_return_5d'] = metals_df['gold_close'].pct_change(5)
        metals_df['gold_return_20d'] = metals_df['gold_close'].pct_change(20)

        # 2. Silver momentum
        metals_df['silver_return_1d'] = metals_df['silver_close'].pct_change()
        metals_df['silver_return_5d'] = metals_df['silver_close'].pct_change(5)
        metals_df['silver_return_20d'] = metals_df['silver_close'].pct_change(20)

        # 3. Gold/Silver ratio (risk appetite indicator)
        metals_df['gold_silver_ratio'] = metals_df['gold_close'] / metals_df['silver_close']
        metals_df['gs_ratio_sma_10'] = metals_df['gold_silver_ratio'].rolling(10).mean()
        metals_df['gs_ratio_vs_sma'] = metals_df['gold_silver_ratio'] / metals_df['gs_ratio_sma_10'] - 1

        # 4. Precious metals sentiment regime
        def classify_precious_metals_sentiment(row):
            gold_20d = row['gold_return_20d']
            gs_ratio = row['gold_silver_ratio']

            # Risk-off: Gold rising AND high gold/silver ratio
            if gold_20d > 0.03 and gs_ratio > 85:
                return 'risk_off_strong'
            elif gold_20d > 0.01 or gs_ratio > 80:
                return 'risk_off_mild'
            # Risk-on: Gold declining AND low gold/silver ratio
            elif gold_20d < -0.02 and gs_ratio < 75:
                return 'risk_on_strong'
            elif gold_20d < -0.005 or gs_ratio < 78:
                return 'risk_on_mild'
            else:
                return 'neutral'

        metals_df['metals_sentiment'] = metals_df.apply(classify_precious_metals_sentiment, axis=1)

        # 5. Correlation with risk assets (inverse correlation = safe haven behavior)
        metals_df['gold_sma_10'] = metals_df['gold_close'].rolling(10).mean()
        metals_df['gold_vs_sma'] = metals_df['gold_close'] / metals_df['gold_sma_10'] - 1

        # Clean up
        metals_df = metals_df.dropna()

        print(f"✅ Fetched {len(metals_df)} precious metals sentiment points from {metals_df['date'].min().date()} to {metals_df['date'].max().date()}")

        # Cache the data
        metals_df.to_parquet(cache_path, index=False)
        print(f"💾 Cached precious metals sentiment to {cache_path}")

        # Filter to requested lookback period
        cutoff_date = datetime.now().date() - timedelta(days=lookback_days)
        recent_data = metals_df[metals_df['date'].dt.date >= cutoff_date]

        return recent_data

    except Exception as e:
        print(f"❌ Failed to fetch precious metals data: {e}")
        # Return empty DataFrame with correct structure
        return pd.DataFrame(columns=[
            'date', 'gold_close', 'silver_close', 'gold_return_1d', 'gold_return_5d', 'gold_return_20d',
            'silver_return_1d', 'silver_return_5d', 'silver_return_20d', 'gold_silver_ratio',
            'gs_ratio_sma_10', 'gs_ratio_vs_sma', 'metals_sentiment', 'gold_sma_10', 'gold_vs_sma'
        ])


def fetch_macro_indicators(cache_dir: str, lookback_days: int = 500) -> Dict[str, pd.DataFrame]:
    """
    Fetch multiple macro indicators in parallel.

    Returns:
        Dictionary with keys: 'vix', 'precious_metals', 'usd_sek', 'oil', etc.
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

    # Add precious metals sentiment
    metals_data = fetch_precious_metals_sentiment(cache_dir, lookback_days)
    if not metals_data.empty:
        indicators['precious_metals'] = metals_data
        print(f"✅ Precious Metals: {len(metals_data)} data points")
    else:
        print("❌ Precious Metals: No data available")

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


def get_latest_precious_metals_sentiment(metals_data: pd.DataFrame) -> Dict:
    """
    Get the latest precious metals sentiment information for reporting.

    Returns:
        Dict with current gold/silver sentiment and market implications
    """

    if metals_data.empty:
        return {
            'gold_level': None,
            'silver_level': None,
            'metals_sentiment': 'unknown',
            'gold_silver_ratio': None,
            'explanation': 'Precious metals data not available'
        }

    latest = metals_data.iloc[-1]
    gold_level = latest['gold_close']
    silver_level = latest['silver_close']
    gold_silver_ratio = latest['gold_silver_ratio']
    metals_sentiment = latest['metals_sentiment']

    # Create explanation
    sentiment_explanations = {
        'risk_off_strong': f"Strong flight to gold (GLD: ${gold_level:.1f}, G/S ratio: {gold_silver_ratio:.1f}) - defensive positioning favored",
        'risk_off_mild': f"Mild safe-haven demand (GLD: ${gold_level:.1f}, G/S ratio: {gold_silver_ratio:.1f}) - cautious sentiment",
        'neutral': f"Balanced precious metals sentiment (GLD: ${gold_level:.1f}, G/S ratio: {gold_silver_ratio:.1f}) - no clear directional bias",
        'risk_on_mild': f"Mild risk appetite (GLD: ${gold_level:.1f}, G/S ratio: {gold_silver_ratio:.1f}) - modest risk-on sentiment",
        'risk_on_strong': f"Strong risk appetite (GLD: ${gold_level:.1f}, G/S ratio: {gold_silver_ratio:.1f}) - growth assets favored"
    }

    explanation = sentiment_explanations.get(metals_sentiment, f"GLD: ${gold_level:.1f}, SLV: ${silver_level:.1f} - sentiment unknown")

    # Add momentum context
    gold_20d = latest.get('gold_return_20d', 0)
    if gold_20d > 0.02:  # 2% gain over 20 days
        explanation += " (gold rally)"
    elif gold_20d < -0.02:  # 2% decline over 20 days
        explanation += " (gold decline)"

    return {
        'gold_level': gold_level,
        'silver_level': silver_level,
        'gold_silver_ratio': gold_silver_ratio,
        'metals_sentiment': metals_sentiment,
        'explanation': explanation,
        'gold_return_20d': latest.get('gold_return_20d', 0),
        'silver_return_20d': latest.get('silver_return_20d', 0),
        'gs_ratio_vs_sma': latest.get('gs_ratio_vs_sma', 0)
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