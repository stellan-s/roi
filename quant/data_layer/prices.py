from pathlib import Path
from datetime import datetime, timedelta, timezone
import pandas as pd, yfinance as yf

def fetch_prices(tickers, cache_dir: str, lookback_days: int = 500):
    import yfinance as yf, pandas as pd
    from pathlib import Path
    from datetime import datetime, timedelta, timezone
    import concurrent.futures

    def fetch_single_ticker(ticker, start_date, timeout_seconds=30):
        """Fetch price data for a single ticker with timeout"""
        try:
            print(f"Fetching {ticker}...")
            # Use thread-based timeout instead of signal
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(yf.download, ticker, start=start_date, auto_adjust=True, progress=False)
                try:
                    data = future.result(timeout=timeout_seconds)
                    if data is None or data.empty or "Close" not in data.columns:
                        print(f"⚠️ No data for {ticker}")
                        return None

                    close_series = data["Close"].dropna()
                    if close_series.empty:
                        print(f"⚠️ No Close data for {ticker}")
                        return None

                    tmp = close_series.reset_index()
                    tmp.columns = ["date", "close"]
                    tmp["ticker"] = ticker
                    print(f"✅ Fetched {len(tmp)} price points for {ticker}")
                    return tmp
                except concurrent.futures.TimeoutError:
                    print(f"⏰ Timeout ({timeout_seconds}s) for {ticker}")
                    return None
        except Exception as e:
            print(f"❌ Failed to fetch {ticker}: {e}")
            return None

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    start = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).date().isoformat()

    dfs = []
    for t in tickers:
        result = fetch_single_ticker(t, start)
        if result is not None:
            dfs.append(result)
    if not dfs: 
        return pd.DataFrame(columns=["date","ticker","close"])
    df = pd.concat(dfs).sort_values(["ticker","date"])
    df.to_parquet(Path(cache_dir)/"prices.parquet", index=False)
    return df
