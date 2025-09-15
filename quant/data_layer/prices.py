from pathlib import Path
from datetime import datetime, timedelta, timezone
import pandas as pd, yfinance as yf

def fetch_prices(tickers, cache_dir: str, lookback_days: int = 500):
    import yfinance as yf, pandas as pd
    from pathlib import Path
    from datetime import datetime, timedelta, timezone

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    start = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).date().isoformat()

    dfs = []
    for t in tickers:
        try:
            data = yf.download(t, start=start, auto_adjust=True, progress=False)
            if data is None or data.empty or "Close" not in data.columns:
                print(f"⚠️ Ingen data för {t}")
                continue
            # Extract Close price and reshape to long format
            close_series = data["Close"].dropna()
            if close_series.empty:
                print(f"⚠️ Ingen Close data för {t}")
                continue
            tmp = close_series.reset_index()
            tmp.columns = ["date", "close"]  # Ensure proper column names
            tmp["ticker"] = t
            dfs.append(tmp)
        except Exception as e:
            print(f"❌ Misslyckades hämta {t}: {e}")
    if not dfs: 
        return pd.DataFrame(columns=["date","ticker","close"])
    df = pd.concat(dfs).sort_values(["ticker","date"])
    df.to_parquet(Path(cache_dir)/"prices.parquet", index=False)
    return df
