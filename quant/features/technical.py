import pandas as pd

def compute_technical_features(prices: pd.DataFrame, sma_long=200, momentum_window=252) -> pd.DataFrame:
    # prices: date,ticker,close
    df = prices.sort_values(["ticker","date"]).copy()
    df["ret"] = df.groupby("ticker")["close"].pct_change()
    df["sma_long"] = df.groupby("ticker")["close"].transform(lambda s: s.rolling(sma_long, min_periods=20).mean())
    df["above_sma"] = (df["close"] > df["sma_long"]).astype(int)
    df["mom"] = df.groupby("ticker")["close"].transform(lambda s: s.pct_change(momentum_window))
    # normalisera enkelt: z-score per dag på mom
    df["mom_rank"] = df.groupby("date")["mom"].rank(pct=True)
    feats = df[["date","ticker","close","above_sma","mom","mom_rank"]].dropna()
    return feats
