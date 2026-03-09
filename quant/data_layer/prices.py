from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import concurrent.futures
import time

import pandas as pd
import yfinance as yf


def _is_rate_limit_error(exc: Exception) -> bool:
    name = exc.__class__.__name__
    text = str(exc).lower()
    return "ratelimit" in name.lower() or "too many requests" in text or "rate limit" in text


def _extract_close_series(data: pd.DataFrame) -> pd.Series | None:
    if data is None or data.empty:
        return None

    # Typical shape: single-level columns with "Close".
    if "Close" in data.columns:
        series = data["Close"]
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]
        return series.dropna()

    # yfinance can return MultiIndex columns in some cases.
    if isinstance(data.columns, pd.MultiIndex):
        try:
            if "Close" in data.columns.get_level_values(0):
                close_df = data.xs("Close", axis=1, level=0)
                if isinstance(close_df, pd.DataFrame) and not close_df.empty:
                    return close_df.iloc[:, 0].dropna()
            if "Close" in data.columns.get_level_values(-1):
                close_df = data.xs("Close", axis=1, level=-1)
                if isinstance(close_df, pd.DataFrame) and not close_df.empty:
                    return close_df.iloc[:, 0].dropna()
        except Exception:
            return None

    return None


def _normalize_price_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(columns=["date", "ticker", "close"])

    df = raw_df.copy()
    required = {"date", "ticker", "close"}
    if not required.issubset(df.columns):
        return pd.DataFrame(columns=["date", "ticker", "close"])

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["ticker"] = df["ticker"].astype(str)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df[df["date"].notna() & df["ticker"].notna() & df["close"].notna()]
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


def _load_cache(cache_path: Path) -> pd.DataFrame:
    if not cache_path.exists():
        return pd.DataFrame(columns=["date", "ticker", "close"])
    try:
        cached = pd.read_parquet(cache_path)
    except Exception:
        return pd.DataFrame(columns=["date", "ticker", "close"])
    return _normalize_price_frame(cached)


def _cached_ticker_slice(cached_df: pd.DataFrame, ticker: str, start_date: str) -> pd.DataFrame | None:
    if cached_df.empty:
        return None
    start_ts = pd.to_datetime(start_date).normalize()
    out = cached_df[(cached_df["ticker"] == ticker) & (cached_df["date"] >= start_ts)].copy()
    return out if not out.empty else None


def fetch_prices(
    tickers,
    cache_dir: str,
    lookback_days: int = 500,
    *,
    max_retries: int = 4,
    base_backoff_seconds: float = 2.0,
    timeout_seconds: int = 30,
):
    """
    Fetch historical prices with retry/backoff and cache fallback.

    Resilience policy:
    - Retries transient failures (including Yahoo rate limits).
    - Falls back to per-ticker cached data when live fetch fails.
    - Returns all successfully fetched/fallback tickers without failing the full run.
    """

    cache_path = Path(cache_dir) / "prices.parquet"
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    cached_all = _load_cache(cache_path)
    start = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).date().isoformat()

    def fetch_single_ticker(ticker: str) -> pd.DataFrame | None:
        last_error: Exception | None = None

        for attempt in range(max_retries + 1):
            try:
                print(f"Fetching {ticker}...")
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        yf.download,
                        ticker,
                        start=start,
                        auto_adjust=True,
                        progress=False,
                    )
                    data = future.result(timeout=timeout_seconds)

                close_series = _extract_close_series(data)
                if close_series is None or close_series.empty:
                    print(f"⚠️ No Close data for {ticker}")
                    break

                tmp = close_series.reset_index()
                tmp.columns = ["date", "close"]
                tmp["ticker"] = ticker
                tmp = _normalize_price_frame(tmp)
                print(f"✅ Fetched {len(tmp)} price points for {ticker}")
                return tmp

            except concurrent.futures.TimeoutError as exc:
                last_error = exc
                print(f"⏰ Timeout ({timeout_seconds}s) for {ticker} (attempt {attempt + 1}/{max_retries + 1})")
            except Exception as exc:
                last_error = exc
                if _is_rate_limit_error(exc):
                    print(f"⏳ Rate limited for {ticker} (attempt {attempt + 1}/{max_retries + 1})")
                else:
                    print(f"❌ Failed to fetch {ticker} (attempt {attempt + 1}/{max_retries + 1}): {exc}")

            if attempt < max_retries:
                sleep_s = base_backoff_seconds * (2**attempt)
                time.sleep(sleep_s)

        fallback = _cached_ticker_slice(cached_all, ticker, start)
        if fallback is not None:
            print(
                f"♻️ Using cached fallback for {ticker} ({len(fallback)} rows)"
                + (f" after error: {last_error}" if last_error else "")
            )
            return fallback

        if last_error is not None:
            print(f"⚠️ No fallback available for {ticker}; last error: {last_error}")
        return None

    dfs = []
    for ticker in tickers:
        result = fetch_single_ticker(ticker)
        if result is not None and not result.empty:
            dfs.append(result)

    if not dfs:
        # Last resort: return filtered cache for requested tickers.
        if not cached_all.empty:
            start_ts = pd.to_datetime(start).normalize()
            fallback = cached_all[
                (cached_all["ticker"].isin([str(t) for t in tickers])) & (cached_all["date"] >= start_ts)
            ].copy()
            if not fallback.empty:
                print(f"♻️ Returning cached prices for {fallback['ticker'].nunique()} tickers")
                return fallback.sort_values(["ticker", "date"]).reset_index(drop=True)
        return pd.DataFrame(columns=["date", "ticker", "close"])

    df = pd.concat(dfs, ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)
    # Dedupe by ticker/date to keep deterministic history when mixing live + cache fallback.
    df = df.drop_duplicates(subset=["ticker", "date"], keep="last").reset_index(drop=True)
    df.to_parquet(cache_path, index=False)
    return df
