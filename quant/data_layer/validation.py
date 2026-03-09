"""Data normalization and lightweight provider validation helpers."""

from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd


def _ensure_columns(df: pd.DataFrame, required: Sequence[str], dataset: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{dataset} missing required columns: {missing}")


def normalize_prices_schema(prices: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize price data schema for downstream consumers."""
    if prices is None:
        return pd.DataFrame(columns=["date", "ticker", "close"])

    prices = prices.copy()
    _ensure_columns(prices, ["date", "ticker", "close"], "prices")
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce").dt.normalize()
    prices["ticker"] = prices["ticker"].astype(str)
    prices["close"] = pd.to_numeric(prices["close"], errors="coerce")
    prices = prices[prices["date"].notna() & prices["ticker"].notna() & prices["close"].notna()]
    return prices.sort_values(["ticker", "date"]).reset_index(drop=True)


def normalize_news_schema(news: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize news schema used by sentiment processing."""
    if news is None:
        return pd.DataFrame(columns=["published", "title", "summary"])

    news = news.copy()
    _ensure_columns(news, ["published"], "news")
    if "title" not in news.columns:
        news["title"] = ""
    if "summary" not in news.columns:
        news["summary"] = ""

    news["published"] = pd.to_datetime(news["published"], errors="coerce", utc=True).dt.tz_convert(None)
    news = news[news["published"].notna()]
    return news.sort_values("published").reset_index(drop=True)


def filter_as_of(df: pd.DataFrame, column: str, as_of: pd.Timestamp) -> pd.DataFrame:
    """Slice a DataFrame to information available at or before `as_of`."""
    if df is None or df.empty:
        return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()

    out = df.copy()
    out[column] = pd.to_datetime(out[column], errors="coerce")
    out = out[out[column].notna()]
    return out[out[column] <= pd.Timestamp(as_of)].copy()


def is_fresh(df: pd.DataFrame, ts_col: str, as_of: pd.Timestamp, max_age_days: int) -> bool:
    """Check if latest timestamp is fresh enough for a runtime guard."""
    if df is None or df.empty or ts_col not in df.columns:
        return False
    latest = pd.to_datetime(df[ts_col], errors="coerce").max()
    if pd.isna(latest):
        return False
    age_days = (pd.Timestamp(as_of) - latest).days
    return age_days <= max_age_days
