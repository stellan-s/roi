#!/usr/bin/env python3
"""
Clone a master price history across every ticker in universe.yaml.

Usage:
  python replicate_prices.py --source data/msft_history.csv --cache-dir data/cache/prices
"""

import argparse
from pathlib import Path
import pandas as pd
import yaml


def load_prices(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    if "date" not in df.columns or "close" not in df.columns:
        raise ValueError("Expected columns 'date' and 'close' in source file")
    df = df[["date", "close"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="CSV/Parquet file with date,close columns to replicate")
    parser.add_argument("--cache-dir", required=True, help="Target cache directory (settings.yaml -> data.cache_dir)")
    parser.add_argument("--universe", default="quant/config/universe.yaml", help="Path to universe.yaml")
    args = parser.parse_args()

    source_path = Path(args.source).expanduser()
    cache_dir = Path(args.cache_dir).expanduser()
    universe_path = Path(args.universe).expanduser()

    cache_dir.mkdir(parents=True, exist_ok=True)

    base = load_prices(source_path)

    with universe_path.open() as fh:
        universe = yaml.safe_load(fh)
    tickers = universe["tickers"]
    if not tickers:
        raise ValueError("No tickers found in universe.yaml")

    frames = []
    for ticker in tickers:
        df = base.copy()
        df["ticker"] = ticker
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["ticker", "date"])

    output_path = cache_dir / "prices.parquet"
    combined.to_parquet(output_path, index=False)

    print(f"✓ Wrote {len(combined)} rows covering {len(tickers)} tickers to {output_path}")


if __name__ == "__main__":
    main()
