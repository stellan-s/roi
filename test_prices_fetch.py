import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from quant.data_layer.prices import fetch_prices


class YFRateLimitError(Exception):
    pass


def _yf_df(start: str, periods: int = 5) -> pd.DataFrame:
    idx = pd.date_range(start, periods=periods, freq="D")
    return pd.DataFrame({"Close": [100.0 + i for i in range(periods)]}, index=idx)


class FetchPricesTests(unittest.TestCase):
    def test_rate_limited_ticker_uses_cache_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "prices.parquet"
            cached = pd.DataFrame(
                {
                    "date": pd.to_datetime(["2026-03-01", "2026-03-02"]),
                    "ticker": ["TEL2-B.ST", "TEL2-B.ST"],
                    "close": [111.0, 112.0],
                }
            )
            cached.to_parquet(cache_path, index=False)

            call_counts = {"TEL2-B.ST": 0, "AAPL": 0}

            def fake_download(ticker, start=None, auto_adjust=True, progress=False):
                call_counts[ticker] += 1
                if ticker == "TEL2-B.ST":
                    raise YFRateLimitError("Too Many Requests. Rate limited. Try after a while.")
                return _yf_df("2026-03-01", periods=3)

            with patch("quant.data_layer.prices.yf.download", side_effect=fake_download):
                out = fetch_prices(
                    ["TEL2-B.ST", "AAPL"],
                    cache_dir=tmp,
                    lookback_days=30,
                    max_retries=1,
                    base_backoff_seconds=0.0,
                    timeout_seconds=5,
                )

            self.assertFalse(out.empty)
            self.assertIn("TEL2-B.ST", set(out["ticker"]))
            self.assertIn("AAPL", set(out["ticker"]))
            self.assertGreaterEqual(call_counts["TEL2-B.ST"], 2)  # initial + retry
            self.assertGreaterEqual((out["ticker"] == "TEL2-B.ST").sum(), 2)  # from cache fallback


if __name__ == "__main__":
    unittest.main()
