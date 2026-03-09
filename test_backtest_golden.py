import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from quant.backtest_runner import run_backtest_period


class _StubEngine:
    def bayesian_score(self, tech, senti, prices=None, vix_data=None):
        out = tech[["ticker", "close"]].copy()
        out["date"] = pd.to_datetime(tech["date"]).dt.normalize()
        out["decision"] = out["ticker"].map(lambda t: "Buy" if t == "AAA" else "Hold")
        out["expected_return"] = out["ticker"].map(lambda t: 0.01 if t == "AAA" else 0.0)
        out["prob_positive"] = out["ticker"].map(lambda t: 0.6 if t == "AAA" else 0.5)
        out["decision_confidence"] = out["ticker"].map(lambda t: 0.8 if t == "AAA" else 0.5)
        out["uncertainty"] = out["ticker"].map(lambda t: 0.2 if t == "AAA" else 0.5)
        out["regime"] = "neutral"
        return out


def _synthetic_prices():
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    rows = []
    for ticker in ("AAA", "BBB"):
        for d in dates:
            rows.append({"date": d, "ticker": ticker, "close": 100.0})
    return pd.DataFrame(rows)


def _empty_news():
    return pd.DataFrame(columns=["published", "title", "summary"])


class BacktestGoldenTests(unittest.TestCase):
    def test_backtest_regression_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = {
                "universe": {"tickers": ["AAA", "BBB"]},
                "data": {"cache_dir": tmp},
                "signals": {
                    "news_feed_urls": [],
                    "sma_long": 20,
                    "momentum_window": 21,
                },
                "policy": {
                    "max_weight": 0.06,
                    "trade_cost_bps": 3,
                    "target_total_allocation": 0.85,
                },
                "backtesting": {"execution": {"slippage_bps": 0.0, "fee_bps": 0.0}},
                "risk_controls": {
                    "max_drawdown": 0.20,
                    "max_daily_loss": 0.05,
                    "max_top3_exposure": 0.18,
                    "min_history_days": 3,
                },
            }

            with patch("quant.backtest_runner.fetch_prices", return_value=_synthetic_prices()), patch(
                "quant.backtest_runner.fetch_news", return_value=_empty_news()
            ), patch(
                "quant.backtest_runner.fetch_vix", return_value=pd.DataFrame(columns=["date", "vix_close"])
            ):
                result = run_backtest_period(
                    _StubEngine(),
                    config,
                    start_date="2024-01-21",
                    end_date="2024-01-30",
                    engine_type="golden",
                )

        fixture_path = Path("tests/fixtures/golden_backtest_expected.json")
        expected = json.loads(fixture_path.read_text())

        self.assertAlmostEqual(result.total_return, expected["total_return"], places=8)
        self.assertAlmostEqual(result.annualized_return, expected["annualized_return"], places=8)
        self.assertAlmostEqual(result.volatility, expected["volatility"], places=8)
        self.assertAlmostEqual(result.sharpe_ratio, expected["sharpe_ratio"], places=8)
        self.assertAlmostEqual(result.max_drawdown, expected["max_drawdown"], places=8)
        self.assertEqual(result.total_trades, expected["total_trades"])
        self.assertAlmostEqual(result.win_rate, expected["win_rate"], places=8)
        self.assertAlmostEqual(result.var_95, expected["var_95"], places=8)


if __name__ == "__main__":
    unittest.main()
