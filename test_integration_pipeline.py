import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from quant.main import run_live_day


class _StubPolicyEngine:
    def __init__(self, config):
        self.config = config

    def bayesian_score(self, tech, senti, prices=None, vix_data=None):
        out = tech[["ticker", "close"]].copy()
        out["date"] = pd.to_datetime(tech["date"]).dt.normalize()
        out["decision"] = out["ticker"].map(lambda t: "Buy" if t == "AAA" else "Hold")
        out["expected_return"] = out["ticker"].map(lambda t: 0.01 if t == "AAA" else 0.0)
        out["prob_positive"] = out["ticker"].map(lambda t: 0.62 if t == "AAA" else 0.5)
        out["decision_confidence"] = out["ticker"].map(lambda t: 0.85 if t == "AAA" else 0.5)
        out["uncertainty"] = out["ticker"].map(lambda t: 0.15 if t == "AAA" else 0.5)
        out["regime"] = "neutral"
        return out


def _prices():
    dates = pd.date_range("2024-03-01", periods=40, freq="D")
    rows = []
    for ticker in ("AAA", "BBB"):
        for i, d in enumerate(dates):
            rows.append({"date": d, "ticker": ticker, "close": 100 + (i if ticker == "AAA" else 0)})
    return pd.DataFrame(rows)


class IntegrationPipelineTests(unittest.TestCase):
    def test_run_live_day_end_to_end_with_stubs(self):
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp) / "reports"
            config = {
                "universe": {"tickers": ["AAA", "BBB"]},
                "data": {"cache_dir": tmp},
                "signals": {"news_feed_urls": [], "sma_long": 20, "momentum_window": 21},
                "policy": {"max_weight": 0.06, "trade_cost_bps": 3, "target_total_allocation": 0.85},
                "backtesting": {"execution": {"slippage_bps": 0.0, "fee_bps": 0.0}},
                "risk_controls": {"min_history_days": 10, "max_top3_exposure": 0.18},
                "run": {"outdir": str(outdir)},
            }

            with patch("quant.main.fetch_prices", return_value=_prices()), patch(
                "quant.main.fetch_news",
                return_value=pd.DataFrame(columns=["published", "title", "summary"]),
            ), patch(
                "quant.main.fetch_vix",
                return_value=pd.DataFrame(columns=["date", "vix_close"]),
            ), patch(
                "quant.main.BayesianPolicyEngine",
                _StubPolicyEngine,
            ), patch(
                "quant.main.save_daily_markdown",
            ) as save_report:
                decisions = run_live_day(config)

            self.assertFalse(decisions.empty)
            self.assertIn("portfolio_weight", decisions.columns)
            self.assertTrue((Path(tmp) / "portfolio" / "current_state.json").exists())
            save_report.assert_called_once()


if __name__ == "__main__":
    unittest.main()
