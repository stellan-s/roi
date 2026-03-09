import unittest
from datetime import date

import pandas as pd

from quant.engine import DayRunContext, run_engine_day


class _StubStaticEngine:
    def __init__(self):
        self.last_tech = None
        self.last_senti = None

    def bayesian_score(self, tech, senti, prices=None, vix_data=None):
        self.last_tech = tech.copy()
        self.last_senti = senti.copy()
        out = tech[["ticker", "close"]].copy()
        out["date"] = pd.to_datetime(tech["date"]).dt.normalize()
        out["decision"] = "Hold"
        out["expected_return"] = 0.0
        out["prob_positive"] = 0.5
        out["decision_confidence"] = 0.5
        out["uncertainty"] = 0.5
        return out


class EngineContractTests(unittest.TestCase):
    def test_run_engine_day_is_point_in_time_safe_for_news(self):
        engine = _StubStaticEngine()
        as_of = pd.Timestamp("2025-02-25")

        dates = pd.date_range("2025-01-01", periods=60, freq="D")
        prices = pd.DataFrame({"date": dates, "ticker": "AAA", "close": [100 + i for i in range(60)]})
        news = pd.DataFrame(
            {
                "published": pd.to_datetime(["2025-02-25T08:00:00Z", "2025-02-26T08:00:00Z"], utc=True),
                "title": ["AAA wins contract", "AAA profit warning"],
                "summary": ["positive", "negative"],
            }
        )
        context = DayRunContext(
            as_of=as_of,
            tickers=["AAA"],
            prices=prices,
            news=news,
        )
        config = {"signals": {"sma_long": 50, "momentum_window": 21}}

        result = run_engine_day(engine, context, config)

        self.assertEqual(result.as_of, as_of)
        self.assertFalse(result.recommendations.empty)
        self.assertTrue((pd.to_datetime(engine.last_senti["date"]).dt.normalize() == as_of).all())
        self.assertIsInstance(engine.last_senti["date"].iloc[0], date)
        # Future article (2025-01-03) must not be present at as_of 2025-01-02.
        self.assertGreaterEqual(engine.last_senti["sent_score"].iloc[0], 0.0)


if __name__ == "__main__":
    unittest.main()
