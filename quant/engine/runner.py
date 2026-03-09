"""Canonical daily engine contract used by live and backtest execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from quant.data_layer.validation import (
    filter_as_of,
    normalize_news_schema,
    normalize_prices_schema,
)
from quant.features.sentiment import naive_sentiment
from quant.features.technical import compute_technical_features


@dataclass(frozen=True)
class DayRunContext:
    """Input context for one point-in-time engine evaluation."""

    as_of: pd.Timestamp
    tickers: List[str]
    prices: pd.DataFrame
    news: pd.DataFrame
    vix_data: Optional[pd.DataFrame] = None
    fundamentals: Optional[pd.DataFrame] = None


@dataclass
class DayRunResult:
    """Standardized output for one day evaluation."""

    as_of: pd.Timestamp
    recommendations: pd.DataFrame
    diagnostics: Dict[str, Any]


def _build_engine_inputs(context: DayRunContext, config: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    as_of = pd.Timestamp(context.as_of).normalize()

    prices = normalize_prices_schema(context.prices)
    prices = filter_as_of(prices, "date", as_of)

    news = normalize_news_schema(context.news)
    news = filter_as_of(news, "published", as_of)

    vix_data = context.vix_data
    if vix_data is not None and not vix_data.empty and "date" in vix_data.columns:
        vix_data = vix_data.copy()
        vix_data["date"] = pd.to_datetime(vix_data["date"], errors="coerce")
        vix_data = vix_data[vix_data["date"].notna()]
        vix_data = vix_data[vix_data["date"] <= as_of]

    signals_cfg = config.get("signals", {})
    tech_all = compute_technical_features(
        prices,
        signals_cfg.get("sma_long", 50),
        signals_cfg.get("momentum_window", 252),
    )
    if tech_all.empty:
        tech_latest = pd.DataFrame(columns=["date", "ticker", "close", "above_sma", "mom_rank"])
    else:
        tech_latest = tech_all.sort_values(["ticker", "date"]).groupby("ticker", as_index=False).tail(1)
        tech_latest = tech_latest[["date", "ticker", "close", "above_sma", "mom_rank"]].copy()
        tech_latest["date"] = as_of

    senti_all = naive_sentiment(news, context.tickers)
    if senti_all.empty:
        senti_latest = pd.DataFrame({"ticker": context.tickers, "date": as_of, "sent_score": 0.0})
    else:
        senti_all = senti_all.copy()
        senti_all["date"] = pd.to_datetime(senti_all["date"], errors="coerce").dt.normalize()
        senti_today = senti_all[senti_all["date"] == as_of][["ticker", "date", "sent_score"]].copy()
        senti_latest = pd.DataFrame({"ticker": context.tickers, "date": as_of}).merge(
            senti_today, on=["ticker", "date"], how="left"
        )
        senti_latest["sent_score"] = senti_latest["sent_score"].fillna(0.0)

    fundamentals = context.fundamentals
    if fundamentals is not None and not fundamentals.empty:
        fundamentals = fundamentals.copy()
        if "ticker" in fundamentals.columns:
            fundamentals["ticker"] = fundamentals["ticker"].astype(str)
        else:
            fundamentals = None

    return {
        "as_of": as_of,
        "prices": prices,
        "tech": tech_latest,
        "senti": senti_latest,
        "vix": vix_data,
        "fundamentals": fundamentals,
    }


def run_engine_day(engine: Any, context: DayRunContext, config: Dict[str, Any]) -> DayRunResult:
    """
    Canonical contract for one daily engine step.

    This function is intentionally the single execution path used by both
    live and backtesting flows.
    """

    prepared = _build_engine_inputs(context, config)
    as_of = prepared["as_of"]
    tech = prepared["tech"]
    senti = prepared["senti"]
    prices = prepared["prices"]
    vix = prepared["vix"]
    fundamentals = prepared["fundamentals"]

    if tech.empty:
        empty = pd.DataFrame(
            columns=[
                "date",
                "ticker",
                "close",
                "decision",
                "expected_return",
                "prob_positive",
                "decision_confidence",
                "uncertainty",
            ]
        )
        return DayRunResult(
            as_of=as_of,
            recommendations=empty,
            diagnostics={"reason": "no_technical_data", "tickers_requested": len(context.tickers)},
        )

    if hasattr(engine, "bayesian_score_adaptive"):
        recs = engine.bayesian_score_adaptive(
            tech=tech,
            senti=senti,
            fundamentals=fundamentals,
            prices=prices,
            vix_data=vix,
        )
        engine_type = "adaptive"
    else:
        # Static engine normalizes technical dates to `datetime.date` before merge.
        # Keep sentiment dates aligned to avoid dtype mismatch in pandas merge.
        static_senti = senti.copy()
        if "date" in static_senti.columns:
            static_senti["date"] = pd.to_datetime(static_senti["date"], errors="coerce").dt.date
        recs = engine.bayesian_score(
            tech=tech,
            senti=static_senti,
            prices=prices,
            vix_data=vix,
        )
        engine_type = "static"

    if recs is None or recs.empty:
        recs = pd.DataFrame(columns=["date", "ticker", "close", "decision", "expected_return", "prob_positive"])
    else:
        recs = recs.copy()
        recs["date"] = as_of
        if "ticker" in recs.columns:
            recs["ticker"] = recs["ticker"].astype(str)

    diagnostics = {
        "engine_type": engine_type,
        "as_of": as_of.isoformat(),
        "prices_rows": len(prices),
        "news_rows": len(context.news) if context.news is not None else 0,
        "features_rows": len(tech),
        "recommendations_rows": len(recs),
    }

    return DayRunResult(as_of=as_of, recommendations=recs, diagnostics=diagnostics)
