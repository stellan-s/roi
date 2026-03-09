"""Adaptive trading entrypoint using the canonical daily engine API."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from quant.bayesian.adaptive_integration import AdaptiveBayesianEngine
from quant.config.loader import load_configuration
from quant.data_layer.macro import fetch_macro_indicators, fetch_precious_metals_sentiment, fetch_vix
from quant.data_layer.news import fetch_news
from quant.data_layer.prices import fetch_prices
from quant.data_layer.validation import normalize_news_schema, normalize_prices_schema
from quant.engine import DayRunContext, run_engine_day
from quant.features.sentiment import naive_sentiment
from quant.features.technical import compute_technical_features
from quant.observability import get_logger, log_event, new_run_id
from quant.portfolio.rules import PortfolioManager
from quant.portfolio.state import PortfolioTracker
from quant.reports.daily_brief import save_daily_markdown


def prepare_historical_data(config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Prepare historical data for adaptive calibration."""
    universe = config["universe"]["tickers"]
    lookback = int(config.get("data", {}).get("lookback_days", 500)) * 2

    prices = normalize_prices_schema(
        fetch_prices(tickers=universe, cache_dir=config["data"]["cache_dir"], lookback_days=lookback)
    )
    news = normalize_news_schema(
        fetch_news(feed_urls=config["signals"]["news_feed_urls"], cache_dir=config["data"]["cache_dir"])
    )
    technical = compute_technical_features(
        prices,
        config["signals"]["sma_long"],
        config["signals"]["momentum_window"],
    )
    sentiment = naive_sentiment(news, universe)
    returns = prices.groupby("ticker").apply(
        lambda x: x.assign(**{"return": x["close"].pct_change()})
    ).reset_index(drop=True)
    return prices, sentiment, technical, returns


def calibrate_adaptive_engine(
    config: Dict,
    prices_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
    technical_df: pd.DataFrame,
    returns_df: pd.DataFrame,
) -> AdaptiveBayesianEngine:
    """Calibrate and return the adaptive Bayesian engine."""
    engine = AdaptiveBayesianEngine(config)
    engine.calibrate_parameters(
        prices_df=prices_df,
        technical_df=technical_df,
        sentiment_df=sentiment_df,
        returns_df=returns_df,
    )
    return engine


def _fetch_live_data(config: Dict, preloaded_prices: Optional[pd.DataFrame] = None) -> Dict:
    tickers = config["universe"]["tickers"]
    if preloaded_prices is not None and not preloaded_prices.empty:
        prices = normalize_prices_schema(preloaded_prices)
    else:
        prices = normalize_prices_schema(fetch_prices(tickers, config["data"]["cache_dir"]))
    news = normalize_news_schema(fetch_news(config["signals"]["news_feed_urls"], config["data"]["cache_dir"]))
    vix = fetch_vix(config["data"]["cache_dir"])

    macro_data = {}
    try:
        macro_data["precious_metals"] = fetch_precious_metals_sentiment(config["data"]["cache_dir"])
        macro_data["indicators"] = fetch_macro_indicators(config["data"]["cache_dir"])
    except Exception:
        macro_data = {}

    return {"tickers": tickers, "prices": prices, "news": news, "vix_data": vix, "macro_data": macro_data}


def _save_adaptive_logs(
    cache_dir: str,
    recommendations: pd.DataFrame,
    engine: AdaptiveBayesianEngine,
    run_id: str,
    diagnostics: Dict,
    trades: List[Dict],
) -> None:
    if recommendations.empty:
        return

    log_dir = Path(cache_dir) / "adaptive_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    as_of = pd.to_datetime(recommendations["date"]).max().date().isoformat()

    rec_path = log_dir / f"adaptive_recommendations_{as_of}.parquet"
    recommendations.to_parquet(rec_path, index=False)

    diag_path = log_dir / f"adaptive_step_diagnostics_{as_of}.json"
    with open(diag_path, "w") as f:
        json.dump({"run_id": run_id, "diagnostics": diagnostics, "trades": trades}, f, indent=2, default=str)

    learning_summary = engine.get_learning_summary()
    summary_path = log_dir / f"learning_summary_{as_of}.json"
    with open(summary_path, "w") as f:
        json.dump({"run_id": run_id, "learning_summary": learning_summary}, f, indent=2, default=str)


def run_live_day(config: Dict) -> pd.DataFrame:
    """Run one adaptive live day through the shared contract."""
    logger = get_logger("quant.adaptive")
    run_id = new_run_id("live_adaptive")
    log_event(logger, "adaptive_start", run_id=run_id)

    prices_hist, sentiment_hist, tech_hist, returns_hist = prepare_historical_data(config)
    engine = calibrate_adaptive_engine(config, prices_hist, sentiment_hist, tech_hist, returns_hist)
    # Reuse historical price fetch to avoid a second full-universe Yahoo request.
    data = _fetch_live_data(config, preloaded_prices=prices_hist)
    if data["prices"].empty:
        raise ValueError("No price data available")

    as_of = pd.to_datetime(data["prices"]["date"]).max().normalize()
    context = DayRunContext(
        as_of=as_of,
        tickers=data["tickers"],
        prices=data["prices"],
        news=data["news"],
        vix_data=data["vix_data"],
    )
    day_result = run_engine_day(engine, context, config)
    recommendations = day_result.recommendations
    if recommendations.empty:
        log_event(logger, "adaptive_no_recommendations", run_id=run_id, as_of=as_of)
        return recommendations

    price_history = data["prices"][data["prices"]["date"] <= as_of].pivot(
        index="date", columns="ticker", values="close"
    ).dropna(axis=1, how="all")
    portfolio_mgr = PortfolioManager(config)
    final_decisions = portfolio_mgr.apply_portfolio_rules(recommendations, price_history)

    current_prices = data["prices"][data["prices"]["date"] == as_of][["ticker", "close"]]
    tracker = PortfolioTracker(config["data"]["cache_dir"] + "/portfolio")
    tracker.update_portfolio_state(current_prices, as_of_date=str(as_of.date()))

    execution_cfg = config.get("backtesting", {}).get("execution", {})
    slippage_bps = float(execution_cfg.get("slippage_bps", 3.0))
    fee_bps = float(execution_cfg.get("fee_bps", config.get("policy", {}).get("trade_cost_bps", 3)))
    executed_trades = tracker.execute_trades(
        final_decisions,
        current_prices,
        slippage_bps=slippage_bps,
        fee_bps=fee_bps,
    )

    portfolio_summary = tracker.get_portfolio_summary()
    _save_adaptive_logs(
        cache_dir=config["data"]["cache_dir"],
        recommendations=final_decisions,
        engine=engine,
        run_id=run_id,
        diagnostics=day_result.diagnostics,
        trades=executed_trades,
    )

    save_daily_markdown(
        final_decisions,
        config.get("run", {}).get("outdir", "reports"),
        portfolio_summary,
        engine,
        data.get("macro_data", {}),
    )

    log_event(
        logger,
        "adaptive_complete",
        run_id=run_id,
        as_of=as_of,
        recommendations=len(final_decisions),
        trades=len(executed_trades),
        portfolio_value=portfolio_summary.get("total_value", 0.0),
    )
    return final_decisions


def main() -> None:
    print("=== ROI Trading System (Adaptive) ===")
    config = load_configuration()
    decisions = run_live_day(config)
    print(f"Generated {len(decisions)} recommendations")


if __name__ == "__main__":
    main()
