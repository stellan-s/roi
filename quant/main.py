"""Static (non-adaptive) trading entrypoint using the canonical daily engine API."""

from __future__ import annotations

from typing import Dict

import pandas as pd

from quant.bayesian.integration import BayesianPolicyEngine
from quant.config.loader import load_configuration
from quant.data_layer.macro import fetch_vix
from quant.data_layer.news import fetch_news
from quant.data_layer.prices import fetch_prices
from quant.data_layer.validation import normalize_news_schema, normalize_prices_schema
from quant.engine import DayRunContext, run_engine_day
from quant.observability import get_logger, log_event, new_run_id
from quant.portfolio.rules import PortfolioManager
from quant.portfolio.state import PortfolioTracker
from quant.reports.daily_brief import save_daily_markdown


def fetch_live_data(config: Dict) -> Dict[str, pd.DataFrame]:
    tickers = config["universe"]["tickers"]
    prices = normalize_prices_schema(fetch_prices(tickers, config["data"]["cache_dir"]))
    news = normalize_news_schema(
        fetch_news(config["signals"]["news_feed_urls"], config["data"]["cache_dir"])
    )
    vix = fetch_vix(config["data"]["cache_dir"])
    return {"tickers": tickers, "prices": prices, "news": news, "vix_data": vix}


def run_live_day(config: Dict) -> pd.DataFrame:
    logger = get_logger("quant.main")
    run_id = new_run_id("live_static")
    log_event(logger, "live_start", run_id=run_id, engine="static")

    data = fetch_live_data(config)
    if data["prices"].empty:
        raise ValueError("No price data available")

    as_of = pd.to_datetime(data["prices"]["date"]).max().normalize()
    engine = BayesianPolicyEngine(config)
    context = DayRunContext(
        as_of=as_of,
        tickers=data["tickers"],
        prices=data["prices"],
        news=data["news"],
        vix_data=data["vix_data"],
    )
    step_result = run_engine_day(engine, context, config)
    recommendations = step_result.recommendations
    if recommendations.empty:
        log_event(logger, "live_no_recommendations", run_id=run_id, as_of=as_of)
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
    save_daily_markdown(
        final_decisions,
        config.get("run", {}).get("outdir", "reports"),
        portfolio_summary,
        engine,
    )
    log_event(
        logger,
        "live_complete",
        run_id=run_id,
        as_of=as_of,
        recommendations=len(final_decisions),
        trades=len(executed_trades),
        portfolio_value=portfolio_summary.get("total_value", 0.0),
    )
    return final_decisions


def main() -> None:
    print("=== ROI Trading System (Static) ===")
    config = load_configuration()
    decisions = run_live_day(config)
    print(f"Generated {len(decisions)} recommendations")


if __name__ == "__main__":
    main()
