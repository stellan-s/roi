from pathlib import Path
import yaml, pandas as pd
from datetime import datetime
from quant.data_layer.prices import fetch_prices
from quant.data_layer.news import fetch_news
from quant.features.technical import compute_technical_features
from quant.features.sentiment import naive_sentiment
from quant.policy_engine.rules import bayesian_score
from quant.portfolio.rules import PortfolioManager
from quant.portfolio.state import PortfolioTracker
from quant.reports.daily_brief import save_daily_markdown

CFG_DIR = Path(__file__).parent / "config"

def load_yaml(name):
    return yaml.safe_load((CFG_DIR / name).read_text(encoding="utf-8"))

def run():
    uni = load_yaml("universe.yaml")["tickers"]
    cfg = load_yaml("settings.yaml")
    cache = cfg["data"]["cache_dir"]

    prices = fetch_prices(uni, cache, cfg["data"]["lookback_days"])
    news   = fetch_news(cfg["signals"]["news_feed_urls"], cache)
    tech   = compute_technical_features(prices, cfg["signals"]["sma_long"], cfg["signals"]["momentum_window"])
    senti  = naive_sentiment(news, uni)
    dec    = bayesian_score(tech, senti, prices, cfg)

    # Portfolio management
    portfolio_mgr = PortfolioManager(cfg)
    final_decisions = portfolio_mgr.apply_portfolio_rules(dec)

    # Portfolio tracking och trade execution
    portfolio_tracker = PortfolioTracker(cfg["data"]["cache_dir"] + "/portfolio")

    # Uppdatera portfolio state med senaste priser
    latest_date = prices['date'].max()
    latest_prices = prices[prices['date'] == latest_date][['ticker', 'close']]
    portfolio_state = portfolio_tracker.update_portfolio_state(latest_prices)

    # Simulera trades baserat på decisions
    executed_trades = portfolio_tracker.execute_trades(final_decisions, latest_prices)

    # Få portfolio summary för rapport
    portfolio_summary = portfolio_tracker.get_portfolio_summary()

    out_md = save_daily_markdown(final_decisions, cfg["run"]["outdir"], portfolio_summary)
    print(f"✅ Roi PoC klar. Rapport: {out_md}")

if __name__ == "__main__":
    run()
