# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
This is "ROI" - a quantitative trading/investment analysis system written in Python. It's a proof-of-concept that fetches Swedish stock data, computes technical indicators and sentiment analysis, then generates daily trading recommendations.

## Architecture
The system follows a pipeline architecture with these main components:

- **Data Layer** (`quant/data_layer/`):
  - `prices.py` - Fetches price data using yfinance, caches as parquet files
  - `news.py` - Fetches news feeds for sentiment analysis

- **Features** (`quant/features/`):
  - `technical.py` - Computes SMA, momentum, and ranking indicators
  - `sentiment.py` - Basic sentiment scoring from news feeds

- **Policy Engine** (`quant/policy_engine/`):
  - `rules.py` - Combines technical and sentiment signals into buy/sell/hold decisions

- **Reports** (`quant/reports/`):
  - `daily_brief.py` - Generates markdown reports with recommendations

## Configuration
The system uses YAML configuration files in `quant/config/`:
- `settings.yaml` - Main configuration (lookback periods, thresholds, data sources)
- `universe.yaml` - Stock tickers to analyze (currently Swedish stocks)

## Running the System
- **Main execution**: `python -m quant.main` (from project root)
- **Python environment**: Uses `.venv` virtual environment with Python 3.12
- **Dependencies**: No requirements.txt found - uses yfinance, pandas, yaml (installed in venv)

## Data Flow
1. Loads configuration from YAML files
2. Fetches price data for configured tickers (cached in `data/`)
3. Fetches news for sentiment analysis
4. Computes technical features (SMA signals, momentum rankings)
5. Applies sentiment analysis
6. Combines signals using scoring rules
7. Generates daily markdown report in `reports/`

## Key Files
- `quant/main.py` - Entry point orchestrating the full pipeline
- `quant/config/settings.yaml` - Core configuration parameters
- `quant/config/universe.yaml` - Stock universe definition

## Output
Reports are generated as markdown files in `reports/` directory with format `daily_YYYY-MM-DD.md`.