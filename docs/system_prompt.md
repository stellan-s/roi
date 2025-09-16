# ROI System Description for AI Collaboration

Use this document when prompting an AI assistant to extend or debug the ROI quantitative trading platform. It captures the system goals, architecture, critical files, and current limitations so new improvements stay aligned with the existing design.

## Mission & Scope
- Provide actionable daily trading recommendations for Swedish and US equities.
- Combine Bayesian signal processing, regime detection, heavy-tail risk modeling, and portfolio construction in a single reproducible pipeline.
- Prioritize risk-aware decision making, transparent reporting, and configurability via declarative settings.

## Core Pipeline
1. **Data Layer (`quant/data_layer/`)** – Ingests and harmonizes price, fundamental, and sentiment data. Interfaces with both historical datasets and live updates.
2. **Feature Engineering (`quant/features/`)** – Builds indicators (momentum, trend, sentiment) consumed by downstream probabilistic models.
3. **Bayesian Engine (`quant/bayesian/`)** – Fuses heterogeneous signals with Bayesian inference, delivering posterior trend probabilities, expected returns, and uncertainty bands.
4. **Regime Detection (`quant/regime/`)** – Hidden Markov Model classifies markets as Bull, Bear, or Neutral and feeds adaptive priors/weights back into the engine.
5. **Risk Modeling (`quant/risk/`)** – Uses Student-t fits, Extreme Value Theory, and Monte Carlo simulation to quantify tail exposure and stress scenarios.
6. **Portfolio Management (`quant/portfolio/`)** – Applies policy constraints (max weights, regime limits, transaction costs) to produce position sizing and drive the paper-trading engine.
7. **Reporting & Logging (`reports/` + `data/`)** – Generates Markdown reports (e.g., `reports/daily_YYYY-MM-DD.md`) and writes recommendation/trade logs under `data/recommendation_logs/` plus simulated portfolio state in `data/portfolio/` for auditability.

## Key Configuration & Interfaces
- Global settings managed in `settings.yaml` (thresholds, regime policies, risk budgets).
- Portfolio state persisted in JSON for cash/holdings tracking with daily trade history under `data/portfolio/`; real trading integration pending.
- Recommendation history persisted as daily parquet/JSON logs under `data/recommendation_logs/`.
- Entry points `python -m quant.main` (legacy) and `python -m quant.adaptive_main` (parameter-learning + paper-trading) orchestrate runs.
- Tests live in `test_*.py` files at the repo root covering Bayesian updates, regime detection, risk tails, and configurable policy behavior.

## Feature Highlights
- Adaptive decision thresholds (buy ≥ 58%, sell ≤ 40%) with transaction-cost-aware filtering.
- Regime-aware exposure caps (e.g., max 60% allocation in bear markets) and diversification checks.
- Automated position sizing to satisfy minimum holdings (≥ 3) and max 10% per asset.
- Detailed reporting with probability-based recommendations, portfolio valuation, risk commentary, and explicit units (E[r]_1d, E[R]_21d, σ in %, Downside VaR_1d, Tail-score).
- Automatic logging of portfolio-adjusted recommendations and executed paper trades for evaluation.

## Current Limitations / Backlog
- No integrated backtesting engine or real brokerage execution yet; evaluation currently uses paper trades and logged recommendations.
- Machine learning signal enhancements, real-time data feeds, and advanced optimization are in progress.
- Notification channels (email/SMS), tax optimisation, and multi-currency support remain TODOs.
- Calibration diagnostics (reliability/Brier/ROC) and automated CV reporting are still manual.

## Prompting Guidelines
When asking the AI to enhance the system, include:
- The specific component or file path you want to change.
- Desired behavior or metric (e.g., faster regime switch detection, new risk metric).
- Any constraints (runtime, data availability, regulatory rules).
- Relevant tests to run or extend.

Encourage the assistant to:
- Respect the modular pipeline boundaries described above.
- Update or add tests alongside code changes.
- Preserve configurability via `settings.yaml` or component-level parameters.
- Keep the reporting/ logging contract intact (parquet + JSON logs) when modifying the daily pipeline.

Keeping this context in prompts will help the AI produce aligned, production-ready improvements.
