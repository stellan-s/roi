# Canonical Engine Contract

This repository now uses one daily execution contract for both live runs and backtests:

- API: `quant.engine.run_engine_day`
- Input type: `quant.engine.DayRunContext`
- Output type: `quant.engine.DayRunResult`

## Contract

`run_engine_day(engine, context, config)`:

1. Slices all inputs to `context.as_of` (point-in-time safe).
2. Normalizes provider schemas (prices/news).
3. Computes day features (`technical`, `sentiment`) from as-of-visible data only.
4. Calls exactly one engine method:
   - adaptive: `bayesian_score_adaptive(...)`
   - static: `bayesian_score(...)`
5. Returns normalized recommendations and diagnostics.

## Why this exists

Before this change, live and backtest used different wiring and stale call paths.
This contract removes interface drift and enforces one scoring path.

## Backtest execution model

- Decisions are produced at `T`.
- Trades execute at `T+1` (next bar), never same bar.
- News and macro are sliced `<= T`.
- Slippage and fees are applied through `PortfolioTracker.execute_trades`.
- Basic kill switches are enforced:
  - `risk_controls.max_drawdown`
  - `risk_controls.max_daily_loss`
