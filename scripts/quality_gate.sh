#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Python binary not found: $PYTHON_BIN"
  echo "Run ./scripts/bootstrap.sh first."
  exit 1
fi

"$PYTHON_BIN" -m compileall quant
"$PYTHON_BIN" -m unittest \
  test_engine_contract.py \
  test_config_loader.py \
  test_prices_fetch.py \
  test_backtest_golden.py \
  test_integration_pipeline.py
"$PYTHON_BIN" -m quant.backtest_runner --help >/dev/null
"$PYTHON_BIN" -m quant.backtesting.cli --help >/dev/null

echo "Quality gate passed."
