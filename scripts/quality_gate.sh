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
  tests.test_engine_contract \
  tests.test_config_loader \
  tests.test_prices_fetch \
  tests.test_backtest_golden \
  tests.test_integration_pipeline
"$PYTHON_BIN" -m quant.backtest_runner --help >/dev/null
"$PYTHON_BIN" -m quant.backtesting.cli --help >/dev/null

echo "Quality gate passed."
