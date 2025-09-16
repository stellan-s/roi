"""Command line helpers for running backtests from the terminal."""

import argparse
from pathlib import Path

from quant.adaptive_main import load_configuration
from quant.backtesting.framework import BacktestEngine
from quant.bayesian.adaptive_integration import AdaptiveBayesianEngine
from quant.bayesian.integration import BayesianPolicyEngine


def _print_single_run(results):
    """Pretty-print a short performance summary for a single backtest."""
    print("\nBacktest summary")
    print("---------------")
    print(f"Engine:           {results.engine_type}")
    print(f"Total return:     {results.total_return:.2%}")
    print(f"Annualized return:{results.annualized_return:.2%}")
    print(f"Volatility:       {results.volatility:.2%}")
    print(f"Sharpe ratio:     {results.sharpe_ratio:.3f}")
    print(f"Max drawdown:     {results.max_drawdown:.2%}")
    print(f"Trades:           {results.total_trades}")
    print(f"Win rate:         {results.win_rate:.2%}")


def _print_comparison(results):
    """Pretty-print the adaptive vs static comparison summary."""
    print("\nComparison summary")
    print("------------------")
    print(f"Adaptive annualized return: {results.adaptive_results.annualized_return:.2%}")
    print(f"Static annualized return:   {results.static_results.annualized_return:.2%}")
    print(f"Return improvement:         {results.return_improvement:.2%}")
    print(f"Adaptive Sharpe:            {results.adaptive_results.sharpe_ratio:.3f}")
    print(f"Static Sharpe:              {results.static_results.sharpe_ratio:.3f}")
    print(f"Sharpe improvement:         {results.sharpe_improvement:.3f}")
    print(f"Drawdown improvement:       {results.drawdown_improvement:.2%}")
    print(f"Tail risk improvement:      {results.tail_risk_improvement:.2%}")
    print(f"P-value (returns):          {results.p_value_returns:.4f}")
    print(
        "Statistically significant: "
        + ("Yes" if results.p_value_returns < 0.05 else "No")
    )


def _resolve_report_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _run_single(args):
    config = load_configuration()
    engine = BacktestEngine(config)

    if args.engine == "adaptive":
        factory = lambda cfg: AdaptiveBayesianEngine(cfg)
    else:
        factory = lambda cfg: BayesianPolicyEngine(cfg)

    result = engine.run_single_backtest(
        start_date=args.start,
        end_date=args.end,
        engine_factory=factory,
        engine_type=args.engine,
    )

    _print_single_run(result)

    if args.report:
        report_path = _resolve_report_path(args.report)
        engine.generate_backtest_report(result, str(report_path))


def _run_compare(args):
    config = load_configuration()
    engine = BacktestEngine(config)
    comparison = engine.compare_adaptive_vs_static(args.start, args.end)
    _print_comparison(comparison)

    if args.report:
        report_path = _resolve_report_path(args.report)
        engine.generate_comparison_report(comparison, str(report_path))


def _prompt_date(prompt: str) -> str:
    while True:
        value = input(f"{prompt} (YYYY-MM-DD): ").strip()
        if value:
            return value
        print("Please enter a date in the format YYYY-MM-DD.")


def _prompt_yes_no(prompt: str) -> bool:
    while True:
        response = input(f"{prompt} [y/N]: ").strip().lower()
        if response in ("y", "yes"):
            return True
        if response in ("", "n", "no"):
            return False
        print("Please answer with 'y' or 'n'.")


def _interactive_menu():
    config = load_configuration()
    engine = BacktestEngine(config)

    while True:
        print("\n=== Backtest Control Panel ===")
        print("1) Run adaptive backtest")
        print("2) Run static backtest")
        print("3) Compare adaptive vs static")
        print("4) Quit")

        choice = input("Select an option: ").strip()

        if choice == "4":
            print("Goodbye!")
            return

        if choice not in {"1", "2", "3"}:
            print("Unknown option, please try again.")
            continue

        start = _prompt_date("Start date")
        end = _prompt_date("End date")

        try:
            if choice in {"1", "2"}:
                engine_type = "adaptive" if choice == "1" else "static"
                factory = (
                    (lambda cfg: AdaptiveBayesianEngine(cfg))
                    if engine_type == "adaptive"
                    else (lambda cfg: BayesianPolicyEngine(cfg))
                )
                result = engine.run_single_backtest(
                    start_date=start,
                    end_date=end,
                    engine_factory=factory,
                    engine_type=engine_type,
                )
                _print_single_run(result)

                if _prompt_yes_no("Save Markdown report?"):
                    path = input("Report path: ").strip()
                    if path:
                        report_path = _resolve_report_path(path)
                        engine.generate_backtest_report(result, str(report_path))
            else:
                comparison = engine.compare_adaptive_vs_static(start, end)
                _print_comparison(comparison)

                if _prompt_yes_no("Save Markdown report?"):
                    path = input("Report path: ").strip()
                    if path:
                        report_path = _resolve_report_path(path)
                        engine.generate_comparison_report(comparison, str(report_path))

        except Exception as exc:
            print(f"⚠️ Backtest failed: {exc}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run portfolio backtests")
    subparsers = parser.add_subparsers(dest="command", required=True)

    single_parser = subparsers.add_parser(
        "single", help="Run a single backtest for a chosen engine"
    )
    single_parser.add_argument("--start", required=True, help="Backtest start date (YYYY-MM-DD)")
    single_parser.add_argument("--end", required=True, help="Backtest end date (YYYY-MM-DD)")
    single_parser.add_argument(
        "--engine",
        choices=("adaptive", "static"),
        default="adaptive",
        help="Which engine to run",
    )
    single_parser.add_argument(
        "--report",
        help="Optional path to write a Markdown backtest report",
    )
    single_parser.set_defaults(func=_run_single)

    compare_parser = subparsers.add_parser(
        "compare", help="Compare adaptive vs static engines"
    )
    compare_parser.add_argument("--start", required=True, help="Backtest start date (YYYY-MM-DD)")
    compare_parser.add_argument("--end", required=True, help="Backtest end date (YYYY-MM-DD)")
    compare_parser.add_argument(
        "--report",
        help="Optional path to write a Markdown comparison report",
    )
    compare_parser.set_defaults(func=_run_compare)

    interactive_parser = subparsers.add_parser(
        "interactive", help="Launch an interactive backtesting console"
    )
    interactive_parser.set_defaults(func=lambda args: _interactive_menu())

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
