"""ANSI dashboard for portfolio snapshot, regime, and recommendations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from quant.policy_engine.rules import get_regime_info

RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
DIM = "\033[2m"

BAR_GLYPH = "█"
SPACER = " "


def load_portfolio_state(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Portfolio state not found: {path}")
    with path.open() as fh:
        return json.load(fh)


def format_currency(value: float) -> str:
    return f"{value:,.0f} SEK"


def colour_for_pnl(pnl: float) -> str:
    if pnl > 0:
        return GREEN
    if pnl < 0:
        return RED
    return YELLOW


def render_weight_bar(weight: float, width: int = 32) -> str:
    filled = int(round(weight * width))
    return BAR_GLYPH * filled + SPACER * (width - filled)


def render_confidence_bar(confidence: float, width: int = 20) -> str:
    filled = int(round(confidence * width))
    bar = BAR_GLYPH * filled + SPACER * (width - filled)
    colour = GREEN if confidence >= 0.66 else YELLOW if confidence >= 0.33 else RED
    return f"{colour}{bar}{RESET}"


def render_regime_section() -> str:
    regime_info = get_regime_info()
    regime = regime_info.get("regime", "Unknown")
    confidence = float(regime_info.get("confidence", 0.0))
    explanation = regime_info.get("explanation", "")
    confidence_pct = confidence * 100
    bar = render_confidence_bar(confidence)
    lines = [
        f"{BOLD}Current Regime:{RESET} {regime}",
        f"{BOLD}Confidence:{RESET} {confidence_pct:5.1f}% {bar}",
    ]
    if explanation:
        lines.append(f"{DIM}{explanation}{RESET}")
    return "\n".join(lines)


def render_portfolio_section(state: Dict, width: int = 32, limit: int = 8) -> str:
    total_value = state.get("total_value", 0.0)
    cash = state.get("cash", 0.0)
    holdings: List[Dict] = state.get("holdings", [])

    lines = [f"{BOLD}Portfolio Snapshot{RESET}", f"Total Value : {format_currency(total_value)}"]
    lines.append(f"Cash        : {format_currency(cash)}")
    total_invested = state.get("total_invested")
    if total_invested is not None:
        lines.append(f"Invested    : {format_currency(total_invested)}")
    pnl = state.get("total_unrealized_pnl")
    if pnl is not None:
        pnl_colour = colour_for_pnl(pnl)
        lines.append(f"PnL         : {pnl_colour}{pnl:,.0f} SEK{RESET}")

    if not holdings:
        lines.append("\nNo active positions.")
        return "\n".join(lines)

    lines.append("\nHoldings (by weight):")
    sorted_holdings = sorted(holdings, key=lambda h: h.get("weight", 0.0), reverse=True)[:limit]
    for holding in sorted_holdings:
        ticker = holding.get("ticker", "?")
        weight = float(holding.get("weight", 0.0))
        weight_pct = weight * 100
        bar = render_weight_bar(weight, width)
        colour = colour_for_pnl(holding.get("unrealized_pnl", 0.0))
        lines.append(f"  {ticker:<10} {weight_pct:5.1f}% |{colour}{bar}{RESET}|")
    if len(holdings) > limit:
        lines.append(f"  {DIM}... {len(holdings) - limit} more holdings{RESET}")
    return "\n".join(lines)


def find_latest_recommendations(directory: Path) -> Optional[Path]:
    files = sorted(directory.glob("recommendations_*.parquet"))
    return files[-1] if files else None


def load_recommendations(path: Optional[Path]) -> pd.DataFrame:
    target = path
    if target is None:
        default_dir = Path("data/recommendation_logs")
        target = find_latest_recommendations(default_dir)
        if target is None:
            return pd.DataFrame()
    if not target.exists():
        raise FileNotFoundError(f"Recommendations not found: {target}")
    return pd.read_parquet(target)


def format_recommendations(recs: pd.DataFrame, limit: int) -> str:
    if recs.empty:
        return "No recommendation data available."

    sections: List[str] = []
    for decision, title, colour in (
        ("Buy", "Buy Signals", GREEN),
        ("Sell", "Sell Signals", RED),
    ):
        subset = (
            recs[recs["decision"].str.lower() == decision.lower()]
            if "decision" in recs.columns
            else pd.DataFrame()
        )
        if subset.empty:
            continue
        ordered = subset.sort_values(
            by=[col for col in ["decision_confidence", "prob_positive", "expected_return"] if col in subset.columns],
            ascending=False,
        )
        sections.append(f"{colour}{title}:{RESET}")
        for _, row in ordered.head(limit).iterrows():
            ticker = row.get("ticker", "?")
            er = row.get("expected_return")
            prob = row.get("prob_positive")
            confidence = row.get("decision_confidence")
            weight = row.get("portfolio_weight")
            pieces = [f"  {ticker:<10}"]
            if pd.notna(er):
                pieces.append(f"E[r]: {er * 100:>+.2f}%")
            if pd.notna(prob):
                pieces.append(f"Pr(↑): {prob * 100:>5.1f}%")
            if pd.notna(confidence):
                pieces.append(f"Conf: {confidence * 100:>5.1f}% {render_confidence_bar(float(confidence))}")
            if pd.notna(weight):
                pieces.append(f"Wt: {weight * 100:>5.1f}%")
            sections.append(" | ".join(pieces))
    return "\n".join(sections) if sections else "No Buy/Sell signals in latest recommendations."


def render_recommendations_section(path: Optional[Path], limit: int) -> str:
    try:
        recs = load_recommendations(path)
    except FileNotFoundError as exc:
        return f"{RED}Recommendation error:{RESET} {exc}"

    lines = [f"{BOLD}Recommendations{RESET}"]
    lines.append(format_recommendations(recs, limit))
    return "\n".join(lines)


def build_dashboard(state_path: Path, bar_width: int, show_limit: int, rec_path: Optional[Path], rec_limit: int) -> str:
    state = load_portfolio_state(state_path)
    sections = [
        render_regime_section(),
        "",
        render_portfolio_section(state, bar_width, show_limit),
        "",
        render_recommendations_section(rec_path, rec_limit),
    ]
    return "\n".join(sections)


def main() -> None:
    parser = argparse.ArgumentParser(description="Display portfolio snapshot and current regime dashboard.")
    parser.add_argument(
        "--state",
        default="data/portfolio/current_state.json",
        help="Path to portfolio state JSON (default: %(default)s)",
    )
    parser.add_argument(
        "--bar-width",
        type=int,
        default=32,
        help="Width of the weight bars (default: %(default)s)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=8,
        help="Number of holdings to display (default: %(default)s)",
    )
    parser.add_argument(
        "--recommendations",
        type=Path,
        default=None,
        help="Path to recommendations parquet (defaults to latest in data/recommendation_logs)",
    )
    parser.add_argument(
        "--rec-limit",
        type=int,
        default=5,
        help="Number of recommendations per decision to show (default: %(default)s)",
    )
    args = parser.parse_args()

    try:
        dashboard = build_dashboard(
            Path(args.state),
            args.bar_width,
            args.limit,
            args.recommendations,
            args.rec_limit,
        )
    except FileNotFoundError as exc:
        print(f"{RED}Error:{RESET} {exc}")
        raise SystemExit(1) from exc

    print("\033[2J\033[H", end="")  # Clear screen
    print(dashboard)


if __name__ == "__main__":
    main()
