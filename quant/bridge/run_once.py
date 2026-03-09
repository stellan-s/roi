"""Run one quant job and emit normalized JSON for app-layer ingestion."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from typing import Any, Dict, List

import pandas as pd

# Ensure repository root is on sys.path when executed as a file.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from quant.adaptive_main import run_live_day as run_adaptive_day
from quant.config.loader import load_configuration
from quant.main import run_live_day as run_static_day


@dataclass
class BridgeResult:
    mode: str
    started_at_utc: str
    ended_at_utc: str
    summary: Dict[str, Any]
    recommendations: List[Dict[str, Any]]


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _to_json_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if isinstance(value, (datetime, date, pd.Timestamp)):
        return pd.to_datetime(value).isoformat()
    if isinstance(value, (bool, int, float, str)):
        return value
    try:
        return float(value)
    except Exception:
        return str(value)


def _build_audit_payload(row: pd.Series, mode: str, config: Dict[str, Any]) -> Dict[str, Any]:
    bayes_cfg = config.get("bayesian", {})
    thresholds_cfg = bayes_cfg.get("decision_thresholds", {})
    filtering_cfg = config.get("stock_factor_profiles", {}).get("dynamic_filtering", {})

    buy_probability = float(thresholds_cfg.get("buy_probability", 0.65))
    sell_probability = float(thresholds_cfg.get("sell_probability", 0.35))
    min_expected_return = float(thresholds_cfg.get("min_expected_return", 0.001))
    max_uncertainty = float(thresholds_cfg.get("max_uncertainty", 0.30))

    cost_threshold = float(filtering_cfg.get("cost_threshold", 0.002))
    conviction_threshold = float(filtering_cfg.get("conviction_threshold", 0.55))

    expected_return = _safe_float(row.get("expected_return"))
    prob_positive = _safe_float(row.get("prob_positive"))
    decision_confidence = _safe_float(row.get("decision_confidence"))
    uncertainty = _safe_float(row.get("uncertainty"))
    portfolio_weight = _safe_float(row.get("portfolio_weight"))
    decision = str(row.get("decision", "Hold"))

    buy_checks = {
        "probability_ge_buy_threshold": (
            prob_positive is not None and prob_positive >= buy_probability
        ),
        "expected_return_ge_min": (
            expected_return is not None and expected_return >= min_expected_return
        ),
        "uncertainty_le_max": uncertainty is not None and uncertainty <= max_uncertainty,
    }
    sell_checks = {
        "probability_le_sell_threshold": (
            prob_positive is not None and prob_positive <= sell_probability
        ),
        "expected_return_le_negative_min": (
            expected_return is not None and expected_return <= -min_expected_return
        ),
        "uncertainty_le_max": uncertainty is not None and uncertainty <= max_uncertainty,
    }
    filter_checks = {
        "expected_return_ge_cost_threshold": (
            expected_return is not None and expected_return >= cost_threshold
        ),
        "decision_confidence_ge_conviction_threshold": (
            decision_confidence is not None and decision_confidence >= conviction_threshold
        ),
        "probability_ge_50pct": prob_positive is not None and prob_positive >= 0.50,
    }

    if decision == "Buy":
        summary = "Buy decision: expected return and probability were strong enough after filtering/risk checks."
    elif decision == "Sell":
        summary = "Sell decision: downside probability and return profile triggered defensive action."
    else:
        summary = "Hold decision: score did not fully clear execution checks or was down-ranked by portfolio controls."

    if portfolio_weight is not None and portfolio_weight > 0:
        summary += f" Portfolio allocation set to {portfolio_weight:.1%}."
    else:
        summary += " Portfolio allocation is 0%."

    raw_fields: Dict[str, Any] = {}
    for col in row.index:
        raw_fields[str(col)] = _to_json_value(row.get(col))

    return {
        "summary": summary,
        "engine_mode": mode,
        "decision": decision,
        "thresholds": {
            "buy_probability": buy_probability,
            "sell_probability": sell_probability,
            "min_expected_return": min_expected_return,
            "max_uncertainty": max_uncertainty,
            "cost_threshold": cost_threshold,
            "conviction_threshold": conviction_threshold,
        },
        "checks": {
            "buy_checks": buy_checks,
            "sell_checks": sell_checks,
            "filter_checks": filter_checks,
            "allocated_capital": portfolio_weight is not None and portfolio_weight > 0,
        },
        "metrics": {
            "close": _safe_float(row.get("close")),
            "expected_return": expected_return,
            "prob_positive": prob_positive,
            "decision_confidence": decision_confidence,
            "uncertainty": uncertainty,
            "portfolio_weight": portfolio_weight,
        },
        "signals": {
            "above_sma": _to_json_value(row.get("above_sma")),
            "mom_rank": _safe_float(row.get("mom_rank")),
            "sent_score": _safe_float(row.get("sent_score")),
            "trend_weight": _safe_float(row.get("trend_weight")),
            "momentum_weight": _safe_float(row.get("momentum_weight")),
            "sentiment_weight": _safe_float(row.get("sentiment_weight")),
            "fundamentals_weight": _safe_float(row.get("fundamentals_weight")),
        },
        "regime": {
            "value": _to_json_value(row.get("regime", row.get("market_regime"))),
            "confidence": _safe_float(row.get("regime_confidence")),
        },
        "risk": {
            "tail_risk_score": _safe_float(row.get("tail_risk_score", row.get("tail_risk"))),
            "extreme_move_prob": _safe_float(row.get("extreme_move_prob")),
            "monte_carlo_prob_gain_20": _safe_float(row.get("monte_carlo_prob_gain_20")),
            "monte_carlo_prob_loss_20": _safe_float(row.get("monte_carlo_prob_loss_20")),
        },
        "raw_fields": raw_fields,
    }


def _normalize_recommendations(
    df: pd.DataFrame, mode: str, config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    if df is None or df.empty:
        return []

    normalized: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        date_val = row.get("date")
        if pd.notna(date_val):
            date_str = pd.to_datetime(date_val).isoformat()
        else:
            date_str = None

        audit = _build_audit_payload(row, mode, config)

        normalized.append(
            {
                "date": date_str,
                "ticker": str(row.get("ticker", "")),
                "decision": str(row.get("decision", "Hold")),
                "close": _safe_float(row.get("close")),
                "expected_return": _safe_float(row.get("expected_return")),
                "prob_positive": _safe_float(row.get("prob_positive")),
                "decision_confidence": _safe_float(row.get("decision_confidence")),
                "uncertainty": _safe_float(row.get("uncertainty")),
                "portfolio_weight": _safe_float(row.get("portfolio_weight")),
                "rationale": audit["summary"],
                "audit": audit,
            }
        )
    return normalized


def _summarize(df: pd.DataFrame) -> Dict[str, Any]:
    if df is None or df.empty:
        return {
            "recommendations_total": 0,
            "buy_count": 0,
            "sell_count": 0,
            "hold_count": 0,
        }

    decisions = df.get("decision", pd.Series(dtype=str)).astype(str)
    buy_count = int((decisions == "Buy").sum())
    sell_count = int((decisions == "Sell").sum())
    hold_count = int((decisions == "Hold").sum())
    return {
        "recommendations_total": int(len(df)),
        "buy_count": buy_count,
        "sell_count": sell_count,
        "hold_count": hold_count,
    }


def _run(mode: str, dry_run: bool = False) -> BridgeResult:
    started_at = datetime.now(timezone.utc)
    config = load_configuration()
    if dry_run:
        mock = pd.DataFrame(
            [
                {
                    "date": pd.Timestamp.now().normalize(),
                    "ticker": "AAPL",
                    "decision": "Buy",
                    "close": 200.0,
                    "expected_return": 0.012,
                    "prob_positive": 0.61,
                    "decision_confidence": 0.72,
                    "uncertainty": 0.28,
                    "portfolio_weight": 0.04,
                },
                {
                    "date": pd.Timestamp.now().normalize(),
                    "ticker": "MSFT",
                    "decision": "Hold",
                    "close": 410.0,
                    "expected_return": 0.002,
                    "prob_positive": 0.52,
                    "decision_confidence": 0.58,
                    "uncertainty": 0.42,
                    "portfolio_weight": 0.0,
                },
            ]
        )
        ended_at = datetime.now(timezone.utc)
        return BridgeResult(
            mode=mode,
            started_at_utc=started_at.isoformat(),
            ended_at_utc=ended_at.isoformat(),
            summary=_summarize(mock),
            recommendations=_normalize_recommendations(mock, mode, config),
        )

    with contextlib.redirect_stdout(sys.stderr):
        if mode == "adaptive":
            decisions = run_adaptive_day(config)
        else:
            decisions = run_static_day(config)

    ended_at = datetime.now(timezone.utc)
    return BridgeResult(
        mode=mode,
        started_at_utc=started_at.isoformat(),
        ended_at_utc=ended_at.isoformat(),
        summary=_summarize(decisions),
        recommendations=_normalize_recommendations(decisions, mode, config),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Bridge quant engine execution to JSON")
    parser.add_argument("--mode", choices=("static", "adaptive"), default="static")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    result = _run(mode=args.mode, dry_run=args.dry_run)
    print(json.dumps(asdict(result), default=str))


if __name__ == "__main__":
    main()
