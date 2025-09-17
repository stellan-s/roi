import pandas as pd
from typing import Optional, Dict
from ..bayesian.integration import BayesianPolicyEngine

# Global Bayesian engine instance (maintains state across calls)
_bayesian_engine = None

def get_bayesian_engine(config: Optional[Dict] = None) -> BayesianPolicyEngine:
    """Singleton pattern för Bayesian engine"""
    global _bayesian_engine
    if _bayesian_engine is None:
        _bayesian_engine = BayesianPolicyEngine(config)
    return _bayesian_engine

def simple_score(tech: pd.DataFrame, senti: pd.DataFrame) -> pd.DataFrame:
    """Legacy simple scoring - behålls för backward compatibility"""
    # merge dagsnivå
    t = tech.copy()
    t["date"]=pd.to_datetime(t["date"]).dt.date
    s = senti.rename(columns={"date":"date"})
    df = t.merge(s, on=["date","ticker"], how="left")
    df["sent_score"] = df["sent_score"].fillna(0).infer_objects()
    df["score"] = (df["above_sma"]*1) + (df["mom_rank"]>0.6).astype(int) + df["sent_score"]
    # beslut
    cond_buy   = df["score"] >= 2
    cond_sell  = df["score"] <= -1
    df["decision"] = "Hold"
    df.loc[cond_buy, "decision"]="Buy"
    df.loc[cond_sell,"decision"]="Sell"
    return df[["date","ticker","close","above_sma","mom_rank","sent_score","score","decision"]]

def bayesian_score(tech: pd.DataFrame, senti: pd.DataFrame, prices: Optional[pd.DataFrame] = None, config: Optional[Dict] = None) -> pd.DataFrame:
    """
    Bayesian signal combination med E[r], Pr(↑) och uncertainty quantification

    Returns enriched DataFrame med:
    - expected_return: E[r] daglig förväntad return
    - prob_positive: Pr(↑) sannolikhet för positiv return
    - confidence_lower/upper: Uncertainty bands
    - decision: Buy/Sell/Hold med Bayesian logic
    - decision_confidence: Confidence i beslut (0-1)
    - signal weights: Dynamiska vikter för trend/momentum/sentiment
    """
    engine = get_bayesian_engine(config)
    return engine.bayesian_score(tech, senti, prices)

def update_bayesian_beliefs(historical_predictions: pd.DataFrame,
                           actual_returns: pd.DataFrame,
                           horizon_days: int = 21) -> None:
    """
    Uppdatera Bayesian priors baserat på faktisk performance
    """
    engine = get_bayesian_engine()
    engine.update_with_performance(historical_predictions, actual_returns, horizon_days)

def get_bayesian_diagnostics() -> pd.DataFrame:
    """Hämta diagnostics om signal effectiveness"""
    engine = get_bayesian_engine()
    return engine.get_diagnostics()

def get_signal_history() -> pd.DataFrame:
    """Hämta historik av signal observations"""
    engine = get_bayesian_engine()
    return engine.get_signal_history()

def get_regime_info() -> dict:
    """Hämta aktuell marknadsregim information"""
    try:
        # Try to get regime info from latest recommendations file
        import os
        from pathlib import Path
        from datetime import datetime

        # Look for today's recommendations file
        today = datetime.now().strftime("%Y-%m-%d")
        rec_file = Path(f"data/recommendation_logs/recommendations_{today}.parquet")

        if rec_file.exists():
            recs = pd.read_parquet(rec_file)
            # Check for either column name (market_regime or regime)
            regime_col = None
            if 'market_regime' in recs.columns:
                regime_col = 'market_regime'
            elif 'regime' in recs.columns:
                regime_col = 'regime'

            if not recs.empty and regime_col:
                # Calculate regime distribution across stocks
                regime_counts = recs[regime_col].value_counts()
                total_stocks = len(recs)

                # Get the most common regime
                most_common_regime = regime_counts.index[0]
                regime_percentage = (regime_counts.iloc[0] / total_stocks) * 100

                # Map regime names to display format
                regime_display = {
                    'bull': 'Bull Market',
                    'bear': 'Bear Market',
                    'neutral': 'Neutral Market',
                    'unknown': 'Unknown'
                }.get(most_common_regime, most_common_regime.title())

                # Create distribution summary
                regime_distribution = []
                for regime, count in regime_counts.items():
                    pct = (count / total_stocks) * 100
                    display_name = {
                        'bull': 'Bull',
                        'bear': 'Bear',
                        'neutral': 'Neutral',
                        'unknown': 'Unknown'
                    }.get(regime, regime)
                    regime_distribution.append(f"{display_name}: {count} stocks ({pct:.0f}%)")

                return {
                    "regime": regime_display,
                    "confidence": regime_percentage / 100,
                    "explanation": f"Stock regime distribution: {', '.join(regime_distribution)}"
                }
    except Exception as e:
        print(f"Failed to load regime from recommendations: {e}")

    # Fallback to engine if file-based approach fails
    engine = get_bayesian_engine()

    # If it's an adaptive engine, get detailed explanation
    if hasattr(engine, 'get_regime_explanation'):
        detailed_explanation = engine.get_regime_explanation()
        current_regime = getattr(engine, 'current_regime', None)
        regime_probabilities = getattr(engine, 'regime_probabilities', {})

        if current_regime and regime_probabilities:
            regime_name = current_regime.value if hasattr(current_regime, 'value') else str(current_regime)
            confidence = regime_probabilities.get(current_regime, 0.33)

            return {
                "regime": regime_name.title(),
                "confidence": confidence,
                "explanation": detailed_explanation
            }

    # Final fallback to basic engine
    return engine.get_regime_info()

def get_regime_history() -> pd.DataFrame:
    """Hämta historik av regime detections"""
    engine = get_bayesian_engine()
    return engine.get_regime_history()
