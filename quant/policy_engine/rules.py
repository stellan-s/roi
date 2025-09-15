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
    engine = get_bayesian_engine()
    return engine.get_regime_info()

def get_regime_history() -> pd.DataFrame:
    """Hämta historik av regime detections"""
    engine = get_bayesian_engine()
    return engine.get_regime_history()
