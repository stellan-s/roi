import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from ..regime.detector import MarketRegime
from ..risk.analytics import RiskAnalytics, PortfolioRiskProfile

@dataclass
class PortfolioConstraints:
    """Portfolio-level constraints och regler"""
    max_weight_per_stock: float = 0.10          # Max 10% per aktie
    max_single_regime_exposure: float = 0.85    # Max 85% i samma regim
    min_regime_diversification: bool = True     # Kräv minst 2 regimer
    pre_earnings_freeze_days: int = 5           # Ingen handel före earnings
    trade_cost_bps: int = 15                    # Transaction costs
    min_portfolio_positions: int = 3            # Minst 3 aktiva positioner
    bear_market_allocation: float = 0.60        # Max allokering i bear market

@dataclass
class PortfolioPosition:
    """Enskild position i portfolion"""
    ticker: str
    weight: float                    # Portfolio vikt (0-1)
    decision: str                    # Buy/Sell/Hold
    expected_return: float           # E[r] daglig
    prob_positive: float             # Pr(↑)
    regime: str                      # Marknadsregim
    decision_confidence: float       # Beslutssäkerhet

class PortfolioManager:
    """
    Portfolio-level management med regime diversification och disciplinmotor

    Ansvarar för:
    1. Position sizing baserat på risk och confidence
    2. Regime diversification rules
    3. Pre-earnings freeze
    4. Transaction cost optimization
    5. Risk budgeting över regimer
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config

        # Initialize risk analytics
        self.risk_analytics = RiskAnalytics(config)

        # Load constraints from config
        if config and 'policy' in config:
            policy = config['policy']
            self.constraints = PortfolioConstraints(
                max_weight_per_stock=policy.get('max_weight', 0.10),
                max_single_regime_exposure=policy.get('max_single_regime_exposure', 0.85),
                min_regime_diversification=policy.get('regime_diversification', True),
                pre_earnings_freeze_days=policy.get('pre_earnings_freeze_days', 5),
                trade_cost_bps=policy.get('trade_cost_bps', 15),
                min_portfolio_positions=policy.get('min_portfolio_positions', 3),
                bear_market_allocation=policy.get('bear_market_allocation', 0.60)
            )
        else:
            self.constraints = PortfolioConstraints()

    def apply_portfolio_rules(self, decisions: pd.DataFrame) -> pd.DataFrame:
        """
        Applicera portfolio-level regler på Bayesian decisions

        Input: DataFrame med Bayesian beslut
        Output: DataFrame med portfolio-justerade beslut och vikter
        """

        # Bara senaste datum för portfolio construction
        latest_date = decisions['date'].max()
        latest_decisions = decisions[decisions['date'] == latest_date].copy()

        if latest_decisions.empty:
            return decisions

        # Skapa PortfolioPosition objects
        positions = []
        for _, row in latest_decisions.iterrows():
            pos = PortfolioPosition(
                ticker=row['ticker'],
                weight=0.0,  # Kommer beräknas
                decision=row['decision'],
                expected_return=row['expected_return'],
                prob_positive=row['prob_positive'],
                regime=row['market_regime'],
                decision_confidence=row['decision_confidence']
            )
            positions.append(pos)

        # Applicera portfolio regler
        adjusted_positions = self._apply_regime_diversification(positions)
        adjusted_positions = self._apply_position_sizing(adjusted_positions)
        adjusted_positions = self._apply_transaction_cost_filter(adjusted_positions)

        # Konvertera tillbaka till DataFrame
        adjusted_df = self._positions_to_dataframe(adjusted_positions, latest_decisions)

        # Mergea med original data (bara latest date uppdaterad)
        other_dates = decisions[decisions['date'] != latest_date]
        final_result = pd.concat([other_dates, adjusted_df], ignore_index=True)

        return final_result.sort_values(['date', 'ticker'])

    def _apply_regime_diversification(self, positions: List[PortfolioPosition]) -> List[PortfolioPosition]:
        """
        Applicera regime diversification rules
        """

        if not self.constraints.min_regime_diversification:
            return positions

        # Räkna positions per regim (endast Buy/Sell decisions)
        active_positions = [p for p in positions if p.decision in ['Buy', 'Sell']]

        if len(active_positions) == 0:
            return positions

        # Regime distribution
        regime_counts = {}
        for pos in active_positions:
            regime_counts[pos.regime] = regime_counts.get(pos.regime, 0) + 1

        # Om alla är i samma regim, reducera exposure
        if len(regime_counts) == 1 and len(active_positions) > 3:
            single_regime = list(regime_counts.keys())[0]
            print(f"⚠️ Regime diversification varning: Alla {len(active_positions)} positioner i {single_regime} regim")

            # Behåll bara de starkaste signalerna, men säkerställ minimum antal
            active_positions.sort(key=lambda p: p.decision_confidence, reverse=True)
            max_positions = max(
                self.constraints.min_portfolio_positions,
                int(len(active_positions) * self.constraints.max_single_regime_exposure)
            )

            # Downgrade svagare positioner till Hold, men behåll minst min_positions
            downgrades = 0
            for i, pos in enumerate(positions):
                if (pos.decision in ['Buy', 'Sell'] and
                    i >= max_positions and
                    len(active_positions) - downgrades > self.constraints.min_portfolio_positions):
                    pos.decision = 'Hold'
                    downgrades += 1
                    print(f"  Downgraded {pos.ticker} till Hold (regime diversification)")

        return positions

    def _apply_position_sizing(self, positions: List[PortfolioPosition]) -> List[PortfolioPosition]:
        """
        Beräkna position sizes baserat på expected return, confidence och heavy-tail risk
        """

        # Bara aktiva positioner (Buy/Sell)
        active_buys = [p for p in positions if p.decision == 'Buy']
        active_sells = [p for p in positions if p.decision == 'Sell']  # För short eller hedge

        if not active_buys:
            # Ingen köp → alla vikter 0
            for pos in positions:
                pos.weight = 0.0
            return positions

        # Kelly-inspirerad position sizing med regime och tail risk adjustment
        total_weight_budget = 1.0

        # Justera budget baserat på regim
        if active_buys and active_buys[0].regime == 'bear':
            total_weight_budget = self.constraints.bear_market_allocation
            print(f"🐻 Bear market: Reducerar allokering till {total_weight_budget*100}%")

        # Risk-adjusted expected returns med tail risk penalty
        risk_adjusted_returns = []
        for pos in active_buys:
            # Justera E[r] med confidence och volatility proxy
            confidence_adj = pos.decision_confidence  # 0-1
            regime_stability = 0.8 if pos.regime != 'neutral' else 0.6  # Regime stability

            # Heavy-tail risk adjustment
            # High tail risk score → reducera position size
            tail_risk_penalty = 1.0 - (getattr(pos, 'tail_risk_score', 0.0) * 0.3)  # Max 30% reduction
            tail_risk_penalty = max(0.5, tail_risk_penalty)  # Minimum 50% of original size

            risk_adj_return = pos.expected_return * confidence_adj * regime_stability * tail_risk_penalty
            risk_adjusted_returns.append(max(0.0001, risk_adj_return))  # Minimum threshold

        # Proportional allocation baserat på risk-adjusted returns
        total_risk_adj_return = sum(risk_adjusted_returns)

        for i, pos in enumerate(active_buys):
            if total_risk_adj_return > 0:
                base_weight = (risk_adjusted_returns[i] / total_risk_adj_return) * total_weight_budget
                # Constrain till max weight
                pos.weight = min(base_weight, self.constraints.max_weight_per_stock)
            else:
                pos.weight = 0.0

        # Normalisera om total weight överstiger 1.0
        total_weight = sum(pos.weight for pos in active_buys)
        if total_weight > 1.0:
            for pos in active_buys:
                pos.weight = pos.weight / total_weight

        return positions

    def _apply_transaction_cost_filter(self, positions: List[PortfolioPosition]) -> List[PortfolioPosition]:
        """
        Filtrera bort trades med för höga transaction costs relativt expected return
        """

        cost_threshold = self.constraints.trade_cost_bps / 10000.0  # Convert bps to decimal

        for pos in positions:
            if pos.decision in ['Buy', 'Sell'] and pos.weight > 0:
                # Expected return måste överstiga transaction costs
                if abs(pos.expected_return) < cost_threshold * 2:  # 2x safety margin
                    print(f"⚠️ {pos.ticker}: Expected return {pos.expected_return*100:.3f}% < transaction costs {cost_threshold*2*100:.3f}%")
                    pos.decision = 'Hold'
                    pos.weight = 0.0

        return positions

    def _positions_to_dataframe(self, positions: List[PortfolioPosition], original_df: pd.DataFrame) -> pd.DataFrame:
        """Konvertera PortfolioPosition tillbaka till DataFrame format"""

        result_rows = []

        for _, orig_row in original_df.iterrows():
            # Hitta motsvarande position
            pos = next((p for p in positions if p.ticker == orig_row['ticker']), None)

            # Kopiera original rad
            new_row = orig_row.copy()

            # Uppdatera med portfolio adjustments
            if pos:
                new_row['decision'] = pos.decision
                new_row['portfolio_weight'] = pos.weight
                new_row['portfolio_adjusted'] = True
            else:
                new_row['portfolio_weight'] = 0.0
                new_row['portfolio_adjusted'] = False

            result_rows.append(new_row)

        return pd.DataFrame(result_rows)

    def get_portfolio_summary(self, decisions: pd.DataFrame) -> Dict:
        """Generate portfolio-level summary och diagnostics"""

        latest_date = decisions['date'].max()
        latest = decisions[decisions['date'] == latest_date]

        if 'portfolio_weight' not in latest.columns:
            latest = self.apply_portfolio_rules(decisions)
            latest = latest[latest['date'] == latest_date]

        active_positions = latest[latest['portfolio_weight'] > 0]

        summary = {
            'total_positions': len(active_positions),
            'total_weight': active_positions['portfolio_weight'].sum(),
            'avg_expected_return': active_positions['expected_return'].mean() if len(active_positions) > 0 else 0,
            'avg_confidence': active_positions['decision_confidence'].mean() if len(active_positions) > 0 else 0,
            'regime_distribution': active_positions['market_regime'].value_counts().to_dict(),
            'largest_position': active_positions['portfolio_weight'].max() if len(active_positions) > 0 else 0,
            'decision_distribution': latest['decision'].value_counts().to_dict()
        }

        return summary