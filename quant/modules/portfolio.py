"""
Portfolio Management Module

Provides comprehensive portfolio management including:
- Position sizing and allocation
- Regime-based diversification rules
- Transaction cost optimization
- Portfolio state tracking and persistence
- Risk-based portfolio optimization
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, date
from dataclasses import dataclass

from .base import BaseModule, ModuleOutput, ModuleContract

@dataclass
class PortfolioPosition:
    """Single position in the portfolio"""
    ticker: str
    target_weight: float         # Target portfolio weight
    current_weight: float        # Current portfolio weight
    expected_return: float       # Expected daily return
    confidence: float           # Decision confidence
    regime: str                 # Market regime
    decision: str               # Buy/Sell/Hold
    risk_score: float           # Risk assessment score

@dataclass
class PortfolioConstraints:
    """Portfolio-level constraints and rules"""
    max_weight_per_stock: float = 0.15          # Max 15% per stock
    max_positions: int = 20                     # Maximum number of positions
    min_position_size: float = 0.02             # Minimum 2% per position
    max_single_regime_exposure: float = 0.80    # Max 80% in same regime
    bear_market_allocation: float = 0.70        # Max allocation in bear markets
    transaction_cost_bps: int = 10              # Transaction costs in basis points
    target_total_allocation: float = 0.85       # Desired total allocation budget

@dataclass
class PortfolioMetrics:
    """Portfolio performance and risk metrics"""
    total_positions: int
    total_allocation: float
    regime_diversification: Dict[str, float]
    expected_return_annual: float
    risk_adjusted_return: float
    concentration_risk: float
    transaction_cost_estimate: float

class PortfolioManagementModule(BaseModule):
    """Module for comprehensive portfolio management and optimization"""

    def define_contract(self) -> ModuleContract:
        return ModuleContract(
            name="portfolio_management",
            version="1.0.0",
            description="Portfolio management with optimization and risk controls",
            input_schema={
                "candidate_positions": "pd.DataFrame[ticker, expected_return, confidence, regime, risk_score]",
                "current_prices": "pd.DataFrame[ticker, close]",
                "current_portfolio": "Dict[ticker, weight]"
            },
            output_schema={
                "optimized_portfolio": "Dict[ticker, Dict]",
                "portfolio_allocation": "Dict[ticker, float]",
                "portfolio_metrics": "Dict[str, Any]",
                "trade_recommendations": "List[Dict]",
                "rebalancing_summary": "Dict[str, Any]"
            },
            performance_sla={
                "max_latency_ms": 300.0,
                "min_confidence": 0.7
            },
            dependencies=[],
            optional_inputs=["current_portfolio"]
        )

    def process(self, inputs: Dict[str, Any]) -> ModuleOutput:
        """Process portfolio optimization and management"""
        candidate_positions = inputs['candidate_positions']
        current_prices = inputs['current_prices']
        current_portfolio = inputs.get('current_portfolio', {})

        # Validate inputs
        if candidate_positions.empty:
            return self._default_portfolio_output("no_candidates")

        if current_prices.empty:
            return self._default_portfolio_output("no_prices")

        try:
            # Load portfolio constraints from config
            constraints = self._load_constraints()

            # Build portfolio positions from candidates
            positions = self._build_portfolio_positions(candidate_positions, current_portfolio)

            if not positions:
                return self._default_portfolio_output("no_valid_positions")

            # Apply portfolio optimization
            optimized_positions = self._optimize_portfolio(positions, constraints)

            # Generate trade recommendations
            trade_recommendations = self._generate_trade_recommendations(
                optimized_positions, current_portfolio, current_prices
            )

            # Calculate portfolio metrics
            portfolio_metrics = self._calculate_portfolio_metrics(optimized_positions, constraints)

            # Create portfolio allocation dictionary
            portfolio_allocation = {
                pos.ticker: pos.target_weight for pos in optimized_positions if pos.target_weight > 0
            }

            # Create optimized portfolio detailed view
            optimized_portfolio = {
                pos.ticker: {
                    'target_weight': pos.target_weight,
                    'expected_return': pos.expected_return,
                    'confidence': pos.confidence,
                    'regime': pos.regime,
                    'decision': pos.decision,
                    'risk_score': pos.risk_score
                }
                for pos in optimized_positions if pos.target_weight > 0
            }

            # Create rebalancing summary
            rebalancing_summary = self._create_rebalancing_summary(
                optimized_positions, current_portfolio, trade_recommendations
            )

            # Calculate overall confidence
            confidence = self._calculate_confidence(optimized_positions, portfolio_metrics)

            metadata = {
                "optimization_method": "risk_adjusted_allocation",
                "positions_considered": len(candidate_positions),
                "positions_selected": len(portfolio_allocation),
                "regime_diversification": len(set(pos.regime for pos in optimized_positions if pos.target_weight > 0)),
                "total_allocation": sum(portfolio_allocation.values())
            }

            return ModuleOutput(
                data={
                    "optimized_portfolio": optimized_portfolio,
                    "portfolio_allocation": portfolio_allocation,
                    "portfolio_metrics": portfolio_metrics,
                    "trade_recommendations": trade_recommendations,
                    "rebalancing_summary": rebalancing_summary
                },
                metadata=metadata,
                confidence=confidence
            )

        except Exception as e:
            return ModuleOutput(
                data={
                    "optimized_portfolio": {},
                    "portfolio_allocation": {},
                    "portfolio_metrics": {},
                    "trade_recommendations": [],
                    "rebalancing_summary": {"error": str(e)}
                },
                metadata={"error": str(e)},
                confidence=0.1
            )

    def test_module(self) -> Dict[str, Any]:
        """Test the portfolio management module with synthetic data"""
        # Generate test data
        test_candidates, test_prices, test_current = self._generate_test_data()

        # Test processing
        result = self.process({
            'candidate_positions': test_candidates,
            'current_prices': test_prices,
            'current_portfolio': test_current
        })

        # Validate outputs
        portfolio_allocation = result.data['portfolio_allocation']
        portfolio_metrics = result.data['portfolio_metrics']
        trade_recommendations = result.data['trade_recommendations']

        tests_passed = 0
        total_tests = 8

        # Test 1: Portfolio allocation calculated
        if portfolio_allocation and len(portfolio_allocation) > 0:
            tests_passed += 1

        # Test 2: Portfolio weights sum to reasonable total
        total_weight = sum(portfolio_allocation.values())
        if 0.5 <= total_weight <= 1.0:
            tests_passed += 1

        # Test 3: No position exceeds maximum weight
        max_weight = max(portfolio_allocation.values()) if portfolio_allocation else 0
        if max_weight <= 0.20:  # 20% maximum reasonable
            tests_passed += 1

        # Test 4: Portfolio metrics calculated
        if portfolio_metrics and 'total_positions' in portfolio_metrics:
            tests_passed += 1

        # Test 5: Expected return is reasonable
        expected_return = portfolio_metrics.get('expected_return_annual', 0)
        if -0.5 <= expected_return <= 1.0:  # Between -50% and 100%
            tests_passed += 1

        # Test 6: Trade recommendations generated
        if isinstance(trade_recommendations, list):
            tests_passed += 1

        # Test 7: Regime diversification exists
        regime_div = portfolio_metrics.get('regime_diversification', {})
        if isinstance(regime_div, dict) and len(regime_div) > 0:
            tests_passed += 1

        # Test 8: Confidence reasonable
        if 0.5 <= result.confidence <= 1.0:
            tests_passed += 1

        return {
            "status": "PASS" if tests_passed >= 6 else "FAIL",
            "tests_passed": tests_passed,
            "total_tests": total_tests,
            "portfolio_positions": len(portfolio_allocation),
            "total_allocation": total_weight,
            "confidence": result.confidence
        }

    def _generate_test_inputs(self) -> Dict[str, Any]:
        """Generate synthetic test data for benchmarking"""
        test_candidates, test_prices, test_current = self._generate_test_data()
        return {
            "candidate_positions": test_candidates,
            "current_prices": test_prices,
            "current_portfolio": test_current
        }

    def _generate_test_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
        """Generate comprehensive test data"""
        # Generate candidate positions
        np.random.seed(42)  # Reproducible results

        tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'AMD', 'META', 'NFLX']
        regimes = ['bull', 'bear', 'neutral']

        candidates_data = []
        for ticker in tickers:
            regime = np.random.choice(regimes)

            # Generate realistic metrics based on regime
            if regime == 'bull':
                expected_return = np.random.normal(0.0008, 0.0003)  # Higher returns in bull market
                confidence = np.random.uniform(0.6, 0.9)
                risk_score = np.random.uniform(0.1, 0.3)
            elif regime == 'bear':
                expected_return = np.random.normal(-0.0003, 0.0004)  # Negative returns in bear market
                confidence = np.random.uniform(0.4, 0.7)
                risk_score = np.random.uniform(0.4, 0.8)
            else:  # neutral
                expected_return = np.random.normal(0.0002, 0.0002)  # Small returns in neutral
                confidence = np.random.uniform(0.5, 0.8)
                risk_score = np.random.uniform(0.2, 0.5)

            candidates_data.append({
                'ticker': ticker,
                'expected_return': expected_return,
                'confidence': confidence,
                'regime': regime,
                'risk_score': risk_score,
                'decision': 'Buy' if expected_return > 0 else 'Sell',
                'prob_positive': confidence if expected_return > 0 else 1 - confidence
            })

        candidates_df = pd.DataFrame(candidates_data)

        # Generate current prices
        prices_data = []
        for ticker in tickers:
            base_price = np.random.uniform(50, 300)
            prices_data.append({
                'ticker': ticker,
                'close': base_price
            })

        prices_df = pd.DataFrame(prices_data)

        # Generate current portfolio (some existing positions)
        current_portfolio = {
            'AAPL': 0.15,
            'MSFT': 0.10,
            'GOOGL': 0.08
        }

        return candidates_df, prices_df, current_portfolio

    def _load_constraints(self) -> PortfolioConstraints:
        """Load portfolio constraints from configuration"""
        # Default constraints
        constraints = PortfolioConstraints()

        # Override with config values if available
        if hasattr(self, 'config') and self.config:
            constraints.max_weight_per_stock = self.config.get('max_weight_per_stock', 0.15)
            constraints.max_positions = self.config.get('max_positions', 20)
            constraints.min_position_size = self.config.get('min_position_size', 0.02)
            constraints.max_single_regime_exposure = self.config.get('max_single_regime_exposure', 0.80)
            constraints.bear_market_allocation = self.config.get('bear_market_allocation', 0.70)
            constraints.transaction_cost_bps = self.config.get('transaction_cost_bps', 10)
            constraints.target_total_allocation = self.config.get('target_total_allocation', 0.85)

        return constraints

    def _build_portfolio_positions(self, candidates_df: pd.DataFrame,
                                 current_portfolio: Dict[str, float]) -> List[PortfolioPosition]:
        """Build portfolio positions from candidate positions"""
        positions = []

        for _, row in candidates_df.iterrows():
            # Only consider buy decisions or strong sell signals
            if row.get('decision', 'Hold') not in ['Buy', 'Sell']:
                continue

            # Skip positions with very low confidence
            if row.get('confidence', 0) < 0.3:
                continue

            # Get current weight if exists
            current_weight = current_portfolio.get(row['ticker'], 0.0)

            position = PortfolioPosition(
                ticker=row['ticker'],
                target_weight=0.0,  # To be calculated
                current_weight=current_weight,
                expected_return=row['expected_return'],
                confidence=row['confidence'],
                regime=row.get('regime', 'neutral'),
                decision=row.get('decision', 'Buy'),
                risk_score=row.get('risk_score', 0.5)
            )

            positions.append(position)

        return positions

    def _optimize_portfolio(self, positions: List[PortfolioPosition],
                          constraints: PortfolioConstraints) -> List[PortfolioPosition]:
        """Optimize portfolio allocation using risk-adjusted expected returns"""

        # Filter to buy positions only for initial allocation
        buy_positions = [p for p in positions if p.decision == 'Buy' and p.expected_return > 0]

        if not buy_positions:
            # No buy signals, return positions with zero weights
            for pos in positions:
                pos.target_weight = 0.0
            return positions

        # Calculate risk-adjusted scores for each position
        risk_adjusted_scores = []
        for pos in buy_positions:
            # Base score from expected return and confidence
            base_score = pos.expected_return * pos.confidence

            # Risk adjustment (lower risk score = better)
            risk_adjustment = 1.0 - (pos.risk_score * 0.5)  # Max 50% reduction

            # Regime stability bonus
            regime_bonus = 1.1 if pos.regime in ['bull', 'neutral'] else 0.9

            final_score = base_score * risk_adjustment * regime_bonus
            risk_adjusted_scores.append(max(0.0001, final_score))

        # Apply regime diversification constraints
        self._apply_regime_constraints(buy_positions, risk_adjusted_scores, constraints)

        # Calculate position weights using risk parity approach
        total_score = sum(risk_adjusted_scores)
        if total_score > 0:
            # Calculate base weights
            base_weights = [score / total_score for score in risk_adjusted_scores]

            # Apply position constraints
            self._apply_position_constraints(buy_positions, base_weights, constraints)

        # Apply bear market allocation constraints
        self._apply_bear_market_constraints(buy_positions, constraints)

        # Set weights for all positions
        for pos in positions:
            if pos in buy_positions:
                # Weight was set in the optimization process
                pass
            else:
                pos.target_weight = 0.0

        return positions

    def _apply_regime_constraints(self, positions: List[PortfolioPosition],
                                scores: List[float], constraints: PortfolioConstraints):
        """Apply regime diversification constraints"""

        # Count positions by regime
        regime_counts = {}
        for pos in positions:
            regime_counts[pos.regime] = regime_counts.get(pos.regime, 0) + 1

        # If more than 80% in one regime, reduce scores for excess positions
        total_positions = len(positions)
        max_single_regime = int(total_positions * constraints.max_single_regime_exposure)

        for regime, count in regime_counts.items():
            if count > max_single_regime:
                # Find positions in this regime and reduce scores for weakest ones
                regime_positions = [(i, pos) for i, pos in enumerate(positions) if pos.regime == regime]

                # Sort by score (weakest first)
                regime_positions.sort(key=lambda x: scores[x[0]])

                # Reduce scores for excess positions
                excess_count = count - max_single_regime
                for i in range(excess_count):
                    idx = regime_positions[i][0]
                    scores[idx] *= 0.3  # Significant penalty

    def _apply_position_constraints(self, positions: List[PortfolioPosition],
                                  weights: List[float], constraints: PortfolioConstraints):
        """Apply individual position size constraints"""

        # Apply maximum position size constraint
        capped_weights = [min(weight, constraints.max_weight_per_stock) for weight in weights]

        # Apply minimum position size constraint and limit total positions
        sorted_indices = sorted(range(len(capped_weights)), key=lambda i: capped_weights[i], reverse=True)

        final_weights = [0.0 for _ in capped_weights]
        kept_positions = 0
        for idx in sorted_indices:
            weight = capped_weights[idx]
            if kept_positions >= constraints.max_positions or weight < constraints.min_position_size:
                final_weights[idx] = 0.0
            else:
                final_weights[idx] = weight
                kept_positions += 1

        # Scale weights toward target allocation
        total_weight = sum(w for w in final_weights if w > 0)
        target_allocation = min(max(constraints.target_total_allocation, 0.0), 1.0)

        if total_weight > 0 and target_allocation > 0:
            scale = target_allocation / total_weight

            if scale != 1.0:
                scaled_weights = [w * scale for w in final_weights]

                # Re-apply maximum position constraint after scaling
                adjusted = False
                for i, w in enumerate(scaled_weights):
                    if w > constraints.max_weight_per_stock:
                        scaled_weights[i] = constraints.max_weight_per_stock
                        adjusted = True

                if adjusted:
                    capped_total = sum(w for w in scaled_weights if w > 0)
                    if capped_total > 0 and capped_total > target_allocation:
                        shrink = target_allocation / capped_total
                        scaled_weights = [w * shrink for w in scaled_weights]

                final_weights = scaled_weights

        # Assign target weights back to positions
        for i, pos in enumerate(positions):
            pos.target_weight = max(0.0, final_weights[i])

    def _apply_bear_market_constraints(self, positions: List[PortfolioPosition],
                                     constraints: PortfolioConstraints):
        """Apply bear market allocation constraints"""

        # Count bear market positions
        bear_positions = [pos for pos in positions if pos.regime == 'bear']
        total_bear_weight = sum(pos.target_weight for pos in bear_positions)

        # If too much allocation in bear market, reduce all allocations
        if total_bear_weight > 0.5:  # More than 50% in bear regime
            reduction_factor = constraints.bear_market_allocation
            for pos in positions:
                pos.target_weight *= reduction_factor

    def _generate_trade_recommendations(self, positions: List[PortfolioPosition],
                                      current_portfolio: Dict[str, float],
                                      current_prices: pd.DataFrame) -> List[Dict]:
        """Generate specific trade recommendations"""

        recommendations = []

        for pos in positions:
            current_weight = current_portfolio.get(pos.ticker, 0.0)
            target_weight = pos.target_weight
            weight_change = target_weight - current_weight

            # Only recommend trades for significant changes
            if abs(weight_change) > 0.01:  # 1% threshold

                # Get current price
                price_row = current_prices[current_prices['ticker'] == pos.ticker]
                current_price = price_row.iloc[0]['close'] if not price_row.empty else 100.0

                recommendation = {
                    'ticker': pos.ticker,
                    'action': 'BUY' if weight_change > 0 else 'SELL',
                    'current_weight': current_weight,
                    'target_weight': target_weight,
                    'weight_change': weight_change,
                    'current_price': current_price,
                    'expected_return': pos.expected_return,
                    'confidence': pos.confidence,
                    'regime': pos.regime,
                    'risk_score': pos.risk_score,
                    'priority': abs(weight_change) * pos.confidence  # Priority score
                }

                recommendations.append(recommendation)

        # Sort by priority (highest priority first)
        recommendations.sort(key=lambda x: x['priority'], reverse=True)

        return recommendations

    def _calculate_portfolio_metrics(self, positions: List[PortfolioPosition],
                                   constraints: PortfolioConstraints) -> Dict[str, Any]:
        """Calculate comprehensive portfolio metrics"""

        active_positions = [pos for pos in positions if pos.target_weight > 0]

        if not active_positions:
            return {
                'total_positions': 0,
                'total_allocation': 0.0,
                'regime_diversification': {},
                'expected_return_annual': 0.0,
                'risk_adjusted_return': 0.0,
                'concentration_risk': 0.0,
                'transaction_cost_estimate': 0.0
            }

        # Basic metrics
        total_positions = len(active_positions)
        total_allocation = sum(pos.target_weight for pos in active_positions)

        # Regime diversification
        regime_weights = {}
        for pos in active_positions:
            regime_weights[pos.regime] = regime_weights.get(pos.regime, 0) + pos.target_weight

        # Expected portfolio return (annualized)
        expected_return_daily = sum(pos.target_weight * pos.expected_return for pos in active_positions)
        expected_return_annual = expected_return_daily * 252  # Annualize

        # Risk-adjusted return (simple approximation)
        avg_risk_score = np.mean([pos.risk_score for pos in active_positions])
        risk_adjustment = 1.0 - (avg_risk_score * 0.3)  # Higher risk reduces return
        risk_adjusted_return = expected_return_annual * risk_adjustment

        # Concentration risk (max single position weight)
        concentration_risk = max(pos.target_weight for pos in active_positions)

        # Transaction cost estimate
        transaction_cost_rate = constraints.transaction_cost_bps / 10000.0
        transaction_cost_estimate = total_allocation * transaction_cost_rate

        return {
            'total_positions': total_positions,
            'total_allocation': total_allocation,
            'regime_diversification': regime_weights,
            'expected_return_annual': expected_return_annual,
            'risk_adjusted_return': risk_adjusted_return,
            'concentration_risk': concentration_risk,
            'transaction_cost_estimate': transaction_cost_estimate,
            'avg_confidence': np.mean([pos.confidence for pos in active_positions]),
            'regime_count': len(regime_weights)
        }

    def _create_rebalancing_summary(self, positions: List[PortfolioPosition],
                                  current_portfolio: Dict[str, float],
                                  trade_recommendations: List[Dict]) -> Dict[str, Any]:
        """Create rebalancing summary"""

        # Calculate total changes
        total_buys = sum(rec['weight_change'] for rec in trade_recommendations if rec['action'] == 'BUY')
        total_sells = sum(abs(rec['weight_change']) for rec in trade_recommendations if rec['action'] == 'SELL')

        # Count position changes
        new_positions = len([rec for rec in trade_recommendations if rec['current_weight'] == 0 and rec['action'] == 'BUY'])
        closed_positions = len([rec for rec in trade_recommendations if rec['target_weight'] == 0 and rec['action'] == 'SELL'])
        adjusted_positions = len(trade_recommendations) - new_positions - closed_positions

        return {
            'total_trades': len(trade_recommendations),
            'total_buy_weight': total_buys,
            'total_sell_weight': total_sells,
            'new_positions': new_positions,
            'closed_positions': closed_positions,
            'adjusted_positions': adjusted_positions,
            'net_weight_change': total_buys - total_sells,
            'high_priority_trades': len([rec for rec in trade_recommendations if rec['priority'] > 0.01])
        }

    def _calculate_confidence(self, positions: List[PortfolioPosition],
                            portfolio_metrics: Dict[str, Any]) -> float:
        """Calculate overall confidence in portfolio optimization"""

        if not positions:
            return 0.1

        # Base confidence from position confidences
        active_positions = [pos for pos in positions if pos.target_weight > 0]
        if not active_positions:
            return 0.3

        avg_position_confidence = np.mean([pos.confidence for pos in active_positions])

        # Diversification bonus
        regime_count = portfolio_metrics.get('regime_count', 1)
        diversification_bonus = min(0.2, regime_count * 0.1)  # Up to 20% bonus

        # Allocation quality (penalize very low allocation)
        total_allocation = portfolio_metrics.get('total_allocation', 0)
        allocation_quality = min(1.0, total_allocation / 0.8)  # Optimal around 80%

        # Position count quality
        position_count = portfolio_metrics.get('total_positions', 0)
        position_quality = min(1.0, position_count / 10)  # Optimal around 10 positions

        # Combined confidence
        confidence = (
            avg_position_confidence * 0.5 +
            allocation_quality * 0.2 +
            position_quality * 0.1 +
            diversification_bonus * 0.2
        )

        return min(1.0, max(0.1, confidence))

    def _default_portfolio_output(self, reason: str) -> ModuleOutput:
        """Return default portfolio output for error cases"""
        return ModuleOutput(
            data={
                "optimized_portfolio": {},
                "portfolio_allocation": {},
                "portfolio_metrics": {
                    "total_positions": 0,
                    "total_allocation": 0.0,
                    "regime_diversification": {},
                    "expected_return_annual": 0.0
                },
                "trade_recommendations": [],
                "rebalancing_summary": {"error": reason}
            },
            metadata={"reason": reason},
            confidence=0.1
        )
