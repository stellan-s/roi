import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, date
from dataclasses import dataclass, asdict

@dataclass
class PortfolioHolding:
    """Single holding inside the portfolio."""
    ticker: str
    shares: float
    avg_cost: float
    current_price: float
    market_value: float
    weight: float
    unrealized_pnl: float
    date_acquired: str

    @property
    def cost_basis(self) -> float:
        return self.shares * self.avg_cost

@dataclass
class PortfolioState:
    """Complete portfolio state at a specific point in time."""
    date: str
    total_value: float
    cash: float
    holdings: List[PortfolioHolding]
    total_invested: float
    total_unrealized_pnl: float

    @property
    def total_cost_basis(self) -> float:
        return sum(h.cost_basis for h in self.holdings)

    @property
    def portfolio_return(self) -> float:
        if self.total_cost_basis == 0:
            return 0.0
        return self.total_unrealized_pnl / self.total_cost_basis

class PortfolioTracker:
    """
    Portfolio state tracking and persistence layer.

    Keeps track of:
    - Current holdings and valuations
    - Historical performance
    - Trade execution history
    - Portfolio metrics over time
    """

    def __init__(self, data_dir: str = "data/portfolio"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.state_file = self.data_dir / "current_state.json"
        self.history_file = self.data_dir / "portfolio_history.json"
        self.trades_file = self.data_dir / "trade_history.json"

        # Load existing state
        self.current_state = self._load_current_state()
        self.portfolio_history = self._load_portfolio_history()
        self.trade_history = self._load_trade_history()

    def _load_current_state(self) -> Optional[PortfolioState]:
        """Load current portfolio state from disk"""
        if not self.state_file.exists():
            return None

        try:
            with open(self.state_file, 'r') as f:
                data = json.load(f)

            holdings = [PortfolioHolding(**h) for h in data.get('holdings', [])]

            return PortfolioState(
                date=data['date'],
                total_value=data['total_value'],
                cash=data['cash'],
                holdings=holdings,
                total_invested=data['total_invested'],
                total_unrealized_pnl=data['total_unrealized_pnl']
            )
        except Exception as e:
            print(f"⚠️ Could not load portfolio state: {e}")
            return None

    def _load_portfolio_history(self) -> List[Dict]:
        """Load historical portfolio snapshots"""
        if not self.history_file.exists():
            return []

        try:
            with open(self.history_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Could not load portfolio history: {e}")
            return []

    def _load_trade_history(self) -> List[Dict]:
        """Load trade execution history"""
        if not self.trades_file.exists():
            return []

        try:
            with open(self.trades_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Could not load trade history: {e}")
            return []

    def update_portfolio_state(self,
                             new_prices: pd.DataFrame,
                             as_of_date: Optional[str] = None) -> PortfolioState:
        """
        Update portfolio state with the latest prices.

        Args:
            new_prices: DataFrame with columns ['ticker', 'close']
            as_of_date: Date for the update (defaults to today)
        """

        if as_of_date is None:
            as_of_date = date.today().isoformat()

        if self.current_state is None:
            # Initialize empty portfolio
            self.current_state = PortfolioState(
                date=as_of_date,
                total_value=100000.0,  # Start with 100k cash
                cash=100000.0,
                holdings=[],
                total_invested=0.0,
                total_unrealized_pnl=0.0
            )

        # Update existing holdings with the new prices
        updated_holdings = []
        total_market_value = 0.0
        total_unrealized_pnl = 0.0

        for holding in self.current_state.holdings:
            # Locate the new price
            price_row = new_prices[new_prices['ticker'] == holding.ticker]

            if not price_row.empty:
                new_price = price_row.iloc[0]['close']
                new_market_value = holding.shares * new_price
                new_unrealized_pnl = new_market_value - holding.cost_basis

                updated_holding = PortfolioHolding(
                    ticker=holding.ticker,
                    shares=holding.shares,
                    avg_cost=holding.avg_cost,
                    current_price=new_price,
                    market_value=new_market_value,
                    weight=0.0,  # Recalculated after all holdings are updated
                    unrealized_pnl=new_unrealized_pnl,
                    date_acquired=holding.date_acquired
                )

                updated_holdings.append(updated_holding)
                total_market_value += new_market_value
                total_unrealized_pnl += new_unrealized_pnl
            else:
                # Keep the previous price if no fresh price is provided
                updated_holdings.append(holding)
                total_market_value += holding.market_value
                total_unrealized_pnl += holding.unrealized_pnl

        # Compute total value and weights
        total_portfolio_value = total_market_value + self.current_state.cash

        # Update weights
        for holding in updated_holdings:
            holding.weight = holding.market_value / total_portfolio_value if total_portfolio_value > 0 else 0

        # Create the updated state
        updated_state = PortfolioState(
            date=as_of_date,
            total_value=total_portfolio_value,
            cash=self.current_state.cash,
            holdings=updated_holdings,
            total_invested=self.current_state.total_invested,
            total_unrealized_pnl=total_unrealized_pnl
        )

        self.current_state = updated_state
        self._save_current_state()

        return updated_state

    def execute_trades(self,
                      trade_decisions: pd.DataFrame,
                      current_prices: pd.DataFrame,
                      cash_per_position: float = 10000.0) -> List[Dict]:
        """
        Simulate trade execution based on ROI system decisions.

        Args:
            trade_decisions: DataFrame containing Buy/Sell decisions and weights
            current_prices: DataFrame with current prices
            cash_per_position: Cash allocation per position

        Returns:
            List of executed trades
        """

        executed_trades = []

        # Filter for buy decisions with weight > 0
        buy_decisions = trade_decisions[
            (trade_decisions['decision'] == 'Buy') &
            (trade_decisions['portfolio_weight'] > 0)
        ]

        for _, row in buy_decisions.iterrows():
            ticker = row['ticker']
            target_weight = row['portfolio_weight']

            # Find the current price
            price_row = current_prices[current_prices['ticker'] == ticker]
            if price_row.empty:
                continue

            current_price = price_row.iloc[0]['close']

            # Calculate position size
            position_value = target_weight * self.current_state.total_value
            shares_to_buy = position_value / current_price

            # Check if we have enough cash
            cost = shares_to_buy * current_price
            if cost <= self.current_state.cash:
                # Execute trade
                trade = {
                    'date': self.current_state.date,
                    'ticker': ticker,
                    'action': 'BUY',
                    'shares': shares_to_buy,
                    'price': current_price,
                    'value': cost,
                    'expected_return': row['expected_return'],
                    'prob_positive': row['prob_positive'],
                    'regime': row.get('market_regime', row.get('regime', 'unknown')),
                    'decision_confidence': row['decision_confidence']
                }

                executed_trades.append(trade)

                # Update portfolio state
                self._add_holding(ticker, shares_to_buy, current_price)
                self.current_state.cash -= cost
                self.current_state.total_invested += cost

                print(f"📈 BOUGHT {shares_to_buy:.0f} {ticker} @ {current_price:.2f} (value: {cost:.0f})")

        # Save trades
        self.trade_history.extend(executed_trades)
        self._save_trade_history()
        self._save_current_state()

        return executed_trades

    def _add_holding(self, ticker: str, shares: float, price: float):
        """Add or update a holding."""

        # Check if the position already exists
        existing_holding = None
        for i, holding in enumerate(self.current_state.holdings):
            if holding.ticker == ticker:
                existing_holding = i
                break

        if existing_holding is not None:
            # Update existing position (recalculate average cost)
            old_holding = self.current_state.holdings[existing_holding]
            total_shares = old_holding.shares + shares
            total_cost = old_holding.cost_basis + (shares * price)
            new_avg_cost = total_cost / total_shares

            self.current_state.holdings[existing_holding] = PortfolioHolding(
                ticker=ticker,
                shares=total_shares,
                avg_cost=new_avg_cost,
                current_price=price,
                market_value=total_shares * price,
                weight=0.0,  # Will be updated later
                unrealized_pnl=total_shares * price - total_cost,
                date_acquired=old_holding.date_acquired
            )
        else:
            # Create a new holding
            new_holding = PortfolioHolding(
                ticker=ticker,
                shares=shares,
                avg_cost=price,
                current_price=price,
                market_value=shares * price,
                weight=0.0,
                unrealized_pnl=0.0,
                date_acquired=self.current_state.date
            )

            self.current_state.holdings.append(new_holding)

    def get_portfolio_summary(self) -> Dict:
        """Return a portfolio summary for reporting."""

        if self.current_state is None:
            return {
                'total_value': 0,
                'cash': 0,
                'invested': 0,
                'positions': 0,
                'unrealized_pnl': 0,
                'portfolio_return': 0
            }

        return {
            'date': self.current_state.date,
            'total_value': self.current_state.total_value,
            'cash': self.current_state.cash,
            'total_invested': self.current_state.total_invested,  # Match daily report field name
            'invested': self.current_state.total_invested,        # Backward compatibility
            'positions': len(self.current_state.holdings),
            'total_unrealized_pnl': self.current_state.total_unrealized_pnl,  # Match daily report field name
            'unrealized_pnl': self.current_state.total_unrealized_pnl,        # Backward compatibility
            'portfolio_return': self.current_state.portfolio_return,
            'largest_position': max((h.weight for h in self.current_state.holdings), default=0),
            'holdings': [
                {
                    'ticker': h.ticker,
                    'shares': h.shares,
                    'avg_cost': h.avg_cost,                       # Add field needed by daily report
                    'current_price': h.current_price,            # Add field needed by daily report
                    'market_value': h.market_value,              # Add field needed by daily report
                    'weight': h.weight,
                    'value': h.market_value,                     # Backward compatibility
                    'unrealized_pnl': h.unrealized_pnl,         # Add field needed by daily report
                    'pnl': h.unrealized_pnl,                    # Backward compatibility
                    'return': h.unrealized_pnl / h.cost_basis if h.cost_basis > 0 else 0,
                    'date_acquired': h.date_acquired             # Add field needed by daily report
                }
                for h in self.current_state.holdings
            ]
        }

    def _save_current_state(self):
        """Save current state to disk"""
        if self.current_state:
            data = asdict(self.current_state)
            with open(self.state_file, 'w') as f:
                json.dump(data, f, indent=2)

    def _save_portfolio_history(self):
        """Save portfolio history"""
        with open(self.history_file, 'w') as f:
            json.dump(self.portfolio_history, f, indent=2)

    def _save_trade_history(self):
        """Save trade history"""
        with open(self.trades_file, 'w') as f:
            json.dump(self.trade_history, f, indent=2)
