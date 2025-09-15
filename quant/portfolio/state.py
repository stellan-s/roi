import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, date
from dataclasses import dataclass, asdict

@dataclass
class PortfolioHolding:
    """Enskild holding i portfolio"""
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
    """Complete portfolio state vid given tidpunkt"""
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
    Portfolio state tracking och persistence

    Håller koll på:
    - Aktuella holdings och values
    - Historisk performance
    - Trade execution history
    - Portfolio metrics över tid
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
            print(f"⚠️ Kunde inte ladda portfolio state: {e}")
            return None

    def _load_portfolio_history(self) -> List[Dict]:
        """Load historical portfolio snapshots"""
        if not self.history_file.exists():
            return []

        try:
            with open(self.history_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Kunde inte ladda portfolio history: {e}")
            return []

    def _load_trade_history(self) -> List[Dict]:
        """Load trade execution history"""
        if not self.trades_file.exists():
            return []

        try:
            with open(self.trades_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Kunde inte ladda trade history: {e}")
            return []

    def update_portfolio_state(self,
                             new_prices: pd.DataFrame,
                             as_of_date: Optional[str] = None) -> PortfolioState:
        """
        Uppdatera portfolio state med nya priser

        Args:
            new_prices: DataFrame med columns ['ticker', 'close']
            as_of_date: Datum för uppdatering (default: today)
        """

        if as_of_date is None:
            as_of_date = date.today().isoformat()

        if self.current_state is None:
            # Initialize empty portfolio
            self.current_state = PortfolioState(
                date=as_of_date,
                total_value=100000.0,  # Start med 100k cash
                cash=100000.0,
                holdings=[],
                total_invested=0.0,
                total_unrealized_pnl=0.0
            )

        # Update existing holdings med nya priser
        updated_holdings = []
        total_market_value = 0.0
        total_unrealized_pnl = 0.0

        for holding in self.current_state.holdings:
            # Hitta nya priset
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
                    weight=0.0,  # Kommer beräknas efter alla holdings uppdaterade
                    unrealized_pnl=new_unrealized_pnl,
                    date_acquired=holding.date_acquired
                )

                updated_holdings.append(updated_holding)
                total_market_value += new_market_value
                total_unrealized_pnl += new_unrealized_pnl
            else:
                # Behåll old price om inget nytt pris
                updated_holdings.append(holding)
                total_market_value += holding.market_value
                total_unrealized_pnl += holding.unrealized_pnl

        # Beräkna total value och weights
        total_portfolio_value = total_market_value + self.current_state.cash

        # Uppdatera weights
        for holding in updated_holdings:
            holding.weight = holding.market_value / total_portfolio_value if total_portfolio_value > 0 else 0

        # Skapa uppdaterad state
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
        Simulera trade execution baserat på beslut från ROI system

        Args:
            trade_decisions: DataFrame med Buy/Sell decisions och weights
            current_prices: DataFrame med aktuella priser
            cash_per_position: Cash att allokera per position

        Returns:
            List av executed trades
        """

        executed_trades = []

        # Filter för buy decisions med weight > 0
        buy_decisions = trade_decisions[
            (trade_decisions['decision'] == 'Buy') &
            (trade_decisions['portfolio_weight'] > 0)
        ]

        for _, row in buy_decisions.iterrows():
            ticker = row['ticker']
            target_weight = row['portfolio_weight']

            # Hitta current price
            price_row = current_prices[current_prices['ticker'] == ticker]
            if price_row.empty:
                continue

            current_price = price_row.iloc[0]['close']

            # Beräkna position size
            position_value = target_weight * self.current_state.total_value
            shares_to_buy = position_value / current_price

            # Check om vi har cash
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
                    'regime': row['market_regime'],
                    'decision_confidence': row['decision_confidence']
                }

                executed_trades.append(trade)

                # Uppdatera portfolio state
                self._add_holding(ticker, shares_to_buy, current_price)
                self.current_state.cash -= cost
                self.current_state.total_invested += cost

                print(f"📈 KÖPT {shares_to_buy:.0f} {ticker} @ {current_price:.2f} (värde: {cost:.0f})")

        # Save trades
        self.trade_history.extend(executed_trades)
        self._save_trade_history()
        self._save_current_state()

        return executed_trades

    def _add_holding(self, ticker: str, shares: float, price: float):
        """Lägg till eller uppdatera holding"""

        # Check om vi redan har position
        existing_holding = None
        for i, holding in enumerate(self.current_state.holdings):
            if holding.ticker == ticker:
                existing_holding = i
                break

        if existing_holding is not None:
            # Uppdatera existing position (average cost)
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
                weight=0.0,  # Kommer uppdateras senare
                unrealized_pnl=total_shares * price - total_cost,
                date_acquired=old_holding.date_acquired
            )
        else:
            # Skapa ny holding
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
        """Hämta portfolio summary för rapporter"""

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
            'invested': self.current_state.total_invested,
            'positions': len(self.current_state.holdings),
            'unrealized_pnl': self.current_state.total_unrealized_pnl,
            'portfolio_return': self.current_state.portfolio_return,
            'largest_position': max((h.weight for h in self.current_state.holdings), default=0),
            'holdings': [
                {
                    'ticker': h.ticker,
                    'shares': h.shares,
                    'weight': h.weight,
                    'value': h.market_value,
                    'pnl': h.unrealized_pnl,
                    'return': h.unrealized_pnl / h.cost_basis if h.cost_basis > 0 else 0
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