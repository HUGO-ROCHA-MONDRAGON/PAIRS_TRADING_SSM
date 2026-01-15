"""
Transaction cost models for backtesting.

Default: 20bp per asset (0.002) as in Zhang (2021).
"""

from dataclasses import dataclass


@dataclass
class TransactionCosts:
    """
    Transaction cost model.
    
    Attributes
    ----------
    cost_per_trade : float
        Cost per unit traded per asset (default 20bp = 0.002)
    half_spread : float
        Half bid-ask spread (alternative cost model)
    slippage : float
        Expected slippage per trade
    
    The paper uses 20bp per asset, so round-trip cost is 40bp per asset
    (80bp total for the pair).
    """
    
    cost_per_trade: float = 0.002  # 20bp per asset
    half_spread: float = 0.0
    slippage: float = 0.0
    
    def single_trade_cost(self, notional: float = 1.0) -> float:
        """Cost for trading one side of the spread."""
        return notional * (self.cost_per_trade + self.half_spread + self.slippage)
    
    def pair_trade_cost(self, notional: float = 1.0) -> float:
        """Cost for trading both legs of the spread (entry OR exit)."""
        return 2.0 * self.single_trade_cost(notional)
    
    def round_trip_cost(self, notional: float = 1.0) -> float:
        """Full round-trip cost (entry + exit for both legs)."""
        return 2.0 * self.pair_trade_cost(notional)


# Convenience constructors
def zero_costs() -> TransactionCosts:
    """No transaction costs (for comparison)."""
    return TransactionCosts(0.0, 0.0, 0.0)


def paper_costs() -> TransactionCosts:
    """Default costs from Zhang (2021): 20bp per asset."""
    return TransactionCosts(cost_per_trade=0.002)


def retail_costs() -> TransactionCosts:
    """Higher costs for retail trading (50bp per asset)."""
    return TransactionCosts(cost_per_trade=0.005)


def institutional_costs() -> TransactionCosts:
    """Lower costs for institutional trading (5bp per asset)."""
    return TransactionCosts(cost_per_trade=0.0005)
