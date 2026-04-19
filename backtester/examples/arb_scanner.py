"""
Arb Scanner - Complete-set arbitrage.

Exploits cases where YES + NO prices sum to less than $1 (buy both sides for
guaranteed profit at settlement) or more than $1 (sell both sides if held).

In v1 (no short selling), we can only buy the complete set when YES + NO < 1.
"""

from backtester.strategy import BaseStrategy, MarketState, Order, Side, Token


class ArbScanner(BaseStrategy):
    """
    Buy complete sets (YES + NO) when their combined cost < $1.

    Guaranteed profit = $1 - (yes_ask + no_ask) per share at settlement.
    """

    def __init__(self, min_edge: float = 0.02, size: float = 50.0):
        self.min_edge = min_edge  # minimum profit per share to trade
        self.size = size
        self._arbed: dict[str, int] = {}  # slug -> count of arb entries

    def on_tick(self, state: MarketState) -> list[Order]:
        orders = []

        for slug, market in state.markets.items():
            yes_ask = market.yes_ask
            no_ask = market.no_ask

            if yes_ask <= 0 or no_ask <= 0:
                continue

            combined_cost = yes_ask + no_ask
            edge = 1.0 - combined_cost

            if edge < self.min_edge:
                continue

            # Don't over-trade same market
            entries = self._arbed.get(slug, 0)
            if entries >= 3:
                continue

            # Cost for both sides
            total_cost = self.size * combined_cost
            if total_cost > state.cash:
                continue

            # Buy YES
            orders.append(Order(
                market_slug=slug,
                token=Token.YES,
                side=Side.BUY,
                size=self.size,
                limit_price=yes_ask,
            ))
            # Buy NO
            orders.append(Order(
                market_slug=slug,
                token=Token.NO,
                side=Side.BUY,
                size=self.size,
                limit_price=no_ask,
            ))

            self._arbed[slug] = entries + 1

        return orders
