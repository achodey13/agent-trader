"""
Buy & Hold - Always buy YES on the first available market.

Baseline strategy: buys YES tokens once per market as soon as it becomes active.
Useful as a null benchmark to see if directional bias in BTC produces positive P&L.
"""

from backtester.strategy import BaseStrategy, MarketState, Order, Side, Token


class BuyAndHold(BaseStrategy):
    """Buy 50 YES shares on each new market, hold until settlement."""

    def __init__(self, size: float = 50.0):
        self.size = size
        self._bought: set[str] = set()

    def on_tick(self, state: MarketState) -> list[Order]:
        orders = []
        for slug, market in state.markets.items():
            if slug in self._bought:
                continue

            # Buy YES at market (take best ask)
            if market.yes_ask > 0 and state.cash >= self.size * market.yes_ask:
                orders.append(Order(
                    market_slug=slug,
                    token=Token.YES,
                    side=Side.BUY,
                    size=self.size,
                    limit_price=market.yes_ask,
                ))
                self._bought.add(slug)

        return orders
