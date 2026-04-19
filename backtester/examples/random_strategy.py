"""
Random Strategy - Makes random trades as a null model.

Useful for establishing a baseline: any real strategy should outperform random.
"""

import random

from backtester.strategy import BaseStrategy, MarketState, Order, Side, Token


class RandomStrategy(BaseStrategy):
    """
    Randomly buy YES or NO on active markets with a configurable probability.
    """

    def __init__(self, trade_prob: float = 0.02, size: float = 10.0, seed: int = 42):
        self.trade_prob = trade_prob
        self.size = size
        self.rng = random.Random(seed)

    def on_tick(self, state: MarketState) -> list[Order]:
        orders = []

        for slug, market in state.markets.items():
            if self.rng.random() > self.trade_prob:
                continue

            token = Token.YES if self.rng.random() > 0.5 else Token.NO
            if token == Token.YES:
                ask = market.yes_ask
            else:
                ask = market.no_ask

            if ask <= 0:
                continue

            cost = self.size * ask
            if cost > state.cash:
                continue

            orders.append(Order(
                market_slug=slug,
                token=token,
                side=Side.BUY,
                size=self.size,
                limit_price=ask,
            ))

        return orders
