"""
Fair Value - Black-Scholes model adapted from appendix/btc_fair_value_estimator.py.

Computes P(YES) = N(d2) using realized volatility and trades when the market
price diverges from the model price by more than a threshold.

Supports BTC, ETH, and SOL markets. Each market is routed to its matching
Chainlink oracle via `_asset_from_slug` and uses a per-asset 15-minute
volatility estimate, so the d2 inputs are always self-consistent.
"""

import math

from backtester.data_loader import _asset_from_slug
from backtester.strategy import BaseStrategy, Fill, MarketState, Order, Side, Token


# Rough 15-minute realized volatility per asset. Values are illustrative;
# a production strategy should calibrate these from recent returns.
_DEFAULT_VOL_15M: dict[str, float] = {"BTC": 0.005, "ETH": 0.007, "SOL": 0.012}


def _standard_normal_cdf(x: float) -> float:
    """Approximate Phi(x) using the error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _compute_fair_prob(
    spot_current: float,
    spot_open: float,
    vol_15m: float,
    time_remaining_frac: float,
) -> float:
    """Compute Black-Scholes P(YES) = N(d2)."""
    if spot_open <= 0 or spot_current <= 0:
        return 0.5

    tau = max(time_remaining_frac, 0.001)
    sigma = vol_15m if vol_15m > 0 else 0.005
    sigma_sqrt_tau = sigma * math.sqrt(tau)

    if sigma_sqrt_tau < 1e-8:
        return 0.99 if spot_current >= spot_open else 0.01

    log_moneyness = math.log(spot_current / spot_open)
    d2 = log_moneyness / sigma_sqrt_tau - sigma_sqrt_tau / 2.0

    prob = _standard_normal_cdf(d2)
    return max(0.01, min(0.99, prob))


class FairValue(BaseStrategy):
    """
    Trade based on Black-Scholes fair value vs market price.

    Buys YES when market is cheap (fair > market + threshold).
    Buys NO when market is rich (fair < market - threshold).
    """

    def __init__(
        self,
        vol_15m: float | dict[str, float] | None = None,
        threshold: float = 0.08,  # Trade when |fair - market| > 8 cents.
        size: float = 30.0,
    ):
        # Accept None (use defaults), a per-asset dict, or a single float
        # applied uniformly to BTC, ETH, and SOL.
        if vol_15m is None:
            self.vol_by_asset = dict(_DEFAULT_VOL_15M)
        elif isinstance(vol_15m, dict):
            self.vol_by_asset = {**_DEFAULT_VOL_15M, **vol_15m}
        else:
            self.vol_by_asset = {a: float(vol_15m) for a in ("BTC", "ETH", "SOL")}
        self.threshold = threshold
        self.size = size
        # Map of slug -> oracle price captured at the market's first tick.
        self._spot_open: dict[str, float] = {}

    def _oracle_for(self, state: MarketState, asset: str) -> float:
        if asset == "BTC":
            return state.chainlink_btc
        if asset == "ETH":
            return state.chainlink_eth
        if asset == "SOL":
            return state.chainlink_sol
        return 0.0

    def on_tick(self, state: MarketState) -> list[Order]:
        orders = []

        for slug, market in state.markets.items():
            asset = _asset_from_slug(slug)
            spot = self._oracle_for(state, asset)
            if spot <= 0:
                continue

            # Record the oracle price the first tick we see this market.
            if slug not in self._spot_open:
                self._spot_open[slug] = spot

            spot_open = self._spot_open[slug]
            fair = _compute_fair_prob(
                spot, spot_open,
                self.vol_by_asset.get(asset, 0.005),
                market.time_remaining_frac,
            )
            market_mid = market.yes_price

            if market_mid <= 0:
                continue

            delta = fair - market_mid

            if delta > self.threshold and market.yes_ask > 0:
                # Market is cheap relative to fair value - buy YES.
                if state.cash >= self.size * market.yes_ask:
                    orders.append(Order(
                        market_slug=slug,
                        token=Token.YES,
                        side=Side.BUY,
                        size=self.size,
                        limit_price=min(fair, market.yes_ask + 0.02),
                    ))

            elif delta < -self.threshold and market.no_ask > 0:
                # Market is rich relative to fair value - buy NO.
                if state.cash >= self.size * market.no_ask:
                    orders.append(Order(
                        market_slug=slug,
                        token=Token.NO,
                        side=Side.BUY,
                        size=self.size,
                        limit_price=min(1 - fair, market.no_ask + 0.02),
                    ))

        return orders
