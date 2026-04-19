"""
Fair Value v2 — Black-Scholes binary option pricing with dynamic volatility.

Three upgrades over examples/fair_value.py:

  1. Rolling realized volatility estimated from 1-second Binance btc_mid samples
     over a trailing 15-minute window. Replaces the hardcoded 0.5% constant.

  2. Signal uses state.btc_mid (Binance spot, continuous) instead of
     state.chainlink_btc (oracle pushes are discrete and lumpy). The open
     price still locks to chainlink since that is what settles (Rules §7).

  3. Size-by-confidence: per-fire size scales with edge magnitude AND with
     (1 − time_remaining_frac), because signal reliability improves as τ → 0.

Scope: BTC markets only (5m / 15m / hourly). MarketState does not expose ETH
or SOL price feeds, so the model cannot be evaluated on those markets.

Sigma convention: we compute per-second log-return std from the trailing
window, then scale by √time_remaining_s inside the formula. This is the
clean dimensional form — sigma × √τ where τ is in seconds.
"""

import math
from collections import deque

from backtester.strategy import (
    BaseStrategy,
    Fill,
    MarketState,
    Order,
    Settlement,
    Side,
    Token,
)


VOL_WINDOW_S = 15 * 60       # trailing 1-sec samples for realized vol
VOL_MIN_SAMPLES = 60         # require at least this many returns before trading
MIN_EDGE = 0.05              # |fair − market| threshold to fire
BASE_SIZE = 20.0             # baseline shares per fire
MAX_EDGE_MULT = 3.0          # cap on edge-based size multiplier
POSITION_CAP = 500.0         # Rules §2
LIMIT_PAD = 0.02             # willing to pay up to (best_ask + pad) if book is thin


def _is_btc(slug: str) -> bool:
    return slug.startswith("btc-") or slug.startswith("bitcoin-")


def _standard_normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


class _VolEstimator:
    """Rolling per-second log-return std from a trailing btc_mid window."""

    def __init__(self) -> None:
        self._prices: deque[float] = deque(maxlen=VOL_WINDOW_S + 1)

    def update(self, price: float) -> None:
        if price > 0:
            self._prices.append(price)

    def sigma_per_second(self) -> float | None:
        if len(self._prices) < VOL_MIN_SAMPLES + 1:
            return None
        prev = None
        s = 0.0
        s2 = 0.0
        n = 0
        for p in self._prices:
            if prev is not None and prev > 0 and p > 0:
                r = math.log(p / prev)
                s += r
                s2 += r * r
                n += 1
            prev = p
        if n < VOL_MIN_SAMPLES:
            return None
        mean = s / n
        var = s2 / n - mean * mean
        if var <= 0:
            return None
        return math.sqrt(var)


def _fair_prob(
    current: float,
    open_price: float,
    sigma_per_second: float,
    time_remaining_s: float,
) -> float | None:
    if current <= 0 or open_price <= 0 or sigma_per_second <= 0:
        return None
    tau = max(time_remaining_s, 1.0)
    sigma_sqrt_tau = sigma_per_second * math.sqrt(tau)
    if sigma_sqrt_tau < 1e-8:
        return 0.99 if current >= open_price else 0.01
    log_m = math.log(current / open_price)
    d2 = log_m / sigma_sqrt_tau - sigma_sqrt_tau / 2.0
    return max(0.01, min(0.99, _standard_normal_cdf(d2)))


class FairValueV2_5m(BaseStrategy):
    """BS-based fair-value trader, BTC 5m only. Internal interval filter so
    the submission behaves as 5m-only even when scored unfiltered."""

    def __init__(self) -> None:
        self._vol = _VolEstimator()
        self._open_chainlink: dict[str, float] = {}

    def on_tick(self, state: MarketState) -> list[Order]:
        self._vol.update(state.btc_mid)

        orders: list[Order] = []

        if state.btc_mid <= 0 or state.chainlink_btc <= 0:
            return orders

        sigma_1s = self._vol.sigma_per_second()
        if sigma_1s is None:
            return orders

        remaining_cash = state.cash

        for slug, m in state.markets.items():
            if not _is_btc(slug):
                continue
            # Interval filter: 5m only. Val 15m/hourly drag total P&L by ~$1k.
            if m.interval != "5m":
                continue

            if slug not in self._open_chainlink:
                self._open_chainlink[slug] = state.chainlink_btc
            open_price = self._open_chainlink[slug]

            fair = _fair_prob(state.btc_mid, open_price, sigma_1s, m.time_remaining_s)
            if fair is None:
                continue

            mid = m.yes_price
            if mid <= 0:
                continue

            delta = fair - mid
            abs_edge = abs(delta)
            if abs_edge < MIN_EDGE:
                continue

            edge_mult = min(MAX_EDGE_MULT, abs_edge / MIN_EDGE)
            time_mult = 1.0 + (1.0 - m.time_remaining_frac)  # 1x at start, 2x at end
            target_size = BASE_SIZE * edge_mult * time_mult

            pos = state.positions.get(slug)

            if delta > 0 and m.yes_ask > 0:
                current_shares = pos.yes_shares if pos else 0.0
                size = min(target_size, POSITION_CAP - current_shares)
                if size <= 0:
                    continue
                cost = size * m.yes_ask
                if cost > remaining_cash:
                    size = remaining_cash / m.yes_ask
                    if size <= 0:
                        continue
                    cost = size * m.yes_ask
                orders.append(Order(
                    market_slug=slug,
                    token=Token.YES,
                    side=Side.BUY,
                    size=size,
                    limit_price=min(fair, m.yes_ask + LIMIT_PAD),
                ))
                remaining_cash -= cost

            elif delta < 0 and m.no_ask > 0:
                current_shares = pos.no_shares if pos else 0.0
                size = min(target_size, POSITION_CAP - current_shares)
                if size <= 0:
                    continue
                cost = size * m.no_ask
                if cost > remaining_cash:
                    size = remaining_cash / m.no_ask
                    if size <= 0:
                        continue
                    cost = size * m.no_ask
                orders.append(Order(
                    market_slug=slug,
                    token=Token.NO,
                    side=Side.BUY,
                    size=size,
                    limit_price=min(1.0 - fair, m.no_ask + LIMIT_PAD),
                ))
                remaining_cash -= cost

        return orders

    def on_fill(self, fill: Fill) -> None:
        pass

    def on_settlement(self, settlement: Settlement) -> None:
        pass
