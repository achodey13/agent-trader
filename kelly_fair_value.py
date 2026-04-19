"""
Kelly Fair Value — same BS fair-value signal as fair_value_v2, sized by the
Kelly criterion instead of ad-hoc multipliers.

For a YES buy at ask a with model probability p:
    f_YES = (p − a) / (1 − a)          # fraction of bankroll (full Kelly)
    target_dollars = κ · f_YES · equity

Symmetrically for NO (using p_NO = 1 − p, a_NO = no_ask).

Why κ < 1:
    Full Kelly assumes p is exact. Our p comes from a BS model using estimated
    volatility, so p is noisy — full Kelly compounds that estimation error
    into over-aggressive sizing. Quarter-Kelly (κ = 0.25) captures roughly
    95% of long-run growth at about 25% of the drawdown variance.

Everything upstream of the sizing block — vol estimation, chainlink-anchored
open, Binance signal, BTC-only scope — matches fair_value_v2 exactly, so that
a head-to-head backtest isolates the sizing choice.
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


# ─── shared with fair_value_v2 ─────────────────────────────────────────

VOL_WINDOW_S = 15 * 60
VOL_MIN_SAMPLES = 60
POSITION_CAP = 500.0
LIMIT_PAD = 0.02

# ─── Kelly-specific knobs ──────────────────────────────────────────────

KAPPA = 0.25                  # quarter-Kelly
MIN_SIZE = 5.0                # don't fire if Kelly wants fewer than this
MAX_PER_MARKET_FRAC = 0.10    # single-market concentration cap (risk layer)


def _is_btc(slug: str) -> bool:
    return slug.startswith("btc-") or slug.startswith("bitcoin-")


def _standard_normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


class _VolEstimator:
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


def _kelly_fraction(prob_win: float, ask: float) -> float:
    """Fraction of bankroll Kelly allocates to a binary bet at price ask.

    Returns 0 if the bet has no edge or the ask is degenerate.
    """
    if ask <= 0 or ask >= 1.0:
        return 0.0
    f = (prob_win - ask) / (1.0 - ask)
    return max(0.0, f)


class KellyFairValue(BaseStrategy):
    """BS fair-value signal, Kelly-sized positions."""

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

        equity = state.total_portfolio_value
        if equity <= 0:
            return orders

        per_market_cap_dollars = equity * MAX_PER_MARKET_FRAC
        remaining_cash = state.cash

        for slug, m in state.markets.items():
            if not _is_btc(slug):
                continue

            if slug not in self._open_chainlink:
                self._open_chainlink[slug] = state.chainlink_btc
            open_price = self._open_chainlink[slug]

            fair = _fair_prob(state.btc_mid, open_price, sigma_1s, m.time_remaining_s)
            if fair is None:
                continue

            pos = state.positions.get(slug)

            # Decide each side independently. In a well-priced market only one
            # side has positive Kelly; both positive = mispricing on both tokens
            # (rare, looks like a latent arb), and we fire both.
            for token, ask, current_shares, prob_win in (
                (
                    Token.YES,
                    m.yes_ask,
                    pos.yes_shares if pos else 0.0,
                    fair,
                ),
                (
                    Token.NO,
                    m.no_ask,
                    pos.no_shares if pos else 0.0,
                    1.0 - fair,
                ),
            ):
                f = _kelly_fraction(prob_win, ask)
                if f <= 0.0 or ask <= 0.0:
                    continue

                target_dollars = KAPPA * f * equity
                # Risk layer: cap per-market exposure
                target_dollars = min(target_dollars, per_market_cap_dollars)
                if target_dollars <= 0:
                    continue

                target_size = target_dollars / ask

                # Position cap (Rules §2)
                headroom = POSITION_CAP - current_shares
                size = min(target_size, headroom)
                if size < MIN_SIZE:
                    continue

                # Cash constraint
                cost = size * ask
                if cost > remaining_cash:
                    size = remaining_cash / ask
                    if size < MIN_SIZE:
                        continue
                    cost = size * ask

                # Limit price: use model prob as a cap, allow small walk into the book
                limit_price = min(prob_win, ask + LIMIT_PAD)

                orders.append(Order(
                    market_slug=slug,
                    token=token,
                    side=Side.BUY,
                    size=size,
                    limit_price=limit_price,
                ))
                remaining_cash -= cost

        return orders

    def on_fill(self, fill: Fill) -> None:
        pass

    def on_settlement(self, settlement: Settlement) -> None:
        pass
