"""
Ensemble v1 — four sub-agents with equal weights, hand-reconciled.

Sub-agents (priority order = Rules §6 resolution order):
    0. ArbAgent        — complete-set arb, all assets/intervals
    1. FairValueAgent  — Black-Scholes vs. chainlink_btc, BTC only
    2. ImbalanceAgent  — YES book bid/ask imbalance near expiry, all assets
    3. BuyHoldAgent    — buy YES on every new market, all assets

The reconciliation layer:
    - processes agents in priority order so low-variance edges (arb) get
      liquidity first (Rules §6 depletes in list order)
    - clips per-market YES/NO sizes to the 500-share cap (§2)
    - skips markets with an empty relevant book (§5 proxy)
    - enforces a running cash budget
    - caps any single market at MAX_PER_MARKET_CASH_FRAC of equity
"""

import math

from backtester.strategy import (
    BaseStrategy,
    Fill,
    MarketState,
    Order,
    Settlement,
    Side,
    Token,
)


POSITION_CAP = 500.0
MAX_PER_MARKET_CASH_FRAC = 0.05


# ───────────────────────────── helpers ─────────────────────────────


def _standard_normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _fair_prob(
    btc_current: float,
    btc_open: float,
    vol_15m: float,
    time_remaining_frac: float,
) -> float:
    if btc_open <= 0 or btc_current <= 0:
        return 0.5
    tau = max(time_remaining_frac, 0.001)
    sigma = vol_15m if vol_15m > 0 else 0.005
    sigma_sqrt_tau = sigma * math.sqrt(tau)
    if sigma_sqrt_tau < 1e-8:
        return 0.99 if btc_current >= btc_open else 0.01
    log_m = math.log(btc_current / btc_open)
    d2 = log_m / sigma_sqrt_tau - sigma_sqrt_tau / 2.0
    return max(0.01, min(0.99, _standard_normal_cdf(d2)))


def _is_btc(slug: str) -> bool:
    return slug.startswith("btc-") or slug.startswith("bitcoin-")


# ───────────────────────────── sub-agents ─────────────────────────────


class _ArbAgent:
    """yes_ask + no_ask < 1 (no fees per Rules §7) → buy both, guaranteed $1 at settle."""

    priority = 0
    MIN_EDGE = 0.02
    SIZE = 50.0
    MAX_ENTRIES = 3

    def __init__(self) -> None:
        self._entries: dict[str, int] = {}

    def propose(self, state: MarketState, budget: float) -> list[Order]:
        orders: list[Order] = []
        remaining = budget
        for slug, m in state.markets.items():
            if m.yes_ask <= 0 or m.no_ask <= 0:
                continue
            combined = m.yes_ask + m.no_ask
            edge = 1.0 - combined
            if edge < self.MIN_EDGE:
                continue
            if self._entries.get(slug, 0) >= self.MAX_ENTRIES:
                continue
            cost = self.SIZE * combined
            if cost > remaining:
                continue
            orders.append(Order(slug, Token.YES, Side.BUY, self.SIZE, m.yes_ask))
            orders.append(Order(slug, Token.NO, Side.BUY, self.SIZE, m.no_ask))
            self._entries[slug] = self._entries.get(slug, 0) + 1
            remaining -= cost
        return orders


class _FairValueAgent:
    """Black-Scholes P(YES) vs market; trade when |fair − market| > threshold. BTC only."""

    priority = 1
    VOL_15M = 0.005
    THRESHOLD = 0.08
    SIZE = 30.0

    def __init__(self) -> None:
        self._open_price: dict[str, float] = {}

    def propose(self, state: MarketState, budget: float) -> list[Order]:
        orders: list[Order] = []
        if state.chainlink_btc <= 0:
            return orders
        remaining = budget
        for slug, m in state.markets.items():
            if not _is_btc(slug):
                continue
            if slug not in self._open_price:
                self._open_price[slug] = state.chainlink_btc
            fair = _fair_prob(
                state.chainlink_btc,
                self._open_price[slug],
                self.VOL_15M,
                m.time_remaining_frac,
            )
            mid = m.yes_price
            if mid <= 0:
                continue
            delta = fair - mid
            if delta > self.THRESHOLD and m.yes_ask > 0:
                cost = self.SIZE * m.yes_ask
                if cost <= remaining:
                    orders.append(Order(
                        slug, Token.YES, Side.BUY, self.SIZE,
                        min(fair, m.yes_ask + 0.02),
                    ))
                    remaining -= cost
            elif delta < -self.THRESHOLD and m.no_ask > 0:
                cost = self.SIZE * m.no_ask
                if cost <= remaining:
                    orders.append(Order(
                        slug, Token.NO, Side.BUY, self.SIZE,
                        min(1.0 - fair, m.no_ask + 0.02),
                    ))
                    remaining -= cost
        return orders


class _ImbalanceAgent:
    """YES book imbalance near expiry → scalp direction."""

    priority = 2
    THRESHOLD = 0.5
    SIZE = 10.0
    MAX_TIME_S = 60
    COOLDOWN_S = 5

    def __init__(self) -> None:
        self._last_fire: dict[str, int] = {}

    def propose(self, state: MarketState, budget: float) -> list[Order]:
        orders: list[Order] = []
        remaining = budget
        for slug, m in state.markets.items():
            if m.time_remaining_s > self.MAX_TIME_S:
                continue
            if state.timestamp - self._last_fire.get(slug, 0) < self.COOLDOWN_S:
                continue
            bid_sz = m.yes_book.total_bid_size
            ask_sz = m.yes_book.total_ask_size
            if bid_sz + ask_sz <= 0:
                continue
            imb = (bid_sz - ask_sz) / (bid_sz + ask_sz)
            if imb > self.THRESHOLD and m.yes_ask > 0:
                cost = self.SIZE * m.yes_ask
                if cost <= remaining:
                    orders.append(Order(slug, Token.YES, Side.BUY, self.SIZE, m.yes_ask))
                    remaining -= cost
                    self._last_fire[slug] = state.timestamp
            elif imb < -self.THRESHOLD and m.no_ask > 0:
                cost = self.SIZE * m.no_ask
                if cost <= remaining:
                    orders.append(Order(slug, Token.NO, Side.BUY, self.SIZE, m.no_ask))
                    remaining -= cost
                    self._last_fire[slug] = state.timestamp
        return orders


class _BuyHoldAgent:
    """Directional null model: buy 50 YES on first sight of each market."""

    priority = 3
    SIZE = 50.0

    def __init__(self) -> None:
        self._bought: set[str] = set()

    def propose(self, state: MarketState, budget: float) -> list[Order]:
        orders: list[Order] = []
        remaining = budget
        for slug, m in state.markets.items():
            if slug in self._bought:
                continue
            if m.yes_ask <= 0:
                continue
            cost = self.SIZE * m.yes_ask
            if cost > remaining:
                continue
            orders.append(Order(slug, Token.YES, Side.BUY, self.SIZE, m.yes_ask))
            self._bought.add(slug)
            remaining -= cost
        return orders


# ───────────────────────────── ensemble ─────────────────────────────


class EnsembleV1(BaseStrategy):
    """Equal-weight ensemble of four sub-agents with reconciliation."""

    def __init__(self) -> None:
        self.agents = [
            _ArbAgent(),
            _FairValueAgent(),
            _ImbalanceAgent(),
            _BuyHoldAgent(),
        ]
        self.weights = [0.25, 0.25, 0.25, 0.25]

    def on_tick(self, state: MarketState) -> list[Order]:
        # Priority-ordered proposals; each agent sees its own cash slice.
        ordered = sorted(
            zip(self.agents, self.weights), key=lambda p: p[0].priority
        )
        proposals: list[Order] = []
        for agent, w in ordered:
            proposals.extend(agent.propose(state, state.cash * w))
        return self._reconcile(proposals, state)

    def _reconcile(
        self, proposals: list[Order], state: MarketState
    ) -> list[Order]:
        out: list[Order] = []
        # Projected YES/NO holdings after earlier-in-list orders fill.
        proj_yes = {slug: p.yes_shares for slug, p in state.positions.items()}
        proj_no = {slug: p.no_shares for slug, p in state.positions.items()}
        remaining_cash = state.cash
        max_per_market = state.total_portfolio_value * MAX_PER_MARKET_CASH_FRAC

        for o in proposals:
            m = state.markets.get(o.market_slug)
            if m is None:
                continue

            # Empty-book guard (§5 proxy — no recent liquidity on the side we'd trade).
            book = m.yes_book if o.token == Token.YES else m.no_book
            if o.side == Side.BUY and not book.asks:
                continue
            if o.side == Side.SELL and not book.bids:
                continue

            # Position cap (§2).
            proj = proj_yes if o.token == Token.YES else proj_no
            current = proj.get(o.market_slug, 0.0)
            if o.side == Side.BUY:
                headroom = POSITION_CAP - current
                if headroom <= 0:
                    continue
                size = min(o.size, headroom)
            else:
                if current <= 0:
                    continue
                size = min(o.size, current)

            if size <= 0:
                continue

            # Cash + per-market concentration cap (BUY only — SELL frees cash).
            if o.side == Side.BUY:
                price = o.limit_price if o.limit_price else book.best_ask
                if price <= 0:
                    continue
                cost = size * price
                if cost > remaining_cash:
                    size = remaining_cash / price
                    if size <= 0:
                        continue
                    cost = size * price
                if cost > max_per_market:
                    size = max_per_market / price
                    if size <= 0:
                        continue
                    cost = size * price
                remaining_cash -= cost

            out.append(Order(
                market_slug=o.market_slug,
                token=o.token,
                side=o.side,
                size=float(size),
                limit_price=o.limit_price,
            ))

            delta = size if o.side == Side.BUY else -size
            proj[o.market_slug] = current + delta

        return out

    def on_fill(self, fill: Fill) -> None:
        pass

    def on_settlement(self, settlement: Settlement) -> None:
        pass
