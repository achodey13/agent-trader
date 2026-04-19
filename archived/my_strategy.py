"""
Late-Window Drift - an intuitive baseline strategy.

Intuition
---------
A Polymarket binary "BTC up-or-down over the next 5 minutes" market must
eventually pay $1 to whichever side Chainlink declares the winner. As the
market nears expiry, the outcome is *almost* deterministic:

    btc_now > btc_open  =>  YES will win
    btc_now < btc_open  =>  NO  will win

An efficient market should price the winning side near $1 in the final
minute. But the Polymarket book is thin and laggy; we often see the losing
side still trading at 0.30-0.50 with 30-60 seconds left. That is free edge
IF our feed is clean.

Rule:
  - Wait until `time_remaining_frac <= LATE_WINDOW` (default 0.40, i.e. the
    last 40% of the market's life).
  - Look at Chainlink oracle drift: `btc_now - btc_open`. A small deadband
    filters out microscopic moves where the outcome is genuinely uncertain.
  - If drift > +DEADBAND, the fair value of YES is ~1.0; buy YES if its ask
    price leaves us meaningful room (< MAX_PAY).
  - Symmetric on the NO side.

This deliberately skips:
  - Volatility / Black-Scholes. The whole point is that with <40% time left
    and a clear drift, sigma*sqrt(tau) is tiny and the probability collapses
    to 0 or 1. No model needed.
  - Order-book imbalance, momentum, spread capture, etc.

We focus on DATA ENGINEERING:
  - Per-market `btc_open` captured from the first tick the market is seen.
  - Chainlink staleness guard: if the oracle feed has not updated recently,
    we don't trust `btc_now` and abstain.
  - Book sanity: skip markets with no ask quote or a nonsense ask.
  - Position limit and cash guards applied before every order.
  - Single entry per (market, side) - no stacking into the same idea.
"""

from backtester.strategy import (
    BaseStrategy,
    Fill,
    MarketState,
    Order,
    Settlement,
    Side,
    Token,
)


class LateWindowDrift(BaseStrategy):
    # ── Knobs (all chosen for BTC 5m markets) ────────────────────────────────
    LATE_WINDOW = 0.40   # trade only in the last 40% of a market's life
    DEADBAND_USD = 10.0  # ignore drifts smaller than $10 on BTC (noise)
    MAX_PAY = 0.85       # don't pay more than 85c even for the "right" side
    MIN_PAY = 0.05       # ignore tokens trading below 5c (can't get fills)
    SIZE = 20.0          # shares per entry
    MAX_STALE_S = 30     # Chainlink freshness guard

    def __init__(self) -> None:
        # Data engineering state kept across ticks:
        self._btc_open: dict[str, float] = {}   # slug -> Chainlink BTC at first tick
        self._last_cl_update_ts: int = 0        # last tick Chainlink advanced
        self._last_cl_price: float = 0.0        # last observed Chainlink
        self._entered: set[tuple[str, Token]] = set()  # (slug, side) we've already entered

    # ── Helpers ──────────────────────────────────────────────────────────────
    @staticmethod
    def _is_btc_slug(slug: str) -> bool:
        s = slug.lower()
        return s.startswith("btc-") or s.startswith("bitcoin-")

    def _chainlink_is_fresh(self, ts: int) -> bool:
        return (ts - self._last_cl_update_ts) <= self.MAX_STALE_S

    # ── Main callback ────────────────────────────────────────────────────────
    def on_tick(self, state: MarketState) -> list[Order]:
        ts = state.timestamp
        cl = state.chainlink_btc

        # 1. Update Chainlink freshness tracker. The data loader forward-fills
        #    the oracle, so `cl` is always non-zero once we've seen it once,
        #    but we only want to ACT when it actually changed recently.
        if cl > 0 and cl != self._last_cl_price:
            self._last_cl_price = cl
            self._last_cl_update_ts = ts

        if cl <= 0 or not self._chainlink_is_fresh(ts):
            return []

        orders: list[Order] = []

        for slug, market in state.markets.items():
            # 2. Scope: BTC 5m markets only. Keeps the strategy honest and
            #    avoids cross-asset noise (SOL/ETH would use different opens).
            if not self._is_btc_slug(slug):
                continue
            if market.interval != "5m":
                continue

            # 3. Capture btc_open on the FIRST tick we ever see this market.
            #    This is the real data-engineering gotcha: markets appear
            #    mid-second, and the backtester hands them to us the tick
            #    they become ACTIVE. That's our open.
            if slug not in self._btc_open:
                self._btc_open[slug] = cl
                # Don't trade on the very first tick - no drift yet.
                continue

            btc_open = self._btc_open[slug]
            drift = cl - btc_open

            # 4. Late-window gate.
            if market.time_remaining_frac > self.LATE_WINDOW:
                continue

            # 5. Deadband - require a meaningful directional drift.
            if abs(drift) < self.DEADBAND_USD:
                continue

            if drift > 0:
                # BTC is above open → YES is the favored side.
                key = (slug, Token.YES)
                if key in self._entered:
                    continue
                ask = market.yes_ask
                if ask <= self.MIN_PAY or ask >= self.MAX_PAY:
                    continue
                cost = self.SIZE * ask
                if state.cash < cost:
                    continue
                # Book-depth sanity: don't trade if there's no ask size at all.
                if market.yes_book.total_ask_size < self.SIZE:
                    continue
                orders.append(Order(
                    market_slug=slug,
                    token=Token.YES,
                    side=Side.BUY,
                    size=self.SIZE,
                    limit_price=ask,
                ))
                self._entered.add(key)

            else:
                # BTC is below open → NO is the favored side.
                key = (slug, Token.NO)
                if key in self._entered:
                    continue
                ask = market.no_ask
                if ask <= self.MIN_PAY or ask >= self.MAX_PAY:
                    continue
                cost = self.SIZE * ask
                if state.cash < cost:
                    continue
                if market.no_book.total_ask_size < self.SIZE:
                    continue
                orders.append(Order(
                    market_slug=slug,
                    token=Token.NO,
                    side=Side.BUY,
                    size=self.SIZE,
                    limit_price=ask,
                ))
                self._entered.add(key)

        return orders

    def on_settlement(self, settlement: Settlement) -> None:
        # Free memory for markets that have resolved - keeps state bounded
        # on long backtests.
        self._btc_open.pop(settlement.market_slug, None)
