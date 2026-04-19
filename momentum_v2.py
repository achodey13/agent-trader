"""
Momentum v2 — re-implementation of the original "4 rising bars → buy YES"
idea, rebuilt on top of everything we've learned since v1 was abandoned.

Original momentum_v1 design: sample btc_mid at t = 0, 60, 120, 180, 240s.
If all four 1-minute bars rose by ≥ 3 bps, buy YES at minute 4 and hold to
settlement. Never backtested — user discarded before running.

v2 rebuilds that spec with post-v1 lessons baked in:

  1. Never trust yes_price (mid). Data exploration showed yes_price is a
     degenerate mid when one book side empties near expiry. v2 uses only
     yes_bid / yes_ask / no_bid / no_ask explicitly.

  2. Tight 50-75¢ price band. Outside this band, Polymarket has already
     priced in the momentum and Kelly sizing produces negative edge.

  3. Kelly-criterion sizing. A strong 4-bar pattern maps to a baseline
     P(correct) ≈ 0.75. Quarter-Kelly f = (0.75 − ask) / (1 − ask).

  4. Profit-take in the fifth minute. User's spec: "sell on 5th minute if
     profitable." Implementation: if yes_bid ≥ entry_price + 8¢, sell at
     bid; otherwise hold to settlement. This is a free option — take it
     when offered, don't demand it.

  5. Symmetric: if 4 bars all FALL by ≥ MIN_BAR_BPS, buy NO. Inversion of
     the same logic.

Scope: BTC 5m markets only. 5m is where 4-bar-minute momentum makes sense.
BTC is the only asset where we've measured the Binance→Polymarket lag.
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


# ── Pattern detection ─────────────────────────────────────────────────
MIN_BAR_BPS = 1.0               # each 60-sec bar must move by ≥ this
CHECKPOINTS = (0, 60, 120, 180, 240)
EARLY_JOIN_TOLERANCE_S = 15     # must see market within this many sec of start

# ── Probability / sizing ──────────────────────────────────────────────
# With looser MIN_BAR_BPS = 1, pattern is weaker → lower P estimate.
P_AFTER_PATTERN = 0.70          # baseline P(correct) after qualifying pattern
KELLY_FRACTION = 0.25           # quarter-Kelly (p is an estimate, not precise)
P_MIN_EDGE = 0.02
MIN_SHARES = 1.0

# ── Price band ─────────────────────────────────────────────────────────
# Widened to 0.90: 4-bar pattern often implies yes_ask in 0.80-0.90 by t=240s.
# Kelly self-limits when P - ask is small.
MIN_ASK = 0.50
MAX_ASK = 0.90

# ── Profit-take ────────────────────────────────────────────────────────
MIN_PROFIT_CENTS = 0.08         # sell if bid ≥ entry_px + this

# ── Risk ───────────────────────────────────────────────────────────────
POSITION_CAP = 500.0
PER_MARKET_FRAC = 0.10
LIMIT_PAD = 0.02
EXIT_SLIP_FRAC = 0.10


def _is_btc(slug: str) -> bool:
    return slug.startswith("btc-") or slug.startswith("bitcoin-")


def _kelly_fraction(p: float, ask: float) -> float:
    if ask <= 0.0 or ask >= 1.0:
        return 0.0
    return max(0.0, (p - ask) / (1.0 - ask))


class MomentumV2(BaseStrategy):
    def __init__(self) -> None:
        # Per-market checkpoint samples — dict[slug, dict[bucket_0..4, price]]
        self._samples: dict[str, dict[int, float]] = {}
        # Markets we've already decided on (entered or passed) — never re-evaluate
        self._decided: set[str] = set()
        # Markets we can't track because we saw them too late
        self._skipped: set[str] = set()

    def on_tick(self, state: MarketState) -> list[Order]:
        if state.btc_mid <= 0:
            return []

        equity = state.total_portfolio_value
        if equity <= 0:
            return []

        per_market_budget = equity * PER_MARKET_FRAC
        remaining_cash = state.cash
        orders: list[Order] = []

        for slug, m in state.markets.items():
            if m.interval != "5m":
                continue
            if not _is_btc(slug):
                continue
            if slug in self._skipped:
                continue

            total_s = m.end_ts - m.start_ts
            elapsed = total_s - m.time_remaining_s

            # First sight: must be near start
            if slug not in self._samples:
                if elapsed > EARLY_JOIN_TOLERANCE_S:
                    self._skipped.add(slug)
                    continue
                self._samples[slug] = {}

            samples = self._samples[slug]

            # Record the current checkpoint bucket if not already sampled
            bucket = min(4, int(elapsed // 60))
            if bucket not in samples:
                samples[bucket] = state.btc_mid

            pos = state.positions.get(slug)
            yes_held = pos.yes_shares if pos else 0.0
            no_held = pos.no_shares if pos else 0.0
            holding = yes_held > 0 or no_held > 0
            cost_basis = pos.cost_basis if pos else 0.0

            # ─── PROFIT-TAKE EXIT (minute 5) ───
            if holding:
                if yes_held > 0:
                    entry_px = cost_basis / yes_held if yes_held > 0 else 0.0
                    if entry_px > 0 and m.yes_bid >= entry_px + MIN_PROFIT_CENTS and m.yes_bid > 0:
                        limit = max(0.01, m.yes_bid * (1.0 - EXIT_SLIP_FRAC))
                        orders.append(Order(
                            market_slug=slug, token=Token.YES, side=Side.SELL,
                            size=yes_held, limit_price=limit,
                        ))
                        continue
                if no_held > 0:
                    entry_px = cost_basis / no_held if no_held > 0 else 0.0
                    if entry_px > 0 and m.no_bid >= entry_px + MIN_PROFIT_CENTS and m.no_bid > 0:
                        limit = max(0.01, m.no_bid * (1.0 - EXIT_SLIP_FRAC))
                        orders.append(Order(
                            market_slug=slug, token=Token.NO, side=Side.SELL,
                            size=no_held, limit_price=limit,
                        ))
                        continue

            # ─── ENTRY: check 4-bar pattern at minute 4 ───
            if slug in self._decided:
                continue
            if len(samples) < 5:
                continue  # still accumulating

            prices = [samples[i] for i in range(5)]
            if any(p <= 0 for p in prices):
                self._skipped.add(slug)
                continue

            bars_up = sum(
                (prices[i + 1] - prices[i]) / prices[i] * 10_000 >= MIN_BAR_BPS
                for i in range(4)
            )
            bars_down = sum(
                (prices[i] - prices[i + 1]) / prices[i] * 10_000 >= MIN_BAR_BPS
                for i in range(4)
            )

            if bars_up == 4:
                token, ask = Token.YES, m.yes_ask
            elif bars_down == 4:
                token, ask = Token.NO, m.no_ask
            else:
                self._decided.add(slug)
                continue

            if ask <= MIN_ASK or ask >= MAX_ASK:
                self._decided.add(slug)
                continue

            f_full = _kelly_fraction(P_AFTER_PATTERN, ask)
            if f_full <= P_MIN_EDGE:
                self._decided.add(slug)
                continue

            target_dollars = KELLY_FRACTION * f_full * equity
            target_dollars = min(target_dollars, per_market_budget)
            if target_dollars <= 0:
                self._decided.add(slug)
                continue

            size = target_dollars / ask
            size = min(size, POSITION_CAP)
            if size < MIN_SHARES:
                self._decided.add(slug)
                continue

            cost = size * ask
            if cost > remaining_cash:
                size = remaining_cash / ask
                if size < MIN_SHARES:
                    self._decided.add(slug)
                    continue
                cost = size * ask

            orders.append(Order(
                market_slug=slug, token=token, side=Side.BUY,
                size=size, limit_price=min(1.0, ask + LIMIT_PAD),
            ))
            self._decided.add(slug)
            remaining_cash -= cost

        return orders

    def on_fill(self, fill: Fill) -> None:
        pass

    def on_settlement(self, settlement: Settlement) -> None:
        pass
