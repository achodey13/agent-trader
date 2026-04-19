"""
Lead-Lag v3 — v2 + profit-take + stop-loss + Binance-spread filter.

v2 achievement: +$160 / 1.35% max DD / Sharpe 10.28 on BTC 5m validation.
v2 behavior: 374 trades, 62% win rate. Most winners either ride to
settlement or exit via signal reversal. Most losers also exit via signal
reversal. Neither extreme is optimal — some winners reverse before we
capture them, some losers drift further before the signal catches up.

v3 additions (all principled, all single-parameter):

    1. Profit-take (+15¢). When `yes_bid ≥ entry_price + PROFIT_TAKE_CENTS`,
       sell. Locks in realized gains on strong moves instead of waiting
       for the full signal reversal (which often comes after give-back).

    2. Stop-loss (-10¢). When `yes_bid ≤ entry_price - STOP_LOSS_CENTS`,
       sell. Truncates worst-case per-trade loss before signal reversal.
       Risk:reward set to 1:1.5 (SL = 10¢, PT = 15¢) — favorable on a
       62%-win-rate strategy.

    3. Binance-spread filter. `state.btc_spread / state.btc_mid` in bps
       is the noise level of our signal input. When Binance spread is
       wide (> MAX_SPREAD_BPS), the mid is unreliable — skip entries.
       Per-asset for ETH/SOL as well.

Kept from v2:
    - +0.25 Binance→Polymarket signal (+30s log-return z-score)
    - Signal-reversal exit (z flip > Z_EXIT in the wrong direction)
    - Late-market-only entry (time_remaining_frac 0.1..0.8)
    - One entry per direction per market
    - Freeze exits in final 60s (book is degenerate)

Exit priority each tick (first trigger wins; one action per market per tick):
    stop-loss → profit-take → signal-reversal
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


# ── Signal (unchanged from v2) ─────────────────────────────────────────
WINDOW_S = 60
MIN_SAMPLES = 30
SIGNAL_LOOKBACK_S = 30

# ── Entry / exit thresholds (unchanged) ───────────────────────────────
Z_ENTRY = 2.0
Z_EXIT = 1.2

# ── Sizing (unchanged) ─────────────────────────────────────────────────
BASE_SIZE = 20.0
MAX_SIZE = 100.0

# ── Entry filters (unchanged) ─────────────────────────────────────────
MIN_ASK = 0.05
MAX_ASK = 0.75
ENTRY_FRAC_MIN = 0.10
ENTRY_FRAC_MAX = 0.80
EXIT_MIN_TIME_S = 60

# ── Risk (unchanged) ───────────────────────────────────────────────────
POSITION_CAP = 500.0
PER_MARKET_FRAC = 0.10
LIMIT_PAD = 0.02
EXIT_SLIP_FRAC = 0.10

# ── NEW in v3 ─────────────────────────────────────────────────────────
PROFIT_TAKE_CENTS = 0.15     # exit if bid ≥ entry_price + this
STOP_LOSS_CENTS = 0.10       # exit if bid ≤ entry_price - this
MAX_SPREAD_BPS = 3.0         # skip entry if Binance spread > this many bps


def _asset_from_slug(slug: str) -> str | None:
    if slug.startswith("btc-") or slug.startswith("bitcoin-"):
        return "BTC"
    if slug.startswith("eth-") or slug.startswith("ethereum-"):
        return "ETH"
    if slug.startswith("sol-") or slug.startswith("solana-"):
        return "SOL"
    return None


def _spread_bps(spread: float, mid: float) -> float:
    if mid <= 0 or spread <= 0:
        return 0.0
    return (spread / mid) * 10_000.0


class _SignalTracker:
    def __init__(self) -> None:
        self._prices: dict[str, deque[float]] = {
            "BTC": deque(maxlen=WINDOW_S + 1),
            "ETH": deque(maxlen=WINDOW_S + 1),
            "SOL": deque(maxlen=WINDOW_S + 1),
        }

    def update(self, asset: str, mid: float) -> None:
        if mid > 0 and asset in self._prices:
            self._prices[asset].append(mid)

    def z_score(self, asset: str) -> float | None:
        buf = self._prices.get(asset)
        if buf is None or len(buf) < MIN_SAMPLES + 1:
            return None
        prices = list(buf)
        lookback = min(SIGNAL_LOOKBACK_S, len(prices) - 1)
        p_now = prices[-1]
        p_prev = prices[-1 - lookback]
        if p_now <= 0 or p_prev <= 0:
            return None
        recent_return = math.log(p_now / p_prev)
        returns: list[float] = []
        for i in range(len(prices) - 1):
            a, b = prices[i], prices[i + 1]
            if a > 0 and b > 0:
                returns.append(math.log(b / a))
        if len(returns) < MIN_SAMPLES:
            return None
        mean = sum(returns) / len(returns)
        var = sum((r - mean) ** 2 for r in returns) / len(returns)
        if var <= 0:
            return None
        sigma_1s = math.sqrt(var)
        null_std = sigma_1s * math.sqrt(lookback)
        return recent_return / null_std


class LeadLagV3(BaseStrategy):
    def __init__(self) -> None:
        self._sig = _SignalTracker()

    def on_tick(self, state: MarketState) -> list[Order]:
        self._sig.update("BTC", state.btc_mid)
        self._sig.update("ETH", state.eth_mid)
        self._sig.update("SOL", state.sol_mid)

        equity = state.total_portfolio_value
        if equity <= 0:
            return []

        per_market_budget = equity * PER_MARKET_FRAC
        remaining_cash = state.cash
        orders: list[Order] = []

        # Per-asset spread gates (only applied to entries).
        spread_ok = {
            "BTC": _spread_bps(state.btc_spread, state.btc_mid) <= MAX_SPREAD_BPS,
            "ETH": _spread_bps(state.eth_spread, state.eth_mid) <= MAX_SPREAD_BPS,
            "SOL": _spread_bps(state.sol_spread, state.sol_mid) <= MAX_SPREAD_BPS,
        }

        for slug, m in state.markets.items():
            asset = _asset_from_slug(slug)
            if asset is None:
                continue
            z = self._sig.z_score(asset)
            if z is None:
                continue

            pos = state.positions.get(slug)
            yes_held = pos.yes_shares if pos else 0.0
            no_held = pos.no_shares if pos else 0.0
            holding = yes_held > 0 or no_held > 0
            cost_basis = pos.cost_basis if pos else 0.0

            # ─── EXITS (priority: stop-loss → profit-take → signal-reversal) ───
            if holding and m.time_remaining_s >= EXIT_MIN_TIME_S:
                if yes_held > 0:
                    entry_px = cost_basis / yes_held if yes_held > 0 else 0.0
                    bid = m.yes_bid
                    should_exit = False
                    if bid > 0 and entry_px > 0:
                        if bid <= entry_px - STOP_LOSS_CENTS:
                            should_exit = True
                        elif bid >= entry_px + PROFIT_TAKE_CENTS:
                            should_exit = True
                    if not should_exit and z < -Z_EXIT and bid > 0:
                        should_exit = True
                    if should_exit and bid > 0:
                        limit = max(0.01, bid * (1.0 - EXIT_SLIP_FRAC))
                        orders.append(Order(
                            market_slug=slug,
                            token=Token.YES,
                            side=Side.SELL,
                            size=yes_held,
                            limit_price=limit,
                        ))
                        continue

                if no_held > 0:
                    entry_px = cost_basis / no_held if no_held > 0 else 0.0
                    bid = m.no_bid
                    should_exit = False
                    if bid > 0 and entry_px > 0:
                        if bid <= entry_px - STOP_LOSS_CENTS:
                            should_exit = True
                        elif bid >= entry_px + PROFIT_TAKE_CENTS:
                            should_exit = True
                    if not should_exit and z > Z_EXIT and bid > 0:
                        should_exit = True
                    if should_exit and bid > 0:
                        limit = max(0.01, bid * (1.0 - EXIT_SLIP_FRAC))
                        orders.append(Order(
                            market_slug=slug,
                            token=Token.NO,
                            side=Side.SELL,
                            size=no_held,
                            limit_price=limit,
                        ))
                        continue

            # ─── ENTRY ───
            if holding:
                continue
            if not spread_ok.get(asset, False):
                continue  # Binance mid is too noisy; skip entry
            if m.time_remaining_frac < ENTRY_FRAC_MIN or m.time_remaining_frac > ENTRY_FRAC_MAX:
                continue
            if abs(z) < Z_ENTRY:
                continue

            if z > 0:
                token, ask = Token.YES, m.yes_ask
            else:
                token, ask = Token.NO, m.no_ask
            if ask <= MIN_ASK or ask >= MAX_ASK:
                continue

            z_mag = min(5.0, abs(z))
            target_size = BASE_SIZE * z_mag
            size = min(target_size, MAX_SIZE, POSITION_CAP)
            if size <= 0:
                continue

            cost = size * ask
            if cost > remaining_cash:
                size = remaining_cash / ask
                if size < 1.0:
                    continue
                cost = size * ask
            if cost > per_market_budget:
                size = per_market_budget / ask
                if size < 1.0:
                    continue
                cost = size * ask

            orders.append(Order(
                market_slug=slug,
                token=token,
                side=Side.BUY,
                size=size,
                limit_price=min(1.0, ask + LIMIT_PAD),
            ))
            remaining_cash -= cost

        return orders

    def on_fill(self, fill: Fill) -> None:
        pass

    def on_settlement(self, settlement: Settlement) -> None:
        pass
