"""
Algo Adaptive — algo03 with rolling-win-rate Kelly throttling.

Problem observed: during stagnant periods (portfolio sits at 11-12k), v4/algo03
keeps firing trades that wash out to breakeven while paying spread costs.
The strategy doesn't know it's in a bad regime.

Fix: track the last N settled markets. If recent win rate drops, throttle Kelly
down to bleed less during bad regimes. When win rate recovers, Kelly ramps back
up to capture trending regimes fully.

Throttle tiers:
    win_rate >= 0.75: Kelly = 0.15 (normal)
    win_rate in [0.60, 0.75): Kelly = 0.12 (slight throttle)
    win_rate in [0.45, 0.60): Kelly = 0.08 (defensive)
    win_rate < 0.45: Kelly = 0.05 (crisis — something's wrong)

Rolling window: last 30 settlements (gives ~30-min to 1-hour reactive horizon).

Everything else identical to algo03: FRAC=0.10, RHO=0.25, 50-70c band,
Z_ENTRY=2.0, Z_EXIT=1.2, hourly skip.
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


# ── Signal / Kelly base ───────────────────────────────────────────────
WINDOW_S = 60
MIN_SAMPLES = 30
SIGNAL_LOOKBACK_S = 30

Z_ENTRY = 2.0
Z_EXIT = 1.2
RHO = 0.25
P_MAX = 0.85
P_MIN_EDGE = 0.01

# Adaptive Kelly — tier thresholds
ROLLING_WINDOW = 30                 # last N settlements for win-rate estimate
MIN_SETTLEMENTS_FOR_ADAPT = 10      # before we have this many, use default
KELLY_NORMAL = 0.15
KELLY_SLIGHT = 0.12
KELLY_DEFENSIVE = 0.08
KELLY_CRISIS = 0.05
WR_NORMAL = 0.75
WR_SLIGHT = 0.60
WR_DEFENSIVE = 0.45

MIN_SHARES = 1.0
MIN_ASK = 0.50
MAX_ASK = 0.70
ENTRY_FRAC_MIN = 0.10
ENTRY_FRAC_MAX = 0.80
EXIT_MIN_TIME_S = 60
POSITION_CAP = 500.0
PER_MARKET_FRAC = 0.10
LIMIT_PAD = 0.02
EXIT_SLIP_FRAC = 0.10


def _asset_from_slug(slug: str) -> str | None:
    if slug.startswith("btc-") or slug.startswith("bitcoin-"):
        return "BTC"
    if slug.startswith("eth-") or slug.startswith("ethereum-"):
        return "ETH"
    if slug.startswith("sol-") or slug.startswith("solana-"):
        return "SOL"
    return None


def _z_to_probability(z: float) -> float:
    beta = RHO / math.sqrt(1.0 - RHO * RHO)
    arg = beta * abs(z) / math.sqrt(2.0)
    return min(P_MAX, 0.5 + 0.5 * math.erf(arg))


def _kelly_fraction(p: float, ask: float) -> float:
    if ask <= 0.0 or ask >= 1.0:
        return 0.0
    return max(0.0, (p - ask) / (1.0 - ask))


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
        return recent_return / (sigma_1s * math.sqrt(lookback))


class AlgoAdaptive(BaseStrategy):
    def __init__(self) -> None:
        self._sig = _SignalTracker()
        self._last_buy_side: dict[str, Token] = {}
        # Rolling settlement outcomes (1=win, 0=loss)
        self._recent_outcomes: deque[int] = deque(maxlen=ROLLING_WINDOW)

    def _current_kelly(self) -> float:
        """Adaptive Kelly based on rolling win rate."""
        n = len(self._recent_outcomes)
        if n < MIN_SETTLEMENTS_FOR_ADAPT:
            return KELLY_NORMAL
        win_rate = sum(self._recent_outcomes) / n
        if win_rate >= WR_NORMAL:
            return KELLY_NORMAL
        if win_rate >= WR_SLIGHT:
            return KELLY_SLIGHT
        if win_rate >= WR_DEFENSIVE:
            return KELLY_DEFENSIVE
        return KELLY_CRISIS

    def on_tick(self, state: MarketState) -> list[Order]:
        self._sig.update("BTC", state.btc_mid)
        self._sig.update("ETH", state.eth_mid)
        self._sig.update("SOL", state.sol_mid)

        equity = state.total_portfolio_value
        if equity <= 0:
            return []

        kelly_now = self._current_kelly()
        per_market_budget = equity * PER_MARKET_FRAC
        remaining_cash = state.cash
        orders: list[Order] = []

        for slug, m in state.markets.items():
            if m.interval == "hourly":
                continue
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

            # Exits unchanged
            if holding and m.time_remaining_s >= EXIT_MIN_TIME_S:
                if yes_held > 0 and z < -Z_EXIT and m.yes_bid > 0:
                    limit = max(0.01, m.yes_bid * (1.0 - EXIT_SLIP_FRAC))
                    orders.append(Order(
                        market_slug=slug, token=Token.YES, side=Side.SELL,
                        size=yes_held, limit_price=limit,
                    ))
                    continue
                if no_held > 0 and z > Z_EXIT and m.no_bid > 0:
                    limit = max(0.01, m.no_bid * (1.0 - EXIT_SLIP_FRAC))
                    orders.append(Order(
                        market_slug=slug, token=Token.NO, side=Side.SELL,
                        size=no_held, limit_price=limit,
                    ))
                    continue

            if holding:
                continue
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

            p = _z_to_probability(z)
            if p - ask < P_MIN_EDGE:
                continue
            f_full = _kelly_fraction(p, ask)
            if f_full <= 0:
                continue

            # Adaptive Kelly applied here
            target_dollars = kelly_now * f_full * equity
            target_dollars = min(target_dollars, per_market_budget)
            if target_dollars <= 0:
                continue

            size = target_dollars / ask
            size = min(size, POSITION_CAP)
            if size < MIN_SHARES:
                continue

            cost = size * ask
            if cost > remaining_cash:
                size = remaining_cash / ask
                if size < MIN_SHARES:
                    continue
                cost = size * ask

            orders.append(Order(
                market_slug=slug, token=token, side=Side.BUY,
                size=size, limit_price=min(1.0, ask + LIMIT_PAD),
            ))
            remaining_cash -= cost

        return orders

    def on_fill(self, fill: Fill) -> None:
        if fill.side == Side.BUY and fill.size > 0:
            self._last_buy_side[fill.market_slug] = fill.token

    def on_settlement(self, settlement: Settlement) -> None:
        our_side = self._last_buy_side.get(settlement.market_slug)
        if our_side is None:
            return
        outcome_win = 1 if our_side == settlement.outcome else 0
        self._recent_outcomes.append(outcome_win)
        self._last_buy_side.pop(settlement.market_slug, None)
