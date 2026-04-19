"""
Algo SOL-Conservative — algo03 with tighter SOL-specific parameters.

Borrows v5's insight that SOL is more volatile and benefits from tighter
params, WITHOUT the BTC-confirmation gate that caused v5 to overfit on val.

BTC/ETH use algo03 parameters (Kelly=0.15, band=0.50-0.70).
SOL uses tighter parameters (Kelly=0.12, band=0.52-0.68, position_cap=320).

No BTC-confirmation gate. No separate RHO.

Tests whether just tightening SOL (without the confirmation layer)
captures the SOL-is-more-volatile insight without the overfitting.
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


# ── Signal ────────────────────────────────────────────────────────────
WINDOW_S = 60
MIN_SAMPLES = 30
SIGNAL_LOOKBACK_S = 30

Z_ENTRY = 2.0
Z_EXIT = 1.2
RHO = 0.25
P_MAX = 0.85
P_MIN_EDGE = 0.01

# ── BTC / ETH params (same as algo03) ─────────────────────────────────
KELLY_FRACTION_BTCETH = 0.15
MIN_ASK_BTCETH = 0.50
MAX_ASK_BTCETH = 0.70
POSITION_CAP_BTCETH = 500.0
PER_MARKET_FRAC_BTCETH = 0.10

# ── SOL params (tighter) ──────────────────────────────────────────────
KELLY_FRACTION_SOL = 0.12
MIN_ASK_SOL = 0.52
MAX_ASK_SOL = 0.68
POSITION_CAP_SOL = 320.0
PER_MARKET_FRAC_SOL = 0.07

ENTRY_FRAC_MIN = 0.10
ENTRY_FRAC_MAX = 0.80
EXIT_MIN_TIME_S = 60
MIN_SHARES = 1.0
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


def _params_for(asset: str) -> tuple[float, float, float, float, float]:
    """Returns (kelly, min_ask, max_ask, position_cap, per_market_frac)."""
    if asset == "SOL":
        return (KELLY_FRACTION_SOL, MIN_ASK_SOL, MAX_ASK_SOL,
                POSITION_CAP_SOL, PER_MARKET_FRAC_SOL)
    return (KELLY_FRACTION_BTCETH, MIN_ASK_BTCETH, MAX_ASK_BTCETH,
            POSITION_CAP_BTCETH, PER_MARKET_FRAC_BTCETH)


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


class LeadLagV6(BaseStrategy):
    def __init__(self) -> None:
        self._sig = _SignalTracker()

    def on_tick(self, state: MarketState) -> list[Order]:
        self._sig.update("BTC", state.btc_mid)
        self._sig.update("ETH", state.eth_mid)
        self._sig.update("SOL", state.sol_mid)

        equity = state.total_portfolio_value
        if equity <= 0:
            return []

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

            kelly, min_ask, max_ask, pos_cap, pm_frac = _params_for(asset)
            per_market_budget = equity * pm_frac

            pos = state.positions.get(slug)
            yes_held = pos.yes_shares if pos else 0.0
            no_held = pos.no_shares if pos else 0.0
            holding = yes_held > 0 or no_held > 0

            # Exits same as algo03
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
            if ask <= min_ask or ask >= max_ask:
                continue

            p = _z_to_probability(z)
            if p - ask < P_MIN_EDGE:
                continue
            f_full = _kelly_fraction(p, ask)
            if f_full <= 0:
                continue

            target_dollars = kelly * f_full * equity
            target_dollars = min(target_dollars, per_market_budget)
            if target_dollars <= 0:
                continue

            size = target_dollars / ask
            size = min(size, pos_cap)
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
        pass

    def on_settlement(self, settlement: Settlement) -> None:
        pass
