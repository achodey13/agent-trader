"""
Algo Vol-Adaptive — algo03 with vol-regime-based Kelly multiplier.

Finding from Path C tests:
    Low-vol regime (bottom 15%):    $4.19/trade  — 2x algo03 average
    Mid-vol regime (middle 70%):    $0.77/trade  — weak
    High-vol regime (top 15%):     −$1.80/trade  — losing

Instead of gating, SCALE position size per regime:
    Low-vol:  Kelly × 2.0  (press the edge)
    Mid-vol:  Kelly × 1.0  (normal algo03 behavior)
    High-vol: Kelly × 0    (skip — these lose money)

Uses the same 30-min rolling σ_1s window and 15%/85% percentile bounds
as algo_vol_gate. Only the response rule differs.
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


# ── Signal / Kelly (algo03) ──────────────────────────────────────────
WINDOW_S = 60
MIN_SAMPLES = 30
SIGNAL_LOOKBACK_S = 30

Z_ENTRY = 2.0
Z_EXIT = 1.2
RHO = 0.25
P_MAX = 0.85
P_MIN_EDGE = 0.01
KELLY_FRACTION = 0.15        # base Kelly (algo03 value)
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

# ── Regime-based Kelly multipliers ───────────────────────────────────
SIGMA_WINDOW_S = 30 * 60
SIGMA_MIN_SAMPLES = 300
SIGMA_RECALC_S = 60
LOW_PCTILE = 0.15
HIGH_PCTILE = 0.85

KELLY_MULT_LOW = 2.0     # low vol: double size — this is where our edge lives
KELLY_MULT_MID = 1.0     # mid vol: algo03 behavior
KELLY_MULT_HIGH = 0.0    # high vol: skip (trades lose money here)


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
        self._sigma_history: dict[str, deque[float]] = {
            "BTC": deque(maxlen=SIGMA_WINDOW_S),
            "ETH": deque(maxlen=SIGMA_WINDOW_S),
            "SOL": deque(maxlen=SIGMA_WINDOW_S),
        }
        self._sigma_bounds: dict[str, tuple[float, float] | None] = {
            "BTC": None, "ETH": None, "SOL": None,
        }
        self._last_bounds_update: dict[str, int] = {"BTC": 0, "ETH": 0, "SOL": 0}

    def update(self, asset: str, mid: float) -> None:
        if mid > 0 and asset in self._prices:
            self._prices[asset].append(mid)

    def z_and_sigma(self, asset: str) -> tuple[float, float] | None:
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
        z = recent_return / (sigma_1s * math.sqrt(lookback))
        return z, sigma_1s

    def record_sigma(self, asset: str, sigma_1s: float, now_ts: int) -> None:
        hist = self._sigma_history[asset]
        hist.append(sigma_1s)
        if (now_ts - self._last_bounds_update[asset]) >= SIGMA_RECALC_S \
           and len(hist) >= SIGMA_MIN_SAMPLES:
            sv = sorted(hist)
            n = len(sv)
            self._sigma_bounds[asset] = (sv[int(n * LOW_PCTILE)], sv[int(n * HIGH_PCTILE)])
            self._last_bounds_update[asset] = now_ts

    def kelly_mult(self, asset: str, sigma_1s: float) -> float:
        """Returns the regime-based Kelly multiplier for this asset's current vol."""
        bounds = self._sigma_bounds[asset]
        if bounds is None:
            return KELLY_MULT_MID  # warmup: behave like algo03
        low, high = bounds
        if sigma_1s <= low:
            return KELLY_MULT_LOW
        if sigma_1s >= high:
            return KELLY_MULT_HIGH
        return KELLY_MULT_MID


class AlgoVolAdaptive(BaseStrategy):
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

        # Pre-compute per-asset z/sigma/mult
        per_asset: dict[str, tuple[float, float, float]] = {}
        for asset in ("BTC", "ETH", "SOL"):
            res = self._sig.z_and_sigma(asset)
            if res is None:
                continue
            z, sigma = res
            self._sig.record_sigma(asset, sigma, state.timestamp)
            mult = self._sig.kelly_mult(asset, sigma)
            per_asset[asset] = (z, sigma, mult)

        for slug, m in state.markets.items():
            if m.interval == "hourly":
                continue
            asset = _asset_from_slug(slug)
            if asset is None or asset not in per_asset:
                continue
            z, sigma, kelly_mult = per_asset[asset]

            pos = state.positions.get(slug)
            yes_held = pos.yes_shares if pos else 0.0
            no_held = pos.no_shares if pos else 0.0
            holding = yes_held > 0 or no_held > 0

            # EXITS always processed (regardless of regime)
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

            # ENTRY — regime-scaled Kelly
            if holding:
                continue
            if kelly_mult <= 0.0:
                continue  # high-vol regime: skip entirely
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

            # Kelly × regime multiplier
            effective_kelly = KELLY_FRACTION * kelly_mult
            target_dollars = effective_kelly * f_full * equity
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
        pass

    def on_settlement(self, settlement: Settlement) -> None:
        pass
