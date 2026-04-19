"""
Algo Vol-Gate — algo03 with a volatility-percentile regime gate.

Concept: track per-asset realized volatility (σ_1s) over a 30-min window.
If the current σ is in the extreme top 15% or bottom 15% of that window,
the market is either too calm (no moves to capture, spread-eaters win) or
too chaotic (signal dominated by noise) — pause new entries.

Mid-regime (middle 70%) → trade normally with algo03 parameters.

Rationale for user observation: during stagnant periods where the portfolio
sits at $11-12k, v4/algo03 keeps trading through noise regimes it shouldn't.
This gate explicitly detects and skips those regimes. Exits continue normally.

Bounds are recomputed once per minute (cheap — sort 1800 samples).
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
KELLY_FRACTION = 0.15
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

# ── NEW: Vol-regime gate ─────────────────────────────────────────────
SIGMA_WINDOW_S = 30 * 60          # 30-min rolling window of σ_1s samples
SIGMA_MIN_SAMPLES = 300           # need at least 5 min before gating
SIGMA_RECALC_S = 60               # recompute percentile bounds once per minute
LOW_PCTILE = 0.15                 # pause below this
HIGH_PCTILE = 0.85                # pause above this


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
    """Computes z-score AND maintains σ_1s history for the regime gate."""

    def __init__(self) -> None:
        self._prices: dict[str, deque[float]] = {
            "BTC": deque(maxlen=WINDOW_S + 1),
            "ETH": deque(maxlen=WINDOW_S + 1),
            "SOL": deque(maxlen=WINDOW_S + 1),
        }
        # Per-asset rolling history of σ_1s values (for the regime gate)
        self._sigma_history: dict[str, deque[float]] = {
            "BTC": deque(maxlen=SIGMA_WINDOW_S),
            "ETH": deque(maxlen=SIGMA_WINDOW_S),
            "SOL": deque(maxlen=SIGMA_WINDOW_S),
        }
        # Cached percentile bounds, refreshed every SIGMA_RECALC_S
        self._sigma_bounds: dict[str, tuple[float, float] | None] = {
            "BTC": None, "ETH": None, "SOL": None,
        }
        self._last_bounds_update: dict[str, int] = {
            "BTC": 0, "ETH": 0, "SOL": 0,
        }

    def update(self, asset: str, mid: float) -> None:
        if mid > 0 and asset in self._prices:
            self._prices[asset].append(mid)

    def z_and_sigma(self, asset: str) -> tuple[float, float] | None:
        """Returns (z, sigma_1s) — we need both to run the regime gate."""
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
        """Push current σ to the history window, recompute bounds periodically."""
        hist = self._sigma_history[asset]
        hist.append(sigma_1s)
        if (now_ts - self._last_bounds_update[asset]) >= SIGMA_RECALC_S \
           and len(hist) >= SIGMA_MIN_SAMPLES:
            sorted_vals = sorted(hist)
            n = len(sorted_vals)
            low = sorted_vals[int(n * LOW_PCTILE)]
            high = sorted_vals[int(n * HIGH_PCTILE)]
            self._sigma_bounds[asset] = (low, high)
            self._last_bounds_update[asset] = now_ts

    def regime_ok(self, asset: str, sigma_1s: float) -> bool:
        """True if current vol is in the middle 70% of its rolling distribution."""
        bounds = self._sigma_bounds[asset]
        if bounds is None:
            return True  # warmup: allow all entries
        low, high = bounds
        return sigma_1s >= high  # HIGH-ONLY: top 15% only


class AlgoVolHighOnly(BaseStrategy):
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

        # Pre-compute per-asset z/sigma and update sigma history
        per_asset: dict[str, tuple[float, float, bool]] = {}
        for asset in ("BTC", "ETH", "SOL"):
            res = self._sig.z_and_sigma(asset)
            if res is None:
                continue
            z, sigma = res
            self._sig.record_sigma(asset, sigma, state.timestamp)
            regime_ok = self._sig.regime_ok(asset, sigma)
            per_asset[asset] = (z, sigma, regime_ok)

        for slug, m in state.markets.items():
            if m.interval == "hourly":
                continue
            asset = _asset_from_slug(slug)
            if asset is None or asset not in per_asset:
                continue
            z, sigma, regime_ok = per_asset[asset]

            pos = state.positions.get(slug)
            yes_held = pos.yes_shares if pos else 0.0
            no_held = pos.no_shares if pos else 0.0
            holding = yes_held > 0 or no_held > 0

            # EXITS always processed (regardless of regime gate)
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

            # ENTRY — gated by regime check
            if holding:
                continue
            if not regime_ok:
                continue  # regime gate: current vol is abnormal, skip
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

            target_dollars = KELLY_FRACTION * f_full * equity
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
