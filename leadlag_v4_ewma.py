"""
Lead-Lag v4 EWMA — same as v4 but volatility is EWMA instead of sample std.

v4 (sample std): σ_1s = std(last 60 seconds of log returns)
    - Uniform weighting across the window
    - Hard memory cutoff at exactly 60 samples
    - Computed fresh from scratch every tick (n=59 squared-diff ops)

v4_ewma (EWMA):  σ²_t = λ · σ²_{t-1} + (1-λ) · r_t²
    - Exponential decay of older returns
    - Soft "memory" with half-life = ln(2) / ln(1/λ)
    - Updated incrementally (1 multiply + 1 add per tick)
    - Adapts faster to regime shifts (large recent r² immediately bumps σ²)

Choice of λ = 0.97:
    Effective memory ≈ 1/(1-λ) = 33 samples
    Half-life ≈ 0.693 / ln(1/0.97) ≈ 22.8 samples
    Similar horizon to v4's 60-sample sample-std, for fair comparison.

Everything else identical to v4: same signal lookback (30s), same Kelly,
same 50-70c band, same exit-on-reversal, same hourly skip.
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

# NEW: EWMA decay parameter. 0.97 gives half-life ≈ 23 samples, similar
# effective memory to v4's 60-sample sample-std but more responsive to
# regime shifts.
EWMA_LAMBDA = 0.99

# ── Entry / exit thresholds (unchanged from v4) ───────────────────────
Z_ENTRY = 2.0
Z_EXIT = 1.2

# ── Probability estimation (unchanged) ────────────────────────────────
RHO = 0.25
P_MAX = 0.85
P_MIN_EDGE = 0.01

# ── Kelly sizing (unchanged) ──────────────────────────────────────────
KELLY_FRACTION = 0.25
MIN_SHARES = 1.0

# ── Price band (unchanged) ────────────────────────────────────────────
MIN_ASK = 0.50
MAX_ASK = 0.70

# ── Entry window (unchanged) ──────────────────────────────────────────
ENTRY_FRAC_MIN = 0.20
ENTRY_FRAC_MAX = 0.80
EXIT_MIN_TIME_S = 60

# ── Risk (unchanged) ──────────────────────────────────────────────────
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
    """EWMA-based volatility. Maintains per-asset σ² state updated incrementally."""

    def __init__(self) -> None:
        # Rolling price deques (still needed for the recent N-sec return)
        self._prices: dict[str, deque[float]] = {
            "BTC": deque(maxlen=WINDOW_S + 1),
            "ETH": deque(maxlen=WINDOW_S + 1),
            "SOL": deque(maxlen=WINDOW_S + 1),
        }
        # EWMA state — persisted across ticks
        self._ewma_var: dict[str, float | None] = {"BTC": None, "ETH": None, "SOL": None}
        self._prev_price: dict[str, float | None] = {"BTC": None, "ETH": None, "SOL": None}
        self._samples_seen: dict[str, int] = {"BTC": 0, "ETH": 0, "SOL": 0}

    def update(self, asset: str, mid: float) -> None:
        if mid <= 0 or asset not in self._prices:
            return

        # Update EWMA variance on the new log-return (if we have a prev)
        prev = self._prev_price[asset]
        if prev is not None and prev > 0:
            r = math.log(mid / prev)
            r2 = r * r
            current_var = self._ewma_var[asset]
            if current_var is None:
                # Initialize with the first squared return
                self._ewma_var[asset] = r2
            else:
                self._ewma_var[asset] = EWMA_LAMBDA * current_var + (1.0 - EWMA_LAMBDA) * r2
            self._samples_seen[asset] += 1

        self._prev_price[asset] = mid
        self._prices[asset].append(mid)

    def z_score(self, asset: str) -> float | None:
        buf = self._prices.get(asset)
        if buf is None or len(buf) < MIN_SAMPLES + 1:
            return None
        if self._samples_seen[asset] < MIN_SAMPLES:
            return None

        prices = list(buf)
        lookback = min(SIGNAL_LOOKBACK_S, len(prices) - 1)
        p_now = prices[-1]
        p_prev = prices[-1 - lookback]
        if p_now <= 0 or p_prev <= 0:
            return None
        recent_return = math.log(p_now / p_prev)

        ewma_var = self._ewma_var[asset]
        if ewma_var is None or ewma_var <= 0:
            return None
        sigma_1s = math.sqrt(ewma_var)
        null_std = sigma_1s * math.sqrt(lookback)
        return recent_return / null_std


class LeadLagV4Ewma(BaseStrategy):
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

            # EXIT on signal reversal
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

            # ENTRY
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
