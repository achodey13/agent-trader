"""
Lead-Lag Hourly — v4 adapted for 1-hour markets.

Why a separate strategy:
    v4's signal is a 30-second Binance return normalized by the per-second
    return std. On a 5-minute market horizon, 30 sec ≈ 10% of the remaining
    time — a reasonable lookahead. On a 1-hour market, 30 sec is 0.8% of the
    remaining time. The horizon mismatch means v4's signal has little to say
    about where a hourly market will settle.

Design:
    1. Signal uses a 10-minute Binance log-return, normalized against the
       30-minute trailing window's 1-sec return std via the same dimensional
       form as v4:
           z = recent_return / (σ_1s · √lookback)
       where lookback = 600 seconds and the window is 1800 seconds.
    2. Scope: only markets with `interval == "hourly"`.
    3. Same Kelly sizing, same 50-70¢ price band, same exit-on-reversal, same
       late-market-entry window (time_remaining_frac 0.1..0.8) as v4.
    4. Same probability model (ρ=0.25 cross-correlation under Gaussian).

Note on sample size: ~117 hourly markets in the 38h validation slice and
~537 in the 178h training slice. Smaller than 5m (454 val / 1361 train) but
large enough to get a meaningful signal.
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


# ── Signal windows (hourly-tuned) ──────────────────────────────────────
WINDOW_S = 30 * 60              # 30-minute trailing Binance history
MIN_SAMPLES = 10 * 60           # need 10 min of 1-sec samples before signaling
SIGNAL_LOOKBACK_S = 10 * 60     # 10-min log-return as the signal

# ── Entry / exit thresholds (same as v4) ──────────────────────────────
Z_ENTRY = 2.0
Z_EXIT = 1.2

# ── Probability estimation (same as v4) ───────────────────────────────
RHO = 0.25
P_MAX = 0.85
P_MIN_EDGE = 0.01

# ── Kelly sizing (same as v4) ─────────────────────────────────────────
KELLY_FRACTION = 0.25
MIN_SHARES = 1.0

# ── Price band (same as v4) ───────────────────────────────────────────
MIN_ASK = 0.50
MAX_ASK = 0.70

# ── Entry window (same as v4) ─────────────────────────────────────────
ENTRY_FRAC_MIN = 0.10
ENTRY_FRAC_MAX = 0.80
EXIT_MIN_TIME_S = 60

# ── Risk (same as v4) ──────────────────────────────────────────────────
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

        # Subsample — full-window per-tick computation is 1800 log() calls/tick
        # × 3 assets = 5400 per tick × 137k ticks ≈ 740M ops for the val run.
        # Compute variance over every-10th-sample instead; still statistically
        # sound (n ≈ 180) and 10× faster.
        returns: list[float] = []
        for i in range(0, len(prices) - 10, 10):
            a, b = prices[i], prices[i + 10]
            if a > 0 and b > 0:
                returns.append(math.log(b / a) / math.sqrt(10.0))  # scale to per-1-sec
        if len(returns) < 20:
            return None
        mean = sum(returns) / len(returns)
        var = sum((r - mean) ** 2 for r in returns) / len(returns)
        if var <= 0:
            return None
        sigma_1s = math.sqrt(var)
        null_std = sigma_1s * math.sqrt(lookback)
        return recent_return / null_std


class LeadLagHourly(BaseStrategy):
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
            # Scope: hourly markets only
            if m.interval != "hourly":
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

            # ── EXIT on signal reversal ──
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

            # ── ENTRY (Kelly-sized) ──
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
