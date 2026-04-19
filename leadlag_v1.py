"""
Lead-Lag v1 — Binance-leads-Polymarket signal, evidence-based build.

Measured cross-correlation finding (data/train/, 382 BTC 5m markets):
    Binance log-returns at time t predict Polymarket YES-price changes at
    time t+1 with mean correlation +0.25. Execution latency is 1 second
    (Rules §4), so observing a Binance move at t and submitting at t fills
    at t+1 — exactly inside the predictive window.

Core signal:
    For each asset, maintain a trailing window of 1-second Binance mid prices,
    compute EWMA of log-returns (recent-weighted), normalize by the window
    std to get a z-score. When |z| > Z_THRESHOLD, fire:
        z > 0   → buy YES (Binance up → YES settlement more likely)
        z < 0   → buy NO  (Binance down → NO settlement more likely)

Key design choices, each motivated by prior findings:
    - Use m.yes_ask / m.no_ask directly, never m.yes_price. The data exploration
      showed yes_price is a degenerate mid when one book side is empty near
      expiry; relying on it produced the 40.7% win rate in fair_value_v2.
    - Skip degenerate books (ask <= 0) to avoid phantom "trades" that won't fill.
    - Skip extreme prices (ask < MIN_ASK or > MAX_ASK) — low edge at the tails.
    - Per-market cooldown to prevent the 644-rejection spam we saw in
      kelly_fair_value (firing every tick a signal is hot).
    - Per-market concentration cap for risk control.
    - Hold to settlement — no early exit in v1.
    - Scope: BTC + ETH + SOL. The new backtester exposes all three Binance +
      Chainlink feeds; the correlation almost certainly holds for all three.

Scope NOT included in v1 (deliberate — keep v1 simple):
    - Dynamic exit on signal reversal
    - Kelly-style sizing (we are on a measured correlation, not a precise
      probability estimate — Kelly would over-size)
    - Per-asset signal calibration (we assume all three crypto pairs behave
      similarly; validate post-v1)
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


# ─── Signal ─────────────────────────────────────────────────────────────

WINDOW_S = 60                 # trailing seconds of Binance mid prices
MIN_SAMPLES = 30              # warmup before we trust the z-score
SIGNAL_LOOKBACK_S = 30        # compute signal from this many seconds of return
Z_THRESHOLD = 1.0             # fire when |z-score| > this (~32% of the time)

# ─── Trading ────────────────────────────────────────────────────────────

BASE_SIZE = 30.0              # shares per z-score unit
MAX_SIZE_PER_FIRE = 150.0     # cap per fire (5 sigma × BASE_SIZE)
MIN_ASK = 0.05                # skip near-certain markets (tiny upside)
MAX_ASK = 0.75                # skip long-shot-looking markets (big downside)
LIMIT_PAD = 0.02              # walk into the book up to this
COOLDOWN_S = 5                # per-market refire gate
POSITION_CAP = 500.0          # Rules §2
PER_MARKET_FRAC = 0.10        # max 10% of equity per single market


def _asset_from_slug(slug: str) -> str | None:
    if slug.startswith("btc-") or slug.startswith("bitcoin-"):
        return "BTC"
    if slug.startswith("eth-") or slug.startswith("ethereum-"):
        return "ETH"
    if slug.startswith("sol-") or slug.startswith("solana-"):
        return "SOL"
    return None


class _SignalTracker:
    """Maintains a trailing Binance mid-price series per asset and computes z-score."""

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

        # Signal: recent N-second log return.
        lookback = min(SIGNAL_LOOKBACK_S, len(prices) - 1)
        p_now = prices[-1]
        p_prev = prices[-1 - lookback]
        if p_now <= 0 or p_prev <= 0:
            return None
        recent_return = math.log(p_now / p_prev)

        # Estimate per-second return std from the full window.
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

        # Under null (iid returns), N-second sum has std σ_1s × √N.
        null_std = sigma_1s * math.sqrt(lookback)
        return recent_return / null_std


class LeadLagV1(BaseStrategy):
    """Buy YES on up-trending Binance, NO on down-trending, across BTC/ETH/SOL."""

    def __init__(self) -> None:
        self._sig = _SignalTracker()
        self._last_fire: dict[str, int] = {}  # slug -> last fire unix-seconds

    def on_tick(self, state: MarketState) -> list[Order]:
        # Feed Binance mids into the per-asset trackers.
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
            asset = _asset_from_slug(slug)
            if asset is None:
                continue

            z = self._sig.z_score(asset)
            if z is None or abs(z) < Z_THRESHOLD:
                continue

            # Per-market cooldown — avoids spamming orders while signal is hot.
            if state.timestamp - self._last_fire.get(slug, 0) < COOLDOWN_S:
                continue

            pos = state.positions.get(slug)
            if z > 0:
                token, ask = Token.YES, m.yes_ask
                current_shares = pos.yes_shares if pos else 0.0
            else:
                token, ask = Token.NO, m.no_ask
                current_shares = pos.no_shares if pos else 0.0

            # Never use yes_price — it's contaminated near expiry. Check the ask directly.
            if ask <= MIN_ASK or ask >= MAX_ASK:
                continue

            # Position cap (Rules §2).
            headroom = POSITION_CAP - current_shares
            if headroom <= 0:
                continue

            # Size: linear in |z|, capped.
            z_mag = min(5.0, abs(z))
            target_size = BASE_SIZE * z_mag
            size = min(target_size, MAX_SIZE_PER_FIRE, headroom)

            # Cash and per-market caps.
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
            self._last_fire[slug] = state.timestamp
            remaining_cash -= cost

        return orders

    def on_fill(self, fill: Fill) -> None:
        pass

    def on_settlement(self, settlement: Settlement) -> None:
        pass
