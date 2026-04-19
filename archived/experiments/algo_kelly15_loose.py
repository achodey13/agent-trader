"""
Lead-Lag v4 — v2 + Kelly-sized bets + tight 50-70¢ entry band.

v2 baseline: +$160, Sharpe 10.28, Max DD 1.35% on BTC 5m validation. Only
profitable backtest so far. Do not break its signal or exit logic.

v4 changes:

  1. Probability estimation from the signal. Given our measured Binance →
     Polymarket cross-correlation of ρ = +0.25 at lag +1 second, under
     Gaussian assumptions:
         P(direction correct | z_signal = z) = Φ(β · z / √2),
         where β = ρ / √(1 − ρ²)  ≈  0.258
     We cap P at P_MAX = 0.85 because extrapolating beyond is unreliable.

  2. Kelly-criterion sizing. Full Kelly on a binary bet bought at ask:
         f = (p − ask) / (1 − ask)
     Quarter-Kelly (κ = 0.25) target dollars = κ · f · equity. Quarter is
     standard for the "p is estimated" case — full Kelly compounds any
     p-estimation error into over-aggressive sizing.

  3. Tightened price band to [0.50, 0.70]. The hypothesis: outside this band
     Polymarket is efficient (ask too close to 0 or 1 leaves no Kelly room
     for our ~0.25-corr edge). Inside, Kelly actually produces meaningful f.

Everything else is v2:
  - Same z-score signal (30-sec return / σ·√30 over 60s window)
  - Same signal-reversal exit (Z_EXIT = 1.2)
  - Same late-market-only entry (time_remaining_frac 0.1..0.8)
  - Same freeze exits in final 60s
  - Same one entry per direction per market

Kelly notes:
  - If p ≤ ask, f = 0 → we don't fire. Natural edge filter.
  - If equity × κ × f < ask (can't afford 1 share), we skip.
  - POSITION_CAP = 500 (Rules §2) usually binds before cash cap in this band.
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

# ── NEW: Probability estimation ───────────────────────────────────────
RHO = 0.25        # measured Binance→Polymarket cross-correlation at lag +1s
P_MAX = 0.85      # cap on P(direction correct) — beyond this we don't trust the model
P_MIN_EDGE = 0.005 # require p − ask ≥ this (else not worth fees + execution)

# ── NEW: Kelly sizing ─────────────────────────────────────────────────
KELLY_FRACTION = 0.15  # quarter-Kelly
MIN_SHARES = 1.0

# ── NEW: Tightened price band ─────────────────────────────────────────
MIN_ASK = 0.50
MAX_ASK = 0.70

# ── Entry window (unchanged) ──────────────────────────────────────────
ENTRY_FRAC_MIN = 0.1
ENTRY_FRAC_MAX = 0.8
EXIT_MIN_TIME_S = 60

# ── Risk (unchanged) ───────────────────────────────────────────────────
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
    """P(direction correct | signal z-score), from measured ρ=0.25 under Gaussian."""
    beta = RHO / math.sqrt(1.0 - RHO * RHO)  # ≈ 0.258
    arg = beta * abs(z) / math.sqrt(2.0)
    return min(P_MAX, 0.5 + 0.5 * math.erf(arg))


def _kelly_fraction(p: float, ask: float) -> float:
    """Full-Kelly fraction of bankroll for a binary bet paying $1 bought at ask."""
    if ask <= 0.0 or ask >= 1.0:
        return 0.0
    f = (p - ask) / (1.0 - ask)
    return max(0.0, f)


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


class AlgoKelly15Loose(BaseStrategy):
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
            # Skip hourly — the 30-sec signal is horizon-mismatched for 1-hour
            # settlements. Measured: v4 on hourly-only val lost $1,200 with
            # 90% win rate + asymmetric tails. Same pathology as v1 on 5m.
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

            # ─── EXIT on signal reversal (v2 logic, unchanged) ───
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

            # ─── ENTRY (Kelly-sized on probability estimate) ───
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

            # Tight price band — edge only exists here
            if ask <= MIN_ASK or ask >= MAX_ASK:
                continue

            # Signal → probability → Kelly fraction
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
