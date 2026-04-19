"""
Hybrid v6 — Fair Value v2 on BTC + LeadLag v5 on ETH/SOL.

  • BTC: Black-Scholes binary option pricing with rolling realized volatility
    from Binance btc_mid. Size scales with edge magnitude and time confidence.

  • ETH: LeadLag v4 (z-score, Gaussian p, quarter-Kelly, 50–70¢ band).

  • SOL: LeadLag with tighter params + BTC z-score confirmation gate.

Run:
  python run_backtest.py hybrid_v6.py
  python run_backtest.py hybrid_v6.py --data data/validation
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from backtester.strategy import (
    BaseStrategy,
    Fill,
    MarketState,
    Order,
    Settlement,
    Side,
    Token,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Asset detection (shared)
# ═══════════════════════════════════════════════════════════════════════════════


def _asset_from_slug(slug: str) -> str | None:
    if slug.startswith("btc-") or slug.startswith("bitcoin-"):
        return "BTC"
    if slug.startswith("eth-") or slug.startswith("ethereum-"):
        return "ETH"
    if slug.startswith("sol-") or slug.startswith("solana-"):
        return "SOL"
    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  BTC arm — Fair Value v2 (BS binary option pricing)
# ═══════════════════════════════════════════════════════════════════════════════


FV_VOL_WINDOW_S = 15 * 60
FV_VOL_MIN_SAMPLES = 60
FV_MIN_EDGE = 0.05
FV_BASE_SIZE = 20.0
FV_MAX_EDGE_MULT = 3.0
FV_POSITION_CAP = 500.0
FV_LIMIT_PAD = 0.02


def _standard_normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


class _VolEstimator:
    """Rolling per-second log-return std from a trailing btc_mid window."""

    def __init__(self) -> None:
        self._prices: deque[float] = deque(maxlen=FV_VOL_WINDOW_S + 1)

    def update(self, price: float) -> None:
        if price > 0:
            self._prices.append(price)

    def sigma_per_second(self) -> float | None:
        if len(self._prices) < FV_VOL_MIN_SAMPLES + 1:
            return None
        prev = None
        s = 0.0
        s2 = 0.0
        n = 0
        for p in self._prices:
            if prev is not None and prev > 0 and p > 0:
                r = math.log(p / prev)
                s += r
                s2 += r * r
                n += 1
            prev = p
        if n < FV_VOL_MIN_SAMPLES:
            return None
        mean = s / n
        var = s2 / n - mean * mean
        if var <= 0:
            return None
        return math.sqrt(var)


def _fair_prob(
    current: float,
    open_price: float,
    sigma_per_second: float,
    time_remaining_s: float,
) -> float | None:
    if current <= 0 or open_price <= 0 or sigma_per_second <= 0:
        return None
    tau = max(time_remaining_s, 1.0)
    sigma_sqrt_tau = sigma_per_second * math.sqrt(tau)
    if sigma_sqrt_tau < 1e-8:
        return 0.99 if current >= open_price else 0.01
    log_m = math.log(current / open_price)
    d2 = log_m / sigma_sqrt_tau - sigma_sqrt_tau / 2.0
    return max(0.01, min(0.99, _standard_normal_cdf(d2)))


# ═══════════════════════════════════════════════════════════════════════════════
#  ETH / SOL arm — LeadLag v5
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class LeadLagParams:
    z_entry: float
    z_exit: float
    rho_default: float
    rho_floor: float
    rho_cap: float
    p_max: float
    p_min_edge: float
    kelly_fraction: float
    min_ask: float
    max_ask: float
    per_market_frac: float
    position_cap: float


PARAMS_ETH = LeadLagParams(
    z_entry=2.0,
    z_exit=1.2,
    rho_default=0.25,
    rho_floor=0.02,
    rho_cap=0.85,
    p_max=0.85,
    p_min_edge=0.01,
    kelly_fraction=0.25,
    min_ask=0.50,
    max_ask=0.70,
    per_market_frac=0.10,
    position_cap=500.0,
)

PARAMS_SOL = LeadLagParams(
    z_entry=2.6,
    z_exit=1.15,
    rho_default=0.16,
    rho_floor=0.02,
    rho_cap=0.80,
    p_max=0.82,
    p_min_edge=0.018,
    kelly_fraction=0.16,
    min_ask=0.52,
    max_ask=0.68,
    per_market_frac=0.07,
    position_cap=320.0,
)

SOL_BTC_ALIGN_Z = 1.15

LL_WINDOW_S = 60
LL_MIN_SAMPLES = 30
LL_SIGNAL_LOOKBACK_S = 30
LL_ENTRY_FRAC_MIN = 0.10
LL_ENTRY_FRAC_MAX = 0.80
LL_EXIT_MIN_TIME_S = 60
LL_MIN_SHARES = 1.0
LL_LIMIT_PAD = 0.02
LL_EXIT_SLIP_FRAC = 0.10


def _sol_btc_confirms(z_sol: float, z_btc: float | None) -> bool:
    if z_btc is None:
        return False
    if abs(z_btc) < SOL_BTC_ALIGN_Z:
        return False
    return (z_sol > 0.0 and z_btc > 0.0) or (z_sol < 0.0 and z_btc < 0.0)


def _z_to_probability(z: float, p: LeadLagParams) -> float:
    r = max(p.rho_floor, min(p.rho_cap, p.rho_default))
    den = 1.0 - r * r
    if den <= 1e-12:
        r = p.rho_default
        den = 1.0 - r * r
    beta = r / math.sqrt(den)
    arg = beta * abs(z) / math.sqrt(2.0)
    return min(p.p_max, 0.5 + 0.5 * math.erf(arg))


def _kelly_fraction(prob: float, ask: float) -> float:
    if ask <= 0.0 or ask >= 1.0:
        return 0.0
    f = (prob - ask) / (1.0 - ask)
    return max(0.0, f)


class _SignalTracker:
    def __init__(self) -> None:
        self._prices: dict[str, deque[float]] = {
            "BTC": deque(maxlen=LL_WINDOW_S + 1),
            "ETH": deque(maxlen=LL_WINDOW_S + 1),
            "SOL": deque(maxlen=LL_WINDOW_S + 1),
        }

    def update(self, asset: str, mid: float) -> None:
        if mid > 0 and asset in self._prices:
            self._prices[asset].append(mid)

    def z_score(self, asset: str) -> float | None:
        buf = self._prices.get(asset)
        if buf is None or len(buf) < LL_MIN_SAMPLES + 1:
            return None
        prices = list(buf)
        lookback = min(LL_SIGNAL_LOOKBACK_S, len(prices) - 1)
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
        if len(returns) < LL_MIN_SAMPLES:
            return None
        mean = sum(returns) / len(returns)
        var = sum((r - mean) ** 2 for r in returns) / len(returns)
        if var <= 0:
            return None
        sigma_1s = math.sqrt(var)
        null_std = sigma_1s * math.sqrt(lookback)
        return recent_return / null_std


# ═══════════════════════════════════════════════════════════════════════════════
#  Unified strategy
# ═══════════════════════════════════════════════════════════════════════════════


class HybridV6(BaseStrategy):
    """
    BTC  → Fair Value v2 (BS pricing + rolling vol + confidence sizing)
    ETH  → LeadLag v5 (z-score + Gaussian p + quarter-Kelly)
    SOL  → LeadLag v5 (tighter params + BTC z confirmation)
    """

    def __init__(self) -> None:
        self._vol = _VolEstimator()
        self._sig = _SignalTracker()
        self._open_chainlink: dict[str, float] = {}

    # ── LeadLag buy helper (ETH / SOL) ────────────────────────────────────

    def _leadlag_buy(
        self,
        slug: str,
        token: Token,
        ask: float,
        z_abs: float,
        cfg: LeadLagParams,
        equity: float,
        remaining_cash: float,
        orders: list[Order],
    ) -> float:
        if ask <= cfg.min_ask or ask >= cfg.max_ask:
            return remaining_cash
        prob = _z_to_probability(z_abs, cfg)
        if prob - ask < cfg.p_min_edge:
            return remaining_cash
        f_full = _kelly_fraction(prob, ask)
        if f_full <= 0:
            return remaining_cash
        per_market_budget = equity * cfg.per_market_frac
        target_dollars = cfg.kelly_fraction * f_full * equity
        target_dollars = min(target_dollars, per_market_budget)
        if target_dollars <= 0:
            return remaining_cash
        size = target_dollars / ask
        size = min(size, cfg.position_cap)
        if size < LL_MIN_SHARES:
            return remaining_cash
        cost = size * ask
        if cost > remaining_cash:
            size = remaining_cash / ask
            if size < LL_MIN_SHARES:
                return remaining_cash
            cost = size * ask
        orders.append(Order(
            market_slug=slug, token=token, side=Side.BUY,
            size=size, limit_price=min(1.0, ask + LL_LIMIT_PAD),
        ))
        return remaining_cash - cost

    # ── Main tick ─────────────────────────────────────────────────────────

    def on_tick(self, state: MarketState) -> list[Order]:
        self._vol.update(state.btc_mid)
        self._sig.update("BTC", state.btc_mid)
        self._sig.update("ETH", state.eth_mid)
        self._sig.update("SOL", state.sol_mid)

        equity = state.total_portfolio_value
        if equity <= 0:
            return []

        sigma_1s = self._vol.sigma_per_second()
        z_btc = self._sig.z_score("BTC")
        remaining_cash = state.cash
        orders: list[Order] = []

        for slug, m in state.markets.items():
            if m.interval != "5m":
                continue

            asset = _asset_from_slug(slug)
            if asset is None:
                continue

            pos = state.positions.get(slug)
            yes_held = pos.yes_shares if pos else 0.0
            no_held = pos.no_shares if pos else 0.0
            holding = yes_held > 0 or no_held > 0

            # ── BTC: Fair Value v2 ────────────────────────────────────────
            if asset == "BTC":
                if state.btc_mid <= 0 or state.chainlink_btc <= 0:
                    continue
                if sigma_1s is None:
                    continue

                if slug not in self._open_chainlink:
                    self._open_chainlink[slug] = state.chainlink_btc
                open_price = self._open_chainlink[slug]

                fair = _fair_prob(state.btc_mid, open_price, sigma_1s, m.time_remaining_s)
                if fair is None:
                    continue

                mid = m.yes_price
                if mid <= 0:
                    continue

                delta = fair - mid
                abs_edge = abs(delta)
                if abs_edge < FV_MIN_EDGE:
                    continue

                edge_mult = min(FV_MAX_EDGE_MULT, abs_edge / FV_MIN_EDGE)
                time_mult = 1.0 + (1.0 - m.time_remaining_frac)
                target_size = FV_BASE_SIZE * edge_mult * time_mult

                if delta > 0 and m.yes_ask > 0:
                    current_shares = yes_held
                    size = min(target_size, FV_POSITION_CAP - current_shares)
                    if size <= 0:
                        continue
                    cost = size * m.yes_ask
                    if cost > remaining_cash:
                        size = remaining_cash / m.yes_ask
                        if size <= 0:
                            continue
                        cost = size * m.yes_ask
                    orders.append(Order(
                        market_slug=slug,
                        token=Token.YES,
                        side=Side.BUY,
                        size=size,
                        limit_price=min(fair, m.yes_ask + FV_LIMIT_PAD),
                    ))
                    remaining_cash -= cost

                elif delta < 0 and m.no_ask > 0:
                    current_shares = no_held
                    size = min(target_size, FV_POSITION_CAP - current_shares)
                    if size <= 0:
                        continue
                    cost = size * m.no_ask
                    if cost > remaining_cash:
                        size = remaining_cash / m.no_ask
                        if size <= 0:
                            continue
                        cost = size * m.no_ask
                    orders.append(Order(
                        market_slug=slug,
                        token=Token.NO,
                        side=Side.BUY,
                        size=size,
                        limit_price=min(1.0 - fair, m.no_ask + FV_LIMIT_PAD),
                    ))
                    remaining_cash -= cost

                continue

            # ── ETH / SOL: LeadLag ────────────────────────────────────────
            cfg = PARAMS_ETH if asset == "ETH" else PARAMS_SOL if asset == "SOL" else None
            if cfg is None:
                continue

            z = self._sig.z_score(asset)
            if z is None:
                continue

            # Exit on signal reversal
            if holding and m.time_remaining_s >= LL_EXIT_MIN_TIME_S:
                if yes_held > 0 and z < -cfg.z_exit and m.yes_bid > 0:
                    limit = max(0.01, m.yes_bid * (1.0 - LL_EXIT_SLIP_FRAC))
                    orders.append(Order(
                        market_slug=slug, token=Token.YES, side=Side.SELL,
                        size=yes_held, limit_price=limit,
                    ))
                    continue
                if no_held > 0 and z > cfg.z_exit and m.no_bid > 0:
                    limit = max(0.01, m.no_bid * (1.0 - LL_EXIT_SLIP_FRAC))
                    orders.append(Order(
                        market_slug=slug, token=Token.NO, side=Side.SELL,
                        size=no_held, limit_price=limit,
                    ))
                    continue

            if holding:
                continue
            if m.time_remaining_frac < LL_ENTRY_FRAC_MIN or m.time_remaining_frac > LL_ENTRY_FRAC_MAX:
                continue

            # SOL: BTC confirmation gate
            if asset == "SOL":
                if not _sol_btc_confirms(z, z_btc):
                    continue

            if abs(z) < cfg.z_entry:
                continue
            if z > 0:
                token, ask = Token.YES, m.yes_ask
            else:
                token, ask = Token.NO, m.no_ask
            remaining_cash = self._leadlag_buy(
                slug, token, ask, abs(z), cfg, equity, remaining_cash, orders,
            )

        return orders

    def on_fill(self, fill: Fill) -> None:
        pass

    def on_settlement(self, settlement: Settlement) -> None:
        pass