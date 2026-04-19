"""
Final v5 — single file, full strategy.

  • Bitcoin & Ethereum: original LeadLag v4 (z-score, Gaussian p, quarter-Kelly,
    50–70¢ band, same exits as v2/v4).

  • Solana: separate parameter bundle (tighter band, smaller Kelly, higher z_entry)
    plus a BTC confirmation gate: only enter when Binance BTC z agrees in sign and
    |z_btc| is above a floor.

Run:
  python run_backtest.py final_v5.py --data data/validation --intervals 5m
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
#  BTC / ETH — LeadLag v4 (unchanged economics)
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


PARAMS_BTC_ETH = LeadLagParams(
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


# ═══════════════════════════════════════════════════════════════════════════════
#  SOL — own bundle + BTC alignment (not the same as BTC/ETH)
# ═══════════════════════════════════════════════════════════════════════════════


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


# ═══════════════════════════════════════════════════════════════════════════════
#  Shared signal / time / risk (both arms)
# ═══════════════════════════════════════════════════════════════════════════════

WINDOW_S = 60
MIN_SAMPLES = 30
SIGNAL_LOOKBACK_S = 30

ENTRY_FRAC_MIN = 0.10
ENTRY_FRAC_MAX = 0.80
EXIT_MIN_TIME_S = 60
MIN_SHARES = 1.0
LIMIT_PAD = 0.02
EXIT_SLIP_FRAC = 0.10


def _sol_btc_confirms(z_sol: float, z_btc: float | None) -> bool:
    if z_btc is None:
        return False
    if abs(z_btc) < SOL_BTC_ALIGN_Z:
        return False
    return (z_sol > 0.0 and z_btc > 0.0) or (z_sol < 0.0 and z_btc < 0.0)


def _asset_from_slug(slug: str) -> str | None:
    if slug.startswith("btc-") or slug.startswith("bitcoin-"):
        return "BTC"
    if slug.startswith("eth-") or slug.startswith("ethereum-"):
        return "ETH"
    if slug.startswith("sol-") or slug.startswith("solana-"):
        return "SOL"
    return None


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


# ═══════════════════════════════════════════════════════════════════════════════
#  Unified strategy (BTC/ETH arm + SOL arm in one on_tick)
# ═══════════════════════════════════════════════════════════════════════════════


class FinalV5(BaseStrategy):
    """
    Single entrypoint: trades all assets Polymarket slugs map to, using PARAMS_BTC_ETH
    for BTC/ETH markets and PARAMS_SOL for SOL markets (with BTC z filter on entries).
    """

    def __init__(self) -> None:
        self._sig = _SignalTracker()

    @staticmethod
    def _cfg_for_asset(asset: str) -> LeadLagParams | None:
        if asset in ("BTC", "ETH"):
            return PARAMS_BTC_ETH
        if asset == "SOL":
            return PARAMS_SOL
        return None

    def _append_buy(
        self,
        slug: str,
        token: Token,
        ask: float,
        z_for_prob: float,
        cfg: LeadLagParams,
        equity: float,
        remaining_cash: float,
        orders: list[Order],
    ) -> float:
        if ask <= cfg.min_ask or ask >= cfg.max_ask:
            return remaining_cash
        prob = _z_to_probability(z_for_prob, cfg)
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
        if size < MIN_SHARES:
            return remaining_cash
        cost = size * ask
        if cost > remaining_cash:
            size = remaining_cash / ask
            if size < MIN_SHARES:
                return remaining_cash
            cost = size * ask
        orders.append(Order(
            market_slug=slug, token=token, side=Side.BUY,
            size=size, limit_price=min(1.0, ask + LIMIT_PAD),
        ))
        return remaining_cash - cost

    def on_tick(self, state: MarketState) -> list[Order]:
        self._sig.update("BTC", state.btc_mid)
        self._sig.update("ETH", state.eth_mid)
        self._sig.update("SOL", state.sol_mid)

        equity = state.total_portfolio_value
        if equity <= 0:
            return []

        z_btc = self._sig.z_score("BTC")
        remaining_cash = state.cash
        orders: list[Order] = []

        for slug, m in state.markets.items():
            if m.interval == "hourly":
                continue

            asset = _asset_from_slug(slug)
            if asset is None:
                continue
            cfg = self._cfg_for_asset(asset)
            if cfg is None:
                continue

            z = self._sig.z_score(asset)
            if z is None:
                continue

            pos = state.positions.get(slug)
            yes_held = pos.yes_shares if pos else 0.0
            no_held = pos.no_shares if pos else 0.0
            holding = yes_held > 0 or no_held > 0

            if holding and m.time_remaining_s >= EXIT_MIN_TIME_S:
                if yes_held > 0 and z < -cfg.z_exit and m.yes_bid > 0:
                    limit = max(0.01, m.yes_bid * (1.0 - EXIT_SLIP_FRAC))
                    orders.append(Order(
                        market_slug=slug, token=Token.YES, side=Side.SELL,
                        size=yes_held, limit_price=limit,
                    ))
                    continue
                if no_held > 0 and z > cfg.z_exit and m.no_bid > 0:
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

            # ── SOL arm ──────────────────────────────────────────────────────
            if asset == "SOL":
                if not _sol_btc_confirms(z, z_btc):
                    continue
                if abs(z) < cfg.z_entry:
                    continue
                if z > 0:
                    token, ask = Token.YES, m.yes_ask
                else:
                    token, ask = Token.NO, m.no_ask
                remaining_cash = self._append_buy(
                    slug, token, ask, abs(z), cfg, equity, remaining_cash, orders,
                )
                continue

            # ── BTC / ETH arm (classic v4) ─────────────────────────────────
            if abs(z) < cfg.z_entry:
                continue
            if z > 0:
                token, ask = Token.YES, m.yes_ask
            else:
                token, ask = Token.NO, m.no_ask
            remaining_cash = self._append_buy(
                slug, token, ask, abs(z), cfg, equity, remaining_cash, orders,
            )

        return orders

    def on_fill(self, fill: Fill) -> None:
        pass

    def on_settlement(self, settlement: Settlement) -> None:
        pass