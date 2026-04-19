"""
ML v1 INVERSE — same distilled SVM as ml_v1 but with the probability flipped.

Hypothesis under test: ml_v1 lost $9,319 on validation. If the model is
directionally correct (AUC 0.85, 76% accuracy), inverting should make it
catastrophically worse. If the model is subtly anti-predictive in a way
the metrics didn't catch, inverting might produce profit.

Expected outcome: significantly worse than ml_v1 (−$12k to −$16k range).
If it actually makes money, the ml_v1 loss wasn't from Kelly+correlation
but from a hidden direction error, and we'd have found a real edge.

Single change from ml_v1: `p_yes = 1.0 − _predict_p_yes(features)`.
All Kelly sizing, direction selection, and exit logic is untouched.

Original ml_v1 docstring follows.

-------------------------------------------------------------------------
ML v1 — LinearSVC + Platt-scaled probability estimation, Kelly-sized.

Trained on 41,320 feature rows extracted from 178h of training data; validated
on 10,138 rows from the 38h held-out validation slice.

Model performance (out-of-sample validation):
    Val AUC:      0.8517
    Val accuracy: 76.4%
    Val Brier:    0.158

Architecture: same v4 skeleton (hourly-skip, late-market entry, exit-on-reversal,
Kelly sizing), but the probability estimate is now the distilled linear SVM:

    x_std = (x − SCALER_MEAN) / SCALER_SCALE
    score = W · x_std + B                          # SVM decision function
    P(YES) = σ(−(PLATT_A · score + PLATT_B))       # Platt sigmoid

Features (in this order, must match training):
    z30, z120, z240            Binance log-return z-scores at 30/120/240s
    yes_ask, yes_bid           Polymarket top-of-book
    spread                     yes_ask − yes_bid
    time_frac                  time_remaining_s / total_duration_s
    book_imb                   (total_bid − total_ask) / (total_bid + total_ask) on YES
    asset_BTC, asset_ETH, asset_SOL   one-hot encoding

Submission compliance: only imports math, collections.deque, and
backtester.strategy — no sklearn at inference time. Coefficients are
hard-coded numpy-compatible constants below.
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


# ── Distilled model parameters (from train_svm.py) ───────────────────
# LinearSVC + Platt scaling, trained on ['z30','z120','z240','yes_ask',
# 'yes_bid','spread','time_frac','book_imb','asset_BTC','asset_ETH','asset_SOL'].

SCALER_MEAN = [
    0.002617541131239778,     # z30
    0.003798854958497168,     # z120
    0.0032833600986232688,    # z240
    0.5129148838334946,       # yes_ask
    0.49896498063891576,      # yes_bid
    0.013949903194578898,     # spread
    0.4709571636011617,       # time_frac
    0.012059769532969777,     # book_imb
    0.3344627299128751,       # asset_BTC
    0.3324540174249758,       # asset_ETH
    0.33308325266214905,      # asset_SOL
]
SCALER_SCALE = [
    0.9743678713900019,
    0.9809831129787615,
    0.9431289709190199,
    0.27314827307063916,
    0.27351450593098436,
    0.0186893635852069,
    0.21497400603309105,
    0.31397527497708544,
    0.47180230204197,
    0.4710927124494389,
    0.471316029282001,
]
W = [
    0.059976483198962226,     # z30
    0.07909857268613478,      # z120
    0.06435757417342233,      # z240
    0.2379652764885088,       # yes_ask
    0.2376259161647748,       # yes_bid
    0.00030334191858687973,   # spread
    0.003036868392433001,     # time_frac
    0.03649347962663801,      # book_imb
    0.0033171256124823628,    # asset_BTC
    -0.0025535642985204986,   # asset_ETH
    -0.000768193623458446,    # asset_SOL
]
B = 0.02509486030162445
PLATT_A = -2.6599868129486053
PLATT_B = 0.00010154032988550932


# ── Signal tracker (same as v4) ──────────────────────────────────────
WINDOW_S = 300
MIN_SAMPLES = 30

# ── Entry / exit thresholds ──────────────────────────────────────────
P_MIN_EDGE = 0.02           # require p − ask ≥ this to fire
P_REVERSAL_MARGIN = 0.05    # exit YES if model P(YES) < ask − this

# ── Kelly sizing (same as v4) ────────────────────────────────────────
KELLY_FRACTION = 0.25
MIN_SHARES = 1.0

# ── Price band (same as v4) ──────────────────────────────────────────
MIN_ASK = 0.50
MAX_ASK = 0.70

# ── Entry window (same as v4) ────────────────────────────────────────
ENTRY_FRAC_MIN = 0.10
ENTRY_FRAC_MAX = 0.80
EXIT_MIN_TIME_S = 60

# ── Risk (same as v4) ────────────────────────────────────────────────
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


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _predict_p_yes(features: list[float]) -> float:
    """Standardize → linear score → Platt sigmoid. Pure math, no numpy needed."""
    score = B
    for i in range(len(features)):
        x_std = (features[i] - SCALER_MEAN[i]) / SCALER_SCALE[i]
        score += W[i] * x_std
    # Platt scaling gives P = 1/(1+exp(A·score+B)) for the positive class.
    z = PLATT_A * score + PLATT_B
    return _sigmoid(-z)


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

    def _z(self, prices: list[float], lookback_s: int) -> float | None:
        if len(prices) < MIN_SAMPLES + 1:
            return None
        lookback = min(lookback_s, len(prices) - 1)
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

    def z_scores(self, asset: str) -> tuple[float, float, float] | None:
        buf = self._prices.get(asset)
        if buf is None or len(buf) < MIN_SAMPLES + 1:
            return None
        prices = list(buf)
        z30 = self._z(prices, 30)
        z120 = self._z(prices, 120)
        z240 = self._z(prices, 240)
        if z30 is None or z120 is None or z240 is None:
            return None
        return z30, z120, z240


class MLv1Inverse(BaseStrategy):
    """ML-based probability estimation with Kelly sizing, otherwise like v4."""

    def __init__(self) -> None:
        self._sig = _SignalTracker()

    def _features(self, state: MarketState, m, asset: str) -> list[float] | None:
        zs = self._sig.z_scores(asset)
        if zs is None:
            return None
        z30, z120, z240 = zs
        if m.yes_ask <= 0 or m.yes_bid <= 0:
            return None
        spread = m.yes_ask - m.yes_bid
        bid_sz = m.yes_book.total_bid_size
        ask_sz = m.yes_book.total_ask_size
        book_imb = 0.0
        if bid_sz + ask_sz > 0:
            book_imb = (bid_sz - ask_sz) / (bid_sz + ask_sz)
        return [
            z30, z120, z240,
            m.yes_ask, m.yes_bid, spread,
            m.time_remaining_frac, book_imb,
            1.0 if asset == "BTC" else 0.0,
            1.0 if asset == "ETH" else 0.0,
            1.0 if asset == "SOL" else 0.0,
        ]

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

            features = self._features(state, m, asset)
            if features is None:
                continue
            # INVERSION EXPERIMENT: flip the model's probability. Downstream
            # Kelly sizing / direction / exit logic is unchanged — it now just
            # treats the opposite direction as favored.
            p_yes = 1.0 - _predict_p_yes(features)

            pos = state.positions.get(slug)
            yes_held = pos.yes_shares if pos else 0.0
            no_held = pos.no_shares if pos else 0.0
            holding = yes_held > 0 or no_held > 0

            # ── EXIT on model reversal ──
            if holding and m.time_remaining_s >= EXIT_MIN_TIME_S:
                if yes_held > 0 and p_yes < m.yes_ask - P_REVERSAL_MARGIN and m.yes_bid > 0:
                    limit = max(0.01, m.yes_bid * (1.0 - EXIT_SLIP_FRAC))
                    orders.append(Order(
                        market_slug=slug, token=Token.YES, side=Side.SELL,
                        size=yes_held, limit_price=limit,
                    ))
                    continue
                if no_held > 0 and (1.0 - p_yes) < m.no_ask - P_REVERSAL_MARGIN and m.no_bid > 0:
                    limit = max(0.01, m.no_bid * (1.0 - EXIT_SLIP_FRAC))
                    orders.append(Order(
                        market_slug=slug, token=Token.NO, side=Side.SELL,
                        size=no_held, limit_price=limit,
                    ))
                    continue

            # ── ENTRY ──
            if holding:
                continue
            if m.time_remaining_frac < ENTRY_FRAC_MIN or m.time_remaining_frac > ENTRY_FRAC_MAX:
                continue

            # Evaluate YES side
            f_yes = _kelly_fraction(p_yes, m.yes_ask)
            # Evaluate NO side
            f_no = _kelly_fraction(1.0 - p_yes, m.no_ask)

            if f_yes >= f_no and f_yes > 0:
                token, ask = Token.YES, m.yes_ask
                p_direction = p_yes
                f_full = f_yes
            elif f_no > f_yes and f_no > 0:
                token, ask = Token.NO, m.no_ask
                p_direction = 1.0 - p_yes
                f_full = f_no
            else:
                continue

            if ask <= MIN_ASK or ask >= MAX_ASK:
                continue
            if p_direction - ask < P_MIN_EDGE:
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
