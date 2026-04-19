"""
Simple ML — minimum viable use of the trained model.

Rule: compute P(YES) from the ml_v2 distilled model. If P > 0.5, buy YES.
Otherwise, buy NO. Fixed 20-share size. One entry per market. Hold to
settlement. No Kelly, no price band, no edge threshold, no exits.

This is the opposite philosophy of ml_v1/ml_v2:
    - ml_v1/v2: Kelly-size based on edge magnitude, blew up on correlation
    - simple_ml: every directional call is the same size, so no single
      market can dominate the portfolio, no correlation blowup possible

The trade-off: we trade every market the model sees, including weak
signals. No filter for "model barely picks a side". Expected to be noisy
but bounded — losses per trade are capped at ~$0.80 (buying at ask up to
0.80, settling at 0).

Uses the ml_v2 coefficients (no yes_ask / yes_bid in features — forces
the model to use z-scores + book_imb for its direction call).
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


# ── Distilled LogisticRegression from ml_v2 (no market-price features) ─
# Feature order: z30, z120, z240, time_frac, book_imb, asset_BTC, asset_ETH, asset_SOL

SCALER_MEAN = [
    0.002617541131239778, 0.003798854958497168, 0.0032833600986232688,
    0.4709571636011617, 0.012059769532969777,
    0.3344627299128751, 0.3324540174249758, 0.33308325266214905,
]
SCALER_SCALE = [
    0.9743678713900019, 0.9809831129787615, 0.9431289709190199,
    0.21497400603309105, 0.31397527497708544,
    0.47180230204197, 0.4710927124494389, 0.471316029282001,
]
W = [
    0.1132408258571712, 0.26640548717261786, 0.3466012432345144,
    0.004758343852915162, 1.3450649425345669,
    0.003310296303553686, -0.008454662121338167, 0.0051369445227977635,
]
B = 0.06847158861126608


# ── Signal tracker (same as v4 / ml_v2) ──────────────────────────────
WINDOW_S = 300
MIN_SAMPLES = 30

# ── Trading ──────────────────────────────────────────────────────────
FIXED_SIZE = 20.0                # every directional bet is this many shares
ENTRY_FRAC_MIN = 0.20            # enter only when 20% < frac < 80%
ENTRY_FRAC_MAX = 0.80
LIMIT_PAD = 0.02


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
    score = B
    for i in range(len(features)):
        x_std = (features[i] - SCALER_MEAN[i]) / SCALER_SCALE[i]
        score += W[i] * x_std
    return _sigmoid(score)


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


class SimpleML(BaseStrategy):
    """Buy the direction the model prefers; fixed size; one entry; hold to settle."""

    def __init__(self) -> None:
        self._sig = _SignalTracker()
        self._entered: set[str] = set()

    def _features(self, m, asset: str) -> list[float] | None:
        zs = self._sig.z_scores(asset)
        if zs is None:
            return None
        z30, z120, z240 = zs
        bid_sz = m.yes_book.total_bid_size
        ask_sz = m.yes_book.total_ask_size
        book_imb = 0.0
        if bid_sz + ask_sz > 0:
            book_imb = (bid_sz - ask_sz) / (bid_sz + ask_sz)
        return [
            z30, z120, z240,
            m.time_remaining_frac,
            book_imb,
            1.0 if asset == "BTC" else 0.0,
            1.0 if asset == "ETH" else 0.0,
            1.0 if asset == "SOL" else 0.0,
        ]

    def on_tick(self, state: MarketState) -> list[Order]:
        self._sig.update("BTC", state.btc_mid)
        self._sig.update("ETH", state.eth_mid)
        self._sig.update("SOL", state.sol_mid)

        remaining_cash = state.cash
        orders: list[Order] = []

        for slug, m in state.markets.items():
            if slug in self._entered:
                continue
            if m.interval == "hourly":
                continue

            asset = _asset_from_slug(slug)
            if asset is None:
                continue
            if m.time_remaining_frac < ENTRY_FRAC_MIN or m.time_remaining_frac > ENTRY_FRAC_MAX:
                continue
            if m.yes_ask <= 0 or m.no_ask <= 0:
                continue

            features = self._features(m, asset)
            if features is None:
                continue
            p_yes = _predict_p_yes(features)

            if p_yes >= 0.5:
                token, ask = Token.YES, m.yes_ask
            else:
                token, ask = Token.NO, m.no_ask

            cost = FIXED_SIZE * ask
            if cost > remaining_cash:
                continue

            orders.append(Order(
                market_slug=slug, token=token, side=Side.BUY,
                size=FIXED_SIZE, limit_price=min(1.0, ask + LIMIT_PAD),
            ))
            self._entered.add(slug)
            remaining_cash -= cost

        return orders

    def on_fill(self, fill: Fill) -> None:
        pass

    def on_settlement(self, settlement: Settlement) -> None:
        pass
