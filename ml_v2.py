"""
ML v2 — retrained LogisticRegression without market-price features.

ml_v1 failure: the model learned P(YES) ≈ yes_ask, which is trivially true
and contains no alpha. With Kelly sizing this produced correlated blowups —
92% win rate but −93% P&L on unfiltered validation.

v2 drops yes_ask / yes_bid / spread from the feature set. The model must now
find alpha from residual signals only:
    z30, z120, z240           Binance log-return z-scores
    time_frac                 fraction of market life remaining
    book_imb                  (bid_size − ask_size) / (bid_size + ask_size) on YES book
    asset_BTC/ETH/SOL         one-hot

Out-of-sample performance (held-out validation):
    AUC      0.8345
    accuracy 75.1%
    brier    0.166

Only slightly lower AUC than ml_v1 (0.85 → 0.83), but the probability
estimate is now decoupled from the market price. Kelly can actually detect
"market thinks P=0.55 but model says P=0.70" mispricings.

Top predictors by magnitude (standardized coefficients):
    book_imb   1.345    Order-book pressure is the strongest single signal
    z240       0.347    4-minute Binance momentum
    z120       0.266    2-minute Binance momentum
    z30        0.113    30-second Binance momentum

Everything else — late-market entry, hourly skip, 50-70¢ band, Kelly sizing,
exit-on-reversal — matches leadlag_v4.
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


# ── Distilled LogisticRegression (from retrained train_svm.py) ───────
# Feature order: z30, z120, z240, time_frac, book_imb, asset_BTC, asset_ETH, asset_SOL

SCALER_MEAN = [
    0.002617541131239778,
    0.003798854958497168,
    0.0032833600986232688,
    0.4709571636011617,
    0.012059769532969777,
    0.3344627299128751,
    0.3324540174249758,
    0.33308325266214905,
]
SCALER_SCALE = [
    0.9743678713900019,
    0.9809831129787615,
    0.9431289709190199,
    0.21497400603309105,
    0.31397527497708544,
    0.47180230204197,
    0.4710927124494389,
    0.471316029282001,
]
W = [
    0.1132408258571712,       # z30
    0.26640548717261786,      # z120
    0.3466012432345144,       # z240
    0.004758343852915162,     # time_frac
    1.3450649425345669,       # book_imb
    0.003310296303553686,     # asset_BTC
    -0.008454662121338167,    # asset_ETH
    0.0051369445227977635,    # asset_SOL
]
B = 0.06847158861126608
# LR outputs probability directly via sigmoid; use dummy Platt params so the
# inference code is identical to an SVM distillation (A=-1, B=0 → P = sigmoid(score)).
PLATT_A = -1.0
PLATT_B = 0.0


# ── Signal windows (same as v4) ──────────────────────────────────────
WINDOW_S = 300
MIN_SAMPLES = 30

# ── Entry / exit ─────────────────────────────────────────────────────
P_MIN_EDGE = 0.03            # stricter than v4 (p − ask ≥ 3¢) — model probs are less calibrated
P_REVERSAL_MARGIN = 0.05     # exit if model's p flips below ask − this

# ── Kelly + size ─────────────────────────────────────────────────────
KELLY_FRACTION = 0.25
MIN_SHARES = 1.0

# ── Price band ───────────────────────────────────────────────────────
MIN_ASK = 0.50
MAX_ASK = 0.70

# ── Entry window ─────────────────────────────────────────────────────
ENTRY_FRAC_MIN = 0.10
ENTRY_FRAC_MAX = 0.80
EXIT_MIN_TIME_S = 60

# ── Risk ─────────────────────────────────────────────────────────────
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
    score = B
    for i in range(len(features)):
        x_std = (features[i] - SCALER_MEAN[i]) / SCALER_SCALE[i]
        score += W[i] * x_std
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


class MLv2(BaseStrategy):
    """LR-based probability (no market-price features) with Kelly sizing."""

    def __init__(self) -> None:
        self._sig = _SignalTracker()

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

            features = self._features(m, asset)
            if features is None:
                continue
            if m.yes_ask <= 0 or m.yes_bid <= 0:
                continue
            p_yes = _predict_p_yes(features)

            pos = state.positions.get(slug)
            yes_held = pos.yes_shares if pos else 0.0
            no_held = pos.no_shares if pos else 0.0
            holding = yes_held > 0 or no_held > 0

            # ── EXIT ──
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

            f_yes = _kelly_fraction(p_yes, m.yes_ask)
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
