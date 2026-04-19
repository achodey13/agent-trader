"""
Book Imbalance v1 — pure Polymarket order-book signal.

Thesis: when one side of the YES book has much more resting volume than the
other, traders are pushing in that direction. This pressure often precedes
a price move in the same direction.

Signal (per market, per tick):
    imbalance = (total_bid_size - total_ask_size) / (total_bid_size + total_ask_size)

    imbalance > +IMBALANCE_THRESHOLD  → buy YES  (bid pressure)
    imbalance < -IMBALANCE_THRESHOLD  → buy NO   (ask pressure)

No Binance feed, no Chainlink, no BS model, no z-scores. Just the
Polymarket order book. Works on any asset — BTC, ETH, SOL — without
needing per-asset tuning.

Execution: fixed-size positions, one entry per direction per market,
hold to settlement. Same 50-70c price band and entry window as v4.

Hourly skipped (same lesson as v4 — signal horizon mismatch).
"""

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
IMBALANCE_THRESHOLD = 0.2     # fire when |imbalance| > this (range [-1, +1])

# ── Position sizing ────────────────────────────────────────────────────
POSITION_SIZE = 20.0
POSITION_CAP = 500.0
MIN_SHARES = 1.0

# ── Price band (same as v4) ───────────────────────────────────────────
MIN_ASK = 0.50
MAX_ASK = 0.70

# ── Entry window (same as v4) ─────────────────────────────────────────
ENTRY_FRAC_MIN = 0.10
ENTRY_FRAC_MAX = 0.80

# ── Order params ───────────────────────────────────────────────────────
LIMIT_PAD = 0.02


def _asset_from_slug(slug: str) -> str | None:
    if slug.startswith("btc-") or slug.startswith("bitcoin-"):
        return "BTC"
    if slug.startswith("eth-") or slug.startswith("ethereum-"):
        return "ETH"
    if slug.startswith("sol-") or slug.startswith("solana-"):
        return "SOL"
    return None


class BookImbV1(BaseStrategy):
    """Directional bet based on Polymarket YES book imbalance. One entry per market."""

    def __init__(self) -> None:
        # Track which markets we've already entered (one shot each)
        self._entered: set[str] = set()

    def on_tick(self, state: MarketState) -> list[Order]:
        equity = state.total_portfolio_value
        if equity <= 0:
            return []

        remaining_cash = state.cash
        orders: list[Order] = []

        for slug, m in state.markets.items():
            if m.interval == "hourly":
                continue
            asset = _asset_from_slug(slug)
            if asset is None:
                continue
            if slug in self._entered:
                continue

            # Entry window filter
            if m.time_remaining_frac < ENTRY_FRAC_MIN or m.time_remaining_frac > ENTRY_FRAC_MAX:
                continue

            # Compute imbalance from YES book
            bid_sz = m.yes_book.total_bid_size
            ask_sz = m.yes_book.total_ask_size
            total = bid_sz + ask_sz
            if total <= 0:
                continue
            imbalance = (bid_sz - ask_sz) / total

            if abs(imbalance) < IMBALANCE_THRESHOLD:
                continue

            # Direction: positive imbalance → bid pressure → buy YES
            if imbalance > 0:
                token, ask = Token.YES, m.yes_ask
            else:
                token, ask = Token.NO, m.no_ask

            # Price-band filter — skip degenerate / already-priced markets
            if ask <= MIN_ASK or ask >= MAX_ASK:
                continue

            # Sizing — fixed, capped by cash
            size = POSITION_SIZE
            cost = size * ask
            if cost > remaining_cash:
                size = remaining_cash / ask
                if size < MIN_SHARES:
                    continue
                cost = size * ask

            size = min(size, POSITION_CAP)

            orders.append(Order(
                market_slug=slug, token=token, side=Side.BUY,
                size=size, limit_price=min(1.0, ask + LIMIT_PAD),
            ))
            self._entered.add(slug)
            remaining_cash -= cost

        return orders

    def on_fill(self, fill: Fill) -> None:
        pass

    def on_settlement(self, settlement: Settlement) -> None:
        pass
