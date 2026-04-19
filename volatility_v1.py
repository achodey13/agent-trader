"""
Volatility v1 — buy both sides at open, salvage the loser before settlement.

Thesis: at market open, yes_ask + no_ask is typically ~1.01-1.02 (Polymarket
MM's spread). Buying both is a synthetic bond that pays exactly $1 at
settlement regardless of outcome, for a cost of ~$1.02 per unit.

The edge: as the market resolves, one side's bid rises toward 1 and the
other's drops toward 0. If we can sell the losing side at, say, bid 0.15
before it goes to 0, we salvage 15¢ per unit that would otherwise vanish.

EV per unit:
    cost at open                   = yes_ask + no_ask  (e.g., $1.02)
    winner settlement              = $1.00
    loser salvage (via bid sale)   = ~$0.05 to $0.20 (depends on market)
    -------
    profit per unit ≈ salvage − (entry_sum − 1)  ≈ $0.03 to $0.18

Completely orthogonal to v4:
    - No Binance signal dependency
    - No directional bet
    - No Kelly sizing (position sizes are uniform)
    - Path-dependent: needs intermediate price separation, doesn't care about
      final direction
    - Works equally on BTC / ETH / SOL, 5m / 15m / hourly (no signal horizon)

Where it SHOULD fail:
    - Flat markets (outcome uncertain until last seconds) — both sides stay
      near 0.50, no salvage opportunity, loser ends at $0 and winner at $1
      → we lose the opening spread (~1-3¢/unit)
    - Markets where `yes_ask + no_ask` opens too wide (> MAX_ENTRY_SUM) —
      we skip entry

Where it SHOULD succeed:
    - Decisive markets (one side becomes >70% early) — salvage the loser
      at a healthy bid before it collapses
    - Whippy markets (multiple big intermediate moves) — windows where the
      losing side's bid is temporarily high

State machine per market:
    (no position)          → consider entry → buy both YES and NO
    (yes > 0, no > 0)      → watch for winner → sell losing side
    (yes > 0, no == 0)     → already salvaged NO, hold YES to settlement
    (yes == 0, no > 0)     → already salvaged YES, hold NO to settlement
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


# ── Entry gates ─────────────────────────────────────────────────────────
MAX_ENTRY_SUM = 1.03          # skip markets where yes_ask + no_ask exceeds this
ENTRY_FRAC_MIN = 0.80         # must see market within first 20% of life
MIN_ASK = 0.25                # skip if either ask is already extreme (market decided)
MAX_ASK = 0.75                # same — both asks should be in healthy range

# ── Position sizing ────────────────────────────────────────────────────
POSITION_SIZE = 50.0          # base shares of EACH side (so 50 YES + 50 NO)
POSITION_CAP = 500.0          # Rules §2
PER_MARKET_FRAC = 0.05        # max 5% of equity per market (both legs combined)
MIN_SHARES = 1.0

# ── Salvage logic ──────────────────────────────────────────────────────
WIN_THRESHOLD = 0.78          # winning side's mid must exceed this to trigger salvage
MIN_SALVAGE_BID = 0.02        # don't sell loser if its bid is below this (too degenerate)
EXIT_MIN_TIME_S = 60          # don't try to sell in final 60s (book is broken)

# ── Order params ───────────────────────────────────────────────────────
LIMIT_PAD = 0.02              # willing to walk into the book by this much on entry
EXIT_SLIP_FRAC = 0.10         # sell at up to 10% below current bid


def _clean_yes_mid(m) -> float:
    """Mid from bid+ask. Avoids the yes_price contamination we found in Q2."""
    if m.yes_bid > 0 and m.yes_ask > 0:
        return (m.yes_bid + m.yes_ask) / 2.0
    if m.yes_bid > 0:
        return m.yes_bid
    if m.yes_ask > 0:
        return m.yes_ask
    return 0.0


class VolatilityV1(BaseStrategy):
    """Buy YES+NO at open, salvage the losing side mid-market, hold winner to settle."""

    def __init__(self) -> None:
        # Once we have either side in a market, don't consider re-entry.
        self._entered: set[str] = set()

    def on_tick(self, state: MarketState) -> list[Order]:
        equity = state.total_portfolio_value
        if equity <= 0:
            return []

        per_market_budget = equity * PER_MARKET_FRAC
        remaining_cash = state.cash
        orders: list[Order] = []

        for slug, m in state.markets.items():
            pos = state.positions.get(slug)
            yes_held = pos.yes_shares if pos else 0.0
            no_held = pos.no_shares if pos else 0.0

            # ─── ENTRY: no position yet, market still fresh ───
            if yes_held == 0 and no_held == 0 and slug not in self._entered:
                if m.time_remaining_frac < ENTRY_FRAC_MIN:
                    continue
                if m.yes_ask <= 0 or m.no_ask <= 0:
                    continue
                if m.yes_ask < MIN_ASK or m.yes_ask > MAX_ASK:
                    continue
                if m.no_ask < MIN_ASK or m.no_ask > MAX_ASK:
                    continue

                entry_sum = m.yes_ask + m.no_ask
                if entry_sum > MAX_ENTRY_SUM:
                    continue

                # Size: smaller of POSITION_SIZE / per-market budget / cash
                per_unit_cost = entry_sum
                size_budget = per_market_budget / per_unit_cost if per_unit_cost > 0 else 0
                size_cash = remaining_cash / per_unit_cost if per_unit_cost > 0 else 0
                size = min(POSITION_SIZE, POSITION_CAP, size_budget, size_cash)
                if size < MIN_SHARES:
                    continue

                # Submit both legs — §6 says orders execute in the returned order,
                # so we return them together (YES first, NO second). Reconciliation-
                # free because we're the only writer.
                orders.append(Order(
                    market_slug=slug, token=Token.YES, side=Side.BUY,
                    size=size, limit_price=min(1.0, m.yes_ask + LIMIT_PAD),
                ))
                orders.append(Order(
                    market_slug=slug, token=Token.NO, side=Side.BUY,
                    size=size, limit_price=min(1.0, m.no_ask + LIMIT_PAD),
                ))
                self._entered.add(slug)
                remaining_cash -= size * entry_sum
                continue

            # ─── SALVAGE: we own both sides, one is becoming clearly the loser ───
            if yes_held > 0 and no_held > 0:
                if m.time_remaining_s < EXIT_MIN_TIME_S:
                    continue

                yes_mid = _clean_yes_mid(m)
                if yes_mid <= 0:
                    continue

                # YES is winning → sell NO for salvage
                if yes_mid > WIN_THRESHOLD and m.no_bid > MIN_SALVAGE_BID:
                    limit = max(0.01, m.no_bid * (1.0 - EXIT_SLIP_FRAC))
                    orders.append(Order(
                        market_slug=slug, token=Token.NO, side=Side.SELL,
                        size=no_held, limit_price=limit,
                    ))
                    continue

                # NO is winning → sell YES for salvage
                if (1.0 - yes_mid) > WIN_THRESHOLD and m.yes_bid > MIN_SALVAGE_BID:
                    limit = max(0.01, m.yes_bid * (1.0 - EXIT_SLIP_FRAC))
                    orders.append(Order(
                        market_slug=slug, token=Token.YES, side=Side.SELL,
                        size=yes_held, limit_price=limit,
                    ))
                    continue

            # ─── HOLD: already salvaged one side (yes XOR no held) → ride to settlement ───
            # No action needed; engine resolves at settlement.

        return orders

    def on_fill(self, fill: Fill) -> None:
        pass

    def on_settlement(self, settlement: Settlement) -> None:
        pass
