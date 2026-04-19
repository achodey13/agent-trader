"""
Polymarket Crypto Prediction Market - Hackathon Strategy Template
==================================================================

Welcome! This file is your starting point. Copy it, rename it, and implement
your trading logic in the `on_tick` method.

COMPETITION RULES
-----------------
- Starting capital: $10,000
- No short selling - you can only SELL tokens you already own.
- Max 500 shares per token (YES or NO) per market.
- 1-second execution latency - orders placed at tick T fill at tick T+1.
- Scoring: Total P&L is the primary metric; Sharpe ratio breaks ties.
- Submit a single .py file containing your strategy class.

MARKET SELECTION - YOU PICK YOUR SCOPE
---------------------------------------
Your strategy receives ALL active markets across all 3 assets (BTC, SOL, ETH)
and all 3 intervals (5m, 15m, hourly). It is up to you which of these you
actually trade.

Some strategies you might build:

  Specialist       - trade one asset, one interval only (e.g. 5m BTC only)
  Multi-asset      - e.g. all 3 assets, only 5m intervals
  Multi-interval   - e.g. BTC only, across 5m + 15m + hourly
  Generalist       - trade everything

Examples:

    # Trade only 5m BTC
    for slug, market in state.markets.items():
        if market.interval != "5m":
            continue
        if not slug.startswith("btc-"):
            continue
        # ... your logic

    # Trade all 5m across all three assets
    for slug, market in state.markets.items():
        if market.interval != "5m":
            continue
        # ... your logic

    # Trade everything
    for slug, market in state.markets.items():
        # ... your logic

Asset detection by slug prefix:
    BTC → slug starts with "btc-" or "bitcoin-"
    SOL → slug starts with "sol-" or "solana-"
    ETH → slug starts with "eth-" or "ethereum-"

IMPORTANT - Dev-loop filters vs. submission
-------------------------------------------
`python run_backtest.py my_strategy.py --assets BTC --intervals 5m` makes the
backtester load ONLY 5m BTC data, which is much faster for iteration. But the
final scoring run is UNFILTERED - the judge loads every asset and every
interval. Your submitted strategy must filter markets itself (as above);
the CLI filter does NOT carry over to scoring.

HOW TO RUN
----------
    python run_backtest.py my_strategy.py

HOW IT WORKS
------------
Every second, the engine calls your `on_tick(state)` method with a frozen
snapshot of the world: BTC prices, active Polymarket prediction markets,
your portfolio, and full order books. You return a list of `Order` objects
describing the trades you want to make.

Markets are short-duration crypto price prediction markets for BTC, SOL, and
ETH (5-minute, 15-minute, or hourly intervals). Each market resolves YES or
NO based on whether the asset's price (per Chainlink oracle) is above or
below its opening price at expiry. Your job is to identify when market prices
are mispriced relative to your estimate of the true probability and trade
accordingly.
"""

from backtester.strategy import (
    BaseStrategy,
    Fill,
    MarketState,
    MarketView,
    Order,
    OrderBookSnapshot,
    PositionView,
    Settlement,
    Side,
    Token,
)


class MyStrategy(BaseStrategy):
    """
    Your strategy class. Rename it if you like, but it must subclass BaseStrategy.

    The engine will instantiate it once and then call on_tick() every second.
    Use __init__ to set up any state you need to track across ticks.
    """

    def __init__(self) -> None:
        # ── Track whatever state you need between ticks ──────────────────
        # Example: remember which markets we have already traded in.
        self.traded_markets: set[str] = set()

    # ────────────────────────────────────────────────────────────────────────
    #  on_tick - REQUIRED
    # ────────────────────────────────────────────────────────────────────────

    def on_tick(self, state: MarketState) -> list[Order]:
        """
        Called every 1-second tick. Return a list of Orders (or an empty list).

        AVAILABLE DATA - ``state: MarketState``
        ========================================

        Timing
        ------
        state.timestamp          int     Unix epoch seconds of the current tick.
        state.timestamp_utc      str     ISO 8601 timestamp (e.g. "2025-06-01T12:00:00Z").

        Portfolio
        ---------
        state.cash                   float   Available cash balance.
        state.total_portfolio_value  float   Cash + mark-to-market value of all positions.
        state.positions              dict[str, PositionView]
            Keyed by market_slug. Each PositionView has:
                .market_slug    str     Market identifier.
                .yes_shares     float   Number of YES tokens held.
                .no_shares      float   Number of NO tokens held.
                .cost_basis     float   Total cost paid to acquire the position.

        Reference Prices - per asset (BTC, ETH, SOL)
        ---------------------------------------------
        state.btc_mid        float   Binance BTCUSDT mid-price (best bid + best ask) / 2.
        state.btc_spread     float   Binance BTCUSDT spread (best ask - best bid).
        state.chainlink_btc  float   Chainlink on-chain oracle BTC price (used for settlement).
        state.eth_mid        float   Binance ETHUSDT mid-price.
        state.eth_spread     float   Binance ETHUSDT spread.
        state.chainlink_eth  float   Chainlink on-chain oracle ETH price.
        state.sol_mid        float   Binance SOLUSDT mid-price.
        state.sol_spread     float   Binance SOLUSDT spread.
        state.chainlink_sol  float   Chainlink on-chain oracle SOL price.
        Use the oracle that matches your market's asset - pairing an ETH
        market with state.chainlink_btc will not work.

        Active Markets - ``state.markets: dict[str, MarketView]``
        ----------------------------------------------------------
        Keyed by market_slug. Each MarketView has:

            Identification / timing:
                .market_slug         str     Unique market identifier.
                .interval            str     Duration bucket: "5m", "15m", or "hourly".
                .start_ts            int     Market start time (unix seconds).
                .end_ts              int     Market end time (unix seconds).
                .time_remaining_s    float   Seconds until settlement.
                .time_remaining_frac float   Fraction of market duration remaining
                                             (1.0 at open, 0.0 at expiry).

            Top-of-book prices (convenience fields):
                .yes_price   float   Last traded / mid YES price.
                .no_price    float   Last traded / mid NO price.
                .yes_bid     float   Best bid for YES tokens.
                .yes_ask     float   Best ask for YES tokens.
                .no_bid      float   Best bid for NO tokens.
                .no_ask      float   Best ask for NO tokens.

            Full order books:
                .yes_book    OrderBookSnapshot   Full YES order book.
                .no_book     OrderBookSnapshot   Full NO order book.

                Each OrderBookSnapshot exposes:
                    .bids            tuple[OrderBookLevel, ...]  Descending by price.
                    .asks            tuple[OrderBookLevel, ...]  Ascending by price.
                    .best_bid        float   Highest bid price (0.0 if empty).
                    .best_ask        float   Lowest ask price (0.0 if empty).
                    .mid             float   (best_bid + best_ask) / 2.
                    .spread          float   best_ask - best_bid.
                    .total_bid_size  float   Sum of all bid sizes.
                    .total_ask_size  float   Sum of all ask sizes.

                Each OrderBookLevel has:
                    .price   float
                    .size    float

        RETURNING ORDERS
        ================
        Return a list of Order objects:

            Order(
                market_slug = "btc-above-100000-at-12:05",
                token       = Token.YES,        # or Token.NO
                side        = Side.BUY,          # or Side.SELL
                size        = 10.0,              # number of shares
                limit_price = 0.55,              # max price for BUY, min for SELL
                                                 # None = market order (take best)
            )

        Orders are validated by the engine:
        - You must have enough cash to cover BUY orders.
        - You must own the tokens to SELL them (no short selling).
        - Position limit: 500 shares per token per market.
        - Orders execute at the NEXT tick (1-second latency).
        """

        orders: list[Order] = []

        # ── Example logic: buy 10 YES shares in every new 5-minute market ──
        for slug, market in state.markets.items():

            # Skip markets we have already traded in.
            if slug in self.traded_markets:
                continue

            # Only trade 5-minute markets in this example.
            if market.interval != "5m":
                continue

            # Only enter near the start of the market (more than 80% time left).
            if market.time_remaining_frac < 0.80:
                continue

            # Make sure there is a reasonable ask price to buy at.
            if market.yes_ask <= 0 or market.yes_ask > 0.90:
                continue

            # Make sure we have enough cash.
            cost_estimate = 10.0 * market.yes_ask
            if state.cash < cost_estimate:
                continue

            # Place a limit buy order for 10 YES shares.
            orders.append(
                Order(
                    market_slug=slug,
                    token=Token.YES,
                    side=Side.BUY,
                    size=10.0,
                    limit_price=market.yes_ask,  # willing to pay up to the ask
                )
            )

            # Remember that we traded this market.
            self.traded_markets.add(slug)

        return orders

    # ────────────────────────────────────────────────────────────────────────
    #  on_fill - OPTIONAL
    # ────────────────────────────────────────────────────────────────────────

    def on_fill(self, fill: Fill) -> None:
        """
        Called whenever one of your orders is executed.

        Use this to update internal tracking, log trades, or adjust state.

        ``fill`` fields:
            .market_slug   str      Market the fill occurred in.
            .token         Token    YES or NO.
            .side          Side     BUY or SELL.
            .size          float    Number of shares filled.
            .avg_price     float    Volume-weighted average fill price.
            .cost          float    Total cost (size * avg_price).
            .timestamp     int      Unix seconds when the fill happened.
            .order         Order    The original order that generated this fill.
        """
        pass  # Replace with your own bookkeeping logic.

    # ────────────────────────────────────────────────────────────────────────
    #  on_settlement - OPTIONAL
    # ────────────────────────────────────────────────────────────────────────

    def on_settlement(self, settlement: Settlement) -> None:
        """
        Called when a market resolves (reaches its end time).

        Use this to learn from outcomes and refine your model.

        ``settlement`` fields:
            .market_slug      str     Market identifier.
            .interval         str     "5m", "15m", or "hourly".
            .outcome          Token   Token.YES or Token.NO - the winning side.
            .start_ts         int     Market start time (unix seconds).
            .end_ts           int     Market end time (unix seconds).
            .chainlink_open   float   Chainlink BTC price at market open.
            .chainlink_close  float   Chainlink BTC price at market close.
        """
        pass  # Replace with your own learning logic.
