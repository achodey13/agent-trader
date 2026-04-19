"""
Test Fixtures - Synthetic market data for backtester tests.

Creates realistic but deterministic test data:
- 2 markets (one 5m, one 15m) with known outcomes
- Order books with bid/ask levels
- Chainlink prices that determine settlement
- Binance BTC prices
"""

from __future__ import annotations

import json

import pytest

from backtester.data_loader import BacktestData, TickData
from backtester.strategy import (
    MarketLifecycle,
    MarketStatus,
    MarketView,
    Order,
    OrderBookLevel,
    OrderBookSnapshot,
    Settlement,
    Side,
    Token,
)


# ── Constants for synthetic data ─────────────────────────────────────────────

BASE_TS = 1_700_000_000  # Some fixed unix timestamp
BTC_OPEN = 95_000.0
ETH_OPEN = 3_400.0
SOL_OPEN = 180.0

# Market 1: 5m market, BTC goes up -> YES wins
MARKET_5M_SLUG = "btc-updown-5m-1700000000"
MARKET_5M_START = BASE_TS
MARKET_5M_END = BASE_TS + 300  # 5 minutes

# Market 2: 15m market, BTC goes down -> NO wins
MARKET_15M_SLUG = "btc-updown-15m-1700000300"
MARKET_15M_START = BASE_TS + 300
MARKET_15M_END = BASE_TS + 1200  # 15 minutes

# Market 3: 5m ETH market. Included so multi-asset strategies see a non-BTC slug.
MARKET_ETH_SLUG = "eth-updown-5m-1700000000"
MARKET_ETH_START = BASE_TS
MARKET_ETH_END = BASE_TS + 300

# Market 4: 5m SOL market.
MARKET_SOL_SLUG = "sol-updown-5m-1700000000"
MARKET_SOL_START = BASE_TS
MARKET_SOL_END = BASE_TS + 300


# ── Helper functions ─────────────────────────────────────────────────────────


def make_book(
    bid_prices: list[float],
    bid_sizes: list[float],
    ask_prices: list[float],
    ask_sizes: list[float],
) -> OrderBookSnapshot:
    """Create an OrderBookSnapshot from price/size lists."""
    bids = tuple(
        OrderBookLevel(p, s)
        for p, s in zip(bid_prices, bid_sizes)
    )
    asks = tuple(
        OrderBookLevel(p, s)
        for p, s in zip(ask_prices, ask_sizes)
    )
    return OrderBookSnapshot(bids=bids, asks=asks)


def make_yes_book(mid: float = 0.50, spread: float = 0.04, depth: float = 100.0) -> OrderBookSnapshot:
    """Create a YES book centered at mid with given spread."""
    half = spread / 2
    return make_book(
        bid_prices=[mid - half, mid - half - 0.02, mid - half - 0.04],
        bid_sizes=[depth, depth * 2, depth * 3],
        ask_prices=[mid + half, mid + half + 0.02, mid + half + 0.04],
        ask_sizes=[depth, depth * 2, depth * 3],
    )


def make_no_book(mid: float = 0.50, spread: float = 0.04, depth: float = 100.0) -> OrderBookSnapshot:
    """Create a NO book (mirror of YES)."""
    return make_yes_book(mid=1.0 - mid, spread=spread, depth=depth)


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def yes_book() -> OrderBookSnapshot:
    """Standard YES order book: mid=0.50, spread=0.04."""
    return make_yes_book()


@pytest.fixture
def no_book() -> OrderBookSnapshot:
    """Standard NO order book: mid=0.50, spread=0.04."""
    return make_no_book()


@pytest.fixture
def market_5m_lifecycle() -> MarketLifecycle:
    return MarketLifecycle(
        market_slug=MARKET_5M_SLUG,
        interval="5m",
        start_ts=MARKET_5M_START,
        end_ts=MARKET_5M_END,
    )


@pytest.fixture
def market_15m_lifecycle() -> MarketLifecycle:
    return MarketLifecycle(
        market_slug=MARKET_15M_SLUG,
        interval="15m",
        start_ts=MARKET_15M_START,
        end_ts=MARKET_15M_END,
    )


@pytest.fixture
def settlement_5m_yes() -> Settlement:
    """5m market settles YES (BTC went up)."""
    return Settlement(
        market_slug=MARKET_5M_SLUG,
        interval="5m",
        outcome=Token.YES,
        start_ts=MARKET_5M_START,
        end_ts=MARKET_5M_END,
        chainlink_open=BTC_OPEN,
        chainlink_close=BTC_OPEN + 50,
    )


@pytest.fixture
def settlement_15m_no() -> Settlement:
    """15m market settles NO (BTC went down)."""
    return Settlement(
        market_slug=MARKET_15M_SLUG,
        interval="15m",
        outcome=Token.NO,
        start_ts=MARKET_15M_START,
        end_ts=MARKET_15M_END,
        chainlink_open=BTC_OPEN + 50,
        chainlink_close=BTC_OPEN - 10,
    )


@pytest.fixture
def sample_market_view(yes_book, no_book) -> MarketView:
    """A fully populated MarketView for the 5m market."""
    return MarketView(
        market_slug=MARKET_5M_SLUG,
        interval="5m",
        start_ts=MARKET_5M_START,
        end_ts=MARKET_5M_END,
        time_remaining_s=150.0,
        time_remaining_frac=0.5,
        yes_book=yes_book,
        no_book=no_book,
        yes_price=0.50,
        no_price=0.50,
        yes_bid=0.48,
        yes_ask=0.52,
        no_bid=0.48,
        no_ask=0.52,
    )


@pytest.fixture
def synthetic_backtest_data(
    market_5m_lifecycle,
    market_15m_lifecycle,
    settlement_5m_yes,
    settlement_15m_no,
) -> BacktestData:
    """
    Complete synthetic BacktestData with 2 markets over 1200 seconds.

    Market 1 (5m): ts 0-300, YES wins
    Market 2 (15m): ts 300-1200, NO wins
    BTC starts at 95000, rises to 95050 at ts 300, drops to 94990 at ts 1200.
    """
    lifecycles = [market_5m_lifecycle, market_15m_lifecycle]
    settlements = {
        MARKET_5M_SLUG: settlement_5m_yes,
        MARKET_15M_SLUG: settlement_15m_no,
    }

    timeline: list[TickData] = []

    for i in range(1201):  # 0 to 1200 inclusive
        ts = BASE_TS + i
        tick = TickData(ts_sec=ts)

        # Synthesize a BTC path that rises then falls so both YES and NO
        # outcomes are exercised across the two markets.
        if i <= 300:
            btc = BTC_OPEN + (i / 300) * 50
        else:
            btc = BTC_OPEN + 50 - ((i - 300) / 900) * 60
        tick.btc_mid = btc
        tick.btc_spread = 0.10
        tick.chainlink_btc = btc

        # Populate ETH and SOL every tick so multi-asset strategies can read
        # the oracles regardless of which markets happen to be active.
        eth = ETH_OPEN + (i / 1200) * 10
        sol = SOL_OPEN - (i / 1200) * 2
        tick.eth_mid = eth
        tick.eth_spread = 0.05
        tick.chainlink_eth = eth
        tick.sol_mid = sol
        tick.sol_spread = 0.02
        tick.chainlink_sol = sol

        # Market 1 is active during ts 0-300.
        if i < 300:
            frac = (300 - i) / 300
            yes_mid = 0.50 + (btc - BTC_OPEN) / BTC_OPEN * 5
            yes_mid = max(0.05, min(0.95, yes_mid))
            no_mid = 1.0 - yes_mid

            tick.market_prices[MARKET_5M_SLUG] = {
                "yes_price": yes_mid,
                "no_price": no_mid,
                "yes_bid": yes_mid - 0.02,
                "yes_ask": yes_mid + 0.02,
                "no_bid": no_mid - 0.02,
                "no_ask": no_mid + 0.02,
            }

            tick.order_books[MARKET_5M_SLUG] = {
                "yes_book": make_yes_book(yes_mid),
                "no_book": make_no_book(yes_mid),
            }
            tick.book_timestamps[MARKET_5M_SLUG] = ts

        # Market 2 is active during ts 300-1200.
        if 300 <= i < 1200:
            frac = (1200 - i) / 900
            yes_mid = 0.50 + (btc - (BTC_OPEN + 50)) / BTC_OPEN * 5
            yes_mid = max(0.05, min(0.95, yes_mid))
            no_mid = 1.0 - yes_mid

            tick.market_prices[MARKET_15M_SLUG] = {
                "yes_price": yes_mid,
                "no_price": no_mid,
                "yes_bid": yes_mid - 0.02,
                "yes_ask": yes_mid + 0.02,
                "no_bid": no_mid - 0.02,
                "no_ask": no_mid + 0.02,
            }

            tick.order_books[MARKET_15M_SLUG] = {
                "yes_book": make_yes_book(yes_mid),
                "no_book": make_no_book(yes_mid),
            }
            tick.book_timestamps[MARKET_15M_SLUG] = ts

        timeline.append(tick)

    return BacktestData(
        timeline=timeline,
        lifecycles=lifecycles,
        settlements=settlements,
        start_ts=BASE_TS,
        end_ts=BASE_TS + 1200,
    )
