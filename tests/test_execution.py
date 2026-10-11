"""Tests for execution module."""

import pytest

from backtester.execution import (
    MAX_BOOK_STALENESS_S,
    MAX_SHARES_PER_TOKEN,
    ExecutionEngine,
)
from backtester.strategy import MarketView, Order, Side, Token
from backtester.portfolio import Portfolio, Position
from tests.conftest import (
    MARKET_5M_END,
    MARKET_5M_SLUG,
    MARKET_5M_START,
    make_no_book,
    make_yes_book,
)


@pytest.fixture
def engine():
    return ExecutionEngine()


@pytest.fixture
def active_market():
    return {
        MARKET_5M_SLUG: MarketView(
            market_slug=MARKET_5M_SLUG,
            interval="5m",
            start_ts=MARKET_5M_START,
            end_ts=MARKET_5M_END,
            time_remaining_s=150,
            time_remaining_frac=0.5,
            yes_book=make_yes_book(0.50),
            no_book=make_no_book(0.50),
            yes_price=0.50,
            no_price=0.50,
            yes_bid=0.48,
            yes_ask=0.52,
            no_bid=0.48,
            no_ask=0.52,
        )
    }


class TestOrderValidation:
    def test_valid_buy_order(self, engine, active_market):
        order = Order(MARKET_5M_SLUG, Token.YES, Side.BUY, 10, 0.52)
        queued, rejected = engine.queue_orders(
            [order], 100, 10000.0, {}, active_market
        )
        assert len(queued) == 1
        assert len(rejected) == 0

    def test_reject_inactive_market(self, engine):
        order = Order("nonexistent-slug", Token.YES, Side.BUY, 10, 0.52)
        queued, rejected = engine.queue_orders([order], 100, 10000.0, {}, {})
        assert len(queued) == 0
        assert len(rejected) == 1
        assert "not active" in rejected[0].reason

    def test_reject_zero_size(self, engine, active_market):
        order = Order(MARKET_5M_SLUG, Token.YES, Side.BUY, 0, 0.52)
        _, rejected = engine.queue_orders([order], 100, 10000.0, {}, active_market)
        assert len(rejected) == 1
        assert "invalid size" in rejected[0].reason

    def test_reject_invalid_limit_price(self, engine, active_market):
        order = Order(MARKET_5M_SLUG, Token.YES, Side.BUY, 10, 1.5)
        _, rejected = engine.queue_orders([order], 100, 10000.0, {}, active_market)
        assert len(rejected) == 1
        assert "invalid limit price" in rejected[0].reason

    def test_reject_insufficient_cash(self, engine, active_market):
        order = Order(MARKET_5M_SLUG, Token.YES, Side.BUY, 100, 0.52)
        _, rejected = engine.queue_orders([order], 100, 5.0, {}, active_market)
        assert len(rejected) == 1
        assert "insufficient cash" in rejected[0].reason

    def test_reject_position_limit(self, engine, active_market):
        from backtester.portfolio import Position
        pos = Position(MARKET_5M_SLUG, yes_shares=490)
        order = Order(MARKET_5M_SLUG, Token.YES, Side.BUY, 20, 0.52)
        _, rejected = engine.queue_orders(
            [order], 100, 10000.0, {MARKET_5M_SLUG: pos}, active_market
        )
        assert len(rejected) == 1
        assert "position limit" in rejected[0].reason

    def test_reject_short_sell(self, engine, active_market):
        """Cannot sell shares you don't own."""
        order = Order(MARKET_5M_SLUG, Token.YES, Side.SELL, 10, 0.48)
        _, rejected = engine.queue_orders([order], 100, 10000.0, {}, active_market)
        assert len(rejected) == 1
        assert "cannot sell" in rejected[0].reason

    def test_sell_owned_shares(self, engine, active_market):
        """Can sell shares you own."""
        from backtester.portfolio import Position
        pos = Position(MARKET_5M_SLUG, yes_shares=50)
        order = Order(MARKET_5M_SLUG, Token.YES, Side.SELL, 10, 0.48)
        queued, rejected = engine.queue_orders(
            [order], 100, 10000.0, {MARKET_5M_SLUG: pos}, active_market
        )
        assert len(queued) == 1
        assert len(rejected) == 0


class TestPendingReservations:
    @pytest.mark.parametrize("token", [Token.YES, Token.NO])
    def test_batch_buys_reserve_position_capacity(self, engine, active_market, token):
        position = Position(MARKET_5M_SLUG, yes_shares=490, no_shares=490)
        orders = [Order(MARKET_5M_SLUG, token, Side.BUY, 10, 0.55) for _ in range(2)]

        queued, rejected = engine.queue_orders(
            orders, 100, 10000.0, {MARKET_5M_SLUG: position}, active_market
        )

        assert len(queued) == 1
        assert len(rejected) == 1
        assert "position limit" in rejected[0].reason

    @pytest.mark.parametrize("token", [Token.YES, Token.NO])
    def test_batch_sells_cannot_sell_the_same_shares_twice(self, engine, active_market, token):
        portfolio = Portfolio()
        position = portfolio.get_position(MARKET_5M_SLUG)
        position.yes_shares = position.no_shares = 50
        orders = [Order(MARKET_5M_SLUG, token, Side.SELL, size, 0.40) for size in (30, 30, 20)]

        queued, rejected = engine.queue_orders(
            orders, 100, portfolio.cash, portfolio.positions, active_market
        )

        assert [pending.order.size for pending in queued] == [30, 20]
        assert len(rejected) == 1
        assert "cannot sell" in rejected[0].reason
        fills = engine.execute_pending(101, active_market, {MARKET_5M_SLUG: 101})
        for fill in fills:
            portfolio.apply_fill(fill)
        assert sum(fill.size for fill in fills) == 50
        assert (position.yes_shares if token == Token.YES else position.no_shares) == 0

    @pytest.mark.parametrize("side", [Side.BUY, Side.SELL])
    def test_separate_queue_calls_preserve_share_reservations(self, engine, active_market, side):
        owned = 490 if side == Side.BUY else 10
        positions = {MARKET_5M_SLUG: Position(MARKET_5M_SLUG, yes_shares=owned)}
        order = Order(MARKET_5M_SLUG, Token.YES, side, 10, 0.55)
        engine.queue_orders([order], 100, 10000.0, positions, active_market)

        queued, rejected = engine.queue_orders([order], 100, 10000.0, positions, active_market)

        assert queued == []
        assert len(rejected) == 1

    def test_yes_and_no_have_separate_capacity(self, engine, active_market):
        orders = [Order(MARKET_5M_SLUG, token, Side.BUY, 500, 0.55) for token in Token]
        queued, rejected = engine.queue_orders(orders, 100, 10000.0, {}, active_market)

        assert len(queued) == 2
        assert rejected == []

    def test_pending_sell_does_not_free_buy_capacity(self, engine, active_market):
        positions = {MARKET_5M_SLUG: Position(MARKET_5M_SLUG, yes_shares=500)}
        orders = [
            Order(MARKET_5M_SLUG, Token.YES, Side.SELL, 20, 0.60),
            Order(MARKET_5M_SLUG, Token.YES, Side.BUY, 20, 0.55),
        ]
        queued, rejected = engine.queue_orders(orders, 100, 10000.0, positions, active_market)

        assert [pending.order.side for pending in queued] == [Side.SELL]
        assert len(rejected) == 1
        assert "position limit" in rejected[0].reason

    def test_pending_buy_does_not_authorize_a_sell(self, engine, active_market):
        orders = [
            Order(MARKET_5M_SLUG, Token.YES, Side.BUY, 20, 0.50),
            Order(MARKET_5M_SLUG, Token.YES, Side.SELL, 20, 0.40),
        ]
        queued, rejected = engine.queue_orders(orders, 100, 10000.0, {}, active_market)

        assert [pending.order.side for pending in queued] == [Side.BUY]
        assert len(rejected) == 1
        assert "cannot sell" in rejected[0].reason

    def test_separate_queue_calls_preserve_cash_reservations(self, engine, active_market):
        order = Order(MARKET_5M_SLUG, Token.YES, Side.BUY, 10, 0.55)
        engine.queue_orders([order], 100, 10.0, {}, active_market)
        queued, rejected = engine.queue_orders([order], 100, 10.0, {}, active_market)

        assert queued == []
        assert len(rejected) == 1
        assert "insufficient cash" in rejected[0].reason

    def test_rejected_execution_releases_share_and_cash_reservations(self, engine, active_market):
        engine.queue_orders(
            [Order(MARKET_5M_SLUG, Token.YES, Side.BUY, 500, 0.50)],
            100, 300.0, {}, active_market,
        )
        assert engine.execute_pending(101, active_market, {MARKET_5M_SLUG: 101}) == []

        queued, rejected = engine.queue_orders(
            [Order(MARKET_5M_SLUG, Token.YES, Side.BUY, 500, 0.55)],
            101, 300.0, {}, active_market,
        )

        assert len(queued) == 1
        assert rejected == []

    def test_partial_fill_releases_unfilled_reservation(self, engine, active_market):
        portfolio = Portfolio()
        engine.queue_orders(
            [Order(MARKET_5M_SLUG, Token.YES, Side.BUY, 500, 0.52)],
            100, portfolio.cash, portfolio.positions, active_market,
        )
        fills = engine.execute_pending(101, active_market, {MARKET_5M_SLUG: 101})
        assert len(fills) == 1
        assert fills[0].size == 100
        portfolio.apply_fill(fills[0])

        queued, rejected = engine.queue_orders(
            [Order(MARKET_5M_SLUG, Token.YES, Side.BUY, 400, 0.55)],
            101, portfolio.cash, portfolio.positions, active_market,
        )

        assert len(queued) == 1
        assert rejected == []


class TestExecution:
    def test_latency(self, engine, active_market):
        """Orders at tick T should not fill until tick T+1."""
        order = Order(MARKET_5M_SLUG, Token.YES, Side.BUY, 10, 0.55)
        engine.queue_orders([order], 100, 10000.0, {}, active_market)

        # At tick 100 (same tick): nothing should fill
        fills = engine.execute_pending(100, active_market, {MARKET_5M_SLUG: 100})
        assert len(fills) == 0

        # At tick 101 (next tick): should fill
        fills = engine.execute_pending(101, active_market, {MARKET_5M_SLUG: 101})
        assert len(fills) == 1

    def test_walk_the_book(self, engine, active_market):
        """Buy order should fill at ask prices."""
        order = Order(MARKET_5M_SLUG, Token.YES, Side.BUY, 10, 0.60)
        engine.queue_orders([order], 100, 10000.0, {}, active_market)

        fills = engine.execute_pending(101, active_market, {MARKET_5M_SLUG: 101})
        assert len(fills) == 1
        fill = fills[0]
        assert fill.size == 10
        assert fill.avg_price == 0.52  # first ask level
        assert fill.token == Token.YES

    def test_walk_multiple_levels(self, engine, active_market):
        """Large order should consume multiple book levels."""
        order = Order(MARKET_5M_SLUG, Token.YES, Side.BUY, 250, 0.60)
        engine.queue_orders([order], 100, 10000.0, {}, active_market)

        fills = engine.execute_pending(101, active_market, {MARKET_5M_SLUG: 101})
        assert len(fills) == 1
        fill = fills[0]
        assert fill.size == 250
        # Should consume first level (100 @ 0.52) + part of second (150 @ 0.54)
        expected_cost = 100 * 0.52 + 150 * 0.54
        assert abs(fill.cost - expected_cost) < 0.01

    def test_stale_book_rejected(self, engine, active_market):
        """Orders against stale books should be rejected."""
        order = Order(MARKET_5M_SLUG, Token.YES, Side.BUY, 10, 0.55)
        engine.queue_orders([order], 100, 10000.0, {}, active_market)

        # Book timestamp is old
        fills = engine.execute_pending(
            101, active_market, {MARKET_5M_SLUG: 101 - MAX_BOOK_STALENESS_S - 1}
        )
        assert len(fills) == 0
        assert engine.total_rejected > 0

    def test_book_depletion(self, engine, active_market):
        """Two orders on same tick should see depleted book."""
        order1 = Order(MARKET_5M_SLUG, Token.YES, Side.BUY, 80, 0.55)
        order2 = Order(MARKET_5M_SLUG, Token.YES, Side.BUY, 80, 0.55)
        engine.queue_orders([order1], 100, 10000.0, {}, active_market)
        engine.queue_orders([order2], 100, 9000.0, {}, active_market)

        fills = engine.execute_pending(101, active_market, {MARKET_5M_SLUG: 101})
        assert len(fills) == 2

        # First fill: 80 from level 1 (100 @ 0.52)
        assert fills[0].size == 80
        assert fills[0].avg_price == 0.52

        # Second fill: remaining 20 from level 1 + 60 from level 2
        assert fills[1].size == 80
        # 20 @ 0.52 + 60 @ 0.54
        expected_cost = 20 * 0.52 + 60 * 0.54
        assert abs(fills[1].cost - expected_cost) < 0.01

    def test_limit_price_respected(self, engine, active_market):
        """Should not fill above limit price."""
        order = Order(MARKET_5M_SLUG, Token.YES, Side.BUY, 200, 0.52)
        engine.queue_orders([order], 100, 10000.0, {}, active_market)

        fills = engine.execute_pending(101, active_market, {MARKET_5M_SLUG: 101})
        assert len(fills) == 1
        # Only fills at first level (0.52), not second (0.54)
        assert fills[0].size == 100
        assert fills[0].avg_price == 0.52

    def test_sell_fills_at_bids(self, engine, active_market):
        """Sell orders should fill at bid prices."""
        from backtester.portfolio import Position
        pos = Position(MARKET_5M_SLUG, yes_shares=50)
        order = Order(MARKET_5M_SLUG, Token.YES, Side.SELL, 10, 0.40)
        engine.queue_orders(
            [order], 100, 10000.0, {MARKET_5M_SLUG: pos}, active_market
        )

        fills = engine.execute_pending(101, active_market, {MARKET_5M_SLUG: 101})
        assert len(fills) == 1
        assert fills[0].avg_price == 0.48  # best bid

