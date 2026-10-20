"""Submitted order values must survive later edits by the strategy."""

from dataclasses import replace

import pytest

from backtester.engine import BacktestEngine
from backtester.execution import ExecutionEngine
from backtester.portfolio import Portfolio, Position
from backtester.strategy import BaseStrategy, MarketLifecycle, Order, Settlement, Side, Token
from tests.conftest import BASE_TS, MARKET_5M_SLUG


@pytest.fixture
def execution():
    return ExecutionEngine()


@pytest.fixture
def active_market(sample_market_view):
    return {MARKET_5M_SLUG: sample_market_view}


@pytest.mark.parametrize("token", [Token.YES, Token.NO])
@pytest.mark.parametrize("side", [Side.BUY, Side.SELL])
def test_submitted_size_survives_input_mutation(execution, active_market, token, side):
    portfolio = Portfolio(starting_cash=5.5)
    position = portfolio.get_position(MARKET_5M_SLUG)
    position.yes_shares = position.no_shares = 490 if side == Side.BUY else 10
    limit = 0.55 if side == Side.BUY else 0.40
    order = Order(MARKET_5M_SLUG, token, side, 10, limit)
    execution.queue_orders([order], 100, portfolio.cash, portfolio.positions, active_market)

    order.size = 1000
    fills = execution.execute_pending(101, active_market, {MARKET_5M_SLUG: 101})

    assert len(fills) == 1
    assert fills[0].size == 10
    assert fills[0].order == Order(MARKET_5M_SLUG, token, side, 10, limit)
    portfolio.apply_fill(fills[0])
    assert portfolio.cash >= 0
    shares = position.yes_shares if token == Token.YES else position.no_shares
    assert shares == (500 if side == Side.BUY else 0)
    assert order.size == 1000  # The public Order API remains mutable.


@pytest.mark.parametrize("side,limit,changed_limit,should_fill", [
    (Side.BUY, 0.50, None, False),
    (Side.SELL, 0.50, None, False),
    (Side.BUY, None, 0.01, True),
    (Side.SELL, None, 0.99, True),
])
def test_submitted_limit_survives_input_mutation(
    execution, active_market, side, limit, changed_limit, should_fill
):
    positions = {MARKET_5M_SLUG: Position(MARKET_5M_SLUG, yes_shares=10)}
    order = Order(MARKET_5M_SLUG, Token.YES, side, 10, limit)
    execution.queue_orders([order], 100, 10.0, positions, active_market)

    order.limit_price = changed_limit
    fills = execution.execute_pending(101, active_market, {MARKET_5M_SLUG: 101})

    assert len(fills) == int(should_fill)
    if fills:
        assert fills[0].order.limit_price == limit
    else:
        assert execution.total_rejected == 1
        assert "no liquidity" in execution.rejected_orders[0].reason


@pytest.mark.parametrize("field,value", [
    ("market_slug", "eth-updown-5m-1700000000"),
    ("token", Token.NO),
    ("side", Side.SELL),
])
def test_submitted_routing_survives_input_mutation(execution, active_market, field, value):
    other_slug = "eth-updown-5m-1700000000"
    active_market[other_slug] = replace(active_market[MARKET_5M_SLUG], market_slug=other_slug)
    order = Order(MARKET_5M_SLUG, Token.YES, Side.BUY, 10)
    execution.queue_orders([order], 100, 10.0, {}, active_market)

    setattr(order, field, value)
    fills = execution.execute_pending(
        101, active_market, {MARKET_5M_SLUG: 101, other_slug: 101}
    )

    assert len(fills) == 1
    assert fills[0].market_slug == MARKET_5M_SLUG
    assert fills[0].token == Token.YES
    assert fills[0].side == Side.BUY
    assert fills[0].order == Order(MARKET_5M_SLUG, Token.YES, Side.BUY, 10)


@pytest.mark.parametrize("constraint,side,owned,cash,reason", [
    ("cash", Side.BUY, 0, 5.5, "insufficient cash"),
    ("position", Side.BUY, 490, 1000.0, "position limit"),
    ("holdings", Side.SELL, 10, 1000.0, "cannot sell"),
])
def test_input_mutation_cannot_release_pending_reservations(
    execution, active_market, constraint, side, owned, cash, reason
):
    positions = {MARKET_5M_SLUG: Position(MARKET_5M_SLUG, yes_shares=owned)}
    limit = 0.55 if side == Side.BUY else 0.40
    order = Order(MARKET_5M_SLUG, Token.YES, side, 10, limit)
    execution.queue_orders([order], 100, cash, positions, active_market)

    order.size = 0
    queued, rejected = execution.queue_orders(
        [Order(MARKET_5M_SLUG, Token.YES, side, 10, limit)],
        100, cash, positions, active_market,
    )

    assert queued == []
    assert len(rejected) == 1
    assert reason in rejected[0].reason


def test_reusing_input_preserves_each_submission_and_fill_history(execution, active_market):
    portfolio = Portfolio()
    order = Order(MARKET_5M_SLUG, Token.YES, Side.BUY, 10, 0.55)
    execution.queue_orders([order], 100, portfolio.cash, portfolio.positions, active_market)

    order.size, order.limit_price = 20, 0.60
    first = execution.execute_pending(101, active_market, {MARKET_5M_SLUG: 101})[0]
    portfolio.apply_fill(first)
    execution.queue_orders([order], 101, portfolio.cash, portfolio.positions, active_market)

    order.size, order.limit_price = 30, 0.65
    second = execution.execute_pending(102, active_market, {MARKET_5M_SLUG: 102})[0]
    portfolio.apply_fill(second)
    order.market_slug, order.token, order.side = "another-market", Token.NO, Side.SELL

    assert [fill.size for fill in portfolio.fill_history] == [10, 20]
    assert first.order == Order(MARKET_5M_SLUG, Token.YES, Side.BUY, 10, 0.55)
    assert second.order == Order(MARKET_5M_SLUG, Token.YES, Side.BUY, 20, 0.60)


def test_execution_rejection_preserves_submitted_values(execution, active_market):
    order = Order(MARKET_5M_SLUG, Token.YES, Side.BUY, 10, 0.50)
    execution.queue_orders([order], 100, 10.0, {}, active_market)

    order.size, order.limit_price = 1000, None
    assert execution.execute_pending(101, active_market, {MARKET_5M_SLUG: 101}) == []
    order.size, order.limit_price = 1, 0.99

    assert execution.rejected_orders[0].order == Order(
        MARKET_5M_SLUG, Token.YES, Side.BUY, 10, 0.50
    )


def test_settlement_callback_cannot_rewrite_a_pending_order(synthetic_backtest_data):
    trigger = Settlement(
        market_slug="eth-updown-5m-snapshot",
        interval="5m",
        outcome=Token.YES,
        start_ts=BASE_TS - 299,
        end_ts=BASE_TS + 1,
        chainlink_open=3400,
        chainlink_close=3401,
    )
    data = replace(
        synthetic_backtest_data,
        lifecycles=[
            *synthetic_backtest_data.lifecycles,
            MarketLifecycle(trigger.market_slug, trigger.interval, trigger.start_ts, trigger.end_ts),
        ],
        settlements={**synthetic_backtest_data.settlements, trigger.market_slug: trigger},
    )

    class MutatingStrategy(BaseStrategy):
        def __init__(self):
            self.order = None

        def on_tick(self, state):
            if self.order is None:
                self.order = Order(MARKET_5M_SLUG, Token.YES, Side.BUY, 10, 0.55)
                return [self.order]
            return []

        def on_settlement(self, settlement):
            if settlement.market_slug == trigger.market_slug:
                self.order.size = 1000
                self.order.limit_price = 0.99

    strategy = MutatingStrategy()
    result = BacktestEngine(data, strategy, starting_cash=5.5).run()

    assert strategy.order.size == 1000
    assert result.total_trades == 1
    assert result.total_rejected == 0
    assert result.fills[0].size == 10
    assert result.fills[0].timestamp == BASE_TS + 1
    assert result.fills[0].order == Order(MARKET_5M_SLUG, Token.YES, Side.BUY, 10, 0.55)
    assert all(snapshot.cash >= 0 for snapshot in result.portfolio_snapshots)
    assert all(
        position.yes_shares <= 500
        for snapshot in result.portfolio_snapshots
        for position in snapshot.positions.values()
    )
