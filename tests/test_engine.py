"""Tests for the backtest engine (integration tests)."""

import pytest

from backtester.engine import BacktestEngine
from backtester.strategy import BaseStrategy, Fill, MarketState, Order, Settlement, Side, Token
from tests.conftest import MARKET_5M_SLUG, MARKET_15M_SLUG


class DoNothingStrategy(BaseStrategy):
    """Strategy that does nothing - baseline."""
    def on_tick(self, state: MarketState) -> list[Order]:
        return []


class BuyOnceStrategy(BaseStrategy):
    """Buy YES on the first market once, then hold."""
    def __init__(self):
        self.bought = False
        self.fills = []
        self.settlements = []

    def on_tick(self, state: MarketState) -> list[Order]:
        if self.bought or not state.markets:
            return []
        slug = next(iter(state.markets))
        market = state.markets[slug]
        if market.yes_ask > 0:
            self.bought = True
            # Use limit slightly above ask to account for 1-tick book movement
            return [Order(slug, Token.YES, Side.BUY, 50, market.yes_ask + 0.01)]
        return []

    def on_fill(self, fill: Fill):
        self.fills.append(fill)

    def on_settlement(self, settlement: Settlement):
        self.settlements.append(settlement)


class TestBacktestEngine:
    def test_do_nothing(self, synthetic_backtest_data):
        """Do-nothing strategy should end with starting cash."""
        strategy = DoNothingStrategy()
        engine = BacktestEngine(
            data=synthetic_backtest_data,
            strategy=strategy,
            starting_cash=10_000.0,
        )
        result = engine.run()

        assert result.total_pnl == 0.0
        assert result.final_portfolio_value == 10_000.0
        assert result.total_trades == 0
        assert result.total_settlements == 2  # both markets settle

    def test_buy_once_fills(self, synthetic_backtest_data):
        """Buy-once strategy should get exactly one fill."""
        strategy = BuyOnceStrategy()
        engine = BacktestEngine(
            data=synthetic_backtest_data,
            strategy=strategy,
        )
        result = engine.run()

        assert result.total_trades == 1
        assert len(strategy.fills) == 1
        assert strategy.fills[0].token == Token.YES
        assert strategy.fills[0].size == 50

    def test_settlements_trigger_callbacks(self, synthetic_backtest_data):
        """Strategy should receive settlement callbacks."""
        strategy = BuyOnceStrategy()
        engine = BacktestEngine(
            data=synthetic_backtest_data,
            strategy=strategy,
        )
        result = engine.run()

        assert len(strategy.settlements) == 2
        slugs = {s.market_slug for s in strategy.settlements}
        assert MARKET_5M_SLUG in slugs
        assert MARKET_15M_SLUG in slugs

    def test_pnl_from_yes_win(self, synthetic_backtest_data):
        """Buying YES on a market that settles YES should profit."""
        strategy = BuyOnceStrategy()
        engine = BacktestEngine(
            data=synthetic_backtest_data,
            strategy=strategy,
        )
        result = engine.run()

        # 5m market settles YES, strategy bought YES -> should profit
        # 50 shares * ($1 payout - avg_price) = positive P&L
        assert result.total_pnl > 0

    def test_portfolio_snapshots_recorded(self, synthetic_backtest_data):
        """Engine should record portfolio snapshots."""
        strategy = DoNothingStrategy()
        engine = BacktestEngine(
            data=synthetic_backtest_data,
            strategy=strategy,
            snapshot_interval=60,
        )
        result = engine.run()

        assert len(result.portfolio_snapshots) > 0
        # All snapshots should have consistent value for do-nothing
        for snap in result.portfolio_snapshots:
            assert snap.total_value == pytest.approx(10_000.0)

    def test_result_timing(self, synthetic_backtest_data):
        """Result should track correct time range."""
        strategy = DoNothingStrategy()
        engine = BacktestEngine(
            data=synthetic_backtest_data,
            strategy=strategy,
        )
        result = engine.run()

        assert result.start_ts == synthetic_backtest_data.start_ts
        assert result.end_ts == synthetic_backtest_data.end_ts
        assert result.elapsed_seconds > 0


class ErrorStrategy(BaseStrategy):
    """Strategy that raises errors - should not crash engine."""
    def on_tick(self, state: MarketState) -> list[Order]:
        raise ValueError("on_tick error")


class TestEngineErrorHandling:
    def test_strategy_error_no_crash(self, synthetic_backtest_data):
        """Engine should handle strategy errors gracefully."""
        strategy = ErrorStrategy()
        engine = BacktestEngine(
            data=synthetic_backtest_data,
            strategy=strategy,
        )
        # Should complete without raising
        result = engine.run()
        assert result.total_trades == 0
        assert result.final_portfolio_value == 10_000.0
