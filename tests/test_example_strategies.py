"""Tests for example strategies - integration tests running on synthetic data."""

import pytest

from backtester.engine import BacktestEngine


class TestBuyAndHold:
    def test_runs_and_trades(self, synthetic_backtest_data):
        from backtester.examples.buy_and_hold import BuyAndHold

        strategy = BuyAndHold(size=50)
        engine = BacktestEngine(
            data=synthetic_backtest_data,
            strategy=strategy,
        )
        result = engine.run()

        assert result.total_trades > 0
        assert result.total_settlements == 2
        # Should have bought on both markets
        traded_slugs = {f.market_slug for f in result.fills}
        assert len(traded_slugs) >= 1


class TestFairValue:
    def test_runs_without_error(self, synthetic_backtest_data):
        from backtester.examples.fair_value import FairValue

        strategy = FairValue(vol_15m=0.005, threshold=0.05, size=30)
        engine = BacktestEngine(
            data=synthetic_backtest_data,
            strategy=strategy,
        )
        result = engine.run()

        # Fair value should make some trades when price diverges
        assert result.total_settlements == 2
        assert result.elapsed_seconds > 0


class TestArbScanner:
    def test_runs_without_error(self, synthetic_backtest_data):
        from backtester.examples.arb_scanner import ArbScanner

        strategy = ArbScanner(min_edge=0.02, size=50)
        engine = BacktestEngine(
            data=synthetic_backtest_data,
            strategy=strategy,
        )
        result = engine.run()

        # Arb scanner may or may not find arbs in synthetic data
        assert result.total_settlements == 2


class TestRandomStrategy:
    def test_runs_deterministically(self, synthetic_backtest_data):
        from backtester.examples.random_strategy import RandomStrategy

        # Run twice with same seed
        strategy1 = RandomStrategy(trade_prob=0.05, size=10, seed=42)
        engine1 = BacktestEngine(data=synthetic_backtest_data, strategy=strategy1)
        result1 = engine1.run()

        strategy2 = RandomStrategy(trade_prob=0.05, size=10, seed=42)
        engine2 = BacktestEngine(data=synthetic_backtest_data, strategy=strategy2)
        result2 = engine2.run()

        assert result1.total_trades == result2.total_trades
        assert result1.total_pnl == pytest.approx(result2.total_pnl)

    def test_makes_some_trades(self, synthetic_backtest_data):
        from backtester.examples.random_strategy import RandomStrategy

        strategy = RandomStrategy(trade_prob=0.1, size=10, seed=123)
        engine = BacktestEngine(data=synthetic_backtest_data, strategy=strategy)
        result = engine.run()

        # With 10% probability over 1200 ticks, should have some trades
        assert result.total_trades > 0
