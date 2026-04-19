"""
Backtest Engine - Main tick loop orchestrator.

Coordinates data loading, market management, execution, and portfolio tracking
to run a strategy over historical data.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .data_loader import BacktestData, TickData
from .execution import ExecutionEngine
from .market_manager import MarketManager
from .portfolio import Portfolio, PortfolioSnapshot
from .strategy import (
    BaseStrategy,
    Fill,
    MarketLifecycle,
    MarketState,
    MarketView,
    Order,
    PositionView,
    Settlement,
)

logger = logging.getLogger(__name__)


@dataclass
class ForecastRecord:
    """One tick's forecast and market state for Brier scoring."""
    timestamp: int
    market_slug: str
    model_forecast: float   # strategy's P(YES)
    market_yes_price: float # market's yes_price at this tick


@dataclass
class BacktestResult:
    """Complete result of a backtest run."""
    strategy_name: str
    start_ts: int
    end_ts: int
    starting_cash: float
    final_cash: float
    final_portfolio_value: float
    total_pnl: float
    total_trades: int
    total_settlements: int
    total_rejected: int
    portfolio_snapshots: list[PortfolioSnapshot]
    fills: list[Fill]
    settlements: list[Settlement]
    elapsed_seconds: float
    forecast_records: list[ForecastRecord] = field(default_factory=list)


class BacktestEngine:
    """
    Main backtest engine that drives the tick loop.

    Per tick:
    1. Settle expired markets -> update portfolio, notify strategy
    2. Update active markets from MarketManager
    3. Execute pending orders (queued from previous tick)
    4. Build MarketState snapshot
    5. Call strategy.on_tick(state) -> get new orders
    6. Validate and queue orders for next tick
    7. Record portfolio snapshot
    """

    def __init__(
        self,
        data: BacktestData,
        strategy: BaseStrategy,
        starting_cash: float = 10_000.0,
        snapshot_interval: int = 1,  # record portfolio every N ticks
        tick_step: int = 1,  # process every Nth tick (1=full fidelity)
    ):
        self.data = data
        self.strategy = strategy
        self.starting_cash = starting_cash
        self.snapshot_interval = snapshot_interval
        self.tick_step = max(1, tick_step)

        # Initialize components - deep-copy lifecycles so data can be reused
        self.portfolio = Portfolio(starting_cash=starting_cash)
        fresh_lifecycles = [
            MarketLifecycle(
                market_slug=lc.market_slug,
                interval=lc.interval,
                start_ts=lc.start_ts,
                end_ts=lc.end_ts,
            )
            for lc in data.lifecycles
        ]
        self.market_manager = MarketManager(
            lifecycles=fresh_lifecycles,
            settlements=data.settlements,
        )
        self.execution = ExecutionEngine()

        # Results
        self.snapshots: list[PortfolioSnapshot] = []
        self.all_fills: list[Fill] = []
        self.all_settlements: list[Settlement] = []
        self.forecast_records: list[ForecastRecord] = []

    def run(self) -> BacktestResult:
        """Run the complete backtest and return results."""
        if not self.data.timeline:
            logger.warning("No timeline data to backtest")
            return self._empty_result()

        strategy_name = type(self.strategy).__name__
        logger.info(
            f"Starting backtest: {strategy_name} | "
            f"{len(self.data.timeline)} ticks | "
            f"${self.starting_cash:,.0f} starting cash | "
            f"{len(self.data.lifecycles)} markets"
        )

        t0 = time.time()
        total_ticks = len(self.data.timeline)

        # Precompute market boundary timestamps for tick_step > 1
        # Always process ticks at market start/end to ensure correct settlements
        boundary_ticks: set[int] = set()
        if self.tick_step > 1:
            for lc in self.data.lifecycles:
                boundary_ticks.add(lc.start_ts)
                boundary_ticks.add(lc.end_ts)

        processed = 0
        for i, tick in enumerate(self.data.timeline):
            # Skip non-boundary ticks based on tick_step
            if self.tick_step > 1 and i % self.tick_step != 0:
                if tick.ts_sec not in boundary_ticks:
                    continue

            self._process_tick(tick)
            processed += 1

            # Progress logging
            if processed % 3600 == 0 or i == total_ticks - 1:
                pv = self.portfolio.mark_to_market(
                    self.market_manager.update(tick.ts_sec)
                )
                logger.info(
                    f"Tick {processed} (idx {i}/{total_ticks}) | "
                    f"ts={tick.ts_sec} | "
                    f"portfolio=${pv:,.2f} | "
                    f"trades={self.portfolio.total_fills} | "
                    f"settled={len(self.all_settlements)}"
                )

        elapsed = time.time() - t0

        # Final valuation
        final_views = {}
        if self.data.timeline:
            last_tick = self.data.timeline[-1]
            final_views = self.market_manager.update(last_tick.ts_sec)
            final_views = self.market_manager.enrich_views(final_views, last_tick)

        final_value = self.portfolio.mark_to_market(final_views)

        result = BacktestResult(
            strategy_name=strategy_name,
            start_ts=self.data.start_ts,
            end_ts=self.data.end_ts,
            starting_cash=self.starting_cash,
            final_cash=self.portfolio.cash,
            final_portfolio_value=final_value,
            total_pnl=final_value - self.starting_cash,
            total_trades=self.portfolio.total_fills,
            total_settlements=len(self.all_settlements),
            total_rejected=self.execution.total_rejected,
            portfolio_snapshots=self.snapshots,
            fills=self.all_fills,
            settlements=self.all_settlements,
            elapsed_seconds=elapsed,
            forecast_records=self.forecast_records,
        )

        logger.info(
            f"Backtest complete: {strategy_name} | "
            f"{elapsed:.1f}s | "
            f"P&L=${result.total_pnl:+,.2f} | "
            f"trades={result.total_trades} | "
            f"rejected={result.total_rejected}"
        )

        return result

    def _process_tick(self, tick: TickData) -> None:
        """Process a single tick of the backtest."""
        ts = tick.ts_sec

        # 1. Update market states (settles expired markets)
        active_views = self.market_manager.update(ts)

        # 2. Handle settlements
        settled = self.market_manager.get_settled_this_tick()
        for settlement in settled:
            self.portfolio.apply_settlement(settlement)
            self.all_settlements.append(settlement)
            try:
                self.strategy.on_settlement(settlement)
            except Exception as e:
                logger.warning(f"Strategy on_settlement error: {e}")

        # 3. Enrich market views with current tick data
        enriched_views = self.market_manager.enrich_views(active_views, tick)

        # 4. Execute pending orders from previous tick
        fills = self.execution.execute_pending(
            current_tick=ts,
            market_views=enriched_views,
            book_timestamps=tick.book_timestamps,
        )

        for fill in fills:
            self.portfolio.apply_fill(fill)
            self.all_fills.append(fill)
            try:
                self.strategy.on_fill(fill)
            except Exception as e:
                logger.warning(f"Strategy on_fill error: {e}")

        # 5. Build MarketState for strategy
        total_value = self.portfolio.mark_to_market(enriched_views)
        state = MarketState(
            timestamp=ts,
            timestamp_utc=datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
            markets=enriched_views,
            btc_mid=tick.btc_mid,
            btc_spread=tick.btc_spread,
            eth_mid=tick.eth_mid,
            eth_spread=tick.eth_spread,
            sol_mid=tick.sol_mid,
            sol_spread=tick.sol_spread,
            chainlink_btc=tick.chainlink_btc,
            chainlink_eth=tick.chainlink_eth,
            chainlink_sol=tick.chainlink_sol,
            cash=self.portfolio.cash,
            positions=self.portfolio.get_position_views(),
            total_portfolio_value=total_value,
        )

        # 6. Call strategy
        try:
            orders = self.strategy.on_tick(state)
        except Exception as e:
            logger.warning(f"Strategy on_tick error: {e}")
            orders = []

        if orders is None:
            orders = []

        # 6b. Collect forecasts for Brier scoring
        try:
            forecasts = self.strategy.get_forecasts(state)
        except Exception:
            forecasts = {}
        if forecasts:
            for slug, prob in forecasts.items():
                if slug in enriched_views:
                    self.forecast_records.append(ForecastRecord(
                        timestamp=ts,
                        market_slug=slug,
                        model_forecast=prob,
                        market_yes_price=enriched_views[slug].yes_price,
                    ))

        # 7. Validate and queue orders
        if orders:
            self.execution.queue_orders(
                orders=orders,
                current_tick=ts,
                cash=self.portfolio.cash,
                positions={
                    slug: self.portfolio.get_position(slug)
                    for slug in enriched_views
                },
                active_markets=enriched_views,
            )

        # 8. Record portfolio snapshot
        if ts % self.snapshot_interval == 0:
            snap = self.portfolio.snapshot(ts, enriched_views)
            self.snapshots.append(snap)

    def _empty_result(self) -> BacktestResult:
        return BacktestResult(
            strategy_name=type(self.strategy).__name__,
            start_ts=0,
            end_ts=0,
            starting_cash=self.starting_cash,
            final_cash=self.starting_cash,
            final_portfolio_value=self.starting_cash,
            total_pnl=0.0,
            total_trades=0,
            total_settlements=0,
            total_rejected=0,
            portfolio_snapshots=[],
            fills=[],
            settlements=[],
            elapsed_seconds=0.0,
        )
