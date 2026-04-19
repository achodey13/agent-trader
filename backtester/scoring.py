"""
Scoring - P&L, Sharpe, drawdown, and competition score computation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .engine import BacktestResult
from .portfolio import PortfolioSnapshot


@dataclass
class ScoreCard:
    """Complete scoring metrics for a backtest run."""
    strategy_name: str
    total_pnl: float
    sharpe_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    win_rate: float
    total_trades: int
    total_settlements: int
    avg_trade_pnl: float
    final_portfolio_value: float
    starting_cash: float
    return_pct: float
    competition_score: float  # primary: total_pnl on OOS period
    avg_implied_prob: float = 0.0  # mean entry price for BUY fills = avg implied P(win)


def compute_score(result: BacktestResult) -> ScoreCard:
    """Compute all scoring metrics from a backtest result."""
    # P&L
    total_pnl = result.total_pnl
    return_pct = (total_pnl / result.starting_cash) * 100 if result.starting_cash > 0 else 0.0

    # Sharpe ratio from portfolio value time series
    sharpe = _compute_sharpe(result.portfolio_snapshots)

    # Max drawdown
    max_dd, max_dd_pct = _compute_max_drawdown(result.portfolio_snapshots)

    # Win rate from settlements
    wins = 0
    for settlement in result.settlements:
        # Check if we had a position that paid out
        for fill in result.fills:
            if fill.market_slug == settlement.market_slug:
                # Simplified: count settlement as win if we had shares on the right side
                if fill.token == settlement.outcome:
                    wins += 1
                    break

    settlements_with_positions = len(set(
        f.market_slug for f in result.fills
    ) & set(s.market_slug for s in result.settlements))

    win_rate = wins / settlements_with_positions if settlements_with_positions > 0 else 0.0

    # Average trade P&L
    avg_trade_pnl = total_pnl / result.total_trades if result.total_trades > 0 else 0.0

    # Average implied probability across BUY fills.
    # When buying YES at price P, P is the implied P(YES wins) = implied P(this trade wins).
    # When buying NO at price (1 - yes_bid), that price is the implied P(NO wins).
    # In both cases, fill.avg_price IS the implied probability of winning.
    from .strategy import Side
    buy_fills = [f for f in result.fills if f.side == Side.BUY and f.avg_price > 0]
    avg_implied = sum(f.avg_price for f in buy_fills) / len(buy_fills) if buy_fills else 0.0

    return ScoreCard(
        strategy_name=result.strategy_name,
        total_pnl=total_pnl,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        max_drawdown_pct=max_dd_pct,
        win_rate=win_rate,
        total_trades=result.total_trades,
        total_settlements=result.total_settlements,
        avg_trade_pnl=avg_trade_pnl,
        final_portfolio_value=result.final_portfolio_value,
        starting_cash=result.starting_cash,
        return_pct=return_pct,
        competition_score=total_pnl,  # Primary: total P&L on OOS
        avg_implied_prob=avg_implied,
    )


def _compute_sharpe(snapshots: list[PortfolioSnapshot], periods_per_day: int = 86400) -> float:
    """
    Compute annualized Sharpe ratio from portfolio snapshots.

    Annualizes based on the actual snapshot interval (derived from timestamps),
    not assuming 1-second ticks. This avoids inflating Sharpe from highly
    autocorrelated high-frequency observations.
    """
    if len(snapshots) < 2:
        return 0.0

    values = [s.total_value for s in snapshots]
    returns = []
    for i in range(1, len(values)):
        if values[i - 1] > 0:
            returns.append((values[i] - values[i - 1]) / values[i - 1])

    if not returns:
        return 0.0

    mean_ret = sum(returns) / len(returns)
    if len(returns) < 2:
        return 0.0

    variance = sum((r - mean_ret) ** 2 for r in returns) / (len(returns) - 1)
    std_ret = math.sqrt(variance)

    if std_ret < 1e-12:
        return 0.0

    # Compute actual snapshot interval from timestamps
    total_duration_s = snapshots[-1].timestamp - snapshots[0].timestamp
    if total_duration_s <= 0:
        return 0.0
    snapshot_interval_s = total_duration_s / len(returns)

    # Annualize: periods_per_year = seconds_per_year / snapshot_interval
    seconds_per_year = 365 * 86400
    periods_per_year = seconds_per_year / max(snapshot_interval_s, 1)
    annualization = math.sqrt(periods_per_year)

    return (mean_ret / std_ret) * annualization


def _compute_max_drawdown(
    snapshots: list[PortfolioSnapshot],
) -> tuple[float, float]:
    """
    Compute maximum drawdown in $ and % terms.

    Returns (max_dd_dollars, max_dd_percent).
    """
    if not snapshots:
        return 0.0, 0.0

    values = [s.total_value for s in snapshots]
    peak = values[0]
    max_dd = 0.0
    max_dd_pct = 0.0

    for v in values:
        if v > peak:
            peak = v
        dd = peak - v
        if dd > max_dd:
            max_dd = dd
            max_dd_pct = dd / peak if peak > 0 else 0.0

    return max_dd, max_dd_pct * 100
