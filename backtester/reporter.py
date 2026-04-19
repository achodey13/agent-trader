"""
Reporter - JSON/CSV/terminal output for backtest results.
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from .engine import BacktestResult
from .scoring import ScoreCard, compute_score

logger = logging.getLogger(__name__)


def print_report(result: BacktestResult) -> ScoreCard:
    """Print a formatted terminal report and return the ScoreCard."""
    score = compute_score(result)

    duration_s = result.end_ts - result.start_ts if result.end_ts > result.start_ts else 0
    duration_h = duration_s / 3600

    print("\n" + "=" * 60)
    print(f"  BACKTEST REPORT: {score.strategy_name}")
    print("=" * 60)
    print(f"  Period:     {_fmt_ts(result.start_ts)} -> {_fmt_ts(result.end_ts)}")
    print(f"  Duration:   {duration_h:.1f} hours ({duration_s:,} seconds)")
    print(f"  Runtime:    {result.elapsed_seconds:.1f}s")
    print()
    print(f"  Starting:   ${score.starting_cash:>10,.2f}")
    print(f"  Final:      ${score.final_portfolio_value:>10,.2f}")
    print(f"  P&L:        ${score.total_pnl:>+10,.2f} ({score.return_pct:+.2f}%)")
    print()
    print(f"  Sharpe:     {score.sharpe_ratio:>10.2f}")
    print(f"  Max DD:     ${score.max_drawdown:>10,.2f} ({score.max_drawdown_pct:.2f}%)")
    print(f"  Win Rate:   {score.win_rate * 100:>9.1f}%")
    print()
    print(f"  Trades:     {score.total_trades:>10}")
    print(f"  Settlements:{score.total_settlements:>10}")
    print(f"  Rejected:   {result.total_rejected:>10}")
    print(f"  Avg P&L:    ${score.avg_trade_pnl:>+10,.4f}")
    print()
    print(f"  Competition Score: ${score.competition_score:+,.2f}")
    print("=" * 60 + "\n")

    return score


def export_json(result: BacktestResult, output_path: Path) -> None:
    """Export full backtest result to JSON."""
    score = compute_score(result)

    data = {
        "strategy": score.strategy_name,
        "period": {
            "start": _fmt_ts(result.start_ts),
            "end": _fmt_ts(result.end_ts),
            "start_ts": result.start_ts,
            "end_ts": result.end_ts,
        },
        "performance": {
            "starting_cash": score.starting_cash,
            "final_value": score.final_portfolio_value,
            "total_pnl": round(score.total_pnl, 4),
            "return_pct": round(score.return_pct, 4),
            "sharpe_ratio": round(score.sharpe_ratio, 4),
            "max_drawdown": round(score.max_drawdown, 4),
            "max_drawdown_pct": round(score.max_drawdown_pct, 4),
            "win_rate": round(score.win_rate, 4),
            "competition_score": round(score.competition_score, 4),
        },
        "activity": {
            "total_trades": score.total_trades,
            "total_settlements": score.total_settlements,
            "total_rejected": result.total_rejected,
            "avg_trade_pnl": round(score.avg_trade_pnl, 4),
        },
        "fills": [
            {
                "timestamp": f.timestamp,
                "market_slug": f.market_slug,
                "token": f.token.value,
                "side": f.side.value,
                "size": round(f.size, 4),
                "avg_price": round(f.avg_price, 4),
                "cost": round(f.cost, 4),
            }
            for f in result.fills
        ],
        "settlements": [
            {
                "market_slug": s.market_slug,
                "interval": s.interval,
                "outcome": s.outcome.value,
                "chainlink_open": round(s.chainlink_open, 2),
                "chainlink_close": round(s.chainlink_close, 2),
            }
            for s in result.settlements
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Report exported to {output_path}")


def export_portfolio_csv(result: BacktestResult, output_path: Path) -> None:
    """Export portfolio value time series to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "datetime_utc", "cash", "total_value",
                         "realized_pnl", "unrealized_pnl"])
        for snap in result.portfolio_snapshots:
            writer.writerow([
                snap.timestamp,
                _fmt_ts(snap.timestamp),
                round(snap.cash, 4),
                round(snap.total_value, 4),
                round(snap.realized_pnl, 4),
                round(snap.unrealized_pnl, 4),
            ])

    logger.info(f"Portfolio CSV exported to {output_path}")


def format_leaderboard(scores: list[ScoreCard]) -> str:
    """Format a leaderboard table from multiple strategy scores."""
    # Sort by competition score (total P&L) descending
    ranked = sorted(scores, key=lambda s: s.competition_score, reverse=True)

    lines = [
        "",
        "=" * 80,
        "  COMPETITION LEADERBOARD",
        "=" * 80,
        f"  {'Rank':<6}{'Strategy':<25}{'P&L':>12}{'Sharpe':>10}{'MaxDD%':>10}{'Trades':>10}",
        "-" * 80,
    ]

    for i, s in enumerate(ranked, 1):
        lines.append(
            f"  {i:<6}{s.strategy_name:<25}"
            f"${s.total_pnl:>+10,.2f}"
            f"{s.sharpe_ratio:>10.2f}"
            f"{s.max_drawdown_pct:>9.2f}%"
            f"{s.total_trades:>10}"
        )

    lines.append("=" * 80)
    return "\n".join(lines)


def _fmt_ts(ts: int) -> str:
    """Format unix timestamp as ISO 8601 UTC string."""
    if ts <= 0:
        return "N/A"
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
