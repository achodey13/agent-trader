"""
CLI Runner - Run a single strategy backtest.

Usage:
    python -m backtester.runner --strategy backtester/examples/buy_and_hold.py
    python -m backtester.runner --strategy backtester/examples/buy_and_hold.py --data data/live/
    python -m backtester.runner --strategy my_strat.py --output results/
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from pathlib import Path

from .data_loader import build_timeline
from .engine import BacktestEngine
from .reporter import export_json, export_portfolio_csv, print_report
from .strategy import BaseStrategy

logger = logging.getLogger(__name__)


def load_strategy_from_file(path: Path) -> BaseStrategy:
    """
    Dynamically load a strategy from a .py file.

    The file must contain a class that inherits from BaseStrategy.
    If multiple are found, the first one is used.
    """
    spec = importlib.util.spec_from_file_location("user_strategy", str(path))
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load module from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["user_strategy"] = module
    spec.loader.exec_module(module)

    # Find BaseStrategy subclass
    candidates = []
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if (
            isinstance(attr, type)
            and issubclass(attr, BaseStrategy)
            and attr is not BaseStrategy
        ):
            candidates.append(attr)

    if not candidates:
        raise ValueError(
            f"No BaseStrategy subclass found in {path}. "
            f"Your file must define a class that inherits from BaseStrategy."
        )

    strategy_cls = candidates[0]
    if len(candidates) > 1:
        logger.warning(
            f"Multiple strategies found in {path}, using {strategy_cls.__name__}"
        )

    return strategy_cls()


def main():
    parser = argparse.ArgumentParser(
        description="Run a single strategy backtest",
        prog="python -m backtester.runner",
    )
    parser.add_argument(
        "--strategy", "-s", required=True,
        help="Path to strategy .py file",
    )
    parser.add_argument(
        "--data", "-d", default=None,
        help="Path to data directory (default: data/live/)",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output directory for results (default: no file output)",
    )
    parser.add_argument(
        "--cash", type=float, default=10_000.0,
        help="Starting cash (default: $10,000)",
    )
    parser.add_argument(
        "--intervals", nargs="+", default=["5m", "15m", "hourly"],
        help="Market intervals to include (default: 5m 15m hourly)",
    )
    parser.add_argument(
        "--snapshot-interval", type=int, default=60,
        help="Record portfolio snapshot every N seconds (default: 60)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load strategy
    strategy_path = Path(args.strategy)
    if not strategy_path.exists():
        print(f"Error: strategy file not found: {strategy_path}")
        sys.exit(1)

    logger.info(f"Loading strategy from {strategy_path}")
    strategy = load_strategy_from_file(strategy_path)
    logger.info(f"Strategy loaded: {type(strategy).__name__}")

    # Load data
    data_dir = Path(args.data) if args.data else None
    logger.info("Loading data...")
    data = build_timeline(data_dir=data_dir, intervals=args.intervals)

    if not data.timeline:
        print("Error: no data found. Check your --data path.")
        sys.exit(1)

    # Run backtest
    engine = BacktestEngine(
        data=data,
        strategy=strategy,
        starting_cash=args.cash,
        snapshot_interval=args.snapshot_interval,
    )
    result = engine.run()

    # Print report
    score = print_report(result)

    # Export results if output dir specified
    if args.output:
        output_dir = Path(args.output)
        name = type(strategy).__name__.lower()
        export_json(result, output_dir / f"{name}_result.json")
        export_portfolio_csv(result, output_dir / f"{name}_portfolio.csv")


if __name__ == "__main__":
    main()
