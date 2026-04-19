"""Tests for data_loader module."""

import json
import sqlite3
import tempfile
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

from backtester.data_loader import (
    build_timeline,
    compute_settlements,
    load_chainlink_prices,
    load_market_outcomes,
    load_market_prices,
    load_orderbooks,
    parse_slug_lifecycle,
)
from backtester.strategy import MarketLifecycle, Token


class TestParseSlugLifecycle:
    def test_5m_slug(self):
        lc = parse_slug_lifecycle("btc-updown-5m-1700000000")
        assert lc is not None
        assert lc.interval == "5m"
        assert lc.start_ts == 1700000000
        assert lc.end_ts == 1700000300  # +300s

    def test_15m_slug(self):
        lc = parse_slug_lifecycle("btc-updown-15m-1700000000")
        assert lc is not None
        assert lc.interval == "15m"
        assert lc.start_ts == 1700000000
        assert lc.end_ts == 1700000900  # +900s

    def test_unknown_slug(self):
        lc = parse_slug_lifecycle("some-random-slug")
        assert lc is None

    def test_hourly_slug_supported(self):
        lc = parse_slug_lifecycle("bitcoin-up-or-down-march-26-2026-5am-et")
        assert lc is not None
        assert lc.interval == "hourly"
        assert lc.end_ts - lc.start_ts == 3600


class TestLoadMarketPrices:
    def test_missing_db(self, tmp_path):
        df = load_market_prices(tmp_path / "nonexistent.db")
        assert df.empty

    def test_load_from_sqlite(self, tmp_path):
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE market_prices (
                id INTEGER PRIMARY KEY,
                timestamp_us INTEGER,
                interval TEXT,
                market_slug TEXT,
                condition_id TEXT,
                yes_token_id TEXT, no_token_id TEXT,
                yes_price REAL, no_price REAL,
                yes_bid REAL, yes_ask REAL,
                no_bid REAL, no_ask REAL,
                volume REAL, liquidity REAL
            )
        """)
        conn.execute(
            "INSERT INTO market_prices VALUES (1, 1700000000000000, '5m', 'btc-updown-5m-1700000000', '', '', '', 0.55, 0.45, 0.54, 0.56, 0.44, 0.46, 100, 50)"
        )
        conn.commit()
        conn.close()

        df = load_market_prices(db_path)
        assert len(df) == 1
        assert "ts_sec" in df.columns
        assert df.iloc[0]["ts_sec"] == 1700000000


class TestLoadOrderbooks:
    def test_empty_dir(self, tmp_path):
        df = load_orderbooks(tmp_path)
        assert df.empty

    def test_load_csv(self, tmp_path):
        csv_path = tmp_path / "2024-01-01.csv"
        data = {
            "timestamp_us": [1700000000000000],
            "interval": ["5m"],
            "market_slug": ["btc-updown-5m-1700000000"],
            "yes_bids_json": ['[[0.48, 100]]'],
            "yes_asks_json": ['[[0.52, 100]]'],
            "no_bids_json": ['[[0.48, 100]]'],
            "no_asks_json": ['[[0.52, 100]]'],
            "yes_best_bid": [0.48],
            "yes_best_ask": [0.52],
            "no_best_bid": [0.48],
            "no_best_ask": [0.52],
            "yes_n_bids": [1],
            "yes_n_asks": [1],
            "no_n_bids": [1],
            "no_n_asks": [1],
            "yes_total_bid_size": [100],
            "yes_total_ask_size": [100],
            "no_total_bid_size": [100],
            "no_total_ask_size": [100],
        }
        pd.DataFrame(data).to_csv(csv_path, index=False)

        df = load_orderbooks(tmp_path)
        assert len(df) == 1
        assert df.iloc[0]["yes_best_bid"] == 0.48

    def test_load_legacy_jsonl(self, tmp_path):
        jsonl_path = tmp_path / "2024-01-01.jsonl"
        record = {
            "timestamp_us": 1700000000000000,
            "interval": "15m",
            "market_slug": "btc-updown-15m-1700000000",
            "yes_book": {
                "bids": [{"price": "0.48", "size": "100"}],
                "asks": [{"price": "0.52", "size": "100"}],
            },
            "no_book": {
                "bids": [{"price": "0.48", "size": "100"}],
                "asks": [{"price": "0.52", "size": "100"}],
            },
        }
        with open(jsonl_path, "w") as f:
            f.write(json.dumps(record) + "\n")

        df = load_orderbooks(tmp_path)
        assert len(df) == 1
        assert df.iloc[0]["yes_best_bid"] == 0.48


class TestComputeSettlements:
    def test_yes_outcome(self):
        lc = MarketLifecycle("test-slug", "5m", 1000, 1300)
        chainlink = pd.DataFrame({
            "ts_sec": [1000, 1300],
            "price": [95000.0, 95100.0],  # went up -> YES
        })
        settlements = compute_settlements([lc], chainlink)
        assert "test-slug" in settlements
        assert settlements["test-slug"].outcome == Token.YES

    def test_no_outcome(self):
        lc = MarketLifecycle("test-slug", "5m", 1000, 1300)
        chainlink = pd.DataFrame({
            "ts_sec": [1000, 1300],
            "price": [95000.0, 94900.0],  # went down -> NO
        })
        settlements = compute_settlements([lc], chainlink)
        assert settlements["test-slug"].outcome == Token.NO

    def test_known_outcome_override(self):
        lc = MarketLifecycle("test-slug", "5m", 1000, 1300)
        chainlink = pd.DataFrame({"ts_sec": [1000, 1300], "price": [95000, 95100]})
        # Known outcome says NO even though Chainlink says YES
        settlements = compute_settlements([lc], chainlink, {"test-slug": "NO"})
        assert settlements["test-slug"].outcome == Token.NO

    def test_empty_chainlink(self):
        lc = MarketLifecycle("test-slug", "5m", 1000, 1300)
        settlements = compute_settlements([lc], pd.DataFrame())
        assert "test-slug" not in settlements


class TestLoadMarketOutcomes:
    def test_missing_db(self, tmp_path):
        outcomes = load_market_outcomes(tmp_path / "nonexistent.db")
        assert outcomes == {}

    def test_no_table(self, tmp_path):
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE dummy (id INTEGER)")
        conn.commit()
        conn.close()
        outcomes = load_market_outcomes(db_path)
        assert outcomes == {}

    def test_load_outcomes(self, tmp_path):
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE market_outcomes (
                market_slug TEXT PRIMARY KEY, interval TEXT, question TEXT,
                status TEXT DEFAULT 'pending', outcome TEXT,
                end_ts INTEGER, first_seen_us INTEGER,
                checked_at_us INTEGER, resolved_at_us INTEGER
            )
        """)
        conn.execute(
            "INSERT INTO market_outcomes VALUES ('slug-1', '5m', 'q', 'resolved', 'YES', 0, 0, 0, 0)"
        )
        conn.commit()
        conn.close()
        outcomes = load_market_outcomes(db_path)
        assert outcomes == {"slug-1": "YES"}


class TestChainlinkPerAssetAggregation:
    """Oracle prices must stay in their own per-asset slot.

    The fixture inserts BTC, ETH, and SOL rows at every second with ETH and
    SOL arriving last within the second. A per-asset aggregation must keep
    `chainlink_btc` in the BTC range (~$60k) regardless of row ordering;
    any cross-asset leak would be caught by the range asserts below.
    """

    @staticmethod
    def _make_data_dir(tmp_path):
        """Build a minimal polymarket.db with empty books and binance dirs."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        books_dir = data_dir / "polymarket_books"
        books_dir.mkdir()
        binance_dir = data_dir / "binance_lob"
        binance_dir.mkdir()

        db_path = data_dir / "polymarket.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE market_prices (
                id INTEGER PRIMARY KEY,
                timestamp_us INTEGER,
                interval TEXT,
                market_slug TEXT,
                condition_id TEXT,
                yes_token_id TEXT, no_token_id TEXT,
                yes_price REAL, no_price REAL,
                yes_bid REAL, yes_ask REAL,
                no_bid REAL, no_ask REAL,
                volume REAL, liquidity REAL
            )
        """)
        conn.execute("""
            CREATE TABLE rtds_prices (
                id INTEGER PRIMARY KEY,
                timestamp_us INTEGER,
                source TEXT,
                symbol TEXT,
                price REAL,
                raw_payload TEXT
            )
        """)

        # One BTC 5m market covering ts 1_700_000_000 .. +300.
        slug = "btc-updown-5m-1700000000"
        for i in range(0, 301, 30):
            ts_us = (1_700_000_000 + i) * 1_000_000
            conn.execute(
                "INSERT INTO market_prices VALUES (NULL, ?, '5m', ?, '', '', '', 0.5, 0.5, 0.49, 0.51, 0.49, 0.51, 0, 0)",
                (ts_us, slug),
            )

        # Insert BTC first, then ETH, then SOL within each second so that
        # a naive `groupby("ts_sec").last()` would pick the SOL row.
        for i in range(0, 301):
            ts_us = (1_700_000_000 + i) * 1_000_000
            conn.execute(
                "INSERT INTO rtds_prices VALUES (NULL, ?, 'chainlink', 'BTC/USD', ?, '')",
                (ts_us, 60_000.0 + i),
            )
            conn.execute(
                "INSERT INTO rtds_prices VALUES (NULL, ?, 'chainlink', 'ETH/USD', ?, '')",
                (ts_us + 10, 3_400.0 + i * 0.1),
            )
            conn.execute(
                "INSERT INTO rtds_prices VALUES (NULL, ?, 'chainlink', 'SOL/USD', ?, '')",
                (ts_us + 20, 180.0 + i * 0.01),
            )
        conn.commit()
        conn.close()
        return data_dir

    def test_chainlink_per_asset_ranges(self, tmp_path):
        data_dir = self._make_data_dir(tmp_path)
        bt = build_timeline(data_dir=data_dir, intervals=["5m"])

        assert bt.timeline, "timeline should be non-empty"

        for tick in bt.timeline:
            assert tick.chainlink_btc > 10_000, (
                f"chainlink_btc out of BTC range: got {tick.chainlink_btc} at ts={tick.ts_sec}"
            )
            assert 100 < tick.chainlink_eth < 20_000, (
                f"chainlink_eth out of ETH range: {tick.chainlink_eth}"
            )
            assert 5 < tick.chainlink_sol < 1_000, (
                f"chainlink_sol out of SOL range: {tick.chainlink_sol}"
            )
