# Data Format Reference

This document is the authoritative schema reference for every data source in the hackathon bundles.

## Conventions

- **All timestamps:** `timestamp_us` = int64 epoch microseconds in **UTC**.
  - Convert to unix seconds: `ts_sec = timestamp_us // 1_000_000`
  - Convert to Python datetime: `datetime.fromtimestamp(timestamp_us / 1e6, tz=timezone.utc)`
- **Sampling:** 1 snapshot per second per market (train and validation bundles are pre-downsampled).
- **Currency:** USD throughout.
- **Price units:** Polymarket YES/NO prices are in USD (range 0.00 to 1.00). Binance LOB prices are in USD per unit of the base asset. Chainlink oracle prices are in USD.

## Directory layout after `python download_data.py`

```
data/
├── train/
│   ├── polymarket.db               # SQLite
│   ├── polymarket_books/
│   │   └── orderbooks.csv          # Full CLOB depth, 1s cadence
│   └── binance_lob/
│       ├── btcusdt.parquet         # 10-level LOB
│       ├── ethusdt.parquet
│       └── solusdt.parquet
└── validation/
    ├── polymarket.db
    ├── polymarket_books/orderbooks.csv
    └── binance_lob/{btcusdt,ethusdt,solusdt}.parquet
```

Both bundles have identical schemas; only the time range differs.

---

## `polymarket.db` (SQLite)

WAL mode. Three tables of interest.

### Table: `market_prices`

Top-of-book snapshots for every active Polymarket prediction market, 1 row per market per second.

| Column | Type | Description |
|---|---|---|
| `timestamp_us` | INTEGER | Epoch microseconds (UTC) |
| `interval` | TEXT | `"5m"`, `"15m"`, or `"hourly"` |
| `market_slug` | TEXT | Unique market id (e.g. `btc-updown-5m-1776283500`) |
| `condition_id` | TEXT | Polymarket condition ID |
| `yes_token_id` | TEXT | YES token contract ID |
| `no_token_id` | TEXT | NO token contract ID |
| `yes_price` | REAL | YES mid-price = (yes_bid + yes_ask) / 2 |
| `no_price` | REAL | NO mid-price |
| `yes_bid` | REAL | Best bid for YES token |
| `yes_ask` | REAL | Best ask for YES token |
| `no_bid` | REAL | Best bid for NO token |
| `no_ask` | REAL | Best ask for NO token |
| `volume` | REAL | Reserved; currently 0 |
| `liquidity` | REAL | Reserved; currently 0 |

**Example load:**
```python
import sqlite3, pandas as pd
conn = sqlite3.connect("data/train/polymarket.db")
df = pd.read_sql("SELECT * FROM market_prices WHERE interval='5m' LIMIT 10000", conn)
```

### Table: `rtds_prices`

Chainlink on-chain oracle reference prices. Used by the engine for settlement.

| Column | Type | Description |
|---|---|---|
| `timestamp_us` | INTEGER | Epoch microseconds (UTC) |
| `source` | TEXT | Always `"chainlink"` |
| `symbol` | TEXT | `"BTC/USD"`, `"ETH/USD"`, or `"SOL/USD"` |
| `price` | REAL | Oracle price in USD |
| `raw_payload` | TEXT | JSON with full-precision raw oracle value |

### Table: `market_outcomes`

Resolution records. Populated when a market settles.

| Column | Type | Description |
|---|---|---|
| `market_slug` | TEXT | Market identifier |
| `interval` | TEXT | `"5m"`, `"15m"`, or `"hourly"` |
| `question` | TEXT | Human-readable market question |
| `status` | TEXT | `"resolved"` or `"pending"` |
| `outcome` | TEXT | `"YES"` or `"NO"` (null if pending) |
| `end_ts` | INTEGER | Market close time (unix seconds) |

---

## `polymarket_books/orderbooks.csv`

Full depth-of-book snapshots for every market, 1 row per market per second.

| Column | Description |
|---|---|
| `timestamp_us` | Epoch microseconds |
| `interval` | Market interval (`"5m"`, `"15m"`, `"hourly"`) |
| `market_slug` | Market identifier |
| `yes_best_bid`, `yes_best_ask` | YES token top-of-book |
| `no_best_bid`, `no_best_ask` | NO token top-of-book |
| `yes_n_bids`, `yes_n_asks` | Number of price levels on each side of YES book |
| `no_n_bids`, `no_n_asks` | Number of price levels on each side of NO book |
| `yes_total_bid_size`, `yes_total_ask_size` | Sum of sizes across all YES levels |
| `no_total_bid_size`, `no_total_ask_size` | Sum of sizes across all NO levels |
| `yes_bids_json`, `yes_asks_json` | JSON array of `[price, size]` pairs, all levels |
| `no_bids_json`, `no_asks_json` | JSON array of `[price, size]` pairs, all levels |

**Example load:**
```python
import pandas as pd, json
df = pd.read_csv("data/train/polymarket_books/orderbooks.csv")
df["yes_bids"] = df["yes_bids_json"].apply(json.loads)
# first row's YES bids: list of [price, size] pairs, sorted desc by price
```

---

## `binance_lob/*.parquet`

One file per asset: `btcusdt.parquet`, `ethusdt.parquet`, `solusdt.parquet`. Each contains 10-level limit order book snapshots from the Binance spot exchange, sampled at 1 second.

| Column | Type | Description |
|---|---|---|
| `timestamp_us` | int64 | Epoch microseconds (UTC) |
| `event_time_ms` | int64 | Binance exchange event time (milliseconds since epoch) |
| `symbol` | string | `"BTCUSDT"`, `"ETHUSDT"`, or `"SOLUSDT"` |
| `ask_price_1` .. `ask_price_10` | float64 | Ask prices, level 1 = best (lowest) ask |
| `ask_vol_1` .. `ask_vol_10` | float64 | Ask sizes in base-asset units |
| `bid_price_1` .. `bid_price_10` | float64 | Bid prices, level 1 = best (highest) bid |
| `bid_vol_1` .. `bid_vol_10` | float64 | Bid sizes in base-asset units |

Snappy compression. Total of 44 columns per row (3 metadata + 40 LOB features - the "DeepLOB" format).

**Example load:**
```python
import pandas as pd
df = pd.read_parquet("data/train/binance_lob/btcusdt.parquet")
df["btc_mid"] = (df["ask_price_1"] + df["bid_price_1"]) / 2
df["btc_spread"] = df["ask_price_1"] - df["bid_price_1"]
```

---

## Merging datasets

The backtester merges all data sources automatically. For your own analysis, join on the unix-second key:

```python
import sqlite3, pandas as pd

# 1-second buckets everywhere
def to_sec(df, col="timestamp_us"):
    df["ts_sec"] = df[col] // 1_000_000
    return df

conn = sqlite3.connect("data/train/polymarket.db")
prices = to_sec(pd.read_sql("SELECT * FROM market_prices", conn))

btc = to_sec(pd.read_parquet("data/train/binance_lob/btcusdt.parquet"))
btc_subset = btc[["ts_sec", "bid_price_1", "ask_price_1"]].rename(
    columns={"bid_price_1": "btc_bid", "ask_price_1": "btc_ask"}
)

merged = prices.merge(btc_subset, on="ts_sec", how="left")
```

Within the backtester, this merge is performed by `backtester.data_loader.build_timeline()` which returns a `BacktestData` object with a unified per-second timeline, market lifecycle map, and settlement prices.

---

## Market slug format

| Interval | Slug format | Examples |
|---|---|---|
| 5-minute | `{asset}-updown-5m-{epoch_seconds}` | `btc-updown-5m-1776283500`<br>`sol-updown-5m-1776283500`<br>`eth-updown-5m-1776283500` |
| 15-minute | `{asset}-updown-15m-{epoch_seconds}` | `btc-updown-15m-1776283500` |
| Hourly | `{asset}-up-or-down-{month}-{day}-{year}-{hour}{ampm}-et` | `bitcoin-up-or-down-june-1-2025-12pm-et` |

The slug prefix identifies the underlying asset:
- `btc-` / `bitcoin-` → BTC
- `sol-` / `solana-` → SOL
- `eth-` / `ethereum-` → ETH

The trailing integer in 5m/15m slugs is the market's **start time** in unix seconds. The market settles exactly 5 or 15 minutes later at the Chainlink price.

---

## Data volume

Approximate sizes (uncompressed, after extraction):

| Asset | train | validation |
|---|---|---|
| `polymarket.db` | ~300 MB | ~150 MB |
| `polymarket_books/orderbooks.csv` | ~1.2 GB | ~600 MB |
| `binance_lob/*.parquet` (all 3) | ~400 MB | ~200 MB |

The compressed `.tar.gz` bundles are ~900 MB (train) and ~425 MB (validation).
