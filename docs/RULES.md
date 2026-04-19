# Competition Rules

## 1. Capital

- **Starting cash:** $10,000 USDC (simulated).
- No deposits or withdrawals during the run.
- Only the final portfolio value counts - see [SCORING.md](SCORING.md).

## 2. Position limits

- **Max 500 shares per token per market.** YES and NO are separate tokens, so the practical max per market is 500 YES + 500 NO = 1,000 shares combined.
- If you submit an order that would exceed this limit, it is **partially filled** up to the cap or **rejected** depending on the engine's constraint - inspect `on_fill` vs. order-rejection logging.

## 3. No short selling

- You can only **SELL tokens you already own.**
- Attempting to sell shares you don't have → order rejected.
- To bet against YES, buy NO. To exit a YES position, sell YES.

## 4. Execution latency

- Orders submitted at tick T **fill at tick T+1** (1-second delay).
- The engine uses the order book as of tick T+1 to match your order, simulating real-world latency.
- Your strategy cannot peek at tick T+1's state before submitting.

## 5. Stale-book guard

- If the most recent order book snapshot is **more than 5 seconds old**, orders against that market are **rejected**.
- Rejections are counted in the report (`Rejected: N`).
- This prevents exploiting known data gaps.

## 6. Walk-the-book execution

- BUY orders walk the book up to `limit_price`:
  - If level 1 has enough size → fill fully at level 1.
  - If level 1 depletes → fill the rest at level 2, 3, etc., until you hit your `limit_price` or the book is exhausted.
- SELL orders walk down similarly.
- `limit_price=None` is a market order (take all liquidity up to whatever price).
- **Multiple orders on the same tick see depleted liquidity from earlier orders.** The engine processes orders within a tick in the order they are returned from `on_tick`.

## 7. Settlement

- Each market resolves at its `end_ts`.
- **YES** pays $1 if the underlying asset's Chainlink price at `end_ts` >= price at `start_ts`, else $0.
- **NO** is the inverse.
- Settlement payouts are added to your `cash` balance automatically and `on_settlement` is called on your strategy.

## 8. Submission format

- Submit **one `.py` file** containing **one class** that subclasses `BaseStrategy`.
- File name suggestion: `{teamname}_strategy.py`.
- The engine instantiates your class once with no args and calls `on_tick()` every second.

## 9. Allowed imports

Your strategy may import from:
- **Python standard library**
- **`numpy`**
- **`pandas`**
- **`scipy`**
- **`backtester`** (only the public symbols - `BaseStrategy`, `MarketState`, `MarketView`, `Order`, `Side`, `Token`, `Fill`, `Settlement`, `OrderBookSnapshot`, `OrderBookLevel`, `PositionView`)

Any other import may cause disqualification.

## 10. Forbidden behavior

- **No filesystem access.** Do not read from or write to disk.
- **No network access.** Do not open sockets or make HTTP requests.
- **No subprocess / shell.** Do not spawn processes.
- **No introspection of engine internals.** Don't reach into private state; only use what's provided in `MarketState`.

Strategies that violate these will be disqualified before scoring.

## 11. Evaluation

- Strategies are run against a **held-out test dataset** that was not provided during the competition.
- The test set is collected after the submission deadline, from the same live data pipeline.
- All submissions run against the same test set, with the same starting cash, in the same engine version.

See [SCORING.md](SCORING.md) for the ranking formula.

## 12. Dev-loop flags (not for submission - for your iteration speed only)

`run_backtest.py` accepts these optional flags that **reduce which data gets loaded**:

| Flag | Example | Effect |
|---|---|---|
| `--hours N` | `--hours 4` | Load only the last N hours of data |
| `--assets A [B ...]` | `--assets BTC` | Load only the listed assets (`BTC`, `ETH`, `SOL`) |
| `--intervals I [J ...]` | `--intervals 5m` | Load only the listed intervals (`5m`, `15m`, `hourly`) |

Final scoring on the held-out test set is **always unfiltered** - your strategy is evaluated on every market across every interval and every asset. These flags are only for speeding up your own iteration on train/validation.
