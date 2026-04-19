# Scoring

## TL;DR

- **Primary metric (competition rank):** total P&L in USD over the evaluation period.
- **Tiebreaker:** annualized Sharpe ratio.
- **Also reported:** max drawdown, win rate, trade count - for transparency, not ranking.

Starting cash is **$10,000**. Evaluation is performed on a held-out test dataset that participants do not have access to during the competition.

---

## 1. Total P&L (primary)

Defined as final portfolio value minus starting cash:

```
total_pnl = final_portfolio_value - starting_cash
```

Where `final_portfolio_value` at the end of the backtest is:

```
final_portfolio_value = cash + sum(shares_i * mark_price_i) + settlement_payouts
```

For **unsettled positions at the end of the test period**, shares are marked at the YES/NO mid-price. For **settled positions**, the payout ($1 × winning-side shares) has already been added to cash.

Higher is better. Ranking is strictly by this number.

### Worked example

| Event | Cash | Shares held | Portfolio value |
|---|---|---|---|
| Start | $10,000.00 | - | $10,000.00 |
| Buy 100 YES at $0.40 | $9,960.00 | 100 YES | $9,960 + 100 × $0.40 = $10,000 (zero slippage) |
| 5m later: market resolves YES | $10,060.00 | - | $10,060 ($9,960 cash + 100 × $1 payout) |
| Final P&L | | | **+$60.00** |

---

## 2. Annualized Sharpe (tiebreaker)

Computed from the portfolio-value time series that the engine samples every 60 seconds:

```python
# From backtester/scoring.py
returns = [(values[i] - values[i-1]) / values[i-1] for i in range(1, len(values))]
mean_ret = mean(returns)
std_ret = std(returns, ddof=1)

snapshot_interval_s = total_duration_s / len(returns)
periods_per_year = (365 * 86400) / snapshot_interval_s
annualization = sqrt(periods_per_year)

sharpe = (mean_ret / std_ret) * annualization
```

### Why annualize from the actual snapshot interval?

Naively annualizing as if every tick were an independent observation would massively inflate Sharpe because successive 1-second portfolio values are almost perfectly autocorrelated. The backtester snapshots portfolio value every 60 seconds and annualizes from there, which is more faithful to the true variance of the strategy's returns.

### Zero-risk floor

If `std_ret < 1e-12` (e.g., a strategy that never trades), Sharpe is reported as `0.00` rather than infinity.

---

## 3. Max Drawdown (reported)

Largest peak-to-trough decline in portfolio value, in both dollars and percent:

```python
peak = values[0]
max_dd = 0
max_dd_pct = 0
for v in values:
    if v > peak:
        peak = v
    dd = peak - v
    if dd > max_dd:
        max_dd = dd
        max_dd_pct = dd / peak if peak > 0 else 0
```

Lower is safer. Not used for ranking - only for transparency.

---

## 4. Win rate (reported)

Fraction of markets in which a fill on the winning side occurred:

```
win_rate = markets_with_winning_fill / markets_with_any_fill
```

Not used for ranking. A high win rate with negative P&L means you won many small trades and lost a few big ones. A low win rate with positive P&L means the opposite.

---

## 5. Total trades, settlements, avg trade P&L (reported)

Informational only - shown in the report so you can sanity-check your strategy's activity level against your expectations.

---

## 6. Example `BACKTEST REPORT` output

```
============================================================
  BACKTEST REPORT: MyStrategy
============================================================
  Period:     2026-04-10 00:00:00 UTC -> 2026-04-10 12:00:00 UTC
  Duration:   12.0 hours (43,200 seconds)
  Runtime:    8.4s

  Starting:   $ 10,000.00
  Final:      $ 10,342.10
  P&L:        $   +342.10 (+3.42%)

  Sharpe:           2.31
  Max DD:     $     42.00 (0.41%)
  Win Rate:      56.3%

  Trades:            156
  Settlements:        89
  Rejected:            2
  Avg P&L:    $  +2.1929

  Competition Score: $+342.10
============================================================
```

`Competition Score` = `total P&L`. This is the number that ranks you.

---

## 7. The exact scoring code

Reproduced from `backtester/scoring.py` so there's no ambiguity:

```python
@dataclass
class ScoreCard:
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
    competition_score: float   # = total_pnl
    avg_implied_prob: float
```

The organizer will run the exact same code against the test dataset.

---

## 8. What you can do to optimize

1. **Focus on total P&L.** Sharpe only matters if you tie with someone - and ties are unlikely.
2. **Watch drawdown.** A strategy that makes $500 but drew down $2,000 midway through the run is fragile. Production scoring uses the full run, so no, you can't just cherry-pick a good period.
3. **Don't over-trade.** Each trade costs you the spread. A higher trade count with the same P&L means each trade was less edgy.
4. **Evaluate on `data/validation/` before the deadline.** The test set is held out, but validation is the closest proxy. If your training-set P&L is +$800 and validation is -$100, you are overfitting.
