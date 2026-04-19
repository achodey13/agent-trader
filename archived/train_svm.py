"""
Train a linear SVM on engineered features for Polymarket 5m/15m direction
prediction, then distill it into numpy constants for the submission strategy.

Pipeline:
    1. Use the backtester engine as a "recording" harness — a stub BaseStrategy
       extracts feature vectors at 60-second intervals during each market's
       tradeable window.
    2. Labels come from on_settlement (market outcome: YES=1, NO=0).
    3. Fit LinearSVC + Platt-scaled probabilities; also fit LogisticRegression
       as a sanity baseline.
    4. Evaluate on validation data (out-of-sample).
    5. Print the full coefficient / intercept / scaler state needed for the
       numpy-only inference in ml_v1.py.

Notes:
    - sklearn is a training-time dep; the submission does NOT import it.
    - Features are designed to be computable at tick time from state.markets
      and a per-asset rolling Binance buffer — same machinery leadlag_v4 uses.
    - Scope: BTC + ETH + SOL, 5m + 15m only. Hourly excluded (v4 lesson).
"""

import math
import os
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from backtester.data_loader import build_timeline
from backtester.engine import BacktestEngine
from backtester.strategy import (
    BaseStrategy,
    Fill,
    MarketState,
    Order,
    Settlement,
    Token,
)


# ── Feature config (must match inference code) ───────────────────────
WINDOW_S = 300
MIN_SAMPLES = 30
LOOKBACKS = (30, 120, 240)

# ── Sampling config ──────────────────────────────────────────────────
SAMPLE_INTERVAL_S = 60       # one feature row per market every 60 sec
ENTRY_FRAC_MIN = 0.10
ENTRY_FRAC_MAX = 0.80


def _asset_from_slug(slug: str) -> str | None:
    if slug.startswith("btc-") or slug.startswith("bitcoin-"):
        return "BTC"
    if slug.startswith("eth-") or slug.startswith("ethereum-"):
        return "ETH"
    if slug.startswith("sol-") or slug.startswith("solana-"):
        return "SOL"
    return None


class _SignalTracker:
    def __init__(self) -> None:
        self._prices: dict[str, deque[float]] = {
            "BTC": deque(maxlen=WINDOW_S + 1),
            "ETH": deque(maxlen=WINDOW_S + 1),
            "SOL": deque(maxlen=WINDOW_S + 1),
        }

    def update(self, asset: str, mid: float) -> None:
        if mid > 0 and asset in self._prices:
            self._prices[asset].append(mid)

    def z_score(self, asset: str, lookback_s: int) -> float | None:
        buf = self._prices.get(asset)
        if buf is None or len(buf) < MIN_SAMPLES + 1:
            return None
        prices = list(buf)
        lookback = min(lookback_s, len(prices) - 1)
        p_now = prices[-1]
        p_prev = prices[-1 - lookback]
        if p_now <= 0 or p_prev <= 0:
            return None
        recent_return = math.log(p_now / p_prev)
        returns: list[float] = []
        for i in range(len(prices) - 1):
            a, b = prices[i], prices[i + 1]
            if a > 0 and b > 0:
                returns.append(math.log(b / a))
        if len(returns) < MIN_SAMPLES:
            return None
        mean = sum(returns) / len(returns)
        var = sum((r - mean) ** 2 for r in returns) / len(returns)
        if var <= 0:
            return None
        sigma_1s = math.sqrt(var)
        null_std = sigma_1s * math.sqrt(lookback)
        return recent_return / null_std


class FeatureRecorder(BaseStrategy):
    """Records (features, slug, timestamp) rows; labels attached post-hoc."""

    ASSET_CODES = {"BTC": 0, "ETH": 1, "SOL": 2}

    def __init__(self) -> None:
        self._sig = _SignalTracker()
        self._last_sampled: dict[str, int] = {}
        # slug → list[feature-row]. Label appended after on_settlement.
        self._rows_by_slug: dict[str, list[list[float]]] = {}
        self._labels: dict[str, int] = {}

    def on_tick(self, state: MarketState) -> list[Order]:
        self._sig.update("BTC", state.btc_mid)
        self._sig.update("ETH", state.eth_mid)
        self._sig.update("SOL", state.sol_mid)

        for slug, m in state.markets.items():
            if m.interval == "hourly":
                continue  # scope: 5m + 15m
            if m.time_remaining_frac < ENTRY_FRAC_MIN or m.time_remaining_frac > ENTRY_FRAC_MAX:
                continue

            asset = _asset_from_slug(slug)
            if asset is None:
                continue

            last = self._last_sampled.get(slug, -10_000_000)
            if state.timestamp - last < SAMPLE_INTERVAL_S:
                continue

            z30 = self._sig.z_score(asset, 30)
            z120 = self._sig.z_score(asset, 120)
            z240 = self._sig.z_score(asset, 240)
            if z30 is None or z120 is None or z240 is None:
                continue

            if m.yes_ask <= 0 or m.yes_bid <= 0:
                continue

            spread = m.yes_ask - m.yes_bid
            bid_sz = m.yes_book.total_bid_size
            ask_sz = m.yes_book.total_ask_size
            book_imb = 0.0
            if bid_sz + ask_sz > 0:
                book_imb = (bid_sz - ask_sz) / (bid_sz + ask_sz)

            feature_row = [
                z30, z120, z240,
                m.yes_ask, m.yes_bid, spread,
                m.time_remaining_frac,
                book_imb,
                float(self.ASSET_CODES[asset]),  # raw code, will one-hot at post-proc
            ]
            self._rows_by_slug.setdefault(slug, []).append(feature_row)
            self._last_sampled[slug] = state.timestamp

        return []

    def on_fill(self, fill: Fill) -> None:
        pass

    def on_settlement(self, settlement: Settlement) -> None:
        self._labels[settlement.market_slug] = (
            1 if settlement.outcome == Token.YES else 0
        )

    def flatten(self) -> pd.DataFrame:
        cols = [
            "z30", "z120", "z240",
            "yes_ask", "yes_bid", "spread",
            "time_frac", "book_imb", "asset_code",
            "label",
        ]
        out = []
        for slug, rows in self._rows_by_slug.items():
            label = self._labels.get(slug)
            if label is None:
                continue
            for r in rows:
                out.append(r + [label])
        return pd.DataFrame(out, columns=cols)


def _extract(data_dir: str) -> pd.DataFrame:
    print(f"Loading {data_dir} …", flush=True)
    data = build_timeline(
        data_dir=Path(data_dir),
        assets=["BTC", "ETH", "SOL"],
        intervals=["5m", "15m"],
    )
    print(f"  ticks: {len(data.timeline):,} lifecycles: {len(data.lifecycles):,}", flush=True)

    rec = FeatureRecorder()
    engine = BacktestEngine(data=data, strategy=rec, starting_cash=10_000)
    engine.run()

    df = rec.flatten()
    print(f"  feature rows: {len(df):,} (labeled markets: {df.market_slug.nunique() if 'market_slug' in df.columns else len(rec._labels):,})", flush=True)
    return df


def main() -> None:
    # Training + validation feature extraction
    df_train = _extract("data/train")
    df_val = _extract("data/validation")

    # One-hot encode asset_code → asset_BTC / asset_ETH / asset_SOL
    for d in (df_train, df_val):
        d["asset_BTC"] = (d["asset_code"] == 0).astype(int)
        d["asset_ETH"] = (d["asset_code"] == 1).astype(int)
        d["asset_SOL"] = (d["asset_code"] == 2).astype(int)

    # v2: drop market-price features (yes_ask, yes_bid, spread) so the model
    # can't shortcut to "P ≈ market price" — force learning residual alpha.
    feature_cols = [
        "z30", "z120", "z240",
        "time_frac", "book_imb",
        "asset_BTC", "asset_ETH", "asset_SOL",
    ]
    X_train = df_train[feature_cols].to_numpy(dtype=np.float64)
    y_train = df_train["label"].to_numpy(dtype=np.int64)
    X_val = df_val[feature_cols].to_numpy(dtype=np.float64)
    y_val = df_val["label"].to_numpy(dtype=np.int64)

    print(f"\nTraining set: {X_train.shape}, YES rate: {y_train.mean():.3f}")
    print(f"Validation set: {X_val.shape}, YES rate: {y_val.mean():.3f}")

    # Fit scaler
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    Xs_train = scaler.fit_transform(X_train)
    Xs_val = scaler.transform(X_val)

    # Baseline: logistic regression
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss

    lr = LogisticRegression(C=1.0, max_iter=2000)
    lr.fit(Xs_train, y_train)
    lr_p_train = lr.predict_proba(Xs_train)[:, 1]
    lr_p_val = lr.predict_proba(Xs_val)[:, 1]
    print(f"\nLogistic Regression:")
    print(f"  train AUC: {roc_auc_score(y_train, lr_p_train):.4f}")
    print(f"  val   AUC: {roc_auc_score(y_val, lr_p_val):.4f}")
    print(f"  val   acc: {accuracy_score(y_val, lr_p_val > 0.5):.4f}")
    print(f"  val   brier: {brier_score_loss(y_val, lr_p_val):.4f}")

    # SVM: LinearSVC + Platt scaling
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV

    print("\nFitting LinearSVC + Platt scaling...")
    base_svm = LinearSVC(C=1.0, max_iter=5000, dual="auto")
    svm = CalibratedClassifierCV(base_svm, method="sigmoid", cv=5)
    svm.fit(Xs_train, y_train)
    svm_p_train = svm.predict_proba(Xs_train)[:, 1]
    svm_p_val = svm.predict_proba(Xs_val)[:, 1]
    print(f"Linear SVM (Platt):")
    print(f"  train AUC: {roc_auc_score(y_train, svm_p_train):.4f}")
    print(f"  val   AUC: {roc_auc_score(y_val, svm_p_val):.4f}")
    print(f"  val   acc: {accuracy_score(y_val, svm_p_val > 0.5):.4f}")
    print(f"  val   brier: {brier_score_loss(y_val, svm_p_val):.4f}")

    # Pick winner
    lr_auc = roc_auc_score(y_val, lr_p_val)
    svm_auc = roc_auc_score(y_val, svm_p_val)

    if svm_auc >= lr_auc:
        print("\n→ SVM wins. Distilling LinearSVC (refit without calibration) for numpy inference.")
        raw_svm = LinearSVC(C=1.0, max_iter=5000, dual="auto")
        raw_svm.fit(Xs_train, y_train)
        # Platt parameters from CalibratedClassifierCV internals
        # Each calibrator has {A_, B_} (sigmoid params). Use the mean across CV folds.
        A_vals = [c.calibrators[0].a_ for c in svm.calibrated_classifiers_]
        B_vals = [c.calibrators[0].b_ for c in svm.calibrated_classifiers_]
        A = float(np.mean(A_vals))
        B = float(np.mean(B_vals))
        w = raw_svm.coef_.ravel().astype(float)
        b = float(raw_svm.intercept_[0])
        model_name = "LinearSVC+Platt"
    else:
        print("\n→ LR wins. Distilling LogisticRegression for numpy inference.")
        w = lr.coef_.ravel().astype(float)
        b = float(lr.intercept_[0])
        A = -1.0  # For LR, P = 1/(1+exp(-score)) directly; use A=-1, B=0 with score=wx+b (actually: P = sigmoid(wx+b))
        B = 0.0
        model_name = "LogisticRegression"

    mean = scaler.mean_.astype(float)
    scale = scaler.scale_.astype(float)

    print("\n" + "=" * 72)
    print(f"DISTILLED {model_name} — paste into ml_v1.py:")
    print("=" * 72)
    print()
    print(f"FEATURE_NAMES = {feature_cols!r}")
    print(f"SCALER_MEAN = {list(mean)!r}")
    print(f"SCALER_SCALE = {list(scale)!r}")
    print(f"W = {list(w)!r}")
    print(f"B = {b!r}")
    print(f"PLATT_A = {A!r}")
    print(f"PLATT_B = {B!r}")
    print(f"MODEL_KIND = {model_name!r}  # 'LinearSVC+Platt' or 'LogisticRegression'")
    print()


if __name__ == "__main__":
    main()
