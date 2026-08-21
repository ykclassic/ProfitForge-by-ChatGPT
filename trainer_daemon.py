from __future__ import annotations

"""ProfitForge hourly signal trainer.

P0 responsibilities:
- use Bitget as the sole market-data source;
- consume only fully closed candles;
- write through the canonical database owner;
- generate at most one signal per strategy/symbol/timeframe/candle;
- attach an explicit expiry;
- calculate risk and position size before a signal can become ACTIVE.

P1 strategy-integrity responsibilities:
- build features through one canonical feature engine;
- align multiple closed timeframes point-in-time;
- interpret trend, volatility, structure and liquidity regimes;
- train on trade-outcome labels rather than next-candle direction;
- include research fees and slippage in those labels.

This module does not place live orders.
"""

import math
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

from adapters.market_data import create_market_data_adapter, timeframe_to_ms
from config import CONFIG
from db.db_handler import TradingDatabaseHandler
from notifications.discord import send_discord_signal
from research.feature_engine import FeatureConfig, build_canonical_features, feature_columns
from research.outcome_labeler import OutcomeLabelConfig, label_trade_outcomes
from risk.risk_manager import RiskValidationError, calculate_position_size

warnings.filterwarnings("ignore", category=FutureWarning)

FEATURE_CONFIG = FeatureConfig()


def _to_dataframe(candles) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "timestamp_ms": candle.timestamp_ms,
                "open": candle.open,
                "high": candle.high,
                "low": candle.low,
                "close": candle.close,
                "volume": candle.volume,
            }
            for candle in candles
        ]
    )


def _build_outcome_model(
    base_df: pd.DataFrame,
    informative: dict[str, pd.DataFrame],
):
    """Train the baseline classifier on net trade outcomes only."""
    features = build_canonical_features(
        base_df,
        informative=informative,
        cfg=FEATURE_CONFIG,
        base_timeframe=CONFIG.timeframe,
    )
    labels = label_trade_outcomes(
        base_df,
        OutcomeLabelConfig(
            horizon_bars=CONFIG.research_max_hold_bars,
            stop_distance_pct=CONFIG.min_stop_distance_pct,
            reward_risk_ratio=CONFIG.reward_risk_ratio,
            fee_bps_per_side=CONFIG.research_fee_bps_per_side,
            slippage_bps_per_side=CONFIG.research_slippage_bps_per_side,
        ),
    )

    columns = feature_columns(features)
    dataset = pd.concat([features, labels], axis=1)
    train_mask = dataset[columns].notna().all(axis=1) & dataset["trade_exit_bar"].ge(0)
    train = dataset.loc[train_mask].copy()

    if len(train) < 30:
        raise ValueError(f"Insufficient point-in-time labeled history: {len(train)} rows.")

    X = train[columns].to_numpy(dtype=float)
    y = train["trade_outcome"].to_numpy(dtype=int)
    if len(np.unique(y)) < 2:
        raise ValueError("Trade-outcome labels contain fewer than two classes.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    classifier = SGDClassifier(
        loss="log_loss",
        alpha=0.0001,
        max_iter=2000,
        tol=1e-4,
        random_state=42,
        class_weight="balanced",
    )
    classifier.fit(X_scaled, y)

    latest = features.iloc[[-1]]
    if latest[columns].isna().any(axis=None):
        raise ValueError("Latest canonical feature vector contains unavailable values.")

    latest_scaled = scaler.transform(latest[columns].to_numpy(dtype=float))
    probabilities = classifier.predict_proba(latest_scaled)[0]
    classes = classifier.classes_
    best_index = int(np.argmax(probabilities))
    predicted_class = int(classes[best_index])
    confidence = float(probabilities[best_index])

    return predicted_class, confidence, latest.iloc[0], float(train["trade_net_return"].mean())


def _signal_expiry(candle_timestamp_ms: int, timeframe_ms: int, validity_bars: int) -> datetime:
    candle_close_ms = candle_timestamp_ms + timeframe_ms
    expiry_ms = candle_close_ms + timeframe_ms * validity_bars
    return datetime.fromtimestamp(expiry_ms / 1000, tz=timezone.utc)


def run_nexus_cycle() -> None:
    db = TradingDatabaseHandler(CONFIG.db_path)
    adapter = create_market_data_adapter(CONFIG.market_data_exchange_id)

    generated = 0
    duplicates = 0
    risk_blocked = 0
    errors = 0
    neutral_skipped = 0
    timeframe_ms = timeframe_to_ms(CONFIG.timeframe)

    for symbol in CONFIG.symbols:
        try:
            base_candles = adapter.fetch_closed_ohlcv(
                symbol,
                CONFIG.timeframe,
                CONFIG.ohlcv_limit,
            )
            if not base_candles:
                raise ValueError("No closed candles returned.")

            informative: dict[str, pd.DataFrame] = {}
            for timeframe in CONFIG.mtf_timeframes:
                if timeframe == CONFIG.timeframe:
                    continue
                informative[timeframe] = _to_dataframe(
                    adapter.fetch_closed_ohlcv(
                        symbol,
                        timeframe,
                        CONFIG.mtf_ohlcv_limit,
                    )
                )

            base_df = _to_dataframe(base_candles)
            now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
            latest = base_candles[-1]
            if latest.timestamp_ms + timeframe_ms > now_ms:
                raise ValueError(f"Latest candle for {symbol} is not fully closed.")

            predicted_class, confidence, latest_features, mean_labeled_return = _build_outcome_model(
                base_df,
                informative,
            )

            if predicted_class == 0:
                neutral_skipped += 1
                print(f"ℹ️ No positive trade-outcome edge for {symbol}; signal skipped.")
                continue

            if not (
                math.isfinite(confidence)
                and 0.0 <= confidence <= 1.0
                and predicted_class in (-1, 1)
            ):
                raise ValueError(
                    f"Invalid outcome-model output for {symbol}: "
                    f"class={predicted_class}, confidence={confidence}"
                )

            side = "LONG" if predicted_class == 1 else "SHORT"
            entry = float(latest.close)
            atr_pct = float(latest_features["atr_pct"])
            stop_distance_pct = max(atr_pct, CONFIG.min_stop_distance_pct)
            if not math.isfinite(stop_distance_pct) or stop_distance_pct <= 0:
                raise ValueError(f"Invalid ATR-derived stop distance for {symbol}")

            move = entry * stop_distance_pct
            stop_loss = entry - move if side == "LONG" else entry + move
            take_profit = (
                entry + move * CONFIG.reward_risk_ratio
                if side == "LONG"
                else entry - move * CONFIG.reward_risk_ratio
            )

            signal_timestamp = datetime.now(timezone.utc)
            expires_at = _signal_expiry(
                latest.timestamp_ms,
                timeframe_ms,
                CONFIG.signal_validity_bars,
            )

            position_size = None
            risk_amount = None
            status = "ACTIVE"
            outcome = "PENDING"

            try:
                sized = calculate_position_size(
                    equity_usdt=CONFIG.account_equity_usdt,
                    risk_fraction=CONFIG.risk_per_trade,
                    entry_price=entry,
                    stop_loss=stop_loss,
                )
                position_size = sized.quantity
                risk_amount = sized.risk_amount_usdt
            except RiskValidationError as exc:
                status = "RISK_BLOCKED"
                outcome = "REJECTED_RISK"
                risk_blocked += 1
                print(f"⚠️ Risk blocked {symbol}: {exc}")

            signal_key = db.build_signal_key(
                strategy_id=CONFIG.strategy_id,
                symbol=symbol,
                timeframe=CONFIG.timeframe,
                candle_timestamp_ms=latest.timestamp_ms,
            )

            signal_id = db.insert_signal(
                {
                    "signal_key": signal_key,
                    "timestamp": signal_timestamp.isoformat(),
                    "symbol": symbol,
                    "signal_type": side,
                    "timeframe": CONFIG.timeframe,
                    "strategy_id": CONFIG.strategy_id,
                    "candle_timestamp_ms": latest.timestamp_ms,
                    "candle_closed": 1,
                    "entry": entry,
                    "sl": stop_loss,
                    "tp": take_profit,
                    "confidence": confidence,
                    "outcome": outcome,
                    "pred_move": mean_labeled_return,
                    "created_at": signal_timestamp.isoformat(),
                    "expires_at": expires_at.isoformat(),
                    "status": status,
                    "exchange": CONFIG.market_data_exchange_id,
                    "risk_per_trade": CONFIG.risk_per_trade,
                    "risk_amount_usdt": risk_amount,
                    "position_size": position_size,
                }
            )

            if signal_id is None:
                duplicates += 1
                print(
                    f"ℹ️ Duplicate suppressed: {symbol} "
                    f"{CONFIG.timeframe} candle={latest.timestamp_ms}"
                )
                continue

            generated += 1

            if status == "ACTIVE" and CONFIG.discord_webhook and position_size is not None:
                send_discord_signal(
                    CONFIG.discord_webhook,
                    symbol,
                    side,
                    entry,
                    stop_loss,
                    take_profit,
                    confidence,
                )

        except Exception as exc:
            errors += 1
            print(f"❌ Error {symbol}: {exc}")

    print(
        "✅ Cycle finished: "
        f"generated={generated}, duplicates_suppressed={duplicates}, "
        f"risk_blocked={risk_blocked}, neutral_skipped={neutral_skipped}, "
        f"errors={errors}"
    )


if __name__ == "__main__":
    run_nexus_cycle()
