from __future__ import annotations

"""ProfitForge hourly signal trainer.

P0 responsibilities:
- use Bitget as the sole market-data source;
- consume only fully closed candles;
- write through the canonical database owner;
- generate at most one signal per strategy/symbol/timeframe/candle;
- attach an explicit expiry;
- calculate risk and position size before a signal can become ACTIVE.

This module does not place live orders.
"""

import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.preprocessing import StandardScaler

from adapters.market_data import create_market_data_adapter, timeframe_to_ms
from config import CONFIG
from db.db_handler import TradingDatabaseHandler
from notifications.discord import send_discord_signal
from risk.risk_manager import RiskValidationError, calculate_position_size

warnings.filterwarnings("ignore", category=FutureWarning)


def _to_dataframe(candles) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ts": candle.timestamp_ms,
                "o": candle.open,
                "h": candle.high,
                "l": candle.low,
                "c": candle.close,
                "v": candle.volume,
            }
            for candle in candles
        ]
    )


def _build_models(df: pd.DataFrame):
    """Train the existing baseline models without adding a new strategy."""
    df = df.copy()
    df["ret"] = df["c"].pct_change().fillna(0.0)
    df["vol"] = (df["h"] - df["l"]) / df["c"]

    feature_columns = ["ret", "vol"]
    supervised = df.iloc[:-1].copy()

    if len(supervised) < 20:
        raise ValueError("Insufficient closed-candle history for model training.")

    X = supervised[feature_columns].to_numpy(dtype=float)
    y_class = (
        df["ret"].shift(-1).iloc[:-1].to_numpy(dtype=float) > 0
    ).astype(int)
    y_reg = df["ret"].shift(-1).abs().iloc[:-1].to_numpy(dtype=float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = SGDClassifier(loss="log_loss", random_state=42)
    clf.fit(X_scaled, y_class)

    reg = SGDRegressor(
        loss="epsilon_insensitive",
        learning_rate="pa1",
        eta0=1.0,
        epsilon=0.01,
        random_state=42,
    )
    reg.fit(X_scaled, y_reg)

    latest = df.iloc[-1]
    latest_features = scaler.transform(
        [[float(latest["ret"]), float(latest["vol"])] ]
    )

    probability_up = float(clf.predict_proba(latest_features)[0][1])
    predicted_magnitude = float(reg.predict(latest_features)[0])

    return probability_up, predicted_magnitude


def _signal_expiry(
    candle_timestamp_ms: int,
    timeframe_ms: int,
    validity_bars: int,
) -> datetime:
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

    timeframe_ms = timeframe_to_ms(CONFIG.timeframe)

    for symbol in CONFIG.symbols:
        try:
            candles = adapter.fetch_closed_ohlcv(
                symbol,
                CONFIG.timeframe,
                CONFIG.ohlcv_limit,
            )

            if not candles:
                raise ValueError("No closed candles returned.")

            # The adapter guarantees closed candles. Recheck the invariant here
            # so a future adapter cannot silently violate the trading contract.
            now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
            latest = candles[-1]
            if latest.timestamp_ms + timeframe_ms > now_ms:
                raise ValueError(f"Latest candle for {symbol} is not fully closed.")

            df = _to_dataframe(candles)
            probability_up, predicted_magnitude = _build_models(df)

            side = "LONG" if probability_up > 0.5 else "SHORT"
            confidence = probability_up if side == "LONG" else 1.0 - probability_up
            entry = float(latest.close)

            # Preserve the current baseline stop model for P0, but move the
            # account-risk calculation into the dedicated risk layer.
            move = entry * max(
                abs(predicted_magnitude), CONFIG.min_stop_distance_pct
            )
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
                # Do not invent an account balance or position size.
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
                    "pred_move": predicted_magnitude,
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

            if (
                status == "ACTIVE"
                and CONFIG.discord_webhook
                and position_size is not None
            ):
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
        f"risk_blocked={risk_blocked}, errors={errors}"
    )


if __name__ == "__main__":
    run_nexus_cycle()
