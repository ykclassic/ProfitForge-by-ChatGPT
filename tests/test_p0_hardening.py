from datetime import datetime, timezone

import pytest

from db.db_handler import TradingDatabaseHandler
from risk.risk_manager import RiskValidationError, calculate_position_size


def test_position_size_uses_account_risk():
    sizing = calculate_position_size(
        equity_usdt=10_000,
        risk_fraction=0.0075,
        entry_price=100,
        stop_loss=95,
    )

    assert sizing.risk_amount_usdt == pytest.approx(75)
    assert sizing.stop_distance == pytest.approx(5)
    assert sizing.quantity == pytest.approx(15)


def test_position_size_rejects_missing_equity():
    with pytest.raises(RiskValidationError):
        calculate_position_size(
            equity_usdt=0,
            risk_fraction=0.0075,
            entry_price=100,
            stop_loss=95,
        )


def test_signal_key_is_deterministic():
    key1 = TradingDatabaseHandler.build_signal_key(
        strategy_id="baseline_ml_v1",
        symbol="BTC/USDT",
        timeframe="1h",
        candle_timestamp_ms=123456789,
    )
    key2 = TradingDatabaseHandler.build_signal_key(
        strategy_id="baseline_ml_v1",
        symbol="BTC/USDT",
        timeframe="1h",
        candle_timestamp_ms=123456789,
    )

    assert key1 == key2


def test_duplicate_signal_is_suppressed(tmp_path):
    db = TradingDatabaseHandler(tmp_path / "trading.db")
    signal = {
        "signal_key": db.build_signal_key(
            strategy_id="baseline_ml_v1",
            symbol="BTC/USDT",
            timeframe="1h",
            candle_timestamp_ms=123456789,
        ),
        "timestamp": "2026-08-20T10:00:00+00:00",
        "symbol": "BTC/USDT",
        "signal_type": "LONG",
        "timeframe": "1h",
        "strategy_id": "baseline_ml_v1",
        "candle_timestamp_ms": 123456789,
        "candle_closed": 1,
        "entry": 100,
        "sl": 99,
        "tp": 101.5,
        "confidence": 0.75,
        "outcome": "PENDING",
        "pred_move": 0.01,
        "created_at": "2026-08-20T10:00:00+00:00",
        "expires_at": "2026-08-20T11:00:00+00:00",
        "status": "ACTIVE",
        "exchange": "bitget",
        "risk_per_trade": 0.0075,
        "risk_amount_usdt": 75,
        "position_size": 75,
    }

    first_id = db.insert_signal(signal)
    second_id = db.insert_signal(signal)

    assert first_id is not None
    assert second_id is None


def test_expired_signal_is_marked_expired(tmp_path):
    db = TradingDatabaseHandler(tmp_path / "trading.db")
    signal = {
        "signal_key": "expiry-test",
        "timestamp": "2026-08-20T08:00:00+00:00",
        "symbol": "ETH/USDT",
        "signal_type": "SHORT",
        "timeframe": "1h",
        "strategy_id": "baseline_ml_v1",
        "candle_timestamp_ms": 123,
        "candle_closed": 1,
        "entry": 100,
        "sl": 101,
        "tp": 98.5,
        "confidence": 0.7,
        "outcome": "PENDING",
        "pred_move": 0.01,
        "created_at": "2026-08-20T08:00:00+00:00",
        "expires_at": "2026-08-20T09:00:00+00:00",
        "status": "ACTIVE",
        "exchange": "bitget",
        "risk_per_trade": 0.0075,
        "risk_amount_usdt": 75,
        "position_size": 75,
    }

    db.insert_signal(signal)
    changed = db.expire_due_signals(
        datetime(2026, 8, 20, 9, 1, tzinfo=timezone.utc)
    )

    assert changed == 1
    row = db.get_latest_signal_status("ETH/USDT")
    assert row["outcome"] == "EXPIRED"
    assert row["status"] == "EXPIRED"
