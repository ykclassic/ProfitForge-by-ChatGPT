from __future__ import annotations

"""Outcome monitor for ProfitForge signals.

P0 uses only fully closed Bitget candles. It never uses a live ticker to decide
whether a historical signal hit SL/TP, because ticker snapshots cannot reliably
establish intra-candle ordering.
"""

from datetime import datetime, timezone

from adapters.market_data import create_market_data_adapter
from config import CONFIG
from db.db_handler import TradingDatabaseHandler
from notifications.discord import send_discord_outcome


def check_outcomes() -> None:
    db = TradingDatabaseHandler(CONFIG.db_path)
    adapter = create_market_data_adapter(CONFIG.market_data_exchange_id)

    active_signals = db.get_active_signals()
    if not active_signals:
        print("No active signals to monitor.")
        return

    now = datetime.now(timezone.utc)
    closed_count = 0
    expired_count = 0
    ambiguous_count = 0
    errors = 0

    for signal in active_signals:
        try:
            if not signal["candle_closed"]:
                db.mark_signal_outcome(
                    signal["id"],
                    outcome="REJECTED_DATA",
                    outcome_price=None,
                    outcome_at=now.isoformat(),
                    status="REJECTED",
                )
                continue

            expires_at = datetime.fromisoformat(
                signal["expires_at"].replace("Z", "+00:00")
            )

            candles = adapter.fetch_closed_ohlcv(
                signal["symbol"],
                signal["timeframe"],
                CONFIG.ohlcv_limit,
            )

            post_signal = [
                candle
                for candle in candles
                if candle.timestamp_ms > signal["candle_timestamp_ms"]
                and candle.close_datetime <= expires_at
            ]

            resolved = False
            for candle in post_signal:
                if signal["signal_type"] == "LONG":
                    hit_sl = candle.low <= signal["sl"]
                    hit_tp = candle.high >= signal["tp"]
                else:
                    hit_sl = candle.high >= signal["sl"]
                    hit_tp = candle.low <= signal["tp"]

                if hit_sl and hit_tp:
                    db.mark_signal_outcome(
                        signal["id"],
                        outcome="AMBIGUOUS",
                        outcome_price=None,
                        outcome_at=candle.close_datetime.isoformat(),
                        status="CLOSED_AMBIGUOUS",
                    )
                    ambiguous_count += 1
                    resolved = True
                    break

                if hit_sl:
                    db.mark_signal_outcome(
                        signal["id"],
                        outcome="STOP_LOSS",
                        outcome_price=signal["sl"],
                        outcome_at=candle.close_datetime.isoformat(),
                        status="CLOSED",
                    )
                    closed_count += 1
                    resolved = True
                    if CONFIG.discord_webhook:
                        send_discord_outcome(
                            CONFIG.discord_webhook,
                            signal,
                            "STOP_LOSS",
                            signal["sl"],
                        )
                    break

                if hit_tp:
                    db.mark_signal_outcome(
                        signal["id"],
                        outcome="TAKE_PROFIT",
                        outcome_price=signal["tp"],
                        outcome_at=candle.close_datetime.isoformat(),
                        status="CLOSED",
                    )
                    closed_count += 1
                    resolved = True
                    if CONFIG.discord_webhook:
                        send_discord_outcome(
                            CONFIG.discord_webhook,
                            signal,
                            "TAKE_PROFIT",
                            signal["tp"],
                        )
                    break

            if not resolved and now >= expires_at:
                db.mark_signal_outcome(
                    signal["id"],
                    outcome="EXPIRED",
                    outcome_price=None,
                    outcome_at=now.isoformat(),
                    status="EXPIRED",
                )
                expired_count += 1

        except Exception as exc:
            errors += 1
            print(
                f"❌ Error monitoring signal {signal['id']} "
                f"{signal['symbol']}: {exc}"
            )

    print(
        "✅ Outcome monitor finished: "
        f"closed={closed_count}, expired={expired_count}, "
        f"ambiguous={ambiguous_count}, errors={errors}"
    )


if __name__ == "__main__":
    check_outcomes()
