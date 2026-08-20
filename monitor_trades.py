from __future__ import annotations

"""Run one finite outcome-monitoring cycle for ProfitForge signals.

P0 requirements:
- Process only ACTIVE/PENDING signals owned by the canonical database layer.
- Use fully closed Bitget candles only.
- Never use a live ticker to decide historical SL/TP ordering.
- Expire due signals before making network requests.
- Cache market-data requests within this cycle.
- Never wait for the next scheduled cycle; return after one pass.
"""

from datetime import datetime, timezone

from adapters.market_data import Candle, MarketDataError, create_market_data_adapter
from config import CONFIG
from db.db_handler import TradingDatabaseHandler
from notifications.discord import send_discord_outcome


def _parse_expiry(value: str) -> datetime:
    """Parse an ISO timestamp and normalize it to UTC."""
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _evaluate_signal(
    db: TradingDatabaseHandler,
    signal: dict,
    candles: list[Candle],
    now: datetime,
) -> str:
    """Evaluate one signal against closed post-entry candles."""
    expires_at = _parse_expiry(signal["expires_at"])

    for candle in candles:
        if candle.timestamp_ms <= signal["candle_timestamp_ms"]:
            continue
        if candle.close_datetime > expires_at:
            continue

        if signal["signal_type"] == "LONG":
            hit_sl = candle.low <= signal["sl"]
            hit_tp = candle.high >= signal["tp"]
        else:
            hit_sl = candle.high >= signal["sl"]
            hit_tp = candle.low <= signal["tp"]

        # OHLCV cannot establish whether SL or TP happened first inside one
        # candle, so do not invent an ordering.
        if hit_sl and hit_tp:
            db.mark_signal_outcome(
                signal["id"],
                outcome="AMBIGUOUS",
                outcome_price=None,
                outcome_at=candle.close_datetime.isoformat(),
                status="CLOSED_AMBIGUOUS",
            )
            return "ambiguous"

        if hit_sl:
            db.mark_signal_outcome(
                signal["id"],
                outcome="STOP_LOSS",
                outcome_price=signal["sl"],
                outcome_at=candle.close_datetime.isoformat(),
                status="CLOSED",
            )
            if CONFIG.discord_webhook:
                send_discord_outcome(
                    CONFIG.discord_webhook,
                    signal,
                    "STOP_LOSS",
                    signal["sl"],
                )
            return "closed"

        if hit_tp:
            db.mark_signal_outcome(
                signal["id"],
                outcome="TAKE_PROFIT",
                outcome_price=signal["tp"],
                outcome_at=candle.close_datetime.isoformat(),
                status="CLOSED",
            )
            if CONFIG.discord_webhook:
                send_discord_outcome(
                    CONFIG.discord_webhook,
                    signal,
                    "TAKE_PROFIT",
                    signal["tp"],
                )
            return "closed"

    if now >= expires_at:
        db.mark_signal_outcome(
            signal["id"],
            outcome="EXPIRED",
            outcome_price=None,
            outcome_at=now.isoformat(),
            status="EXPIRED",
        )
        return "expired"

    return "pending"


def check_outcomes() -> None:
    """Run exactly one bounded monitoring pass and then exit."""
    db = TradingDatabaseHandler(CONFIG.db_path)
    now = datetime.now(timezone.utc)

    # Expire stale rows before reading active signals. This prevents old
    # signals from causing unnecessary exchange requests and makes expiry
    # deterministic even when no new scheduler cycle has run recently.
    expired_before_fetch = db.expire_due_signals(now)
    active_signals = db.get_active_signals()

    if not active_signals:
        print(
            "✅ Outcome monitor finished: no active signals to monitor; "
            f"expired_before_fetch={expired_before_fetch}"
        )
        return

    adapter = create_market_data_adapter(CONFIG.market_data_exchange_id)
    candle_cache: dict[tuple[str, str], list[Candle]] = {}

    closed_count = 0
    expired_count = expired_before_fetch
    ambiguous_count = 0
    pending_count = 0
    errors = 0

    for signal_row in active_signals:
        signal = dict(signal_row)

        try:
            if not bool(signal["candle_closed"]):
                db.mark_signal_outcome(
                    signal["id"],
                    outcome="REJECTED_DATA",
                    outcome_price=None,
                    outcome_at=now.isoformat(),
                    status="REJECTED",
                )
                continue

            expires_at = _parse_expiry(signal["expires_at"])
            if now >= expires_at:
                db.mark_signal_outcome(
                    signal["id"],
                    outcome="EXPIRED",
                    outcome_price=None,
                    outcome_at=now.isoformat(),
                    status="EXPIRED",
                )
                expired_count += 1
                continue

            cache_key = (signal["symbol"], signal["timeframe"])
            if cache_key not in candle_cache:
                candle_cache[cache_key] = adapter.fetch_closed_ohlcv(
                    signal["symbol"],
                    signal["timeframe"],
                    CONFIG.ohlcv_limit,
                )

            result = _evaluate_signal(
                db,
                signal,
                candle_cache[cache_key],
                now,
            )

            if result == "closed":
                closed_count += 1
            elif result == "expired":
                expired_count += 1
            elif result == "ambiguous":
                ambiguous_count += 1
            elif result == "pending":
                pending_count += 1

        except (MarketDataError, ValueError, TypeError, KeyError) as exc:
            errors += 1
            print(
                f"❌ Error monitoring signal {signal['id']} "
                f"{signal['symbol']}: {exc}"
            )
        except Exception as exc:
            errors += 1
            print(
                f"❌ Unexpected error monitoring signal {signal['id']} "
                f"{signal['symbol']}: {exc}"
            )

    print(
        "✅ Outcome monitor finished: "
        f"closed={closed_count}, expired={expired_count}, "
        f"ambiguous={ambiguous_count}, pending={pending_count}, "
        f"errors={errors}"
    )


if __name__ == "__main__":
    check_outcomes()
