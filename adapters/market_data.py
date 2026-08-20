from __future__ import annotations

"""Market-data adapters.

This module owns market-data access only. It must never place orders.
"""

from dataclasses import dataclass
from datetime import datetime, timezone

import ccxt


class MarketDataError(RuntimeError):
    """Raised when market data cannot be fetched or validated."""


@dataclass(frozen=True)
class Candle:
    timestamp_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def close_datetime(self) -> datetime:
        return datetime.fromtimestamp(
            self.timestamp_ms / 1000, tz=timezone.utc
        )


class MarketDataAdapter:
    """Abstract market-data contract."""

    exchange_id: str

    def fetch_closed_ohlcv(
        self, symbol: str, timeframe: str, limit: int
    ) -> list[Candle]:
        raise NotImplementedError


def timeframe_to_ms(timeframe: str) -> int:
    """Convert a CCXT-style timeframe such as 1m/1h/1d to milliseconds."""
    if not timeframe or len(timeframe) < 2:
        raise MarketDataError(f"Unsupported timeframe: {timeframe}")

    unit = timeframe[-1]
    try:
        value = int(timeframe[:-1])
    except ValueError as exc:
        raise MarketDataError(f"Unsupported timeframe: {timeframe}") from exc

    multipliers = {
        "s": 1_000,
        "m": 60_000,
        "h": 3_600_000,
        "d": 86_400_000,
        "w": 604_800_000,
    }

    if unit not in multipliers or value <= 0:
        raise MarketDataError(f"Unsupported timeframe: {timeframe}")

    return value * multipliers[unit]


class BitgetMarketDataAdapter(MarketDataAdapter):
    """Primary Bitget public market-data adapter."""

    exchange_id = "bitget"

    def __init__(self) -> None:
        self.exchange = ccxt.bitget({"enableRateLimit": True})

    def fetch_closed_ohlcv(
        self, symbol: str, timeframe: str, limit: int
    ) -> list[Candle]:
        if not self.exchange.has.get("fetchOHLCV"):
            raise MarketDataError("Bitget does not advertise fetchOHLCV support.")

        try:
            self.exchange.load_markets()
            raw = self.exchange.fetch_ohlcv(
                symbol,
                timeframe,
                limit=limit + 1,
            )
        except Exception as exc:
            raise MarketDataError(
                f"Bitget OHLCV fetch failed for {symbol} {timeframe}: {exc}"
            ) from exc

        if not raw:
            raise MarketDataError(f"Bitget returned no candles for {symbol}.")

        duration_ms = timeframe_to_ms(timeframe)
        now_ms = self.exchange.milliseconds()

        closed: list[Candle] = []
        for row in raw:
            if len(row) < 6:
                continue

            timestamp_ms = int(row[0])
            if timestamp_ms + duration_ms > now_ms:
                continue

            try:
                candle = Candle(
                    timestamp_ms=timestamp_ms,
                    open=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                    volume=float(row[5]),
                )
            except (TypeError, ValueError) as exc:
                raise MarketDataError(
                    f"Invalid OHLCV row for {symbol}: {row}"
                ) from exc

            if not (
                candle.open > 0
                and candle.high > 0
                and candle.low > 0
                and candle.close > 0
                and candle.high >= max(candle.open, candle.close)
                and candle.low <= min(candle.open, candle.close)
            ):
                raise MarketDataError(
                    f"Invalid OHLCV values for {symbol}: {row}"
                )

            closed.append(candle)

        if len(closed) < limit:
            raise MarketDataError(
                f"Only {len(closed)} closed candles available for {symbol}; "
                f"{limit} required."
            )

        return closed[-limit:]


def create_market_data_adapter(exchange_id: str) -> MarketDataAdapter:
    exchange_id = exchange_id.strip().lower()

    if exchange_id == "bitget":
        return BitgetMarketDataAdapter()

    raise ValueError(
        f"Unsupported market-data exchange '{exchange_id}'. "
        "P0 permits Bitget only; failover is intentionally deferred."
    )
