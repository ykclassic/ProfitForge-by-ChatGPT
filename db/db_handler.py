from __future__ import annotations

"""Canonical SQLite database owner for ProfitForge.

All application writes to trading.db must go through this module.
"""

import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = ROOT_DIR / "data" / "trading.db"


class TradingDatabaseHandler:
    """Owns SQLite connection, schema initialization, migrations and writes."""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA busy_timeout = 30000;")
        # The database is Git-tracked; DELETE journaling avoids untracked
        # -wal/-shm sidecar files in scheduled GitHub Actions runs.
        conn.execute("PRAGMA journal_mode = DELETE;")
        return conn

    def _initialize_schema(self) -> None:
        with self._get_connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_key TEXT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    timeframe TEXT NOT NULL DEFAULT '1h',
                    strategy_id TEXT NOT NULL DEFAULT 'baseline_ml_v1',
                    candle_timestamp_ms INTEGER,
                    candle_closed INTEGER NOT NULL DEFAULT 1,
                    entry REAL NOT NULL,
                    sl REAL NOT NULL,
                    tp REAL NOT NULL,
                    confidence REAL NOT NULL,
                    outcome TEXT NOT NULL DEFAULT 'PENDING',
                    outcome_price REAL,
                    outcome_at TEXT,
                    pred_move REAL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'ACTIVE',
                    exchange TEXT NOT NULL DEFAULT 'bitget',
                    risk_per_trade REAL,
                    risk_amount_usdt REAL,
                    position_size REAL
                );

                CREATE INDEX IF NOT EXISTS idx_signals_symbol_status
                    ON signals(symbol, status);

                CREATE INDEX IF NOT EXISTS idx_signals_expiry
                    ON signals(expires_at);

                CREATE INDEX IF NOT EXISTS idx_signals_candle
                    ON signals(symbol, timeframe, candle_timestamp_ms);

                CREATE INDEX IF NOT EXISTS idx_signals_created
                    ON signals(created_at);
                """
            )

            self._migrate_legacy_columns(conn)
            self._backfill_legacy_rows(conn)
            self._ensure_signal_key_uniqueness(conn)

    @staticmethod
    def _migrate_legacy_columns(conn: sqlite3.Connection) -> None:
        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(signals)").fetchall()
        }

        additions: dict[str, str] = {
            "signal_key": "TEXT",
            "timeframe": "TEXT NOT NULL DEFAULT '1h'",
            "strategy_id": "TEXT NOT NULL DEFAULT 'baseline_ml_v1'",
            "candle_timestamp_ms": "INTEGER",
            "candle_closed": "INTEGER NOT NULL DEFAULT 1",
            "outcome_price": "REAL",
            "outcome_at": "TEXT",
            "created_at": "TEXT",
            "expires_at": "TEXT",
            "status": "TEXT NOT NULL DEFAULT 'ACTIVE'",
            "exchange": "TEXT NOT NULL DEFAULT 'bitget'",
            "risk_per_trade": "REAL",
            "risk_amount_usdt": "REAL",
            "position_size": "REAL",
        }

        for name, definition in additions.items():
            if name not in columns:
                conn.execute(
                    f"ALTER TABLE signals ADD COLUMN {name} {definition}"
                )

    @staticmethod
    def _backfill_legacy_rows(conn: sqlite3.Connection) -> None:
        rows = conn.execute(
            """
            SELECT id, timestamp, symbol
            FROM signals
            WHERE signal_key IS NULL
               OR created_at IS NULL
               OR expires_at IS NULL
               OR candle_timestamp_ms IS NULL
            """
        ).fetchall()

        for row in rows:
            timestamp = row["timestamp"] or datetime.now(timezone.utc).isoformat()
            try:
                created = datetime.fromisoformat(
                    timestamp.replace("Z", "+00:00")
                )
            except ValueError:
                created = datetime.now(timezone.utc)

            candle_timestamp_ms = int(created.timestamp() * 1000)
            expires_at = datetime.fromtimestamp(
                created.timestamp() + 3600,
                tz=timezone.utc,
            ).isoformat()
            key = TradingDatabaseHandler.build_signal_key(
                strategy_id="baseline_ml_v1",
                symbol=row["symbol"],
                timeframe="1h",
                candle_timestamp_ms=candle_timestamp_ms,
            )

            conn.execute(
                """
                UPDATE signals
                SET signal_key = COALESCE(signal_key, ?),
                    created_at = COALESCE(created_at, ?),
                    expires_at = COALESCE(expires_at, ?),
                    timeframe = COALESCE(timeframe, '1h'),
                    strategy_id = COALESCE(strategy_id, 'baseline_ml_v1'),
                    candle_timestamp_ms = COALESCE(candle_timestamp_ms, ?),
                    candle_closed = COALESCE(candle_closed, 1),
                    status = COALESCE(status, 'ACTIVE'),
                    exchange = COALESCE(exchange, 'bitget')
                WHERE id = ?
                """,
                (
                    key,
                    timestamp,
                    expires_at,
                    candle_timestamp_ms,
                    row["id"],
                ),
            )

    def _ensure_signal_key_uniqueness(self, conn: sqlite3.Connection) -> None:
        duplicate_rows = conn.execute(
            """
            SELECT signal_key, GROUP_CONCAT(id) AS ids
            FROM signals
            WHERE signal_key IS NOT NULL
            GROUP BY signal_key
            HAVING COUNT(*) > 1
            """
        ).fetchall()

        for duplicate in duplicate_rows:
            ids = [int(value) for value in duplicate["ids"].split(",")]
            for row_id in ids[1:]:
                conn.execute(
                    """
                    UPDATE signals
                    SET signal_key = signal_key || '|legacy-' || id
                    WHERE id = ?
                    """,
                    (row_id,),
                )

        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS uq_signals_signal_key
            ON signals(signal_key)
            """
        )

    @staticmethod
    def build_signal_key(
        *,
        strategy_id: str,
        symbol: str,
        timeframe: str,
        candle_timestamp_ms: int,
    ) -> str:
        raw = f"{strategy_id}|{symbol}|{timeframe}|{candle_timestamp_ms}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def insert_signal(self, signal: dict[str, Any]) -> int | None:
        """Insert a signal once; return None when its signal key already exists."""
        required = (
            "signal_key",
            "timestamp",
            "symbol",
            "signal_type",
            "timeframe",
            "strategy_id",
            "candle_timestamp_ms",
            "entry",
            "sl",
            "tp",
            "confidence",
            "created_at",
            "expires_at",
            "exchange",
        )
        missing = [field for field in required if field not in signal]
        if missing:
            raise ValueError(f"Missing signal fields: {', '.join(missing)}")

        columns = [
            "signal_key",
            "timestamp",
            "symbol",
            "signal_type",
            "timeframe",
            "strategy_id",
            "candle_timestamp_ms",
            "candle_closed",
            "entry",
            "sl",
            "tp",
            "confidence",
            "outcome",
            "pred_move",
            "created_at",
            "expires_at",
            "status",
            "exchange",
            "risk_per_trade",
            "risk_amount_usdt",
            "position_size",
        ]
        values = [signal.get(column) for column in columns]
        placeholders = ", ".join("?" for _ in columns)
        column_sql = ", ".join(columns)

        with self._get_connection() as conn:
            cursor = conn.execute(
                f"""
                INSERT INTO signals ({column_sql})
                VALUES ({placeholders})
                ON CONFLICT(signal_key) DO NOTHING
                """,
                values,
            )
            return int(cursor.lastrowid) if cursor.rowcount else None

    def get_active_signals(self) -> list[sqlite3.Row]:
        with self._get_connection() as conn:
            return conn.execute(
                """
                SELECT *
                FROM signals
                WHERE status = 'ACTIVE'
                  AND outcome = 'PENDING'
                ORDER BY id ASC
                """
            ).fetchall()

    def mark_signal_outcome(
        self,
        signal_id: int,
        *,
        outcome: str,
        outcome_price: float | None,
        outcome_at: str,
        status: str,
    ) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE signals
                SET outcome = ?,
                    outcome_price = ?,
                    outcome_at = ?,
                    status = ?
                WHERE id = ?
                """,
                (outcome, outcome_price, outcome_at, status, signal_id),
            )

    def expire_due_signals(self, now: datetime) -> int:
        now_iso = now.astimezone(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                UPDATE signals
                SET outcome = 'EXPIRED',
                    outcome_at = ?,
                    status = 'EXPIRED'
                WHERE status = 'ACTIVE'
                  AND outcome = 'PENDING'
                  AND expires_at <= ?
                """,
                (now_iso, now_iso),
            )
            return int(cursor.rowcount)

    def get_latest_signal_status(
        self, symbol: str, strategy_id: str = "baseline_ml_v1"
    ) -> dict[str, Any] | None:
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM signals
                WHERE symbol = ?
                  AND strategy_id = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (symbol, strategy_id),
            ).fetchone()
            return dict(row) if row else None
