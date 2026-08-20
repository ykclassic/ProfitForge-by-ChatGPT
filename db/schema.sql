-- Canonical ProfitForge SQLite schema.
-- db/db_handler.py is the sole database owner and applies migrations.
-- PostgreSQL-specific types from the previous prototype are intentionally gone.

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

CREATE UNIQUE INDEX IF NOT EXISTS uq_signals_signal_key
    ON signals(signal_key);
