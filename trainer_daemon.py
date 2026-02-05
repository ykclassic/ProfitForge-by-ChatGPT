import sqlite3
import os
import requests
import ccxt
from datetime import datetime, timezone

# =============================
# CONFIG
# =============================
DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK")
DB_PATH = "data/trading.db"

SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "ADA/USDT",
    "XRP/USDT", "DOGE/USDT", "SUI/USDT", "LTC/USDT", "LINK/USDT"
]

os.makedirs("data", exist_ok=True)

# =============================
# SCHEMA MIGRATION (OPTION A)
# =============================
def get_columns(cursor, table):
    cursor.execute(f"PRAGMA table_info({table})")
    return [row[1] for row in cursor.fetchall()]

def migrate_signals_schema(conn):
    cursor = conn.cursor()

    # Base table (minimal, backward-compatible)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT
        )
    """)

    existing = get_columns(cursor, "signals")

    required_columns = {
        "symbol": "TEXT",
        "signal_type": "TEXT",
        "entry": "REAL",
        "sl": "REAL",
        "tp": "REAL",
        "confidence": "REAL",
        "outcome": "TEXT"
    }

    for col, col_type in required_columns.items():
        if col not in existing:
            cursor.execute(
                f"ALTER TABLE signals ADD COLUMN {col} {col_type}"
            )

    conn.commit()

# =============================
# OUTCOME CHECKING
# =============================
def check_previous_outcomes(exchange, conn):
    cursor = conn.cursor()

    cursor.execute("""
        SELECT rowid, symbol, signal_type, sl, tp, entry
        FROM signals
        WHERE timestamp > datetime('now', '-2 hours')
        AND (outcome IS NULL OR outcome = '')
    """)

    recent_trades = cursor.fetchall()

    for rowid, symbol, side, sl, tp, entry in recent_trades:
        if not symbol or not side:
            continue

        try:
            ohlcv = exchange.fetch_ohlcv(symbol, "1h", limit=2)

            if not ohlcv or len(ohlcv) < 2:
                continue

            high = ohlcv[-1][2]
            low = ohlcv[-1][3]
            close = ohlcv[-1][4]

            status = None

            if side == "LONG":
                if low <= sl:
                    status = "STOP_LOSS"
                elif high >= tp:
                    status = "TAKE_PROFIT"
            elif side == "SHORT":
                if high >= sl:
                    status = "STOP_LOSS"
                elif low <= tp:
                    status = "TAKE_PROFIT"

            if status:
                cursor.execute(
                    "UPDATE signals SET outcome = ? WHERE rowid = ?",
                    (status, rowid)
                )
                conn.commit()

                if DISCORD_WEBHOOK:
                    msg = (
                        f"🔔 **Trade Outcome**\n"
                        f"Symbol: {symbol}\n"
                        f"Result: {status}\n"
                        f"Entry: ${entry:,.4f}\n"
                        f"Last Price: ${close:,.4f}"
                    )
                    requests.post(
                        DISCORD_WEBHOOK,
                        json={"content": msg},
                        timeout=10
                    )

        except Exception as e:
            print(f"[WARN] Outcome check failed for {symbol}: {e}")

# =============================
# MAIN CYCLE
# =============================
def run_nexus_cycle():
    exchange = ccxt.gateio({
        "enableRateLimit": True,
        "timeout": 10000,
    })

    with sqlite3.connect(DB_PATH) as conn:
        migrate_signals_schema(conn)
        check_previous_outcomes(exchange, conn)

    print("✅ Nexus cycle completed successfully.")

if __name__ == "__main__":
    run_nexus_cycle()
