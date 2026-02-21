import sqlite3
import os
import requests
import ccxt
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier, PassiveAggressiveRegressor
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
# SCHEMA REPAIR ENGINE
# =============================
def migrate_signals_schema(conn):
    cursor = conn.cursor()
    try:
        # Check if 'id' exists by trying to select it
        cursor.execute("SELECT id FROM signals LIMIT 1")
    except sqlite3.OperationalError:
        print("⚠️ Schema mismatch detected (missing 'id'). Rebuilding table...")
        # Rename old table to preserve data just in case, or just drop it
        cursor.execute("DROP TABLE IF EXISTS signals_old")
        cursor.execute("ALTER TABLE signals RENAME TO signals_old")
        
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            symbol TEXT,
            signal_type TEXT,
            entry REAL,
            sl REAL,
            tp REAL,
            confidence REAL,
            outcome TEXT DEFAULT 'PENDING',
            pred_move REAL
        )
    """)
    conn.commit()

# =============================
# OUTCOME CHECKING
# =============================
def check_previous_outcomes(exchange, conn):
    cursor = conn.cursor()
    # Using 'id' safely now
    cursor.execute("""
        SELECT id, symbol, signal_type, sl, tp, entry
        FROM signals
        WHERE outcome = 'PENDING'
        AND timestamp > datetime('now', '-3 hours')
    """)
    recent_trades = cursor.fetchall()

    for db_id, symbol, side, sl, tp, entry in recent_trades:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, "1h", limit=2)
            if not ohlcv: continue
            high, low, close = ohlcv[-1][2], ohlcv[-1][3], ohlcv[-1][4]
            status = None

            if side == "LONG":
                if low <= sl: status = "STOP_LOSS"
                elif high >= tp: status = "TAKE_PROFIT"
            else: # SHORT
                if high >= sl: status = "STOP_LOSS"
                elif low <= tp: status = "TAKE_PROFIT"

            if status:
                cursor.execute("UPDATE signals SET outcome = ? WHERE id = ?", (status, db_id))
                if DISCORD_WEBHOOK:
                    msg = f"🔔 **Nexus Outcome: {symbol}**\nResult: {status}\nEntry: ${entry:,.4f} | Exit: ${close:,.4f}"
                    requests.post(DISCORD_WEBHOOK, json={"content": msg}, timeout=5)
        except: continue
    conn.commit()

# =============================
# PREDICTIVE ENGINE
# =============================
def run_nexus_cycle():
    exchange = ccxt.gateio({"enableRateLimit": True})
    conn = sqlite3.connect(DB_PATH)
    
    # 1. Fix Schema
    migrate_signals_schema(conn)
    
    # 2. Check Outcomes
    check_previous_outcomes(exchange, conn)
    
    # 3. Generate Signals
    ts = datetime.now(timezone.utc).isoformat()
    for symbol in SYMBOLS:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, "1h", limit=100)
            df = pd.DataFrame(ohlcv, columns=["ts","o","h","l","c","v"])
            
            df["ret"] = df["c"].pct_change().fillna(0)
            df["vol"] = (df["h"] - df["l"]) / df["c"]
            X = df[["ret", "vol"]].values
            X_scaled = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)

            # Train micro-models
            y_class = (df["ret"].shift(-1) > 0).astype(int).values[:-1]
            y_reg = df["ret"].shift(-1).abs().values[:-1]

            clf = SGDClassifier(loss="log_loss").fit(X_scaled[:-1], y_class)
            reg = PassiveAggressiveRegressor().fit(X_scaled[:-1], y_reg)

            last_feat = X_scaled[-1].reshape(1, -1)
            prob_up = float(clf.predict_proba(last_feat)[0][1])
            pred_mag = float(reg.predict(last_feat)[0])

            side = "LONG" if prob_up > 0.5 else "SHORT"
            conf = prob_up if side == "LONG" else (1 - prob_up)
            entry = df["c"].iloc[-1]
            
            move = entry * max(pred_mag, 0.008) 
            sl = (entry - move) if side == "LONG" else (entry + move)
            tp = (entry + move * 1.5) if side == "LONG" else (entry - move * 1.5)

            conn.execute("""
                INSERT INTO signals (timestamp, symbol, signal_type, entry, sl, tp, confidence, outcome, pred_move)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (ts, symbol, side, entry, sl, tp, conf, 'PENDING', pred_mag))

        except Exception as e:
            print(f"Error on {symbol}: {e}")

    conn.commit()
    conn.close()
    print(f"✅ Cycle Finished. File updated: {os.path.getsize(DB_PATH)} bytes")

if __name__ == "__main__":
    run_nexus_cycle()
