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
# SCHEMA MIGRATION
# =============================
def migrate_signals_schema(conn):
    cursor = conn.cursor()
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
            outcome TEXT,
            pred_move REAL
        )
    """)
    conn.commit()

# =============================
# OUTCOME CHECKING
# =============================
def check_previous_outcomes(exchange, conn):
    cursor = conn.cursor()
    # Look back at the last 3 hours of signals that are still 'PENDING'
    cursor.execute("""
        SELECT id, symbol, signal_type, sl, tp, entry
        FROM signals
        WHERE (outcome IS NULL OR outcome = 'PENDING')
        AND timestamp > datetime('now', '-3 hours')
    """)
    recent_trades = cursor.fetchall()

    for db_id, symbol, side, sl, tp, entry in recent_trades:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, "1h", limit=2)
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
                    msg = f"🔔 **Trade Update: {symbol}**\nResult: {status}\nEntry: ${entry:,.4f} | Exit: ${close:,.4f}"
                    requests.post(DISCORD_WEBHOOK, json={"content": msg})
        except: continue
    conn.commit()

# =============================
# PREDICTIVE ENGINE
# =============================
def run_nexus_cycle():
    exchange = ccxt.gateio({"enableRateLimit": True})
    conn = sqlite3.connect(DB_PATH)
    migrate_signals_schema(conn)
    
    # 1. Clean up old trades first
    check_previous_outcomes(exchange, conn)
    
    current_signals = []
    ts = datetime.now(timezone.utc).isoformat()

    # 2. Generate New Signals
    for symbol in SYMBOLS:
        try:
            # Fetch fresh data (params to bypass cache)
            ohlcv = exchange.fetch_ohlcv(symbol, "1h", limit=200, params={'nonce': exchange.milliseconds()})
            df = pd.DataFrame(ohlcv, columns=["ts","o","h","l","c","v"])
            
            # Feature Engineering
            df["ret"] = df["c"].pct_change().fillna(0)
            df["vol"] = (df["h"] - df["l"]) / df["c"]
            X = df[["ret", "vol"]].values
            X_scaled = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)

            # Simple ML Logic
            y_class = (df["ret"].shift(-1) > 0).astype(int).values[:-1]
            y_reg = df["ret"].shift(-1).abs().values[:-1]

            clf = SGDClassifier(loss="log_loss").fit(X_scaled[:-1], y_class)
            reg = PassiveAggressiveRegressor().fit(X_scaled[:-1], y_reg)

            # Prediction
            last_feat = X_scaled[-1].reshape(1, -1)
            prob_up = float(clf.predict_proba(last_feat)[0][1])
            pred_mag = float(reg.predict(last_feat)[0])

            side = "LONG" if prob_up > 0.5 else "SHORT"
            conf = prob_up if side == "LONG" else (1 - prob_up)
            entry = df["c"].iloc[-1]
            
            # SL/TP calculation
            move = entry * max(pred_mag, 0.005) # Min 0.5% move
            sl = (entry - move) if side == "LONG" else (entry + move)
            tp = (entry + move * 1.5) if side == "LONG" else (entry - move * 1.5)

            # Insert into DB
            conn.execute("""
                INSERT INTO signals (timestamp, symbol, signal_type, entry, sl, tp, confidence, outcome, pred_move)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (ts, symbol, side, entry, sl, tp, conf, 'PENDING', pred_mag))
            
            current_signals.append({"symbol": symbol, "side": side, "conf": conf})

        except Exception as e:
            print(f"❌ Error processing {symbol}: {e}")

    conn.commit()
    conn.close()
    
    # Force Git to see the change by updating the file timestamp
    os.utime(DB_PATH, None)
    
    print(f"✅ Nexus cycle completed. Generated {len(current_signals)} signals.")

if __name__ == "__main__":
    run_nexus_cycle()
