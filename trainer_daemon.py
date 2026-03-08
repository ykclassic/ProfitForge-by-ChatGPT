import sqlite3
import os
import ccxt
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier, SGDRegressor
from datetime import datetime, timezone
import warnings

# CRITICAL: Import your specific discord function
try:
    from notifications.discord import send_discord_signal
except ImportError:
    def send_discord_signal(*args, **kwargs):
        print("⚠️ Warning: notifications.discord module not found.")

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
warnings.filterwarnings("ignore", category=FutureWarning)

# =============================
# SCHEMA ENGINE
# =============================
def migrate_signals_schema(conn):
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT id FROM signals LIMIT 1")
    except sqlite3.OperationalError:
        print("⚠️ Rebuilding table for schema compliance...")
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
# PREDICTIVE ENGINE
# =============================
def run_nexus_cycle():
    # Using Gate.io for consistent OHLCV data
    exchange = ccxt.gateio({"enableRateLimit": True})
    
    with sqlite3.connect(DB_PATH) as conn:
        migrate_signals_schema(conn)
        ts = datetime.now(timezone.utc).isoformat()
        signals_generated = 0

        for symbol in SYMBOLS:
            try:
                # Fetch fresh data
                ohlcv = exchange.fetch_ohlcv(symbol, "1h", limit=100)
                df = pd.DataFrame(ohlcv, columns=["ts","o","h","l","c","v"])
                
                # Feature Engineering (Return and Volatility)
                df["ret"] = df["c"].pct_change().fillna(0)
                df["vol"] = (df["h"] - df["l"]) / df["c"]
                X = df[["ret", "vol"]].values
                X_scaled = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)
                
                # Target Labels: Binary (Up/Down) and Absolute Magnitude
                y_class = (df["ret"].shift(-1) > 0).astype(int).values[:-1]
                y_reg = df["ret"].shift(-1).abs().values[:-1]

                # Train Models
                clf = SGDClassifier(loss="log_loss").fit(X_scaled[:-1], y_class)
                
                # FIX: Added loss='epsilon_insensitive' to support 'pa1' learning rate
                reg = SGDRegressor(
                    loss='epsilon_insensitive', 
                    learning_rate='pa1', 
                    eta0=1.0, 
                    epsilon=0.01
                ).fit(X_scaled[:-1], y_reg)

                # Prediction for the next hour
                last_feat = X_scaled[-1].reshape(1, -1)
                prob_up = float(clf.predict_proba(last_feat)[0][1])
                pred_mag = float(reg.predict(last_feat)[0])

                side = "LONG" if prob_up > 0.5 else "SHORT"
                conf = prob_up if side == "LONG" else (1 - prob_up)
                entry = df["c"].iloc[-1]
                
                # Dynamic Risk Management based on predicted magnitude
                move = entry * max(pred_mag, 0.008) # Floor of 0.8%
                sl = (entry - move) if side == "LONG" else (entry + move)
                tp = (entry + move * 1.5) if side == "LONG" else (entry - move * 1.5)

                # Save to Database
                conn.execute("""
                    INSERT INTO signals (timestamp, symbol, signal_type, entry, sl, tp, confidence, outcome, pred_move)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (ts, symbol, side, entry, sl, tp, conf, 'PENDING', pred_mag))

                # Dispatch Notification
                if DISCORD_WEBHOOK:
                    send_discord_signal(DISCORD_WEBHOOK, symbol, side, entry, sl, tp, conf)
                
                signals_generated += 1

            except Exception as e:
                print(f"❌ Error {symbol}: {e}")

        conn.commit()

    # Touch file to force Git detection
    os.utime(DB_PATH, None)
    print(f"✅ Cycle Finished. Generated {signals_generated} signals.")

if __name__ == "__main__":
    run_nexus_cycle()
