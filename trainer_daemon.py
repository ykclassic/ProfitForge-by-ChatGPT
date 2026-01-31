import sqlite3
import os
import requests
import ccxt
import pandas as pd
from sklearn.linear_model import SGDClassifier
from hmmlearn.hmm import GaussianHMM
from datetime import datetime, timezone

# --- Config ---
DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK")
DB_DIR = "data"
DB_PATH = os.path.join(DB_DIR, "trading.db")

if not os.path.exists(DB_DIR):
    os.makedirs(DB_DIR)

def migrate_db(conn):
    """Lead Dev Fix: Automatically adds missing columns to existing database."""
    cursor = conn.cursor()
    # Get existing columns
    cursor.execute("PRAGMA table_info(signals)")
    columns = [column[1] for column in cursor.fetchall()]
    
    # Check for missing columns and add them
    required_columns = {
        "signal_type": "TEXT",
        "sl": "REAL",
        "tp": "REAL"
    }
    
    for col_name, col_type in required_columns.items():
        if col_name not in columns:
            print(f"Migrating: Adding column {col_name} to signals table.")
            cursor.execute(f"ALTER TABLE signals ADD COLUMN {col_name} {col_type}")
    
    conn.commit()

def run_nexus_cycle():
    # 1. Fetch
    exchange = ccxt.gateio()
    ohlcv = exchange.fetch_ohlcv("BTC/USDT", "1h", limit=100)
    df = pd.DataFrame(ohlcv, columns=["ts","o","h","l","c","v"])
    df["ret"] = df["c"].pct_change().fillna(0)
    
    # 2. Regime Classify
    hmm = GaussianHMM(n_components=3, random_state=42)
    X = df[["ret"]].values
    hmm.fit(X)
    current_regime = int(hmm.predict(X)[-1])
    
    # 3. Predict
    model = SGDClassifier(loss="log_loss")
    y = (df["ret"].shift(-1) > 0).astype(int).values[:-1]
    X_train = X[:-1]
    model.partial_fit(X_train, y, classes=[0, 1])
    
    prob_up = float(model.predict_proba(X[-1].reshape(1,-1))[0][1])
    
    # Logic for Long/Short
    signal_type = "LONG" if prob_up >= 0.5 else "SHORT"
    confidence = prob_up if prob_up >= 0.5 else (1 - prob_up)

    entry = df["c"].iloc[-1]
    sl = entry * 0.99 if signal_type == "LONG" else entry * 1.01
    tp = entry * 1.02 if signal_type == "LONG" else entry * 0.98

    # 4. Persistence with Migration
    conn = sqlite3.connect(DB_PATH)
    
    # Ensure table exists first
    conn.execute('''CREATE TABLE IF NOT EXISTS signals 
                    (timestamp TEXT, entry REAL, confidence REAL, regime INTEGER)''')
    
    # Run migration to add new columns if they are missing
    migrate_db(conn)
    
    conn.execute("INSERT INTO signals (timestamp, entry, confidence, regime, signal_type, sl, tp) VALUES (?, ?, ?, ?, ?, ?, ?)", 
                 (datetime.now(timezone.utc).isoformat(), entry, confidence, 
                  current_regime, signal_type, sl, tp))
    
    conn.commit()
    conn.close()
    print(f"Cycle Complete: {signal_type} @ {entry}")

if __name__ == "__main__":
    run_nexus_cycle()
