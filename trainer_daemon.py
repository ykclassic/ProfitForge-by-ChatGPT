import sqlite3
import os
import requests
import ccxt
import pandas as pd
import numpy as np
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
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(signals)")
    columns = [column[1] for column in cursor.fetchall()]
    
    required_columns = {
        "signal_type": "TEXT",
        "sl": "REAL",
        "tp": "REAL",
        "atr": "REAL" # Added ATR for volatility tracking
    }
    
    for col_name, col_type in required_columns.items():
        if col_name not in columns:
            cursor.execute(f"ALTER TABLE signals ADD COLUMN {col_name} {col_type}")
    conn.commit()

def run_nexus_cycle():
    # 1. Fetch High-Precision Data
    exchange = ccxt.gateio()
    ohlcv = exchange.fetch_ohlcv("BTC/USDT", "1h", limit=200)
    df = pd.DataFrame(ohlcv, columns=["ts","o","h","l","c","v"])
    
    # Calculate Returns and ATR (Volatility)
    df["ret"] = df["c"].pct_change().fillna(0)
    high_low = df["h"] - df["l"]
    high_close = (df["h"] - df["c"].shift()).abs()
    low_close = (df["l"] - df["c"].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df["atr"] = true_range.rolling(14).mean().fillna(method='bfill')

    # 2. Stable Regime Classify
    X = df[["ret"]].values
    hmm = GaussianHMM(n_components=3, covariance_type="full", random_state=42)
    hmm.fit(X)
    
    # Lead Dev Fix: Sort regimes by variance so 0 is always 'Low Vol'
    regime_map = np.argsort(np.diagonal(hmm.covars_).mean(axis=1).flatten())
    raw_regimes = hmm.predict(X)
    # Map the raw output to our stable labels
    stable_regimes = [np.where(regime_map == r)[0][0] for r in raw_regimes]
    current_regime = int(stable_regimes[-1])
    
    # 3. Predict Direction
    model = SGDClassifier(loss="log_loss")
    y = (df["ret"].shift(-1) > 0).astype(int).values[:-1]
    X_train = X[:-1]
    model.partial_fit(X_train, y, classes=[0, 1])
    
    prob_up = float(model.predict_proba(X[-1].reshape(1,-1))[0][1])
    signal_type = "LONG" if prob_up >= 0.5 else "SHORT"
    confidence = prob_up if prob_up >= 0.5 else (1 - prob_up)

    # 4. Volatility-Adjusted Risk (2x ATR for SL, 4x ATR for TP)
    entry = df["c"].iloc[-1]
    current_atr = df["atr"].iloc[-1]
    
    if signal_type == "LONG":
        sl = entry - (current_atr * 2)
        tp = entry + (current_atr * 4)
    else:
        sl = entry + (current_atr * 2)
        tp = entry - (current_atr * 4)

    # 5. Persistence
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''CREATE TABLE IF NOT EXISTS signals 
                    (timestamp TEXT, entry REAL, confidence REAL, regime INTEGER)''')
    migrate_db(conn)
    
    conn.execute("""INSERT INTO signals (timestamp, entry, confidence, regime, signal_type, sl, tp, atr) 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)""", 
                 (datetime.now(timezone.utc).isoformat(), entry, confidence, 
                  current_regime, signal_type, sl, tp, current_atr))
    
    conn.commit()
    conn.close()
    print(f"Cycle Complete: {signal_type} | Regime: {current_regime} | ATR: {current_atr:.2f}")

if __name__ == "__main__":
    run_nexus_cycle()
