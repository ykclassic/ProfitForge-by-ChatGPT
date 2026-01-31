import time
import sqlite3
import requests
import ccxt
import numpy as np
import pandas as pd
import os
from sklearn.linear_model import SGDClassifier
from hmmlearn.hmm import GaussianHMM
from river.drift import ADWIN
from datetime import datetime, timezone

# --- Configuration ---
DISCORD_WEBHOOK = "YOUR_DISCORD_WEBHOOK"
DB_DIR = "data"
DB_PATH = os.path.join(DB_DIR, "trading.db")

# Ensure directory exists
if not os.path.exists(DB_DIR):
    os.makedirs(DB_DIR)

# --- Model Initialization ---
exchange = ccxt.gateio()
model = SGDClassifier(loss="log_loss")
drift_detector = ADWIN()
hmm = GaussianHMM(n_components=3, covariance_type="full", random_state=42)

def init_db():
    """Initializes the database schema if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS signals (
            timestamp TEXT,
            symbol TEXT,
            entry REAL,
            sl REAL,
            tp REAL,
            confidence REAL,
            regime INTEGER,
            signal_type TEXT
        )
    ''')
    conn.commit()
    conn.close()

def send_discord(msg):
    if DISCORD_WEBHOOK == "YOUR_DISCORD_WEBHOOK":
        return
    try:
        requests.post(DISCORD_WEBHOOK, json={"content": msg})
    except Exception as e:
        print(f"Discord Error: {e}")

def fetch_data(symbol="BTC/USDT", timeframe="1h", limit=300):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["ts","o","h","l","c","v"])
    df["ret"] = df["c"].pct_change()
    return df.dropna()

def classify_regime(df):
    X = df[["ret"]].values
    hmm.fit(X)
    df["regime"] = hmm.predict(X)
    return df

def save_signal(data):
    """Lead Dev Addition: Persist signal to SQLite for the Dashboard."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO signals (timestamp, symbol, entry, sl, tp, confidence, regime, signal_type)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        datetime.now(timezone.utc).isoformat(),
        "BTC/USDT",
        data['entry'],
        data['sl'],
        data['tp'],
        data['confidence'],
        int(data['regime']),
        "LONG" if data['confidence'] > 0.5 else "SHORT"
    ))
    conn.commit()
    conn.close()

def train_and_signal():
    try:
        df = fetch_data()
        df = classify_regime(df)

        X = df[["ret"]].values
        # Target: 1 if next candle is positive, else 0
        y = (df["ret"].shift(-1) > 0).astype(int).values[:-1]
        X_train = X[:-1]

        # Partial fit for online learning
        model.partial_fit(X_train, y, classes=[0, 1])
        
        # Latest prediction
        current_x = X[-1].reshape(1, -1)
        probs = model.predict_proba(current_x)[0][1]

        # Drift detection
        drift_detector.update(int(probs < 0.5))
        if drift_detector.drift_detected:
            send_discord("⚠️ **Nexus Alert**: Concept drift detected. Model adapting.")

        entry = df["c"].iloc[-1]
        stop = entry * 0.99
        tp = entry * 1.02
        current_regime = df['regime'].iloc[-1]

        signal_data = {
            'entry': entry,
            'sl': stop,
            'tp': tp,
            'confidence': probs,
            'regime': current_regime
        }

        # Save to DB for Streamlit
        save_signal(signal_data)

        # Notify Discord
        msg = (
            f"📈 **BTC Signal**\n"
            f"Entry: {entry:.2f}\n"
            f"SL: {stop:.2f}\n"
            f"TP: {tp:.2f}\n"
            f"Confidence: {probs:.2%}\n"
            f"Regime: {current_regime}"
        )
        send_discord(msg)
        print(f"[{datetime.now()}] Signal processed and saved.")

    except Exception as e:
        print(f"Error in training loop: {e}")

def main():
    print("Starting Nexus HybridTrader Daemon...")
    init_db()
    while True:
        train_and_signal()
        # Sleep for 1 hour (3600s)
        time.sleep(3600)

if __name__ == "__main__":
    main()
