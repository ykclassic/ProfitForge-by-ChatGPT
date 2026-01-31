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
DB_PATH = "data/trading.db"

if not os.path.exists("data"):
    os.makedirs("data")

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
    
    # Probability of price going UP
    prob_up = float(model.predict_proba(X[-1].reshape(1,-1))[0][1])
    
    # Determine Signal Type
    signal_type = "LONG" if prob_up >= 0.5 else "SHORT"
    confidence = prob_up if prob_up >= 0.5 else (1 - prob_up)

    # Calculate SL/TP (1% SL, 2% TP)
    entry = df["c"].iloc[-1]
    if signal_type == "LONG":
        sl = entry * 0.99
        tp = entry * 1.02
    else:
        sl = entry * 1.01
        tp = entry * 0.98

    # 4. Persistence
    conn = sqlite3.connect(DB_PATH)
    # Lead Dev Note: Explicitly adding columns for SL, TP, and Signal
    conn.execute('''CREATE TABLE IF NOT EXISTS signals 
                    (timestamp TEXT, entry REAL, confidence REAL, 
                     regime INTEGER, signal_type TEXT, sl REAL, tp REAL)''')
    
    conn.execute("INSERT INTO signals VALUES (?, ?, ?, ?, ?, ?, ?)", 
                 (datetime.now(timezone.utc).isoformat(), entry, confidence, 
                  current_regime, signal_type, sl, tp))
    conn.commit()
    conn.close()
    print(f"Cycle Complete. {signal_type} at {entry} (Conf: {confidence:.2%})")

if __name__ == "__main__":
    run_nexus_cycle()
