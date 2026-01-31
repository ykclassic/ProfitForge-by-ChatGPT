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
    confidence = float(model.predict_proba(X[-1].reshape(1,-1))[0][1])

    # 4. Persistence
    conn = sqlite3.connect(DB_PATH)
    conn.execute("CREATE TABLE IF NOT EXISTS signals (timestamp TEXT, entry REAL, confidence REAL, regime INTEGER)")
    conn.execute("INSERT INTO signals VALUES (?, ?, ?, ?)", 
                 (datetime.now(timezone.utc).isoformat(), df["c"].iloc[-1], confidence, current_regime))
    conn.commit()
    conn.close()
    print(f"Cycle Complete. Confidence: {confidence:.2%}")

if __name__ == "__main__":
    run_nexus_cycle()
