import sqlite3
import os
import requests
import ccxt
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier, PassiveAggressiveRegressor
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
        "signal_type": "TEXT", "sl": "REAL", "tp": "REAL", 
        "atr": "REAL", "pred_move": "REAL"
    }
    
    for col_name, col_type in required_columns.items():
        if col_name not in columns:
            cursor.execute(f"ALTER TABLE signals ADD COLUMN {col_name} {col_type}")
    conn.commit()

def run_nexus_cycle():
    # 1. Fetch High-Precision Data
    exchange = ccxt.gateio()
    ohlcv = exchange.fetch_ohlcv("BTC/USDT", "1h", limit=250)
    df = pd.DataFrame(ohlcv, columns=["ts","o","h","l","c","v"])
    
    # Feature Engineering
    df["ret"] = df["c"].pct_change().fillna(0)
    df["volatility"] = (df["h"] - df["l"]) / df["c"]
    
    # 2. Stable Regime Classify
    X = df[["ret", "volatility"]].values
    hmm = GaussianHMM(n_components=3, random_state=42)
    hmm.fit(X)
    regime_map = np.argsort(np.diagonal(hmm.covars_).mean(axis=1).flatten())
    current_regime = int(np.where(regime_map == hmm.predict(X)[-1])[0][0])
    
    # 3. PREDICTIVE ENGINE
    # Model A: Direction (Classifier)
    # Model B: Magnitude (Regressor - predicting the next high/low range)
    clf = SGDClassifier(loss="log_loss")
    reg = PassiveAggressiveRegressor(max_iter=1000, tol=1e-3)
    
    y_class = (df["ret"].shift(-1) > 0).astype(int).values[:-1]
    y_reg = df["ret"].shift(-1).abs().values[:-1] # Predicting absolute magnitude of move
    
    X_train = X[:-1]
    clf.partial_fit(X_train, y_class, classes=[0, 1])
    reg.partial_fit(X_train, y_reg)
    
    # Forecasts for the NEXT candle
    next_x = X[-1].reshape(1, -1)
    prob_up = float(clf.predict_proba(next_x)[0][1])
    predicted_magnitude = float(reg.predict(next_x)[0])
    
    signal_type = "LONG" if prob_up >= 0.5 else "SHORT"
    confidence = prob_up if prob_up >= 0.5 else (1 - prob_up)

    # 4. Predictive Risk Management
    # Instead of ATR, we use the model's predicted magnitude for the next candle
    entry = df["c"].iloc[-1]
    expected_move_price = entry * predicted_magnitude
    
    if signal_type == "LONG":
        tp = entry + (expected_move_price * 1.5) # Aim for 1.5x the predicted move
        sl = entry - expected_move_price         # Stop at 1x the predicted move
    else:
        tp = entry - (expected_move_price * 1.5)
        sl = entry + expected_move_price

    # 5. Persistence
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''CREATE TABLE IF NOT EXISTS signals 
                    (timestamp TEXT, entry REAL, confidence REAL, regime INTEGER)''')
    migrate_db(conn)
    
    conn.execute("""INSERT INTO signals (timestamp, entry, confidence, regime, signal_type, sl, tp, pred_move) 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)""", 
                 (datetime.now(timezone.utc).isoformat(), entry, confidence, 
                  current_regime, signal_type, sl, tp, predicted_magnitude))
    
    conn.commit()
    conn.close()
    print(f"Forecasted {signal_type} with {predicted_magnitude:.2%} expected move.")

if __name__ == "__main__":
    run_nexus_cycle()
