# trainer_daemon.py
import time
import sqlite3
import requests
import ccxt
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from hmmlearn.hmm import GaussianHMM
from river.drift import ADWIN
from datetime import datetime, timezone

DISCORD_WEBHOOK = "YOUR_DISCORD_WEBHOOK"

DB_PATH = "data/trading.db"

exchange = ccxt.gateio()

model = SGDClassifier(loss="log_loss")
meta_model = SGDClassifier(loss="log_loss")
drift_detector = ADWIN()

hmm = GaussianHMM(n_components=3, covariance_type="full")

def send_discord(msg):
    requests.post(DISCORD_WEBHOOK, json={"content": msg})

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

def train_and_signal():
    df = fetch_data()
    df = classify_regime(df)

    X = df[["ret"]].values
    y = (df["ret"].shift(-1) > 0).astype(int).values[:-1]
    X = X[:-1]

    model.partial_fit(X, y, classes=[0,1])
    probs = model.predict_proba(X)[-1][1]

    drift_detector.update(int(probs < 0.5))
    if drift_detector.drift_detected:
        send_discord("⚠️ Concept drift detected. Model adapting.")

    entry = df["c"].iloc[-1]
    stop = entry * 0.99
    tp = entry * 1.02

    msg = (
        f"📈 **BTC Signal**\n"
        f"Entry: {entry:.2f}\n"
        f"SL: {stop:.2f}\n"
        f"TP: {tp:.2f}\n"
        f"Confidence: {probs:.2%}\n"
        f"Regime: {df['regime'].iloc[-1]}"
    )
    send_discord(msg)

def main():
    while True:
        train_and_signal()
        time.sleep(3600)

if __name__ == "__main__":
    main()
