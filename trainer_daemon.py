import time
import numpy as np
import pandas as pd
from models.online_model import OnlineModel
from models.regime_hmm import detect_regimes
from risk.drift_adwin import DriftDetector
from risk.bet_sizing import fractional_kelly

model = OnlineModel()
drift = DriftDetector()

while True:
    df = pd.read_csv("latest_features.csv", index_col=0, parse_dates=True)

    df = detect_regimes(df)

    X = df[["open","high","low","close","volume"]].values
    y = (df["close"].shift(-1) > df["close"]).astype(int).values[:-1]

    model.update(X[:-1], y)

    probs = model.predict_proba(X[-1].reshape(1, -1))[0][1]
    error = int(probs < 0.5)

    if drift.update(error):
        print("⚠️ Concept drift detected — suppressing signals")

    bet = fractional_kelly(probs, avg_win=1.2, avg_loss=1.0)
    print(f"Confidence={probs:.3f} | Bet size={bet:.3f}")

    time.sleep(60)
