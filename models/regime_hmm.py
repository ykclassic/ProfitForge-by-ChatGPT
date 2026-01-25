import numpy as np
from hmmlearn.hmm import GaussianHMM

def detect_regimes(df):
    X = np.column_stack([
        df["log_return"],
        df["atr"],
        df["volatility"]
    ])

    hmm = GaussianHMM(
        n_components=3,
        covariance_type="full",
        n_iter=200
    )
    hmm.fit(X)

    regimes = hmm.predict(X)
    probs = hmm.predict_proba(X).max(axis=1)

    df["regime"] = regimes
    df["regime_prob"] = probs
    return df
