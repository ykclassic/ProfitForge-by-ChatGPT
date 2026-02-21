import sqlite3
import os
import requests
import ccxt
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier, SGDRegressor
from datetime import datetime, timezone
import warnings

# =============================
# CONFIG
# =============================
DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK")
DB_PATH = "data/trading.db"
SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "ADA/USDT",
    "XRP/USDT", "DOGE/USDT", "SUI/USDT", "LTC/USDT", "LINK/USDT"
]

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

# Silence deprecation warnings for cleaner GitHub logs
warnings.filterwarnings("ignore", category=FutureWarning)

# =============================
# SCHEMA REPAIR ENGINE
# =============================
def migrate_signals_schema(conn):
    """
    Maintains existing data while ensuring the table has the correct 
    structure (including the PRIMARY KEY 'id' and 'pred_move').
    """
    cursor = conn.cursor()
    try:
        # Check if 'id' exists by trying to select it
        cursor.execute("SELECT id FROM signals LIMIT 1")
    except sqlite3.OperationalError:
        print("⚠️ Schema mismatch detected (missing 'id'). Rebuilding table...")
        # Rename old table to preserve data
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
# OUTCOME CHECKING
# =============================
def check_previous_outcomes(exchange, conn):
    """
    Checks if PENDING trades from the last 3 hours have hit SL or TP.
    """
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, symbol, signal_type, sl, tp, entry
        FROM signals
        WHERE outcome = 'PENDING'
        AND timestamp > datetime('now', '-3 hours')
    """)
    recent_trades = cursor.fetchall()

    for db_id, symbol, side, sl, tp, entry in recent_trades:
        try:
            # Fetch last 2 candles to check price action
            ohlcv = exchange.fetch_ohlcv(symbol, "1h", limit=2)
            if not ohlcv: continue
            
            high, low, close = ohlcv[-1][2], ohlcv[-1][3], ohlcv[-1][4]
            status = None

            if side == "LONG":
                if low <= sl: status = "STOP_LOSS"
                elif high >= tp: status = "TAKE_PROFIT"
            else: # SHORT
                if high >= sl: status = "STOP_LOSS"
                elif low <= tp: status = "TAKE_PROFIT"

            if status:
                cursor.execute("UPDATE signals SET outcome = ? WHERE id = ?", (status, db_id))
                if DISCORD_WEBHOOK:
                    msg = (
                        f"🔔 **Nexus Trade Update**\n"
                        f"Asset: {symbol} | Result: {status}\n"
                        f"Entry: ${entry:,.4f} | Final: ${close:,.4f}"
                    )
                    requests.post(DISCORD_WEBHOOK, json={"content": msg}, timeout=5)
        except Exception as e:
            print(f"Outcome check error for {symbol}: {e}")
    conn.commit()

# =============================
# PREDICTIVE ENGINE
# =============================
def run_nexus_cycle():
    """
    Fetches live market data, trains micro-models per asset, 
    and generates new trading signals.
    """
    # Use Gate.io as the primary data source
    exchange = ccxt.gateio({"enableRateLimit": True})
    
    with sqlite3.connect(DB_PATH) as conn:
        # 1. Ensure DB is healthy
        migrate_signals_schema(conn)
        
        # 2. Update status of existing trades
        check_previous_outcomes(exchange, conn)
        
        # 3. Generate New Signals
        ts = datetime.now(timezone.utc).isoformat()
        signals_count = 0

        for symbol in SYMBOLS:
            try:
                # Fetch OHLCV data with cache-busting nonce
                ohlcv = exchange.fetch_ohlcv(symbol, "1h", limit=100, params={'nonce': exchange.milliseconds()})
                df = pd.DataFrame(ohlcv, columns=["ts","o","h","l","c","v"])
                
                # Feature Engineering
                df["ret"] = df["c"].pct_change().fillna(0)
                df["vol"] = (df["h"] - df["l"]) / df["c"]
                X = df[["ret", "vol"]].values
                X_scaled = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)

                # Target Labels
                y_class = (df["ret"].shift(-1) > 0).astype(int).values[:-1]
                y_reg = df["ret"].shift(-1).abs().values[:-1]

                # Train Online Learners
                clf = SGDClassifier(loss="log_loss").fit(X_scaled[:-1], y_class)
                reg = SGDRegressor(
                    loss='epsilon_insensitive', 
                    penalty=None, 
                    learning_rate='pa1', 
                    eta0=1.0
                ).fit(X_scaled[:-1], y_reg)

                # Predict Next Candle
                last_feat = X_scaled[-1].reshape(1, -1)
                prob_up = float(clf.predict_proba(last_feat)[0][1])
                pred_mag = float(reg.predict(last_feat)[0])

                # Determine Signal Type
                side = "LONG" if prob_up > 0.5 else "SHORT"
                conf = prob_up if side == "LONG" else (1 - prob_up)
                entry = df["c"].iloc[-1]
                
                # Risk Management (SL/TP)
                move = entry * max(pred_mag, 0.008) # Floor of 0.8% volatility
                sl = (entry - move) if side == "LONG" else (entry + move)
                tp = (entry + move * 1.5) if side == "LONG" else (entry - move * 1.5)

                # Save Signal
                conn.execute("""
                    INSERT INTO signals (timestamp, symbol, signal_type, entry, sl, tp, confidence, outcome, pred_move)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (ts, symbol, side, entry, sl, tp, conf, 'PENDING', pred_mag))
                
                signals_count += 1

            except Exception as e:
                print(f"❌ Failed to process {symbol}: {e}")

        conn.commit()

    # FORCE UPDATE: Change file metadata so Git detects a change even if data is similar
    os.utime(DB_PATH, None)
    print(f"✅ Cycle Finished. Generated {signals_count} signals. DB Size: {os.path.getsize(DB_PATH)} bytes")

if __name__ == "__main__":
    run_nexus_cycle()
