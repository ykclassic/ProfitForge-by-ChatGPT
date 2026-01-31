import sqlite3
import os
import requests
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# --- Config ---
DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK")
DB_PATH = "data/trading.db"
SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "ADA/USDT", 
           "XRP/USDT", "DOGE/USDT", "SUI/USDT", "LTC/USDT", "LINK/USDT"]

def check_previous_outcomes(exchange, conn):
    """Lead Dev Fix: Scans last hour's signals to see if SL or TP was hit."""
    cursor = conn.cursor()
    # Find signals from the last 2 hours that don't have a recorded outcome yet
    cursor.execute("SELECT rowid, symbol, signal_type, sl, tp, entry FROM signals WHERE timestamp > datetime('now', '-2 hours')")
    recent_trades = cursor.fetchall()
    
    for rowid, symbol, side, sl, tp, entry in recent_trades:
        try:
            # Fetch the actual price movement since that signal
            ohlcv = exchange.fetch_ohlcv(symbol, "1h", limit=2)
            high = ohlcv[-1][2]
            low = ohlcv[-1][3]
            current_price = ohlcv[-1][4]

            status = None
            if side == "LONG":
                if low <= sl: status = "❌ STOP LOSS HIT"
                elif high >= tp: status = "✅ TAKE PROFIT HIT"
            else: # SHORT
                if high <= sl: status = "❌ STOP LOSS HIT" # For short, SL is above entry
                elif low >= tp: status = "✅ TAKE PROFIT HIT"

            if status and DISCORD_WEBHOOK:
                msg = f"🔔 **Trade Outcome: {symbol}**\nResult: {status}\nEntry: ${entry:,.4f} | Final Price: ${current_price:,.4f}"
                requests.post(DISCORD_WEBHOOK, json={"content": msg})
                # We could delete or mark the row here, but for now, we just alert.
        except Exception as e:
            print(f"Error checking outcome for {symbol}: {e}")

def send_discord_report(top_pick, all_signals):
    if not DISCORD_WEBHOOK: return
    report = f"🚀 **Nexus Alpha Report: {datetime.now().strftime('%H:%M UTC')}**\n\n"
    report += f"🔥 **TOP PICK:** {top_pick['symbol']} | {top_pick['signal_type']} ({top_pick['confidence']:.2%} Conf)\n"
    report += f"🎯 TP: ${top_pick['tp']:,.4f} | SL: ${top_pick['sl']:,.4f}\n\n"
    report += "**Other Signals:** " + ", ".join([f"{s['symbol']} ({s['signal_type']})" for s in all_signals[:5]])
    requests.post(DISCORD_WEBHOOK, json={"content": report})

def run_nexus_cycle():
    exchange = ccxt.gateio()
    conn = sqlite3.connect(DB_PATH)
    
    # 1. Recon: Check what happened to our last alerts
    check_previous_outcomes(exchange, conn)
    
    # 2. Generate New Signals (Existing logic continues here...)
    # [Rest of your training/prediction logic remains the same]
    # ...
    # Ensure you are passing 'current_signals' and 'top_pick' to send_discord_report()
