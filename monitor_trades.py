import sqlite3
import os
import ccxt
from notifications.discord import send_discord_signal
from datetime import datetime

# --- CONFIG ---
DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK")
DB_PATH = "data/trading.db"

def check_outcomes():
    if not os.path.exists(DB_PATH):
        print("No database found. Skipping monitor.")
        return

    exchange = ccxt.gateio({"enableRateLimit": True})
    
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        # Only fetch trades that are still 'PENDING'
        cursor.execute("""
            SELECT id, symbol, signal_type, entry, sl, tp 
            FROM signals 
            WHERE outcome = 'PENDING'
        """)
        active_trades = cursor.fetchall()

        if not active_trades:
            print("No active pending trades to monitor.")
            return

        for db_id, symbol, side, entry, sl, tp in active_trades:
            try:
                ticker = exchange.fetch_ticker(symbol)
                last_price = ticker['last']
                status = None

                # Logic for LONG
                if side == "LONG":
                    if last_price <= sl: status = "STOP_LOSS"
                    elif last_price >= tp: status = "TAKE_PROFIT"
                
                # Logic for SHORT
                else:
                    if last_price >= sl: status = "STOP_LOSS"
                    elif last_price <= tp: status = "TAKE_PROFIT"

                if status:
                    # Update Database
                    cursor.execute("UPDATE signals SET outcome = ? WHERE id = ?", (status, db_id))
                    print(f"🎯 {symbol} hit {status} at ${last_price}")

                    # Send Alert
                    if DISCORD_WEBHOOK:
                        # Re-using your discord module with a 'Result' flavor
                        from requests import post
                        msg = {
                            "content": (
                                f"🏁 **TRADE CLOSED: {symbol}**\n"
                                f"**Result:** {status} {'✅' if status == 'TAKE_PROFIT' else '❌'}\n"
                                f"**Entry:** ${entry:,.4f}\n"
                                f"**Exit Price:** ${last_price:,.4f}"
                            )
                        }
                        post(DISCORD_WEBHOOK, json=msg)
            except Exception as e:
                print(f"Error monitoring {symbol}: {e}")
        
        conn.commit()

if __name__ == "__main__":
    check_outcomes()
