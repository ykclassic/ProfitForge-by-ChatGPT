import sqlite3
import os
import requests
import ccxt
from datetime import datetime

DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK")
DB_PATH = "data/trading.db"

SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "ADA/USDT",
    "XRP/USDT", "DOGE/USDT", "SUI/USDT", "LTC/USDT", "LINK/USDT"
]

os.makedirs("data", exist_ok=True)

def check_previous_outcomes(exchange, conn):
    cursor = conn.cursor()

    cursor.execute("""
        SELECT rowid, symbol, signal_type, sl, tp, entry
        FROM signals
        WHERE timestamp > datetime('now', '-2 hours')
        AND (outcome IS NULL OR outcome = '')
    """)

    recent_trades = cursor.fetchall()

    for rowid, symbol, side, sl, tp, entry in recent_trades:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, "1h", limit=2)

            if not ohlcv or len(ohlcv) < 2:
                print(f"No OHLCV data for {symbol}")
                continue

            high = ohlcv[-1][2]
            low = ohlcv[-1][3]
            current_price = ohlcv[-1][4]

            status = None

            if side == "LONG":
                if low <= sl:
                    status = "❌ STOP LOSS HIT"
                elif high >= tp:
                    status = "✅ TAKE PROFIT HIT"
            else:  # SHORT
                if high >= sl:
                    status = "❌ STOP LOSS HIT"
                elif low <= tp:
                    status = "✅ TAKE PROFIT HIT"

            if status:
                print(f"{symbol} outcome: {status}")

                cursor.execute(
                    "UPDATE signals SET outcome = ? WHERE rowid = ?",
                    (status, rowid)
                )
                conn.commit()

                if DISCORD_WEBHOOK:
                    msg = (
                        f"🔔 **Trade Outcome: {symbol}**\n"
                        f"Result: {status}\n"
                        f"Entry: ${entry:,.4f} | Final Price: ${current_price:,.4f}"
                    )
                    requests.post(DISCORD_WEBHOOK, json={"content": msg}, timeout=10)

        except Exception as e:
            print(f"Error checking outcome for {symbol}: {e}")

def send_discord_report(top_pick, all_signals):
    if not DISCORD_WEBHOOK:
        return

    report = f"🚀 **Nexus Alpha Report: {datetime.utcnow().strftime('%H:%M UTC')}**\n\n"
    report += f"🔥 **TOP PICK:** {top_pick['symbol']} | {top_pick['signal_type']} ({top_pick['confidence']:.2%} Conf)\n"
    report += f"🎯 TP: ${top_pick['tp']:,.4f} | SL: ${top_pick['sl']:,.4f}\n\n"
    report += "**Other Signals:** " + ", ".join(
        [f"{s['symbol']} ({s['signal_type']})" for s in all_signals[:5]]
    )

    requests.post(DISCORD_WEBHOOK, json={"content": report}, timeout=10)

def run_nexus_cycle():
    exchange = ccxt.gateio({
        "enableRateLimit": True,
        "timeout": 10000,
    })

    with sqlite3.connect(DB_PATH) as conn:
        check_previous_outcomes(exchange, conn)

    print("Nexus cycle completed successfully.")

if __name__ == "__main__":
    run_nexus_cycle()
