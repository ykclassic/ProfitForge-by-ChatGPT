import requests

def send_discord_signal(
    webhook_url,
    symbol,
    direction,
    entry,
    stop_loss,
    take_profit,
    confidence
):
    payload = {
        "content": (
            f"📊 **TRADE SIGNAL**\n"
            f"**Symbol:** {symbol}\n"
            f"**Direction:** {direction}\n\n"
            f"**Entry:** {entry:.4f}\n"
            f"**Stop Loss:** {stop_loss:.4f}\n"
            f"**Take Profit:** {take_profit:.4f}\n\n"
            f"**Confidence:** {confidence:.2%}"
        )
    }

    response = requests.post(webhook_url, json=payload)
    response.raise_for_status()
