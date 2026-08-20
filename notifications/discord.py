from __future__ import annotations

"""Discord notification helpers with bounded network timeouts."""

from typing import Mapping, Any

import requests


REQUEST_TIMEOUT_SECONDS = 10


def _post(webhook_url: str, payload: dict[str, Any]) -> None:
    response = requests.post(
        webhook_url,
        json=payload,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()


def send_discord_signal(
    webhook_url: str,
    symbol: str,
    direction: str,
    entry: float,
    stop_loss: float,
    take_profit: float,
    confidence: float,
) -> None:
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
    _post(webhook_url, payload)


def send_discord_outcome(
    webhook_url: str,
    signal: Mapping[str, Any],
    outcome: str,
    exit_price: float,
) -> None:
    result_icon = "✅" if outcome == "TAKE_PROFIT" else "❌"
    payload = {
        "content": (
            f"🏁 **TRADE CLOSED: {signal['symbol']}**\n"
            f"**Result:** {outcome} {result_icon}\n"
            f"**Direction:** {signal['signal_type']}\n"
            f"**Entry:** ${signal['entry']:,.4f}\n"
            f"**Exit Price:** ${exit_price:,.4f}"
        )
    }
    _post(webhook_url, payload)
