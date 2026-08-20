from __future__ import annotations

"""Central runtime configuration for ProfitForge.

P0 establishes Bitget as the sole primary market-data source.
Execution remains deliberately disabled until a separate execution gateway is
implemented and validated.
"""

import os
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
DB_PATH = DATA_DIR / "trading.db"

MARKET_DATA_EXCHANGE_ID = os.getenv("MARKET_DATA_EXCHANGE_ID", "bitget").strip().lower()
EXECUTION_EXCHANGE_ID = os.getenv("EXECUTION_EXCHANGE_ID", "").strip().lower() or None

TIMEFRAME = os.getenv("SIGNAL_TIMEFRAME", "1h")
OHLCV_LIMIT = int(os.getenv("OHLCV_LIMIT", "100"))
SIGNAL_VALIDITY_BARS = int(os.getenv("SIGNAL_VALIDITY_BARS", "1"))

# Risk sizing is paper/research-only until an execution gateway is explicitly enabled.
# No live order is placed by this P0 implementation.
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.0075"))
ACCOUNT_EQUITY_USDT = float(os.getenv("ACCOUNT_EQUITY_USDT", "0"))

MIN_STOP_DISTANCE_PCT = float(os.getenv("MIN_STOP_DISTANCE_PCT", "0.008"))
REWARD_RISK_RATIO = float(os.getenv("REWARD_RISK_RATIO", "1.5"))

STRATEGY_ID = os.getenv("STRATEGY_ID", "baseline_ml_v1")

SYMBOLS = tuple(
    symbol.strip()
    for symbol in os.getenv(
        "SYMBOLS",
        "BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT,ADA/USDT,"
        "XRP/USDT,DOGE/USDT,SUI/USDT,LTC/USDT,LINK/USDT",
    ).split(",")
    if symbol.strip()
)

DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK")

if not 0 < RISK_PER_TRADE < 1:
    raise ValueError("RISK_PER_TRADE must be greater than 0 and less than 1.")

if SIGNAL_VALIDITY_BARS < 1:
    raise ValueError("SIGNAL_VALIDITY_BARS must be at least 1.")

if OHLCV_LIMIT < 20:
    raise ValueError("OHLCV_LIMIT must be at least 20.")

if MIN_STOP_DISTANCE_PCT <= 0:
    raise ValueError("MIN_STOP_DISTANCE_PCT must be greater than 0.")

if REWARD_RISK_RATIO <= 0:
    raise ValueError("REWARD_RISK_RATIO must be greater than 0.")


@dataclass(frozen=True)
class RuntimeConfig:
    db_path: Path = DB_PATH
    market_data_exchange_id: str = MARKET_DATA_EXCHANGE_ID
    execution_exchange_id: str | None = EXECUTION_EXCHANGE_ID
    timeframe: str = TIMEFRAME
    ohlcv_limit: int = OHLCV_LIMIT
    signal_validity_bars: int = SIGNAL_VALIDITY_BARS
    risk_per_trade: float = RISK_PER_TRADE
    account_equity_usdt: float = ACCOUNT_EQUITY_USDT
    min_stop_distance_pct: float = MIN_STOP_DISTANCE_PCT
    reward_risk_ratio: float = REWARD_RISK_RATIO
    strategy_id: str = STRATEGY_ID
    symbols: tuple[str, ...] = SYMBOLS
    discord_webhook: str | None = DISCORD_WEBHOOK


CONFIG = RuntimeConfig()
