from __future__ import annotations

"""Central runtime configuration for ProfitForge.

P0 establishes Bitget as the sole primary market-data source.
P1 adds canonical multi-timeframe research configuration and explicit
transaction-cost/slippage assumptions. Execution remains deliberately
separate from research and is not enabled by these settings.
"""

import os
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
DB_PATH = DATA_DIR / "trading.db"


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return float(raw)


MARKET_DATA_EXCHANGE_ID = os.getenv("MARKET_DATA_EXCHANGE_ID", "bitget").strip().lower()
EXECUTION_EXCHANGE_ID = os.getenv("EXECUTION_EXCHANGE_ID", "").strip().lower() or None

TIMEFRAME = os.getenv("SIGNAL_TIMEFRAME", "1h")
MTF_TIMEFRAMES = tuple(
    value.strip()
    for value in os.getenv("MTF_TIMEFRAMES", "15m,4h,1d").split(",")
    if value.strip()
)
OHLCV_LIMIT = int(os.getenv("OHLCV_LIMIT", "100"))
MTF_OHLCV_LIMIT = int(os.getenv("MTF_OHLCV_LIMIT", "100"))
SIGNAL_VALIDITY_BARS = int(os.getenv("SIGNAL_VALIDITY_BARS", "1"))

# Risk sizing is paper/research-only until an execution gateway is explicitly enabled.
RISK_PER_TRADE = _env_float("RISK_PER_TRADE", 0.0075)
ACCOUNT_EQUITY_USDT = _env_float("ACCOUNT_EQUITY_USDT", 10000.0)

MIN_STOP_DISTANCE_PCT = _env_float("MIN_STOP_DISTANCE_PCT", 0.008)
REWARD_RISK_RATIO = _env_float("REWARD_RISK_RATIO", 1.5)

# Research friction assumptions. These are charged on both entry and exit.
RESEARCH_FEE_BPS_PER_SIDE = _env_float("RESEARCH_FEE_BPS_PER_SIDE", 5.0)
RESEARCH_SLIPPAGE_BPS_PER_SIDE = _env_float("RESEARCH_SLIPPAGE_BPS_PER_SIDE", 2.0)
RESEARCH_MAX_HOLD_BARS = int(os.getenv("RESEARCH_MAX_HOLD_BARS", "6"))

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
if OHLCV_LIMIT < 20 or MTF_OHLCV_LIMIT < 20:
    raise ValueError("OHLCV limits must be at least 20.")
if MIN_STOP_DISTANCE_PCT <= 0:
    raise ValueError("MIN_STOP_DISTANCE_PCT must be greater than 0.")
if REWARD_RISK_RATIO <= 0:
    raise ValueError("REWARD_RISK_RATIO must be greater than 0.")
if RESEARCH_FEE_BPS_PER_SIDE < 0 or RESEARCH_SLIPPAGE_BPS_PER_SIDE < 0:
    raise ValueError("Research fee/slippage cannot be negative.")
if RESEARCH_MAX_HOLD_BARS < 1:
    raise ValueError("RESEARCH_MAX_HOLD_BARS must be at least 1.")


@dataclass(frozen=True)
class RuntimeConfig:
    db_path: Path = DB_PATH
    market_data_exchange_id: str = MARKET_DATA_EXCHANGE_ID
    execution_exchange_id: str | None = EXECUTION_EXCHANGE_ID
    timeframe: str = TIMEFRAME
    mtf_timeframes: tuple[str, ...] = MTF_TIMEFRAMES
    ohlcv_limit: int = OHLCV_LIMIT
    mtf_ohlcv_limit: int = MTF_OHLCV_LIMIT
    signal_validity_bars: int = SIGNAL_VALIDITY_BARS
    risk_per_trade: float = RISK_PER_TRADE
    account_equity_usdt: float = ACCOUNT_EQUITY_USDT
    min_stop_distance_pct: float = MIN_STOP_DISTANCE_PCT
    reward_risk_ratio: float = REWARD_RISK_RATIO
    research_fee_bps_per_side: float = RESEARCH_FEE_BPS_PER_SIDE
    research_slippage_bps_per_side: float = RESEARCH_SLIPPAGE_BPS_PER_SIDE
    research_max_hold_bars: int = RESEARCH_MAX_HOLD_BARS
    strategy_id: str = STRATEGY_ID
    symbols: tuple[str, ...] = SYMBOLS
    discord_webhook: str | None = DISCORD_WEBHOOK


CONFIG = RuntimeConfig()
