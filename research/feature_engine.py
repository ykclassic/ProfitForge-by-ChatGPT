from __future__ import annotations

"""Canonical, point-in-time feature engine for ProfitForge research.

Features are calculated from completed candles. Multi-timeframe alignment is
performed on candle-close timestamps, not candle-open timestamps, so a higher
timeframe candle cannot leak its eventual close into lower-timeframe rows that
occurred while that candle was still forming.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureConfig:
    ema_fast: int = 20
    ema_slow: int = 50
    atr_period: int = 14
    adx_period: int = 14
    rsi_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0
    structure_lookback: int = 10
    liquidity_lookback: int = 20


REQUIRED_COLUMNS = {"timestamp_ms", "open", "high", "low", "close", "volume"}


def _timeframe_to_ms(timeframe: str) -> int:
    if not timeframe or len(timeframe) < 2:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    units = {"s": 1_000, "m": 60_000, "h": 3_600_000, "d": 86_400_000, "w": 604_800_000}
    try:
        value = int(timeframe[:-1])
    except ValueError as exc:
        raise ValueError(f"Unsupported timeframe: {timeframe}") from exc
    unit = timeframe[-1]
    if value <= 0 or unit not in units:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    return value * units[unit]


def _validate_frame(df: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing OHLCV columns: {sorted(missing)}")
    result = df.copy()
    result = result.sort_values("timestamp_ms").drop_duplicates("timestamp_ms")
    if result.empty:
        raise ValueError("OHLCV frame is empty")
    if not result["timestamp_ms"].is_monotonic_increasing:
        raise ValueError("OHLCV timestamps must be monotonic")
    return result


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    previous_close = df["close"].shift(1)
    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _adx(df: pd.DataFrame, period: int) -> pd.Series:
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    atr = _atr(df, period)
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def _single_timeframe_features(df: pd.DataFrame, cfg: FeatureConfig, prefix: str = "") -> pd.DataFrame:
    frame = _validate_frame(df)
    close = frame["close"]
    high = frame["high"]
    low = frame["low"]
    volume = frame["volume"]

    atr = _atr(frame, cfg.atr_period)
    ema_fast = close.ewm(span=cfg.ema_fast, adjust=False, min_periods=cfg.ema_fast).mean()
    ema_slow = close.ewm(span=cfg.ema_slow, adjust=False, min_periods=cfg.ema_slow).mean()
    rolling_mid = close.rolling(cfg.bb_period, min_periods=cfg.bb_period).mean()
    rolling_std = close.rolling(cfg.bb_period, min_periods=cfg.bb_period).std(ddof=0)
    bb_width = (2 * cfg.bb_std * rolling_std) / rolling_mid.replace(0, np.nan)

    rolling_high = high.rolling(cfg.structure_lookback, min_periods=cfg.structure_lookback).max().shift(1)
    rolling_low = low.rolling(cfg.structure_lookback, min_periods=cfg.structure_lookback).min().shift(1)
    previous_swing_high = high.shift(1).rolling(cfg.structure_lookback, min_periods=cfg.structure_lookback).max()
    previous_swing_low = low.shift(1).rolling(cfg.structure_lookback, min_periods=cfg.structure_lookback).min()

    frame[f"{prefix}return_1"] = close.pct_change()
    frame[f"{prefix}return_5"] = close.pct_change(5)
    frame[f"{prefix}atr"] = atr
    frame[f"{prefix}atr_pct"] = atr / close.replace(0, np.nan)
    frame[f"{prefix}ema_fast_dist"] = (close - ema_fast) / close.replace(0, np.nan)
    frame[f"{prefix}ema_slow_dist"] = (close - ema_slow) / close.replace(0, np.nan)
    frame[f"{prefix}rsi"] = _rsi(close, cfg.rsi_period)
    frame[f"{prefix}adx"] = _adx(frame, cfg.adx_period)
    frame[f"{prefix}bb_width"] = bb_width

    frame[f"{prefix}bos_up"] = (close > rolling_high).astype(float)
    frame[f"{prefix}bos_down"] = (close < rolling_low).astype(float)
    frame[f"{prefix}structure_bias"] = np.select(
        [close > previous_swing_high, close < previous_swing_low],
        [1.0, -1.0],
        default=0.0,
    )

    volume_mean = volume.rolling(cfg.liquidity_lookback, min_periods=cfg.liquidity_lookback).mean()
    volume_std = volume.rolling(cfg.liquidity_lookback, min_periods=cfg.liquidity_lookback).std(ddof=0)
    frame[f"{prefix}volume_z"] = (volume - volume_mean) / volume_std.replace(0, np.nan)
    frame[f"{prefix}dist_prior_high_atr"] = (close - rolling_high) / atr.replace(0, np.nan)
    frame[f"{prefix}dist_prior_low_atr"] = (close - rolling_low) / atr.replace(0, np.nan)
    frame[f"{prefix}sweep_high"] = ((high > rolling_high) & (close < rolling_high)).astype(float)
    frame[f"{prefix}sweep_low"] = ((low < rolling_low) & (close > rolling_low)).astype(float)

    trend_up = (close > ema_slow) & (frame[f"{prefix}adx"] >= 25)
    trend_down = (close < ema_slow) & (frame[f"{prefix}adx"] >= 25)
    high_vol = frame[f"{prefix}atr_pct"] >= frame[f"{prefix}atr_pct"].rolling(100, min_periods=20).median()
    frame[f"{prefix}regime"] = np.select(
        [trend_up & high_vol, trend_down & high_vol, trend_up, trend_down, high_vol],
        [2.0, -2.0, 1.0, -1.0, 0.0],
        default=0.0,
    )

    return frame


def build_canonical_features(
    base: pd.DataFrame,
    informative: dict[str, pd.DataFrame] | None = None,
    cfg: FeatureConfig | None = None,
    base_timeframe: str = "1h",
) -> pd.DataFrame:
    """Build the canonical point-in-time feature matrix.

    Base and informative candles are timestamped by candle open. Alignment is
    performed using ``open + timeframe_duration`` (the candle close). Thus a
    higher-timeframe candle becomes available only after it has actually closed.
    Backward as-of matching selects the latest completed informative candle.
    """
    cfg = cfg or FeatureConfig()
    base_frame = _single_timeframe_features(base, cfg)
    base_duration = _timeframe_to_ms(base_timeframe)
    result = base_frame.copy()
    result["_decision_close_ms"] = result["timestamp_ms"] + base_duration
    result = result.sort_values("_decision_close_ms").reset_index(drop=True)

    for timeframe, higher in (informative or {}).items():
        higher_features = _single_timeframe_features(higher, cfg, prefix=f"{timeframe}_").copy()
        higher_duration = _timeframe_to_ms(timeframe)
        higher_features["_decision_close_ms"] = higher_features["timestamp_ms"] + higher_duration
        columns = [column for column in higher_features.columns if column != "timestamp_ms"]
        higher_features = higher_features[["_decision_close_ms", *columns]].sort_values("_decision_close_ms")
        result = pd.merge_asof(
            result,
            higher_features,
            on="_decision_close_ms",
            direction="backward",
            allow_exact_matches=True,
        )

    result = result.drop(columns=["_decision_close_ms"])
    return result.replace([np.inf, -np.inf], np.nan)


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric feature columns, excluding raw OHLCV and timestamps."""
    excluded = REQUIRED_COLUMNS
    return [
        column
        for column in df.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(df[column])
    ]
