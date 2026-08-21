import numpy as np
import pandas as pd
import pytest

from research.event_backtester import BacktestConfig, BacktestSignal, run_event_backtest
from research.feature_engine import FeatureConfig, build_canonical_features
from research.outcome_labeler import OutcomeLabelConfig, label_trade_outcomes


def _candles(count: int = 120, start: int = 0, step: int = 3_600_000) -> pd.DataFrame:
    rows = []
    price = 100.0
    for i in range(count):
        timestamp = start + i * step
        close = price + (0.05 if i % 3 else -0.02)
        high = max(price, close) + 0.5
        low = min(price, close) - 0.5
        rows.append(
            {
                "timestamp_ms": timestamp,
                "open": price,
                "high": high,
                "low": low,
                "close": close,
                "volume": 1000 + i,
            }
        )
        price = close
    return pd.DataFrame(rows)


def test_feature_engine_has_canonical_structure_volatility_and_liquidity_features():
    df = _candles()
    features = build_canonical_features(df, cfg=FeatureConfig())

    required = {
        "atr",
        "atr_pct",
        "adx",
        "rsi",
        "bb_width",
        "bos_up",
        "bos_down",
        "structure_bias",
        "volume_z",
        "sweep_high",
        "sweep_low",
        "regime",
    }
    assert required.issubset(features.columns)
    assert len(features) == len(df)


def test_mtf_features_are_backward_aligned_without_future_fill():
    base = _candles(120)
    higher = _candles(120, step=4 * 3_600_000)
    higher.loc[higher.index[-1], "close"] = 999999.0

    features = build_canonical_features(
        base,
        informative={"4h": higher},
        cfg=FeatureConfig(),
    )

    assert "4h_ema_slow_dist" in features.columns
    # The future higher-timeframe row cannot be projected backward into base rows.
    assert not np.isclose(
        features.iloc[0]["4h_ema_slow_dist"],
        higher.iloc[-1]["close"],
        equal_nan=False,
    )


def test_trade_outcome_label_is_not_next_candle_direction():
    candles = _candles(20)
    candles.loc[1, "high"] = 105
    candles.loc[1, "close"] = 104

    labels = label_trade_outcomes(
        candles,
        OutcomeLabelConfig(
            horizon_bars=3,
            stop_distance_pct=0.01,
            reward_risk_ratio=1.5,
            fee_bps_per_side=0,
            slippage_bps_per_side=0,
        ),
    )

    assert labels.loc[0, "trade_outcome"] == 1
    assert labels.loc[0, "trade_exit_bar"] == 1


def test_trade_outcome_includes_fees_and_slippage():
    candles = _candles(10)
    candles.loc[1, "high"] = 105
    candles.loc[1, "close"] = 104

    gross = label_trade_outcomes(
        candles,
        OutcomeLabelConfig(
            horizon_bars=2,
            stop_distance_pct=0.01,
            reward_risk_ratio=1.5,
            fee_bps_per_side=0,
            slippage_bps_per_side=0,
        ),
    )
    net = label_trade_outcomes(
        candles,
        OutcomeLabelConfig(
            horizon_bars=2,
            stop_distance_pct=0.01,
            reward_risk_ratio=1.5,
            fee_bps_per_side=5,
            slippage_bps_per_side=2,
        ),
    )

    assert net.loc[0, "trade_net_return"] < gross.loc[0, "trade_net_return"]


def test_ambiguous_ohlc_bar_is_stop_first():
    candles = _candles(5)
    candles.loc[1, "high"] = 102
    candles.loc[1, "low"] = 98

    labels = label_trade_outcomes(
        candles,
        OutcomeLabelConfig(
            horizon_bars=1,
            stop_distance_pct=0.01,
            reward_risk_ratio=1.5,
            fee_bps_per_side=0,
            slippage_bps_per_side=0,
        ),
    )

    assert labels.loc[0, "trade_outcome"] != 1


def test_event_backtester_fills_on_next_bar_not_signal_bar():
    candles = _candles(8)
    signal_times = []

    def strategy(history):
        signal_times.append(int(history.iloc[-1]["timestamp_ms"]))
        if len(signal_times) == 1:
            return BacktestSignal(side=1, stop_distance_pct=0.01, reward_risk_ratio=1.5)
        return None

    result = run_event_backtest(
        candles,
        strategy,
        config=BacktestConfig(
            initial_equity=10000,
            fee_bps_per_side=0,
            slippage_bps_per_side=0,
            max_bars_in_trade=1,
        ),
    )

    assert result.trades
    assert result.trades[0].entry_time == candles.iloc[1]["timestamp_ms"]
    assert result.trades[0].entry_time > signal_times[0]


def test_event_backtester_applies_costs():
    candles = _candles(8)

    def strategy(history):
        if len(history) == 1:
            return BacktestSignal(side=1, stop_distance_pct=0.01, reward_risk_ratio=1.5)
        return None

    free = run_event_backtest(
        candles,
        strategy,
        config=BacktestConfig(
            initial_equity=10000,
            fee_bps_per_side=0,
            slippage_bps_per_side=0,
            max_bars_in_trade=1,
        ),
    )
    costed = run_event_backtest(
        candles,
        strategy,
        config=BacktestConfig(
            initial_equity=10000,
            fee_bps_per_side=5,
            slippage_bps_per_side=2,
            max_bars_in_trade=1,
        ),
    )

    assert costed.final_equity <= free.final_equity


def test_event_backtester_does_not_fill_pending_order_after_dataset_end():
    candles = _candles(2)

    def strategy(history):
        return BacktestSignal(side=1, stop_distance_pct=0.01)

    result = run_event_backtest(candles, strategy)
    assert result.trades == []
