from __future__ import annotations

"""Trade-outcome labels for research.

Labels are generated from future bars only and are never exposed to the feature
engine for the same decision timestamp. Costs and adverse slippage are included
in the realized net return used by the label.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OutcomeLabelConfig:
    horizon_bars: int = 6
    stop_distance_pct: float = 0.008
    reward_risk_ratio: float = 1.5
    fee_bps_per_side: float = 5.0
    slippage_bps_per_side: float = 2.0


def _execution_price(price: float, side: int, slippage_bps: float, entry: bool) -> float:
    slip = slippage_bps / 10_000.0
    if side == 1:
        return price * (1 + slip if entry else 1 - slip)
    return price * (1 - slip if entry else 1 + slip)


def _net_return(entry_price: float, exit_price: float, side: int, fee_bps: float) -> float:
    fee = 2 * fee_bps / 10_000.0
    gross = side * (exit_price - entry_price) / entry_price
    return gross - fee


def label_trade_outcomes(
    candles: pd.DataFrame,
    cfg: OutcomeLabelConfig | None = None,
) -> pd.DataFrame:
    """Return point-in-time long/short/flat trade-outcome labels.

    Outcome classes:
      1  = long target reached first
     -1  = short target reached first
      0  = neither target nor stop reached inside the horizon

    If stop and target are both touched in one OHLC bar, stop is assumed first.
    This is intentionally conservative because OHLC data cannot establish
    intrabar ordering.
    """
    cfg = cfg or OutcomeLabelConfig()
    required = {"timestamp_ms", "open", "high", "low", "close"}
    missing = required.difference(candles.columns)
    if missing:
        raise ValueError(f"Missing candle columns: {sorted(missing)}")
    if cfg.horizon_bars < 1 or cfg.stop_distance_pct <= 0 or cfg.reward_risk_ratio <= 0:
        raise ValueError("Invalid outcome-label configuration")

    frame = candles.sort_values("timestamp_ms").reset_index(drop=True).copy()
    labels = np.zeros(len(frame), dtype=np.int8)
    net_returns = np.zeros(len(frame), dtype=float)
    exit_bars = np.full(len(frame), -1, dtype=int)

    for i in range(len(frame) - 1):
        entry_raw = float(frame.at[i, "close"])
        if not np.isfinite(entry_raw) or entry_raw <= 0:
            continue

        for side in (1, -1):
            entry = _execution_price(entry_raw, side, cfg.slippage_bps_per_side, entry=True)
            stop_raw = entry_raw * (1 - cfg.stop_distance_pct if side == 1 else 1 + cfg.stop_distance_pct)
            target_raw = entry_raw * (
                1 + cfg.stop_distance_pct * cfg.reward_risk_ratio
                if side == 1
                else 1 - cfg.stop_distance_pct * cfg.reward_risk_ratio
            )
            outcome = 0
            outcome_return = 0.0
            outcome_bar = -1

            end = min(len(frame), i + 1 + cfg.horizon_bars)
            for j in range(i + 1, end):
                high = float(frame.at[j, "high"])
                low = float(frame.at[j, "low"])
                stop_hit = low <= stop_raw if side == 1 else high >= stop_raw
                target_hit = high >= target_raw if side == 1 else low <= target_raw

                if stop_hit:
                    exit_raw = stop_raw
                    exit_price = _execution_price(exit_raw, side, cfg.slippage_bps_per_side, entry=False)
                    outcome_return = _net_return(entry, exit_price, side, cfg.fee_bps_per_side)
                    outcome = -1
                    outcome_bar = j
                    break
                if target_hit:
                    exit_raw = target_raw
                    exit_price = _execution_price(exit_raw, side, cfg.slippage_bps_per_side, entry=False)
                    outcome_return = _net_return(entry, exit_price, side, cfg.fee_bps_per_side)
                    outcome = 1
                    outcome_bar = j
                    break

            if outcome == 0 and end > i + 1:
                exit_raw = float(frame.at[end - 1, "close"])
                exit_price = _execution_price(exit_raw, side, cfg.slippage_bps_per_side, entry=False)
                outcome_return = _net_return(entry, exit_price, side, cfg.fee_bps_per_side)
                outcome_bar = end - 1

            # Select the direction with the better realized net outcome.
            if side == 1:
                long_return = outcome_return
                long_outcome = outcome
                long_exit = outcome_bar
            else:
                short_return = outcome_return
                short_outcome = outcome
                short_exit = outcome_bar

        if long_outcome == 1 and short_outcome != 1:
            labels[i] = 1
            net_returns[i] = long_return
            exit_bars[i] = long_exit
        elif short_outcome == 1 and long_outcome != 1:
            labels[i] = -1
            net_returns[i] = short_return
            exit_bars[i] = short_exit
        elif long_return > 0 and long_return >= short_return:
            labels[i] = 1
            net_returns[i] = long_return
            exit_bars[i] = long_exit
        elif short_return > 0:
            labels[i] = -1
            net_returns[i] = short_return
            exit_bars[i] = short_exit
        else:
            labels[i] = 0
            net_returns[i] = max(long_return, short_return)
            exit_bars[i] = max(long_exit, short_exit)

    result = pd.DataFrame(index=frame.index)
    result["trade_outcome"] = labels
    result["trade_net_return"] = net_returns
    result["trade_exit_bar"] = exit_bars
    return result
