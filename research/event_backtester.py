from __future__ import annotations

"""Strategy-agnostic event-driven backtester.

The engine advances one bar at a time. A strategy sees only the current and
previously emitted bars. Signals are converted to orders and filled on the
next bar's open. Stops/targets are evaluated from subsequent OHLC bars, with a
conservative stop-first rule when both levels occur in one bar.
"""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    initial_equity: float = 10_000.0
    fee_bps_per_side: float = 5.0
    slippage_bps_per_side: float = 2.0
    max_bars_in_trade: int = 6
    allow_short: bool = True


@dataclass(frozen=True)
class BacktestSignal:
    side: int
    stop_distance_pct: float
    reward_risk_ratio: float = 1.5


@dataclass(frozen=True)
class BacktestTrade:
    entry_time: int
    exit_time: int
    side: int
    entry_price: float
    exit_price: float
    quantity: float
    gross_pnl: float
    fees: float
    slippage_cost: float
    net_pnl: float
    reason: str


@dataclass
class BacktestResult:
    initial_equity: float
    final_equity: float
    trades: list[BacktestTrade] = field(default_factory=list)
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def total_return(self) -> float:
        return self.final_equity / self.initial_equity - 1.0

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        return sum(trade.net_pnl > 0 for trade in self.trades) / len(self.trades)

    @property
    def max_drawdown(self) -> float:
        if self.equity_curve.empty:
            return 0.0
        running_max = self.equity_curve["equity"].cummax()
        drawdown = self.equity_curve["equity"] / running_max - 1.0
        return float(drawdown.min())


StrategyFn = Callable[[pd.DataFrame], BacktestSignal | None]


def _adverse_price(price: float, side: int, bps: float, entry: bool) -> float:
    slip = bps / 10_000.0
    if side == 1:
        return price * (1 + slip if entry else 1 - slip)
    return price * (1 - slip if entry else 1 + slip)


def run_event_backtest(
    candles: pd.DataFrame,
    strategy: StrategyFn,
    *,
    quantity_fn: Callable[[float, float, float], float] | None = None,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Run a deterministic one-position-at-a-time event-driven simulation.

    ``quantity_fn(equity, entry_price, stop_price)`` may be supplied for risk
    sizing. If omitted, one unit is traded. The strategy is called at the close
    of each bar, and the resulting order is filled at the next bar's open.
    """
    cfg = config or BacktestConfig()
    required = {"timestamp_ms", "open", "high", "low", "close"}
    missing = required.difference(candles.columns)
    if missing:
        raise ValueError(f"Missing candle columns: {sorted(missing)}")
    frame = candles.sort_values("timestamp_ms").reset_index(drop=True)
    if len(frame) < 2:
        return BacktestResult(cfg.initial_equity, cfg.initial_equity)

    equity = float(cfg.initial_equity)
    position: dict[str, float | int] | None = None
    pending: BacktestSignal | None = None
    pending_time: int | None = None
    trades: list[BacktestTrade] = []
    curve: list[dict[str, float | int]] = []

    for i in range(len(frame)):
        bar = frame.iloc[i]
        timestamp = int(bar["timestamp_ms"])

        # Event 1: execute an order generated from the previous completed bar.
        if pending is not None and position is None and i > 0:
            side = int(pending.side)
            if side not in (-1, 1) or (side == -1 and not cfg.allow_short):
                pending = None
            else:
                raw_entry = float(bar["open"])
                stop_distance = raw_entry * pending.stop_distance_pct
                stop_price = raw_entry - stop_distance if side == 1 else raw_entry + stop_distance
                quantity = (
                    float(quantity_fn(equity, raw_entry, stop_price))
                    if quantity_fn is not None
                    else 1.0
                )
                if quantity > 0 and np.isfinite(quantity):
                    entry_price = _adverse_price(raw_entry, side, cfg.slippage_bps_per_side, True)
                    entry_fee = abs(entry_price * quantity) * cfg.fee_bps_per_side / 10_000.0
                    equity -= entry_fee
                    position = {
                        "side": side,
                        "entry_price": entry_price,
                        "raw_entry": raw_entry,
                        "quantity": quantity,
                        "entry_time": timestamp,
                        "stop": stop_price,
                        "target": raw_entry + stop_distance * pending.reward_risk_ratio if side == 1 else raw_entry - stop_distance * pending.reward_risk_ratio,
                        "bars_held": 0,
                        "entry_fee": entry_fee,
                        "pending_time": pending_time or timestamp,
                    }
                pending = None
                pending_time = None

        # Event 2: manage the existing position using this bar.
        if position is not None:
            position["bars_held"] = int(position["bars_held"]) + 1
            side = int(position["side"])
            stop = float(position["stop"])
            target = float(position["target"])
            high = float(bar["high"])
            low = float(bar["low"])
            stop_hit = low <= stop if side == 1 else high >= stop
            target_hit = high >= target if side == 1 else low <= target

            reason: str | None = None
            raw_exit: float | None = None
            if stop_hit:
                reason = "STOP"
                raw_exit = stop
            elif target_hit:
                reason = "TARGET"
                raw_exit = target
            elif int(position["bars_held"]) >= cfg.max_bars_in_trade:
                reason = "TIME"
                raw_exit = float(bar["close"])

            if reason is not None and raw_exit is not None:
                exit_price = _adverse_price(raw_exit, side, cfg.slippage_bps_per_side, False)
                quantity = float(position["quantity"])
                gross_pnl = side * (exit_price - float(position["entry_price"])) * quantity
                exit_fee = abs(exit_price * quantity) * cfg.fee_bps_per_side / 10_000.0
                total_fees = float(position["entry_fee"]) + exit_fee
                theoretical_exit = raw_exit
                slippage_cost = abs(theoretical_exit - exit_price) * quantity + abs(float(position["raw_entry"]) - float(position["entry_price"])) * quantity
                net_pnl = gross_pnl - exit_fee
                equity += gross_pnl - exit_fee
                trades.append(
                    BacktestTrade(
                        entry_time=int(position["entry_time"]),
                        exit_time=timestamp,
                        side=side,
                        entry_price=float(position["entry_price"]),
                        exit_price=exit_price,
                        quantity=quantity,
                        gross_pnl=gross_pnl,
                        fees=total_fees,
                        slippage_cost=slippage_cost,
                        net_pnl=net_pnl,
                        reason=reason,
                    )
                )
                position = None

        # Event 3: generate a signal from information available at this close.
        # It cannot fill until the next bar, structurally preventing same-close fills.
        if position is None and i < len(frame) - 1:
            history = frame.iloc[: i + 1].copy()
            signal = strategy(history)
            if signal is not None:
                pending = signal
                pending_time = timestamp

        curve.append({"timestamp_ms": timestamp, "equity": equity})

    # Pending orders are deliberately not filled after the dataset ends.
    return BacktestResult(
        initial_equity=cfg.initial_equity,
        final_equity=equity,
        trades=trades,
        equity_curve=pd.DataFrame(curve),
    )
