from __future__ import annotations

"""Deterministic risk and position-sizing layer."""

from dataclasses import dataclass


class RiskValidationError(ValueError):
    """Raised when a trade cannot satisfy risk constraints."""


@dataclass(frozen=True)
class PositionSize:
    equity_usdt: float
    risk_fraction: float
    risk_amount_usdt: float
    entry_price: float
    stop_loss: float
    stop_distance: float
    quantity: float


def calculate_position_size(
    *,
    equity_usdt: float,
    risk_fraction: float,
    entry_price: float,
    stop_loss: float,
) -> PositionSize:
    """Calculate base-asset quantity from fixed fractional account risk.

    Formula:
        risk_amount = equity * risk_fraction
        quantity = risk_amount / abs(entry - stop)

    Leverage is intentionally not part of this calculation. It must never be
    used to increase the account-risk budget.
    """
    if equity_usdt <= 0:
        raise RiskValidationError(
            "ACCOUNT_EQUITY_USDT must be greater than zero before a "
            "position size can be calculated."
        )

    if not 0 < risk_fraction < 1:
        raise RiskValidationError(
            "risk_fraction must be greater than zero and less than one."
        )

    if entry_price <= 0 or stop_loss <= 0:
        raise RiskValidationError("entry_price and stop_loss must be positive.")

    stop_distance = abs(entry_price - stop_loss)
    if stop_distance <= 0:
        raise RiskValidationError("Entry and stop-loss prices must be different.")

    risk_amount = equity_usdt * risk_fraction
    quantity = risk_amount / stop_distance

    if quantity <= 0:
        raise RiskValidationError("Calculated position quantity is not positive.")

    return PositionSize(
        equity_usdt=equity_usdt,
        risk_fraction=risk_fraction,
        risk_amount_usdt=risk_amount,
        entry_price=entry_price,
        stop_loss=stop_loss,
        stop_distance=stop_distance,
        quantity=quantity,
    )
