from __future__ import annotations

"""Execution adapter boundary.

P0 deliberately contains no live execution implementation.
Market-data acquisition and order execution are separate contracts.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class OrderRequest:
    symbol: str
    side: str
    quantity: float
    entry_price: float
    stop_loss: float
    take_profit: float


class ExecutionAdapter(ABC):
    """Contract for future order execution implementations."""

    @abstractmethod
    def submit_order(self, request: OrderRequest) -> str:
        raise NotImplementedError

    @abstractmethod
    def cancel_order(self, order_id: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def reconcile_order(self, order_id: str) -> dict:
        raise NotImplementedError


class ExecutionDisabledError(RuntimeError):
    """Raised when execution is requested before the execution gateway is enabled."""


class DisabledExecutionAdapter(ExecutionAdapter):
    """Safety boundary: all live execution is disabled in P0."""

    def submit_order(self, request: OrderRequest) -> str:
        raise ExecutionDisabledError(
            "Live execution is disabled in P0. "
            "Use a future explicitly enabled execution adapter."
        )

    def cancel_order(self, order_id: str) -> None:
        raise ExecutionDisabledError("Live execution is disabled in P0.")

    def reconcile_order(self, order_id: str) -> dict:
        raise ExecutionDisabledError("Live execution is disabled in P0.")
