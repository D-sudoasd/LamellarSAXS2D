"""Cooperative cancellation contract shared by all fitting stages."""

from __future__ import annotations

from typing import Any


class AnalysisCancelled(RuntimeError):
    """Raised when a caller requests cancellation at a safe fit boundary."""

    def __init__(self, stage: str = "analysis") -> None:
        self.stage = str(stage or "analysis")
        super().__init__(f"analysis cancelled during {self.stage}")


def cancellation_requested(cancel_event: Any) -> bool:
    if cancel_event is None:
        return False
    if callable(cancel_event):
        return bool(cancel_event())
    checker = getattr(cancel_event, "is_set", None)
    if callable(checker):
        return bool(checker())
    return bool(getattr(cancel_event, "cancelled", False))


def raise_if_cancelled(cancel_event: Any, stage: str = "analysis") -> None:
    if cancellation_requested(cancel_event):
        raise AnalysisCancelled(stage)


__all__ = ["AnalysisCancelled", "cancellation_requested", "raise_if_cancelled"]
