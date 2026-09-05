"""Background analysis jobs for the Qt workbench.

Workers only receive plain payloads and a callable.  They never access a
widget; the main window applies results on the GUI thread and rejects stale
generations when a newer preview/optimization has already been requested.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable

from .qt_compat import QT_AVAILABLE, QtCore


@dataclass(frozen=True)
class JobRequest:
    generation: int
    kind: str
    parameters: dict[str, Any]
    payload: Any = None


class GenerationGuard:
    """Monotonically increasing request generation used for stale-result gates."""

    def __init__(self) -> None:
        self.current = 0

    def next(self) -> int:
        self.current += 1
        return self.current

    def is_current(self, generation: int) -> bool:
        return generation == self.current


def _invoke(job: Callable[..., Any], request: JobRequest) -> Any:
    """Call an engine adapter without requiring one fixed core signature."""

    try:
        signature = inspect.signature(job)
    except (TypeError, ValueError):
        signature = None
    kwargs = {
        "kind": request.kind,
        "parameters": request.parameters,
        "payload": request.payload,
        "request": request,
    }
    if signature is not None:
        parameters = signature.parameters
        accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values())
        if accepts_kwargs:
            return job(**kwargs)
        accepted = {name: value for name, value in kwargs.items() if name in parameters}
        required = [
            p
            for p in parameters.values()
            if p.default is inspect.Parameter.empty
            and p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        if accepted or not required:
            return job(**accepted)
        if len(required) == 1:
            return job(request.parameters)
        if len(required) == 2:
            return job(request.parameters, request.payload)
        return job(request.kind, request.parameters, request.payload)
    return job(request.kind, request.parameters, request.payload)


if QT_AVAILABLE:

    class WorkerSignals(QtCore.QObject):
        finished = QtCore.Signal(int, str, object)
        error = QtCore.Signal(int, str, object)
        progress = QtCore.Signal(int, str, object)


    class AnalysisWorker(QtCore.QRunnable):
        """Run one preview/optimization/batch request on ``QThreadPool``."""

        def __init__(
            self,
            job: Callable[..., Any],
            *,
            generation: int,
            kind: str = "preview",
            parameters: dict[str, Any] | None = None,
            payload: Any = None,
        ) -> None:
            super().__init__()
            self.setAutoDelete(True)
            self.job = job
            self.request = JobRequest(
                generation=int(generation),
                kind=str(kind),
                parameters=dict(parameters or {}),
                payload=payload,
            )
            self.generation = self.request.generation
            self.kind = self.request.kind
            self.signals = WorkerSignals()

        def report_progress(self, payload: Any) -> None:
            """Thread-safe callback passed to batch/service progress hooks."""

            self.signals.progress.emit(self.generation, self.kind, payload)

        @QtCore.Slot()
        def run(self) -> None:  # noqa: D401 - QRunnable API
            try:
                result = _invoke(self.job, self.request)
            except Exception as exc:  # worker errors are reported on GUI thread
                self.signals.error.emit(self.generation, self.kind, exc)
                return
            self.signals.finished.emit(self.generation, self.kind, result)


    class BatchWorker(AnalysisWorker):
        """Semantic alias for callers that want to identify batch jobs."""

        def __init__(self, job: Callable[..., Any], **kwargs: Any) -> None:
            kwargs.setdefault("kind", "batch")
            super().__init__(job, **kwargs)


else:

    class _Signal:
        def __init__(self) -> None:
            self._callbacks: list[Callable[..., Any]] = []

        def connect(self, callback: Callable[..., Any]) -> None:
            self._callbacks.append(callback)

        def emit(self, *args: Any) -> None:
            for callback in tuple(self._callbacks):
                callback(*args)


    class WorkerSignals:
        def __init__(self) -> None:
            self.finished = _Signal()
            self.error = _Signal()
            self.progress = _Signal()


    class AnalysisWorker:
        """Synchronous fallback with QRunnable-like ``run`` semantics."""

        def __init__(
            self,
            job: Callable[..., Any],
            *,
            generation: int,
            kind: str = "preview",
            parameters: dict[str, Any] | None = None,
            payload: Any = None,
        ) -> None:
            self.job = job
            self.request = JobRequest(int(generation), str(kind), dict(parameters or {}), payload)
            self.generation = self.request.generation
            self.kind = self.request.kind
            self.signals = WorkerSignals()

        def report_progress(self, payload: Any) -> None:
            self.signals.progress.emit(self.generation, self.kind, payload)

        def run(self) -> None:
            try:
                result = _invoke(self.job, self.request)
            except Exception as exc:
                self.signals.error.emit(self.generation, self.kind, exc)
                return
            self.signals.finished.emit(self.generation, self.kind, result)


    class BatchWorker(AnalysisWorker):
        def __init__(self, job: Callable[..., Any], **kwargs: Any) -> None:
            kwargs.setdefault("kind", "batch")
            super().__init__(job, **kwargs)


RefinementWorker = AnalysisWorker

__all__ = [
    "AnalysisWorker",
    "BatchWorker",
    "GenerationGuard",
    "JobRequest",
    "RefinementWorker",
    "WorkerSignals",
]
