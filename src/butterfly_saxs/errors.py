"""User-facing recoverable errors shared by CLI and the scientific pipeline.

Kept in a standard-library module so ``bsaxs describe`` / missing-input
failures do not import NumPy just to construct an error type.
"""

from __future__ import annotations


class PipelineError(RuntimeError):
    """A user-facing, recoverable pipeline error."""


__all__ = ["PipelineError"]
