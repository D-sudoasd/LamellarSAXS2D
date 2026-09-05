"""CSV boundary encoding without changing scientific numeric cells."""

from __future__ import annotations

import math
import re
from typing import Any


_NUMERIC_TEXT = re.compile(
    r"^[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?$"
)


def safe_csv_cell(value: Any) -> Any:
    """Protect formula-like text while preserving numeric values/types.

    Only text values beginning with ``=``/``+``/``-``/``@`` are considered
    dangerous.  A string that is a valid signed number remains numeric text,
    and Python/numpy numeric values remain numeric for downstream readers.
    """

    if value is None:
        return ""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return ""
        return value
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return safe_csv_cell(item())
        except Exception:  # pragma: no cover
            pass
    if isinstance(value, str):
        # Spreadsheet engines also evaluate formulas after ignorable leading
        # whitespace, control characters, or a UTF-8 BOM.  Keep the original
        # text for display, but inspect the normalized prefix.
        candidate = value.lstrip(" \t\r\n\v\f").lstrip("\ufeff").lstrip()
        if candidate.startswith(("=", "+", "-", "@")) and not _NUMERIC_TEXT.fullmatch(candidate):
            return "'" + value
        return value
    return value


__all__ = ["safe_csv_cell"]
