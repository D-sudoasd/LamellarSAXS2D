"""Small, dependency-light serializers shared by public result boundaries.

Scientific fitting code is allowed to keep NumPy values and non-finite
diagnostics while it is running.  Values crossing a JSON boundary use
``json_safe`` so that non-finite numbers become JSON ``null`` and NumPy
containers/scalars become ordinary Python values.  The conversion is a
recursive copy; it never mutates the numerical inputs.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
import math
from pathlib import Path
from typing import Any


def json_safe(value: Any) -> Any:
    """Return a strict-JSON-compatible copy of ``value``.

    ``numpy`` is intentionally not imported here.  Array/scalar duck typing
    keeps this helper safe to import from low-level modules and still handles
    NumPy values without introducing an import cycle.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [json_safe(item) for item in value]
    if is_dataclass(value):
        return {
            item.name: json_safe(getattr(value, item.name))
            for item in fields(value)
        }

    # NumPy arrays and scalars both expose one of these methods.  Convert
    # through the returned built-in value and recurse so NaN/Inf is normalized
    # in arrays as well as at scalar leaves.
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            converted = tolist()
        except Exception:  # pragma: no cover - unusual array-like object
            converted = value
        if converted is not value:
            return json_safe(converted)
    item = getattr(value, "item", None)
    if callable(item):
        try:
            converted = item()
        except Exception:  # pragma: no cover - unusual scalar-like object
            converted = value
        if converted is not value:
            return json_safe(converted)

    # Parameter/result objects in the public seams commonly expose one of
    # these explicit mapping methods.  Keep this fallback after array/scalar
    # conversion so an ndarray is never stringified.
    for method_name in ("to_dict", "as_dict", "to_mapping"):
        method = getattr(value, method_name, None)
        if not callable(method):
            continue
        try:
            try:
                converted = method(include_specs=True)
            except TypeError:
                converted = method()
        except Exception:  # pragma: no cover - defensive compatibility path
            continue
        if converted is not value:
            return json_safe(converted)

    if hasattr(value, "__dict__"):
        try:
            return json_safe(
                {
                    key: item
                    for key, item in vars(value).items()
                    if not str(key).startswith("_")
                }
            )
        except Exception:  # pragma: no cover - defensive compatibility path
            pass
    return repr(value)


# Descriptive alias for callers that want to make the strict boundary
# explicit in their code without copying the implementation.
strict_jsonable = json_safe


__all__ = ["json_safe", "strict_jsonable"]
