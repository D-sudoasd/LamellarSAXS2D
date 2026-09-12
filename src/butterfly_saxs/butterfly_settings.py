"""Serializable recipe for observed butterfly arcs, shared by every entrypoint."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math

from .settings import strict_int


# v2 records all-arc holdout and per-arc support diagnostics; v2.1 preserves
# the input validity domain through include edits and records the prominence-
# gated radial hint selection. Keep earlier outputs identifiable.
METHOD_VERSION = "butterfly-curvature-arcs-v2.1"
MAX_RESAMPLES = 10_000


def normalize_butterfly_settings(settings=None) -> dict:
    """Validate scientific controls and graph edits without reading any image."""
    if settings is not None and not isinstance(settings, Mapping):
        raise ValueError("analysis.butterfly must be a mapping")
    result = deepcopy(dict(settings or {}))
    result.setdefault("stage", "evaluate")
    result.setdefault("resamples", 0)
    result.setdefault("seed", 20260906)
    result.setdefault("edits", [])
    result.setdefault("sensitivity", True)
    if result["stage"] not in {"trace", "evaluate"}:
        raise ValueError("butterfly stage must be trace or evaluate")
    for key in ("resamples", "seed"):
        result[key] = strict_int(result[key], f"butterfly {key}", minimum=0)
    if "max_nfev" in result:
        result["max_nfev"] = strict_int(result["max_nfev"], "butterfly max_nfev", minimum=1)
    if result["resamples"] > MAX_RESAMPLES:
        raise ValueError(f"butterfly resamples must be <= {MAX_RESAMPLES}")
    if not isinstance(result["sensitivity"], bool):
        raise ValueError("butterfly sensitivity must be boolean")
    edits = result["edits"]
    if not isinstance(edits, list):
        raise ValueError("butterfly edits must be a list")
    for edit in edits:
        if not isinstance(edit, Mapping):
            raise ValueError("each butterfly edit must be a mapping")
        kind = edit.get("type")
        if kind in {"include_polygon", "exclude_polygon"}:
            vertices = edit.get("points", [])
            if not isinstance(vertices, (list, tuple)) or len(vertices) < 3:
                raise ValueError("polygon requires at least three q coordinate pairs")
            xy = [_pair(vertex) for vertex in vertices]
            area = sum(xy[i][0] * xy[(i + 1) % len(xy)][1]
                       - xy[(i + 1) % len(xy)][0] * xy[i][1]
                       for i in range(len(xy)))
            if abs(area) <= 1e-15:
                raise ValueError("polygon requires a nonzero area")
        elif kind == "seed":
            _pair([edit.get("qx"), edit.get("qy")])
            if isinstance(edit.get("branch_id", 0), bool) or edit.get("branch_id", 0) not in (0, 1):
                raise ValueError("seed branch_id must be 0 or 1")
            if edit.get("side", "unknown") not in ("upper", "lower", "unknown"):
                raise ValueError("seed side must be upper, lower or unknown")
        elif kind == "exclude_point":
            if not isinstance(edit.get("point_id"), str) or not edit["point_id"]:
                raise ValueError("exclude_point requires point_id")
        else:
            raise ValueError(f"unknown butterfly edit type: {kind!r}")
    return result


def _pair(value):
    try:
        if len(value) != 2:
            raise ValueError
        if any(isinstance(item, bool) for item in value):
            raise ValueError
        pair = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError("q coordinates must be finite pairs") from exc
    if not all(math.isfinite(item) for item in pair):
        raise ValueError("q coordinates must be finite pairs")
    return pair
