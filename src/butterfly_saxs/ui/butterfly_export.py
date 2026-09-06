"""Non-overwriting export bundle for one butterfly geometry result.

This module is intentionally UI-owned.  It serializes the result already
returned by the service and never runs fitting or changes the scientific core.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

try:
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if _np is not None:
        if isinstance(value, _np.ndarray):
            return _jsonable(value.tolist())
        if isinstance(value, _np.generic):
            return _jsonable(value.item())
    return str(value)


def _csv_value(value: Any) -> str:
    value = _jsonable(value)
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def _safe_write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _csv_value(row.get(column)) for column in columns})


def _parameter_rows(quantities: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not isinstance(quantities, Mapping):
        return rows
    for name, payload in quantities.items():
        payload = payload if isinstance(payload, Mapping) else {"value": payload}
        rows.append(
            {
                "parameter": str(name),
                "value": payload.get("value"),
                "candidate_value": payload.get("candidate_value", payload.get("candidate")),
                "status": payload.get("status"),
                "reason": payload.get("reason"),
                "interval": payload.get("interval", payload.get("confidence_interval")),
                "unit": payload.get("unit"),
            }
        )
    return rows


def _point_rows(points: Any) -> list[dict[str, Any]]:
    if not isinstance(points, Sequence) or isinstance(points, (str, bytes)):
        return []
    rows: list[dict[str, Any]] = []
    for point in points:
        if isinstance(point, Mapping):
            rows.append({str(key): value for key, value in point.items()})
    return rows


def _review_payload(workbench: Any, context: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return only a review state explicitly tied to the butterfly page result."""

    review = getattr(workbench, "manual_review", None)
    if callable(review):
        review = review()
    if not isinstance(review, Mapping):
        context_review = (context or {}).get("butterfly_review") if isinstance(context, Mapping) else None
        review = context_review if isinstance(context_review, Mapping) else {}
    status = str(review.get("manual_status", review.get("status", "unreviewed")) or "unreviewed").lower()
    if status not in {"unreviewed", "accepted", "rejected"}:
        status = "unreviewed"
    # The workbench property clears stale/missing revisions.  A caller that
    # supplies a raw state must provide the audit identity for a decision to be
    # meaningful; otherwise it remains explicitly unreviewed.
    reviewer = str(review.get("reviewed_by", review.get("reviewer", "")) or "")
    reviewed_at = review.get("reviewed_at")
    if status in {"accepted", "rejected"} and (not reviewer or not reviewed_at):
        status = "unreviewed"
        reviewer = ""
        reviewed_at = None
    return {
        "manual_status": status,
        "reviewed_by": reviewer,
        "reviewed_at": reviewed_at,
        "review_notes": str(review.get("review_notes", review.get("notes", "")) or ""),
        # A manual review is an audit state only.  It never certifies the
        # scientific result on behalf of the quantitative gates.
        "scientific_acceptance": False,
    }


def export_butterfly_analysis(
    workbench: Any,
    target: str | os.PathLike[str] | Path,
    *,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    """Write a fresh butterfly analysis bundle and reject all overwrites."""

    if not bool(getattr(workbench, "result_fresh", False)):
        raise ValueError("butterfly analysis result is stale; rerun Identify or Evaluate")
    result = getattr(workbench, "current_result", {})
    if not isinstance(result, Mapping) or not result:
        raise ValueError("no butterfly analysis result is available")
    target_path = Path(target).expanduser().resolve()
    if target_path.exists():
        raise FileExistsError(f"export target already exists: {target_path}")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target_path.name}.", dir=target_path.parent))
    try:
        review = _review_payload(workbench, context)
        analysis = {
            "schema_version": 1,
            "method": "butterfly_curvature",
            "analysis": _jsonable(getattr(workbench, "butterfly_settings", {})),
            "butterfly": _jsonable(result),
            **review,
        }
        _safe_write_json(temporary / "butterfly_analysis.json", analysis)

        quantities = result.get("quantitative_parameters", {})
        _write_csv(
            temporary / "parameters.csv",
            _parameter_rows(quantities),
            ("parameter", "value", "candidate_value", "status", "reason", "interval", "unit"),
        )
        point_rows = _point_rows(result.get("points", []))
        columns = sorted({key for row in point_rows for key in row})
        _write_csv(temporary / "ridge_points.csv", point_rows, columns or ("point_id",))

        provenance = {
            "schema_version": 1,
            **review,
            "source": _jsonable(dict(context or getattr(workbench, "export_context", {}) or {})),
            "display": _jsonable(getattr(workbench, "display_settings", {})),
            "frame_data": {
                "q_unit": _jsonable(getattr(workbench, "_frame_data", {}).get("q_unit")),
                "shape": list(getattr(getattr(workbench, "qspace", None), "observed", _np.empty((0,))).shape)
                if _np is not None and getattr(getattr(workbench, "qspace", None), "observed", None) is not None
                else None,
            },
        }
        _safe_write_json(temporary / "provenance.json", provenance)

        qspace = getattr(workbench, "qspace", None)
        if qspace is not None and hasattr(qspace, "grab"):
            if not bool(qspace.grab().save(str(temporary / "qspace.png"))):
                raise OSError("could not save qspace.png")
        for name, panel in (
            ("normal_profile.png", getattr(workbench, "normal_profile", None)),
            ("ellipse_local.png", getattr(workbench, "ellipse_diagnostic", None)),
        ):
            if panel is not None and hasattr(panel, "grab"):
                if not bool(panel.grab().save(str(temporary / name))):
                    raise OSError(f"could not save {name}")

        manifest = {
            "files": sorted(path.name for path in temporary.iterdir()),
            "point_count": len(point_rows),
            "profile_count": len(result.get("profiles", {}) or {}) if isinstance(result.get("profiles", {}), Mapping) else 0,
            **review,
        }
        _safe_write_json(temporary / "manifest.json", manifest)
        os.replace(temporary, target_path)
        return {path.name: target_path / path.name for path in target_path.iterdir()}
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
        raise


__all__ = ["export_butterfly_analysis"]
