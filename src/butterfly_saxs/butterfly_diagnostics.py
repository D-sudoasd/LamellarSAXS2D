"""Small, lossless display diagnostics derived from observed points and a candidate."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
import numpy as np


def profile_residuals(profiles: dict) -> None:
    """Attach the actual raw-minus-fit series; masked samples remain missing."""
    for profile in profiles.values():
        raw = np.asarray(profile.get("raw_intensity", []), dtype=float)
        fit = np.asarray(profile.get("fit_intensity", []), dtype=float)
        if raw.ndim == 1 and raw.size and raw.shape == fit.shape:
            profile["residual"] = (raw - fit).tolist()
            profile["residual_definition"] = "raw_intensity_minus_profile_fit"


def arc_diagnostic_rows(trace: Mapping | Sequence | None, candidate: Mapping | None = None) -> dict:
    """Expose per-arc support/evidence rows without inventing values.

    Rows are copied losslessly from the observed trace or candidate payload.
    The result is explicitly candidate diagnostic data: callers may render it
    or export it, but it does not certify scientific acceptance.
    """

    candidates = []
    if isinstance(trace, Mapping):
        candidates.extend((trace.get("arc_diagnostics"), trace.get("arc_support")))
        diagnostics = trace.get("diagnostics")
        if isinstance(diagnostics, Mapping):
            candidates.extend((diagnostics.get("arc_diagnostics"), diagnostics.get("arc_support")))
    if isinstance(candidate, Mapping):
        candidates.extend((candidate.get("arc_diagnostics"), candidate.get("arc_support")))
    elif isinstance(trace, Sequence) and not isinstance(trace, (str, bytes)):
        candidates.append(trace)
    rows = []
    for value in candidates:
        if isinstance(value, Mapping):
            value = value.get("rows", value.get("arcs", value.get("records")))
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            rows = [deepcopy(item) for item in value if isinstance(item, Mapping)]
            if rows:
                break
    return {
        "rows": rows,
        "source": "observed_arc_support",
        "candidate_only": True,
        "scientific_status": "NOT_ACCEPTED",
        "field_provenance": {
            "support": "frozen observed q-space corridor",
            "projection_distance_q": "valid bounded-support projection only",
            "support_violation_q": "finite infeasible-support penalty; not a projection distance",
            "normal_residual_q": "fixed source branch and side labels",
            "mirror_symmetry": "fixed source labels and q-space reflection; unavailable remains explicit",
        },
    }


# Descriptive aliases keep the helper easy to discover from existing local
# view/export callers without creating a second representation.
arc_evidence_view = arc_diagnostic_rows
arc_support_diagnostics = arc_diagnostic_rows
arc_support_rows = arc_diagnostic_rows
observed_arc_diagnostics = arc_diagnostic_rows


def ellipse_local_views(points: list[dict], candidate: dict) -> dict:
    """Return branch-local v versus u, without multiplying the scientific data.

    A display may magnify v, but that factor belongs only to the view. These
    coordinates and the side curves are labelled as candidate diagnostics,
    including when the candidate has not passed identifiability checks.
    """
    try:
        a, b = float(candidate["a"]), float(candidate["b"])
        theta = float(candidate["theta_deg"])
        reference = float(candidate.get("reference_axis_deg", 0.))
        cx, cy = float(candidate.get("center_qx", 0.)), float(candidate.get("center_qy", 0.))
    except (KeyError, TypeError, ValueError):
        return {}
    if not all(math.isfinite(x) for x in (a, b, theta, reference, cx, cy)) or not a >= b > 0:
        return {}
    result = {}
    for branch in (0, 1):
        canonical = 1 - branch if candidate.get("branch_swap_applied", False) else branch
        angle = math.radians(reference + (1 if canonical == 0 else -1) * theta)
        c, s = math.cos(angle), math.sin(angle)
        rows = []
        for point in points:
            if point.get("branch_id") != branch:
                continue
            try:
                x, y = float(point["qx"]) - cx, float(point["qy"]) - cy
            except (KeyError, TypeError, ValueError):
                continue
            if not math.isfinite(x + y):
                continue
            rows.append({"point_id": str(point.get("point_id", "")), "u": c*x + s*y,
                         "v": -s*x + c*y, "side": point.get("side", "unknown"),
                         "accepted": bool(point.get("accepted", False))})
        supported = [row["u"] for row in rows if row["accepted"]]
        curves = {}
        if supported:
            low, high = max(-a, min(supported)), min(a, max(supported))
            if high > low:
                u = np.linspace(low, high, 160)
                v = b * np.sqrt(np.maximum(0., 1. - (u / a)**2))
                curves = {side: {"u": u.tolist(), "v": (sign*v).tolist()}
                          for side, sign in (("upper", 1), ("lower", -1))}
        result[str(branch)] = {"points": rows, "curves": curves, "a": a, "b": b,
                               "q_unit": candidate.get("q_unit", "unknown"),
                               "source": "candidate_geometry", "angle_deg": math.degrees(angle),
                               "solver_success": bool(candidate.get("success"))}
    return result
