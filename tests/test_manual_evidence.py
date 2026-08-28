from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from butterfly_saxs.manual_evidence import OUTPUT_NAMES, PARAMETER_COLUMNS, export_manual_fit


def _result(tmp_path: Path) -> tuple[dict, dict]:
    observed = np.arange(36, dtype=float).reshape(6, 6)
    model = observed * 0.9 + 0.25
    residual = observed - model
    q = np.linspace(-0.8, 0.8, 6)
    qx, qy = np.meshgrid(q, q)
    valid = np.ones_like(observed, dtype=bool)
    valid[0, 0] = False
    source = tmp_path / "frame.edf"
    poni = tmp_path / "geometry.poni"
    mask = tmp_path / "mask.npy"
    source.write_bytes(b"source frame")
    poni.write_text("Distance: 0.1\n", encoding="utf-8")
    mask.write_bytes(b"mask file")
    result = {
        "observed": observed,
        "model": model,
        "residual": residual,
        "qx": qx,
        "qy": qy,
        "q_unit": "nm^-1",
        "valid_mask": valid,
        "parameters": {
            "a": {"value": 0.42, "min": 0.1, "max": 1.0, "vary": True, "unit": "nm^-1", "stderr": 0.01},
            "axis_ratio": {"value": 0.7, "min": 0.01, "max": 1.0, "vary": False, "expr": "", "unit": ""},
        },
        "ridges": [{"qx": 0.1, "qy": 0.2}, {"qx": -0.2, "qy": 0.3}],
        "ellipse_fit": {"parameters": {"cx": 0.0, "cy": 0.0, "a": 0.42, "axis_ratio": 0.7, "theta_deg": 18.0}},
        "flags": {"apparent_geometry_only": True},
        "analysis": {"max_pixels": 1400, "residual": "sampson"},
    }
    context = {
        "source": source,
        "frame": "frame-001",
        "dataset": "/entry/data",
        "poni": poni,
        "mask_path": mask,
        "roi": {"type": "ellipse", "cx": 0.0, "cy": 0.0},
        "current_model_ellipses": [{"cx": 0.0, "cy": 0.0, "a": 0.42, "axis_ratio": 0.7, "theta_deg": 22.0}],
    }
    return result, context


def test_manual_evidence_writes_exactly_seven_nonempty_files_and_strict_json(tmp_path: Path) -> None:
    result, context = _result(tmp_path)
    output = tmp_path / "evidence"

    paths = export_manual_fit(result, output, context=context)

    assert set(paths) == set(OUTPUT_NAMES)
    assert {path.name for path in output.iterdir()} == set(OUTPUT_NAMES)
    assert all(path.stat().st_size > 0 for path in paths.values())
    session_text = (output / "fit_session.json").read_text(encoding="utf-8")
    provenance_text = (output / "provenance.json").read_text(encoding="utf-8")
    assert "NaN" not in session_text
    assert "Infinity" not in session_text
    session = json.loads(session_text)
    provenance = json.loads(provenance_text)
    assert session["manual_status"] == "unreviewed"
    assert session["empirical_model_only"] is True
    assert session["human_review_required"] is True
    assert session["context"]["frame"] == "frame-001"
    assert provenance["q_unit"] == "nm^-1"
    assert provenance["inputs"]["source"]["sha256"] == hashlib.sha256(b"source frame").hexdigest()
    assert provenance["inputs"]["poni"]["exists"] is True
    assert provenance["inputs"]["mask"]["exists"] is True

    with (output / "parameters.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == list(PARAMETER_COLUMNS)
    assert [row["name"] for row in rows] == ["a", "axis_ratio"]
    assert rows[0]["unit"] == "nm^-1"


@pytest.mark.parametrize(
    "review",
    [
        {"manual_status": "accepted"},
        {"manual_status": "rejected", "reviewed_by": "reviewer"},
        {"manual_status": "accepted", "reviewed_by": "reviewer", "reviewed_at": "not-a-time"},
        {"manual_status": "pending"},
    ],
)
def test_manual_review_requires_explicit_valid_status_and_metadata(tmp_path: Path, review: dict) -> None:
    result, context = _result(tmp_path)

    with pytest.raises(ValueError):
        export_manual_fit(result, tmp_path / "evidence", context=context, review=review)


def test_manual_review_acceptance_is_explicit_and_retained(tmp_path: Path) -> None:
    result, context = _result(tmp_path)
    review = {
        "manual_status": "accepted",
        "reviewed_by": "Dr. Reviewer",
        "reviewed_at": "2026-08-28T16:00:00+08:00",
        "review_notes": "four lobes remain visible",
    }

    export_manual_fit(result, tmp_path / "evidence", context=context, review=review)
    session = json.loads((tmp_path / "evidence" / "fit_session.json").read_text(encoding="utf-8"))
    assert session["review"] == review
    assert session["manual_status"] == "accepted"


def test_default_does_not_overwrite_and_force_only_replaces_named_targets(tmp_path: Path) -> None:
    result, context = _result(tmp_path)
    output = tmp_path / "evidence"
    output.mkdir()
    sentinel = output / "unrelated.txt"
    sentinel.write_text("keep me", encoding="utf-8")
    existing = output / "model.png"
    existing.write_bytes(b"old model")

    with pytest.raises(FileExistsError, match="force=True"):
        export_manual_fit(result, output, context=context)
    assert existing.read_bytes() == b"old model"
    assert not (output / "observed.png").exists()

    export_manual_fit(result, output, context=context, force=True)
    assert existing.read_bytes() != b"old model"
    assert sentinel.read_text(encoding="utf-8") == "keep me"
    assert set(path.name for path in output.iterdir()) == set(OUTPUT_NAMES) | {"unrelated.txt"}


def test_shape_and_mask_validation_happens_before_writing(tmp_path: Path) -> None:
    result, context = _result(tmp_path)
    result["model"] = np.zeros((5, 5), dtype=float)
    with pytest.raises(ValueError, match="shape"):
        export_manual_fit(result, tmp_path / "bad-shape", context=context)
    assert not (tmp_path / "bad-shape").exists()

    result, context = _result(tmp_path)
    result["valid_mask"] = np.zeros((6, 5), dtype=bool)
    with pytest.raises(ValueError, match="valid_mask"):
        export_manual_fit(result, tmp_path / "bad-mask", context=context)
    assert not (tmp_path / "bad-mask").exists()

    result, context = _result(tmp_path)
    result["valid_mask"] = np.zeros((6, 6), dtype=bool)
    with pytest.raises(ValueError, match="没有有效像素"):
        export_manual_fit(result, tmp_path / "empty-domain", context=context)
    assert not (tmp_path / "empty-domain").exists()
