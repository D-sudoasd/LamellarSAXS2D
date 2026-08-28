from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from butterfly_saxs.p4_quality import evaluate_p4_ellipse_quality


def _ridge(
    *,
    width: float = 0.08,
    count: int = 72,
    continuity_fraction: float = 1.0,
    q_unit: str = "nm^-1",
) -> SimpleNamespace:
    points = [
        {
            "valid": True,
            "radial_fwhm": width,
            "snr": 20.0,
            "score": 20.0,
            "trajectory_id": 0,
        }
        for _ in range(count)
    ]
    return SimpleNamespace(
        points=points,
        valid_fraction=1.0,
        continuity_fraction=continuity_fraction,
        continuity_score=continuity_fraction,
        q_unit=q_unit,
    )


def _fit(**overrides):
    values = {
        "a": 0.86,
        "b": 0.54,
        "theta": np.deg2rad(12.0),
        "axes_ratio": 0.54 / 0.86,
        "rmse": 0.012,
        "success": True,
        "condition_number": 20.0,
        "coverage": SimpleNamespace(angular_coverage=0.95),
        "branch_counts": (36, 36),
        "bound_flags": {},
        "candidate_solutions": (),
        "multistart_count": 1,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_p4_engineering_quality_passes_well_supported_geometry() -> None:
    result = evaluate_p4_ellipse_quality(_ridge(), _fit())

    assert result["status"] == "PASS"
    assert result["scientific_status"] == "NOT_ACCEPTED"
    assert result["thresholds_frozen"] is False


def test_p4_engineering_quality_rejects_nonellipse_residual() -> None:
    result = evaluate_p4_ellipse_quality(_ridge(width=0.08), _fit(rmse=0.04))

    assert result["status"] == "FAIL"
    assert "residual_vs_local_width" in result["flags"]


def test_p4_engineering_quality_warns_when_theta_is_unidentifiable() -> None:
    result = evaluate_p4_ellipse_quality(
        _ridge(),
        _fit(axes_ratio=0.999, b=0.85914),
    )

    assert result["status"] == "WARN"
    assert "orientation_identifiability" in result["flags"]


def test_p4_engineering_quality_never_passes_uncalibrated_pixel_q() -> None:
    result = evaluate_p4_ellipse_quality(_ridge(q_unit="pixel-q"), _fit())

    assert result["status"] == "WARN"
    assert result["metrics"]["q_unit"] == "pixel-q"
    assert "physical_q_declared" in result["flags"]


def test_p4_engineering_quality_fails_without_two_branch_support() -> None:
    result = evaluate_p4_ellipse_quality(_ridge(), _fit(branch_counts=(72, 0)))

    assert result["status"] == "FAIL"
    assert "branch_support" in result["flags"]


def test_p4_engineering_quality_rejects_discontinuous_ridge() -> None:
    result = evaluate_p4_ellipse_quality(
        _ridge(continuity_fraction=0.1),
        _fit(),
    )

    assert result["status"] == "FAIL"
    assert "ridge_continuity" in result["flags"]


def test_p4_minimum_point_threshold_must_be_an_integer() -> None:
    with pytest.raises(ValueError, match="must be an integer"):
        evaluate_p4_ellipse_quality(
            _ridge(),
            _fit(),
            thresholds={"minimum_points": 5.5},
        )
