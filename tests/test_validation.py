from __future__ import annotations

import json

import numpy as np
import pytest

from butterfly_saxs.validation import AnalysisDomainError, build_analysis_domain


def _grid(shape: tuple[int, int] = (4, 5)) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y, x = np.indices(shape, dtype=float)
    qx = x - 2.0
    qy = y - 1.5
    return qx, qy, np.hypot(qx, qy)


def test_analysis_domain_reports_cumulative_counts_and_sample_mask() -> None:
    image = np.ones((4, 5), dtype=float)
    image[0, 0] = np.nan
    qx, qy, q = _grid(image.shape)
    detector_valid = np.ones(image.shape, dtype=bool)
    detector_valid[0, 1] = False
    external_mask = np.zeros(image.shape, dtype=bool)
    external_mask[0, 2] = True
    roi = np.zeros(image.shape, dtype=bool)
    roi[0, 3] = True

    domain = build_analysis_domain(
        image,
        qx,
        qy,
        q=q,
        detector_valid=detector_valid,
        external_mask=external_mask,
        roi_exclusion=roi,
        q_window=(0.0, 10.0),
    )
    indices = np.flatnonzero(domain.fit_valid_mask)[::2]
    sampled = domain.with_sampled_indices(indices)

    assert domain.counts == {
        "image_pixel_count": 20,
        "finite_pixel_count": 19,
        "detector_valid_count": 18,
        "external_mask_excluded_count": 1,
        "external_valid_count": 17,
        "q_window_pixel_count": 17,
        "roi_excluded_count": 1,
        "weight_invalid_count": 0,
        "fit_pixel_count": 16,
        "sampled_pixel_count": 16,
    }
    assert sampled.counts["sampled_pixel_count"] == len(indices)
    assert np.all(sampled.sampled_valid_mask <= sampled.fit_valid_mask)
    json.dumps(sampled.to_summary(), allow_nan=False)


@pytest.mark.parametrize("name", ["detector_valid", "external_mask", "sigma", "weights"])
def test_analysis_domain_rejects_shape_mismatch(name: str) -> None:
    image = np.ones((4, 5))
    qx, qy, _ = _grid(image.shape)
    kwargs = {name: np.ones((3, 5))}
    with pytest.raises(AnalysisDomainError, match="shape"):
        build_analysis_domain(image, qx, qy, **kwargs)


def test_analysis_domain_rejects_invalid_window_and_weights() -> None:
    image = np.ones((4, 5))
    qx, qy, _ = _grid(image.shape)
    with pytest.raises(AnalysisDomainError, match="max > min"):
        build_analysis_domain(image, qx, qy, q_window=(2.0, 1.0))

    sigma = np.ones(image.shape)
    sigma[1, 1] = 0.0
    with pytest.raises(AnalysisDomainError, match="non-finite or non-positive"):
        build_analysis_domain(image, qx, qy, sigma=sigma)


def test_analysis_domain_rejects_non_numeric_image_and_bad_sample_indices() -> None:
    qx, qy, _ = _grid()
    with pytest.raises(AnalysisDomainError, match="numeric"):
        build_analysis_domain(np.full((4, 5), "x"), qx, qy)

    detector_valid = np.ones((4, 5), dtype=bool)
    detector_valid[0, 0] = False
    domain = build_analysis_domain(
        np.ones((4, 5)), qx, qy, detector_valid=detector_valid
    )
    with pytest.raises(AnalysisDomainError, match="outside"):
        domain.with_sampled_indices([0])
    with pytest.raises(AnalysisDomainError, match="integers"):
        domain.with_sampled_indices([1.5])
