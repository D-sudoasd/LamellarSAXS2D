"""Detector geometry and physical coordinate maps.

All PONI values are interpreted by pyFAI.  This module intentionally does not
reimplement detector rotations or pixel coordinates: ``pyFAI.load`` creates the
official :class:`AzimuthalIntegrator`, and its ``qArray`` plus
``center_array('chi_rad')`` methods are used for every pixel.  The in-plane
components are then the unambiguous polar decomposition
``qx = q*cos(chi)``, ``qy = q*sin(chi)`` (0 degrees is +qx, 90 degrees is
+qy).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


class GeometryError(ValueError):
    """A PONI file or geometry map cannot satisfy the strict contract."""


class GeometryShapeError(GeometryError):
    """The requested map shape is invalid or does not match a mask."""


@dataclass
class GeometryMaps:
    """Physical coordinates and the mask associated with one detector shape."""

    q_nm_inv: np.ndarray
    chi_rad: np.ndarray
    qx_nm_inv: np.ndarray
    qy_nm_inv: np.ndarray
    valid_mask: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    fingerprint: str = ""
    chi_deg: np.ndarray | None = None

    def __post_init__(self) -> None:
        arrays = {
            "q_nm_inv": self.q_nm_inv,
            "chi_rad": self.chi_rad,
            "qx_nm_inv": self.qx_nm_inv,
            "qy_nm_inv": self.qy_nm_inv,
        }
        converted: dict[str, np.ndarray] = {}
        for name, value in arrays.items():
            array = np.asarray(value, dtype=np.float64)
            converted[name] = array
        shapes = {name: value.shape for name, value in converted.items()}
        if len(set(shapes.values())) != 1:
            raise GeometryShapeError(f"geometry maps have inconsistent shapes: {shapes!r}")
        if len(next(iter(shapes.values()))) != 2:
            raise GeometryShapeError(
                f"geometry maps must be 2-D, got {next(iter(shapes.values()))!r}"
            )
        mask = np.asarray(self.valid_mask, dtype=bool)
        if mask.shape != next(iter(shapes.values())):
            raise GeometryShapeError(
                f"valid_mask shape {mask.shape!r} does not match map shape "
                f"{next(iter(shapes.values()))!r}"
            )
        for name, value in converted.items():
            setattr(self, name, value)
        self.valid_mask = mask
        if self.chi_deg is None:
            self.chi_deg = np.degrees(self.chi_rad) % 360.0
        else:
            self.chi_deg = np.asarray(self.chi_deg, dtype=np.float64)
            if self.chi_deg.shape != mask.shape:
                raise GeometryShapeError("chi_deg shape does not match geometry maps")
        self.metadata = dict(self.metadata)
        self.metadata.setdefault("shape", list(mask.shape))
        self.metadata.setdefault("q_unit", "nm^-1")
        self.metadata.setdefault("chi_unit", "rad")
        self.metadata.setdefault(
            "chi_convention",
            {
                "zero_degrees": "+qx",
                "ninety_degrees": "+qy",
                "positive_direction": "counter-clockwise in the qx/qy plane",
            },
        )
        self.metadata.setdefault(
            "mask_semantics",
            {
                "valid_mask_true": "valid pixel",
                "external_mask_true": "invalid pixel",
            },
        )

    @property
    def q(self) -> np.ndarray:
        return self.q_nm_inv

    @property
    def chi(self) -> np.ndarray:
        return self.chi_rad

    @property
    def qx(self) -> np.ndarray:
        return self.qx_nm_inv

    @property
    def qy(self) -> np.ndarray:
        return self.qy_nm_inv

    @property
    def mask(self) -> np.ndarray:
        """Detector-style negative mask (``True`` means invalid).

        The canonical positive representation remains :attr:`valid_mask`;
        this alias lets the result pass through the domain ``QMap`` seam,
        whose historical ``mask`` field follows FabIO/pyFAI convention.
        """

        return ~self.valid_mask

    @property
    def valid(self) -> np.ndarray:
        return self.valid_mask

    @property
    def geometry_fingerprint(self) -> str:
        return self.fingerprint

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.q_nm_inv.shape)  # type: ignore[return-value]

    def as_dict(self) -> dict[str, Any]:
        """Return metadata and fingerprint without duplicating large arrays."""

        return {
            "fingerprint": self.fingerprint,
            "metadata": dict(self.metadata),
            "shape": list(self.shape),
        }


# QMap is the vocabulary used by the domain models and an intuitive public
# alias for callers that do not need to distinguish map construction details.
QMap = GeometryMaps
GeometryMap = GeometryMaps


def load_poni(poni: str | Path | Any) -> Any:
    """Load a PONI through pyFAI's supported loader.

    An already-created AzimuthalIntegrator is accepted so a UI can reuse a
    live, user-refined geometry without serialising it first.
    """

    if _looks_like_azimuthal_integrator(poni):
        return poni
    source = Path(poni).expanduser()
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(f"PONI file does not exist: {source}")
    try:
        import pyFAI

        # pyFAI.load parses both legacy and current PONI representations and
        # preserves all detector rotations.  Do not parse rot1/rot2/rot3 here.
        return pyFAI.load(str(source))
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise GeometryError("PONI geometry requires pyFAI") from exc
    except Exception as exc:
        raise GeometryError(f"could not load PONI file {source}: {exc}") from exc


def build_geometry(
    shape: Sequence[int] | str | Path,
    poni: str | Path | Any | Sequence[int] | None = None,
    *,
    valid_mask: Any | None = None,
    external_mask: Any | None = None,
    mask: Any | None = None,
) -> GeometryMaps:
    """Build q/chi/qx/qy maps for exactly ``shape`` pixels.

    The canonical call is ``build_geometry((rows, cols), poni_path)``.  For
    compatibility with callers that naturally put the PONI first,
    ``build_geometry(poni_path, (rows, cols))`` is also accepted.
    """

    if _looks_like_shape(poni) and not _looks_like_shape(shape):
        shape, poni = poni, shape
    if poni is None:
        raise GeometryError("a PONI path or AzimuthalIntegrator is required")
    shape_tuple = _validate_shape(shape)
    ai = load_poni(poni)

    try:
        q = np.asarray(ai.qArray(shape_tuple), dtype=np.float64)
        # center_array is the current pyFAI precision path.  chiArray is kept
        # as a compatibility fallback for old pyFAI installations only.
        try:
            chi = np.asarray(
                ai.center_array(shape_tuple, unit="chi_rad", scale=True),
                dtype=np.float64,
            )
        except (AttributeError, TypeError):  # pragma: no cover - old pyFAI
            chi = np.asarray(ai.chiArray(shape_tuple), dtype=np.float64)
    except Exception as exc:
        raise GeometryError(
            f"pyFAI could not calculate q/chi maps for shape {shape_tuple!r}: {exc}"
        ) from exc

    if q.shape != shape_tuple or chi.shape != shape_tuple:
        raise GeometryShapeError(
            f"pyFAI returned q shape {q.shape!r} and chi shape {chi.shape!r}; "
            f"requested {shape_tuple!r}"
        )
    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(chi)):
        raise GeometryError("PONI produced non-finite q/chi values")

    # pyFAI's chi is an azimuth in radians.  Explicitly normalise its phase so
    # the public convention remains 0 deg=+qx and 90 deg=+qy for every PONI.
    chi = np.mod(chi, 2.0 * np.pi)
    qx = q * np.cos(chi)
    qy = q * np.sin(chi)

    finite_mask = np.isfinite(q) & np.isfinite(chi) & np.isfinite(qx) & np.isfinite(qy)
    if valid_mask is None and external_mask is None and mask is None:
        combined_mask = finite_mask
    else:
        if mask is not None:
            if external_mask is not None:
                raise GeometryError("provide only one of mask and external_mask")
            external_mask = mask
        from .io import combine_masks  # local import avoids an import cycle

        user_mask = combine_masks(
            shape_tuple, valid_mask=valid_mask, external_mask=external_mask
        )
        combined_mask = finite_mask if user_mask is None else finite_mask & user_mask

    metadata = _geometry_metadata(ai, poni, shape_tuple, combined_mask)
    fingerprint = _fingerprint(ai, poni, shape_tuple, metadata)
    metadata["fingerprint"] = fingerprint
    return GeometryMaps(
        q_nm_inv=q,
        chi_rad=chi,
        qx_nm_inv=qx,
        qy_nm_inv=qy,
        valid_mask=combined_mask,
        metadata=metadata,
        fingerprint=fingerprint,
    )


def geometry_from_poni(
    poni: str | Path | Any,
    shape: Sequence[int],
    **kwargs: Any,
) -> GeometryMaps:
    """PONI-first spelling of :func:`build_geometry`."""

    return build_geometry(shape, poni, **kwargs)


def build_q_map(
    poni: str | Path | Any,
    shape: Sequence[int],
    **kwargs: Any,
) -> GeometryMaps:
    """Alias used by q-map consumers."""

    return geometry_from_poni(poni, shape, **kwargs)


def q_map_from_poni(
    poni: str | Path | Any,
    shape: Sequence[int],
    **kwargs: Any,
) -> GeometryMaps:
    """Alias for :func:`build_q_map`."""

    return build_q_map(poni, shape, **kwargs)


def qmap_from_poni(
    poni: str | Path | Any,
    shape: Sequence[int],
    **kwargs: Any,
) -> GeometryMaps:
    """Underscore-free alias used by the pipeline adapter discovery."""

    return build_q_map(poni, shape, **kwargs)


def build_qmap(
    image: Any | None = None,
    *,
    poni: str | Path | Any | None = None,
    shape: Sequence[int] | None = None,
    valid_mask: Any | None = None,
    external_mask: Any | None = None,
    mask: Any | None = None,
    **_: Any,
) -> GeometryMaps:
    """Pipeline-facing q-map adapter.

    ``pipeline.build_qmap`` supplies an image and a PONI keyword.  This
    adapter derives only the image shape; every physical coordinate still
    comes from :func:`build_geometry` and pyFAI.
    """

    if shape is None:
        if image is None or not hasattr(image, "shape"):
            raise GeometryShapeError("build_qmap requires shape or a 2-D image")
        shape = tuple(np.asarray(image).shape)
    if poni is None:
        raise GeometryError("build_qmap requires a PONI path or integrator")
    return build_geometry(
        shape,
        poni,
        valid_mask=valid_mask,
        external_mask=external_mask,
        mask=mask,
    )


compute_qmap = build_qmap
make_qmap = build_qmap
pixel_qmap = build_qmap


def _looks_like_azimuthal_integrator(value: Any) -> bool:
    return value is not None and all(
        callable(getattr(value, name, None)) for name in ("qArray", "center_array")
    )


def _looks_like_shape(value: Any) -> bool:
    if value is None or isinstance(value, (str, bytes, Path)):
        return False
    try:
        items = tuple(value)
    except TypeError:
        return False
    return len(items) == 2 and all(
        not isinstance(item, bool) and isinstance(item, (int, np.integer))
        for item in items
    )


def _validate_shape(shape: Any) -> tuple[int, int]:
    if not _looks_like_shape(shape):
        raise GeometryShapeError(f"shape must be two integer dimensions, got {shape!r}")
    result = (int(shape[0]), int(shape[1]))
    if result[0] <= 0 or result[1] <= 0:
        raise GeometryShapeError(f"shape must be positive, got {result!r}")
    return result


def _geometry_metadata(
    ai: Any,
    poni: Any,
    shape: tuple[int, int],
    valid_mask: np.ndarray,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "shape": list(shape),
        "q_unit": "nm^-1",
        "chi_unit": "rad",
        "chi_zero_deg": "+qx",
        "chi_ninety_deg": "+qy",
        "chi_convention": {
            "zero_degrees": "+qx",
            "ninety_degrees": "+qy",
            "positive_direction": "counter-clockwise in the qx/qy plane",
        },
        "mask_semantics": {
            "valid_mask_true": "valid pixel",
            "external_mask_true": "invalid pixel",
        },
        "valid_pixel_count": int(np.count_nonzero(valid_mask)),
        "invalid_pixel_count": int(valid_mask.size - np.count_nonzero(valid_mask)),
    }
    try:
        import pyFAI

        metadata["pyfai_version"] = str(getattr(pyFAI, "__version__", getattr(pyFAI, "version", "unknown")))
    except Exception:  # pragma: no cover - pyFAI import was already successful
        metadata["pyfai_version"] = "unknown"
    for name in (
        "dist",
        "poni1",
        "poni2",
        "rot1",
        "rot2",
        "rot3",
        "wavelength",
        "orientation",
    ):
        try:
            value = getattr(ai, name)
        except Exception:
            continue
        metadata[name] = _json_value(value)
    detector = getattr(ai, "detector", None)
    if detector is not None:
        detector_meta: dict[str, Any] = {"class": type(detector).__name__}
        for name in ("name", "pixel1", "pixel2", "max_shape", "shape", "orientation"):
            try:
                detector_meta[name] = _json_value(getattr(detector, name))
            except Exception:
                continue
        metadata["detector"] = detector_meta
    if isinstance(poni, (str, Path)):
        source = Path(poni).expanduser()
        metadata["poni_source"] = str(source)
        try:
            metadata["poni_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        except OSError:
            pass
    else:
        metadata["poni_source"] = "in-memory AzimuthalIntegrator"
    return metadata


def _fingerprint(ai: Any, poni: Any, shape: tuple[int, int], metadata: Mapping[str, Any]) -> str:
    payload: dict[str, Any] = {
        "shape": list(shape),
        "metadata": _json_value(dict(metadata)),
    }
    # The bytes digest captures user edits to a PONI that do not happen to be
    # exposed as an AI attribute.  For in-memory geometry, metadata is the
    # complete serialisable parameter record available to us.
    if isinstance(poni, (str, Path)):
        try:
            payload["poni_bytes_sha256"] = hashlib.sha256(
                Path(poni).expanduser().read_bytes()
            ).hexdigest()
        except OSError:
            pass
    payload["ai_class"] = type(ai).__name__
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _json_value(value: Any) -> Any:
    """Convert numpy/scalar metadata to deterministic JSON-compatible values."""

    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


__all__ = [
    "GeometryError",
    "GeometryMap",
    "GeometryMaps",
    "GeometryShapeError",
    "QMap",
    "build_geometry",
    "build_qmap",
    "build_q_map",
    "compute_qmap",
    "geometry_from_poni",
    "load_poni",
    "make_qmap",
    "pixel_qmap",
    "qmap_from_poni",
    "q_map_from_poni",
]
