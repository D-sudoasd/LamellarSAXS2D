"""Input and mask handling for two-dimensional SAXS detector data.

The loader in this module deliberately has a small, strict contract:

* detector values are returned without normalisation or an implicit cast;
* a data source with more than one frame must be selected explicitly;
* HDF5/NPZ datasets are selected explicitly whenever the file is ambiguous;
* ``valid_mask`` uses ``True`` for a usable pixel, while an ``external_mask``
  uses ``True`` for a rejected pixel.

The latter distinction is important because both conventions are common in
beamline software.  The public result stores only the positive ``valid_mask``
and records the convention in its metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np


class DataIOError(ValueError):
    """Base error for a data source that cannot satisfy the loader contract."""


class UnsupportedFormatError(DataIOError):
    """The input suffix is not supported by the installed readers."""


class FrameSelectionError(DataIOError):
    """A frame was required, missing, malformed, or out of range."""


class DatasetSelectionError(DataIOError):
    """A dataset/key was required, missing, or ambiguous."""


class DataShapeError(DataIOError):
    """The selected object is not exactly one two-dimensional image."""


@dataclass
class LoadedImage:
    """A selected two-dimensional image and its source provenance.

    ``data`` is kept in the source dtype.  In particular, absolute intensity
    values are not divided by a monitor, exposure, or maximum value here.
    The caller can therefore apply a beamline-specific correction explicitly
    and retain an auditable record of that operation.
    """

    data: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    source: Path | None = None
    frame: int | None = None
    dataset: str | None = None
    valid_mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data)
        if self.data.ndim != 2:
            raise DataShapeError(
                f"selected image must be 2-D, got shape {self.data.shape!r}"
            )
        if self.valid_mask is not None:
            self.valid_mask = _coerce_mask_array(
                self.valid_mask, self.data.shape, name="valid_mask"
            )
        # Keep the source object useful even when callers construct it directly.
        self.metadata = dict(self.metadata)
        self.metadata.setdefault("shape", list(self.data.shape))
        self.metadata.setdefault("dtype", str(self.data.dtype))
        self.metadata.setdefault("absolute_intensity_preserved", True)
        self.metadata.setdefault(
            "intensity_semantics",
            "source values unchanged; no implicit normalisation",
        )
        self.metadata.setdefault(
            "mask_semantics",
            {
                "valid_mask_true": "valid pixel",
                "external_mask_true": "invalid pixel",
            },
        )

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.data.shape)  # type: ignore[return-value]

    @property
    def dtype(self) -> np.dtype[Any]:
        return self.data.dtype

    @property
    def mask(self) -> np.ndarray | None:
        """Compatibility alias using the detector convention (``True`` bad).

        New code should use :attr:`valid_mask` or :func:`combine_masks` to make
        the polarity explicit.  The domain ``ImageFrame`` uses this negative
        mask convention as well.
        """

        if self.valid_mask is None:
            return None
        return ~self.valid_mask

    @property
    def valid(self) -> np.ndarray | None:
        """Short alias for the positive valid mask."""

        return self.valid_mask

    @property
    def path(self) -> str | None:
        """Domain-model compatible source-path alias."""

        return None if self.source is None else str(self.source)

    @property
    def frame_id(self) -> int | None:
        return self.frame

    @property
    def preserves_absolute_intensity(self) -> bool:
        return bool(self.metadata.get("absolute_intensity_preserved", True))


# Names used by early clients and by the domain models are kept as aliases.
ImageFrame = LoadedImage
DataFrame = LoadedImage


_IMAGE_SUFFIXES = {
    ".cbf": "fabio",
    ".edf": "fabio",
    ".tif": "tiff",
    ".tiff": "tiff",
    ".npy": "npy",
    ".npz": "npz",
    ".csv": "csv",
    ".txt": "csv",
    ".h5": "hdf5",
    ".hdf5": "hdf5",
    ".hdf": "hdf5",
}


def load_image(
    path: str | Path,
    *,
    frame: int | None = None,
    dataset: str | None = None,
    valid_mask: Any | None = None,
    external_mask: Any | None = None,
    mask: Any | None = None,
) -> LoadedImage:
    """Read one two-dimensional detector frame.

    Parameters
    ----------
    path:
        CBF/EDF/TIF/TIFF/NPY/NPZ/CSV/TXT/HDF5 input path.
    frame:
        Zero-based frame index.  It is mandatory for a selected array whose
        first dimension contains more than one image.  ``frame=0`` is accepted
        for a single 2-D image as a convenient explicit form.
    dataset:
        HDF5 dataset path or NPZ key.  Ambiguous files fail closed when this is
        omitted.
    valid_mask:
        Optional boolean-like array/path where ``True`` means valid.
    external_mask (or ``mask``):
        Optional boolean-like array/path where ``True`` means masked/invalid.
        The two mask arguments cannot be supplied together.
    """

    source = Path(path).expanduser()
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(f"image source does not exist: {source}")
    if mask is not None:
        if external_mask is not None:
            raise DataIOError("provide only one of mask and external_mask")
        external_mask = mask
    _validate_frame_argument(frame)
    if dataset is not None and not isinstance(dataset, str):
        raise DatasetSelectionError("dataset must be a string path/key")

    suffix = source.suffix.lower()
    kind = _IMAGE_SUFFIXES.get(suffix)
    if kind is None:
        raise UnsupportedFormatError(
            f"unsupported image format {suffix or '<no suffix>'!r}; "
            f"supported: {', '.join(sorted(_IMAGE_SUFFIXES))}"
        )

    if kind == "fabio":
        array, metadata, selected_frame = _read_fabio(source, frame=frame)
        selected_dataset = None
    elif kind == "tiff":
        array, metadata, selected_frame = _read_tiff(source, frame=frame)
        selected_dataset = None
    elif kind == "npy":
        if dataset is not None:
            raise DatasetSelectionError("dataset is not applicable to NPY")
        array = _safe_numpy_load(source)
        metadata = {"format": "npy"}
        array, selected_frame = _select_frame(
            array, frame=frame, source=source, source_kind="NPY"
        )
        selected_dataset = None
    elif kind == "npz":
        array, metadata, selected_frame, selected_dataset = _read_npz(
            source, frame=frame, dataset=dataset
        )
    elif kind == "csv":
        if dataset is not None:
            raise DatasetSelectionError("dataset is not applicable to CSV/TXT")
        try:
            array = np.loadtxt(source, delimiter="," if suffix == ".csv" else None)
        except (OSError, ValueError) as exc:
            raise DataIOError(f"could not read tabular image {source}: {exc}") from exc
        metadata = {"format": "csv" if suffix == ".csv" else "txt"}
        array, selected_frame = _select_frame(
            array, frame=frame, source=source, source_kind="CSV/TXT"
        )
        selected_dataset = None
    else:
        array, metadata, selected_frame, selected_dataset = _read_hdf5(
            source, frame=frame, dataset=dataset
        )

    data = np.asarray(array)
    if data.ndim != 2:
        raise DataShapeError(
            f"selected image from {source} must be exactly 2-D, got {data.shape!r}"
        )
    if data.dtype.kind == "O":
        raise DataShapeError("object-valued image arrays are not supported")

    # HDF5 views must be detached before the file closes; copying every selected
    # array also makes the result independent of FabIO/TIFF file handles.
    data = np.array(data, copy=True)
    metadata = dict(metadata)
    metadata.update(
        {
            "source": str(source),
            "format": metadata.get("format", kind),
            "shape": list(data.shape),
            "dtype": str(data.dtype),
            "frame": selected_frame,
            "dataset": selected_dataset,
            "absolute_intensity_preserved": True,
            "intensity_semantics": "source values unchanged; no implicit normalisation",
            "mask_semantics": {
                "valid_mask_true": "valid pixel",
                "external_mask_true": "invalid pixel",
            },
        }
    )

    combined_mask = combine_masks(
        data.shape,
        valid_mask=_mask_value(valid_mask, frame=frame, dataset=dataset),
        external_mask=_mask_value(external_mask, frame=frame, dataset=dataset),
    )
    if combined_mask is not None:
        metadata["valid_pixel_count"] = int(np.count_nonzero(combined_mask))
        metadata["invalid_pixel_count"] = int(combined_mask.size - np.count_nonzero(combined_mask))

    return LoadedImage(
        data=data,
        metadata=metadata,
        source=source,
        frame=selected_frame,
        dataset=selected_dataset,
        valid_mask=combined_mask,
    )


def read_image(*args: Any, **kwargs: Any) -> LoadedImage:
    """Alias for :func:`load_image`."""

    return load_image(*args, **kwargs)


def load_frame(*args: Any, **kwargs: Any) -> LoadedImage:
    """Alias for :func:`load_image`."""

    return load_image(*args, **kwargs)


def read_data(*args: Any, **kwargs: Any) -> LoadedImage:
    """Alias for :func:`load_image`."""

    return load_image(*args, **kwargs)


def combine_masks(
    shape: Sequence[int],
    *,
    valid_mask: Any | None = None,
    external_mask: Any | None = None,
    mask: Any | None = None,
) -> np.ndarray | None:
    """Combine positive and external mask conventions into ``valid_mask``.

    ``valid_mask=True`` means a pixel is retained; ``external_mask=True``
    means a pixel is rejected.  The returned array is always boolean and has
    exactly ``shape``; broadcasting is intentionally not allowed.
    """

    if mask is not None:
        if external_mask is not None:
            raise DataIOError("provide only one of mask and external_mask")
        external_mask = mask
    shape_tuple = _validate_shape(shape)
    if valid_mask is None and external_mask is None:
        return None
    result = np.ones(shape_tuple, dtype=bool)
    if valid_mask is not None:
        result &= _load_mask_value(valid_mask, shape_tuple, name="valid_mask")
    if external_mask is not None:
        result &= ~_load_mask_value(
            external_mask, shape_tuple, name="external_mask"
        )
    return result


def resolve_valid_mask(
    shape: Sequence[int],
    *,
    valid_mask: Any | None = None,
    external_mask: Any | None = None,
    mask: Any | None = None,
) -> np.ndarray | None:
    """Explicitly named alias for :func:`combine_masks`."""

    return combine_masks(
        shape,
        valid_mask=valid_mask,
        external_mask=external_mask,
        mask=mask,
    )


def _validate_shape(shape: Sequence[int]) -> tuple[int, int]:
    if isinstance(shape, (str, bytes)):
        raise DataShapeError(f"image shape must contain two integers, got {shape!r}")
    try:
        values = tuple(shape)
    except TypeError as exc:
        raise DataShapeError(f"image shape must contain two integers, got {shape!r}") from exc
    if len(values) != 2 or any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in values):
        raise DataShapeError(f"image shape must contain two integers, got {shape!r}")
    out = (int(values[0]), int(values[1]))
    if out[0] <= 0 or out[1] <= 0:
        raise DataShapeError(f"image shape must be positive, got {out!r}")
    return out


def _validate_frame_argument(frame: int | None) -> None:
    if frame is None:
        return
    if isinstance(frame, bool) or not isinstance(frame, (int, np.integer)):
        raise FrameSelectionError("frame must be a non-negative integer or None")
    if int(frame) < 0:
        raise FrameSelectionError("frame must be a non-negative integer")


def _select_frame(
    array: Any,
    *,
    frame: int | None,
    source: Path,
    source_kind: str,
) -> tuple[np.ndarray, int | None]:
    ary = np.asarray(array)
    if ary.ndim < 2:
        raise DataShapeError(
            f"selected object from {source} must be 2-D, got {ary.shape!r}"
        )
    if ary.ndim == 2:
        if frame is not None and int(frame) != 0:
            raise FrameSelectionError(
                f"{source_kind} source {source} contains one image; frame must be 0"
            )
        return ary, 0 if frame is not None else None
    if frame is None:
        raise FrameSelectionError(
            f"{source_kind} source {source} contains multiple dimensions/frames; "
            "select an explicit frame=..."
        )
    index = int(frame)
    if index >= ary.shape[0]:
        raise FrameSelectionError(
            f"frame {index} is outside {source_kind} source {source} "
            f"with {ary.shape[0]} frames"
        )
    selected = ary[index]
    if np.asarray(selected).ndim != 2:
        raise DataShapeError(
            f"frame {index} from {source} is not a 2-D image: {np.asarray(selected).shape!r}"
        )
    return np.asarray(selected), index


def _safe_numpy_load(source: Path) -> np.ndarray:
    try:
        value = np.load(source, allow_pickle=False)
    except (OSError, ValueError, EOFError) as exc:
        raise DataIOError(f"could not read NPY source {source}: {exc}") from exc
    if not isinstance(value, np.ndarray):
        raise DataShapeError(f"NPY source {source} did not contain an ndarray")
    return value


def _read_npz(
    source: Path,
    *,
    frame: int | None,
    dataset: str | None,
) -> tuple[np.ndarray, dict[str, Any], int | None, str]:
    if dataset is not None and not dataset:
        raise DatasetSelectionError("NPZ dataset/key cannot be empty")
    try:
        archive = np.load(source, allow_pickle=False)
    except (OSError, ValueError, EOFError) as exc:
        raise DataIOError(f"could not read NPZ source {source}: {exc}") from exc
    try:
        keys = list(archive.files)
        if dataset is None:
            if len(keys) != 1:
                raise DatasetSelectionError(
                    f"NPZ source {source} has datasets {keys!r}; select dataset='...'"
                )
            selected_dataset = keys[0]
        else:
            selected_dataset = dataset
            if selected_dataset not in keys:
                raise DatasetSelectionError(
                    f"dataset {selected_dataset!r} not found in NPZ source {source}; "
                    f"available: {keys!r}"
                )
        array = archive[selected_dataset]
        array, selected_frame = _select_frame(
            array, frame=frame, source=source, source_kind="NPZ"
        )
        metadata = {"format": "npz", "datasets": keys}
        return array, metadata, selected_frame, selected_dataset
    finally:
        archive.close()


def _read_tiff(
    source: Path, *, frame: int | None
) -> tuple[np.ndarray, dict[str, Any], int | None]:
    try:
        import tifffile

        with tifffile.TiffFile(source) as tif:
            pages = len(tif.pages)
            series_axes = [getattr(s, "axes", None) for s in tif.series]
            imagej_metadata = tif.imagej_metadata
            ome_metadata = tif.ome_metadata
            tags: dict[str, Any] = {}
            if pages:
                for key, tag in tif.pages[0].tags.items():
                    try:
                        tags[str(key)] = tag.value
                    except Exception:
                        # A malformed vendor tag must not prevent intensity data
                        # from being read; the tag itself is simply omitted.
                        continue
            array = tif.asarray()
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise DataIOError("TIFF support requires tifffile") from exc
    except (OSError, ValueError, RuntimeError) as exc:
        raise DataIOError(f"could not read TIFF source {source}: {exc}") from exc
    metadata: dict[str, Any] = {
        "format": "tiff",
        "page_count": pages,
        "series_axes": series_axes,
        "imagej_metadata": imagej_metadata,
        "ome_metadata": ome_metadata,
        "tags": tags,
    }
    selected, selected_frame = _select_frame(
        array, frame=frame, source=source, source_kind="TIFF"
    )
    return selected, metadata, selected_frame


def _read_fabio(
    source: Path, *, frame: int | None
) -> tuple[np.ndarray, dict[str, Any], int | None]:
    try:
        import fabio

        first = fabio.open(str(source))
        nframes = int(getattr(first, "nframes", 1) or 1)
        if frame is None:
            if nframes > 1:
                raise FrameSelectionError(
                    f"FabIO source {source} contains {nframes} frames; "
                    "select an explicit frame=..."
                )
            image = first
            selected_frame = None
        else:
            if int(frame) >= nframes:
                raise FrameSelectionError(
                    f"frame {int(frame)} is outside FabIO source {source} "
                    f"with {nframes} frames"
                )
            image = fabio.open(str(source), frame=int(frame))
            selected_frame = int(frame)
        data = np.asarray(image.data)
        header = dict(getattr(image, "header", {}) or {})
        metadata = {
            "format": source.suffix.lower().lstrip("."),
            "fabio_class": type(image).__name__,
            "frame_count": nframes,
            "header": header,
        }
        return data, metadata, selected_frame
    except FrameSelectionError:
        raise
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise DataIOError("CBF/EDF support requires fabio") from exc
    except (OSError, ValueError, RuntimeError, TypeError) as exc:
        raise DataIOError(f"could not read FabIO source {source}: {exc}") from exc


def _read_hdf5(
    source: Path,
    *,
    frame: int | None,
    dataset: str | None,
) -> tuple[np.ndarray, dict[str, Any], int | None, str]:
    if dataset is not None and not dataset:
        raise DatasetSelectionError("HDF5 dataset path cannot be empty")
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise DataIOError("HDF5 support requires h5py; install butterfly-saxs[hdf5]") from exc
    try:
        with h5py.File(source, "r") as handle:
            datasets: list[str] = []

            def visit(name: str, obj: Any) -> None:
                if isinstance(obj, h5py.Dataset):
                    datasets.append(name)

            handle.visititems(visit)
            if dataset is None:
                if len(datasets) != 1:
                    raise DatasetSelectionError(
                        f"HDF5 source {source} has datasets {datasets!r}; "
                        "select dataset='...' explicitly"
                    )
                selected_dataset = datasets[0]
            else:
                selected_dataset = dataset.lstrip("/")
                if selected_dataset not in datasets:
                    raise DatasetSelectionError(
                        f"dataset {dataset!r} not found in HDF5 source {source}; "
                        f"available: {datasets!r}"
                    )
            node = handle[selected_dataset]
            array = np.asarray(node[...])
            array, selected_frame = _select_frame(
                array, frame=frame, source=source, source_kind="HDF5"
            )
            metadata: dict[str, Any] = {
                "format": "hdf5",
                "datasets": datasets,
                "file_attrs": _attribute_dict(handle.attrs),
                "dataset_attrs": _attribute_dict(node.attrs),
            }
            return array, metadata, selected_frame, selected_dataset
    except (DatasetSelectionError, FrameSelectionError, DataShapeError):
        raise
    except (OSError, ValueError, RuntimeError, TypeError) as exc:
        raise DataIOError(f"could not read HDF5 source {source}: {exc}") from exc


def _attribute_dict(attrs: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in attrs:
        try:
            value = attrs[key]
            if isinstance(value, np.generic):
                value = value.item()
            result[str(key)] = value
        except Exception:
            continue
    return result


def _mask_value(value: Any | None, *, frame: int | None, dataset: str | None) -> Any | None:
    """Load a mask path using the same explicit selectors as the image."""

    if value is None or not isinstance(value, (str, Path)):
        return value
    loaded = load_image(value, frame=frame, dataset=dataset)
    return loaded.data


def _load_mask_value(value: Any, shape: tuple[int, int], *, name: str) -> np.ndarray:
    if isinstance(value, (str, Path)):
        value = load_image(value).data
    return _coerce_mask_array(value, shape, name=name)


def _coerce_mask_array(value: Any, shape: Sequence[int], *, name: str) -> np.ndarray:
    expected = _validate_shape(shape)
    array = np.asarray(value)
    if array.shape != expected:
        raise DataShapeError(
            f"{name} shape {array.shape!r} does not exactly match image shape {expected!r}"
        )
    if array.dtype.kind == "O":
        raise DataShapeError(f"{name} cannot be object-valued")
    return np.asarray(array != 0, dtype=bool)


__all__ = [
    "DataFrame",
    "DataIOError",
    "DataShapeError",
    "DatasetSelectionError",
    "FrameSelectionError",
    "ImageFrame",
    "LoadedImage",
    "UnsupportedFormatError",
    "combine_masks",
    "load_frame",
    "load_image",
    "read_data",
    "read_image",
    "resolve_valid_mask",
]
