"""Qt-free parameter records and the refinement parameter table model.

The model accepts the small parameter objects used by the core package as
well as ordinary mappings.  Keeping that conversion here lets the workbench
remain useful while the scientific engine evolves without making the UI
depend on one particular core implementation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from math import isfinite
import math
from numbers import Real
from typing import Any, Callable

from .i18n import translate, validate_language
from .qt_compat import QT_AVAILABLE, QtCore


_HEADER_KEYS = (
    "header.parameter",
    "header.value",
    "header.min",
    "header.max",
    "header.vary",
    "header.expr",
    "header.unit",
    "header.stderr",
)

# Tooltip keys intentionally use the visible column names from the public
# parameter-table contract (Parameter/Value/Min/Max/Vary/Expr/Unit/Stderr),
# rather than the internal ``ParameterRow`` attribute names.  The latter are
# implementation details and should not leak into the translation catalog.
_HEADER_TOOLTIP_KEYS = (
    "tooltip.header.parameter",
    "tooltip.header.value",
    "tooltip.header.minimum",
    "tooltip.header.maximum",
    "tooltip.header.vary",
    "tooltip.header.expression",
    "tooltip.header.unit",
    "tooltip.header.stderr",
)

# These are the model parameters for which the application can state a
# scientifically meaningful description.  Descriptions live in i18n so the
# same table can be rendered in either supported language.  A parameter that
# is not in this map deliberately uses the neutral ``unknown`` description;
# the UI must not infer material meaning from an arbitrary model extension.
_PARAMETER_TOOLTIP_KEYS = {
    name: f"tooltip.parameter.{name}"
    for name in (
        "a",
        "b",
        "cx",
        "cy",
        "axis_ratio",
        "theta",
        "theta_deg",
        "lobe_angle",
        "lobe_angle_deg",
        "angular_width",
        "angular_width_deg",
        "radial_sigma",
        "radial_gamma",
        "radial_fwhm",
        "radial_width",
        "eta",
        "amplitude",
        "amplitude_plus",
        "amplitude_minus",
        "background",
        "background_slope",
        "background_curvature",
        "background_amplitude",
        "background_width",
        "q_center",
        "q_major",
        "q_minor",
        "ellipticity",
        "intensity",
        "ridge_width",
    )
}


def _read_value(source: Any, names: tuple[str, ...], default: Any = None) -> Any:
    if isinstance(source, Mapping):
        for name in names:
            if name in source:
                return source[name]
        return default
    for name in names:
        if hasattr(source, name):
            return getattr(source, name)
    return default


def _number(value: Any, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if isfinite(result) else default


def _format_number(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not isfinite(number):
        return str(value)
    return f"{number:.8g}"


@dataclass
class ParameterRow:
    """A single editable fitting parameter.

    ``minimum``/``maximum`` deliberately use ``None`` for an open bound.
    Values are kept numeric whenever possible but can be strings while a user
    edits an expression or when a core parameter is categorical.
    """

    name: str
    value: Any = 0.0
    minimum: float | None = None
    maximum: float | None = None
    vary: bool = True
    expression: str = ""
    unit: str = ""
    stderr: float | None = None

    @classmethod
    def from_any(cls, source: Any, name: str | None = None) -> "ParameterRow":
        if isinstance(source, cls):
            return cls(**asdict(source))
        resolved_name = name or _read_value(source, ("name", "label", "key"), "parameter")
        if isinstance(source, Real) and not isinstance(source, bool):
            return cls(name=str(resolved_name), value=source)
        return cls(
            name=str(resolved_name),
            value=_read_value(source, ("value", "val", "initial"), 0.0),
            minimum=_number(_read_value(source, ("minimum", "min", "lower", "min_value"))),
            maximum=_number(_read_value(source, ("maximum", "max", "upper", "max_value"))),
            vary=bool(_read_value(source, ("vary", "free", "variable"), True)),
            expression=str(_read_value(source, ("expression", "expr", "constraint"), "") or ""),
            unit=str(_read_value(source, ("unit", "units"), "") or ""),
            stderr=_number(_read_value(source, ("stderr", "standard_error", "error"))),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def coerce_parameter_rows(parameters: Any = None) -> list[ParameterRow]:
    """Convert common engine parameter containers to :class:`ParameterRow`.

    Supported inputs include a mapping of names to values/specifications,
    objects exposing ``parameters``, iterables of parameter objects and a
    single parameter object.  Unknown values are represented conservatively
    instead of raising during UI construction.
    """

    if parameters is None:
        return []
    nested = _read_value(parameters, ("parameters", "parameter_set", "params"), None)
    if nested is not None and nested is not parameters:
        parameters = nested
    # ``ParameterSet.items()`` intentionally exposes resolved numeric values
    # for analysis code.  Editors need the richer ``ParameterSpec`` objects
    # from its explicit spec_items/specs seam instead.
    spec_items = getattr(parameters, "spec_items", None)
    if callable(spec_items):
        try:
            parameters = dict(spec_items())
        except (TypeError, ValueError):
            pass
    elif hasattr(parameters, "specs"):
        try:
            parameters = dict(getattr(parameters, "specs"))
        except (TypeError, ValueError):
            pass
    if isinstance(parameters, Mapping):
        rows = [ParameterRow.from_any(value, str(name)) for name, value in parameters.items()]
    elif isinstance(parameters, (str, bytes)):
        rows = [ParameterRow.from_any(parameters)]
    else:
        try:
            rows = [ParameterRow.from_any(value) for value in parameters]
        except TypeError:
            rows = [ParameterRow.from_any(parameters)]
    # The scientific core uses radians for its optimizer, but the public UI
    # contract is explicit and degree-labelled.  Infer units only when a
    # caller did not provide one, preserving beamline-specific units.
    for row in rows:
        if not row.unit:
            if row.name in {"theta_deg", "orientation_deg", "angle_deg"}:
                row.unit = "degree"
            elif row.name in {"theta", "orientation", "angle"}:
                row.unit = "radian"
    return rows


if QT_AVAILABLE:

    class ParameterTableModel(QtCore.QAbstractTableModel):
        """Editable ``QAbstractTableModel`` for refinement parameters."""

        parameterChanged = QtCore.Signal(str, str, object)
        parametersChanged = QtCore.Signal()

        COLUMNS = (
            ("name", "Parameter"),
            ("value", "Value"),
            ("minimum", "Min"),
            ("maximum", "Max"),
            ("vary", "Vary"),
            ("expression", "Expr"),
            ("unit", "Unit"),
            ("stderr", "Stderr"),
        )
        COLUMN_KEYS = tuple(key for key, _ in COLUMNS)
        HEADER_LABELS = tuple(label for _, label in COLUMNS)
        # Suggested widths keep the constraint/uncertainty columns visible in
        # the dock; views may still resize them interactively.
        COLUMN_WIDTHS = (128, 82, 72, 72, 58, 112, 70, 76)
        headers = HEADER_LABELS

        def __init__(
            self,
            parameters: Any = None,
            parent: Any = None,
            *,
            language: str = "en",
        ) -> None:
            super().__init__(parent)
            self._language = validate_language(language)
            self._rows: list[ParameterRow] = coerce_parameter_rows(parameters)

        @property
        def language(self) -> str:
            return self._language

        def set_language(self, language: str) -> None:
            resolved = validate_language(language)
            if resolved == self._language:
                return
            self._language = resolved
            self.headerDataChanged.emit(
                QtCore.Qt.Orientation.Horizontal,
                0,
                len(self.COLUMNS) - 1,
            )
            if self._rows:
                self.dataChanged.emit(
                    self.index(0, 0),
                    self.index(len(self._rows) - 1, len(self.COLUMNS) - 1),
                    [QtCore.Qt.ItemDataRole.ToolTipRole],
                )

        @property
        def rows(self) -> list[ParameterRow]:
            return self._rows

        def rowCount(self, parent: Any = None) -> int:  # noqa: N802 - Qt API
            return 0 if parent is not None and parent.isValid() else len(self._rows)

        def columnCount(self, parent: Any = None) -> int:  # noqa: N802 - Qt API
            return 0 if parent is not None and parent.isValid() else len(self.COLUMNS)

        def headerData(self, section: int, orientation: Any, role: Any = None) -> Any:  # noqa: N802
            if orientation == QtCore.Qt.Orientation.Horizontal:
                if not 0 <= section < len(self.COLUMNS):
                    return None
                if role == QtCore.Qt.ItemDataRole.ToolTipRole:
                    return translate(self._language, _HEADER_TOOLTIP_KEYS[section])
                if role not in (None, QtCore.Qt.ItemDataRole.DisplayRole):
                    return None
                return translate(self._language, _HEADER_KEYS[section])
            if role not in (None, QtCore.Qt.ItemDataRole.DisplayRole):
                return None
            return str(section + 1)

        def _value_for(self, row: ParameterRow, key: str) -> Any:
            return getattr(row, key)

        def _parameter_tooltip(self, row: ParameterRow, column: int) -> str:
            """Build one localized cell tooltip without changing row data."""

            description_key = _PARAMETER_TOOLTIP_KEYS.get(
                row.name,
                "tooltip.parameter.unknown",
            )
            description = translate(self._language, description_key, name=row.name)
            field = translate(self._language, _HEADER_TOOLTIP_KEYS[column])
            unit = row.unit or translate(self._language, "tooltip.parameter.unit_none")
            return translate(
                self._language,
                "tooltip.parameter.cell",
                description=description,
                field=field,
                unit=unit,
            )

        def data(self, index: Any, role: Any = None) -> Any:  # noqa: N802 - Qt API
            if not index.isValid() or not (0 <= index.row() < len(self._rows)):
                return None
            if not 0 <= index.column() < len(self.COLUMNS):
                return None
            row = self._rows[index.row()]
            key = self.COLUMNS[index.column()][0]
            value = self._value_for(row, key)
            if key == "vary" and role == QtCore.Qt.ItemDataRole.CheckStateRole:
                return QtCore.Qt.CheckState.Checked if value else QtCore.Qt.CheckState.Unchecked
            if role == QtCore.Qt.ItemDataRole.ToolTipRole:
                return self._parameter_tooltip(row, index.column())
            if role in (None, QtCore.Qt.ItemDataRole.DisplayRole, QtCore.Qt.ItemDataRole.EditRole):
                if key in {"value", "minimum", "maximum", "stderr"}:
                    return _format_number(value)
                return value
            return None

        def flags(self, index: Any) -> Any:  # noqa: N802 - Qt API
            if not index.isValid():
                return QtCore.Qt.ItemFlag.NoItemFlags
            flags = QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable
            row = self._rows[index.row()]
            key = self.COLUMNS[index.column()][0]
            if key in {"minimum", "maximum", "vary", "expression"}:
                flags |= QtCore.Qt.ItemFlag.ItemIsEditable
            if key == "value" and not row.expression:
                flags |= QtCore.Qt.ItemFlag.ItemIsEditable
            if key == "vary":
                flags |= QtCore.Qt.ItemFlag.ItemIsUserCheckable
            return flags

        def _set_row_value(self, row: ParameterRow, key: str, value: Any) -> Any:
            if key in {"value", "minimum", "maximum", "stderr"}:
                # Keep a text value during editing only when it cannot be
                # parsed; the core can then report a useful validation error.
                parsed = _number(value)
                return parsed if parsed is not None else (None if value == "" else value)
            if key == "vary":
                if isinstance(value, bool):
                    return value
                return bool(value in (True, 1, "1", "true", "True", "yes", "Yes"))
            return "" if value is None else str(value)

        def setData(self, index: Any, value: Any, role: Any = None) -> bool:  # noqa: N802
            if not index.isValid() or not (0 <= index.row() < len(self._rows)):
                return False
            key = self.COLUMNS[index.column()][0]
            if key == "vary" and role == QtCore.Qt.ItemDataRole.CheckStateRole:
                value = value == QtCore.Qt.CheckState.Checked
            elif role not in (None, QtCore.Qt.ItemDataRole.EditRole, QtCore.Qt.ItemDataRole.CheckStateRole):
                return False
            row = self._rows[index.row()]
            new_value = self._set_row_value(row, key, value)
            if getattr(row, key) == new_value:
                return False
            setattr(row, key, new_value)
            self.dataChanged.emit(index, index, [QtCore.Qt.ItemDataRole.DisplayRole, role])
            if key == "vary":
                self.dataChanged.emit(index, index, [QtCore.Qt.ItemDataRole.CheckStateRole])
            self.parameterChanged.emit(row.name, key, new_value)
            self.parametersChanged.emit()
            return True

        def set_rows(self, parameters: Any) -> None:
            rows = coerce_parameter_rows(parameters)
            self.beginResetModel()
            self._rows = rows
            self.endResetModel()
            self.parametersChanged.emit()

        def set_parameters(self, parameters: Any) -> None:
            self.set_rows(parameters)

        def parameter_values(self) -> dict[str, Any]:
            values = {row.name: row.value for row in self._rows}
            # Keep derived degree readouts synchronized with the editable
            # radians parameter when a core ParameterSet is supplied.
            for row in self._rows:
                if row.name.endswith("_deg") and row.expression:
                    base = row.name[: -len("_deg")]
                    if base in values:
                        try:
                            values[row.name] = math.degrees(float(values[base]))
                        except (TypeError, ValueError):
                            pass
            return values

        def parameter_dict(self) -> dict[str, dict[str, Any]]:
            values = self.parameter_values()
            result = {row.name: row.to_dict() for row in self._rows}
            for name, value in values.items():
                result[name]["value"] = value
            return result

        def to_dict(self) -> dict[str, dict[str, Any]]:
            return self.parameter_dict()

        def set_parameter(self, name: str, value: Any) -> bool:
            for row_index, row in enumerate(self._rows):
                if row.name == name:
                    if row.expression and name.endswith("_deg"):
                        base = name[: -len("_deg")]
                        for base_index, base_row in enumerate(self._rows):
                            if base_row.name == base:
                                return self.setData(
                                    self.index(base_index, 1),
                                    math.radians(float(value)),
                                    QtCore.Qt.ItemDataRole.EditRole,
                                )
                        return False
                    index = self.index(row_index, 1)
                    return self.setData(index, value, QtCore.Qt.ItemDataRole.EditRole)
            return False


else:

    class ParameterTableModel:
        """Qt-free fallback with the same data API as the Qt model.

        It is intentionally small: it allows scripts and the core package to
        inspect/edit parameter state without making PySide6 a hard dependency.
        """

        COLUMNS = (
            ("name", "Parameter"),
            ("value", "Value"),
            ("minimum", "Min"),
            ("maximum", "Max"),
            ("vary", "Vary"),
            ("expression", "Expr"),
            ("unit", "Unit"),
            ("stderr", "Stderr"),
        )
        COLUMN_KEYS = tuple(key for key, _ in COLUMNS)
        HEADER_LABELS = tuple(label for _, label in COLUMNS)
        COLUMN_WIDTHS = (128, 82, 72, 72, 58, 112, 70, 76)
        headers = HEADER_LABELS

        def __init__(
            self,
            parameters: Any = None,
            parent: Any = None,
            *,
            language: str = "en",
        ) -> None:
            del parent
            self._language = validate_language(language)
            self._rows = coerce_parameter_rows(parameters)
            self._callbacks: list[Callable[[str, str, Any], None]] = []

        @property
        def language(self) -> str:
            return self._language

        def set_language(self, language: str) -> None:
            self._language = validate_language(language)

        @property
        def rows(self) -> list[ParameterRow]:
            return self._rows

        def rowCount(self, parent: Any = None) -> int:  # noqa: N802
            del parent
            return len(self._rows)

        def columnCount(self, parent: Any = None) -> int:  # noqa: N802
            del parent
            return len(self.COLUMNS)

        def connect_parameter_changed(self, callback: Callable[[str, str, Any], None]) -> None:
            self._callbacks.append(callback)

        def set_rows(self, parameters: Any) -> None:
            self._rows = coerce_parameter_rows(parameters)

        set_parameters = set_rows

        def parameter_values(self) -> dict[str, Any]:
            values = {row.name: row.value for row in self._rows}
            for row in self._rows:
                if row.name.endswith("_deg") and row.expression:
                    base = row.name[: -len("_deg")]
                    if base in values:
                        try:
                            values[row.name] = math.degrees(float(values[base]))
                        except (TypeError, ValueError):
                            pass
            return values

        def parameter_dict(self) -> dict[str, dict[str, Any]]:
            values = self.parameter_values()
            result = {row.name: row.to_dict() for row in self._rows}
            for name, value in values.items():
                result[name]["value"] = value
            return result

        to_dict = parameter_dict

        def set_parameter(self, name: str, value: Any) -> bool:
            for row in self._rows:
                if row.name == name:
                    if row.expression and name.endswith("_deg"):
                        base = name[: -len("_deg")]
                        return self.set_parameter(base, math.radians(float(value)))
                    row.value = _number(value, value)
                    for callback in self._callbacks:
                        callback(name, "value", row.value)
                    return True
            return False


__all__ = ["ParameterRow", "ParameterTableModel", "coerce_parameter_rows"]
