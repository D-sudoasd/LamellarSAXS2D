"""Parameter definitions used by the butterfly SAXS refinement models.

The parameter layer is deliberately small and dependency free.  It provides
the pieces that a GUI and a batch runner both need: editable values, bounds,
fixed/free state and safely evaluated tied expressions.  Expressions are
parsed with :mod:`ast`; Python evaluation is never used, so a parameter file
cannot execute arbitrary code.
"""

from __future__ import annotations

import ast
import math
import operator
import re
from collections.abc import Iterator, Mapping, MutableMapping
from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np

_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_EPS = float(np.finfo(float).eps)

# A deliberately small expression language.  These names are useful in a
# parameter editor, while attribute access, comprehensions and imports are
# intentionally not accepted.
_CONSTANTS: dict[str, float] = {"pi": math.pi, "e": math.e, "tau": math.tau}
_FUNCTIONS: dict[str, Any] = {
    "abs": abs,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "exp": math.exp,
    "log": math.log,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "floor": math.floor,
    "ceil": math.ceil,
}

_BINOPS: dict[type[ast.operator], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
_UNARYOPS: dict[type[ast.unaryop], Any] = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _as_finite_float(value: Any, field: str, *, allow_infinite: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{field} must be a real number")
    result = float(value)
    if not allow_infinite and not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    if math.isnan(result):
        raise ValueError(f"{field} cannot be NaN")
    return result


def _validate_identifier(name: str) -> str:
    if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
        raise ValueError(f"invalid parameter name: {name!r}")
    return name


def _expression_tree(expr: str) -> ast.Expression:
    if not isinstance(expr, str) or not expr.strip():
        raise ValueError("expr must be a non-empty string")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"invalid parameter expression {expr!r}: {exc.msg}") from exc
    return tree


def _expression_names(expr: str) -> set[str]:
    tree = _expression_tree(expr)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
    return names - set(_CONSTANTS) - set(_FUNCTIONS)


def _evaluate_tree(node: ast.AST, values: Mapping[str, float]) -> float:
    """Evaluate an already validated expression AST.

    Keeping this interpreter explicit makes the accepted syntax auditable and
    prevents accidental expansion of the expression language when Python
    changes its AST implementation.
    """

    if isinstance(node, ast.Expression):
        return _evaluate_tree(node.body, values)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (bool, np.bool_)) or not isinstance(node.value, Real):
            raise ValueError("parameter expressions only support numeric constants")
        return _as_finite_float(node.value, "expression constant")
    if isinstance(node, ast.Name):
        if node.id in values:
            return float(values[node.id])
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        raise ValueError(f"unknown variable in parameter expression: {node.id}")
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARYOPS:
        return float(_UNARYOPS[type(node.op)](_evaluate_tree(node.operand, values)))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        left = _evaluate_tree(node.left, values)
        right = _evaluate_tree(node.right, values)
        try:
            result = _BINOPS[type(node.op)](left, right)
        except (ArithmeticError, ValueError, OverflowError) as exc:
            raise ValueError(f"could not evaluate parameter expression: {exc}") from exc
        return _as_finite_float(result, "expression result")
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCTIONS:
        if node.keywords:
            raise ValueError("keyword arguments are not allowed in parameter expressions")
        args = [_evaluate_tree(arg, values) for arg in node.args]
        try:
            result = _FUNCTIONS[node.func.id](*args)
        except (ArithmeticError, ValueError, OverflowError) as exc:
            raise ValueError(f"could not evaluate parameter expression: {exc}") from exc
        return _as_finite_float(result, "expression result")
    raise ValueError(f"unsupported syntax in parameter expression: {ast.dump(node, include_attributes=False)}")


@dataclass
class ParameterSpec:
    """Description of one fit parameter.

    ``expr`` makes a parameter tied/derived.  Tied parameters are never part
    of the free optimization vector, even if a caller accidentally supplies
    ``vary=True``.  ``min`` and ``max`` are inclusive bounds.
    """

    value: float = 0.0
    min: float | None = None
    max: float | None = None
    vary: bool = True
    expr: str | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        if self.name is not None:
            self.name = _validate_identifier(self.name)
        if self.min is not None:
            self.min = _as_finite_float(self.min, "min", allow_infinite=True)
        if self.max is not None:
            self.max = _as_finite_float(self.max, "max", allow_infinite=True)
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError("min must not exceed max")
        if not isinstance(self.vary, (bool, np.bool_)):
            raise TypeError("vary must be a boolean")
        self.vary = bool(self.vary)
        if self.expr is not None:
            _expression_tree(self.expr)
            # Tied parameters are derived, not independently optimized.
            self.vary = False
        else:
            self.value = _as_finite_float(self.value, "value")
            self._validate_value(self.value)
        if self.name in {"axis_ratio", "ratio"}:
            if self.min is not None and self.min <= 0:
                raise ValueError("axis_ratio must be greater than zero")
            if self.max is not None and self.max > 1:
                raise ValueError("axis_ratio must not exceed one")
            if self.expr is None and not (0 < self.value <= 1):
                raise ValueError("axis_ratio must be in (0, 1]")

    @property
    def is_tied(self) -> bool:
        return self.expr is not None

    @property
    def lower(self) -> float | None:
        """UI-friendly alias for the inclusive lower bound."""

        return self.min

    @property
    def upper(self) -> float | None:
        """UI-friendly alias for the inclusive upper bound."""

        return self.max

    @property
    def bounds(self) -> tuple[float | None, float | None]:
        """Compatibility view used by generic fit engines."""

        return self.min, self.max

    @property
    def is_fixed(self) -> bool:
        return self.expr is None and not self.vary

    def _validate_value(self, value: float) -> None:
        if self.min is not None and value < self.min:
            raise ValueError(f"value {value} is below min {self.min}")
        if self.max is not None and value > self.max:
            raise ValueError(f"value {value} exceeds max {self.max}")

    def set_value(self, value: float) -> "ParameterSpec":
        if self.expr is not None:
            raise ValueError(f"cannot set the value of tied parameter {self.name or '<unnamed>'}")
        value = _as_finite_float(value, "value")
        self._validate_value(value)
        if self.name in {"axis_ratio", "ratio"} and not (0 < value <= 1):
            raise ValueError("axis_ratio must be in (0, 1]")
        self.value = value
        return self

    def copy(self, **changes: Any) -> "ParameterSpec":
        data = {
            "value": self.value,
            "min": self.min,
            "max": self.max,
            "vary": self.vary,
            "expr": self.expr,
            "name": self.name,
        }
        data.update(changes)
        return ParameterSpec(**data)


class ParameterSet(MutableMapping[str, ParameterSpec]):
    """Ordered collection of :class:`ParameterSpec` objects.

    A mapping can contain ``ParameterSpec`` objects, plain numbers, or small
    mappings such as ``{"value": 1, "min": 0, "vary": False}``.  The
    constructor validates expression references and cycles immediately so a
    malformed UI/batch configuration fails before fitting starts.
    """

    def __init__(self, parameters: Mapping[str, Any] | "ParameterSet" | None = None, **kwargs: Any) -> None:
        self._parameters: dict[str, ParameterSpec] = {}
        self._suspend_checks = True
        try:
            if parameters is not None:
                if isinstance(parameters, ParameterSet):
                    parameters = parameters._parameters
                if not isinstance(parameters, Mapping):
                    raise TypeError("parameters must be a mapping or ParameterSet")
                for name, spec in parameters.items():
                    self[name] = spec
            for name, spec in kwargs.items():
                self[name] = spec
        finally:
            self._suspend_checks = False
        self._check_dependencies()

    @staticmethod
    def _coerce_spec(name: str, spec: Any) -> ParameterSpec:
        name = _validate_identifier(name)
        if isinstance(spec, ParameterSpec):
            result = spec.copy(name=name)
        elif isinstance(spec, Mapping):
            data = dict(spec)
            data.setdefault("name", name)
            if "value" not in data and "expr" not in data:
                raise ValueError(f"parameter {name!r} needs a value or expr")
            result = ParameterSpec(**data)
        elif isinstance(spec, Real) and not isinstance(spec, (bool, np.bool_)):
            result = ParameterSpec(value=float(spec), name=name)
        else:
            raise TypeError(f"invalid specification for parameter {name!r}")
        return result

    def __getitem__(self, name: str) -> ParameterSpec:
        return self._parameters[name]

    def __setitem__(self, name: str, spec: Any) -> None:
        name = _validate_identifier(name)
        old = self._parameters.get(name)
        self._parameters[name] = self._coerce_spec(name, spec)
        # Dependency checks are repeated after every update so an interactive
        # editor cannot hold an invalid graph between operations.
        if not getattr(self, "_suspend_checks", False):
            try:
                self._check_dependencies()
            except Exception:
                if old is None:
                    del self._parameters[name]
                else:
                    self._parameters[name] = old
                raise

    def __delitem__(self, name: str) -> None:
        del self._parameters[name]
        self._check_dependencies()

    def __iter__(self) -> Iterator[str]:
        return iter(self._parameters)

    def __len__(self) -> int:
        return len(self._parameters)

    def __repr__(self) -> str:
        return f"ParameterSet({self._parameters!r})"

    def items(self):  # type: ignore[override]
        """Return resolved numeric values for UI/batch model adapters.

        Individual editable specifications remain available through
        ``params[name]`` and ``params.specs``.  Returning numeric items keeps
        this object interoperable with model code that treats a parameter set
        as an ordinary value mapping (notably the intensity refinement path).
        """

        values = self.resolve()
        return ((name, values[name]) for name in self._parameters)

    def spec_items(self):
        """Iterate ``(name, ParameterSpec)`` pairs for editors and validators."""

        return self._parameters.items()

    @property
    def specs(self) -> dict[str, ParameterSpec]:
        return dict(self._parameters)

    @property
    def fixed(self) -> dict[str, bool]:
        return {name: spec.is_fixed for name, spec in self._parameters.items()}

    def copy(self) -> "ParameterSet":
        return ParameterSet(self)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._parameters)

    @property
    def free_names(self) -> tuple[str, ...]:
        return tuple(name for name, spec in self._parameters.items() if spec.expr is None and spec.vary)

    @property
    def fixed_names(self) -> tuple[str, ...]:
        return tuple(name for name, spec in self._parameters.items() if spec.is_fixed)

    @property
    def tied_names(self) -> tuple[str, ...]:
        return tuple(name for name, spec in self._parameters.items() if spec.is_tied)

    @property
    def varying_names(self) -> tuple[str, ...]:
        return self.free_names

    def dependency_graph(self) -> dict[str, frozenset[str]]:
        graph: dict[str, frozenset[str]] = {}
        available = set(self._parameters)
        for name, spec in self._parameters.items():
            deps = _expression_names(spec.expr) if spec.expr is not None else set()
            unknown = deps - available
            if unknown:
                unknown_text = ", ".join(sorted(unknown))
                raise ValueError(f"parameter {name!r} references unknown variable(s): {unknown_text}")
            graph[name] = frozenset(deps)
        return graph

    def topological_order(self) -> tuple[str, ...]:
        graph = self.dependency_graph()
        state: dict[str, int] = {}
        order: list[str] = []
        stack: list[str] = []

        def visit(name: str) -> None:
            status = state.get(name, 0)
            if status == 1:
                start = stack.index(name) if name in stack else 0
                cycle = " -> ".join(stack[start:] + [name])
                raise ValueError(f"cyclic parameter expression: {cycle}")
            if status == 2:
                return
            state[name] = 1
            stack.append(name)
            for dep in graph[name]:
                visit(dep)
            stack.pop()
            state[name] = 2
            order.append(name)

        for name in self._parameters:
            visit(name)
        return tuple(order)

    def _check_dependencies(self) -> None:
        # Empty sets are valid and useful while a UI builds a configuration.
        if self._parameters:
            self.topological_order()

    def resolve(self, *, check_bounds: bool = True) -> dict[str, float]:
        values: dict[str, float] = {}
        for name in self.topological_order():
            spec = self._parameters[name]
            if spec.expr is None:
                value = spec.value
            else:
                tree = _expression_tree(spec.expr)
                value = _evaluate_tree(tree, values)
            value = _as_finite_float(value, f"resolved value for {name}")
            if check_bounds:
                spec._validate_value(value)
                if name in {"axis_ratio", "ratio"} and not (0 < value <= 1):
                    raise ValueError("axis_ratio must be in (0, 1]")
            values[name] = value
        return values

    evaluate = resolve
    valuesdict = resolve

    def update_values(self, values: Mapping[str, Any], *, allow_tied: bool = False) -> "ParameterSet":
        unknown = set(values) - set(self._parameters)
        if unknown:
            raise KeyError(f"unknown parameter(s): {', '.join(sorted(unknown))}")
        for name, value in values.items():
            spec = self._parameters[name]
            if spec.is_tied and not allow_tied:
                raise ValueError(f"cannot directly set tied parameter {name!r}")
            if spec.is_tied:
                raise ValueError(f"tied parameter {name!r} is defined by expr")
            spec.set_value(value)
        self.resolve()
        return self

    set_values = update_values

    def free_vector(self) -> np.ndarray:
        return np.asarray([self[name].value for name in self.free_names], dtype=float)

    def free_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lower = np.asarray(
            [-np.inf if self[name].min is None else self[name].min for name in self.free_names], dtype=float
        )
        upper = np.asarray(
            [np.inf if self[name].max is None else self[name].max for name in self.free_names], dtype=float
        )
        return lower, upper

    bounds = free_bounds

    def set_free_vector(self, vector: Any) -> "ParameterSet":
        vector = np.asarray(vector, dtype=float)
        if vector.ndim != 1 or vector.size != len(self.free_names):
            raise ValueError(f"expected {len(self.free_names)} free values, got shape {vector.shape}")
        for name, value in zip(self.free_names, vector):
            self[name].set_value(float(value))
        return self

    set_vector = set_free_vector

    def resolved_with_vector(self, vector: Any, *, check_bounds: bool = True) -> dict[str, float]:
        candidate = self.copy()
        candidate.set_free_vector(vector)
        return candidate.resolve(check_bounds=check_bounds)

    def as_dict(self, *, resolved: bool = False, include_specs: bool = False) -> dict[str, Any]:
        if resolved or not include_specs:
            return self.resolve()
        if include_specs:
            return {
                name: {
                    "value": spec.value,
                    "min": spec.min,
                    "max": spec.max,
                    "vary": spec.vary,
                    "expr": spec.expr,
                }
                for name, spec in self._parameters.items()
            }
        return {name: spec.value for name, spec in self._parameters.items()}

    to_dict = as_dict


def default_ellipse_parameters(*, center: tuple[float, float] = (0.0, 0.0), a: float = 1.0,
                               axis_ratio: float = 0.7, theta: float = 0.0) -> ParameterSet:
    """Return the canonical editable parameter set for one ellipse.

    The same set is also the shared parameterization of a symmetric pair:
    the pair differs only by using ``+theta`` and ``-theta``.  ``b`` is always
    tied to ``a*axis_ratio`` and is therefore not an independent fit degree
    of freedom.
    """

    cx, cy = center
    a = _as_finite_float(a, "a")
    ratio = _as_finite_float(axis_ratio, "axis_ratio")
    theta = _as_finite_float(theta, "theta")
    if a <= 0:
        raise ValueError("a must be greater than zero")
    if not 0 < ratio <= 1:
        raise ValueError("axis_ratio must be in (0, 1]")
    return ParameterSet(
        {
            "cx": ParameterSpec(cx, name="cx"),
            "cy": ParameterSpec(cy, name="cy"),
            "a": ParameterSpec(a, min=_EPS, name="a"),
            "axis_ratio": ParameterSpec(ratio, min=_EPS, max=1.0, name="axis_ratio"),
            "b": ParameterSpec(a * ratio, vary=False, expr="a*axis_ratio", name="b"),
            "theta": ParameterSpec(theta, min=-math.pi / 2, max=math.pi / 2, name="theta"),
            # The optimizer uses radians, while the UI and exported batch
            # tables get an unambiguous degree-valued readout.
            "theta_deg": ParameterSpec(0.0, vary=False, expr="theta*180/pi", name="theta_deg"),
        }
    )


# Common spelling variants used by UI code and notebooks.
default_ellipse_parameter_set = default_ellipse_parameters
ellipse_parameter_defaults = default_ellipse_parameters
default_ellipse_params = default_ellipse_parameters


__all__ = ["ParameterSpec", "ParameterSet", "default_ellipse_parameters", "default_ellipse_parameter_set",
           "default_ellipse_params", "ellipse_parameter_defaults"]
