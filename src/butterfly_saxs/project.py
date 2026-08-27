"""TOML project configuration for LamellarSAXS2D.

The project file is deliberately small.  It describes inputs, geometry, the
analysis switches, and the export location; scientific implementations remain
in the analysis modules.  ``ProjectConfig`` accepts a few historical aliases
(``inputs``, ``poni``, and ``full2d``) so a project file can be used by both the
CLI and a future GUI without a migration step.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import copy
import os
import tomllib


class ProjectConfigError(ValueError):
    """Raised when a TOML project file cannot be represented safely."""


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, os.PathLike)):
        return [os.fspath(value)]
    if isinstance(value, (list, tuple)):
        return [os.fspath(item) for item in value]
    raise ProjectConfigError("输入文件必须是字符串或字符串列表")


def _copy_mapping(value: Any, *, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ProjectConfigError(f"{name} 必须是 TOML 表")
    return copy.deepcopy(dict(value))


@dataclass(init=False)
class ProjectConfig:
    """A serialisable description of one LamellarSAXS2D analysis project.

    Paths are intentionally kept as user-written strings.  A caller can use
    :meth:`resolve_paths` when it wants paths relative to the TOML file.  This
    makes load/save round trips lossless and keeps exported provenance useful.
    """

    input_paths: list[str]
    poni_path: str | None
    output_dir: str
    q_unit: str
    full2d: bool
    analysis: dict[str, Any]
    export: dict[str, Any]
    metadata: dict[str, Any]

    def __init__(
        self,
        input_paths: Any = None,
        poni_path: str | os.PathLike[str] | None = None,
        output_dir: str | os.PathLike[str] = "results",
        q_unit: str = "1/nm",
        full2d: bool = False,
        analysis: Mapping[str, Any] | None = None,
        export: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        *,
        # User-facing aliases.  They are accepted in code as well as in TOML.
        inputs: Any = None,
        poni: str | os.PathLike[str] | None = None,
        do_full2d: bool | None = None,
        input: Any = None,
        output: str | os.PathLike[str] | None = None,
        **extra: Any,
    ) -> None:
        if input_paths is None:
            input_paths = inputs if inputs is not None else input
        if poni_path is None:
            poni_path = poni
        if output is not None and output_dir == "results":
            output_dir = output
        if do_full2d is not None:
            full2d = bool(do_full2d)

        # Unknown top-level fields are retained as metadata.  This is useful
        # for beamline-specific knobs and avoids silently dropping provenance.
        metadata_values = _copy_mapping(metadata, name="metadata")
        metadata_values.update(extra)

        self.input_paths = _as_list(input_paths)
        self.poni_path = None if poni_path is None else os.fspath(poni_path)
        self.output_dir = os.fspath(output_dir)
        self.q_unit = str(q_unit)
        self.full2d = bool(full2d)
        self.analysis = _copy_mapping(analysis, name="analysis")
        self.export = _copy_mapping(export, name="export")
        self.metadata = metadata_values

    @property
    def inputs(self) -> list[str]:
        """Alias used by the short CLI vocabulary."""

        return self.input_paths

    @inputs.setter
    def inputs(self, value: Any) -> None:
        self.input_paths = _as_list(value)

    @property
    def poni(self) -> str | None:
        return self.poni_path

    @poni.setter
    def poni(self, value: str | os.PathLike[str] | None) -> None:
        self.poni_path = None if value is None else os.fspath(value)

    @property
    def output(self) -> str:
        return self.output_dir

    @output.setter
    def output(self, value: str | os.PathLike[str]) -> None:
        self.output_dir = os.fspath(value)

    @property
    def do_full2d(self) -> bool:
        return self.full2d

    @do_full2d.setter
    def do_full2d(self, value: bool) -> None:
        self.full2d = bool(value)

    # Frequently used analysis controls are exposed as read-only convenience
    # properties while the lossless mapping remains in ``analysis``.
    def analysis_value(self, name: str, default: Any = None) -> Any:
        return self.analysis.get(name, default)

    @property
    def q_window(self) -> Any:
        return self.analysis.get("q_window", self.analysis.get("q_range"))

    @property
    def mask(self) -> Any:
        return self.analysis.get("mask")

    @property
    def valid_mask(self) -> Any:
        return self.analysis.get("valid_mask")

    @property
    def ridge_method(self) -> str:
        return str(self.analysis.get("ridge_method", "radial_peak"))

    @property
    def n_angles(self) -> int:
        try:
            return int(self.analysis.get("n_angles", self.analysis.get("n_ridge_angles", 72)))
        except (TypeError, ValueError):
            return 72

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ProjectConfig":
        """Create a config from either a flat mapping or common TOML groups."""

        if not isinstance(data, Mapping):
            raise ProjectConfigError("项目配置根节点必须是 TOML 表")

        root = dict(data)
        # Accept [project] as the canonical group while allowing [inputs],
        # [output], and [analysis] to sit beside it.
        project = _copy_mapping(root.pop("project", None), name="project")
        inputs_group = _copy_mapping(root.pop("inputs", None), name="inputs")
        input_group = _copy_mapping(root.pop("input", None), name="input")
        output_group = _copy_mapping(root.pop("output", None), name="output")
        analysis_group = _copy_mapping(root.pop("analysis", None), name="analysis")
        export_group = _copy_mapping(root.pop("export", None), name="export")
        metadata_group = _copy_mapping(root.pop("metadata", None), name="metadata")

        # Merge in increasing precedence: project defaults, dedicated groups,
        # then flat keys.  Flat keys are convenient for tiny programmatic TOML.
        merged: dict[str, Any] = {}
        merged.update(project)
        for group in (inputs_group, input_group, output_group, analysis_group, export_group):
            merged.update(group)
        merged.update(root)

        input_value = merged.pop("input_paths", None)
        if input_value is None:
            input_value = merged.pop("inputs", None)
        if input_value is None:
            input_value = merged.pop("input", None)
        # [inputs] commonly uses ``files``; [input] commonly uses ``path``.
        if input_value is None:
            input_value = inputs_group.get("files", input_group.get("path"))

        poni_value = merged.pop("poni_path", None)
        if poni_value is None:
            poni_value = merged.pop("poni", None)
        if poni_value is None:
            poni_value = inputs_group.get("poni", input_group.get("poni"))

        output_value = merged.pop("output_dir", None)
        if output_value is None:
            output_value = merged.pop("output", None)
        if output_value is None:
            output_value = output_group.get("directory", output_group.get("dir", "results"))

        # These group-local keys have already been consumed above.  Do not
        # leak them into metadata merely because all groups share one merged
        # convenience mapping during parsing.
        for consumed_key in ("files", "path", "directory", "dir", "json", "npz"):
            merged.pop(consumed_key, None)

        q_unit = merged.pop("q_unit", analysis_group.get("q_unit", "1/nm"))
        full2d = merged.pop("full2d", merged.pop("do_full2d", analysis_group.get("full2d", False)))

        analysis = _copy_mapping(data.get("analysis"), name="analysis")
        # Beamline files often put frame selectors and detector masks beside
        # the input path.  Promote them into the same analysis namespace used
        # by CLI/pipeline while retaining the original values on round-trip.
        for key in (
            "frame",
            "dataset",
            "mask",
            "valid_mask",
            "mask_frame",
            "mask_dataset",
        ):
            if key in analysis:
                continue
            source_group = input_group if key in input_group else inputs_group
            if key in source_group:
                analysis[key] = copy.deepcopy(source_group[key])
        # Flat project knobs not consumed above are retained in analysis when
        # explicitly supplied by the [project] group.
        for key in (
            "q_range", "q_window", "q_scale", "center", "mask", "valid_mask",
            "frame", "dataset", "ridge", "ridge_method", "ridge_bins", "n_angles",
            "n_ridge_angles", "n_angular_bins", "n_radial_bins", "curvature",
            "curvature_sigma", "curvature_percentile", "curvature_normal_step", "ellipse",
        ):
            if key in merged:
                analysis[key] = merged.pop(key)

        export = _copy_mapping(data.get("export"), name="export")
        metadata = _copy_mapping(data.get("metadata"), name="metadata")
        metadata.update(metadata_group)
        # Preserve a project name/description and unknown values, rather than
        # throwing away user-authored provenance.
        for key in ("name", "description", "sample", "beamline"):
            if key in merged:
                metadata[key] = merged.pop(key)
        metadata.update(merged)

        return cls(
            input_paths=input_value,
            poni_path=poni_value,
            output_dir=output_value,
            q_unit=q_unit,
            full2d=bool(full2d),
            analysis=analysis,
            export=export,
            metadata=metadata,
        )

    @classmethod
    def from_toml(cls, path: str | os.PathLike[str]) -> "ProjectConfig":
        return load_project(path)

    def to_mapping(self) -> dict[str, Any]:
        """Return a TOML-friendly canonical mapping."""

        return {
            "project": {"q_unit": self.q_unit, "full2d": self.full2d},
            "inputs": {
                "files": list(self.input_paths),
                **({"poni": self.poni_path} if self.poni_path is not None else {}),
            },
            "output": {"directory": self.output_dir},
            "analysis": copy.deepcopy(self.analysis),
            "export": copy.deepcopy(self.export),
            "metadata": copy.deepcopy(self.metadata),
        }

    def resolve_paths(self, base_dir: str | os.PathLike[str]) -> "ProjectConfig":
        """Return a copy with relative input/poni/output paths resolved."""

        base = Path(base_dir)

        def resolve(value: str | None) -> str | None:
            if value is None:
                return None
            path = Path(value)
            return os.fspath(path if path.is_absolute() else base / path)

        analysis = copy.deepcopy(self.analysis)
        for key in ("mask", "valid_mask", "manifest", "checkpoint", "sigma", "weights"):
            value = analysis.get(key)
            if isinstance(value, (str, os.PathLike)):
                analysis[key] = resolve(os.fspath(value))
            elif isinstance(value, (list, tuple)):
                analysis[key] = [
                    resolve(os.fspath(item)) if isinstance(item, (str, os.PathLike)) else item
                    for item in value
                ]

        return ProjectConfig(
            input_paths=[resolve(item) or item for item in self.input_paths],
            poni_path=resolve(self.poni_path),
            output_dir=resolve(self.output_dir) or self.output_dir,
            q_unit=self.q_unit,
            full2d=self.full2d,
            analysis=analysis,
            export=copy.deepcopy(self.export),
            metadata=copy.deepcopy(self.metadata),
        )


def load_project(path: str | os.PathLike[str]) -> ProjectConfig:
    """Load a TOML project file with a concise Chinese error on failure."""

    source = Path(path)
    try:
        with source.open("rb") as handle:
            data = tomllib.load(handle)
    except FileNotFoundError as exc:
        raise ProjectConfigError(f"找不到项目配置：{source}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ProjectConfigError(f"项目配置 TOML 无法解析：{source}（{exc}）") from exc
    except OSError as exc:
        raise ProjectConfigError(f"无法读取项目配置：{source}（{exc}）") from exc
    return ProjectConfig.from_mapping(data)


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        return f'"{escaped}"'
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    raise ProjectConfigError(f"不支持写入 TOML 的值类型：{type(value).__name__}")


def _emit_table(lines: list[str], name: str, values: Mapping[str, Any]) -> None:
    scalars: list[tuple[str, Any]] = []
    nested: list[tuple[str, Mapping[str, Any]]] = []
    for key, value in values.items():
        if not isinstance(key, str) or not key or any(char in key for char in " \t\r\n."):
            raise ProjectConfigError(f"TOML 键名无效：{key!r}")
        if isinstance(value, Mapping):
            nested.append((key, value))
        elif value is not None:
            scalars.append((key, value))
    if scalars:
        lines.append(f"[{name}]")
        for key, value in scalars:
            lines.append(f"{key} = {_toml_value(value)}")
        lines.append("")
    for key, value in nested:
        _emit_table(lines, f"{name}.{key}", value)


def save_project(
    config: ProjectConfig | Mapping[str, Any],
    path: str | os.PathLike[str],
    *,
    force: bool = False,
) -> Path:
    """Write a project TOML file; existing files require ``force=True``."""

    destination = Path(path)
    if destination.exists() and not force:
        raise FileExistsError(f"输出已存在，未覆盖：{destination}（需要 --force）")
    if not isinstance(config, ProjectConfig):
        config = ProjectConfig.from_mapping(config)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = ["# LamellarSAXS2D project configuration", ""]
    for name, values in config.to_mapping().items():
        if values:
            _emit_table(lines, name, values)
    text = "\n".join(lines).rstrip() + "\n"
    try:
        destination.write_text(text, encoding="utf-8", newline="\n")
    except OSError as exc:
        raise ProjectConfigError(f"无法写入项目配置：{destination}（{exc}）") from exc
    return destination


def write_project(
    config: ProjectConfig | Mapping[str, Any],
    path: str | os.PathLike[str],
    *,
    force: bool = False,
) -> Path:
    """Compatibility alias for :func:`save_project`."""

    return save_project(config, path, force=force)


def load_project_config(path: str | os.PathLike[str]) -> ProjectConfig:
    """Compatibility alias for callers that spell out ``config``."""

    return load_project(path)


def save_project_config(
    config: ProjectConfig | Mapping[str, Any],
    path: str | os.PathLike[str],
    *,
    force: bool = False,
) -> Path:
    return save_project(config, path, force=force)


__all__ = [
    "ProjectConfig",
    "ProjectConfigError",
    "load_project",
    "save_project",
    "write_project",
    "load_project_config",
    "save_project_config",
]
