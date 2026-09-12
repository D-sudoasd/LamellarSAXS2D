"""Shared, installed-font selection for scientific figure renderers."""
from __future__ import annotations

from pathlib import Path

from matplotlib import font_manager
from matplotlib.font_manager import FontProperties

_WINDOWS_CJK_FONT_PATHS = (
    Path(r"C:\Windows\Fonts\msyh.ttc"),
    Path(r"C:\Windows\Fonts\simhei.ttf"),
    Path(r"C:\Windows\Fonts\simsun.ttc"),
    Path(r"C:\Windows\Fonts\NotoSansCJK-Regular.ttc"),
)
_CJK_FONT_FAMILIES = (
    "Microsoft YaHei",
    "Microsoft YaHei UI",
    "SimHei",
    "SimSun",
    "Noto Sans CJK SC",
    "Noto Sans SC",
    "Noto Sans CJK TC",
    "Noto Sans TC",
    "Noto Sans CJK JP",
    "Noto Sans JP",
    "Noto Sans CJK HK",
    "Noto Sans HK",
    "Noto Sans CJK KR",
    "WenQuanYi Micro Hei",
    "WenQuanYi Zen Hei",
    "Hiragino Sans GB",
    "Hiragino Sans",
    "PingFang SC",
)
_LATIN_FONT_FAMILIES = ("Arial", "DejaVu Sans")


def _font_family_key(name: str) -> str:
    return " ".join(str(name).casefold().split())


def _installed_family(candidates: tuple[str, ...]) -> str | None:
    installed = {
        _font_family_key(entry.name): entry.name
        for entry in font_manager.fontManager.ttflist
    }
    return next(
        (
            installed[_font_family_key(candidate)]
            for candidate in candidates
            if _font_family_key(candidate) in installed
        ),
        None,
    )


def font_properties(
    size: float | None = 6.5,
    *,
    weight: str = "normal",
    language: str = "en",
) -> FontProperties:
    """Return properties for an installed CJK or Latin font, with its path."""
    is_cjk = str(language).lower().startswith("zh")
    if is_cjk and weight == "normal":
        for path in _WINDOWS_CJK_FONT_PATHS:
            if path.is_file():
                return FontProperties(fname=str(path), size=size, weight=weight)

    candidates = _CJK_FONT_FAMILIES if is_cjk else _LATIN_FONT_FAMILIES
    family = _installed_family(candidates)
    if family is None and is_cjk:
        family = _installed_family(_LATIN_FONT_FAMILIES)
    if family is None:
        properties = FontProperties(size=size, weight=weight)
        try:
            path = font_manager.findfont(properties, fallback_to_default=True)
        except (OSError, ValueError):  # pragma: no cover - defensive fallback
            return properties
        return FontProperties(fname=path, size=size, weight=weight)

    properties = FontProperties(family=[family], size=size, weight=weight)
    try:
        path = font_manager.findfont(properties, fallback_to_default=False)
    except (OSError, ValueError):  # pragma: no cover - defensive fallback
        return properties
    return FontProperties(family=[family], fname=path, size=size, weight=weight)
