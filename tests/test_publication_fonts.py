from __future__ import annotations

import pytest
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties

from butterfly_saxs import font_support
from butterfly_saxs.lamellar_render import _font as _lamellar_font
from butterfly_saxs.publication_render import _font as _publication_font


@pytest.fixture(autouse=True)
def _clear_font_manager_cache(monkeypatch):
    yield
    font_manager.fontManager._findfont_cached.cache_clear()


def _entry(name: str, path: str, weight: int) -> font_manager.FontEntry:
    return font_manager.FontEntry(
        fname=path,
        name=name,
        style="normal",
        variant="normal",
        weight=weight,
        stretch="normal",
        size="scalable",
    )


def _catalog(monkeypatch, entries: list[font_manager.FontEntry]) -> None:
    monkeypatch.setattr(font_support, "_WINDOWS_CJK_FONT_PATHS", ())
    monkeypatch.setattr(font_manager.fontManager, "ttflist", entries)
    font_manager.fontManager._findfont_cached.cache_clear()


def _font_entries(name: str, path: str) -> list[font_manager.FontEntry]:
    return [_entry(name, path, weight) for weight in (400, 700)]


def test_chinese_font_selects_installed_noto_jp_when_it_is_the_only_cjk_family(monkeypatch):
    path = font_manager.findfont(FontProperties(family=["DejaVu Sans"]))
    _catalog(monkeypatch, _font_entries("Noto Sans CJK JP", path))

    selected = _publication_font(size=8.0, weight="bold", language="zh_CN")

    assert selected.get_family() == ["Noto Sans CJK JP"]
    assert selected.get_file() == path
    assert selected.get_size_in_points() == 8.0
    assert selected.get_weight() == "bold"


def test_shared_chinese_font_choice_keeps_windows_yahei_preference(monkeypatch):
    path = font_manager.findfont(FontProperties(family=["DejaVu Sans"]))
    entries = _font_entries("Noto Sans CJK JP", path)
    entries += _font_entries("Microsoft YaHei", path)
    _catalog(monkeypatch, entries)

    publication_font = _publication_font(language="zh_CN")
    lamellar_font = _lamellar_font()

    assert publication_font.get_family() == ["Microsoft YaHei"]
    assert lamellar_font.get_family() == ["Microsoft YaHei"]
    assert publication_font.get_file() == lamellar_font.get_file() == path


def test_lamellar_font_keeps_matplotlib_default_size(monkeypatch):
    path = font_manager.findfont(FontProperties(family=["DejaVu Sans"]))
    _catalog(monkeypatch, _font_entries("Noto Sans CJK JP", path))

    selected = _lamellar_font()

    assert selected.get_size_in_points() == FontProperties().get_size_in_points()


def test_english_font_prefers_arial_then_installed_dejavu(monkeypatch):
    path = font_manager.findfont(FontProperties(family=["DejaVu Sans"]))
    _catalog(monkeypatch, _font_entries("DejaVu Sans", path) + _font_entries("Arial", path))

    assert _publication_font(language="en").get_family() == ["Arial"]

    _catalog(monkeypatch, _font_entries("DejaVu Sans", path))
    assert _publication_font(language="en").get_family() == ["DejaVu Sans"]
