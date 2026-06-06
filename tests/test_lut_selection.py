from pathlib import Path

from config import config_from_dict
from processing.lut import (
    available_luts,
    parse_lut_selection_input,
    resolve_lut_selection,
)


def test_list_lut_files_from_assets(tmp_path: Path) -> None:
    lut_dir = tmp_path / "assets" / "luts"
    lut_dir.mkdir(parents=True)
    (lut_dir / "one.cube").write_text("lut", encoding="utf-8")
    (lut_dir / "two.cube").write_text("lut", encoding="utf-8")
    (lut_dir / "skip.txt").write_text("x", encoding="utf-8")

    files = available_luts(tmp_path)
    assert [path.name for path in files] == ["one.cube", "two.cube"]


def test_resolve_selected_luts_warns_when_missing(tmp_path: Path, caplog) -> None:
    lut_dir = tmp_path / "assets" / "luts"
    lut_dir.mkdir(parents=True)
    (lut_dir / "present.cube").write_text("lut", encoding="utf-8")

    config = config_from_dict(
        {
            "color": {
                "lut_enabled": True,
                "max_luts": 3,
                "selected_luts": ["present.cube", "missing.cube"],
                "auto_select_luts": False,
            }
        }
    ).color

    result = resolve_lut_selection(tmp_path, config)
    assert [path.name for path in result.resolved_paths] == ["present.cube"]
    assert any("Không tìm thấy file LUT đã chọn" in record.message for record in caplog.records)


def test_parse_lut_selection_input() -> None:
    assert parse_lut_selection_input("1,3", available_count=4, max_luts=3) == [1, 3]
    assert parse_lut_selection_input("5", available_count=4, max_luts=3) == []
