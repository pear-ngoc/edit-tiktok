from pathlib import Path

from utils.files import safe_output_path, sanitize_filename


def test_sanitize_filename_removes_unsafe_chars() -> None:
    assert sanitize_filename('bad<>:"/\\|?* name.MP4') == "bad_ name.mp4"


def test_safe_output_path_preserves_subfolders_and_unique(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    source = input_root / "nested" / "bad:name.mov"
    source.parent.mkdir(parents=True)
    source.write_text("x")
    output_root = tmp_path / "output"
    first = safe_output_path(source, input_root, output_root)
    first.write_text("x")
    second = safe_output_path(source, input_root, output_root)
    assert first.parent == output_root / "nested"
    assert second.name == "bad_name_1.mp4"
