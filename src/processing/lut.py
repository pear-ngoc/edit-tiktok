from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from models import ColorConfig
from utils.files import list_lut_files

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class LutSelectionResult:
    available: list[Path]
    selected_names: list[str]
    resolved_paths: list[Path]
    applied: bool
    warnings: list[str]


def resolve_selected_luts(project_root: Path, color: ColorConfig) -> list[Path]:
    return resolve_lut_selection(project_root, color).resolved_paths


def resolve_lut_selection(project_root: Path, color: ColorConfig) -> LutSelectionResult:
    lut_dir = project_root / "assets" / "luts"
    available = list_lut_files(lut_dir)
    LOGGER.info("Số LUT khả dụng: %s", len(available))

    warnings: list[str] = []
    if not color.lut_enabled:
        LOGGER.info("LUT đã tắt trong cấu hình.")
        return LutSelectionResult(available, [], [], False, warnings)

    selected: list[Path] = []
    selected_names = list(color.selected_luts[: color.max_luts])

    if not selected_names:
        if color.auto_select_luts:
            selected_names = [path.name for path in available[: color.max_luts]]
            LOGGER.info("Tự động chọn LUT: %s", ", ".join(selected_names) or "không có")
        else:
            warning = "LUT is enabled but no LUT is selected. No LUT will be applied."
            LOGGER.warning(warning)
            warnings.append(warning)
            return LutSelectionResult(available, [], [], False, warnings)

    LOGGER.info("LUT đã chọn: %s", ", ".join(selected_names))
    for raw in selected_names:
        candidate = Path(raw)
        if not candidate.is_absolute():
            direct_candidate = project_root / candidate
            candidate = direct_candidate if direct_candidate.exists() else lut_dir / candidate
        if candidate.exists() and candidate.suffix.lower() == ".cube":
            selected.append(candidate)
            LOGGER.info("Đã phân giải LUT: %s", candidate)
        else:
            warning = f"Không tìm thấy file LUT đã chọn: {candidate}"
            LOGGER.warning(warning)
            warnings.append(warning)
    if not selected:
        warning = "Không có LUT hợp lệ nào được áp dụng."
        LOGGER.warning(warning)
        warnings.append(warning)
    return LutSelectionResult(available, selected_names, selected[: color.max_luts], bool(selected), warnings)


def parse_lut_selection_input(raw: str, available_count: int, max_luts: int) -> list[int]:
    text = raw.strip()
    if not text:
        return []
    normalized = text.lower()
    if normalized in {"0", "n", "no", "none", "no lut", "null"}:
        return []

    values: list[int] = []
    for chunk in text.split(","):
        item = chunk.strip()
        if not item:
            continue
        index = int(item)
        if index < 1 or index > available_count + 1:
            raise ValueError("Lựa chọn LUT không hợp lệ.")
        if index == available_count + 1:
            return []
        if index not in values:
            values.append(index)

    if len(values) > max_luts:
        raise ValueError(f"Chỉ được chọn tối đa {max_luts} LUT.")
    return values


def available_luts(project_root: Path) -> list[Path]:
    return list_lut_files(project_root / "assets" / "luts")
