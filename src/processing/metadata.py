from __future__ import annotations

from models import MetadataConfig


def metadata_args(config: MetadataConfig) -> list[str]:
    mode = config.mode.lower()
    if mode == "keep":
        return []
    if mode == "remove":
        return ["-map_metadata", "-1"]
    if mode == "custom":
        args = ["-map_metadata", "-1"]
        for key, value in config.custom.items():
            if value:
                args.extend(["-metadata", f"{key}={value}"])
        return args
    raise ValueError(f"Chế độ metadata không được hỗ trợ: {config.mode}")
