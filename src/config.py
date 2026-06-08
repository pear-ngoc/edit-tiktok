from __future__ import annotations

import ast
import json
import os
import shutil
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, TypeVar

from models import (
    AppConfig,
    AudioConfig,
    AmdAmfConfig,
    ColorConfig,
    CpuConfig,
    EncoderConfig,
    GroqTranscriptionConfig,
    MetadataConfig,
    FormattingConfig,
    LoggingConfig,
    ProcessingConfig,
    RuntimeConfig,
    QueueConfig,
    NvidiaConfig,
    RevidAPIConfig,
    StorageConfig,
    StorageGoogleDriveConfig,
    StorageTelegramConfig,
    TiktokdlFallbackConfig,
    VaapiConfig,
    TelegramConfig,
    SubtitlesConfig,
    VideotoolboxConfig,
    VideoConfig,
    CenterCropBlurConfig,
)

T = TypeVar("T")

LOGGER = logging.getLogger(__name__)


CONFIG_FILENAME = "config.yaml"
EXAMPLE_CONFIG_FILENAME = "config.example.yaml"
SUPPORTED_STORAGE_PROVIDERS = {"local", "telegram", "google_drive", "both"}


def default_config() -> AppConfig:
    return AppConfig()


def default_config_dict() -> dict[str, Any]:
    return asdict(default_config())


def ensure_config_file(project_root: Path) -> Path:
    config_path = project_root / CONFIG_FILENAME
    if config_path.exists():
        return config_path

    example_path = project_root / EXAMPLE_CONFIG_FILENAME
    if example_path.exists():
        shutil.copyfile(example_path, config_path)
    else:
        config_path.write_text(dump_simple_yaml(default_config_dict()), encoding="utf-8")
    return config_path


def load_config(project_root: Path, config_path: Path | None = None) -> AppConfig:
    path = config_path or ensure_config_file(project_root)
    load_env_file(project_root)
    raw = _load_raw_config(path)
    merged = deep_merge(default_config_dict(), raw)
    config = config_from_dict(merged)
    return apply_environment_overrides(config)


def save_config(project_root: Path, config: AppConfig, config_path: Path | None = None) -> Path:
    path = config_path or (project_root / CONFIG_FILENAME)
    path.write_text(dump_simple_yaml(asdict(config)), encoding="utf-8")
    return path


def save_default_config(path: Path) -> None:
    path.write_text(dump_simple_yaml(default_config_dict()), encoding="utf-8")


def _load_raw_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return {}

    if path.suffix.lower() == ".json":
        return json.loads(text)

    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text)
        return loaded or {}
    except ModuleNotFoundError:
        return parse_simple_yaml(text)


def config_from_dict(data: dict[str, Any]) -> AppConfig:
    storage = _storage_section(data.get("storage", {}))
    subtitles_raw = data.get("subtitles", {})
    groq_config = _section(GroqTranscriptionConfig, subtitles_raw.get("groq", {}), section_name="subtitles.groq")
    subtitles = SubtitlesConfig(
        **{k: v for k, v in subtitles_raw.items() if k != "groq"},
        groq=groq_config,
    )
    return AppConfig(
        processing=_section(ProcessingConfig, data.get("processing", {}), section_name="processing"),
        queue=_section(QueueConfig, data.get("queue", {}), section_name="queue"),
        video=_video_section(data.get("video", {})),
        color=_section(ColorConfig, data.get("color", {}), section_name="color"),
        audio=_section(AudioConfig, data.get("audio", {}), section_name="audio"),
        metadata=_section(MetadataConfig, data.get("metadata", {}), section_name="metadata"),
        encoder=_section(EncoderConfig, data.get("encoder", {}), section_name="encoder"),
        subtitles=subtitles,
        telegram=_section(TelegramConfig, data.get("telegram", {}), section_name="telegram"),
        storage=storage,
        revid_api=_section(RevidAPIConfig, data.get("revid_api", {}), section_name="revid_api"),
        tiktokdl_fallback=_section(TiktokdlFallbackConfig, data.get("tiktokdl_fallback", {}), section_name="tiktokdl_fallback"),
        formatting=_section(FormattingConfig, data.get("formatting", {}), section_name="formatting"),
        runtime=_section(RuntimeConfig, data.get("runtime", {}), section_name="runtime"),
        nvidia=_section(NvidiaConfig, data.get("nvidia", {}), section_name="nvidia"),
        amd_amf=_section(AmdAmfConfig, data.get("amd_amf", {}), section_name="amd_amf"),
        vaapi=_section(VaapiConfig, data.get("vaapi", {}), section_name="vaapi"),
        videotoolbox=_section(VideotoolboxConfig, data.get("videotoolbox", {}), section_name="videotoolbox"),
        cpu=_section(CpuConfig, data.get("cpu", {}), section_name="cpu"),
        logging=_section(LoggingConfig, data.get("logging", {}), section_name="logging"),
    )


def apply_overrides(config: AppConfig, overrides: dict[str, Any]) -> AppConfig:
    data = asdict(config)
    mapping = {
        "input": ("processing", "input_dir"),
        "output": ("processing", "output_dir"),
        "aspect": ("video", "aspect_ratio"),
        "mode": ("video", "mode"),
        "encoder": ("encoder", "backend"),
        "preset": ("encoder", "preset"),
        "workers": ("processing", "max_workers"),
        "speed": ("video", "speed"),
        "target_resolution": ("video", "target_resolution"),
        "lut": ("color", "selected_luts"),
        "subtitle_language": ("subtitles", "language"),
        "whisper_model": ("subtitles", "model_size"),
        "transcription_backend": ("subtitles", "backend"),
        "groq_transcription_model": ("subtitles", "groq", "model"),
        "groq_fallback_local": ("subtitles", "groq", "fallback_to_local"),
        "caption_max_chars_per_line": ("formatting", "max_chars_per_line"),
        "caption_max_lines": ("formatting", "max_lines"),
        "caption_max_words": ("formatting", "max_words_per_cue"),
        "caption_max_duration": ("formatting", "max_duration"),
        "caption_position": ("formatting", "caption_position"),
        "caption_font_size": ("formatting", "caption_font_size"),
        "debug_ffmpeg": ("processing", "debug_ffmpeg"),
        "log_level": ("logging", "level"),
        "console_level": ("logging", "console_level"),
        "file_level": ("logging", "file_level"),
        "debug_ffmpeg_commands": ("logging", "debug_ffmpeg_commands"),
        "ffmpeg_stderr_tail_lines": ("logging", "ffmpeg_stderr_tail_lines"),
        "per_job_logs": ("logging", "per_job_logs"),
        "retain_failed_temp": ("logging", "retain_failed_temp"),
        "progress_interval_seconds": ("logging", "progress_interval_seconds"),
        "show_runtime_plan": ("logging", "show_runtime_plan"),
        "storage": ("storage", "provider"),
        "telegram_chat_id": ("storage", "telegram", "default_chat_id"),
        "drive_folder_id": ("storage", "google_drive", "folder_id"),
    }
    if overrides.get("no_lut"):
        data["color"]["lut_enabled"] = False
        data["color"]["selected_luts"] = []
        overrides = {key: value for key, value in overrides.items() if key != "lut"}
    if overrides.get("no_subtitles") is True:
        data["subtitles"]["enabled"] = False
        data["subtitles"]["burn_in"] = False
    if overrides.get("subtitles") is True:
        data["subtitles"]["enabled"] = True
    if overrides.get("burn_captions") is True:
        data["subtitles"]["enabled"] = True
        data["subtitles"]["burn_in"] = True
    if overrides.get("burn_captions") is False and "burn_captions" in overrides:
        data["subtitles"]["burn_in"] = False
    for key, value in overrides.items():
        if value is None or key not in mapping:
            continue
        mapping_value = mapping[key]
        section = mapping_value[0]
        if key == "lut":
            field_name = mapping_value[1]
            data[section][field_name] = list(value)
            data["color"]["lut_enabled"] = True
        elif len(mapping_value) == 3:
            _, nested_section, field_name = mapping_value
            data[section][nested_section][field_name] = value
        elif key in {"subtitle_language", "whisper_model"}:
            field_name = mapping_value[1]
            data[section][field_name] = value
            if key == "subtitle_language":
                data["subtitles"]["burn_language"] = value
        else:
            field_name = mapping_value[1]
            data[section][field_name] = value
    if "subtitle_language" in overrides:
        data["subtitles"]["language"] = overrides["subtitle_language"] or "auto"
        data["subtitles"]["burn_language"] = overrides["subtitle_language"] or "auto"
    if "whisper_model" in overrides:
        data["subtitles"]["model_size"] = overrides["whisper_model"]
    if "progress_language" in overrides and overrides["progress_language"] is not None:
        data["telegram"]["progress_language"] = overrides["progress_language"]
    if "log_level" in overrides and overrides["log_level"] is not None:
        data["logging"]["level"] = overrides["log_level"]
        data["logging"]["console_level"] = overrides["log_level"]
        data["logging"]["file_level"] = overrides["log_level"]
    if "console_level" in overrides and overrides["console_level"] is not None:
        data["logging"]["console_level"] = overrides["console_level"]
    if "file_level" in overrides and overrides["file_level"] is not None:
        data["logging"]["file_level"] = overrides["file_level"]
    if "debug_ffmpeg_commands" in overrides and overrides["debug_ffmpeg_commands"] is not None:
        data["logging"]["debug_ffmpeg_commands"] = overrides["debug_ffmpeg_commands"]
    if "ffmpeg_stderr_tail_lines" in overrides and overrides["ffmpeg_stderr_tail_lines"] is not None:
        data["logging"]["ffmpeg_stderr_tail_lines"] = int(overrides["ffmpeg_stderr_tail_lines"])
    if "per_job_logs" in overrides and overrides["per_job_logs"] is not None:
        data["logging"]["per_job_logs"] = bool(overrides["per_job_logs"])
    if "retain_failed_temp" in overrides and overrides["retain_failed_temp"] is not None:
        data["logging"]["retain_failed_temp"] = bool(overrides["retain_failed_temp"])
    if "progress_interval_seconds" in overrides and overrides["progress_interval_seconds"] is not None:
        data["logging"]["progress_interval_seconds"] = float(overrides["progress_interval_seconds"])
    if "show_runtime_plan" in overrides and overrides["show_runtime_plan"] is not None:
        data["logging"]["show_runtime_plan"] = bool(overrides["show_runtime_plan"])
    return config_from_dict(data)


def apply_environment_overrides(config: AppConfig) -> AppConfig:
    data = asdict(config)
    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    telegram_api_base_url = os.getenv("TELEGRAM_BOT_API_BASE_URL", "").strip()
    telegram_api_file_url = os.getenv("TELEGRAM_BOT_API_FILE_URL", "").strip()
    telegram_local_mode = os.getenv("TELEGRAM_BOT_API_LOCAL_MODE", "").strip().lower()
    revid_key = os.getenv("REVID_API_KEY", "").strip()
    google_credentials = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    google_drive_folder_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID", "").strip()
    google_drive_auth_method = os.getenv("GOOGLE_DRIVE_AUTH_METHOD", "").strip()
    google_oauth_client = os.getenv("GOOGLE_OAUTH_CLIENT_SECRETS_FILE", "").strip()
    google_oauth_token = os.getenv("GOOGLE_OAUTH_TOKEN_FILE", "").strip()
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if telegram_token and not data["telegram"].get("bot_token"):
        data["telegram"]["bot_token"] = telegram_token
    if telegram_api_base_url and not data["telegram"].get("api_base_url"):
        data["telegram"]["api_base_url"] = telegram_api_base_url
    if telegram_api_file_url and not data["telegram"].get("api_file_url"):
        data["telegram"]["api_file_url"] = telegram_api_file_url
    if telegram_local_mode in {"1", "true", "yes", "y", "on"}:
        data["telegram"]["local_mode"] = True
    if revid_key and not data["revid_api"].get("api_key"):
        data["revid_api"]["api_key"] = revid_key
    if google_credentials and not data["storage"]["google_drive"].get("credentials_file"):
        data["storage"]["google_drive"]["credentials_file"] = google_credentials
    if google_drive_folder_id and not data["storage"]["google_drive"].get("folder_id"):
        data["storage"]["google_drive"]["folder_id"] = google_drive_folder_id
    if google_drive_auth_method:
        data["storage"]["google_drive"]["auth_method"] = google_drive_auth_method
    if google_oauth_client:
        data["storage"]["google_drive"]["oauth_client_secrets_file"] = google_oauth_client
    if google_oauth_token:
        data["storage"]["google_drive"]["oauth_token_file"] = google_oauth_token
    if groq_key and not data["subtitles"]["groq"].get("api_key"):
        data["subtitles"]["groq"]["api_key"] = groq_key
    return config_from_dict(data)


def load_env_file(project_root: Path) -> None:
    env_path = project_root / ".env"
    if not env_path.exists():
        return

    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        if key in os.environ and os.environ[key].strip():
            continue
        os.environ[key] = _unquote_env_value(value.strip())


def _unquote_env_value(value: str) -> str:
    if not value:
        return ""
    if value[0] in {"'", '"'} and value[-1:] == value[0]:
        return value[1:-1]
    if " #" in value:
        value = value.split(" #", 1)[0].rstrip()
    return value


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def parse_simple_yaml(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if ":" not in stripped:
            continue
        key, raw_value = stripped.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        current = stack[-1][1]
        if raw_value == "":
            child: dict[str, Any] = {}
            current[key] = child
            stack.append((indent, child))
        else:
            current[key] = _parse_scalar(raw_value)
    return root


def dump_simple_yaml(data: dict[str, Any], indent: int = 0) -> str:
    lines: list[str] = []
    pad = " " * indent
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"{pad}{key}:")
            lines.append(dump_simple_yaml(value, indent + 2).rstrip())
        else:
            lines.append(f"{pad}{key}: {_format_scalar(value)}")
    return "\n".join(lines) + "\n"


def _section(cls: type[T], data: dict[str, Any], *, section_name: str = "") -> T:
    allowed = set(getattr(cls, "__dataclass_fields__", {}).keys())
    unknown = sorted(key for key in data.keys() if key not in allowed)
    if unknown:
        LOGGER.warning(
            "Cấu hình %s có key không được hỗ trợ và sẽ bị bỏ qua: %s",
            section_name or cls.__name__,
            ", ".join(unknown),
        )
    filtered = {key: value for key, value in data.items() if key in allowed}
    return cls(**filtered)


def _storage_section(data: dict[str, Any]) -> StorageConfig:
    raw = data or {}
    base = {key: value for key, value in raw.items() if key not in {"telegram", "google_drive"}}
    provider = str(base.get("provider", "local")).strip().lower()
    if provider not in SUPPORTED_STORAGE_PROVIDERS:
        supported = ", ".join(sorted(SUPPORTED_STORAGE_PROVIDERS))
        raise ValueError(f"storage.provider không hợp lệ: {provider!r}. Giá trị hỗ trợ: {supported}.")
    base["provider"] = provider
    telegram = _section(
        StorageTelegramConfig,
        raw.get("telegram", {}),
        section_name="storage.telegram",
    )
    google_drive = _section(
        StorageGoogleDriveConfig,
        raw.get("google_drive", {}),
        section_name="storage.google_drive",
    )
    config = _section(StorageConfig, base, section_name="storage")
    config.telegram = telegram
    config.google_drive = google_drive
    return config


def _video_section(data: dict[str, Any]) -> VideoConfig:
    raw = data or {}
    base = {key: value for key, value in raw.items() if key != "center_crop_blur"}
    video = _section(VideoConfig, base, section_name="video")
    video.center_crop_blur = _section(
        CenterCropBlurConfig,
        raw.get("center_crop_blur", {}),
        section_name="video.center_crop_blur",
    )
    return video


def _parse_scalar(value: str) -> Any:
    lower = value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower in {"null", "none"}:
        return None
    if value in {"[]", "{}"} or (value.startswith("[") and value.endswith("]")):
        return ast.literal_eval(value)
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _format_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        if value == "" or any(char in value for char in ":#[]{}"):
            return json.dumps(value)
        return value
    if isinstance(value, list):
        return json.dumps(value)
    return str(value)
