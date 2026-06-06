from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path

from config import apply_overrides, ensure_config_file, load_config, save_default_config
from ffmpeg_tools.encoders import detect_available_encoders, probe_nvidia_runtime, select_encoder
from logging_config import configure_logging
from models import AppConfig
from processing.batch import run_batch
from processing.caption_renderer import available_caption_fonts
from processing.lut import available_luts, parse_lut_selection_input
from processing.preflight import (
    ensure_input_videos_exist,
    print_preflight_summary,
    run_preflight_checks,
)
from queue_manager import QueueManager
from integrations.telegram_bot import TelegramBotService
from utils.paths import ensure_gitkeep_files, ensure_runtime_dirs, resolve_project_path
from utils.cleanup import clear_workspace as clear_runtime_workspace
from utils.platform import get_platform_info, is_python_supported
from utils.runtime_logging import log_startup_summary, resolve_whisper_runtime
from utils.profiles import (
    list_saved_profiles,
    load_profile_config,
    merge_profile_config,
    profile_path,
    sanitize_profile_name,
    save_profile_config,
)


LOGGER = logging.getLogger(__name__)


def init_project(project_root: Path) -> None:
    ensure_gitkeep_files(project_root)
    config_path = project_root / "config.yaml"
    if not config_path.exists():
        example_path = project_root / "config.example.yaml"
        if example_path.exists():
            shutil.copyfile(example_path, config_path)
        else:
            save_default_config(config_path)
    print("Đã khởi tạo dự án.")
    print(f"Cấu hình: {config_path}")


def process_default(
    project_root: Path,
    overrides: dict[str, object] | None = None,
    *,
    config_profile: str | None = None,
    skip_preflight: bool = False,
) -> None:
    config = _load_effective_config(project_root, config_profile=config_profile, overrides=overrides)
    if config is None:
        return
    if not skip_preflight and not ensure_input_videos_exist(project_root, config):
        return
    ensure_runtime_dirs(project_root, config)
    configure_logging(project_root / "logs", config=config, debug=config.processing.debug_ffmpeg)
    info = get_platform_info()
    encoders = detect_available_encoders()
    selected = select_encoder(config.encoder, encoders, system=info.system, machine=info.machine)
    log_startup_summary(
        project_root,
        config,
        ffmpeg_path=info.ffmpeg_path,
        ffprobe_path=info.ffprobe_path,
        available_encoders=encoders,
        encoder=selected,
        whisper_runtime=resolve_whisper_runtime(config.subtitles),
    )
    run_batch(project_root, config)


def preflight_only(
    project_root: Path,
    overrides: dict[str, object] | None = None,
    *,
    config_profile: str | None = None,
) -> None:
    config = _load_effective_config(project_root, config_profile=config_profile, overrides=overrides)
    if config is None:
        return
    if not ensure_input_videos_exist(project_root, config):
        return
    ensure_runtime_dirs(project_root, config)
    configure_logging(project_root / "logs", config=config, debug=config.processing.debug_ffmpeg)
    result = run_preflight_checks(project_root, config)
    print_preflight_summary(result)


def doctor(
    project_root: Path,
    *,
    config_profile: str | None = None,
    overrides: dict[str, object] | None = None,
) -> None:
    config = _load_effective_config(project_root, config_profile=config_profile, overrides=overrides)
    if config is None:
        return
    ensure_runtime_dirs(project_root, config)
    configure_logging(project_root / "logs", config=config, debug=config.processing.debug_ffmpeg)

    info = get_platform_info()
    encoders = detect_available_encoders()
    selected = select_encoder(config.encoder, encoders, system=info.system, machine=info.machine)
    nvenc_runtime = probe_nvidia_runtime()

    print("Kiểm tra môi trường edit-tiktok")
    print(f"Python: {info.python_version} ({'đạt' if is_python_supported() else 'cần >=3.11'})")
    print(f"Hệ điều hành: {info.system} {info.machine}")
    print(f"Apple Silicon: {'có' if info.is_apple_silicon else 'không'}")
    print(f"FFmpeg: {info.ffmpeg_path or 'thiếu'}")
    print(f"FFprobe: {info.ffprobe_path or 'thiếu'}")
    print(f"Encoder build hỗ trợ: {', '.join(_interesting_encoders(encoders)) or 'không có'}")
    print(
        "NVENC runtime: "
        + ("có" if nvenc_runtime.nvidia_runtime_available else f"không ({nvenc_runtime.nvidia_runtime_reason or 'unknown'})")
    )
    print(f"Backend khuyến nghị: {selected.description}")
    log_startup_summary(
        project_root,
        config,
        ffmpeg_path=info.ffmpeg_path,
        ffprobe_path=info.ffprobe_path,
        available_encoders=encoders,
        encoder=selected,
        whisper_runtime=resolve_whisper_runtime(config.subtitles),
    )


def list_luts(project_root: Path) -> None:
    ensure_runtime_dirs(project_root)
    luts = available_luts(project_root)
    if not luts:
        print("Không tìm thấy file LUT .cube nào trong assets/luts")
        return
    for path in luts:
        print(path.relative_to(project_root))


def list_fonts(project_root: Path) -> None:
    ensure_runtime_dirs(project_root)
    fonts = available_caption_fonts(project_root)
    if not fonts:
        print("Không tìm thấy font caption nào trong assets/font")
        return
    print("Available caption fonts:")
    for index, path in enumerate(fonts, start=1):
        print(f"{index}. {path.name}")


def wizard(
    project_root: Path,
    *,
    config_profile: str | None = None,
    overrides: dict[str, object] | None = None,
) -> None:
    config = _load_effective_config(project_root, config_profile=config_profile, overrides=overrides)
    if config is None:
        return
    if not ensure_input_videos_exist(project_root, config):
        return
    print("Trình hướng dẫn tương tác")
    _configure_subtitles_interactively(config)
    config.video.aspect_ratio = _ask("Tỷ lệ khung hình", config.video.aspect_ratio)
    config.video.mode = _ask("Chế độ (crop/blur/original/target)", config.video.mode)
    config.video.noise_overlay = _ask_bool("Bật lớp nhiễu", config.video.noise_overlay)
    config.audio.pitch_shift_semitones = float(
        _ask("Dịch cao độ (semitone)", str(config.audio.pitch_shift_semitones))
    )
    config.audio.ambient_enabled = _ask_bool("Bật âm thanh môi trường", config.audio.ambient_enabled)
    config.audio.bgm_enabled = _ask_bool("Bật nhạc nền", config.audio.bgm_enabled)
    config.metadata.mode = _ask("Chế độ metadata (keep/remove/custom)", config.metadata.mode)
    config.encoder.backend = _ask("Backend bộ mã hóa", config.encoder.backend)
    _configure_luts_interactively(project_root, config)
    ensure_runtime_dirs(project_root, config)
    configure_logging(project_root / "logs", config=config, debug=config.processing.debug_ffmpeg)
    _print_wizard_summary(config)
    _maybe_save_wizard_profile(project_root, config)
    run_batch(project_root, config)


def watch(project_root: Path, *, config_profile: str | None = None, overrides: dict[str, object] | None = None) -> None:
    config = _load_effective_config(project_root, config_profile=config_profile, overrides=overrides)
    if config is None:
        return
    ensure_runtime_dirs(project_root, config)
    configure_logging(project_root / "logs", config=config, debug=config.processing.debug_ffmpeg)
    _run_queue_runtime(project_root, config, watch_input=True, enable_telegram=False)


def telegram(project_root: Path, *, config_profile: str | None = None, overrides: dict[str, object] | None = None) -> None:
    config = _load_effective_config(project_root, config_profile=config_profile, overrides=overrides)
    if config is None:
        return
    ensure_runtime_dirs(project_root, config)
    configure_logging(project_root / "logs", config=config, debug=config.processing.debug_ffmpeg)
    _run_queue_runtime(project_root, config, watch_input=False, enable_telegram=True)


def worker(
    project_root: Path,
    *,
    config_profile: str | None = None,
    overrides: dict[str, object] | None = None,
    watch_input: bool | None = None,
    enable_telegram: bool | None = None,
) -> None:
    config = _load_effective_config(project_root, config_profile=config_profile, overrides=overrides)
    if config is None:
        return
    ensure_runtime_dirs(project_root, config)
    configure_logging(project_root / "logs", config=config, debug=config.processing.debug_ffmpeg)
    run_watch = config.queue.watch_input if watch_input is None else watch_input
    telegram_enabled = _telegram_enabled(config) if enable_telegram is None else enable_telegram
    _run_queue_runtime(project_root, config, watch_input=run_watch, enable_telegram=telegram_enabled)


def list_configs(project_root: Path) -> None:
    profiles = list_saved_profiles(project_root)
    if not profiles:
        print("Chưa có cấu hình nào được lưu.")
        return
    print("Saved configs:")
    for index, path in enumerate(profiles, start=1):
        print(f"\n{index}. {path.stem}")


def show_config(project_root: Path, profile_name: str) -> None:
    try:
        profile = load_profile_config(project_root, profile_name)
    except FileNotFoundError:
        print(f"Không tìm thấy cấu hình '{profile_name}'.")
        _print_available_profiles(project_root)
        return

    meta = profile.get("profile", {})
    print("Saved config details:")
    print(f"- profile name: {meta.get('name', sanitize_profile_name(profile_name))}")
    print(f"- created_at: {meta.get('created_at', 'unknown')}")
    print(f"- description: {meta.get('description', '')}")
    print(f"- aspect ratio: {profile.get('video', {}).get('aspect_ratio', 'unknown')}")
    print(f"- mode: {profile.get('video', {}).get('mode', 'unknown')}")
    print(f"- encoder: {profile.get('encoder', {}).get('backend', 'unknown')}")
    print(f"- preset: {profile.get('encoder', {}).get('preset', 'unknown')}")
    color = profile.get("color", {})
    print(
        f"- LUT enabled/selected LUTs: {color.get('lut_enabled', False)} / {', '.join(color.get('selected_luts', [])) or 'none'}"
    )
    subtitles = profile.get("subtitles", {})
    print(
        f"- subtitles enabled/burn-in/language: {subtitles.get('enabled', False)} / {subtitles.get('burn_in', False)} / {subtitles.get('language', 'auto')}"
    )
    audio = profile.get("audio", {})
    print(
        f"- audio options: volume={audio.get('volume', 'unknown')}, ambient={audio.get('ambient_enabled', False)}, bgm={audio.get('bgm_enabled', False)}"
    )


def clear_workspace(
    project_root: Path,
    *,
    config_profile: str | None = None,
    overrides: dict[str, object] | None = None,
    scope: str = "all",
    yes: bool = False,
    dry_run: bool = False,
) -> None:
    config = _load_effective_config(project_root, config_profile=config_profile, overrides=overrides)
    if config is None:
        return

    include_input = scope in {"all", "input"}
    include_generated = scope in {"all", "generated"}
    targets = clear_runtime_workspace(
        project_root,
        config,
        include_input=include_input,
        include_generated=include_generated,
        dry_run=True,
    ).removed_paths
    if not targets:
        print("Không có gì để xóa.")
        return

    print("Clear sẽ xóa các thư mục sau:")
    for target in targets:
        print(f"- {target.relative_to(project_root) if target.is_relative_to(project_root) else target}")

    if not yes and not dry_run:
        if not _ask_bool("Xác nhận xóa toàn bộ dữ liệu đã sinh ra?", False):
            print("Đã hủy thao tác clear.")
            return

    result = clear_runtime_workspace(
        project_root,
        config,
        include_input=include_input,
        include_generated=include_generated,
        dry_run=dry_run,
    )
    action = "dự kiến xóa" if dry_run else "đã xóa"
    print(f"Clear {action} {result.removed_count} thư mục.")


def _configure_subtitles_interactively(config: AppConfig) -> None:
    raw = input("Burn captions into the video? [y/N]: ").strip().lower()
    if raw in {"y", "yes", "true", "1"}:
        language = input("Subtitle language? Enter language code or leave blank for auto: ").strip()
        selected_language = language or "auto"
        config.subtitles.enabled = True
        config.subtitles.burn_in = True
        config.subtitles.output_srt = True
        config.subtitles.language = selected_language
        config.subtitles.burn_language = selected_language
    else:
        config.subtitles.burn_in = False


def _ask(prompt: str, default: str) -> str:
    value = input(f"{prompt} [{default}]: ").strip()
    return value or default


def _ask_bool(prompt: str, default: bool) -> bool:
    value = input(f"{prompt} [{'Y/n' if default else 'y/N'}]: ").strip().lower()
    if not value:
        return default
    return value in {"y", "yes", "true", "1"}


def _configure_luts_interactively(project_root: Path, config: AppConfig) -> None:
    available = available_luts(project_root)
    if not available:
        print("Không có LUT nào trong assets/luts.")
        config.color.lut_enabled = False
        config.color.selected_luts = []
        return

    print("LUT khả dụng:")
    for index, path in enumerate(available, start=1):
        print(f"{index}. {path.name}")
    print(f"{len(available) + 1}. Không dùng LUT")
    selected_indexes: list[int] = []
    while True:
        raw = _ask(
            f"Chọn LUT, phân tách bằng dấu phẩy, tối đa {config.color.max_luts}",
            "Không dùng LUT",
        )
        try:
            selected_indexes = parse_lut_selection_input(raw, len(available), config.color.max_luts)
            break
        except ValueError as exc:
            print(f"Lựa chọn LUT không hợp lệ: {exc}")

    if not selected_indexes:
        config.color.lut_enabled = False
        config.color.selected_luts = []
        return

    config.color.lut_enabled = True
    config.color.selected_luts = [available[index - 1].name for index in selected_indexes]


def _print_wizard_summary(config: AppConfig) -> None:
    lut_summary = ", ".join(config.color.selected_luts) if config.color.selected_luts else "Không dùng LUT"
    subtitle_summary = "bật" if config.subtitles.enabled else "tắt"
    burn_summary = "bật" if config.subtitles.burn_in else "tắt"
    print("Tóm tắt cấu hình trước khi xử lý:")
    print(f"- Tỷ lệ khung hình: {config.video.aspect_ratio}")
    print(f"- Chế độ video: {config.video.mode}")
    print(f"- LUT: {lut_summary}")
    print(f"- Phụ đề: {subtitle_summary}")
    print(f"- Burn captions: {burn_summary}")
    print(f"- Âm thanh môi trường: {'bật' if config.audio.ambient_enabled else 'tắt'}")
    print(f"- Nhạc nền: {'bật' if config.audio.bgm_enabled else 'tắt'}")
    print(f"- Metadata: {config.metadata.mode}")
    print(f"- Encoder: {config.encoder.backend}")


def _maybe_save_wizard_profile(project_root: Path, config: AppConfig) -> None:
    if not _ask_bool("Do you want to save this configuration for later?", False):
        return

    while True:
        raw_name = input("Configuration name: ").strip()
        safe_name = sanitize_profile_name(raw_name)
        if not safe_name:
            print("Tên cấu hình không hợp lệ.")
            continue

        path = profile_path(project_root, safe_name)
        overwrite = False
        if path.exists():
            if not _ask_bool("Configuration already exists. Overwrite?", False):
                another = input("Enter another name or leave blank to skip saving: ").strip()
                if not another:
                    return
                raw_name = another
                continue
            overwrite = True

        try:
            saved_path = save_profile_config(project_root, safe_name, config, overwrite=overwrite)
            print(f"Đã lưu cấu hình vào: {saved_path}")
            return
        except Exception as exc:
            LOGGER.exception("Không lưu được cấu hình wizard")
            print(f"Không lưu được cấu hình: {exc}")
            _ask_bool("Tiếp tục xử lý mà không lưu?", True)
            return


def _load_effective_config(
    project_root: Path,
    *,
    config_profile: str | None = None,
    overrides: dict[str, object] | None = None,
) -> AppConfig | None:
    ensure_config_file(project_root)
    config = load_config(project_root)
    if config_profile:
        try:
            profile_data = load_profile_config(project_root, config_profile)
        except FileNotFoundError:
            print(f"Không tìm thấy cấu hình đã lưu '{config_profile}'.")
            _print_available_profiles(project_root)
            return None
        config = merge_profile_config(config, profile_data)
    if overrides:
        config = apply_overrides(config, overrides)
    return config


def _print_available_profiles(project_root: Path) -> None:
    profiles = list_saved_profiles(project_root)
    if not profiles:
        print("Không có cấu hình nào đã lưu.")
        return
    print("Cấu hình khả dụng:")
    for path in profiles:
        print(f"- {path.stem}")


def _interesting_encoders(encoders: list[str]) -> list[str]:
    wanted = [
        "libx264",
        "libx265",
        "h264_nvenc",
        "hevc_nvenc",
        "h264_videotoolbox",
        "hevc_videotoolbox",
    ]
    return [encoder for encoder in wanted if encoder in encoders]


def _run_queue_runtime(
    project_root: Path,
    config: AppConfig,
    *,
    watch_input: bool,
    enable_telegram: bool,
) -> None:
    from processing.lut import resolve_lut_selection
    from processing.pipeline import process_video
    info = get_platform_info()
    encoders = detect_available_encoders()
    selected = select_encoder(config.encoder, encoders, system=info.system, machine=info.machine)
    log_startup_summary(
        project_root,
        config,
        ffmpeg_path=info.ffmpeg_path,
        ffprobe_path=info.ffprobe_path,
        available_encoders=encoders,
        encoder=selected,
        whisper_runtime=resolve_whisper_runtime(config.subtitles),
    )

    resolved_luts = resolve_lut_selection(project_root, config.color)
    for warning in resolved_luts.warnings:
        print(warning)
    if resolved_luts.resolved_paths:
        print("Đang áp dụng LUT: " + ", ".join(path.name for path in resolved_luts.resolved_paths))

    input_root = resolve_project_path(project_root, config.processing.input_dir)
    output_root = resolve_project_path(project_root, config.processing.output_dir)
    temp_root = resolve_project_path(project_root, config.processing.temp_dir)

    def process_callback(job, progress_callback=None, *, worker_slot=None, worker_total=None):
        input_path = Path(job.input_path)
        return process_video(
            input_path,
            input_root=input_root,
            output_root=output_root,
            temp_root=temp_root,
            project_root=project_root,
            config=config,
            lut_paths=list(resolved_luts.resolved_paths),
            progress_callback=progress_callback,
            job=job,
            worker_slot=worker_slot,
            worker_total=worker_total,
        )

    queue_manager = QueueManager(project_root, config, process_callback)
    telegram_service: TelegramBotService | None = None
    telegram_active = enable_telegram and _telegram_enabled(config)
    if enable_telegram and not telegram_active:
        print("Telegram bot chưa được bật vì thiếu bot_token hoặc cấu hình Telegram đang tắt.")
    if telegram_active:
        telegram_service = TelegramBotService(project_root, config, queue_manager)
        queue_manager.notifier = telegram_service

    workers = max(1, int(config.queue.max_workers))
    print(
        f"Khởi động runtime hàng đợi | workers={workers} | watch_input={'có' if watch_input else 'không'} | telegram={'có' if telegram_service else 'không'}"
    )
    if telegram_service is not None:
        telegram_service.start()
    queue_manager.start(watch_input=watch_input, worker_count=workers)

    try:
        if not watch_input and telegram_service is None:
            queue_manager.wait_for_idle()
            return
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        LOGGER.info("Nhận Ctrl+C, đang dừng runtime...")
    finally:
        queue_manager.stop()
        queue_manager.join(timeout=10)
        if telegram_service is not None:
            telegram_service.stop()
            telegram_service.join(timeout=10)
        queue_manager.save_state()
        LOGGER.info("Runtime hàng đợi đã dừng")


def _telegram_enabled(config: AppConfig) -> bool:
    return bool(config.telegram.enabled or config.telegram.bot_token.strip())
