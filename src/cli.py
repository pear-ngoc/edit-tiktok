from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import app


def main(argv: Sequence[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    args, global_options = _extract_global_options(args)
    project_root = Path(__file__).resolve().parent.parent

    if not args:
        app.process_default(
            project_root,
            config_profile=global_options["config_profile"],
            overrides=global_options["overrides"],
            skip_preflight=global_options["skip_preflight"],
        )
        return

    command = args[0]
    rest = args[1:]
    if command == "init":
        app.init_project(project_root)
    elif command == "doctor":
        app.doctor(
            project_root,
            config_profile=global_options["config_profile"],
            overrides=global_options["overrides"],
        )
    elif command == "list-luts":
        app.list_luts(project_root)
    elif command == "list-fonts":
        app.list_fonts(project_root)
    elif command == "wizard":
        app.wizard(
            project_root,
            config_profile=global_options["config_profile"],
            overrides=global_options["overrides"],
        )
    elif command == "watch":
        app.watch(
            project_root,
            config_profile=global_options["config_profile"],
            overrides=global_options["overrides"],
        )
    elif command == "telegram":
        app.telegram(
            project_root,
            config_profile=global_options["config_profile"],
            overrides=global_options["overrides"],
        )
    elif command == "worker":
        namespace = _worker_parser().parse_args(rest)
        app.worker(
            project_root,
            config_profile=global_options["config_profile"],
            overrides=global_options["overrides"],
            watch_input=namespace.watch_input,
            enable_telegram=namespace.telegram,
        )
    elif command == "preflight":
        app.preflight_only(
            project_root,
            config_profile=global_options["config_profile"],
            overrides=global_options["overrides"],
        )
    elif command == "process":
        namespace = _process_parser().parse_args(rest)
        app.process_default(
            project_root,
            {**vars(namespace), **global_options["overrides"]},
            config_profile=global_options["config_profile"],
            skip_preflight=global_options["skip_preflight"],
        )
    elif command == "storage":
        _handle_storage_command(project_root, rest, global_options)
    elif command == "storage-doctor":
        app.storage_doctor(
            project_root,
            config_profile=global_options["config_profile"],
            overrides=global_options["overrides"],
        )
    elif command == "upload-file":
        if not rest:
            print("Thiếu đường dẫn file upload.")
            raise SystemExit(2)
        app.storage_upload_file(
            project_root,
            Path(rest[0]),
            config_profile=global_options["config_profile"],
            overrides=global_options["overrides"],
        )
    elif command == "retry-uploads":
        app.storage_retry(
            project_root,
            config_profile=global_options["config_profile"],
            overrides=global_options["overrides"],
        )
    elif command == "storage-status":
        app.storage_status(
            project_root,
            config_profile=global_options["config_profile"],
            overrides=global_options["overrides"],
        )
    elif command == "storage-auth":
        app.storage_auth(
            project_root,
            config_profile=global_options["config_profile"],
            overrides=global_options["overrides"],
        )
    elif command == "configs":
        _handle_configs_command(project_root, rest)
    elif command == "list-configs":
        app.list_configs(project_root)
    elif command == "show-config":
        if not rest:
            print("Thiếu tên cấu hình.")
            raise SystemExit(2)
        app.show_config(project_root, rest[0])
    elif command == "use-config":
        if not rest:
            print("Thiếu tên cấu hình.")
            raise SystemExit(2)
        app.process_default(
            project_root,
            overrides=global_options["overrides"],
            config_profile=rest[0],
            skip_preflight=global_options["skip_preflight"],
        )
    elif command == "clear":
        namespace = _clear_parser().parse_args(rest)
        app.clear_workspace(
            project_root,
            config_profile=global_options["config_profile"],
            overrides=global_options["overrides"],
            scope=namespace.scope,
            yes=namespace.yes,
            dry_run=namespace.dry_run,
        )
    elif command in {"-h", "--help", "help"}:
        _root_parser().print_help()
    else:
        print(f"Lệnh không hợp lệ: {command}", file=sys.stderr)
        _root_parser().print_help()
        raise SystemExit(2)


def _root_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="Xử lý video hàng loạt bằng FFmpeg. Không truyền tham số sẽ chạy ngay workflow mặc định.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        help="Lệnh: init, doctor, list-luts, list-fonts, preflight, process, wizard, watch, telegram, worker, storage, clear, configs, list-configs, show-config, use-config",
    )
    return parser


def _extract_global_options(args: list[str]) -> tuple[list[str], dict[str, object]]:
    cleaned: list[str] = []
    config_profile: str | None = None
    skip_preflight = False
    overrides: dict[str, object] = {}
    index = 0
    while index < len(args):
        current = args[index]
        if current == "--config-profile" and index + 1 < len(args):
            config_profile = args[index + 1]
            index += 2
            continue
        if current.startswith("--config-profile="):
            config_profile = current.split("=", 1)[1]
            index += 1
            continue
        if current == "--skip-preflight":
            skip_preflight = True
            index += 1
            continue
        if current == "--log-level" and index + 1 < len(args):
            overrides["log_level"] = args[index + 1]
            index += 2
            continue
        if current.startswith("--log-level="):
            overrides["log_level"] = current.split("=", 1)[1]
            index += 1
            continue
        if current == "--debug-ffmpeg":
            overrides["debug_ffmpeg"] = True
            overrides["debug_ffmpeg_commands"] = True
            index += 1
            continue
        if current == "--retain-failed-temp":
            overrides["retain_failed_temp"] = True
            index += 1
            continue
        if current == "--no-retain-failed-temp":
            overrides["retain_failed_temp"] = False
            index += 1
            continue
        if current == "--progress-interval" and index + 1 < len(args):
            overrides["progress_interval_seconds"] = float(args[index + 1])
            index += 2
            continue
        if current.startswith("--progress-interval="):
            overrides["progress_interval_seconds"] = float(current.split("=", 1)[1])
            index += 1
            continue
        if current == "--show-runtime-plan":
            overrides["show_runtime_plan"] = True
            index += 1
            continue
        cleaned.append(current)
        index += 1
    return cleaned, {"config_profile": config_profile, "skip_preflight": skip_preflight, "overrides": overrides}


def _handle_configs_command(project_root: Path, rest: list[str]) -> None:
    if not rest:
        app.list_configs(project_root)
        return
    subcommand = rest[0]
    if subcommand == "list":
        app.list_configs(project_root)
    elif subcommand == "show":
        if len(rest) < 2:
            print("Thiếu tên cấu hình.")
            raise SystemExit(2)
        app.show_config(project_root, rest[1])
    else:
        print(f"Lệnh con không hợp lệ: {subcommand}", file=sys.stderr)
        raise SystemExit(2)


def _handle_storage_command(project_root: Path, rest: list[str], global_options: dict[str, object]) -> None:
    if not rest:
        print("Thiếu lệnh storage: doctor, auth, upload, retry, status.")
        raise SystemExit(2)
    subcommand = rest[0]
    overrides = global_options["overrides"]
    if subcommand == "doctor":
        app.storage_doctor(
            project_root,
            config_profile=global_options["config_profile"],
            overrides=overrides,
        )
    elif subcommand == "upload":
        if len(rest) < 2:
            print("Thiếu đường dẫn file upload.")
            raise SystemExit(2)
        app.storage_upload_file(
            project_root,
            Path(rest[1]),
            config_profile=global_options["config_profile"],
            overrides=overrides,
        )
    elif subcommand == "retry":
        app.storage_retry(
            project_root,
            config_profile=global_options["config_profile"],
            overrides=overrides,
        )
    elif subcommand == "status":
        app.storage_status(
            project_root,
            config_profile=global_options["config_profile"],
            overrides=overrides,
        )
    elif subcommand == "auth":
        app.storage_auth(
            project_root,
            config_profile=global_options["config_profile"],
            overrides=overrides,
        )
    else:
        print(f"Lệnh storage không hợp lệ: {subcommand}", file=sys.stderr)
        raise SystemExit(2)


def _process_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python main.py process",
        description="Ghi đè nhanh một số thiết lập xử lý.",
    )
    parser.add_argument("--input", dest="input", help="Thư mục đầu vào")
    parser.add_argument("--output", dest="output", help="Thư mục đầu ra")
    parser.add_argument("--aspect", dest="aspect", help="Tỷ lệ khung hình")
    parser.add_argument("--mode", dest="mode", help="Chế độ xử lý video")
    parser.add_argument("--encoder", dest="encoder", help="Backend bộ mã hóa")
    parser.add_argument("--preset", dest="preset", help="Mức chất lượng")
    parser.add_argument("--workers", dest="workers", help="Số luồng")
    parser.add_argument("--speed", dest="speed", type=float, help="Tốc độ video")
    parser.add_argument(
        "--target-resolution",
        dest="target_resolution",
        help="Độ phân giải mục tiêu",
    )
    subtitle_group = parser.add_mutually_exclusive_group()
    subtitle_group.add_argument(
        "--subtitles",
        dest="subtitles",
        action="store_true",
        default=None,
        help="Bật tạo phụ đề",
    )
    subtitle_group.add_argument(
        "--no-subtitles",
        dest="no_subtitles",
        action="store_true",
        default=None,
        help="Tắt tạo phụ đề",
    )
    burn_group = parser.add_mutually_exclusive_group()
    burn_group.add_argument(
        "--burn-captions",
        dest="burn_captions",
        action="store_true",
        default=None,
        help="Burn phụ đề vào video",
    )
    burn_group.add_argument(
        "--no-burn-captions",
        dest="burn_captions",
        action="store_false",
        default=None,
        help="Không burn phụ đề",
    )
    parser.add_argument(
        "--subtitle-language",
        dest="subtitle_language",
        help="Ngôn ngữ phụ đề, ví dụ vi, en, ja hoặc auto",
    )
    parser.add_argument(
        "--whisper-model",
        dest="whisper_model",
        help="Kích thước model faster-whisper",
    )
    parser.add_argument(
        "--caption-max-chars-per-line",
        dest="caption_max_chars_per_line",
        type=int,
        help="Số ký tự tối đa mỗi dòng caption",
    )
    parser.add_argument(
        "--caption-max-lines",
        dest="caption_max_lines",
        type=int,
        help="Số dòng tối đa mỗi caption",
    )
    parser.add_argument(
        "--caption-max-words",
        dest="caption_max_words",
        type=int,
        help="Số từ tối đa mỗi caption",
    )
    parser.add_argument(
        "--caption-max-duration",
        dest="caption_max_duration",
        type=float,
        help="Thời lượng tối đa mỗi caption",
    )
    parser.add_argument(
        "--caption-position",
        dest="caption_position",
        help="Vị trí caption, mặc định bottom",
    )
    parser.add_argument(
        "--caption-font-size",
        dest="caption_font_size",
        type=int,
        help="Cỡ chữ caption khi burn-in ASS",
    )
    parser.add_argument(
        "--lut",
        dest="lut",
        action="append",
        help="Tên file LUT .cube trong assets/luts, có thể lặp lại nhiều lần",
    )
    parser.add_argument(
        "--no-lut",
        dest="no_lut",
        action="store_true",
        help="Tắt toàn bộ LUT",
    )
    parser.add_argument(
        "--storage",
        dest="storage",
        choices=["local", "telegram", "google_drive", "both"],
        help="Provider lưu/gửi output sau render",
    )
    parser.add_argument(
        "--telegram-chat-id",
        dest="telegram_chat_id",
        type=int,
        help="Chat ID mặc định cho local job khi storage telegram/both",
    )
    parser.add_argument(
        "--drive-folder-id",
        dest="drive_folder_id",
        help="Google Drive folder ID cho storage google_drive/both",
    )
    return parser


def _worker_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python main.py worker",
        description="Chạy runtime nền liên tục cho input watcher và Telegram bot.",
    )
    telegram_group = parser.add_mutually_exclusive_group()
    telegram_group.add_argument(
        "--telegram",
        dest="telegram",
        action="store_true",
        default=None,
        help="Bật Telegram bot",
    )
    telegram_group.add_argument(
        "--no-telegram",
        dest="telegram",
        action="store_false",
        default=None,
        help="Tắt Telegram bot",
    )
    watch_group = parser.add_mutually_exclusive_group()
    watch_group.add_argument(
        "--watch-input",
        dest="watch_input",
        action="store_true",
        default=None,
        help="Bật theo dõi input liên tục",
    )
    watch_group.add_argument(
        "--no-watch-input",
        dest="watch_input",
        action="store_false",
        default=None,
        help="Tắt theo dõi input liên tục",
    )
    return parser


def _clear_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python main.py clear",
        description="Xóa dữ liệu sinh ra và/hoặc input theo phạm vi được chọn.",
    )
    parser.add_argument(
        "scope",
        nargs="?",
        choices=["all", "input", "generated"],
        default="all",
        help="Phạm vi xóa: all, input hoặc generated",
    )
    parser.add_argument(
        "--yes",
        dest="yes",
        action="store_true",
        help="Không hỏi xác nhận trước khi xóa",
    )
    parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="Chỉ in danh sách cần xóa, không xóa thật",
    )
    return parser


if __name__ == "__main__":
    main()
