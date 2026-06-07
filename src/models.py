from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ProcessingConfig:
    input_dir: str = "input"
    output_dir: str = "output"
    temp_dir: str = "temp"
    recursive: bool = True
    delete_source: bool = False
    max_workers: int | str = "auto"
    debug_ffmpeg: bool = False
    preflight_enabled: bool = True
    skip_invalid_videos: bool = True
    fail_fast_on_preflight_error: bool = False


@dataclass(slots=True)
class QueueConfig:
    enabled: bool = True
    max_workers: int = 5
    watch_input: bool = True
    scan_interval_seconds: float = 3.0
    stable_file_check_seconds: float = 2.0
    move_failed_to: str | None = "failed"
    move_completed_to: str | None = None
    state_file: str = "data/jobs.json"


@dataclass(slots=True)
class VideoConfig:
    aspect_ratio: str = "9:16"
    mode: str = "blur"
    target_resolution: str = "720p"
    keep_original_resolution: bool = False
    speed: float = 1.1
    segment_mode: str = "random"
    min_segment_seconds: float = 3.0
    max_segment_seconds: float = 5.0
    scene_threshold: float = 0.3
    alternating_flip: bool = True
    base_zoom: float = 1.52
    alternating_zoom: list[float] = field(default_factory=lambda: [1.0, 1.05])
    fade_seconds: float = 0.5
    noise_overlay: bool = False
    noise_alpha: float = 0.01


@dataclass(slots=True)
class ColorConfig:
    lut_enabled: bool = True
    max_luts: int = 3
    selected_luts: list[str] = field(default_factory=list)
    auto_select_luts: bool = False
    contrast: float = 1.03
    saturation: float = 1.08
    sharpen: bool = True


@dataclass(slots=True)
class AudioConfig:
    volume: float = 0.95
    tempo_match_speed: bool = True
    pitch_shift_semitones: float = 0
    random_eq: bool = False
    eq_bass_range: list[float] = field(default_factory=lambda: [-6, 6])
    eq_treble_range: list[float] = field(default_factory=lambda: [-6, 6])
    ambient_enabled: bool = False
    ambient_dir: str = "assets/ambient"
    ambient_volume: float = 0.01
    bgm_enabled: bool = False
    bgm_dir: str = "assets/bgm"
    bgm_volume: float = 0.05


@dataclass(slots=True)
class MetadataConfig:
    mode: str = "keep"
    custom: dict[str, str] = field(
        default_factory=lambda: {"title": "", "artist": "", "comment": ""}
    )


@dataclass(slots=True)
class EncoderConfig:
    backend: str = "auto"
    codec: str = "h264"
    preset: str = "balanced"
    pix_fmt: str = "yuv420p"
    faststart: bool = True
    allow_cpu_fallback: bool = True
    smoke_test_on_startup: bool = True
    cache_capability_results: bool = True


@dataclass(slots=True)
class RuntimeConfig:
    prefer_native_hardware_acceleration: bool = True
    container_gpu_mode: str = "auto"


@dataclass(slots=True)
class NvidiaConfig:
    enabled: bool = True


@dataclass(slots=True)
class AmdAmfConfig:
    enabled: bool = True


@dataclass(slots=True)
class VaapiConfig:
    enabled: bool = True
    device: str = "auto"


@dataclass(slots=True)
class VideotoolboxConfig:
    enabled: bool = True


@dataclass(slots=True)
class CpuConfig:
    enabled: bool = True


@dataclass(slots=True)
class GroqTranscriptionConfig:
    enabled: bool = True
    api_key: str = ""
    base_url: str = "https://api.groq.com/openai/v1"
    model: str = "whisper-large-v3-turbo"
    temperature: float = 0.0
    response_format: str = "verbose_json"
    timestamp_granularities: list[str] = field(default_factory=lambda: ["segment", "word"])
    timeout_seconds: int = 600
    connect_timeout_seconds: int = 30
    retry_attempts: int = 3
    retry_delay_seconds: float = 5.0
    fallback_to_local: bool = True
    max_concurrent_requests: int = 2
    audio_format: str = "mp3"
    audio_sample_rate: int = 16000
    audio_channels: int = 1
    audio_bitrate: str = "96k"
    chunking_enabled: bool = True
    chunk_duration_seconds: int = 600
    chunk_overlap_seconds: float = 1.0


@dataclass(slots=True)
class SubtitlesConfig:
    enabled: bool = True
    backend: str = "auto"
    model_size: str = "medium"
    language: str = "auto"
    output_srt: bool = True
    output_vtt: bool = False
    burn_in: bool = False
    burn_language: str = "auto"
    device: str = "auto"
    compute_type: str = "auto"
    output_dir: str = "output/subtitles"
    word_timestamps: bool = True
    groq: GroqTranscriptionConfig = field(default_factory=GroqTranscriptionConfig)


@dataclass(slots=True)
class TelegramConfig:
    enabled: bool = False
    bot_token: str = ""
    allowed_chat_ids: list[int] = field(default_factory=list)
    allow_all_chats_if_empty: bool = True
    send_progress_messages: bool = True
    edit_progress_message: bool = True
    progress_language: str = "vi"
    send_output_video: bool = True
    max_video_send_mb: int = 49
    api_base_url: str = ""
    api_file_url: str = ""
    local_mode: bool = False
    input_subdir: str = "input/telegram"
    message_when_queued: str = "Video received and queued."
    message_when_done: str = "Your video is ready."
    message_when_failed: str = "Video processing failed."


@dataclass(slots=True)
class StorageTelegramConfig:
    enabled: bool = False
    default_chat_id: int | None = None
    send_as_document: bool = True
    send_caption: bool = True
    max_file_size_mb: int = 49


@dataclass(slots=True)
class StorageGoogleDriveConfig:
    enabled: bool = False
    auth_method: str = "service_account"
    credentials_file: str = "secrets/google-drive-service-account.json"
    oauth_client_secrets_file: str = "secrets/google-drive-oauth-client.json"
    oauth_token_file: str = "data/google-drive-token.json"
    folder_id: str = ""
    shared_drive_id: str = ""
    make_public: bool = False
    overwrite_existing: bool = False
    chunk_size_mb: int = 8


@dataclass(slots=True)
class StorageConfig:
    provider: str = "local"
    keep_local_file: bool = True
    delete_local_after_upload: bool = False
    upload_only_final_output: bool = True
    upload_subtitles: bool = False
    retry_attempts: int = 3
    retry_delay_seconds: float = 5.0
    timeout_seconds: int = 600
    max_concurrent_uploads: int = 2
    state_file: str = "data/storage_uploads.json"
    telegram: StorageTelegramConfig = field(default_factory=StorageTelegramConfig)
    google_drive: StorageGoogleDriveConfig = field(default_factory=StorageGoogleDriveConfig)


@dataclass(slots=True)
class RevidAPIConfig:
    enabled: bool = True
    api_key: str = ""
    endpoint: str = "https://api.revidapi.com/paid/tiktok/download"
    timeout_seconds: int = 60
    download_timeout_seconds: int = 300


@dataclass(slots=True)
class TiktokdlFallbackConfig:
    enabled: bool = False
    endpoint: str = "https://tiktokios.id/wp-admin/admin-ajax.php"
    tkdl_nonce: str = "458d52a803"
    timeout_seconds: int = 60
    download_timeout_seconds: int = 300


@dataclass(slots=True)
class FormattingConfig:
    max_chars_per_line: int = 20
    max_lines: int = 2
    max_chars_per_cue: int = 40
    max_words_per_cue: int = 7
    min_duration: float = 0.7
    max_duration: float = 2.6
    pause_threshold: float = 0.45
    caption_renderer: str = "rounded_box"
    caption_position: str = "bottom"
    caption_vertical_offset: int = 0
    caption_font_name: str = "Arial"
    caption_font_file: str = "BeVietnamPro-Bold.ttf"
    caption_font_fallback: str = "Arial"
    caption_font_size: int = 54
    caption_text_color: str = "#111111"
    caption_text_opacity: float = 1.0
    caption_background_color: str = "#FFFFFF"
    caption_background_opacity: float = 0.95
    caption_padding_x: int = 28
    caption_padding_y: int = 14
    caption_border_radius: int = 18
    caption_outline_color: str = "#000000"
    caption_outline_opacity: float = 0.0
    caption_outline: int = 0
    caption_shadow_enabled: bool = True
    caption_shadow_color: str = "#000000"
    caption_shadow_opacity: float = 0.25
    caption_shadow_offset_x: int = 0
    caption_shadow_offset_y: int = 4
    caption_shadow_blur: int = 6
    caption_margin_v: int = 140
    caption_max_width_percent: int = 82
    caption_box_enabled: bool = True
    caption_shadow: int = 1


@dataclass(slots=True)
class AppConfig:
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    queue: QueueConfig = field(default_factory=QueueConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    color: ColorConfig = field(default_factory=ColorConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    metadata: MetadataConfig = field(default_factory=MetadataConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    subtitles: SubtitlesConfig = field(default_factory=SubtitlesConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    revid_api: RevidAPIConfig = field(default_factory=RevidAPIConfig)
    tiktokdl_fallback: TiktokdlFallbackConfig = field(default_factory=TiktokdlFallbackConfig)
    formatting: FormattingConfig = field(default_factory=FormattingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    nvidia: NvidiaConfig = field(default_factory=NvidiaConfig)
    amd_amf: AmdAmfConfig = field(default_factory=AmdAmfConfig)
    vaapi: VaapiConfig = field(default_factory=VaapiConfig)
    videotoolbox: VideotoolboxConfig = field(default_factory=VideotoolboxConfig)
    cpu: CpuConfig = field(default_factory=CpuConfig)
    logging: "LoggingConfig" = field(default_factory=lambda: LoggingConfig())


@dataclass(slots=True)
class VideoInfo:
    path: Path
    duration: float
    width: int
    height: int
    fps: float
    has_audio: bool
    video_codec: str = ""
    audio_codec: str = ""
    sample_aspect_ratio: str = ""
    display_aspect_ratio: str = ""
    time_base: str = ""
    audio_sample_rate: int = 0
    audio_channels: int = 0
    audio_bitrate: int = 0
    audio_duration: float = 0.0
    video_bitrate: int = 0


class JobSource(str, Enum):
    LOCAL_INPUT = "local_input"
    TELEGRAM_TIKTOK = "telegram_tiktok"


class JobStatus(str, Enum):
    PENDING = "pending"
    DOWNLOADING = "downloading"
    QUEUED = "queued"
    PROCESSING = "processing"
    RENDERED = "rendered"
    UPLOADING = "uploading"
    COMPLETED = "completed"
    UPLOAD_FAILED = "upload_failed"
    FAILED = "failed"


@dataclass(slots=True)
class VideoJob:
    job_id: str
    source: JobSource
    status: JobStatus
    input_path: str
    output_path: str | None = None
    chat_id: int | None = None
    telegram_chat_id: int | None = None
    telegram_status_message_id: int | None = None
    telegram_status_text: str = ""
    original_url: str | None = None
    created_at: str = ""
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None
    file_size: int = 0
    modified_time: float = 0.0
    identity: str = ""
    metadata_path: str | None = None
    output_size: int = 0


@dataclass(slots=True)
class EncoderSelection:
    backend: str
    codec_name: str
    args: list[str]
    description: str
    requested_backend: str = ""
    fallback_reason: str | None = None


@dataclass(slots=True)
class LoggingConfig:
    level: str = "INFO"
    console_level: str = "INFO"
    file_level: str = "DEBUG"
    third_party_level: str = "WARNING"
    compact_console: bool = True
    debug_ffmpeg_commands: bool = False
    ffmpeg_stderr_tail_lines: int = 40
    per_job_logs: bool = True
    retain_failed_temp: bool = True
    show_stage_start: bool = False
    show_stage_done: bool = True
    segment_log_mode: str = "summary"
    caption_log_mode: str = "summary"
    queue_duplicate_log_level: str = "DEBUG"
    queue_heartbeat_seconds: float = 60.0
    progress_interval_seconds: float = 10.0
    suppress_repeated_messages_seconds: float = 60.0
    redact_secrets: bool = True
    show_runtime_plan: bool = False


@dataclass(slots=True)
class ProcessResult:
    source: Path
    output: Path | None
    success: bool
    elapsed_seconds: float
    error: str | None = None


class PreflightStatus(str, Enum):
    VALID = "valid"
    WARNING = "warning"
    INVALID = "invalid"
    FATAL = "fatal"


@dataclass(slots=True)
class PreflightIssue:
    severity: PreflightStatus
    message: str
    code: str = ""


@dataclass(slots=True)
class PreflightVideoResult:
    source: Path
    status: PreflightStatus
    issues: list[PreflightIssue] = field(default_factory=list)
    output: Path | None = None
    probe: VideoInfo | None = None
    supported: bool = True


@dataclass(slots=True)
class PreflightBatchResult:
    input_root: Path
    videos: list[PreflightVideoResult] = field(default_factory=list)
    ffmpeg_available: bool = True
    ffprobe_available: bool = True
    encoder_available: bool = True
    lut_paths: list[Path] = field(default_factory=list)
    lut_warnings: list[str] = field(default_factory=list)
    issues: list[PreflightIssue] = field(default_factory=list)
    fatal_issues: list[PreflightIssue] = field(default_factory=list)

    @property
    def total_count(self) -> int:
        return len(self.videos)

    @property
    def valid_videos(self) -> list[PreflightVideoResult]:
        return [item for item in self.videos if item.status == PreflightStatus.VALID]

    @property
    def warning_videos(self) -> list[PreflightVideoResult]:
        return [item for item in self.videos if item.status == PreflightStatus.WARNING]

    @property
    def invalid_videos(self) -> list[PreflightVideoResult]:
        return [item for item in self.videos if item.status == PreflightStatus.INVALID]

    @property
    def processable_videos(self) -> list[PreflightVideoResult]:
        return self.valid_videos + self.warning_videos

    @property
    def has_fatal_errors(self) -> bool:
        return bool(self.fatal_issues) or not self.ffmpeg_available or not self.ffprobe_available

    @property
    def summary_counts(self) -> tuple[int, int, int]:
        return len(self.valid_videos), len(self.warning_videos), len(self.invalid_videos)


JsonDict = dict[str, Any]
