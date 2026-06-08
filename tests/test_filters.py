from pathlib import Path

from config import default_config
from models import EncoderSelection, JobSource, JobStatus, VideoInfo, VideoJob
import processing.pipeline as pipeline
from ffmpeg_tools.filters import (
    calculate_center_crop,
    build_color_adjust_filter,
    build_cinematic_blur_filter,
    build_center_crop_blur_filter,
    build_crop_filter,
    build_lut_filter,
    parse_aspect_ratio,
    parse_target_resolution,
    suggest_center_crop_aspect_ratio,
)
from processing.pipeline import (
    _build_filter_complex,
    _resolve_center_crop_foreground_ratio,
    _resolve_output_dimensions,
    _segment_zoom,
)
from processing.video import Segment
from utils.runtime_logging import build_job_runtime_context


def test_parse_aspect_ratio() -> None:
    assert parse_aspect_ratio("9:16") == (9, 16)
    assert parse_aspect_ratio("16x9") == (16, 9)


def test_parse_target_resolution_vertical() -> None:
    assert parse_target_resolution("720p", "9:16") == (404, 720)


def test_build_filter_strings() -> None:
    assert "crop=1080:1920" in build_crop_filter(1080, 1920)
    assert "eq=contrast=1.03:saturation=1.08" in build_color_adjust_filter(1.03, 1.08, True)
    chain = build_lut_filter([Path("assets/luts/look.cube"), Path("assets/luts/look2.cube")])
    assert "lut3d=file=" in chain
    assert chain.count("lut3d=file=") == 2


def test_cinematic_blur_defaults_foreground_to_3_by_4() -> None:
    chain = build_cinematic_blur_filter(720, 1280)

    assert "scale=720:1280:force_original_aspect_ratio=decrease" in chain


def test_calculate_center_crop_1080x1920_to_3_by_4() -> None:
    crop = calculate_center_crop(1080, 1920, "3:4")

    assert crop.crop_width == 1080
    assert crop.crop_height == 1440
    assert crop.crop_x == 0
    assert crop.crop_y == 240
    assert crop.crop_width % 2 == 0
    assert crop.crop_height % 2 == 0


def test_calculate_center_crop_720x1280_to_3_by_4() -> None:
    crop = calculate_center_crop(720, 1280, "3:4")

    assert crop.crop_width == 720
    assert crop.crop_height == 960
    assert crop.crop_x == 0
    assert crop.crop_y == 160
    assert crop.crop_width % 2 == 0
    assert crop.crop_height % 2 == 0


def test_calculate_center_crop_1920x1080_to_4_by_3() -> None:
    crop = calculate_center_crop(1920, 1080, "4:3")

    assert crop.crop_width == 1440
    assert crop.crop_height == 1080
    assert crop.crop_x == 240
    assert crop.crop_y == 0
    assert crop.crop_width % 2 == 0
    assert crop.crop_height % 2 == 0


def test_suggest_center_crop_aspect_ratio_matches_input_orientation() -> None:
    assert suggest_center_crop_aspect_ratio(1080, 1920) == "3:4"
    assert suggest_center_crop_aspect_ratio(1920, 1080) == "4:3"


def test_center_crop_blur_filter_uses_crop_and_no_scale() -> None:
    crop = calculate_center_crop(1080, 1920, "3:4")
    chain = build_center_crop_blur_filter(1080, 1920, foreground_aspect_ratio="3:4", blur_sigma=30, crop=crop)

    assert "gblur=sigma=30" in chain
    assert "scale=" not in chain
    assert "crop=1080:1440:0:240" in chain
    assert "overlay=(W-w)/2:(H-h)/2" in chain


def test_center_crop_blur_preserves_input_resolution() -> None:
    config = default_config()
    config.video.mode = "center_crop_blur"
    info = VideoInfo(
        path=Path("input.mp4"),
        duration=10.0,
        width=1080,
        height=1920,
        fps=30.0,
        has_audio=True,
    )

    assert _resolve_output_dimensions(info, config) == (1080, 1920)


def test_center_crop_blur_disables_foreground_zoom_by_default() -> None:
    config = default_config()
    config.video.mode = "center_crop_blur"
    config.video.center_crop_blur.allow_foreground_zoom = False

    assert _segment_zoom(config, 0) == 1.0


def test_segmented_pipeline_final_pass_keeps_job_and_flip(monkeypatch, tmp_path: Path) -> None:
    config = default_config()
    config.video.mode = "center_crop_blur"
    config.video.alternating_flip = True
    config.logging.segment_log_mode = "none"
    info = VideoInfo(
        path=tmp_path / "input.mp4",
        duration=8.0,
        width=720,
        height=1280,
        fps=30.0,
        has_audio=False,
    )
    job = VideoJob(
        job_id="job_test",
        source=JobSource.LOCAL_INPUT,
        status=JobStatus.PROCESSING,
        input_path=str(info.path),
    )
    job_context = build_job_runtime_context(
        job_id=job.job_id,
        source=job.source.value,
        input_path=info.path,
        output_path=tmp_path / "output.mp4",
        worker_slot=None,
        worker_total=None,
    )
    encoder = EncoderSelection(
        backend="cpu_h264",
        codec_name="libx264",
        args=[],
        description="test encoder",
    )
    rendered_filters: list[str] = []
    final_pass_kwargs: dict[str, object] = {}

    def fake_render_segment(**kwargs) -> None:
        rendered_filters.append(kwargs["segment_filter"])
        kwargs["segment_file"].write_bytes(b"segment")

    def fake_concat_segments(concat_source: Path, concat_output: Path, debug: bool, stderr_tail_lines: int) -> None:
        concat_output.write_bytes(b"concat")

    def fake_probe_video(path: Path) -> VideoInfo:
        return VideoInfo(
            path=path,
            duration=8.0,
            width=720,
            height=1280,
            fps=30.0,
            has_audio=False,
        )

    def fake_run_single_pass(**kwargs) -> None:
        final_pass_kwargs.update(kwargs)

    monkeypatch.setattr(pipeline, "_render_segment", fake_render_segment)
    monkeypatch.setattr(pipeline, "_concat_segments", fake_concat_segments)
    monkeypatch.setattr(pipeline, "probe_video", fake_probe_video)
    monkeypatch.setattr(pipeline, "_run_single_pass", fake_run_single_pass)

    pipeline._run_segmented_pipeline(
        input_file=info.path,
        output_file=tmp_path / "output.mp4",
        info=info,
        project_root=tmp_path,
        config=config,
        job=job,
        lut_paths=[],
        work_dir=tmp_path / "work",
        encoder=encoder,
        width=720,
        height=1280,
        segments=[Segment(start=0.0, end=4.0, index=0), Segment(start=4.0, end=8.0, index=1)],
        job_context=job_context,
        ffmpeg_debug=False,
    )

    assert final_pass_kwargs["job"] is job
    assert final_pass_kwargs["apply_visual_effects"] is False
    assert any("hflip" in chain for chain in rendered_filters)


def test_center_crop_blur_does_not_change_audio_speed_path() -> None:
    config = default_config()
    config.video.mode = "center_crop_blur"
    config.video.speed = 1.0
    info = VideoInfo(
        path=Path("input.mp4"),
        duration=10.0,
        width=1080,
        height=1920,
        fps=30.0,
        has_audio=True,
    )
    job_context = build_job_runtime_context(
        job_id="job_test",
        source="local_input",
        input_path=Path("input.mp4"),
        output_path=Path("output.mp4"),
        worker_slot=None,
        worker_total=None,
    )
    width, height = _resolve_output_dimensions(info, config)
    filter_complex, _ = _build_filter_complex(
        info=info,
        width=width,
        height=height,
        encoder_backend="libx264",
        lut_paths=[],
        ambient=None,
        bgm=None,
        config=config,
        apply_visual_effects=True,
        job=None,
        job_context=job_context,
    )

    assert "atempo=" not in filter_complex
    assert "volume=" in filter_complex


def test_center_crop_blur_telegram_jobs_use_nearest_crop_ratio() -> None:
    config = default_config()
    config.video.mode = "center_crop_blur"
    config.video.center_crop_blur.foreground_aspect_ratio = "3:4"
    info = VideoInfo(
        path=Path("input.mp4"),
        duration=10.0,
        width=1920,
        height=1080,
        fps=30.0,
        has_audio=True,
    )
    job = VideoJob(
        job_id="job_telegram",
        source=JobSource.TELEGRAM_TIKTOK,
        status=JobStatus.QUEUED,
        input_path="input.mp4",
    )

    assert _resolve_center_crop_foreground_ratio(info, config, job) == "4:3"
