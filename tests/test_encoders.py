from ffmpeg_tools.encoders import EncoderRuntimeProbe, parse_encoder_names, select_encoder
from models import EncoderConfig


SAMPLE = """
 V....D libx264              libx264 H.264
 V....D h264_videotoolbox    VideoToolbox H.264 Encoder
 V....D h264_nvenc           NVIDIA NVENC H.264 encoder
"""


def test_parse_encoder_names() -> None:
    assert parse_encoder_names(SAMPLE) == ["h264_nvenc", "h264_videotoolbox", "libx264"]


def test_select_encoder_falls_back_to_cpu() -> None:
    selected = select_encoder(
        EncoderConfig(backend="nvidia_h264"),
        ["libx264"],
        system="Windows",
        machine="AMD64",
    )
    assert selected.backend == "cpu_h264"
    assert selected.codec_name == "libx264"


def test_select_encoder_falls_back_when_nvenc_runtime_missing() -> None:
    selected = select_encoder(
        EncoderConfig(backend="nvidia_h264"),
        ["h264_nvenc", "libx264"],
        system="Linux",
        machine="x86_64",
        runtime_probe=EncoderRuntimeProbe(False, "libcuda.so.1 không khả dụng"),
    )
    assert selected.backend == "cpu_h264"
    assert selected.codec_name == "libx264"
    assert "libcuda.so.1" in (selected.fallback_reason or "")
