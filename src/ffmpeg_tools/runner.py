from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass

from utils.runtime_logging import redact_command, redact_text

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class CommandResult:
    args: list[str]
    returncode: int
    stdout: str
    stderr: str


class FFmpegError(RuntimeError):
    def __init__(self, message: str, result: CommandResult) -> None:
        super().__init__(message)
        self.result = result


def run_command(
    args: list[str],
    *,
    debug: bool = False,
    check: bool = True,
    stderr_tail_lines: int = 40,
) -> CommandResult:
    if debug:
        LOGGER.debug("Đang chạy lệnh: %s", redact_command(args))
    completed = subprocess.run(
        args,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    result = CommandResult(
        args=args,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
    if check and result.returncode != 0:
        LOGGER.error("Lệnh thất bại (%s): %s", result.returncode, redact_command(args))
        stderr_tail = _tail_lines(result.stderr, stderr_tail_lines)
        if stderr_tail:
            LOGGER.error("stderr tail:\n%s", stderr_tail)
        raise FFmpegError("Lệnh FFmpeg thất bại", result)
    return result


def _tail_lines(text: str, count: int) -> str:
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    tail = lines[-max(1, count) :]
    return "\n".join(redact_text(line) for line in tail)
