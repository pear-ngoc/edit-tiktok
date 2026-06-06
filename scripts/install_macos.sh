#!/usr/bin/env bash
set -euo pipefail

echo "Đang thiết lập edit-tiktok..."

if ! command -v python3 >/dev/null 2>&1; then
  echo "Không tìm thấy python3. Vui lòng cài Python 3.11+ trước."
  exit 1
fi

python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -e ".[dev]"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "Không tìm thấy FFmpeg."
  if command -v brew >/dev/null 2>&1; then
    echo "Cài bằng lệnh: brew install ffmpeg"
  else
    echo "Cài Homebrew từ https://brew.sh rồi chạy: brew install ffmpeg"
  fi
else
  ffmpeg -version | head -n 1
fi

echo "Chạy: ./.venv/bin/python main.py doctor"
