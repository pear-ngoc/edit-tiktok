$ErrorActionPreference = "Stop"

Write-Host "Đang thiết lập edit-tiktok..."

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    Write-Host "Không tìm thấy trình chạy Python 'py'. Vui lòng cài Python 3.11+ từ https://www.python.org/downloads/"
    exit 1
}

py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Host "Không tìm thấy FFmpeg trong PATH."
    Write-Host "Cài FFmpeg bằng winget:"
    Write-Host "  winget install Gyan.FFmpeg"
    Write-Host "Sau đó hãy mở lại PowerShell."
} else {
    ffmpeg -version | Select-Object -First 1
}

Write-Host "Chạy: .\.venv\Scripts\python.exe main.py doctor"
