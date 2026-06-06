# edit-tiktok

`edit-tiktok` là công cụ xử lý video hàng loạt bằng FFmpeg. Ứng dụng đọc video trong `input/`, chỉnh sửa theo `config.yaml`, rồi xuất kết quả vào `output/`.

Ứng dụng dành cho video bạn sở hữu hoặc có quyền chỉnh sửa. Các tính năng như xoá metadata, LUT, crop, nền blur, trộn nhạc và phụ đề được triển khai như tính năng biên tập, riêng tư và nâng chất lượng nội dung.

## Hệ điều hành hỗ trợ

- Windows 10/11, ưu tiên máy có NVIDIA GPU và FFmpeg hỗ trợ NVENC.
- macOS Apple Silicon, ưu tiên FFmpeg từ Homebrew và VideoToolbox.
- Linux hoặc máy không có GPU, tự fallback về CPU `libx264`.

## Cài đặt nhanh

Yêu cầu Python `>=3.11` và FFmpeg/FFprobe.

Windows PowerShell:

```powershell
scripts\install_windows.ps1
py -3.11 main.py doctor
py -3.11 main.py
```

macOS:

```bash
chmod +x scripts/install_macos.sh
./scripts/install_macos.sh
python3 main.py doctor
python3 main.py
```

## Lệnh chính

Chạy xử lý mặc định:

```bash
python main.py
```

Lệnh trên sẽ tạo thư mục cần thiết, tạo `config.yaml` nếu chưa có, quét `input/`, xử lý video và ghi log vào `logs/video_processing.log`.

Các lệnh khác:

```bash
python main.py init
python main.py doctor
python main.py list-luts
python main.py preflight
python main.py process
python main.py wizard
python main.py configs list
python main.py configs show vertical_blur
python main.py list-configs
python main.py show-config vertical_blur
python main.py use-config vertical_blur
```

`python main.py process` chỉ là alias rõ ràng cho workflow mặc định.

Bạn cũng có thể nạp cấu hình đã lưu trực tiếp:

```bash
python main.py --config-profile vertical_blur
python main.py process --config-profile tiktok_burn_caption_vi
python main.py process --config-profile vertical_blur --input input2 --output output2
```

## Cách xử lý video

1. Chép video vào `input/`.
2. Sửa `config.yaml` nếu cần.
3. Chạy:

```bash
python main.py
```

Ví dụ override nhanh:

```bash
python main.py process --aspect 9:16 --mode blur --encoder auto --preset balanced
```

## LUT, ambient và BGM

- Chép file LUT `.cube` vào `assets/luts/`.
- Xem LUT có sẵn:

```bash
python main.py list-luts
```

- Chép file font `.ttf` hoặc `.otf` vào `assets/font/`.
- Xem font caption có sẵn:

```bash
python main.py list-fonts
```

- Chọn LUT trong `config.yaml`:

```yaml
color:
  lut_enabled: true
  max_luts: 3
  selected_luts:
    - "example.cube"
```

- Chọn LUT nhanh bằng CLI:

```bash
python main.py process --lut example.cube
python main.py process --lut first.cube --lut second.cube
python main.py process --no-lut
```

- `python main.py wizard` sẽ cho chọn LUT tương tác từ danh sách `assets/luts/`.

## Preflight trước khi xử lý

Trước khi chạy batch, app sẽ kiểm tra rất nhẹ xem thư mục đầu vào có ít nhất một video hợp lệ hay không.
Chỉ các đuôi sau được tính:

- `.mp4`
- `.mov`
- `.mkv`
- `.avi`
- `.webm`

Nếu `processing.recursive: true`, app sẽ quét cả thư mục con. Nếu không tìm thấy video nào, workflow sẽ dừng ngay với thông báo thân thiện:

```text
No input videos found. Please add videos to the input/ folder and run again.
```

Chạy riêng preflight:

```bash
python main.py preflight
```

Đây chỉ là bước kiểm tra đầu vào rất nhẹ, không chạy `ffprobe`, không kiểm tra codec, audio, LUT hay phụ đề. Bạn có thể bỏ qua preflight chỉ khi thật sự cần:

```bash
python main.py process --skip-preflight
```

- Chép âm thanh môi trường vào `assets/ambient/`.
- Chép nhạc nền vào `assets/bgm/`.
- Bật trong `config.yaml`:

```yaml
audio:
  ambient_enabled: true
  bgm_enabled: true
```

Nếu file âm thanh ngắn hơn video, FFmpeg sẽ loop an toàn trong quá trình xử lý.

## Phụ đề tự động

Phần phụ đề được tạo bằng `faster-whisper` và chạy trực tiếp trong luồng xử lý video bình thường.

Mặc định cấu hình phụ đề:

```yaml
subtitles:
  enabled: true
  backend: faster-whisper
  model_size: medium
  language: auto
  output_srt: true
  output_vtt: false
  burn_in: false
  burn_language: auto
  device: auto
  compute_type: auto
  output_dir: output/subtitles
  word_timestamps: true

formatting:
  max_chars_per_line: 20
  max_lines: 2
  max_chars_per_cue: 40
  max_words_per_cue: 7
  min_duration: 0.7
  max_duration: 2.6
  pause_threshold: 0.45
  caption_renderer: rounded_box
  caption_position: bottom
  caption_vertical_offset: 0
  caption_font_name: Arial
  caption_font_file: BeVietnamPro-Bold.ttf
  caption_font_fallback: Arial
  caption_font_size: 54
  caption_text_color: "#111111"
  caption_text_opacity: 1.0
  caption_background_color: "#FFFFFF"
  caption_background_opacity: 0.95
  caption_padding_x: 28
  caption_padding_y: 14
  caption_border_radius: 18
  caption_outline_color: "#000000"
  caption_outline_opacity: 0.0
  caption_outline: 0
  caption_shadow_enabled: true
  caption_shadow_color: "#000000"
  caption_shadow_opacity: 0.25
  caption_shadow_offset_x: 0
  caption_shadow_offset_y: 4
  caption_shadow_blur: 6
  caption_margin_v: 140
  caption_max_width_percent: 82
  caption_box_enabled: true
```

Giải thích nhanh các tùy chọn style:

- `caption_renderer`: `rounded_box` để vẽ caption bo góc bằng PNG overlay, hoặc `ass` để dùng renderer ASS dự phòng.
- `caption_font_file`: tên file `.ttf` hoặc `.otf` nằm trong `assets/font/`. Nếu có file này, app sẽ ưu tiên dùng file đó thay vì chỉ dựa vào tên font.
- `caption_font_fallback`: font dự phòng khi file trong `assets/font/` không load được.
- `caption_font_name`: tên font hiển thị/log và dùng làm fallback cũ nếu cần.
- `caption_font_size`: cỡ chữ tham chiếu cho output dọc 1080x1920. App sẽ tự scale theo độ phân giải thật.
- `caption_text_color`: màu chữ dạng `#RRGGBB`.
- `caption_text_opacity`, `caption_outline_opacity`, `caption_background_opacity`, `caption_shadow_opacity`: độ trong suốt từ `0.0` đến `1.0` (`0.0` = trong suốt, `1.0` = đậm nhất). Không bị đảo ngược nữa.
- `caption_background_color`: màu nền hộp chữ. Với renderer `rounded_box`, đây là màu hộp bo góc thật.
- `caption_padding_x`, `caption_padding_y`: khoảng đệm bên trong hộp.
- `caption_border_radius`: độ bo góc của hộp.
- `caption_shadow_enabled`, `caption_shadow_color`, `caption_shadow_offset_x`, `caption_shadow_offset_y`, `caption_shadow_blur`: cấu hình shadow phía dưới hộp.
- `caption_outline`: độ dày outline nếu bạn dùng fallback ASS hoặc muốn stroke chữ.
- `caption_margin_v`: khoảng cách từ mép dưới lên caption.
- `caption_vertical_offset`: đẩy caption lên hoặc xuống thêm so với `caption_margin_v`.
- `caption_max_width_percent`: giới hạn chiều ngang hộp caption theo phần trăm chiều rộng video.
- `caption_box_enabled`: bật/tắt nền hộp.

Nếu bạn muốn caption thấp hơn, giảm `caption_margin_v` hoặc `caption_vertical_offset`.
Nếu muốn caption cao hơn, tăng một trong hai giá trị đó.

### Font caption

Bạn có thể đặt các file `.ttf` hoặc `.otf` vào `assets/font/`. Ví dụ:

```text
assets/font/
├── BeVietnamPro-Bold.ttf
├── Montserrat-SemiBold.ttf
└── .gitkeep
```

Để xem danh sách font khả dụng:

```bash
python main.py list-fonts
```

Khi muốn đổi font, chỉ cần sửa:

```yaml
formatting:
  caption_font_file: "BeVietnamPro-Bold.ttf"
```

Nếu file font không tồn tại hoặc không đọc được, app sẽ tự log cảnh báo và dùng font dự phòng thay vì làm hỏng job.

Ví dụ bật phụ đề từ CLI:

```bash
python main.py process --subtitles
python main.py process --subtitles --burn-captions --subtitle-language vi
python main.py process --subtitles --whisper-model medium
```

Nếu bật `burn-captions`, app sẽ tạo video mới dạng `_burned.mp4` và vẫn giữ nguyên bản video đã xử lý không burn.

Nếu video không có audio, phần tạo phụ đề sẽ được bỏ qua an toàn và sẽ chỉ ghi log cảnh báo.

## Cắt đoạn tự động

Mặc định app sẽ chia video thành các đoạn ngắn ngẫu nhiên khoảng `3-5` giây, rồi áp dụng biến đổi xen kẽ như zoom và lật hình theo từng đoạn trước khi ghép lại.

Nếu bạn muốn thử chế độ khác trong tương lai, phần `segment_mode` đã được tách riêng trong kiến trúc để dễ mở rộng thêm phát hiện cảnh tự động.

## Cấu hình quan trọng

- `processing.input_dir`: thư mục video đầu vào.
- `processing.output_dir`: thư mục xuất.
- `processing.delete_source`: mặc định `false`; chỉ xoá video gốc khi bạn tự đặt `true`.
- `video.aspect_ratio`: ví dụ `9:16`, `16:9`, `1:1`, hoặc custom `4:3`.
- `video.mode`: `crop`, `blur`, `original`, hoặc `target`.
- `video.speed`: mặc định `1.1`.
- `encoder.backend`: `auto`, `cpu_h264`, `cpu_h265`, `nvidia_h264`, `nvidia_h265`, `videotoolbox_h264`, `videotoolbox_h265`.
- `color.selected_luts`: danh sách LUT `.cube` sẽ áp dụng theo thứ tự.
- `color.auto_select_luts`: nếu `true`, app sẽ tự chọn tối đa `max_luts` LUT đầu tiên trong `assets/luts/` khi chưa chọn gì.
- `metadata.mode`: `keep`, `remove`, hoặc `custom`.

## Saved configs / profile cấu hình

Sau khi chạy `python main.py wizard`, app sẽ hỏi có muốn lưu cấu hình hay không. Nếu chọn có, file YAML sẽ được lưu trong `configs/`.

Ví dụ:

```bash
python main.py configs list
python main.py configs show vertical_blur
python main.py --config-profile vertical_blur
```

Thứ tự ưu tiên cấu hình khi chạy app là:

1. Cấu hình mặc định trong code.
2. `config.yaml` ở root.
3. Profile đã lưu trong `configs/<name>.yaml`.
4. Override từ CLI.

Nếu profile không tồn tại, app sẽ báo lỗi thân thiện và liệt kê các profile đang có.

## Chạy hàng đợi liên tục, Telegram và Docker

### Chạy theo dõi `input/` liên tục

```bash
python main.py watch
```

Chế độ này dùng chung queue worker để tự phát hiện video mới trong `input/` và xử lý liên tục.

### Chạy Telegram bot

```bash
python main.py telegram
```

Bot chạy theo kiểu long polling, nhận link TikTok qua tin nhắn, tải video bằng Revid API rồi trả file đã render về đúng `chat_id` đã gửi link.

Bạn có thể giới hạn chat được phép bằng:

```yaml
telegram:
  enabled: true
  bot_token: "..."
  allowed_chat_ids:
    - 123456789
    - 987654321
  allow_all_chats_if_empty: false
  max_video_send_mb: 49
```

Nếu `allowed_chat_ids` để trống và `allow_all_chats_if_empty: true`, bot sẽ nhận link từ mọi chat. Mỗi link là một job riêng và output luôn trả về đúng chat ban đầu.

Bạn có thể đặt `telegram.bot_token` và `revid_api.api_key` trực tiếp trong `config.yaml` hoặc dùng biến môi trường `TELEGRAM_BOT_TOKEN` và `REVID_API_KEY`. Nếu biến môi trường có giá trị, app sẽ ưu tiên dùng chúng khi config đang để trống.

App cũng tự đọc file `.env` ở root khi khởi động, nên nếu bạn đặt key trong `.env` thì `python main.py worker` sẽ nhận luôn mà không cần `source .env` trước.

Nếu bạn muốn gửi file lớn hơn giới hạn 50 MB của Bot API công khai, hãy dùng Bot API server local trong Docker. Khi đó bạn có thể cấu hình thêm:

```yaml
telegram:
  api_base_url: "http://telegram-bot-api:8081/bot"
  api_file_url: "http://telegram-bot-api:8081/file/bot"
  local_mode: true
```

Ba field này giúp ứng dụng trỏ sang service `telegram-bot-api` trong `docker-compose.yml`. Ở chế độ local, bot có thể upload file lớn hơn so với `api.telegram.org`, miễn là server local và tài nguyên máy cho phép. Mình đã mount `./output:/app/output:ro` vào service này để nó đọc file render trực tiếp từ volume dùng chung, tối ưu hơn cho video lớn.
Service `telegram-bot-api` còn cần `TELEGRAM_API_ID` và `TELEGRAM_API_HASH` lấy từ `https://my.telegram.org`.

### Chạy worker nền đầy đủ

```bash
python main.py worker
```

Có thể ghi đè nhanh:

```bash
python main.py worker --telegram
python main.py worker --no-telegram
python main.py worker --watch-input
python main.py worker --no-watch-input
```

### Xoá dữ liệu sinh ra và input

```bash
python main.py clear
python main.py clear input
python main.py clear generated
python main.py clear --yes
python main.py clear --dry-run
```

Mặc định `clear` sẽ xoá cả `input/` lẫn các dữ liệu sinh ra như `output/`, `temp/`, `logs/`, `data/`, `failed/`, `completed/` và các profile đã lưu trong `configs/`. Lệnh sẽ hỏi xác nhận trước khi xoá, trừ khi bạn truyền `--yes`. Nếu chỉ muốn xem trước danh sách cần xoá, dùng `--dry-run`.

### Cấu hình queue

```yaml
queue:
  enabled: true
  max_workers: 5
  watch_input: true
  scan_interval_seconds: 3
  stable_file_check_seconds: 2
```

Lưu ý: `5` job FFmpeg song song có thể khá nặng. Máy yếu hơn nên giảm xuống `1-2`. Bạn hoàn toàn có thể chỉnh `queue.max_workers` theo CPU/GPU của máy.

### Docker

Build và chạy:

```bash
docker compose up -d --build
```

File `.env.example` có sẵn:

```env
TELEGRAM_BOT_TOKEN=
TELEGRAM_API_ID=
TELEGRAM_API_HASH=
TELEGRAM_BOT_API_BASE_URL=
TELEGRAM_BOT_API_FILE_URL=
TELEGRAM_BOT_API_LOCAL_MODE=
REVID_API_KEY=
```

Container mặc định chạy:

```bash
python main.py worker
```

Các volume được mount:

- `./input:/app/input`
- `./output:/app/output`
- `./assets:/app/assets`
- `./configs:/app/configs`
- `./logs:/app/logs`
- `./data:/app/data`
- `./config.yaml:/app/config.yaml`

Compose hiện có thêm service `telegram-bot-api` chạy từ image `ghcr.io/bots-house/docker-telegram-bot-api`. Service này dùng để self-host Telegram Bot API khi bạn muốn upload file lớn hơn giới hạn 50 MB của API công khai.

Để cache model `faster-whisper` không bị tải lại sau khi recreate container, `edit-tiktok` đã mount thêm:

- `./data/huggingface:/root/.cache/huggingface`

Vì vậy model sẽ được giữ lại trên máy host trong `data/huggingface/` và các lần `docker compose down` / `up` sau sẽ dùng lại cache cũ thay vì tải lại từ đầu. Nếu muốn tùy biến vị trí cache, bạn có thể đặt thêm:

```bash
HF_HOME=/root/.cache/huggingface
HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface/hub
```

Trong `docker-compose.yml`, service `edit-tiktok` đã tự gán sẵn các biến này để phù hợp với volume mount ở trên.

### Bật GPU NVIDIA trong Docker

Nếu máy host có NVIDIA GPU và Docker Desktop đang chạy trên WSL2, service `edit-tiktok` đã được cấu hình để xin GPU bằng:

```yaml
gpus: all
environment:
  NVIDIA_VISIBLE_DEVICES: all
  NVIDIA_DRIVER_CAPABILITIES: compute,video,utility
```

Đây là phần cần có để container thực sự nhìn thấy `libcuda.so.1` và dùng được `h264_nvenc` hoặc faster-whisper CUDA. Chỉ thấy `h264_nvenc` trong danh sách encoder của FFmpeg chưa đủ, vì đó mới chỉ là hỗ trợ build-time. Bạn có thể kiểm tra runtime thật bằng:

```bash
docker compose run --rm edit-tiktok nvidia-smi
```

Nếu lệnh này không thấy GPU, hãy kiểm tra:

- Docker Desktop đang bật WSL2 backend
- Driver NVIDIA trên Windows có hỗ trợ WSL2 GPU
- `wsl --update` đã chạy
- máy có GPU NVIDIA thật

## Tăng tốc GPU

Windows NVIDIA:

- Cài driver NVIDIA mới.
- Cài FFmpeg build có NVENC.
- Chạy `python main.py doctor`.
- Nếu thấy `h264_nvenc`, backend `auto` sẽ ưu tiên NVIDIA.

macOS Apple Silicon:

- Cài FFmpeg bằng Homebrew:

```bash
brew install ffmpeg
```

- Chạy `python main.py doctor`.
- Nếu thấy `h264_videotoolbox`, backend `auto` sẽ ưu tiên VideoToolbox.

Nếu GPU encoder không có, ứng dụng tự fallback về CPU `libx264`.

## Xử lý lỗi thường gặp

- `FFmpeg: missing`: cài FFmpeg và đảm bảo `ffmpeg` nằm trong `PATH`.
- Python báo yêu cầu `>=3.11`: cài Python 3.11 hoặc mới hơn.
- Không thấy video: kiểm tra file trong `input/`, hỗ trợ `.mp4`, `.mov`, `.mkv`, `.avi`, `.webm`.
- LUT lỗi: kiểm tra file `.cube` có tồn tại trong `assets/luts/`.
- Video không có audio: ứng dụng vẫn xuất video-only hợp lệ.

## Ghi chú migration

Bản này thay thế kiến trúc cũ kiểu `Main.py`, `app.py`, `video_processor.py`, `lut_processor.py`, `audio_enhancer.py`, `metadata_handler.py`, `utils.py` bằng cấu trúc rõ trách nhiệm hơn:

- `main.py` ở root chỉ launcher.
- `src/cli.py` xử lý CLI.
- `src/app.py` điều phối workflow.
- `src/processing/` xử lý video/audio/LUT/metadata.
- `src/ffmpeg_tools/` chứa probe, runner, encoder và filter FFmpeg.

Không có package `edit_kenh` hoặc `edit_tiktok`.
