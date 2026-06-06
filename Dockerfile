FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY main.py ./
COPY config.example.yaml ./config.example.yaml
COPY config.yaml ./config.yaml

RUN pip install --upgrade pip \
    && pip install .

CMD ["python", "main.py", "worker"]
