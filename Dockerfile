FROM nvidia/cuda:12.2.0-cudnn9-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# ---- System deps ----
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3.10-venv \
    ffmpeg git curl build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ---- Python deps ----
WORKDIR /app
COPY requirements.txt .

RUN pip3 install --upgrade pip && \
    pip3 install -r requirements.txt

# ---- Pre-download faster-whisper model (CPU-safe) ----
RUN python3 - <<EOF
from faster_whisper import WhisperModel
WhisperModel("small", device="cpu", compute_type="int8")
EOF

# ---- App ----
COPY app ./app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
