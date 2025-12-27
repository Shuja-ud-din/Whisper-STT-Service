# Use CUDA 12.3 with CUDNN 9 to prevent "libcudnn_ops.so.9" errors
FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install system dependencies including FFmpeg and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-dev \
    ffmpeg \
    build-essential \
    pkg-config \
    libavdevice-dev \
    libavfilter-dev \
    libavformat-dev \
    libavcodec-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Core environment variables to help Python find CUDA/CUDNN
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu"

# Upgrade pip and install dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# Bake the model into the image (Fast pod startup)
RUN python3 -c "from faster_whisper import WhisperModel; WhisperModel('small', device='cpu', compute_type='int8')"

COPY . .

# Match CMD to the FastAPI variable name "app"
# We use --workers 1 for Whisper as it is highly GPU-intensive
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]