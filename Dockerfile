# Use CUDA 11.8 - The most stable version for Whisper + cuDNN 8
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3-dev ffmpeg build-essential pkg-config \
    libavdevice-dev libavfilter-dev libavformat-dev libavcodec-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# CRITICAL: Install the NVIDIA vendor libraries for Python
# This places the missing .so.8 files directly into your python site-packages
RUN pip3 install --no-cache-dir nvidia-cublas-cu11 nvidia-cudnn-cu11

# Set the LD_LIBRARY_PATH so the system can see the files we just installed
# This is the "Magic Fix" for the libcudnn_ops_infer.so.8 error
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cublas/lib"

# Pre-download model
RUN python3 -c "from faster_whisper import WhisperModel; WhisperModel('small', device='cpu', compute_type='int8')"

COPY . .

EXPOSE 8000

# Explicitly run ldconfig to refresh the library cache before starting
CMD ldconfig && uvicorn main:app --host 0.0.0.0 --port 8000