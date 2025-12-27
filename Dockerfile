FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3-dev ffmpeg build-essential pkg-config \
    libavdevice-dev libavfilter-dev libavformat-dev libavcodec-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip first separately
RUN pip3 install --no-cache-dir --upgrade pip

# Install dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# CRITICAL: Dynamically calculate the path to pip-installed NVIDIA libs
# and use it during the model download step to prevent crashes
RUN export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:$(python3 -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))')" && \
    python3 -c "from faster_whisper import WhisperModel; WhisperModel('small', device='cpu', compute_type='int8')"

COPY . .

# Final Environment Path for Runtime
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu"

EXPOSE 8000

# Start command calculates the NVIDIA lib paths again at runtime
CMD export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python3 -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))') && \
    uvicorn main:app --host 0.0.0.0 --port 8000