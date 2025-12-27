# Use NVIDIA CUDA 12.2 with cuDNN 8 (Stable for ctranslate2 4.4.0)
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/root/.local/bin:$PATH"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Pre-download the model to the image (Production best practice)
# This avoids downloading ~3GB every time the Pod starts
RUN python3 -c "from faster_whisper import WhisperModel; WhisperModel('small', device='cpu', compute_type='int8')"

COPY . .

# RunPod uses port 8000 by default for HTTP traffic
EXPOSE 8000

CMD ["uvicorn", "main:api", "--host", "0.0.0.0", "--port", "8000"]