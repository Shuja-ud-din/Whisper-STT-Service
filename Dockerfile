FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# ---- System deps ----
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    curl \
    ca-certificates \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ---- CUDA + cuDNN ----
RUN curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -o cuda-keyring.deb && \
    dpkg -i cuda-keyring.deb && \
    apt-get update && \
    apt-get install -y \
        cuda-cudart-12-1 \
        libcudnn9 \
        libcudnn9-dev \
    && rm -rf /var/lib/apt/lists/* cuda-keyring.deb

ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64
ENV PATH=/usr/local/cuda/bin:${PATH}

# ---- Python deps ----
WORKDIR /app
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# ---- Pre-download model (CPU-safe) ----
RUN python - <<EOF
from faster_whisper import WhisperModel
WhisperModel("small", device="cpu", compute_type="int8")
EOF

# ---- App ----
COPY app ./app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]