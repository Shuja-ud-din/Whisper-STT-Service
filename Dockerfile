# Use an available CUDA 12.1 image with cuDNN8 support
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# CUDA environment
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# install minimum base system dependencies and ffmpeg
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3.10-venv curl ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# upgrade pip
RUN python3 -m pip install --upgrade pip

WORKDIR /app

# copy dependencies file and install
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# copy app
COPY app/ /app/

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--loop", "asyncio", "--http", "h11"]
