FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3-dev ffmpeg build-essential pkg-config \
    libavdevice-dev libavfilter-dev libavformat-dev libavcodec-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Install dependencies and then force set the library paths
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# This command finds the pip-installed nvidia libraries and exports them
# It's the only way to guarantee the .so.8 files are found regardless of where pip puts them
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu"

# Pre-download the model
RUN python3 -c "from faster_whisper import WhisperModel; WhisperModel('small', device='cpu', compute_type='int8')"

COPY . .

# IMPORTANT: We use a shell-form CMD so we can expand the LD_LIBRARY_PATH 
# with the location of the pip-installed nvidia libraries at runtime
CMD export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python3 -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))') && \
    uvicorn main:app --host 0.0.0.0 --port 8000