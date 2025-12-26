import os
from fastapi import FastAPI, UploadFile, HTTPException
from concurrent.futures import ThreadPoolExecutor
import asyncio
import io
import soundfile as sf
import numpy as np

from app.audio import pcm16_to_float32
from app.model import transcribe  # your Faster-Whisper wrapper

app = FastAPI()

# Controls parallel GPU jobs
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "8"))
EXECUTOR = ThreadPoolExecutor(max_workers=MAX_WORKERS)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/transcribe")
async def transcribe_pcm(file: UploadFile):
    """Raw PCM16 bytes"""
    try:
        pcm_bytes = await file.read()
        audio = pcm16_to_float32(pcm_bytes)

        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(EXECUTOR, transcribe, audio)
        return {"text": text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/transcribe-file")
async def transcribe_file(file: UploadFile):
    """Standard audio files (.wav, .mp3, etc)"""
    try:
        audio_bytes = await file.read()
        # Decode audio using soundfile
        f = io.BytesIO(audio_bytes)
        data, samplerate = sf.read(f, dtype="float32")

        # If stereo, convert to mono
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)

        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(EXECUTOR, transcribe, data)
        return {"text": text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
