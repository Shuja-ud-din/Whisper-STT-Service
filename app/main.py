import os
from fastapi import FastAPI, UploadFile
from concurrent.futures import ThreadPoolExecutor
import asyncio

from app.audio import pcm16_to_float32
from app.model import transcribe

app = FastAPI()

# Controls parallel GPU jobs
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "8"))
EXECUTOR = ThreadPoolExecutor(max_workers=MAX_WORKERS)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile):
    pcm_bytes = await file.read()
    audio = pcm16_to_float32(pcm_bytes)

    loop = asyncio.get_running_loop()
    text = await loop.run_in_executor(EXECUTOR, transcribe, audio)

    return {"text": text}
