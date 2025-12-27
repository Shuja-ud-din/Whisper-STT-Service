import os
import torch
import whisper
import tempfile
import numpy as np
from fastapi import FastAPI, UploadFile, File, Body
from concurrent.futures import ThreadPoolExecutor
import asyncio

app = FastAPI()

# ---- Model Load (once per container) ----
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("small", device=device)

# ThreadPool for CPU-bound work (file I/O, numpy conversion)
executor = ThreadPoolExecutor(max_workers=4)


# ---- Health ----
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": device,
        "gpu_available": torch.cuda.is_available(),
    }


# ---- Transcribe RAW PCM audio ----
@app.post("/transcribe")
async def transcribe_raw(
    audio: bytes = Body(..., media_type="application/octet-stream")
):
    if len(audio) % 2 != 0:
        return {"error": "Invalid PCM buffer length"}

    # Convert bytes to numpy in executor
    loop = asyncio.get_running_loop()

    def process_audio(audio_bytes):
        return np.frombuffer(audio_bytes, np.int16).astype(np.float32) / 32768.0

    audio_np = await loop.run_in_executor(executor, process_audio, audio)

    # Run inference
    with torch.inference_mode(), torch.autocast("cuda", enabled=(device == "cuda")):
        result = model.transcribe(audio_np, fp16=(device == "cuda"))

    return {"text": result["text"]}


# ---- Transcribe audio file ----
@app.post("/transcribe-file")
async def transcribe_file(file: UploadFile = File(...)):
    loop = asyncio.get_running_loop()

    # Save file in executor
    def save_tmp_file(file_bytes):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file_bytes)
            return tmp.name

    file_bytes = await file.read()
    tmp_path = await loop.run_in_executor(executor, save_tmp_file, file_bytes)

    try:
        with torch.inference_mode(), torch.autocast("cuda", enabled=(device == "cuda")):
            result = model.transcribe(tmp_path, fp16=(device == "cuda"), language=None)
    finally:
        os.unlink(tmp_path)

    return {
        "filename": file.filename,
        "text": result["text"],
        "language": result.get("language"),
    }
