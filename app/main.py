import os
import torch
import whisper
import tempfile
import numpy as np
from fastapi import FastAPI, UploadFile, File, Body
from fastapi.responses import JSONResponse

app = FastAPI()

# ---- Model Load (once per container) ----
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("small", device=device)


# ---- Health ----
@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": device,
        "gpu_available": torch.cuda.is_available(),
    }


# ---- Transcribe RAW PCM audio ----
@app.post("/transcribe")
def transcribe_raw(audio: bytes = Body(..., media_type="application/octet-stream")):
    if len(audio) % 2 != 0:
        return {"error": "Invalid PCM buffer length"}

    audio_np = np.frombuffer(audio, np.int16).astype(np.float32) / 32768.0

    result = model.transcribe(audio_np, fp16=(device == "cuda"))
    return {"text": result["text"]}


# ---- Transcribe audio file ----
@app.post("/transcribe-file")
async def transcribe_file(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = model.transcribe(tmp_path, fp16=(device == "cuda"), language=None)
    finally:
        os.unlink(tmp_path)

    return {
        "filename": file.filename,
        "text": result["text"],
        "language": result.get("language"),
    }
