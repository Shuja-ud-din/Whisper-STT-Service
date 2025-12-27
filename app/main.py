import io
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from faster_whisper import WhisperModel
import numpy as np

api = FastAPI()

# Initialize model globally for warm-start
# 'float16' is best for NVIDIA GPUs (3090, A100, etc.)
MODEL_SIZE = "small"
model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")


@api.get("/health")
async def health():
    return {"status": "healthy", "model": MODEL_SIZE, "device": "cuda"}


@api.post("/transcribe")
async def transcribe_raw(audio_bytes: bytes = File(...)):
    """Transcribes raw audio bytes."""
    try:
        # Faster-whisper can take a binary stream
        segments, info = model.transcribe(io.BytesIO(audio_bytes), beam_size=5)
        results = [{"start": s.start, "end": s.end, "text": s.text} for s in segments]
        return {"language": info.language, "transcription": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api.post("/transcribe-file")
async def transcribe_file(file: UploadFile = File(...)):
    """Transcribes an uploaded file (.wav, .mp3, etc.)"""
    try:
        # Read file into memory
        content = await file.read()
        segments, info = model.transcribe(io.BytesIO(content), beam_size=5)
        results = [{"start": s.start, "end": s.end, "text": s.text} for s in segments]
        return {
            "filename": file.filename,
            "language": info.language,
            "transcription": results,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
