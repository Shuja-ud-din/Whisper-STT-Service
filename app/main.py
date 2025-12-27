from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import numpy as np
import io
from model import model
import torch
import asyncio
import ffmpeg

app = FastAPI(title="Whisper-vLLM STT API")


# ---------------- Health Check ----------------
@app.get("/health")
async def health():
    return {"status": "ok", "device": str(model.device)}


# ---------------- Transcribe raw audio ----------------
class TranscribeRequest(BaseModel):
    audio: list  # PCM16 list or base64-decoded PCM bytes
    sample_rate: int = 16000


@app.post("/transcribe")
async def transcribe(req: TranscribeRequest):
    try:
        audio_np = np.array(req.audio, dtype=np.float32)
        segments, _ = model.transcribe(audio_np, beam_size=5)
        text = " ".join([s.text for s in segments])
        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------- Transcribe file upload ----------------
@app.post("/transcribe-file")
async def transcribe_file(file: UploadFile = File(...)):
    try:
        # Save to temp
        tmp_bytes = await file.read()
        in_memory_file = io.BytesIO(tmp_bytes)

        # Convert to PCM16 using ffmpeg
        out, _ = (
            ffmpeg.input("pipe:0")
            .output("pipe:1", format="s16le", acodec="pcm_s16le", ac=1, ar=16000)
            .run(input=in_memory_file.read(), capture_stdout=True, capture_stderr=True)
        )
        audio_np = np.frombuffer(out, np.int16).astype(np.float32) / 32768.0
        segments, _ = model.transcribe(audio_np, beam_size=5)
        text = " ".join([s.text for s in segments])
        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
