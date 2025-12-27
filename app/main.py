import io
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from faster_whisper import WhisperModel

app = FastAPI()

MODEL_SIZE = "small"

# Load model on startup
try:
    # float16 is optimal for RunPod GPUs (3090, A100, etc.)
    model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")
except Exception as e:
    print(f"CRITICAL: Failed to load model on GPU: {e}")
    model = None


@app.get("/health")
async def health():
    if model is None:
        raise HTTPException(status_code=500, detail="Model failed to initialize")
    return {"status": "healthy", "model": MODEL_SIZE, "device": "cuda"}


@app.post("/transcribe")
async def transcribe_raw(audio_bytes: bytes = File(...)):
    try:
        segments, info = model.transcribe(io.BytesIO(audio_bytes), beam_size=5)
        results = [{"start": s.start, "end": s.end, "text": s.text} for s in segments]
        return {"language": info.language, "transcription": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/transcribe-file")
async def transcribe_file(file: UploadFile = File(...)):
    try:
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
