import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from faster_whisper import WhisperModel

# Use "app" as the variable name
app = FastAPI()

MODEL_SIZE = "small"
# Pre-loading the model on the GPU
model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")


@app.get("/health")
async def health():
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
