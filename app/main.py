import os
import tempfile
import numpy as np
from fastapi import FastAPI, UploadFile, File, Body
from faster_whisper import WhisperModel

app = FastAPI()

# ---- Model ----
model = WhisperModel(
    "small",
    device="cuda",  # auto-falls back to CPU if no GPU
    compute_type="float16",  # use "int8_float16" if VRAM constrained
    cpu_threads=4,
    num_workers=4,  # enables parallel decoding
)


# ---- Health ----
@app.get("/health")
def health():
    return {"status": "ok"}


# ---- Raw PCM ----
@app.post("/transcribe")
def transcribe_raw(
    audio: bytes = Body(..., media_type="application/octet-stream"),
    sample_rate: int = 16000,
):
    audio_np = np.frombuffer(audio, np.int16).astype(np.float32) / 32768.0

    segments, info = model.transcribe(audio_np, beam_size=5, vad_filter=False)

    text = "".join(seg.text for seg in segments)
    return {"text": text.strip(), "language": info.language}


# ---- File Upload ----
@app.post("/transcribe-file")
async def transcribe_file(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        segments, info = model.transcribe(tmp_path, beam_size=5, vad_filter=False)
    finally:
        os.unlink(tmp_path)

    text = "".join(seg.text for seg in segments)
    return {"filename": file.filename, "text": text.strip(), "language": info.language}
