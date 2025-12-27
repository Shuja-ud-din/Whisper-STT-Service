from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel
import numpy as np
import io
from pydub import AudioSegment

app = FastAPI(title="Faster Whisper API")

# Load model once on GPU
model_size = "small"  # choose small/base/medium/large
model = WhisperModel(
    model_size, device="cuda", compute_type="float16"
)  # GPU float16 for speed


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/transcribe")
async def transcribe(raw_audio: bytes = File(...), sample_rate: int = 16000):
    try:
        audio = np.frombuffer(raw_audio, dtype=np.float32)
        segments, info = model.transcribe(audio, beam_size=5)
        text = " ".join([segment.text for segment in segments])
        return {"text": text}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/transcribe-file")
async def transcribe_file(file: UploadFile):
    try:
        contents = await file.read()
        audio_file = io.BytesIO(contents)
        audio_segment = AudioSegment.from_file(audio_file)
        audio_segment = audio_segment.set_channels(1).set_frame_rate(16000)
        audio_np = (
            np.array(audio_segment.get_array_of_samples()).astype(np.float32) / 32768.0
        )
        segments, info = model.transcribe(audio_np, beam_size=5)
        text = " ".join([segment.text for segment in segments])
        return {"text": text}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
