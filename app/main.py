import os
import torch
import whisper
import tempfile
import numpy as np
from fastapi import FastAPI, UploadFile, File, Body
import asyncio

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("small", device=device)

# ---- Request queue ----
request_queue = asyncio.Queue()

# ---- Background batch processor ----
BATCH_SIZE = 4  # Adjust based on VRAM and desired latency
PROCESS_INTERVAL = 0.05  # seconds


async def batch_processor():
    while True:
        batch = []
        futures = []

        # Wait for at least 1 request
        req = await request_queue.get()
        batch.append(req["audio"])
        futures.append(req["future"])

        # Try to collect more requests without blocking too long
        try:
            for _ in range(BATCH_SIZE - 1):
                req = request_queue.get_nowait()
                batch.append(req["audio"])
                futures.append(req["future"])
        except asyncio.QueueEmpty:
            pass

        # Process batch
        results = []
        with torch.inference_mode(), torch.autocast("cuda", enabled=(device == "cuda")):
            for audio_np in batch:
                result = model.transcribe(audio_np, fp16=(device == "cuda"))
                results.append(result["text"])

        # Return results to waiting futures
        for fut, text in zip(futures, results):
            fut.set_result(text)

        await asyncio.sleep(PROCESS_INTERVAL)


# Start background processor
asyncio.create_task(batch_processor())


# ---- Helper to enqueue requests ----
async def enqueue_request(audio_np):
    loop = asyncio.get_running_loop()
    fut = loop.create_future()
    await request_queue.put({"audio": audio_np, "future": fut})
    return await fut


# ---- Endpoints ----
@app.post("/transcribe")
async def transcribe_raw(
    audio: bytes = Body(..., media_type="application/octet-stream")
):
    if len(audio) % 2 != 0:
        return {"error": "Invalid PCM buffer length"}
    audio_np = np.frombuffer(audio, np.int16).astype(np.float32) / 32768.0
    text = await enqueue_request(audio_np)
    return {"text": text}


@app.post("/transcribe-file")
async def transcribe_file(file: UploadFile = File(...)):
    file_bytes = await file.read()
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        import soundfile as sf

        audio_np, _ = sf.read(tmp_path, dtype="float32")
        text = await enqueue_request(audio_np)
    finally:
        os.unlink(tmp_path)

    return {"filename": file.filename, "text": text}
