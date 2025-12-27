import os
import torch
import whisper
import tempfile
import numpy as np
from fastapi import FastAPI, UploadFile, File, Body
import asyncio
import soundfile as sf

app = FastAPI(title="Whisper Async Server")

# -------------------------------
# Model Load (once per container)
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("small", device=device)

# -------------------------------
# Request queue for batching
# -------------------------------
request_queue = asyncio.Queue()
BATCH_SIZE = 4  # How many requests to process in one loop
PROCESS_INTERVAL = 0.05  # seconds, time to wait between batches


# -------------------------------
# Background GPU worker
# -------------------------------
async def gpu_worker():
    while True:
        batch = []
        futures = []

        # Wait for at least one request
        req = await request_queue.get()
        batch.append(req["audio"])
        futures.append(req["future"])

        # Try to collect more requests without waiting
        try:
            for _ in range(BATCH_SIZE - 1):
                req = request_queue.get_nowait()
                batch.append(req["audio"])
                futures.append(req["future"])
        except asyncio.QueueEmpty:
            pass

        # Process batch sequentially under inference_mode + autocast
        results = []
        with torch.inference_mode(), torch.autocast("cuda", enabled=(device == "cuda")):
            for audio_np in batch:
                # Reduce beam_size to 3 for speed
                result = model.transcribe(
                    audio_np, fp16=(device == "cuda"), beam_size=3
                )
                results.append(result["text"])

        # Return results
        for fut, text in zip(futures, results):
            fut.set_result(text)

        await asyncio.sleep(PROCESS_INTERVAL)


# Start background GPU worker
asyncio.create_task(gpu_worker())


# -------------------------------
# Helper function to enqueue request
# -------------------------------
async def enqueue_request(audio_np: np.ndarray):
    loop = asyncio.get_running_loop()
    fut = loop.create_future()
    await request_queue.put({"audio": audio_np, "future": fut})
    return await fut


# -------------------------------
# Health endpoint
# -------------------------------
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": device,
        "gpu_available": torch.cuda.is_available(),
    }


# -------------------------------
# Transcribe raw PCM bytes
# -------------------------------
@app.post("/transcribe")
async def transcribe_raw(
    audio: bytes = Body(..., media_type="application/octet-stream")
):
    if len(audio) % 2 != 0:
        return {"error": "Invalid PCM buffer length"}

    # Convert PCM to float32 np array
    audio_np = np.frombuffer(audio, np.int16).astype(np.float32) / 32768.0

    text = await enqueue_request(audio_np)
    return {"text": text}


# -------------------------------
# Transcribe audio file
# -------------------------------
@app.post("/transcribe-file")
async def transcribe_file(file: UploadFile = File(...)):
    file_bytes = await file.read()
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        # Load audio as float32
        audio_np, _ = sf.read(tmp_path, dtype="float32")
        text = await enqueue_request(audio_np)
    finally:
        os.unlink(tmp_path)

    return {"filename": file.filename, "text": text}
