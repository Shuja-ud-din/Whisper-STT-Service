import os
import torch
import whisper
import numpy as np
from fastapi import FastAPI, UploadFile, File, Body
from multiprocessing import Process, Queue, Manager, set_start_method
import asyncio
import tempfile
import soundfile as sf

# -------------------------------
# Force 'spawn' start method for CUDA in multiprocess
# -------------------------------
set_start_method("spawn", force=True)

# -------------------------------
# Shared objects
# -------------------------------
NUM_WORKERS = 5
request_queue = Queue()
manager = Manager()
results_dict = manager.dict()  # Store results with unique request IDs

app = FastAPI(title="Multi-Process Whisper Server")
device = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------------
# Worker function
# -------------------------------
def worker_main(worker_id, request_queue, results_dict):
    print(f"[Worker {worker_id}] Starting...")
    # Each worker loads its own model instance on the GPU
    model = whisper.load_model("small", device=device)

    while True:
        try:
            request_id, audio_np = request_queue.get()
            if audio_np is None:  # shutdown signal
                print(f"[Worker {worker_id}] Shutting down")
                break

            # Inference
            with torch.inference_mode(), torch.autocast(
                "cuda", enabled=(device == "cuda")
            ):
                result = model.transcribe(
                    audio_np, fp16=(device == "cuda"), beam_size=3
                )
            results_dict[request_id] = result["text"]

        except Exception as e:
            results_dict[request_id] = f"ERROR: {str(e)}"


# -------------------------------
# Start worker processes
# -------------------------------
workers = []
for i in range(NUM_WORKERS):
    p = Process(target=worker_main, args=(i, request_queue, results_dict), daemon=True)
    p.start()
    workers.append(p)

# -------------------------------
# Helper to enqueue request
# -------------------------------
request_counter = 0


async def enqueue_request(audio_np):
    global request_counter
    loop = asyncio.get_running_loop()
    request_id = f"req_{request_counter}"
    request_counter += 1

    request_queue.put((request_id, audio_np))

    # Wait until worker sets the result
    while True:
        await asyncio.sleep(0.01)
        if request_id in results_dict and isinstance(results_dict[request_id], str):
            text = results_dict.pop(request_id)
            return text


# -------------------------------
# FastAPI endpoints
# -------------------------------
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": device,
        "gpu_available": torch.cuda.is_available(),
    }


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
        audio_np, _ = sf.read(tmp_path, dtype="float32")
        text = await enqueue_request(audio_np)
    finally:
        os.unlink(tmp_path)

    return {"filename": file.filename, "text": text}
