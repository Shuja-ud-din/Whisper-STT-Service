from faster_whisper import WhisperModel
import torch

MODEL_SIZE = "small"

# Load once per container
model = WhisperModel(
    MODEL_SIZE,
    device="cuda",
    compute_type="float16",  # critical for throughput
)


def transcribe(audio_array):
    segments, info = model.transcribe(
        audio_array,
        beam_size=1,
        vad_filter=True,
    )
    return " ".join([seg.text for seg in segments])
