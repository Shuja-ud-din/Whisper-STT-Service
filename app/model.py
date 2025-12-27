from faster_whisper import WhisperModel
import torch

# Use GPU if available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load Whisper small model
model = WhisperModel(
    "small",
    device=DEVICE,
    compute_type="float16",  # Use float16 for faster GPU inference
)
