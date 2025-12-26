import numpy as np


def pcm16_to_float32(pcm_bytes: bytes):
    audio = np.frombuffer(pcm_bytes, dtype=np.int16)
    audio = audio.astype(np.float32) / 32768.0
    return audio
