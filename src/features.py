import numpy as np
import librosa

SR = 22050 #sample rate per second. 2 seconds per chord = 44100 samples per chord
N_FFT = 2048 # Fast fourier transform. FFT / SR = window, each window of time contains 2048 samples and the time per window is 2048 / 22050 ≈ 0.093 seconds
HOP_LENGTH = 512 # how much we move forward between windows, with 2048 samples window, we overlap three windows at a time -> smoother frequency tracking.

def load_audio(path: str, seconds: float = 2.0, sr: int = SR) -> np.ndarray:
    y, _ = librosa.load(path, sr=sr, mono=True)

    target_len = int(sr * seconds)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    y = y / (np.max(np.abs(y)) + 1e-9)
    return y.astype(np.float32)

def extract_chroma(y: np.ndarray, sr: int = SR) -> np.ndarray:
    chroma = librosa.feature.chroma_stft(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    return chroma.astype(np.float32)

def chroma_stats(chroma: np.ndarray) -> np.ndarray:
    # chroma: (12, frames) -> features: (24,)
    mean = np.mean(chroma, axis=1)
    std = np.std(chroma, axis=1)
    feats = np.concatenate([mean, std], axis=0)
    return feats.astype(np.float32)

def extract_features_from_file(path: str) -> np.ndarray:
    y = load_audio(path)
    c = extract_chroma(y)
    return chroma_stats(c)

def extract_features(y: np.ndarray, sr: int = SR) -> np.ndarray:
    y = y.astype(np.float32)

    # normalize (както в load_audio)
    y = y / (np.max(np.abs(y)) + 1e-9)

    c = extract_chroma(y, sr=sr)
    return chroma_stats(c)