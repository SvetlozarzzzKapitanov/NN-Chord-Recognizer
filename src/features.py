import numpy as np
import librosa

# ================= Audio Configuration =================
# SR: Sample Rate - The number of audio samples captured per second.
# Common values are 22050 or 44100. Higher SR captures higher frequencies.
SR = 22050 

# N_FFT: Fast Fourier Transform size. 
# It determines the "window" of time analyzed at once to identify frequencies.
# A window of 2048 samples at 22050 Hz is ~93ms.
N_FFT = 2048 

# HOP_LENGTH: The number of samples to "step" between successive analysis windows.
# 512 samples at 22050 Hz is ~23ms. Using 512 with 2048 N_FFT means windows overlap by 75%,
# which provides smoother frequency tracking over time.
HOP_LENGTH = 512 

def load_audio(path: str, seconds: float = 2.0, sr: int = SR) -> np.ndarray:
    #Loads an audio file, resamples if necessary, ensures fixed duration, and normalizes volume.
    y, _ = librosa.load(path, sr=sr, mono=True)

    # Pad with silence or truncate to ensure the signal is exactly 'seconds' long
    target_len = int(sr * seconds)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    # Peak normalization: scale signal so the maximum absolute value is 1.0.
    # The 1e-9 prevents division by zero.
    y = y / (np.max(np.abs(y)) + 1e-9)
    return y.astype(np.float32)

def extract_chroma(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """
    Extracts Chroma features from audio signal.
    Chroma features map all frequencies into 12 bins corresponding to the 12 semi-tones
    of the musical scale (C, C#, D, D#, E, F, F#, G, G#, A, A#, B).
    This makes them highly effective for chord recognition regardless of the octave or instrument.
    """
    chroma = librosa.feature.chroma_stft(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    return chroma.astype(np.float32)

def chroma_stats(chroma: np.ndarray) -> np.ndarray:
    """
    Summarizes a time-varying Chroma matrix into a single fixed-size feature vector.
    
    Calculates the mean and standard deviation for each of the 12 semi-tone bins over time.
    Returns:
        A (24,) numpy array: [mean_C...mean_B, std_C...std_B]
    """
    # chroma shape is (12, frames)
    mean = np.mean(chroma, axis=1) # Average intensity of each note
    std = np.std(chroma, axis=1)   # Variation/stability of each note
    feats = np.concatenate([mean, std], axis=0)
    return feats.astype(np.float32)

def extract_features_from_file(path: str) -> np.ndarray:
    """Helper to load a file and extract its statistical chroma features."""
    y = load_audio(path)
    c = extract_chroma(y)
    return chroma_stats(c)

def extract_features(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """
    Extracts statistical chroma features directly from an in-memory audio array.
    Used for real-time processing chunks.
    """
    y = y.astype(np.float32)

    # Normalize volume to handle varying recording levels
    y = y / (np.max(np.abs(y)) + 1e-9)

    c = extract_chroma(y, sr=sr)
    return chroma_stats(c)