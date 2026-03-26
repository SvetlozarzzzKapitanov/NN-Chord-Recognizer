import os
import time
from collections import deque

import numpy as np
import sounddevice as sd
import torch

from features import extract_features

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "chord_mlp.pt")


class MLP(torch.nn.Module):
    def __init__(self, in_dim=24, num_classes=5):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def load_model(model_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    payload = torch.load(model_path, map_location=device, weights_only=False)

    chords = payload["chords"]
    mu = payload["mu"]
    sigma = payload["sigma"]

    model = MLP(in_dim=payload["in_dim"], num_classes=len(chords)).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    return model, chords, mu, sigma, device


def softmax_numpy(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


def predict_probs(model, features: np.ndarray, mu: np.ndarray, sigma: np.ndarray, device: str) -> np.ndarray:
    x = (features[None, :] - mu) / sigma
    x_t = torch.tensor(x, dtype=torch.float32, device=device)

    with torch.no_grad():
        logits = model(x_t).cpu().numpy()[0]

    probs = softmax_numpy(logits)
    return probs


def rms_energy(y: np.ndarray) -> float:
    return float(np.sqrt(np.mean(y ** 2) + 1e-12))


def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")

    model, chords, mu, sigma, device = load_model(MODEL_PATH)

    sr = 22050
    window_seconds = 2.0
    hop_seconds = 0.1

    window_samples = int(sr * window_seconds)
    hop_samples = int(sr * hop_seconds)

    # thresholds
    silence_threshold = 0.005
    confidence_threshold = 0.50

    # smoothing
    prob_history = deque(maxlen=5)

    # display stability
    displayed_label = "Listening..."
    candidate_label = None
    candidate_count = 0
    required_repeats = 3

    print("Loaded model.")
    print(f"Chords: {chords}")
    print("Listening... Press Ctrl+C to stop.\n")

    audio_buffer = np.zeros(window_samples, dtype=np.float32)

    stream = sd.InputStream(
        samplerate=sr,
        channels=1,
        dtype="float32",
        blocksize=hop_samples,
    )

    try:
        with stream:
            while True:
                audio_chunk, _ = stream.read(hop_samples)
                audio_chunk = audio_chunk[:, 0]

                # update rolling buffer
                audio_buffer[:-hop_samples] = audio_buffer[hop_samples:]
                audio_buffer[-hop_samples:] = audio_chunk

                energy = rms_energy(audio_buffer)

                if energy < silence_threshold:
                    prob_history.clear()
                    raw_label = "No chord detected"
                    confidence = 0.0
                    top_text = "signal too weak"
                else:
                    feats = extract_features(audio_buffer, sr=sr)
                    probs = predict_probs(model, feats, mu, sigma, device)

                    prob_history.append(probs)
                    avg_probs = np.mean(np.stack(prob_history, axis=0), axis=0)

                    pred_idx = int(np.argmax(avg_probs))
                    pred_chord = chords[pred_idx]
                    confidence = float(avg_probs[pred_idx])

                    top_indices = np.argsort(avg_probs)[::-1][:3]
                    top_text = " | ".join(
                        f"{chords[i]}: {avg_probs[i]:.3f}" for i in top_indices
                    )

                    if confidence < confidence_threshold:
                        raw_label = "Uncertain"
                    else:
                        raw_label = pred_chord

                # stabilize displayed output
                if raw_label == displayed_label:
                    candidate_label = None
                    candidate_count = 0
                else:
                    if raw_label == candidate_label:
                        candidate_count += 1
                    else:
                        candidate_label = raw_label
                        candidate_count = 1

                    if candidate_count >= required_repeats:
                        displayed_label = raw_label
                        candidate_label = None
                        candidate_count = 0

                print(
                    f"\rDetected: {displayed_label:<18} "
                    f"Raw: {raw_label:<18} "
                    f"Confidence: {confidence:.3f} "
                    f"RMS: {energy:.4f} "
                    f"Top3 -> {top_text}",
                    end=""
                )

                time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()