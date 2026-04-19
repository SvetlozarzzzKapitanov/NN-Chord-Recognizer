import os
import time
import threading
from collections import deque

import numpy as np
import sounddevice as sd
import torch

from features import extract_features

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "chord_mlp.pt")


class MLP(torch.nn.Module):
    """
    Multi-Layer Perceptron (MLP) for chord classification.
    Input: 24 features (12 mean chroma + 12 std chroma).
    Output: Logits for each chord class.
    """
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
    """
    Loads model weights and normalization parameters (mu/mean, sigma/std).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    payload = torch.load(model_path, map_location=device, weights_only=False)

    chords = payload["chords"]
    mu = payload["mu"]       # Mean for feature scaling
    sigma = payload["sigma"] # Standard deviation for feature scaling

    model = MLP(in_dim=payload["in_dim"], num_classes=len(chords)).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    return model, chords, mu, sigma, device


def softmax_numpy(x: np.ndarray) -> np.ndarray:
    """
    Standard Softmax implementation for NumPy to convert logits to probabilities.
    """
    x = x - np.max(x) # Numerical stability
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


def predict_probs(model, features: np.ndarray, mu: np.ndarray, sigma: np.ndarray, device: str) -> np.ndarray:
    """
    Standardizes features and runs inference.
    """
    # Normalize: (X - mu) / sigma
    x = (features[None, :] - mu) / sigma
    x_t = torch.tensor(x, dtype=torch.float32, device=device)

    with torch.no_grad():
        logits = model(x_t).cpu().numpy()[0]

    probs = softmax_numpy(logits)
    return probs


def rms_energy(y: np.ndarray) -> float:
    """
    Calculates Root Mean Square (RMS) energy to represent signal loudness.
    Used to detect silence/weak signals.
    """
    return float(np.sqrt(np.mean(y ** 2) + 1e-12))


class RealTimeChordRecognizer:
    """
    Processes live audio using a sliding window and rolling buffer.
    """
    def __init__(self, model_path: str = MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing model file: {model_path}")

        self.model, self.chords, self.mu, self.sigma, self.device = load_model(model_path)

        self.sr = 22050 # Sample Rate (Hz)
        self.window_seconds = 2.5 # Analysis window length
        self.hop_seconds = 0.5    # Processing interval (step size)

        self.window_samples = int(self.sr * self.window_seconds)
        self.hop_samples = int(self.sr * self.hop_seconds)

        self.silence_threshold = 0.01
        self.confidence_threshold = 0.30

        # Temporal smoothing buffer to avoid flickering predictions
        self.prob_history = deque(maxlen=3)

        self.displayed_label = "Listening..."
        self.candidate_label = None
        self.candidate_count = 0
        self.required_repeats = 2 # Stability: same chord must appear N times to update UI

        # Rolling buffer: holds the latest 'window_samples' for analysis
        self.audio_buffer = np.zeros(self.window_samples, dtype=np.float32)

        self.status = "Idle"
        self.raw_label = "Listening..."
        self.confidence = 0.0
        self.energy = 0.0
        self.top_text = "-"
        self.is_running = False

        self._thread = None
        self._lock = threading.Lock()

    def feed_audio_chunk(self, chunk: np.ndarray):
        """
        Shifts the rolling buffer and appends the new audio chunk.
        """
        chunk = chunk.astype(np.float32)

        if len(chunk) != self.hop_samples:
            if len(chunk) < self.hop_samples:
                chunk = np.pad(chunk, (0, self.hop_samples - len(chunk)))
            else:
                chunk = chunk[:self.hop_samples]

        # Rolling update: discard oldest samples, append newest
        self.audio_buffer[:-self.hop_samples] = self.audio_buffer[self.hop_samples:]
        self.audio_buffer[-self.hop_samples:] = chunk

    def process_buffer(self):
        """
        Main analysis pipeline: Energy check -> Feature Extraction (Chroma) -> Inference.
        """
        energy = rms_energy(self.audio_buffer)

        if energy < self.silence_threshold:
            self.prob_history.clear()
            raw_label = "No chord detected"
            confidence = 0.0
            top_text = "signal too weak"
        else:
            # Extract statistical Chroma features (note intensities)
            feats = extract_features(self.audio_buffer, sr=self.sr)
            probs = predict_probs(self.model, feats, self.mu, self.sigma, self.device)

            # Smooth predictions using max probability over the history window
            self.prob_history.append(probs)
            avg_probs = np.max(np.stack(self.prob_history, axis=0), axis=0)

            pred_idx = int(np.argmax(avg_probs))
            pred_chord = self.chords[pred_idx]
            confidence = float(avg_probs[pred_idx])

            # Prepare debug info for top 3 candidates
            top_indices = np.argsort(avg_probs)[::-1][:3]
            top_text = " | ".join(
                f"{self.chords[i]}: {avg_probs[i]:.3f}" for i in top_indices
            )

            if confidence < self.confidence_threshold:
                raw_label = "Uncertain"
            else:
                raw_label = pred_chord

        self._update_display_label(raw_label)

        with self._lock:
            self.raw_label = raw_label
            self.confidence = confidence
            self.energy = energy
            self.top_text = top_text

    def start(self):
        """Initializes background audio capture thread."""
        if self.is_running:
            return

        self.is_running = True
        with self._lock:
            self.status = "Listening"
            self.displayed_label = "Listening..."
            self.raw_label = "Listening..."
            self.confidence = 0.0
            self.energy = 0.0
            self.top_text = "-"
            self.audio_buffer = np.zeros(self.window_samples, dtype=np.float32)
            self.prob_history.clear()
            self.candidate_label = None
            self.candidate_count = 0

        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.is_running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)

        with self._lock:
            self.status = "Idle"
            self.displayed_label = "Stopped"
            self.raw_label = "-"
            self.confidence = 0.0
            self.energy = 0.0
            self.top_text = "-"
            self.prob_history.clear()

    def get_state(self):
        with self._lock:
            return {
                "status": self.status,
                "detected_label": self.displayed_label,
                "raw_label": self.raw_label,
                "confidence": self.confidence,
                "energy": self.energy,
                "top_text": self.top_text,
                "is_running": self.is_running,
            }

    def _run_loop(self):
        """
        Background loop capturing microphone input via sounddevice.
        """
        try:
            stream = sd.InputStream(
                samplerate=self.sr,
                channels=1,
                dtype="float32",
                blocksize=self.hop_samples, # Process every hop_samples
            )

            with stream:
                while self.is_running:
                    audio_chunk, _ = stream.read(self.hop_samples)
                    audio_chunk = audio_chunk[:, 0]

                    self.feed_audio_chunk(audio_chunk)
                    self.process_buffer()

                    with self._lock:
                        self.status = "Listening"

                    time.sleep(0.01)

        except Exception as e:
            with self._lock:
                self.status = f"Error: {e}"
                self.displayed_label = "Error"
                self.raw_label = "Error"
                self.confidence = 0.0
                self.energy = 0.0
                self.top_text = str(e)
            self.is_running = False

    def _update_display_label(self, raw_label: str):
        """
        Basic hysteresis to prevent UI flickering.
        The detected chord must be stable for 'required_repeats' cycles.
        """
        if raw_label == self.displayed_label:
            self.candidate_label = None
            self.candidate_count = 0
            return

        if raw_label == self.candidate_label:
            self.candidate_count += 1
        else:
            self.candidate_label = raw_label
            self.candidate_count = 1

        if self.candidate_count >= self.required_repeats:
            with self._lock:
                self.displayed_label = raw_label
            self.candidate_label = None
            self.candidate_count = 0


def main():
    recognizer = RealTimeChordRecognizer()

    print("Loaded model.")
    print(f"Chords: {recognizer.chords}")
    print("Listening... Press Ctrl+C to stop.\n")

    recognizer.start()

    try:
        while True:
            state = recognizer.get_state()
            # \r (carriage return) overwrites the current line in terminal
            print(
                f"\rDetected: {state['detected_label']:<18} "
                f"Raw: {state['raw_label']:<18} "
                f"Confidence: {state['confidence']:.3f} "
                f"RMS: {state['energy']:.4f} "
                f"Top3 -> {state['top_text']}",
                end=""
            )
            time.sleep(0.1)
    except KeyboardInterrupt:
        recognizer.stop()
        print("\nStopped.")


if __name__ == "__main__":
    main()