import sys
import numpy as np
import torch
from features import extract_features_from_file

def main():
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py <path_to_wav>")
        raise SystemExit(1)

    wav_path = sys.argv[1]
    # Load model checkpoint containing weights, chord list, and normalization stats
    ckpt = torch.load("models/chord_mlp.pt", map_location="cpu", weights_only=False)

    chords = ckpt["chords"] # List of supported chord names
    mu = ckpt["mu"]         # Mean used for feature standardization
    sigma = ckpt["sigma"]   # Standard deviation used for feature standardization

    # Rebuild model architecture (MLP: Multi-Layer Perceptron)
    import torch.nn as nn

    class MLP(nn.Module):
        def __init__(self, in_dim=24, num_classes=5):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 64), # Input layer to hidden layer
                nn.ReLU(),             # Activation function for non-linearity
                nn.Dropout(0.2),       # Regularization to prevent overfitting
                nn.Linear(64, 32),     # Hidden layer reduction
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, num_classes), # Output layer (scores for each chord)
            )

        def forward(self, x):
            return self.net(x)

    model = MLP(in_dim=ckpt["in_dim"], num_classes=len(chords))
    model.load_state_dict(ckpt["state_dict"])
    model.eval() # Set model to evaluation mode (disables dropout)

    # Extract Chroma features (12 bins representing semi-tones) and their statistics
    feats = extract_features_from_file(wav_path)[None, :]  # Shape (1, 24)
    
    # Standardize features using training set statistics: (X - mu) / sigma
    feats = (feats - mu) / sigma
    x = torch.tensor(feats, dtype=torch.float32)

    with torch.no_grad():
        # Get raw scores and convert to probabilities using Softmax
        probs = torch.softmax(model(x), dim=1).numpy().ravel()

    # Get index of the highest probability
    idx = int(np.argmax(probs))
    print("Prediction:", chords[idx])
    print("Probabilities:")
    # Display sorted results for debugging/analysis
    for c, p in sorted(zip(chords, probs), key=lambda t: -t[1]):
        print(f"  {c}: {p:.3f}")

if __name__ == "__main__":
    main()