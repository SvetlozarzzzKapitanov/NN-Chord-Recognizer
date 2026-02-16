import sys
import numpy as np
import torch
from features import extract_features_from_file

def main():
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py <path_to_wav>")
        raise SystemExit(1)

    wav_path = sys.argv[1]
    ckpt = torch.load("models/chord_mlp.pt", map_location="cpu", weights_only=False)

    chords = ckpt["chords"]
    mu = ckpt["mu"]
    sigma = ckpt["sigma"]

    # Rebuild model
    import torch.nn as nn

    class MLP(nn.Module):
        def __init__(self, in_dim=24, num_classes=5):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, num_classes),
            )

        def forward(self, x):
            return self.net(x)

    model = MLP(in_dim=ckpt["in_dim"], num_classes=len(chords))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    feats = extract_features_from_file(wav_path)[None, :]  # (1, 24)
    feats = (feats - mu) / sigma
    x = torch.tensor(feats, dtype=torch.float32)

    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1).numpy().ravel()

    idx = int(np.argmax(probs))
    print("Prediction:", chords[idx])
    print("Probabilities:")
    for c, p in sorted(zip(chords, probs), key=lambda t: -t[1]):
        print(f"  {c}: {p:.3f}")

if __name__ == "__main__":
    main()
