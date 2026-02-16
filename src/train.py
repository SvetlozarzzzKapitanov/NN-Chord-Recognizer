import os
import random
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from features import extract_features_from_file

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Folder names must match your data folders exactly:
CHORDS = ["Am", "C", "Em", "F", "G"]
CHORD_TO_IDX = {c: i for i, c in enumerate(CHORDS)}
IDX_TO_CHORD = {i: c for c, i in CHORD_TO_IDX.items()}

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def load_dataset():
    X, y = [], []
    for chord in CHORDS:
        chord_dir = os.path.join(DATA_DIR, chord)
        if not os.path.isdir(chord_dir):
            raise FileNotFoundError(f"Missing folder: {chord_dir}")

        wavs = [f for f in os.listdir(chord_dir) if f.lower().endswith(".wav")]
        if len(wavs) == 0:
            raise FileNotFoundError(f"No .wav files in: {chord_dir}")

        for fn in wavs:
            fp = os.path.join(chord_dir, fn)
            feats = extract_features_from_file(fp)  # (24,)
            X.append(feats)
            y.append(CHORD_TO_IDX[chord])

    X = np.stack(X, axis=0)  # (N, 24)
    y = np.array(y, dtype=np.int64)  # (N,)
    return X, y

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

def main():
    X, y = load_dataset()
    print(f"Dataset: X={X.shape}, y={y.shape} (classes={len(CHORDS)})")

    # Stratified split keeps class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=SEED, stratify=y
    )

    # Standardize features using train stats (important)
    mu = X_train.mean(axis=0, keepdims=True)
    sigma = X_train.std(axis=0, keepdims=True) + 1e-9
    X_train = (X_train - mu) / sigma
    X_test = (X_test - mu) / sigma

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MLP(in_dim=X.shape[1], num_classes=len(CHORDS)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.long, device=device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_test_t = torch.tensor(y_test, dtype=torch.long, device=device)

    epochs = 80
    for epoch in range(1, epochs + 1):
        model.train()
        logits = model(X_train_t)
        loss = loss_fn(logits, y_train_t)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                pred = model(X_test_t).argmax(dim=1)
                acc = (pred == y_test_t).float().mean().item()
            print(f"Epoch {epoch:03d} | loss={loss.item():.4f} | test_acc={acc:.3f}")

    # Final evaluation
    model.eval()
    with torch.no_grad():
        pred = model(X_test_t).argmax(dim=1).cpu().numpy()

    print("\nConfusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_test, pred))

    print("\nClassification report:")
    print(classification_report(y_test, pred, target_names=CHORDS))

    # Save model + normalization stats
    save_path = os.path.join(MODELS_DIR, "chord_mlp.pt")
    payload = {
        "state_dict": model.state_dict(),
        "mu": mu.astype(np.float32),
        "sigma": sigma.astype(np.float32),
        "chords": CHORDS,
        "in_dim": X.shape[1],
    }
    torch.save(payload, save_path)
    print(f"\nSaved model to: {save_path}")

if __name__ == "__main__":
    main()
