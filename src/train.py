import os
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from features import extract_features_from_file

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Detect available chords by scanning subdirectories in the data folder
CHORDS = sorted(
    d for d in os.listdir(DATA_DIR)
    if os.path.isdir(os.path.join(DATA_DIR, d))
)
# Mapping between text labels and numeric IDs
CHORD_TO_IDX = {c: i for i, c in enumerate(CHORDS)}
IDX_TO_CHORD = {i: c for c, i in CHORD_TO_IDX.items()}

# Set random seeds for reproducibility of results
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def load_dataset():
    """
    Iterates through folders, extracts features from every .wav file,
    and returns them as numpy arrays X (features) and y (labels).
    """
    X, y = [], []
    for chord in CHORDS:
        chord_dir = os.path.join(DATA_DIR, chord)

        wavs = [f for f in os.listdir(chord_dir) if f.lower().endswith(".wav")]
        if len(wavs) == 0:
            raise FileNotFoundError(f"No .wav files in: {chord_dir}")

        for fn in wavs:
            fp = os.path.join(chord_dir, fn)
            # Feature extraction produces a vector of 24 numbers (12 means, 12 stds)
            feats = extract_features_from_file(fp)
            X.append(feats)
            y.append(CHORD_TO_IDX[chord])

    X = np.stack(X, axis=0)
    y = np.array(y, dtype=np.int64)
    return X, y


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (Simple Neural Network).
    Consists of three linear layers with ReLU activation and Dropout for regularization.
    """
    def __init__(self, in_dim=24, num_classes=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2), # Randomly disables 20% of neurons to prevent overfitting
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes),
        )

    # Passes input through the network.
    def forward(self, x):

        return self.net(x)


def plot_confusion_matrix(cm, class_names):
    """
    Visualizes the model's performance.
    Shows which chords are being correctly identified vs which are being confused.aram cm:
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def main():
    X, y = load_dataset()
    print(f"Dataset: X={X.shape}, y={y.shape} (classes={len(CHORDS)})")
    """
        Stratified split ensures training and testing sets have the same proportion of each chord.
        75% for training, 25% for verifying accuracy. 
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=SEED, stratify=y
    )
    """
    Z-score Normalization: Scales features so they have mean=0 and std=1.
    This helps the neural network learn much faster and more reliably.
    """
    mu = X_train.mean(axis=0, keepdims=True)
    sigma = X_train.std(axis=0, keepdims=True) + 1e-9
    X_train = (X_train - mu) / sigma
    X_test = (X_test - mu) / sigma

    # Use GPU (cuda) for training if available, otherwise fall back to CPU.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    # crating the model | input size = num features | output size = number of chords |
    model = MLP(in_dim=X.shape[1], num_classes=len(CHORDS)).to(device)

    # Adam optimizer, lr = learning rate -> too high = unstable / too low = slow
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # CrossEntropyLoss: Measures the difference between predicted and actual chord.
    loss_fn = nn.CrossEntropyLoss()

    # Convert numpy arrays to PyTorch tensors for processing on GPU/CPU
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.long, device=device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_test_t = torch.tensor(y_test, dtype=torch.long, device=device)

    # Training Loop
    epochs = 80 # Number of times the model sees the entire dataset
    for epoch in range(1, epochs + 1):
        model.train()
        
        # Forward pass: predict chords
        logits = model(X_train_t)
        loss = loss_fn(logits, y_train_t)

        # Backward pass: calculate error and update weights
        opt.zero_grad() # Clear previous gradients
        loss.backward() # Compute new gradients (backpropagation)
        opt.step()      # Apply weight changes

        # Print progress every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            model.eval() # Switch to evaluation mode (turns off Dropout)
            with torch.no_grad():
                pred = model(X_test_t).argmax(dim=1)
                acc = (pred == y_test_t).float().mean().item()
            print(f"Epoch {epoch:03d} | loss={loss.item():.4f} | test_acc={acc:.3f}")

    # Final evaluation
    model.eval()
    with torch.no_grad():
        pred = model(X_test_t).argmax(dim=1).cpu().numpy()

    cm = confusion_matrix(y_test, pred)

    print("\nClassification report:")
    print(classification_report(y_test, pred, target_names=CHORDS))

    plot_confusion_matrix(cm, CHORDS)

    """
    Save Model and Metadata
    We save the model weights along with the normalization stats (mu, sigma) 
    and the chord names so the prediction scripts can use them correctly.
    """
    save_path = os.path.join(MODELS_DIR, "chord_mlp.pt")
    payload = {
        "state_dict": model.state_dict(),
        "mu": mu.astype(np.float32),
        "sigma": sigma.astype(np.float32),
        "chords": CHORDS,
        "in_dim": X.shape[1],
    }
    torch.save(payload, save_path)
    print(f"\nSaved trained model to: {save_path}")


if __name__ == "__main__":
    main()