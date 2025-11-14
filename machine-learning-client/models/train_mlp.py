import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

DATA_DIR = Path("data")
MODELS_DIR = Path("models")

X_PATH = DATA_DIR / "hagrid_keypoints_X.npy"
Y_PATH = DATA_DIR / "hagrid_keypoints_y.npy"
CLASSES_PATH = DATA_DIR / "hagrid_classes.json"

MODEL_PATH = MODELS_DIR / "gesture_mlp.pt"


class GestureMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def load_data():
    X = np.load(X_PATH)  # (N, 63)
    y = np.load(Y_PATH)

    with open(CLASSES_PATH, "r") as f:
        class_names = json.load(f)

    print(f"[INFO] Loaded X: {X.shape}, y: {y.shape}")
    print(f"[INFO] Classes: {class_names}")

    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).long()

    dataset = TensorDataset(X, y)

    # 80% train，20% test
    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    return train_ds, val_ds, class_names


def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    loss_total = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss_total += loss.item() * X_batch.size(0)

            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += X_batch.size(0)

    return loss_total / total, correct / total


def main():
    MODELS_DIR.mkdir(exist_ok=True)

    train_ds, val_ds, class_names = load_data()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    input_dim = train_ds[0][0].shape[0]
    num_classes = len(class_names)
    model = GestureMLP(input_dim, num_classes).to(device)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, 26):
        model.train()
        train_loss = 0.0
        n_samples = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)
            n_samples += X_batch.size(0)

        train_loss /= n_samples
        val_loss, val_acc = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()

    model.load_state_dict(best_state)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "input_dim": input_dim,
            "num_classes": num_classes,
        },
        MODEL_PATH,
    )

    print(f"[RESULT] Best val_acc = {best_val_acc:.4f}")
    print(f"[SAVED] Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
