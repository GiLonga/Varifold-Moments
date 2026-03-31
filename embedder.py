from JC_momenst import *
from JC_functions import *

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder

def invariant_n_moments(n=10):
    """
    Compute the invariant n moments for a given n.
    Parameters:
    n (int): The number of moments to compute (default is 10).
    Returns:    list of tuples: A list of tuples where each tuple contains the moments (a, b, c).   
    """
    solutions = [(a, a + c, c)
                for a in range(n + 1)
                for c in range(n + 1)
                if a + c <= n]
    return solutions


def split_normalize(X, y, seed=42):
    """
    80% train | 10% val | 10% test
    Scaler is fit ONLY on train to prevent leakage.
    """
    # First cut off the test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.10, random_state=seed, stratify=y
    )
    # Then split the remainder into train / val (val = ~11.1% of temp ≈ 10% of total)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.10, random_state=seed, stratify=y_temp
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test


class FeatureDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    

class MLP_simple(nn.Module):
    def __init__(self, input_dim=70, hidden1=256, hidden2=128, 
                 num_classes=2, dropout_rate=0.4):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden2, hidden2 // 2),   # extra bottleneck layer
            nn.BatchNorm1d(hidden2 // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden2 // 2, num_classes)
        )
        
        # Better weight initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.network(x)

class MLP(nn.Module):
    def __init__(self, input_dim=70, hidden1=512, hidden2=256, hidden3=128, hidden4 = 64, num_classes=2):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden2, hidden3),
            nn.BatchNorm1d(hidden3),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden3, num_classes)
        )

    def forward(self, x):
        return self.network(x)


def train_model(X_train, X_val, X_test, y_train, y_val, y_test, MLP, run_id=0):

    train_loader = DataLoader(FeatureDataset(X_train, y_train), batch_size=64,
                              shuffle=True, pin_memory=True)
    val_loader   = DataLoader(FeatureDataset(X_val,   y_val),   batch_size=64,
                              pin_memory=True)
    test_loader  = DataLoader(FeatureDataset(X_test,  y_test),  batch_size=64,
                              pin_memory=True)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                     patience=3000, factor=0.5)

    num_epochs       = 100_000
    patience         = 10_000
    patience_counter = 0
    best_val_loss    = float("inf")
    best_val_acc     = 0.0
    ckpt_path        = f"best_model_run{run_id}.pt"

    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, num_epochs + 1):

        # ── Train ─────────────────────────────────────────────────
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_train_loss = running_loss / len(train_loader)

        # ── Validate ──────────────────────────────────────────────
        model.eval()
        running_val_loss = 0.0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                outputs = model(xb)
                running_val_loss += criterion(outputs, yb).item()
                all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                all_targets.extend(yb.cpu().numpy())

        avg_val_loss = running_val_loss / len(val_loader)
        val_acc      = accuracy_score(all_targets, all_preds)
        best_val_acc = max(best_val_acc, val_acc)

        scheduler.step(avg_val_loss)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)

        if epoch % 1000 == 0 or epoch == 1:
            print(f"  [Run {run_id}] Epoch {epoch:05d} | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | "
                  f"Val Acc: {val_acc:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        # ── Early stopping & checkpointing ────────────────────────
        if avg_val_loss < best_val_loss:
            best_val_loss    = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  [Run {run_id}] Early stopping at epoch {epoch}.")
                break

    # ── Test evaluation with best checkpoint ──────────────────────
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()
    test_preds, test_targets = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            outputs = model(xb)
            test_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            test_targets.extend(yb.cpu().numpy())

    test_acc = accuracy_score(test_targets, test_preds)
    print(f"  [Run {run_id}] Best Val Acc: {best_val_acc:.4f} | Test Acc: {test_acc:.4f}")

    plot_training_history(history, run_id)
    return test_acc, best_val_acc


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_training_history(history, run_id=0):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Training History — Run {run_id}", fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.plot(epochs, history["train_loss"], label="Train Loss",      linewidth=1.5)
    ax.plot(epochs, history["val_loss"],   label="Validation Loss", linewidth=1.5, linestyle="--")
    best_epoch = int(np.argmin(history["val_loss"])) + 1
    ax.axvline(best_epoch, color="red", linestyle=":", linewidth=1.2,
               label=f"Best epoch ({best_epoch})")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Loss vs. Epochs")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(epochs, history["val_acc"], color="green", linewidth=1.5, label="Val Accuracy")
    ax.axhline(max(history["val_acc"]), color="red", linestyle=":", linewidth=1.2,
               label=f"Best acc ({max(history['val_acc']):.4f})")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy"); ax.set_title("Val Accuracy vs. Epochs")
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"training_history_run{run_id}.png", dpi=150, bbox_inches="tight")
    plt.show()


# ── Repeated random-split cross-validation ────────────────────────────────────

def run_cross_validation(X, labels, n_runs=5):
    """
    Runs n_runs independent train/val/test splits with different seeds.
    Encodes labels once upfront to avoid inconsistencies across runs.
    Reports mean ± std of test accuracy.
    """
    le = LabelEncoder()
    y  = le.fit_transform(labels)          # encode once, reuse every run

    num_classes = len(np.unique(y))
    input_dim   = X.shape[1]
    print(f"Input dim: {input_dim}  |  Classes: {num_classes}  |  Samples: {len(y)}\n")

    # Quick sanity checks
    assert not np.isnan(X).any(),  "Data contains NaN values!"
    assert not np.isinf(X).any(),  "Data contains Inf values!"
    for cls, count in zip(*np.unique(y, return_counts=True)):
        print(f"  Class {le.inverse_transform([cls])[0]}: {count} samples "
              f"({100*count/len(y):.1f}%)")
    print()

    BoundMLP = lambda: MLP_simple(input_dim=input_dim, num_classes=num_classes)

    test_accs = []
    val_accs  = []

    for run_id in range(n_runs):
        print(f"{'='*60}")
        print(f"  RUN {run_id + 1} / {n_runs}  (seed={run_id})")
        print(f"{'='*60}")

        X_train, X_val, X_test, y_train, y_val, y_test = split_normalize(X, y, seed=run_id)

        test_acc, val_acc = train_model(
            X_train, X_val, X_test,
            y_train, y_val, y_test,
            BoundMLP, run_id=run_id
        )
        test_accs.append(test_acc)
        val_accs.append(val_acc)

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  CROSS-VALIDATION SUMMARY ({n_runs} runs)")
    print(f"{'='*60}")
    for i, (v, t) in enumerate(zip(val_accs, test_accs)):
        print(f"  Run {i}: Val Acc = {v:.4f}  |  Test Acc = {t:.4f}")
    print(f"\n  Val  Acc — mean: {np.mean(val_accs):.4f}  std: {np.std(val_accs):.4f}")
    print(f"  Test Acc — mean: {np.mean(test_accs):.4f}  std: {np.std(test_accs):.4f}")

    plot_cv_summary(val_accs, test_accs)
    return test_accs, val_accs


# ── CV Summary plot ───────────────────────────────────────────────────────────

def plot_cv_summary(val_accs, test_accs):
    runs = np.arange(1, len(test_accs) + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(runs, val_accs,  marker="o", label="Val Acc",  linewidth=1.5)
    plt.plot(runs, test_accs, marker="s", label="Test Acc", linewidth=1.5)
    plt.axhline(np.mean(test_accs), color="red", linestyle="--", linewidth=1.2,
                label=f"Mean Test Acc ({np.mean(test_accs):.4f})")
    plt.fill_between(runs,
                     np.mean(test_accs) - np.std(test_accs),
                     np.mean(test_accs) + np.std(test_accs),
                     alpha=0.15, color="red", label="±1 std")
    plt.xlabel("Run"); plt.ylabel("Accuracy")
    plt.title("Cross-Validation Summary")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig("cv_summary.png", dpi=150, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    n = 10

    