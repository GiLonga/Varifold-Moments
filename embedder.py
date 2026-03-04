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


def split_normalize(data, labels):
    X_train, X_val, y_train, y_val = train_test_split(
    data, labels, test_size=0.2, random_state=42
)


    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    return X_train,X_val,y_train,y_val


class FeatureDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    


class MLP(nn.Module):
    def __init__(self, input_dim=70, hidden1=128, hidden2=64, num_classes=2):
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

            nn.Linear(hidden2, num_classes)
        )

    def forward(self, x):
        return self.network(x)


def main(X_train, X_val, y_train, y_val, MLP):

    train_dataset = FeatureDataset(X_train, y_train)
    val_dataset = FeatureDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MLP(input_dim=70, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)


    best_val_loss = float("inf")
    patience = 10
    patience_counter = 0
    num_epochs = 100

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
    
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

    # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                outputs = model(xb)
                loss = criterion(outputs, yb)
                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(yb.cpu().numpy())

        val_acc = accuracy_score(all_targets, all_preds)

        print(f"Epoch {epoch+1:03d} | "
          f"Train Loss: {train_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | "
          f"Val Acc: {val_acc:.4f}")

    # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
    print("\nTraining complete. Best model saved as 'best_model.pt'.")


if __name__ == "__main__":
    n = 10

    #
    