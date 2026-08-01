"""
Trains the baseline MLP IDS model on the preprocessed CICIDS2018 data.

Reads:
    data/processed/X_train.csv, y_train.csv
    data/processed/X_val.csv,   y_val.csv
    data/processed/X_test.csv,  y_test.csv
    src/defender/saved_models/feature_columns.json  (to sanity-check shape)
    src/defender/saved_models/label_encoder.pkl       (to sanity-check classes)

Saves:
    src/defender/saved_models/baseline_model.pt        (trained weights)
    src/defender/saved_models/model_architecture.json  (input_dim, hidden
                                                          sizes, num_classes
                                                          -- needed to
                                                          reconstruct the
                                                          model before
                                                          loading weights)
    results/metrics/baseline_metrics.json               (accuracy, per-class
                                                           precision/recall/
                                                           F1, confusion
                                                           matrix)

Usage:
    python -m src.defender.train_baseline
"""

import json
import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.utils.class_weight import compute_class_weight

from src.defender.model_config import (
    IDS_MLP,
    LEARNING_RATE,
    BATCH_SIZE,
    NUM_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    RANDOM_STATE,
)


def set_seed(seed: int = RANDOM_STATE):
    torch.manual_seed(seed)
    np.random.seed(seed)


def load_split(processed_dir: str, split_name: str):
    """Load one split (train/val/test) as (X, y) numpy arrays."""
    X = pd.read_csv(os.path.join(processed_dir, f"X_{split_name}.csv")).values.astype("float32")
    y = pd.read_csv(os.path.join(processed_dir, f"y_{split_name}.csv")).values.ravel().astype("int64")
    return X, y


def make_dataloader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool):
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def compute_class_weights_tensor(y_train: np.ndarray, num_classes: int, device):
    """Class weights help with the remaining mild imbalance after
    downsampling in preprocessing (classes are close but not identical)."""
    classes = np.arange(num_classes)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_loss_and_preds(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        total_loss += loss.item() * X_batch.size(0)

        preds = torch.argmax(logits, dim=1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(y_batch.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    return avg_loss, all_preds, all_labels


def train_baseline_model(
    processed_dir: str = "data/processed",
    model_artifacts_dir: str = "src/defender/saved_models",
    results_dir: str = "results/metrics",
):
    set_seed()
    os.makedirs(model_artifacts_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\nLoading preprocessed data...")
    X_train, y_train = load_split(processed_dir, "train")
    X_val, y_val = load_split(processed_dir, "val")
    X_test, y_test = load_split(processed_dir, "test")

    with open(os.path.join(model_artifacts_dir, "label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)
    class_names = list(label_encoder.classes_)
    num_classes = len(class_names)
    input_dim = X_train.shape[1]

    print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
    print(f"Input dim: {input_dim} | Num classes: {num_classes}")
    print(f"Classes: {class_names}")

    train_loader = make_dataloader(X_train, y_train, BATCH_SIZE, shuffle=True)
    val_loader = make_dataloader(X_val, y_val, BATCH_SIZE, shuffle=False)
    test_loader = make_dataloader(X_test, y_test, BATCH_SIZE, shuffle=False)

    model = IDS_MLP(input_dim=input_dim, num_classes=num_classes).to(device)
    print(f"\nModel architecture:\n{model}")

    class_weights = compute_class_weights_tensor(y_train, num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_model_state = None

    print("\nStarting training...\n")
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_preds, val_labels = evaluate_loss_and_preds(model, val_loader, criterion, device)
        val_acc = accuracy_score(val_labels, val_preds)

        print(f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
              f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping: no val improvement for {EARLY_STOPPING_PATIENCE} epochs.")
                break

    print("\nLoading best model (lowest val loss) for final evaluation...")
    model.load_state_dict(best_model_state)

    print("\nEvaluating on test set...")
    _, test_preds, test_labels = evaluate_loss_and_preds(model, test_loader, criterion, device)

    test_acc = accuracy_score(test_labels, test_preds)
    report = classification_report(
        test_labels, test_preds, target_names=class_names, output_dict=True
    )
    cm = confusion_matrix(test_labels, test_preds).tolist()

    print(f"\nTest accuracy: {test_acc:.4f}\n")
    print(classification_report(test_labels, test_preds, target_names=class_names))

    # --- Save model weights ---
    model_path = os.path.join(model_artifacts_dir, "baseline_model.pt")
    torch.save(best_model_state, model_path)
    print(f"\nSaved model weights to {model_path}")

    # --- Save architecture info so anyone (attacker's FGSM code,
    #     evaluate.py, the API) can reconstruct this exact model before
    #     loading the weights above ---
    architecture_path = os.path.join(model_artifacts_dir, "model_architecture.json")
    with open(architecture_path, "w") as f:
        json.dump({
            "input_dim": input_dim,
            "num_classes": num_classes,
            "hidden_sizes": [128, 64, 32],
            "class_names": class_names,
        }, f, indent=2)
    print(f"Saved model architecture to {architecture_path}")

    # --- Save metrics for the visualizer's dashboard ---
    metrics_path = os.path.join(results_dir, "baseline_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "test_accuracy": test_acc,
            "classification_report": report,
            "confusion_matrix": cm,
            "class_names": class_names,
        }, f, indent=2)
    print(f"Saved metrics to {metrics_path}")

    return model, test_acc


if __name__ == "__main__":
    train_baseline_model()