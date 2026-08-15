"""
Retrains the IDS MLP on original training data + adversarial samples,
to measure and improve robustness against FGSM evasion.

Supports running an ablation study: train on different subsets of
epsilon values (e.g. "all epsilons" vs "only high epsilons") and compare
results -- both models are always EVALUATED against the full adversarial
set (all epsilons), regardless of what they were trained on, so the
comparison is meaningful rather than each model just being tested on
what it already saw.

Reads:
    data/processed/X_train.csv, y_train.csv   (original clean training data)
    data/processed/X_test.csv,  y_test.csv    (original clean test data, untouched)
    data/adversarial/fgsm_<class_name>_<epsilon>.csv   (attacker's adversarial samples)
    src/defender/saved_models/label_encoder.pkl
    src/defender/saved_models/feature_columns.json
    src/defender/saved_models/model_architecture.json

Saves (per experiment):
    src/defender/saved_models/robust_model_<experiment_name>.pt
    results/metrics/comparison_<experiment_name>.json

Usage:
    python -m src.defender.adversarial_training
"""

import glob
import json
import os
import pickle
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report, f1_score

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


def load_clean_split(processed_dir: str, split_name: str):
    X = pd.read_csv(os.path.join(processed_dir, f"X_{split_name}.csv"))
    y = pd.read_csv(os.path.join(processed_dir, f"y_{split_name}.csv")).values.ravel().astype("int64")
    return X, y


def parse_class_and_epsilon_from_filename(filepath: str, class_names: list):
    """
    Adversarial filenames look like: fgsm_Bot_0.3.csv, fgsm_DDOS attack-HOIC_0.1.csv
    Returns (class_name, epsilon_float).
    """
    filename = os.path.basename(filepath)

    class_name = None
    for name in sorted(class_names, key=len, reverse=True):
        if name in filename:
            class_name = name
            break
    if class_name is None:
        raise ValueError(f"Could not determine class label from filename: {filename}")

    eps_match = re.search(r"(\d+\.\d+)\.csv$", filename)
    if not eps_match:
        raise ValueError(f"Could not determine epsilon from filename: {filename}")
    epsilon = float(eps_match.group(1))

    return class_name, epsilon


def load_adversarial_samples(
    adversarial_dir: str,
    feature_columns: list,
    label_encoder,
    epsilon_filter: list = None,
):
    """
    Load adversarial CSVs, infer (class, epsilon) from filename, enforce
    column order, and combine into (X, y).

    If epsilon_filter is given (e.g. [0.1, 0.3]), only files matching
    those epsilon values are loaded. If None, all files are loaded.
    """
    csv_paths = sorted(glob.glob(os.path.join(adversarial_dir, "*.csv")))
    if not csv_paths:
        raise FileNotFoundError(f"No adversarial CSVs found in {adversarial_dir}.")

    class_names = list(label_encoder.classes_)
    X_parts, y_parts = [], []
    skipped = 0

    for path in csv_paths:
        class_name, epsilon = parse_class_and_epsilon_from_filename(path, class_names)

        if epsilon_filter is not None and epsilon not in epsilon_filter:
            skipped += 1
            continue

        label_int = label_encoder.transform([class_name])[0]
        df = pd.read_csv(path)

        missing = set(feature_columns) - set(df.columns)
        extra = set(df.columns) - set(feature_columns)
        if missing or extra:
            raise ValueError(f"{path} has mismatched columns.\nMissing: {missing}\nUnexpected: {extra}")
        df = df[feature_columns]

        X_parts.append(df.values.astype("float32"))
        y_parts.append(np.full(len(df), label_int, dtype="int64"))
        print(f"  [included] {os.path.basename(path):45s} class={class_name:25s} eps={epsilon} rows={len(df):,}")

    if skipped:
        print(f"  [skipped {skipped} files not matching epsilon_filter={epsilon_filter}]")

    if not X_parts:
        raise ValueError(f"No adversarial files matched epsilon_filter={epsilon_filter}")

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    print(f"Total adversarial samples loaded: {len(X):,}\n")
    return X, y


def make_dataloader(X, y, batch_size, shuffle):
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        preds = torch.argmax(model(X_batch), dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(y_batch.numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


def compute_evasion_rate(model, X_adv, y_true, device):
    model.eval()
    with torch.no_grad():
        preds = torch.argmax(model(torch.from_numpy(X_adv).to(device)), dim=1).cpu().numpy()
    return float((preds != y_true).mean())


def run_adversarial_training(
    experiment_name: str,
    train_epsilon_filter: list = None,
    processed_dir: str = "data/processed",
    adversarial_dir: str = "data/adversarial",
    model_artifacts_dir: str = "src/defender/saved_models",
    results_dir: str = "results/metrics",
):
    """
    Runs one adversarial training experiment.

    train_epsilon_filter: which epsilon values to INCLUDE IN TRAINING
        (e.g. [0.1, 0.3]). None = use all available epsilons.
        Evaluation always uses the FULL adversarial set (all epsilons),
        regardless of this filter, so results are comparable across
        experiments.
    """
    set_seed()
    os.makedirs(results_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'#'*60}\n# EXPERIMENT: {experiment_name}\n# train_epsilon_filter={train_epsilon_filter}\n{'#'*60}")
    print(f"Using device: {device}\n")

    with open(os.path.join(model_artifacts_dir, "label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)
    with open(os.path.join(model_artifacts_dir, "feature_columns.json")) as f:
        feature_columns = json.load(f)
    with open(os.path.join(model_artifacts_dir, "model_architecture.json")) as f:
        arch = json.load(f)

    class_names = list(label_encoder.classes_)
    num_classes = len(class_names)
    input_dim = arch["input_dim"]

    print("Loading clean train/test data...")
    X_train_clean_df, y_train_clean = load_clean_split(processed_dir, "train")
    X_test_clean_df, y_test_clean = load_clean_split(processed_dir, "test")
    X_train_clean = X_train_clean_df[feature_columns].values.astype("float32")
    X_test_clean = X_test_clean_df[feature_columns].values.astype("float32")

    print("\nLoading adversarial samples FOR TRAINING (filtered)...")
    X_adv_train, y_adv_train = load_adversarial_samples(
        adversarial_dir, feature_columns, label_encoder, epsilon_filter=train_epsilon_filter
    )

    print("Loading adversarial samples FOR EVALUATION (full set, all epsilons)...")
    X_adv_eval, y_adv_eval = load_adversarial_samples(
        adversarial_dir, feature_columns, label_encoder, epsilon_filter=None
    )

    # Baseline model, for reference numbers
    baseline_model = IDS_MLP(input_dim=input_dim, num_classes=num_classes,
                              hidden_sizes=arch["hidden_sizes"]).to(device)
    baseline_model.load_state_dict(torch.load(
        os.path.join(model_artifacts_dir, "baseline_model.pt"), map_location=device
    ))
    baseline_evasion_rate = compute_evasion_rate(baseline_model, X_adv_eval, y_adv_eval, device)
    test_loader = make_dataloader(X_test_clean, y_test_clean, BATCH_SIZE, shuffle=False)
    preds, labels = evaluate(baseline_model, test_loader, device)
    baseline_clean_acc = accuracy_score(labels, preds)
    baseline_macro_f1 = f1_score(labels, preds, average="macro")

    # Combine clean + (filtered) adversarial training data
    X_combined = np.vstack([X_train_clean, X_adv_train])
    y_combined = np.concatenate([y_train_clean, y_adv_train])
    shuffle_idx = np.random.permutation(len(X_combined))
    X_combined, y_combined = X_combined[shuffle_idx], y_combined[shuffle_idx]
    print(f"Combined training set: {X_combined.shape[0]:,} rows "
          f"({len(X_train_clean):,} clean + {len(X_adv_train):,} adversarial)")

    train_loader = make_dataloader(X_combined, y_combined, BATCH_SIZE, shuffle=True)

    robust_model = IDS_MLP(input_dim=input_dim, num_classes=num_classes,
                            hidden_sizes=arch["hidden_sizes"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(robust_model.parameters(), lr=LEARNING_RATE)

    best_state, best_test_acc, epochs_without_improvement = None, 0.0, 0
    print("\nTraining...")
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(robust_model, train_loader, optimizer, criterion, device)
        preds, labels = evaluate(robust_model, test_loader, device)
        test_acc = accuracy_score(labels, preds)
        print(f"  Epoch {epoch:02d}/{NUM_EPOCHS} | train_loss={train_loss:.4f} | clean_test_acc={test_acc:.4f}")
        if test_acc > best_test_acc:
            best_test_acc, best_state, epochs_without_improvement = test_acc, robust_model.state_dict(), 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(f"  Early stopping at epoch {epoch}.")
                break

    robust_model.load_state_dict(best_state)

    preds, labels = evaluate(robust_model, test_loader, device)
    robust_clean_acc = accuracy_score(labels, preds)
    robust_macro_f1 = f1_score(labels, preds, average="macro")
    robust_clean_report = classification_report(labels, preds, target_names=class_names, output_dict=True)
    robust_evasion_rate = compute_evasion_rate(robust_model, X_adv_eval, y_adv_eval, device)

    model_path = os.path.join(model_artifacts_dir, f"robust_model_{experiment_name}.pt")
    torch.save(robust_model.state_dict(), model_path)

    result = {
        "experiment_name": experiment_name,
        "train_epsilon_filter": train_epsilon_filter,
        "baseline": {
            "clean_test_accuracy": baseline_clean_acc,
            "macro_f1": baseline_macro_f1,
            "adversarial_evasion_rate": baseline_evasion_rate,
        },
        "robust": {
            "clean_test_accuracy": robust_clean_acc,
            "macro_f1": robust_macro_f1,
            "adversarial_evasion_rate": robust_evasion_rate,
            "classification_report": robust_clean_report,
        },
        "improvement": {
            "evasion_rate_reduction": baseline_evasion_rate - robust_evasion_rate,
            "clean_accuracy_change": robust_clean_acc - baseline_clean_acc,
            "macro_f1_change": robust_macro_f1 - baseline_macro_f1,
        },
        "class_names": class_names,
        "num_adversarial_samples_used_for_training": int(len(X_adv_train)),
        "num_adversarial_samples_used_for_eval": int(len(X_adv_eval)),
    }

    comparison_path = os.path.join(results_dir, f"comparison_{experiment_name}.json")
    with open(comparison_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {model_path}")
    print(f"Saved: {comparison_path}")
    print(f"RESULT [{experiment_name}] clean_acc={robust_clean_acc:.4f} "
          f"macro_f1={robust_macro_f1:.4f} evasion_rate={robust_evasion_rate:.4f}\n")

    return result


def run_ablation_study():
    """
    Runs both experiments and prints a final comparison table.
    Both models are always EVALUATED against the same full adversarial
    set, so the comparison is fair regardless of what each was trained on.
    """
    baseline_ref = None
    results = []

    result_a = run_adversarial_training(
        experiment_name="all_eps",
        train_epsilon_filter=None,  # train on everything
    )
    results.append(("AT (All eps)", result_a))
    baseline_ref = result_a["baseline"]  # same for both, just grab once

    result_b = run_adversarial_training(
        experiment_name="high_eps_only",
        train_epsilon_filter=[0.1, 0.3],
    )
    results.append(("AT (0.1 + 0.3 only)", result_b))

    print("\n" + "=" * 70)
    print("ABLATION STUDY SUMMARY")
    print("=" * 70)
    print(f"{'Model':<25}{'Clean Acc':<15}{'Macro F1':<15}{'Evasion Rate':<15}")
    print(f"{'Baseline':<25}{baseline_ref['clean_test_accuracy']:<15.4f}"
          f"{baseline_ref['macro_f1']:<15.4f}{baseline_ref['adversarial_evasion_rate']:<15.4f}")
    for name, res in results:
        r = res["robust"]
        print(f"{name:<25}{r['clean_test_accuracy']:<15.4f}{r['macro_f1']:<15.4f}{r['adversarial_evasion_rate']:<15.4f}")

    return results


if __name__ == "__main__":
    run_ablation_study()