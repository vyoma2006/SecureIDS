"""
Defender service — owned by Team Lead.

Wraps src/defender/ logic (train_baseline, adversarial_training) and the
saved model artifacts (src/defender/saved_models/) for api/routers/defender_routes.py.

Per docs/model_contract.md:
  - Model expects scaled input, feature order == feature_columns.json, shape (n, 78).
  - robust_model_all_eps.pt is the recommended model for live inference
    (falls back to baseline_model.pt if the robust checkpoint isn't present yet).

This layer is the ONLY place that loads the model / scaler / label encoder
for inference, so we don't reload them from disk on every request.
"""

import io
import json
import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch

from src.defender.model_config import IDS_MLP

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_ARTIFACTS_DIR = BASE_DIR / "src" / "defender" / "saved_models"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results" / "metrics"

BASELINE_MODEL_FILE = "baseline_model.pt"
ROBUST_MODEL_FILE = "robust_model_all_eps.pt"  # recommended per docs/model_contract.md

# Cache so we don't re-read weights/scaler/encoder from disk on every request.
_artifact_cache: dict[str, Any] = {}


class ArtifactsNotFoundError(Exception):
    """Raised when a required saved-model artifact is missing from disk."""


def _require(path: Path, what: str) -> Path:
    if not path.exists():
        raise ArtifactsNotFoundError(
            f"{what} not found at '{path}'. Run `python -m src.defender.train_baseline` first."
        )
    return path


def load_inference_artifacts(force_reload: bool = False) -> dict[str, Any]:
    """
    Load (and cache) everything needed to run inference: model, scaler,
    label encoder, and feature column order.

    Prefers the robust (adversarially-trained) model per docs/model_contract.md;
    falls back to the frozen baseline model if adversarial training hasn't
    produced robust_model_all_eps.pt yet.
    """
    if not force_reload and _artifact_cache:
        return _artifact_cache

    arch_path = _require(MODEL_ARTIFACTS_DIR / "model_architecture.json", "Model architecture file")
    with open(arch_path) as f:
        arch = json.load(f)

    robust_path = MODEL_ARTIFACTS_DIR / ROBUST_MODEL_FILE
    baseline_path = _require(MODEL_ARTIFACTS_DIR / BASELINE_MODEL_FILE, "Baseline model weights")

    if robust_path.exists():
        weights_path, model_used = robust_path, "robust_model_all_eps"
    else:
        weights_path, model_used = baseline_path, "baseline_model"

    model = IDS_MLP(
        input_dim=arch["input_dim"],
        num_classes=arch["num_classes"],
        hidden_sizes=arch["hidden_sizes"],
    )
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    model.eval()

    with open(_require(MODEL_ARTIFACTS_DIR / "scaler.pkl", "Feature scaler"), "rb") as f:
        scaler = pickle.load(f)
    with open(_require(MODEL_ARTIFACTS_DIR / "label_encoder.pkl", "Label encoder"), "rb") as f:
        label_encoder = pickle.load(f)
    with open(_require(MODEL_ARTIFACTS_DIR / "feature_columns.json", "Feature column list")) as f:
        feature_columns = json.load(f)

    _artifact_cache.clear()
    _artifact_cache.update({
        "model": model,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "feature_columns": feature_columns,
        "class_names": list(label_encoder.classes_),
        "model_used": model_used,
    })
    return _artifact_cache


MAX_ROWS = 10_000  # keeps the live demo responsive; the UI renders one row per prediction


def _normalise_columns(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """
    Make uploaded headers match feature_columns.json exactly.

    Real CICIDS exports often have stray spaces (' Flow Duration') or different
    capitalisation, which used to make every column look "missing". Match on
    stripped, lower-cased names and rename to the canonical spelling. If the
    same feature appears twice, the first occurrence wins.
    """
    canonical = {c.strip().lower(): c for c in feature_columns}
    rename = {}
    for col in df.columns:
        key = str(col).strip().lower()
        if key in canonical and col != canonical[key]:
            rename[col] = canonical[key]
    if rename:
        df = df.rename(columns=rename)
    return df.loc[:, ~df.columns.duplicated()]


def _find_label_column(df: pd.DataFrame) -> Optional[str]:
    for col in df.columns:
        if str(col).strip().lower() == "label":
            return col
    return None


def predict(csv_bytes: bytes) -> dict[str, Any]:
    """
    Classify uploaded traffic data.

    csv_bytes: raw bytes of an uploaded CSV containing RAW (unscaled) feature
    values, one row per traffic sample. The 78 feature_columns.json columns
    must be present (extra columns are ignored; a 'Label' column, if present,
    is echoed back as `actual_label` so the UI can compare); this function
    scales the data itself using the saved scaler.pkl before running it
    through the model, so callers should never pre-scale.

    Rows with missing / infinite / non-numeric values are NOT classified
    (training dropped such rows too, and NaN input would otherwise silently
    come out as "Benign"). They are reported in `skipped_rows` instead.

    Returns a dict matching api/schemas/prediction_schema.PredictResponse.
    """
    artifacts = load_inference_artifacts()
    feature_columns = artifacts["feature_columns"]
    class_names = artifacts["class_names"]

    try:
        df = pd.read_csv(io.BytesIO(csv_bytes))
    except Exception as e:
        raise ValueError(f"Could not parse uploaded file as CSV: {e}") from e

    if df.empty:
        raise ValueError("Uploaded CSV has no rows.")
    if len(df) > MAX_ROWS:
        raise ValueError(
            f"Uploaded CSV has {len(df):,} rows; the live demo accepts at most {MAX_ROWS:,}. "
            f"Upload a smaller sample."
        )

    df = _normalise_columns(df, feature_columns)

    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"Uploaded CSV is missing {len(missing)} required feature column(s), "
            f"e.g. {missing[:5]}. Expected all {len(feature_columns)} columns from "
            f"src/defender/saved_models/feature_columns.json."
        )

    label_col = _find_label_column(df)
    actual_labels = df[label_col].astype(str).str.strip().tolist() if label_col is not None else None

    # Coerce to numbers ("Infinity" parses to inf; junk text becomes NaN),
    # then find rows the model must not see.
    features = df[feature_columns].apply(pd.to_numeric, errors="coerce")
    X_raw = features.to_numpy(dtype="float64")
    X_raw = np.where(np.abs(X_raw) > np.finfo("float32").max, np.inf, X_raw)  # would overflow float32
    finite = np.isfinite(X_raw)
    row_ok = finite.all(axis=1)

    skipped = []
    for i in np.flatnonzero(~row_ok):
        bad_cols = [feature_columns[j] for j in np.flatnonzero(~finite[i])]
        shown = ", ".join(bad_cols[:3]) + (f" (+{len(bad_cols) - 3} more)" if len(bad_cols) > 3 else "")
        skipped.append({"row_index": int(i), "reason": f"Missing, infinite or non-numeric value in: {shown}"})

    if not row_ok.any():
        raise ValueError(
            f"None of the {len(df)} rows could be classified. First problem: {skipped[0]['reason']}. "
            f"Rows must be clean numeric values (no blanks, NaN or Infinity)."
        )

    good_idx = np.flatnonzero(row_ok)
    X_good = X_raw[good_idx].astype("float32")

    try:
        X_scaled = artifacts["scaler"].transform(X_good)
    except Exception as e:
        raise ValueError(f"Feature scaling failed — check that columns contain numeric data: {e}") from e

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    with torch.no_grad():
        logits = artifacts["model"](X_tensor)
        probs = torch.softmax(logits, dim=1).numpy()

    predictions = []
    for row_probs, original_index in zip(probs, good_idx):
        pred_idx = int(row_probs.argmax())
        predictions.append({
            "row_index": int(original_index),
            "predicted_class": class_names[pred_idx],
            "confidence": float(row_probs[pred_idx]),
            "probabilities": {
                class_names[j]: float(row_probs[j]) for j in range(len(class_names))
            },
            "actual_label": actual_labels[original_index] if actual_labels is not None else None,
        })

    return {
        "model_used": artifacts["model_used"],
        "n_rows": int(len(df)),
        "n_classified": len(predictions),
        "predictions": predictions,
        "skipped_rows": skipped,
    }


def get_feature_info() -> dict[str, Any]:
    """
    What the live demo needs to build a template / manual-entry form:
    required columns in model order, class names, and each feature's
    training-set mean in RAW units (from the saved scaler).
    """
    artifacts = load_inference_artifacts()
    feature_columns = artifacts["feature_columns"]
    means = artifacts["scaler"].mean_
    return {
        "model_used": artifacts["model_used"],
        "class_names": artifacts["class_names"],
        "feature_columns": feature_columns,
        "feature_means": {c: float(m) for c, m in zip(feature_columns, means)},
    }


def train_baseline_model() -> dict[str, Any]:
    """
    Runs the real baseline training pipeline (src/defender/train_baseline.py).

    This is slow (multiple epochs over the full processed dataset) and
    blocks the request thread while it runs — callers (routers) should treat
    this as a long-running admin operation, not something to expose behind a
    snappy end-user button. Requires data/processed/{X,y}_{train,val,test}.csv
    to already exist.
    """
    if not PROCESSED_DIR.exists() or not any(PROCESSED_DIR.glob("*.csv")):
        raise ArtifactsNotFoundError(
            f"No processed data found in '{PROCESSED_DIR}'. "
            f"Run the data_processing pipeline before training."
        )

    from src.defender.train_baseline import train_baseline_model as _train

    _, test_acc = _train(
        processed_dir=str(PROCESSED_DIR),
        model_artifacts_dir=str(MODEL_ARTIFACTS_DIR),
        results_dir=str(RESULTS_DIR),
    )
    load_inference_artifacts(force_reload=True)  # pick up the freshly written weights
    return {"status": "completed", "test_accuracy": float(test_acc)}


def adversarial_retrain(experiment_name: str = "all_eps") -> dict[str, Any]:
    """
    Runs the real adversarial-training pipeline
    (src/defender/adversarial_training.run_adversarial_training) for one
    experiment ("all_eps" or "high_eps_only").

    Slow — trains a fresh model on clean + adversarial data. Requires
    data/processed/ and data/adversarial/ (the latter produced by the
    Attacker's FGSM generation step) to already exist.
    """
    from src.defender.adversarial_training import run_adversarial_training

    adversarial_dir = BASE_DIR / "data" / "adversarial"
    if not PROCESSED_DIR.exists() or not any(PROCESSED_DIR.glob("*.csv")):
        raise ArtifactsNotFoundError(f"No processed data found in '{PROCESSED_DIR}'.")
    if not adversarial_dir.exists() or not any(adversarial_dir.glob("*.csv")):
        raise ArtifactsNotFoundError(
            f"No adversarial samples found in '{adversarial_dir}'. "
            f"Run POST /attacker/generate first."
        )

    epsilon_filter: Optional[list] = [0.1, 0.3] if experiment_name == "high_eps_only" else None
    if experiment_name not in ("all_eps", "high_eps_only"):
        raise ValueError("experiment_name must be 'all_eps' or 'high_eps_only'.")

    result = run_adversarial_training(
        experiment_name=experiment_name,
        train_epsilon_filter=epsilon_filter,
        processed_dir=str(PROCESSED_DIR),
        adversarial_dir=str(adversarial_dir),
        model_artifacts_dir=str(MODEL_ARTIFACTS_DIR),
        results_dir=str(RESULTS_DIR),
    )
    if experiment_name == "all_eps":
        load_inference_artifacts(force_reload=True)  # this is the model /predict prefers

    return {
        "status": "completed",
        "experiment_name": experiment_name,
        "robust_clean_accuracy": result["robust"]["clean_test_accuracy"],
        "robust_evasion_rate": result["robust"]["adversarial_evasion_rate"],
    }
