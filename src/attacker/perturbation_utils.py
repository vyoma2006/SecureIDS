import json
import torch
import pickle
import pandas as pd
from src.defender.model_config import IDS_MLP

def load_model_artifacts(base_path="src/defender/saved_models"):
    with open(f"{base_path}/model_architecture.json") as f:
        arch = json.load(f)
    model = IDS_MLP(
        input_dim=arch["input_dim"],
        num_classes=arch["num_classes"],
        hidden_sizes=arch["hidden_sizes"],
    )
    model.load_state_dict(torch.load(f"{base_path}/baseline_model.pt",weights_only=True))
    model.eval()

    with open(f"{base_path}/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(f"{base_path}/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    with open(f"{base_path}/feature_columns.json") as f:
        feature_columns = json.load(f)

    return model, scaler, label_encoder, feature_columns

def prepare_inputs(X_df, y_df, feature_columns, scaler, label_encoder):
    X_raw = X_df[feature_columns].values
    X_tensor = torch.tensor(X_raw, dtype=torch.float32)
    y_tensor = torch.tensor(y_df["label_encoded"].values, dtype=torch.long)
    return X_tensor, y_tensor