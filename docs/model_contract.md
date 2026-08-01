# Defender Model Contract

Interface contract for anyone consuming the baseline IDS model — primarily the Attacker (FGSM), but also relevant to the Visualizer and Validator.

Owner: Defender (Team Lead)
Last updated: [fill in date]
Status: Baseline model FROZEN as of this version — see "Model Freeze" below.

---

## Artifacts and their locations

| File | Path | Purpose |
|---|---|---|
| Model weights | `src/defender/saved_models/baseline_model.pt` | PyTorch `state_dict` |
| Model architecture | `src/defender/saved_models/model_architecture.json` | Needed to reconstruct the model shape before loading weights |
| Feature scaler | `src/defender/saved_models/scaler.pkl` | `sklearn.StandardScaler`, fit on training data |
| Label encoder | `src/defender/saved_models/label_encoder.pkl` | `sklearn.LabelEncoder`, maps class name ↔ integer |
| Feature column order | `src/defender/saved_models/feature_columns.json` | Exact ordered list of 78 feature names |

## How to load the model

```python
import json
import torch
from src.defender.model_config import IDS_MLP

with open("src/defender/saved_models/model_architecture.json") as f:
    arch = json.load(f)

model = IDS_MLP(
    input_dim=arch["input_dim"],
    num_classes=arch["num_classes"],
    hidden_sizes=arch["hidden_sizes"],
)
model.load_state_dict(torch.load("src/defender/saved_models/baseline_model.pt"))
model.eval()
```

## Input format FGSM must produce

- Input tensor shape: `(batch_size, 78)`, dtype `float32`
- **Features must already be scaled** using the saved `scaler.pkl` before being passed to the model. FGSM perturbations should be computed in this scaled space — do not perturb raw/unscaled feature values.
- Feature order must exactly match `feature_columns.json`. Do not reorder columns.
- Model is fully differentiable (plain Linear/ReLU/Dropout layers, softmax applied only inside the loss function) — standard `torch.autograd` gradient computation works directly, no surrogate model needed.

## Output format

- `model(x)` returns raw logits, shape `(batch_size, 8)`
- Apply `torch.softmax(logits, dim=1)` for class probabilities
- Class index → name mapping (also in `label_encoder.pkl`):
0 -> Benign
1 -> Bot
2 -> DDOS attack-HOIC
3 -> DDoS attacks-LOIC-HTTP
4 -> DoS attacks-Hulk
5 -> FTP-BruteForce
6 -> Infilteration
7 -> SSH-Bruteforce

## What the Attacker should produce, and where

Adversarial samples generated via FGSM should be saved to `data/adversarial/`, in the same scaled/78-column format as the training data (so `adversarial_training.py` can consume them directly later without reformatting). Suggested filename pattern: `data/adversarial/fgsm_<class_name>_<epsilon>.csv`.

## Model Freeze

This baseline model is considered **frozen** — the Attacker should build and test against this exact version. If the Defender retrains and updates `baseline_model.pt`, this file's "Last updated" date must change and the team should be notified, since it may shift adversarial results.

## Known model behavior worth knowing before you attack it

Benign and Infiltration are the two weakest classes in the baseline (see `results/reports/baseline_findings.md` for full analysis) — the model already confuses them with each other before any adversarial perturbation is applied. Keep this in mind when interpreting evasion results on this pair; a "successful evasion" here is less meaningful than on the other 6 classes, which are cleanly and confidently classified pre-attack.