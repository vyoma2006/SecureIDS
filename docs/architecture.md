# Architecture — How the 4 Modules Connect

## Pipeline flow
data/raw/ --(src/data_processing/load_data.py)--> data/processed/merged_raw.csv
|
v
(src/data_processing/preprocess.py)
|
v
data/processed/X_train.csv, y_train.csv, X_val.csv, y_val.csv, X_test.csv, y_test.csv
src/defender/saved_models/scaler.pkl, label_encoder.pkl, feature_columns.json
|
v
(src/defender/train_baseline.py)
|
v
src/defender/saved_models/baseline_model.pt + model_architecture.json
|
+-----------------------------------+-----------------------------------+
| |
v v
(src/attacker/generate_adversarial.py) (src/validator/test_cases.py)
| independent stress test
v
data/adversarial/fgsm_<class_name>_<epsilon>.csv
|
v
(src/defender/adversarial_training.py)
|
v
src/defender/saved_models/robust_model.pt
|
v
results/metrics/*.json --> api/ --> frontend/dashboard_app.py
|
v
(src/validator/test_cases.py) re-runs against robust_model.pt too


## Contracts (don't change without telling the team)

**Processed data schema** (`data/processed/X_train.csv` + `y_train.csv`, same for val/test):
- 78 feature columns, exact order defined in `src/defender/saved_models/feature_columns.json` — never reorder.
- Features are already scaled (StandardScaler, fit on train only). Do not re-scale.
- Labels are integers, encoded via `src/defender/saved_models/label_encoder.pkl`.
- 8 classes: `Benign, Bot, DDOS attack-HOIC, DDoS attacks-LOIC-HTTP, DoS attacks-Hulk, FTP-BruteForce, Infilteration, SSH-Bruteforce`.

**Adversarial samples schema** (`data/adversarial/*.csv`):
- Same 78 scaled columns as `X_train.csv`, same column order.
- Include the *ground truth* label alongside each sample, not the model's (possibly-fooled) prediction — the Defender needs the true label to retrain correctly.

**Model artifacts** (`src/defender/saved_models/`):
- `baseline_model.pt` — PyTorch `state_dict`, frozen once the Attacker starts building against it.
- `model_architecture.json` — required to reconstruct the exact model shape (input_dim, hidden_sizes, num_classes) before loading the weights above.
- Full loading instructions: `docs/model_contract.md`.

**Metrics output** (`results/metrics/*.json`):
- `baseline_metrics.json` — from `src/defender/train_baseline.py` (accuracy, per-class precision/recall/F1, confusion matrix)
- `comparison.json` — from `src/defender/adversarial_training.py` (before/after numbers) — not yet created
- `validation_report.json` — from `src/validator/` — not yet created
- The dashboard (`frontend/dashboard_app.py`) reads these via the API; it should never recompute metrics itself, only display what's already been written.

## Why MLP, not Random Forest/XGBoost

FGSM requires computing a gradient of the loss with respect to the input — Random Forest and XGBoost aren't differentiable, so literal FGSM can't be applied to them without a surrogate model. To avoid that extra complexity, the baseline IDS is a PyTorch MLP instead: fully differentiable, so **FGSM applies directly to the real model, no surrogate needed.** See `results/reports/baseline_findings.md` for why this choice was made and how the baseline performed.

## Weekly integration point

Everyone should be able to run `python -m src.defender.train_baseline` and get a fresh `src/defender/saved_models/baseline_model.pt` at any time — that's the contract the other three roles build against. Keep it working. Once the Attacker starts building FGSM code against a specific version, that version is considered **frozen** (see `docs/model_contract.md`) until the whole team agrees to update it.