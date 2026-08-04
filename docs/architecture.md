# Architecture — How the 4 Modules Connect

## Pipeline flow
CICIDS2018 raw CSVs
|
v
src/data_processing/ --> data/processed/ (train/val/test splits)
--> src/defender/saved_models/ (scaler, label_encoder, feature_columns)
|
v
src/defender/train_baseline.py
|
v
src/defender/saved_models/baseline_model.pt (FROZEN, used by Attacker)
|
v
src/attacker/generate_adversarial.py --> data/adversarial/fgsm_<class>_<epsilon>.csv
|
v
src/defender/adversarial_training.py (ablation study: 2 experiments)
|
+---------------------------+---------------------------+
| |
v v
robust_model_all_eps.pt robust_model_high_eps_only.pt
(PRIMARY -- see defense_findings.md) (comparison / ablation)
|
v
results/metrics/comparison_all_eps.json, comparison_high_eps_only.json
|
v
api/ --> frontend/ dashboard_app.py (visualized for the user)
|
v
src/validator/ (independent sign-off, including generalization check --
see "known caveat" in docs/model_contract.md)


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

## Key design decisions

- **MLP over Random Forest/XGBoost**: chosen specifically because FGSM requires a differentiable model. See `results/reports/baseline_findings.md` for why.
- **8 classes, not all 15**: very low-sample classes (e.g. SQL Injection at 87 rows) were excluded — too few samples to train or evaluate meaningfully.
- **Downsampling Benign**: originally 13.4M rows vs ~2.8M across all attack classes combined; capped to keep the dataset both memory-manageable and reasonably balanced.
- **Ablation study on adversarial training**: rather than assuming which epsilon values to train on, two experiments were run and compared (all epsilons vs. high-epsilon-only), evaluated identically against the full adversarial set. Full reasoning and results in `results/reports/defense_findings.md`.