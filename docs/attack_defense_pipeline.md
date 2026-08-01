# Attack & Defense Pipeline Walkthrough

This is the step-by-step narrative of how data flows through the system, phase by phase. For the exact technical interface (file formats, function signatures), see `docs/model_contract.md`.

## Phase 1: Data preparation (Defender/team, done)

1. Download all 10 CICIDS2018 CSVs from AWS S3 into `data/raw/`
2. `python -m src.data_processing.load_data` — streams and merges all files into `data/processed/merged_raw.csv`
3. `python -m src.data_processing.preprocess` — filters to 8 selected classes, downsamples Benign, cleans, splits (train/val/test), scales, and saves the shared artifacts (`scaler.pkl`, `label_encoder.pkl`, `feature_columns.json`)

Status: Done. See `results/reports/baseline_findings.md` for dataset stats.

## Phase 2: Baseline model (Defender, done)

`python -m src.defender.train_baseline` trains the MLP on the preprocessed data and saves:
- `baseline_model.pt` — trained weights
- `model_architecture.json` — architecture spec, needed to reconstruct the model before loading weights
- `results/metrics/baseline_metrics.json` — accuracy, per-class precision/recall/F1, confusion matrix

Status: Done. 91.45% test accuracy, 6/8 classes at near-perfect precision/recall. Full findings in `results/reports/baseline_findings.md`.

**This model is now frozen** — the Attacker builds against this exact version. See `docs/model_contract.md` for the freeze policy.

## Phase 3: Adversarial attack (Attacker, in progress)

The Attacker loads `baseline_model.pt` (per `docs/model_contract.md`) and generates adversarial traffic samples using FGSM:

1. Compute the gradient of the loss with respect to the (scaled) input features
2. Perturb each feature by `epsilon * sign(gradient)`
3. Save the resulting adversarial samples to `data/adversarial/`, keeping the same scaled/78-column format as training data
4. Measure evasion rate: what fraction of adversarial samples get misclassified by the baseline model that were correctly classified before perturbation

Output expected: adversarial CSVs in `data/adversarial/`, evasion rate metrics for the dashboard.

## Phase 4: Adversarial training (Defender, upcoming)

Once adversarial samples exist:

1. Combine original training data + adversarial samples into a new training set
2. Retrain the MLP on this combined set (`src/defender/adversarial_training.py`)
3. Re-evaluate on the original test set AND on adversarial samples
4. Compare before/after: did detection rate on adversarial samples improve after retraining, without hurting accuracy on normal traffic?

Output expected: a second model checkpoint, before/after metrics for the dashboard.

## Phase 5: Visualization (Visualizer, can start now against dummy data)

Dashboard reads from `results/metrics/` (via the API) to show:
- Baseline accuracy by class
- Evasion rate (before defense)
- Before/after accuracy comparison (after adversarial training)
- Confusion matrices

## Phase 6: Validation (Validator, after Phase 4)

Independent check of the full pipeline's outputs against known attack signatures / MITRE ATT&CK mapping, producing a final sign-off report.