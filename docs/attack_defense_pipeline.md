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

## Phase 3: Adversarial attack (Attacker, done)

The Attacker loaded `baseline_model.pt` (per `docs/model_contract.md`) and generated adversarial traffic samples using FGSM at four epsilon levels (0.01, 0.05, 0.1, 0.3) per class:

1. Computed the gradient of the loss with respect to the (scaled) input features
2. Perturbed each feature by `epsilon * sign(gradient)`
3. Saved adversarial samples to `data/adversarial/fgsm_<class_name>_<epsilon>.csv`, same scaled/78-column format as training data
4. Measured evasion rate per class/epsilon against the baseline model

Status: Done. Baseline model evasion rate averaged 28.53% across all adversarial samples (see `results/reports/defense_findings.md` for the full breakdown). Some per-class evasion patterns (e.g. flat rates across epsilons for Bot and DDOS-HOIC) were flagged as worth investigating further but did not block progress — see `src/attacker/README.md` for the attacker's own notes.

## Phase 4: Adversarial training (Defender, done)

Ran as an ablation study rather than a single retraining pass, per `src/defender/adversarial_training.py`:

1. Combined original clean training data with adversarial samples
2. **Experiment A**: trained on adversarial samples across all 4 epsilons
3. **Experiment B**: trained only on the higher-epsilon (0.1, 0.3) adversarial samples
4. Both models evaluated identically: clean accuracy/macro F1 on the untouched clean test set, evasion rate against the *full* adversarial set (all epsilons) regardless of training subset

Status: Done.

| Model | Clean Accuracy | Macro F1 | Evasion Rate |
|---|---|---|---|
| Baseline | 91.45% | 0.9054 | 28.53% |
| AT (All eps) | 93.75% | **0.9166** | **0.41%** |
| AT (0.1+0.3 only) | **93.94%** | 0.9129 | 1.97% |

**`robust_model_all_eps.pt` is the recommended primary defense model** — nearly 5x lower evasion rate than the alternative, with comparable clean accuracy. Full analysis, including why "all epsilons" outperformed "high epsilon only," is in `results/reports/defense_findings.md`.

**Known caveat, relevant for Phase 6**: robustness was measured against the same FGSM method used to generate the training data. Whether this generalizes to a different attack method or fresh adversarial samples is an open question for the Validator to check.

## Phase 5: Visualization (Visualizer, can start now against dummy data)

Dashboard reads from `results/metrics/` (via the API) to show:
- Baseline accuracy by class
- Evasion rate (before defense)
- Before/after accuracy comparison (after adversarial training)
- Confusion matrices

## Phase 6: Validation (Validator, after Phase 4)

Independent check of the full pipeline's outputs against known attack signatures / MITRE ATT&CK mapping, producing a final sign-off report.