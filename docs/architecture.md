# Architecture — How the 4 Modules Connect

## Pipeline flow

```
data/raw/  --(defense/preprocessing.py)-->  data/processed/train.csv, test.csv
                                                    |
                                                    v
                                    (defense/train_baseline.py)
                                                    |
                                                    v
                                    models/baseline/ids_baseline.pkl
                                                    |
                            +-----------------------+-----------------------+
                            |                                               |
                            v                                               v
        (attack/generate_adversarial_samples.py)              (validation/test_cases.py)
                            |                                     independent stress test
                            v
    data/processed/adversarial/fgsm_samples.csv
                            |
                            v
        (defense/adversarial_training.py)
                            |
                            v
        models/robust/ids_robust.pkl
                            |
                            v
        (visualization/metrics.py) --> results/metrics/*.json --> (visualization/dashboard/app.py)
                            |
                            v
                (validation/test_cases.py) re-runs against models/robust/ too
```

## Contracts (don't change without telling the team)

**Processed data schema** (`data/processed/train.csv`, `test.csv`):
- Same feature columns throughout the whole pipeline.
- Label column name comes from `config/config.yaml -> data.label_column`.
- Classes are exactly: `Normal`, `DDoS`, `PortScan`, `BruteForce` (see `config/config.yaml -> data.classes`).

**Adversarial samples schema** (`data/processed/adversarial/fgsm_samples.csv`):
- Identical columns to `test.csv`.
- The `Label` column holds the *ground truth* label, not what the model predicted after perturbation — the Defender needs the true label to retrain correctly.

**Model artifacts** (`models/baseline/`, `models/robust/`):
- Saved with `joblib`, loaded via `src/defense/model_utils.load_model()`.
- Anyone loading a model uses that shared helper — don't reimplement loading logic in your own module.

**Metrics output** (`results/metrics/*.json`):
- `baseline_eval.json` — from `defense/train_baseline.py`
- `comparison.json` — from `defense/adversarial_training.py` (before/after numbers)
- `validation_report.json` — from `validation/test_cases.py`
- The dashboard (`visualization/dashboard/app.py`) reads all three; it should never recompute metrics itself, only display what's already been written.

## FGSM on tree-based models — heads-up for the Attacker

RandomForest/XGBoost aren't differentiable, so literal FGSM (which needs a
gradient) doesn't apply directly. Pick one and document it:
1. Train a differentiable surrogate model (small PyTorch MLP) that mimics
   the baseline, attack the surrogate with real FGSM, and test how well the
   resulting samples transfer to fooling the actual RF/XGBoost baseline.
2. Use a gradient-free adversarial method suited to tree ensembles (e.g. via
   the Adversarial Robustness Toolbox) and call it out in the report as "FGSM
   adapted for tabular/tree-based models" rather than literal FGSM.

Either is fine for this project scope — just be explicit in the final report
about which approach was used and why, since a validator or reviewer will ask.

## Weekly integration point

Everyone should be able to run `python -m src.defense.train_baseline` and get
a fresh `models/baseline/ids_baseline.pkl` at any time — that's the contract
the other three roles build against. Keep it working.
