# SecureIDS — Attacker (FGSM) Work Report

**Role:** Attacker (FGSM)
**Branch:** `feature/attacker-fgsm`
**Pipeline stage:** Step 3 — `src/attacker/generate_adversarial.py` loads the baseline model and crafts FGSM adversarial samples.

---

## 1. Summary

As Attacker, my job was to implement the Fast Gradient Sign Method (FGSM) against the Defender's frozen baseline IDS model, measure how well the perturbed traffic evades detection, and hand off the results in a format both the Defender (for adversarial retraining) and the Visualizer (for the dashboard) can use directly.

The work is split into four focused modules under `src/attacker/`, consumes five frozen artifacts from the Defender, and produces adversarial CSVs plus two results exports (CSV + JSON).

---

## 2. Inputs Consumed — Defender Model Contract

| Artifact | Path | Purpose |
|---|---|---|
| Model weights | `src/defender/saved_models/baseline_model.pt` | PyTorch `state_dict` |
| Model architecture | `src/defender/saved_models/model_architecture.json` | Rebuilds model shape before loading weights |
| Feature scaler | `src/defender/saved_models/scaler.pkl` | `StandardScaler` fit on training data |
| Label encoder | `src/defender/saved_models/label_encoder.pkl` | Maps class name ↔ integer label |
| Feature column order | `src/defender/saved_models/feature_columns.json` | Exact ordered list of 78 feature names |
| Test features | `data/processed/X_test.csv` | Held-out, pre-scaled test features |
| Test labels | `data/processed/y_test.csv` | Held-out labels, column `label_encoded` |

All five model artifacts were used exactly as specified in the contract, with no modification to the frozen baseline.

---

## 3. Module Structure

```
src/attacker/
├── __init__.py
├── perturbation_utils.py   → loads model + data, prepares tensors
├── fgsm_attack.py          → core FGSM algorithm
├── evasion_metrics.py      → evasion rate + confidence drop
└── generate_adversarial.py → entry point, orchestrates everything
```

| File | Responsibility |
|---|---|
| `perturbation_utils.py` | Loads model weights, architecture, scaler, label encoder, feature columns; builds `X`/`y` tensors from `X_test.csv` / `y_test.csv` |
| `fgsm_attack.py` | Implements `x_adv = x + epsilon * sign(∇x loss)` using `torch.autograd` |
| `evasion_metrics.py` | Computes evasion rate (misclassified fraction) and confidence drop on the true class |
| `generate_adversarial.py` | CLI entry point — loops over classes × epsilons, runs the attack, saves outputs |

---

## 4. Attack Methodology

- Loaded the frozen `IDS_MLP` model, reconstructed from `model_architecture.json`, then loaded `baseline_model.pt` weights.
- Confirmed `X_test.csv` was already scaled by the data-processing pipeline (mean ≈ 0, std ≈ 1), so `scaler.transform()` was not reapplied — avoiding a double-scaling error.
- Confirmed `y_test.csv`'s `label_encoded` column already holds integer labels matching `label_encoder.pkl`'s ordering (`0=Benign … 7=SSH-Bruteforce`), so no re-encoding was needed.
- Reordered feature columns to exactly match `feature_columns.json` before building tensors, preserving the required 78-column order.
- Implemented FGSM using standard autograd:
  ```python
  def fgsm_attack(model, x, y_true, epsilon):
      x = x.clone().detach().requires_grad_(True)
      logits = model(x)
      loss = F.cross_entropy(logits, y_true)
      model.zero_grad()
      loss.backward()
      return (x + epsilon * x.grad.sign()).detach()
  ```
- Ran the attack per attack class (excluding Benign as a source class) across an epsilon sweep: `0.01, 0.05, 0.1, 0.3`.

**Benign/Infiltration caveat:** The Defender flagged that the baseline model already struggles to tell Benign and Infiltration traffic apart, even before any attack. So results for this pair are reported separately rather than folded into the main evasion numbers — a "successful evasion" here doesn't say much when the model was already shaky on these two to begin with. The stronger, more meaningful evidence is in the other 6 classes, which the model classifies cleanly and confidently before any perturbation.

---

## 5. Outputs Produced

| Output | Location | Format |
|---|---|---|
| Adversarial samples | `data/adversarial/fgsm_<class>_<epsilon>.csv` | Scaled, 78-column — matches training data shape, consumable directly by `adversarial_training.py` |
| Summary results (CSV) | `results/reports/attacker_findings.csv` | `class, epsilon, evasion_rate, confidence_drop, n_samples` |
| Summary results (JSON) | `results/metrics/evasion_metrics.json` | Same fields, JSON array — for dashboard/API consumption by the Visualizer |

---

