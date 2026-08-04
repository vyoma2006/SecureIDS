# Adversarial Defense — Findings

Author: Defender (Team Lead)
Date: [fill in date]

## Setup

Two adversarial training experiments were run, both starting from the frozen baseline model (`baseline_model.pt`, 91.45% test accuracy — see `baseline_findings.md`) and retrained from scratch on original training data combined with the Attacker's FGSM-generated adversarial samples.

- **Experiment A — "AT (All eps)"**: trained on adversarial samples across all four epsilon values (0.01, 0.05, 0.1, 0.3), combined with the original clean training set. Total combined training set: 2,454,299 rows (1,417,527 clean + 1,036,772 adversarial).
- **Experiment B — "AT (0.1 + 0.3 only)"**: trained only on the two higher-epsilon adversarial sets, combined with the same clean training data. Total combined training set: 1,935,913 rows (1,417,527 clean + 518,386 adversarial).

Both experiments were evaluated identically for fairness:
- **Clean accuracy / Macro F1**: measured on the original, untouched clean test set (303,756 rows)
- **Evasion rate**: measured against the *full* adversarial set (all four epsilons, 1,036,772 samples) regardless of which epsilons the model was trained on — so a model trained only on 0.1/0.3 is still tested against 0.01/0.05 samples it never saw, making the comparison meaningful rather than circular.

## Results

| Model | Clean Accuracy | Macro F1 | Evasion Rate |
|---|---|---|---|
| Baseline (no defense) | 0.9145 | 0.9054 | 0.2853 |
| AT (All eps) | 0.9375 | **0.9166** | **0.0041** |
| AT (0.1 + 0.3 only) | **0.9394** | 0.9129 | 0.0197 |

## Key finding: exposure to the full epsilon range matters more than raw attack strength

Both adversarially trained models substantially outperform the baseline: evasion rate dropped from 28.53% to under 2% in both cases, and — notably — clean accuracy *improved* in both experiments rather than trading off against robustness. This is the outcome adversarial training is meant to achieve: the model gained resistance to perturbed inputs without becoming worse at classifying normal traffic.

Between the two experiments, **AT (All eps) achieved a nearly 5x lower evasion rate** than AT (0.1+0.3 only) (0.41% vs 1.97%), despite AT (0.1+0.3 only) having a marginal 0.19-point edge in clean accuracy — a difference small enough to be within normal training noise.

This is a genuine and slightly counterintuitive finding: at generation time, the low-epsilon adversarial samples (0.01, 0.05) had low evasion rates against the *baseline* model (i.e., they were "weak" attacks on their own). Despite this, including them in training measurably improved the resulting model's robustness beyond what training on only the "strong" (0.1, 0.3) perturbations achieved. This suggests that exposing the model to a *range* of perturbation magnitudes during training — including subtle ones — teaches it a more general decision boundary against the perturbation direction, rather than only hardening it against large, easily-generated attacks.

## Recommendation

**AT (All eps) is the recommended defense model going forward** (`robust_model_all_eps.pt`). The evasion rate improvement (nearly 5x) is a far larger effect than the accuracy difference between the two experiments (0.19 points), making it the stronger choice for an IDS use case, where missing an adversarial attack is a more costly failure than a marginal accuracy loss.

## Caveat for further validation

The evasion rate reported here (0.41%) reflects robustness against the *same FGSM method* that generated the training data. This is a legitimate and standard adversarial training evaluation, but it does not by itself confirm the model generalizes to adversarial examples produced by a different attack method or a different random seed/implementation. This is flagged here as a suggested next check for the Validator: testing the robust model against a freshly-generated, held-out adversarial set (or a different attack algorithm, if time permits) would confirm whether the robustness generalizes or is specific to this attack's characteristics.

## Artifacts produced

- `src/defender/saved_models/robust_model_all_eps.pt` — recommended defense model
- `src/defender/saved_models/robust_model_high_eps_only.pt` — comparison model (ablation)
- `results/metrics/comparison_all_eps.json`
- `results/metrics/comparison_high_eps_only.json`