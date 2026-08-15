# Baseline IDS Model — Findings

Author: Defender (Team Lead)
Date: [fill in date]

## Dataset

- Source: CICIDS2018, all 10 processed CSV files from the official AWS S3 bucket
- After merging: 16,233,002 total rows across 15 original labels
- Selected 8 classes for this baseline (excluded very low-sample classes like SQL Injection at 87 rows, which cannot support meaningful training or adversarial evaluation)
- Class imbalance addressed via downsampling: majority class (Benign, originally 13.48M rows) capped at ~300,000 rows; minority classes kept in full
- Final dataset: 2,025,039 rows across 8 classes (160k–300k rows each)
- Split: 70% train (1,417,527) / 15% val (303,756) / 15% test (303,756), stratified by class
- Features: 78 numeric flow-level features from CICFlowMeter (identifier columns — IP, port, timestamp — excluded to prevent the model from learning topology instead of behavior)

## Model

- Architecture: feedforward MLP, 78 → 128 → 64 → 32 → 8, ReLU activations, 0.3 dropout
- Training: class-weighted CrossEntropyLoss, Adam optimizer (lr=0.001), early stopping on validation loss
- Chosen over Random Forest/XGBoost specifically because MLPs are differentiable, which the project's FGSM adversarial attack phase requires directly (no surrogate model needed)

## Results

**Overall test accuracy: 91.45%**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Benign | 0.87 | 0.49 | 0.63 |
| Bot | 1.00 | 1.00 | 1.00 |
| DDOS attack-HOIC | 1.00 | 1.00 | 1.00 |
| DDoS attacks-LOIC-HTTP | 1.00 | 1.00 | 1.00 |
| DoS attacks-Hulk | 1.00 | 1.00 | 1.00 |
| FTP-BruteForce | 1.00 | 1.00 | 1.00 |
| Infilteration | 0.48 | 0.86 | 0.62 |
| SSH-Bruteforce | 1.00 | 1.00 | 1.00 |

## Key finding: Benign/Infiltration confusion

6 of 8 classes achieve near-perfect classification (precision and recall ≥ 0.99). The two exceptions, Benign and Infiltration, are confused specifically with each other — of 44,563 actual Benign test samples, 22,599 (~51%) were misclassified as Infiltration; of 24,096 actual Infiltration samples, 3,272 (~14%) were misclassified as Benign. No other class pairs showed meaningful confusion.

This was investigated via feature-level analysis rather than left unexplained: comparing per-feature mean values between the two classes (in scaled feature space) showed a maximum separation of only 0.45 standard deviations across all 78 features (`Fwd Pkt Len Std` was the most differentiating feature). This indicates substantial distributional overlap between the two classes in this feature space, rather than a model training deficiency.

This finding is consistent with the nature of infiltration attacks: they are deliberately designed as slow, low-volume, stealthy connections that mimic legitimate traffic to evade detection during real-world lateral movement — the model has correctly learned that these two classes are difficult to separate because, at the flow-statistics level, they largely are. This is also a well-documented characteristic of the CICIDS2018 dataset in published intrusion-detection literature, not an artifact specific to this implementation.

## Implication for the adversarial phase

Since 6 of 8 classes are cleanly and confidently classified, they provide a strong, well-separated baseline for testing FGSM evasion — a meaningful drop in accuracy on these classes after adversarial perturbation will clearly demonstrate the attack's effectiveness. The Benign/Infiltration pair should be interpreted carefully in adversarial results, since the baseline model already treats them as a weak boundary; evasion success on this pair is less informative than evasion success on the 6 well-separated classes.
