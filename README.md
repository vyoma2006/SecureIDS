# SecureIDS
**Building a Robust Network Intrusion Detection System with Adversarial Attack Simulation and Defense**

SecureIDS trains an ML-based Intrusion Detection System (IDS) on the CICIDS2018 dataset, simulates adversarial evasion attacks using FGSM adapted for tabular data, and applies adversarial training to measurably improve robustness. Results are exposed through a FastAPI backend and visualized in an interactive dashboard.

## Team & Roles

| Role | Owns | Folder(s) |
|---|---|---|
| **Defender (Team Lead)** | Baseline IDS model, adversarial retraining, evaluation | `src/defender/`, `api/routers/defender_routes.py` |
| **Attacker** | FGSM adversarial sample generation, evasion metrics | `src/attacker/`, `api/routers/attacker_routes.py` |
| **Visualizer** | Dashboard, charts, metrics aggregation | `frontend/`, `api/routers/metrics_routes.py` |
| **Cybersecurity Expert (Validator)** | System validation, threat mapping, sign-off report | `src/validator/`, `api/routers/validator_routes.py` |

## Project Structure

```
SecureIDS/
├── data/               # Raw, processed, and adversarial datasets (not committed — see data/README.md)
├── src/                # Core logic: data processing, defender, attacker, validator
├── api/                # FastAPI backend connecting all modules
├── frontend/            # Dashboard (Streamlit) consuming the API
├── notebooks/          # Exploratory work per person
├── results/            # Metrics, figures, reports
├── tests/              # Unit + API tests
├── configs/            # Central config (paths, hyperparams, API settings)
└── docs/               # Architecture, pipeline, API reference, setup guide
```

## Setup

```bash
# 1. Clone and enter the repo
git clone <your-repo-url>
cd SecureIDS

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download CICIDS2018 data — see data/README.md for the link
#    Place raw files in data/raw/

# 5. Run the API
uvicorn api.main:app --reload
# Swagger docs at http://127.0.0.1:8000/docs

# 6. Run the dashboard (in a separate terminal)
streamlit run frontend/dashboard_app.py
```

## Pipeline Overview

1. `src/data_processing/` cleans CICIDS2018 into a shared processed dataset.
2. `src/defender/train_baseline.py` trains the baseline Random Forest / XGBoost IDS.
3. `src/attacker/generate_adversarial.py` loads the baseline model and crafts FGSM adversarial samples.
4. `src/defender/adversarial_training.py` retrains the model on original + adversarial samples.
5. `frontend/dashboard_app.py` calls the API to visualize before/after accuracy, detection rate, and evasion rate.
6. `src/validator/` independently checks the pipeline and produces the final validation report.

See `docs/architecture.md` and `docs/attack_defense_pipeline.md` for details.

## Branching Convention

- `main` — stable, working code only
- `feature/defender-*`, `feature/attacker-*`, `feature/visualizer-*`, `feature/validator-*` — per-person work
- Open a PR into `main` when ready for review
