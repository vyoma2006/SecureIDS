# Setup Guide

For any team member setting up the project fresh.

## 1. Clone the repo

```bash
git clone <repo-url>
cd SecureIDS
```

## 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Download the dataset

Full instructions in `data/README.md`. Short version:

```bash
aws s3 sync --no-sign-request --region us-east-1 \
  "s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/" \
  "./data/raw/"
```

This pulls ~6-7GB of CSVs into `data/raw/`.

## 5. Run the data pipeline

```bash
python -m src.data_processing.load_data
python -m src.data_processing.preprocess
```

Note: `preprocess.py` has a `max_rows_per_class` setting (default 300,000) to control memory usage — lower it if you're on a machine with limited RAM (tested working on 8GB).

## 6. Train the baseline model

```bash
python -m src.defender.train_baseline
```

Takes a while on CPU (no GPU required, but slower) — expect anywhere from 20 minutes to over an hour depending on your machine, though early stopping usually kicks in before all 30 epochs finish.

## 7. Run the API (once endpoints are implemented)

```bash
uvicorn api.main:app --reload
```

Visit `http://127.0.0.1:8000/docs` for interactive endpoint docs.

## 8. Run the dashboard (once built)

```bash
streamlit run frontend/dashboard_app.py
```

## If something goes wrong

- **Out of memory during preprocessing**: lower `max_rows_per_class` or `chunksize` in `preprocess.py`
- **torch install issues**: see `requirements.txt` notes — CPU-only install is the default and should work without a GPU
- **Missing files errors**: make sure you ran steps 5 and 6 in order — later steps depend on earlier ones' saved artifacts