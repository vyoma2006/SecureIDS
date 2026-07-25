# Data

This folder is intentionally empty in git (see root `.gitignore`) because CICIDS2018 is too large to commit.

## Download

CICIDS2018 (CSE-CIC-IDS2018) is available from the Canadian Institute for Cybersecurity:
https://www.unb.ca/cic/datasets/ids-2018.html

## Folder layout

- `raw/` — original downloaded CSVs, untouched
- `processed/` — cleaned/encoded/normalized data, output of `src/data_processing/`
- `adversarial/` — FGSM-generated adversarial samples, output of `src/attacker/generate_adversarial.py`

## Sharing large files with the team

Since the dataset and trained models aren't in git, agree on one shared location, e.g.:
- A shared Google Drive / OneDrive folder, linked here, or
- Git LFS if you want raw/processed data versioned alongside code

(Team: fill in the actual shared link here once you've picked one.)
