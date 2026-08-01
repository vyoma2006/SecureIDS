import pandas as pd
import numpy as np
import json

X_train = pd.read_csv("data/processed/X_train.csv")
y_train = pd.read_csv("data/processed/y_train.csv")

with open("src/defender/saved_models/feature_columns.json") as f:
    feature_names = json.load(f)

X_train.columns = feature_names
df = X_train.copy()
df["label"] = y_train.values

# Compare Benign (0) vs Infiltration (6) specifically
benign = df[df["label"] == 0]
infil = df[df["label"] == 6]

diffs = (benign.drop(columns="label").mean() - infil.drop(columns="label").mean()).abs()
print(diffs.sort_values(ascending=False).head(15))