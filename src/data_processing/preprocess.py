"""
Cleans the merged dataset and prepares train/val/test splits ready for
MLP training -- memory-efficient version for machines with limited RAM
(written/tested with an 8GB RAM constraint in mind).

Two-stage design:
  STAGE A (streaming): read merged_raw.csv in chunks, clean each chunk
  (drop inf/NaN/duplicates), filter to selected classes, and DOWNSAMPLE
  the majority class (Benign) as we go. Write the much smaller result to
  data/processed/filtered.csv. Only one chunk is ever fully in memory.

  STAGE B (in-memory): the filtered file is now small enough to load
  fully, so we do the train/val/test split, label encoding, and scaling
  on it directly, then save the shared artifacts your team depends on:
      - scaler.pkl            (StandardScaler fit on training data)
      - label_encoder.pkl      (maps class name <-> integer)
      - feature_columns.json   (exact ordered list of feature columns)

Do not change the shape/order of these once the team is building against
them -- if you need to change selected classes or features, regenerate
all three together and let the team know.

Usage:
    python -m src.data_processing.preprocess
"""

import json
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


# ---------------------------------------------------------------------------
# STAGE A: streaming filter + clean + downsample
# ---------------------------------------------------------------------------

def filter_and_downsample_streaming(
    merged_csv_path: str,
    output_path: str,
    selected_labels: list,
    label_col: str = "Label",
    max_rows_per_class: int = 300_000,
    chunksize: int = 200_000,
    random_state: int = 42,
) -> None:
    """
    Stream through merged_csv_path, keep only rows in selected_labels,
    clean inf/NaN, downsample any class that exceeds max_rows_per_class,
    and write the result to output_path.

    Downsampling is done via per-chunk random sampling: each chunk keeps
    roughly the fraction of a class needed to hit the overall cap by the
    end. This is approximate (not an exact cap) but keeps memory flat,
    since we never need to hold the full class in memory to sample from it.
    """
    rng = np.random.default_rng(random_state)

    if os.path.exists(output_path):
        os.remove(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Rough estimate of each selected class's total size, so we know what
    # fraction to keep per chunk. Cheap: just count matching label values
    # via a fast streaming pass (still faster/lighter than loading full data).
    print("Estimating class sizes for downsampling ratios...")
    class_counts = {label: 0 for label in selected_labels}
    for chunk in pd.read_csv(merged_csv_path, usecols=[label_col], chunksize=chunksize):
        vc = chunk[label_col].value_counts()
        for label in selected_labels:
            if label in vc:
                class_counts[label] += int(vc[label])

    print("Estimated counts:", class_counts)

    keep_fraction = {
        label: min(1.0, max_rows_per_class / count) if count > 0 else 1.0
        for label, count in class_counts.items()
    }
    print("Keep fraction per class (1.0 = keep everything):")
    for label, frac in keep_fraction.items():
        print(f"  {label}: {frac:.4f}")

    print("\nStreaming through file: cleaning, filtering, downsampling...")
    total_written = 0
    written_counts = {label: 0 for label in selected_labels}

    for chunk in pd.read_csv(merged_csv_path, chunksize=chunksize, low_memory=False):
        # Drop inf/NaN
        chunk = chunk.replace([np.inf, -np.inf], np.nan).dropna()

        # Keep only selected classes
        chunk = chunk[chunk[label_col].isin(selected_labels)]
        if chunk.empty:
            continue

        # Downsample each class in this chunk according to its keep_fraction
        sampled_parts = []
        for label in selected_labels:
            class_rows = chunk[chunk[label_col] == label]
            if class_rows.empty:
                continue
            frac = keep_fraction[label]
            if frac >= 1.0:
                sampled_parts.append(class_rows)
            else:
                mask = rng.random(len(class_rows)) < frac
                sampled_parts.append(class_rows[mask])

        if not sampled_parts:
            continue

        chunk_out = pd.concat(sampled_parts, ignore_index=True)

        # Downcast numeric columns to float32 to roughly halve memory
        # footprint from here on out (default pandas is float64).
        numeric_cols = chunk_out.select_dtypes(include=["float64"]).columns
        chunk_out[numeric_cols] = chunk_out[numeric_cols].astype("float32")

        write_header = not os.path.exists(output_path)
        chunk_out.to_csv(output_path, mode="a", header=write_header, index=False)

        total_written += len(chunk_out)
        for label, count in chunk_out[label_col].value_counts().items():
            written_counts[label] = written_counts.get(label, 0) + count

    print(f"\nDone. {total_written:,} rows written to {output_path}")
    print("Final class distribution:")
    for label, count in sorted(written_counts.items(), key=lambda x: -x[1]):
        print(f"  {label}: {count:,}")


# ---------------------------------------------------------------------------
# STAGE B: in-memory split / encode / scale / save (on the now-small file)
# ---------------------------------------------------------------------------

def encode_labels(df: pd.DataFrame, label_col: str = "Label"):
    """Encode string labels to integers. Returns (df, fitted LabelEncoder)."""
    encoder = LabelEncoder()
    df = df.copy()
    df["label_encoded"] = encoder.fit_transform(df[label_col])

    print("\nLabel encoding:")
    for i, class_name in enumerate(encoder.classes_):
        print(f"  {i} -> {class_name}")

    return df, encoder


def split_features_labels(df: pd.DataFrame, label_col: str = "Label"):
    """Separate feature matrix X from label vector y."""
    drop_cols = [label_col, "label_encoded"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df["label_encoded"]
    return X, y


def run_preprocessing_pipeline(
    merged_csv_path: str,
    selected_labels: list,
    output_dir: str = "data/processed",
    model_artifacts_dir: str = "src/defender/saved_models",
    max_rows_per_class: int = 300_000,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_artifacts_dir, exist_ok=True)
    filtered_path = os.path.join(output_dir, "filtered.csv")

    print("=" * 60)
    print("STAGE A: streaming filter + clean + downsample")
    print("=" * 60)
    filter_and_downsample_streaming(
        merged_csv_path=merged_csv_path,
        output_path=filtered_path,
        selected_labels=selected_labels,
        max_rows_per_class=max_rows_per_class,
        random_state=random_state,
    )

    print("\n" + "=" * 60)
    print("STAGE B: load filtered (much smaller) data into memory")
    print("=" * 60)
    df = pd.read_csv(filtered_path, low_memory=False)
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns into memory")

    print("\n" + "=" * 60)
    print("Encode labels")
    print("=" * 60)
    df, label_encoder = encode_labels(df)

    print("\n" + "=" * 60)
    print("Split features/labels")
    print("=" * 60)
    X, y = split_features_labels(df)
    feature_columns = list(X.columns)
    print(f"{len(feature_columns)} feature columns")
    del df  # free memory now that we've split out what we need

    print("\n" + "=" * 60)
    print("Train/val/test split (stratified)")
    print("=" * 60)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    del X, y
    val_ratio_of_temp = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio_of_temp, stratify=y_temp, random_state=random_state
    )
    del X_temp, y_temp
    print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

    print("\n" + "=" * 60)
    print("Scale features (fit on train only)")
    print("=" * 60)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype("float32")
    X_val_scaled = scaler.transform(X_val).astype("float32")
    X_test_scaled = scaler.transform(X_test).astype("float32")

    print("\n" + "=" * 60)
    print("Save everything")
    print("=" * 60)
    pd.DataFrame(X_train_scaled, columns=feature_columns).to_csv(
        os.path.join(output_dir, "X_train.csv"), index=False
    )
    pd.DataFrame(X_val_scaled, columns=feature_columns).to_csv(
        os.path.join(output_dir, "X_val.csv"), index=False
    )
    pd.DataFrame(X_test_scaled, columns=feature_columns).to_csv(
        os.path.join(output_dir, "X_test.csv"), index=False
    )
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_val.to_csv(os.path.join(output_dir, "y_val.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)
    print(f"Saved train/val/test splits to {output_dir}/")

    with open(os.path.join(model_artifacts_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(model_artifacts_dir, "label_encoder.pkl"), "wb") as f:
        pickle.dump(label_encoder, f)
    with open(os.path.join(model_artifacts_dir, "feature_columns.json"), "w") as f:
        json.dump(feature_columns, f, indent=2)
    print(f"Saved scaler.pkl, label_encoder.pkl, feature_columns.json to {model_artifacts_dir}/")

    print("\nDone. Ready for model training.")


if __name__ == "__main__":
    # EDIT THIS LIST to choose which classes to train on for this run.
    SELECTED_LABELS = [
        "Benign",
        "DDOS attack-HOIC",
        "DDoS attacks-LOIC-HTTP",
        "DoS attacks-Hulk",
        "Bot",
        "FTP-BruteForce",
        "SSH-Bruteforce",
        "Infilteration",
    ]

    run_preprocessing_pipeline(
        merged_csv_path="data/processed/merged_raw.csv",
        selected_labels=SELECTED_LABELS,
        max_rows_per_class=300_000,  # caps Benign (and any other huge class) here
    )