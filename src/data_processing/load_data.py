"""
Loads and merges the raw CICIDS2018 CSV files into a single dataframe,
memory-efficiently.

Instead of reading all 10 files fully into RAM and concatenating, this
version:
  1. Peeks at just the header row of every file to figure out the common
     columns across all files (cheap -- no data loaded).
  2. Streams through each file in chunks (default 100,000 rows), cleaning
     and filtering each chunk, and appends it directly to the output CSV
     on disk. Only one chunk is ever in memory at a time.

This keeps peak RAM usage low regardless of total dataset size, which
matters for the ~3.9GB single-day files in CICIDS2018.

CICIDS2018's raw CSVs also have a few known quirks this module handles:
  - Column names sometimes have leading/trailing whitespace
    (e.g. " Label" instead of "Label")
  - Not all files have exactly the same columns/column order
  - Some files include extra metadata columns (Flow ID, Src IP, Src Port,
    Dst IP, Timestamp) that others don't
  - Label values sometimes have inconsistent casing/whitespace

Usage:
    from src.data_processing.load_data import merge_csvs_streaming

    merge_csvs_streaming("data/raw/", "data/processed/merged_raw.csv")
"""

import glob
import os
import pandas as pd


# Columns that identify WHO is talking, not HOW they're talking.
# We drop these so the model learns attack *behavior*, not specific
# IPs/ports, which would cause it to overfit to this dataset's topology
# and wouldn't generalize (and wouldn't make sense to attack via FGSM either).
IDENTIFIER_COLUMNS = [
    "Flow ID",
    "Src IP",
    "Source IP",
    "Src Port",
    "Source Port",
    "Dst IP",
    "Destination IP",
    "Timestamp",
]


def _clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from column names so files merge cleanly."""
    df.columns = [c.strip() for c in df.columns]
    return df


def _clean_label_column(df: pd.DataFrame, label_col: str = "Label") -> pd.DataFrame:
    """Strip whitespace/normalize casing issues in the label column."""
    if label_col in df.columns:
        df[label_col] = df[label_col].astype(str).str.strip()
    return df


def _get_common_columns(csv_paths: list) -> list:
    """
    Peek at just the header row of each file (no data loaded) to find
    the columns shared across all files.
    """
    common_columns = None
    for path in csv_paths:
        header_df = pd.read_csv(path, nrows=0)
        header_df = _clean_column_names(header_df)
        cols = set(header_df.columns)
        common_columns = cols if common_columns is None else (common_columns & cols)

    common_columns = sorted(common_columns)

    # Drop identifier columns here too, so they never even get read
    # from disk in the streaming pass below.
    common_columns = [c for c in common_columns if c not in IDENTIFIER_COLUMNS]

    return common_columns


def merge_csvs_streaming(
    raw_data_dir: str,
    output_path: str,
    chunksize: int = 100_000,
) -> None:
    """
    Stream-merge every CSV in raw_data_dir into a single output CSV,
    processing one chunk at a time to keep memory usage low.

    Each file is read in chunks of `chunksize` rows. Each chunk is
    cleaned (whitespace stripped, restricted to common+non-identifier
    columns) and appended directly to output_path.
    """
    csv_paths = sorted(glob.glob(os.path.join(raw_data_dir, "*.csv")))

    if not csv_paths:
        raise FileNotFoundError(
            f"No CSV files found in {raw_data_dir}. "
            f"Did the download finish, and is the path correct?"
        )

    print(f"Found {len(csv_paths)} CSV files.\n")

    print("Pass 1: reading headers only to find common columns...")
    keep_columns = _get_common_columns(csv_paths)
    print(f"{len(keep_columns)} columns will be kept (common across all files, "
          f"identifier columns excluded).\n")

    # Make sure output starts fresh
    if os.path.exists(output_path):
        os.remove(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    total_rows_written = 0
    label_counts = {}

    print("Pass 2: streaming each file in chunks, cleaning, appending to disk...")
    for path in csv_paths:
        print(f"\nProcessing {path} ...")
        file_rows = 0

        for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
            chunk = _clean_column_names(chunk)
            chunk = _clean_label_column(chunk)

            # Remove duplicate/embedded header rows (a known CICIDS2018 quirk where
            # # the header row sometimes repeats partway through a file's data)
            if "Label" in chunk.columns:
                chunk = chunk[chunk["Label"] != "Label"]

            # Restrict to the common/non-identifier columns only.
            # (Some files may be missing a column another has -- this
            # keep_columns list is already the safe intersection.)
            available = [c for c in keep_columns if c in chunk.columns]
            chunk = chunk[available]

            # Append this chunk straight to disk. Header only on first write.
            write_header = not os.path.exists(output_path)
            chunk.to_csv(output_path, mode="a", header=write_header, index=False)

            file_rows += len(chunk)
            total_rows_written += len(chunk)

            if "Label" in chunk.columns:
                for label, count in chunk["Label"].value_counts().items():
                    label_counts[label] = label_counts.get(label, 0) + count

        print(f"  -> {file_rows:,} rows written from this file")

    print(f"\nDone. {total_rows_written:,} total rows written to {output_path}")
    print("\nLabel distribution across full merged dataset:")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"  {label}: {count:,}")


if __name__ == "__main__":
    # Adjust chunksize down (e.g. 50_000) if you're on a machine with
    # limited RAM, or up if you have RAM to spare and want fewer disk writes.
    merge_csvs_streaming(
        raw_data_dir="data/raw/",
        output_path="data/processed/merged_raw.csv",
        chunksize=100_000,
    )