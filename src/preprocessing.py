"""
preprocessing.py
Mini Project – Heart Disease Prediction
Team: PRIYADARSHAN S V (RA2311026050153), MUKESH T (RA2311026050205)
SRM Institute of Science and Technology
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

RAW_PATH  = "dataset/raw_data/heart_disease_raw.csv"
PROC_PATH = "dataset/processed_data/heart_disease_processed.csv"


def load_data(path=RAW_PATH):
    df = pd.read_csv(path)
    print(f"[Load] Shape: {df.shape}")
    print(f"[Load] Columns: {list(df.columns)}")
    return df


def handle_missing_values(df):
    print(f"\n[Missing Values]\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    # Fill numerical nulls with median
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"  Filled '{col}' nulls with median = {median_val:.2f}")
    return df


def remove_duplicates(df):
    before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"\n[Duplicates] Removed {before - len(df)} duplicate rows.")
    return df


def encode_features(df):
    # Binary columns are already 0/1; no additional encoding needed for this dataset
    print("\n[Encoding] All features are already numeric.")
    return df


def scale_features(df, target_col="target"):
    features = df.drop(columns=[target_col])
    target   = df[target_col]

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    df_scaled = pd.DataFrame(scaled, columns=features.columns)
    df_scaled[target_col] = target.values

    print(f"\n[Scaling] StandardScaler applied to {len(features.columns)} features.")
    return df_scaled


def generate_summary(df):
    print("\n[Summary Statistics]")
    print(df.describe().round(2))

    print(f"\n[Target Distribution]")
    counts = df["target"].value_counts()
    print(f"  No Disease (0): {counts.get(0, 0)}")
    print(f"  Disease    (1): {counts.get(1, 0)}")


def run_pipeline():
    print("=" * 55)
    print(" Preprocessing Pipeline – Heart Disease Dataset")
    print("=" * 55)

    df = load_data()
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = encode_features(df)
    generate_summary(df)
    df_scaled = scale_features(df)

    os.makedirs(os.path.dirname(PROC_PATH), exist_ok=True)
    df_scaled.to_csv(PROC_PATH, index=False)
    print(f"\n[Saved] Processed data → {PROC_PATH}")
    print("\n[Done] Preprocessing complete.")
    return df_scaled


if __name__ == "__main__":
    run_pipeline()
