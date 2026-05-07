"""
analysis.py
Mini Project – Heart Disease Prediction
Team: PRIYADARSHAN S V (RA2311026050153), MUKESH T (RA2311026050205)
SRM Institute of Science and Technology
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

DATA_PATH   = "dataset/raw_data/heart_disease_raw.csv"
GRAPHS_PATH = "outputs/graphs"

FEATURE_DESC = {
    "age":      "Age (years)",
    "sex":      "Sex (1=Male, 0=Female)",
    "cp":       "Chest Pain Type (0–3)",
    "trestbps": "Resting Blood Pressure (mmHg)",
    "chol":     "Serum Cholesterol (mg/dl)",
    "fbs":      "Fasting Blood Sugar > 120 mg/dl",
    "restecg":  "Resting ECG Results",
    "thalach":  "Max Heart Rate Achieved",
    "exang":    "Exercise Induced Angina",
    "oldpeak":  "ST Depression (Exercise vs Rest)",
    "slope":    "Slope of Peak Exercise ST Segment",
    "ca":       "Number of Major Vessels (0–3)",
    "thal":     "Thalassemia Type",
    "target":   "Heart Disease (1=Yes, 0=No)",
}


def load_data():
    df = pd.read_csv(DATA_PATH)
    df.fillna(df.median(numeric_only=True), inplace=True)
    return df


def plot_target_distribution(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["target"].value_counts()
    colors = ["#42A5F5", "#EF5350"]
    ax.bar(["No Disease", "Heart Disease"], counts.values, color=colors, edgecolor="white", width=0.5)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 2, str(v), ha="center", fontweight="bold", fontsize=11)
    ax.set_title("Target Variable Distribution", fontsize=14, fontweight="bold")
    ax.set_ylabel("Count")
    ax.set_facecolor("#F9F9F9")
    fig.tight_layout()
    fig.savefig(f"{GRAPHS_PATH}/01_target_distribution.png", dpi=150)
    plt.close()
    print("[Graph] Target distribution saved.")


def plot_age_distribution(df):
    fig, ax = plt.subplots(figsize=(8, 4))
    for target, color, label in [(0, "#42A5F5", "No Disease"), (1, "#EF5350", "Heart Disease")]:
        ax.hist(df[df["target"] == target]["age"], bins=20, alpha=0.65,
                color=color, label=label, edgecolor="white")
    ax.set_title("Age Distribution by Heart Disease Status", fontsize=13, fontweight="bold")
    ax.set_xlabel("Age"); ax.set_ylabel("Count")
    ax.legend(); ax.set_facecolor("#F9F9F9")
    fig.tight_layout()
    fig.savefig(f"{GRAPHS_PATH}/02_age_distribution.png", dpi=150)
    plt.close()
    print("[Graph] Age distribution saved.")


def plot_correlation_heatmap(df):
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.5, ax=ax, annot_kws={"size": 8})
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{GRAPHS_PATH}/03_correlation_heatmap.png", dpi=150)
    plt.close()
    print("[Graph] Correlation heatmap saved.")


def plot_chest_pain_vs_target(df):
    fig, ax = plt.subplots(figsize=(7, 4))
    cp_counts = df.groupby(["cp", "target"]).size().unstack(fill_value=0)
    cp_counts.plot(kind="bar", ax=ax, color=["#42A5F5", "#EF5350"],
                   edgecolor="white", width=0.6)
    ax.set_title("Chest Pain Type vs Heart Disease", fontsize=13, fontweight="bold")
    ax.set_xlabel("Chest Pain Type"); ax.set_ylabel("Count")
    ax.legend(["No Disease", "Heart Disease"])
    ax.set_facecolor("#F9F9F9"); ax.tick_params(axis='x', rotation=0)
    fig.tight_layout()
    fig.savefig(f"{GRAPHS_PATH}/04_chest_pain_vs_target.png", dpi=150)
    plt.close()
    print("[Graph] Chest pain vs target saved.")


def plot_cholesterol_boxplot(df):
    fig, ax = plt.subplots(figsize=(6, 5))
    data = [df[df["target"] == 0]["chol"].dropna(),
            df[df["target"] == 1]["chol"].dropna()]
    bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                    boxprops=dict(facecolor="#BBDEFB"),
                    medianprops=dict(color="#E53935", linewidth=2))
    ax.set_xticklabels(["No Disease", "Heart Disease"])
    ax.set_title("Cholesterol Distribution by Target", fontsize=13, fontweight="bold")
    ax.set_ylabel("Cholesterol (mg/dl)"); ax.set_facecolor("#F9F9F9")
    fig.tight_layout()
    fig.savefig(f"{GRAPHS_PATH}/05_cholesterol_boxplot.png", dpi=150)
    plt.close()
    print("[Graph] Cholesterol boxplot saved.")


def plot_max_hr_scatter(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    for target, color, label in [(0, "#42A5F5", "No Disease"), (1, "#EF5350", "Heart Disease")]:
        sub = df[df["target"] == target]
        ax.scatter(sub["age"], sub["thalach"], alpha=0.55, color=color,
                   label=label, s=40, edgecolors="white", linewidths=0.4)
    ax.set_title("Age vs Max Heart Rate by Disease Status", fontsize=13, fontweight="bold")
    ax.set_xlabel("Age"); ax.set_ylabel("Max Heart Rate")
    ax.legend(); ax.set_facecolor("#F9F9F9")
    fig.tight_layout()
    fig.savefig(f"{GRAPHS_PATH}/06_age_vs_maxhr.png", dpi=150)
    plt.close()
    print("[Graph] Age vs Max HR scatter saved.")


def print_eda_summary(df):
    print("\n" + "=" * 55)
    print(" EDA Summary – Heart Disease Dataset")
    print("=" * 55)
    print(f"  Total Records  : {len(df)}")
    print(f"  Total Features : {len(df.columns) - 1}")
    print(f"  Missing Values : {df.isnull().sum().sum()}")
    print(f"  Disease Cases  : {(df['target'] == 1).sum()} ({(df['target']==1).mean()*100:.1f}%)")
    print(f"  Healthy Cases  : {(df['target'] == 0).sum()} ({(df['target']==0).mean()*100:.1f}%)")
    print(f"  Avg Age        : {df['age'].mean():.1f}")
    print(f"  Avg Cholesterol: {df['chol'].mean():.1f} mg/dl")
    print(f"  Avg Max HR     : {df['thalach'].mean():.1f}")
    print("=" * 55)


def run_eda():
    os.makedirs(GRAPHS_PATH, exist_ok=True)
    df = load_data()
    print_eda_summary(df)
    plot_target_distribution(df)
    plot_age_distribution(df)
    plot_correlation_heatmap(df)
    plot_chest_pain_vs_target(df)
    plot_cholesterol_boxplot(df)
    plot_max_hr_scatter(df)
    print(f"\n[Done] All graphs saved to {GRAPHS_PATH}/")
    return df


if __name__ == "__main__":
    run_eda()
