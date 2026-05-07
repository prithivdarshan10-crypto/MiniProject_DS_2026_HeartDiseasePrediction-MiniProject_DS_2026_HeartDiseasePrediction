"""
model.py
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
import os, json

from sklearn.model_selection    import train_test_split, cross_val_score
from sklearn.preprocessing      import StandardScaler
from sklearn.linear_model       import LogisticRegression
from sklearn.ensemble           import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree               import DecisionTreeClassifier
from sklearn.neighbors          import KNeighborsClassifier
from sklearn.metrics            import (accuracy_score, precision_score,
                                         recall_score, f1_score,
                                         confusion_matrix, classification_report,
                                         roc_auc_score, roc_curve)

DATA_PATH    = "dataset/raw_data/heart_disease_raw.csv"
GRAPHS_PATH  = "outputs/graphs"
RESULTS_PATH = "outputs/results"


def load_and_split(test_size=0.2, random_state=42):
    df = pd.read_csv(DATA_PATH)
    df.fillna(df.median(numeric_only=True), inplace=True)

    X = df.drop(columns=["target"])
    y = df["target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y)

    print(f"[Data Split] Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test, X.columns.tolist()


def train_models(X_train, y_train):
    models = {
        "Logistic Regression"      : LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree"            : DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest"            : RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting"        : GradientBoostingClassifier(n_estimators=100, random_state=42),
        "K-Nearest Neighbours"     : KNeighborsClassifier(n_neighbors=7),
    }
    print("\n[Training] Fitting all models ...")
    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f"  ✓ {name}")
    return models


def evaluate_models(models, X_test, y_test):
    results = {}
    for name, model in models.items():
        y_pred  = model.predict(X_test)
        y_prob  = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
        results[name] = {
            "accuracy" : round(accuracy_score(y_test, y_pred) * 100, 2),
            "precision": round(precision_score(y_test, y_pred, zero_division=0) * 100, 2),
            "recall"   : round(recall_score(y_test, y_pred, zero_division=0) * 100, 2),
            "f1"       : round(f1_score(y_test, y_pred, zero_division=0) * 100, 2),
            "auc"      : round(roc_auc_score(y_test, y_prob) * 100, 2),
        }
    return results


def print_results_table(results):
    print("\n" + "=" * 72)
    print(f" {'Model':<28} {'Acc%':>6} {'Prec%':>6} {'Rec%':>6} {'F1%':>6} {'AUC%':>6}")
    print("=" * 72)
    for name, m in results.items():
        print(f" {name:<28} {m['accuracy']:>6} {m['precision']:>6} {m['recall']:>6} {m['f1']:>6} {m['auc']:>6}")
    print("=" * 72)
    best = max(results, key=lambda k: results[k]["f1"])
    print(f"\n Best Model (by F1): {best}  →  F1 = {results[best]['f1']}%\n")
    return best


def plot_model_comparison(results):
    names   = list(results.keys())
    metrics = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(names))
    width = 0.2
    colors = ["#1565C0","#2E7D32","#E65100","#6A1B9A"]

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        vals = [results[n][metric] for n in names]
        bars = ax.bar(x + i*width, vals, width, label=metric.capitalize(),
                      color=color, alpha=0.85, edgecolor="white")
    ax.set_xticks(x + 1.5*width)
    ax.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=8.5)
    ax.set_ylim(0, 115); ax.set_ylabel("Score (%)")
    ax.set_title("Model Comparison – Accuracy / Precision / Recall / F1",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper right"); ax.set_facecolor("#F9F9F9")
    fig.tight_layout()
    fig.savefig(f"{GRAPHS_PATH}/07_model_comparison.png", dpi=150)
    plt.close()
    print("[Graph] Model comparison saved.")


def plot_confusion_matrix(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No Disease", "Disease"],
                yticklabels=["No Disease", "Disease"],
                annot_kws={"size": 14, "weight": "bold"})
    ax.set_title(f"Confusion Matrix – {model_name}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    fig.tight_layout()
    fname = model_name.lower().replace(" ", "_")
    fig.savefig(f"{GRAPHS_PATH}/08_confusion_matrix_{fname}.png", dpi=150)
    plt.close()
    print(f"[Graph] Confusion matrix saved for {model_name}.")


def plot_roc_curves(models, X_test, y_test):
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#1565C0","#2E7D32","#E65100","#6A1B9A","#C62828"]
    for (name, model), color in zip(models.items(), colors):
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc = roc_auc_score(y_test, y_prob)
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})", color=color, linewidth=1.8)
    ax.plot([0,1],[0,1],"k--", linewidth=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves – All Models", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8); ax.set_facecolor("#F9F9F9")
    fig.tight_layout()
    fig.savefig(f"{GRAPHS_PATH}/09_roc_curves.png", dpi=150)
    plt.close()
    print("[Graph] ROC curves saved.")


def plot_feature_importance(model, feature_names):
    if not hasattr(model, "feature_importances_"):
        return
    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(range(len(importances)),
                  importances[indices],
                  color="#1565C0", alpha=0.85, edgecolor="white")
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha="right", fontsize=9)
    ax.set_title("Feature Importance – Random Forest", fontsize=13, fontweight="bold")
    ax.set_ylabel("Importance Score"); ax.set_facecolor("#F9F9F9")
    fig.tight_layout()
    fig.savefig(f"{GRAPHS_PATH}/10_feature_importance.png", dpi=150)
    plt.close()
    print("[Graph] Feature importance saved.")


def save_results(results, best_name):
    os.makedirs(RESULTS_PATH, exist_ok=True)
    with open(f"{RESULTS_PATH}/model_results.json", "w") as f:
        json.dump({"results": results, "best_model": best_name}, f, indent=2)

    rows = [["Model","Accuracy%","Precision%","Recall%","F1%","AUC%"]]
    for name, m in results.items():
        rows.append([name, m["accuracy"], m["precision"], m["recall"], m["f1"], m["auc"]])
    df_res = pd.DataFrame(rows[1:], columns=rows[0])
    df_res.to_csv(f"{RESULTS_PATH}/model_results.csv", index=False)
    print(f"[Saved] Results → {RESULTS_PATH}/")


def run_pipeline():
    os.makedirs(GRAPHS_PATH, exist_ok=True)
    os.makedirs(RESULTS_PATH, exist_ok=True)

    print("=" * 55)
    print(" Model Pipeline – Heart Disease Prediction")
    print("=" * 55)

    X_train, X_test, y_train, y_test, feature_names = load_and_split()
    models  = train_models(X_train, y_train)
    results = evaluate_models(models, X_test, y_test)
    best_name = print_results_table(results)

    plot_model_comparison(results)
    plot_confusion_matrix(models[best_name], X_test, y_test, best_name)
    plot_roc_curves(models, X_test, y_test)
    plot_feature_importance(models["Random Forest"], feature_names)
    save_results(results, best_name)

    print(f"\n[Done] Model pipeline complete. Best: {best_name}")
    return results, best_name


if __name__ == "__main__":
    run_pipeline()
