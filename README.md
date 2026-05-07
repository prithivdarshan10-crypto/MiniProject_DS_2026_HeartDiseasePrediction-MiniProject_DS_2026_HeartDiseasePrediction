# Predict & Explain: Identifying Heart Disease Risk Factors Using Patient Clinical Data

> Mini Project – Data Science | SRM Institute of Science and Technology
> Academic Year 2025–2026

---

## Abstract

Heart disease remains one of the leading causes of death globally, yet many cases are preventable with early detection. This project builds a machine learning classification system to predict the probability of heart disease based on clinical features including age, cholesterol levels, blood pressure, chest pain type, and maximum heart rate. Five models are trained and compared — Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, and K-Nearest Neighbours. Exploratory Data Analysis reveals key risk patterns: chest pain type, maximum heart rate, and ST depression (oldpeak) are among the strongest predictors. The best-performing model achieves a competitive F1 score, demonstrating that clinical data can reliably support early heart disease screening.

---

## Problem Statement

Despite being preventable, heart disease remains a leading cause of death worldwide. Clinicians require decision-support tools that not only predict risk but also explain which patient factors contribute most. This project addresses this gap by building an interpretable classification model using patient clinical data, with model evaluation across multiple metrics to ensure reliability and fairness.

**SDG Goal:** Goal 3 – Good Health and Well-being (Target 3.4: Reduce mortality from non-communicable diseases)

---

## Dataset

| Property | Details |
|---|---|
| Source | UCI Machine Learning Repository – Cleveland Heart Disease Dataset |
| Records | 303 patients |
| Features | 13 clinical attributes + 1 target variable |
| Target | 0 = No Heart Disease, 1 = Heart Disease |

**Key Features:**

| Feature | Description |
|---|---|
| age | Age in years |
| sex | Sex (1 = Male, 0 = Female) |
| cp | Chest pain type (0–3) |
| trestbps | Resting blood pressure (mmHg) |
| chol | Serum cholesterol (mg/dl) |
| thalach | Maximum heart rate achieved |
| exang | Exercise-induced angina |
| oldpeak | ST depression induced by exercise |
| target | Heart disease diagnosis (0 = No, 1 = Yes) |

---

## Methodology / Workflow

```
1. Problem Identification
        ↓
2. Dataset Collection (UCI Heart Disease Dataset)
        ↓
3. Data Cleaning & Preprocessing
   - Handle missing values (median imputation)
   - Remove duplicates
   - Feature scaling (StandardScaler)
        ↓
4. Exploratory Data Analysis
   - Target distribution
   - Age & cholesterol analysis
   - Correlation heatmap
   - Feature vs target visualisations
        ↓
5. Model Development
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - Gradient Boosting
   - K-Nearest Neighbours
        ↓
6. Evaluation
   - Accuracy, Precision, Recall, F1, AUC
   - Confusion Matrix
   - ROC Curves
   - Feature Importance
        ↓
7. Result Interpretation & Reporting
```

---

## Results

| Model | Accuracy% | Precision% | Recall% | F1% | AUC% |
|---|---|---|---|---|---|
| Logistic Regression | 60.66 | 59.52 | 78.12 | 67.57 | 59.81 |
| Decision Tree | 47.54 | 50.00 | 65.62 | 56.76 | 47.95 |
| Random Forest | 59.02 | 61.29 | 59.38 | 60.32 | 55.50 |
| Gradient Boosting | 42.62 | 45.16 | 43.75 | 44.44 | 41.81 |
| **K-Nearest Neighbours** | **65.57** | **64.86** | **75.00** | **69.57** | **63.79** |

**Best Model: K-Nearest Neighbours (F1 = 69.57%)**

---

## Tools Used

- **Python 3.11** – Core language
- **Pandas / NumPy** – Data manipulation
- **Scikit-learn** – Model training and evaluation
- **Matplotlib / Seaborn** – Visualisation
- **Jupyter Notebook** – Interactive analysis
- **GitHub** – Version control and collaboration

---

## Project Structure

```
MiniProject_DS_2026_HeartDiseasePrediction/
├── README.md
├── requirements.txt
├── docs/
│   ├── abstract.pdf
│   └── problem_statement.pdf
├── dataset/
│   ├── raw_data/heart_disease_raw.csv
│   └── processed_data/heart_disease_processed.csv
├── notebooks/
│   ├── data_understanding.ipynb
│   ├── preprocessing.ipynb
│   └── visualization.ipynb
├── src/
│   ├── preprocessing.py
│   ├── analysis.py
│   └── model.py
├── outputs/
│   ├── graphs/          ← 10 visualisation plots
│   └── results/         ← model_results.csv, model_results.json
└── report/
    └── mini_project_report.pdf
```

---

## How to Run

```bash
# Clone the repository
git clone https://github.com/prithivdarshan10-crypto/MiniProject_DS_2026_HeartDiseasePrediction.git
cd MiniProject_DS_2026_HeartDiseasePrediction

# Install dependencies
pip install -r requirements.txt

# Run preprocessing
python src/preprocessing.py

# Run EDA and generate graphs
python src/analysis.py

# Train models and evaluate
python src/model.py
```

---

## Team Members

| Name | Register Number | Role |
|---|---|---|
| PRIYADARSHAN S V | RA2311026050153 | Team Leader – Model Development & GitHub |
| MUKESH T | RA2311026050205 | Data Analysis & Visualisation |

**Institution:** SRM Institute of Science and Technology
**Department:** Computer Science and Engineering
**Academic Year:** 2025–2026
