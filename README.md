# Credit Card Fraud Detection

---

## Overview

Credit card fraud costs the financial industry billions every year — and detecting it is hard, especially when only **492 out of 284,807 transactions (~0.17%)** are actually fraudulent. That extreme imbalance is exactly what makes this problem interesting.

This project covers the full pipeline: EDA, preprocessing, training 4 models, threshold tuning, and SHAP explainability.

**Dataset:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) · `V1`–`V28` (PCA) + `Amount` + `Time` + `Class`

| File | Description |
|---|---|
| `EDA.ipynb` | Class distribution, correlations, PCA-component distributions |
| `ML_FD.py` | Full pipeline: preprocessing → SMOTE → training → evaluation → SHAP |
| `Data/creditcard.csv` | Raw transaction dataset |
| `Models/` | Saved best model + scaler |
| `Visualizations/` | Confusion matrices, ROC/PR curves, threshold plot, SHAP plots |

---

## Technical Approach

- **Scaling:** `StandardScaler` on `Amount` and `Time` only (`V1`–`V28` are already normalized)
- **Imbalance:** SMOTE (`k=5`) applied on training data only — test set stays untouched
- **Models:** Logistic Regression, Random Forest, XGBoost, LightGBM
- **Metrics:** PR-AUC (primary), ROC-AUC, Precision, Recall, F1 — accuracy is useless here
- **Threshold Tuning:** Swept 0.01→0.99, picked the threshold that maximizes fraud F1
- **SHAP:** `TreeExplainer` on the best tree booster — beeswarm + bar plots + top-5 feature analysis

---

## Results

### Model Performance Summary

| Model | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| **Random Forest** | **0.96** | **0.81** | **0.88** | **0.98** | **0.88** |
| LightGBM | 0.94 | 0.80 | 0.86 | 0.98 | 0.86 |
| XGBoost | 0.93 | 0.79 | 0.85 | 0.97 | 0.85 |
| Logistic Regression | 0.07 | 0.93 | 0.12 | 0.97 | 0.72 |

> **Random Forest** was the best model and was saved as the final model.

### Threshold Tuning

| Setting | Threshold | Fraud Precision | Fraud Recall | Fraud F1 |
|---|---|---|---|---|
| Default | 0.50 | 0.96 | 0.81 | 0.88 |
| **Optimal** | **~0.35** | **~0.87** | **~0.88** | **~0.88** |

### Top 5 Features (SHAP)

| Feature | Direction |
|---|---|
| V14 | Lower values → higher fraud score |
| V4 | Higher values → higher fraud score |
| V11 | Lower values → higher fraud score |
| V12 | Lower values → higher fraud score |
| V10 | Lower values → higher fraud score |

---

