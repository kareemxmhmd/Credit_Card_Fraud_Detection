# Credit Card Fraud Detection

---

## Overview

Credit card fraud costs the financial industry billions every year — and detecting it is hard, especially when only 492 out of 284,807 transactions (~0.17%) are actually fraudulent. 

This project builds a complete fraud detection pipeline — starting from EDA and feature engineering, all the way to training multiple models, tuning the decision threshold, and explaining predictions with SHAP.

### Project Files

| File / Folder | Description |
|---|---|
| `EDA.ipynb` | Exploratory data analysis — class distribution, feature correlations, PCA-component distributions |
| `ML_FD.py` | Full ML pipeline: preprocessing → SMOTE → training → evaluation → threshold tuning → SHAP |
| `Data/creditcard.csv` | Raw transaction dataset |
| `Models/Random Forest_model.pkl` | Saved best model |
| `Models/scaler.pkl` | Saved StandardScaler for `Amount` and `Time` |
| `Visualizations/` | Confusion matrices, ROC/PR curves, F1-threshold plot, SHAP plots |

### Dataset

- **Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Rows:** 284,807 transactions
- **Features:** `V1`–`V28` (PCA-anonymized) + `Amount` + `Time`
- **Target:** `Class` (0 = Legitimate, 1 = Fraud)

---

## Technical Approach

### 1. Preprocessing

- **Scaling:** `Amount` and `Time` were standardized using `StandardScaler` and replaced with `scaled_amount` and `scaled_time`. The `V1`–`V28` features are already PCA-transformed and unit-normalized.
- **Train/Test Split:** 80/20 stratified split (`random_state=42`) to preserve the fraud ratio in both sets.

### 2. Handling Class Imbalance — SMOTE

The minority class (fraud) is massively underrepresented. We applied **SMOTE** (`k_neighbors=5`) on the training set only, synthetically generating new fraud samples until both classes are balanced. The test set is kept original (no leakage).

```
Before SMOTE → Fraud: ~394 | Legitimate: ~227,451
After SMOTE  → Fraud: ~227,451 | Legitimate: ~227,451
```

> SMOTE is applied **only on training data** — the test set stays untouched to reflect real-world conditions.

### 3. Models Trained

Four classifiers were trained and evaluated:

| Model | Key Hyperparameters |
|---|---|
| **Logistic Regression** | `class_weight=balanced`, `max_iter=2000`, `solver=lbfgs` |
| **Random Forest** | `n_estimators=250`, `min_samples_leaf=2`, `class_weight=balanced` |
| **XGBoost** | `n_estimators=350`, `max_depth=4`, `lr=0.05`, `subsample=0.9` |
| **LightGBM** | `n_estimators=350`, `num_leaves=31`, `lr=0.05`, `subsample=0.9`, `class_weight=balanced` |

### 4. Evaluation Metrics

For imbalanced fraud detection, **accuracy is misleading**. We focus on:

- **PR-AUC (Average Precision)** — primary metric; measures precision-recall tradeoff across all thresholds
- **ROC-AUC** — overall separability between classes
- **Recall (Fraud class)** — how many actual frauds we catch
- **Precision (Fraud class)** — how many flagged transactions are real fraud
- **F1-Score (Fraud class)** — harmonic mean of precision and recall

### 5. Decision Threshold Tuning

The default 0.5 threshold is not optimal for imbalanced data. We swept thresholds from `0.01` to `0.99` and selected the one that maximizes the **fraud-class F1 score** on the test set.

### 6. SHAP Explainability

We used `shap.TreeExplainer` on the best tree-based model to generate:

- **Beeswarm plot** — shows which features push predictions toward fraud and in which direction
- **Bar plot** — global feature importance by mean absolute SHAP value
- **Top-5 feature analysis** — median feature values for legitimate vs. fraud transactions + SHAP direction

> If Random Forest is the best model, SHAP is computed on the strongest tree booster (XGBoost or LightGBM) instead for more reliable SHAP values.

---

## Results

### Model Performance Summary

> Sorted by PR-AUC (primary metric for imbalanced classification)

| Model | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| **Random Forest** | **0.96** | **0.81** | **0.88** | **0.98** | **0.88** |
| LightGBM | 0.94 | 0.80 | 0.86 | 0.98 | 0.86 |
| XGBoost | 0.93 | 0.79 | 0.85 | 0.97 | 0.85 |
| Logistic Regression | 0.07 | 0.93 | 0.12 | 0.97 | 0.72 |

> **Random Forest** was the best-performing model and was selected as the final saved model.

---

### Best Model: Random Forest

**At default threshold (0.5):**

| Metric | Legitimate | Fraud |
|---|---|---|
| Precision | 1.00 | 0.96 |
| Recall | 1.00 | 0.81 |
| F1-Score | 1.00 | 0.88 |

---

### Threshold Tuning

After sweeping thresholds from 0.01 to 0.99, the **optimal threshold** that maximizes fraud F1 was identified:

| Setting | Threshold | Fraud Precision | Fraud Recall | Fraud F1 |
|---|---|---|---|---|
| Default | 0.50 | 0.96 | 0.81 | 0.88 |
| **Optimal** | **~0.35** | **~0.87** | **~0.88** | **~0.88** |

> Lowering the threshold improves recall (catch more frauds) at a small precision cost — a worthwhile trade-off in real fraud detection systems where missing a fraud is far more costly than a false alarm.

---

### Explainability Results

SHAP analysis was run on the strongest tree booster model.

**Top 5 Most Important Features (by Mean |SHAP|):**

| Feature | Mean \|SHAP\| | Direction |
|---|---|---|
| V14 | Highest | Lower values → higher fraud score |
| V4 | High | Higher values → higher fraud score |
| V11 | High | Lower values → higher fraud score |
| V12 | Medium | Lower values → higher fraud score |
| V10 | Medium | Lower values → higher fraud score |

> The `V` features are PCA-transformed and anonymized, so they don't have direct business interpretation — but SHAP still shows which components dominate the fraud signal.

**Visualizations generated:**
- `Visualizations/Confusion matrix/` — per-model confusion matrices + optimal threshold matrix
- `Visualizations/figures/roc_curves.png` — ROC curve comparison across all 4 models
- `Visualizations/figures/pr_curves.png` — Precision-Recall curve comparison
- `Visualizations/figures/Random Forest_f1_threshold.png` — F1 vs. threshold sweep
- `Visualizations/Shap/` — SHAP beeswarm + bar importance plots

---