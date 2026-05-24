# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")  # for cleaner output

# Scikit-learn utilities
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, ConfusionMatrixDisplay, roc_auc_score,
    average_precision_score, roc_curve, precision_recall_curve,
    precision_recall_fscore_support, f1_score
)

# Gradient-boosting libraries
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# SMOTE for synthetic minority oversampling
from imblearn.over_sampling import SMOTE

# SHAP for model interpretability
import shap

# Joblib for serialising trained models
import joblib

# DATA LOADING
df = pd.read_csv('Data/creditcard.csv')

# FEATURE / TARGET SPLIT
X = df.drop(columns='Class')
y = df[['Class']]

# TRAIN / TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# FEATURE SCALING
scaler = StandardScaler()
X_train[['scaled_amount', 'scaled_time']] = scaler.fit_transform(X_train[['Amount', 'Time']])
X_test[['scaled_amount', 'scaled_time']] = scaler.transform(X_test[['Amount', 'Time']])

# Drop the original un-scaled columns now that scaled versions exist
X_train = X_train.drop(columns=["Amount", "Time"])
X_test = X_test.drop(columns=["Amount", "Time"])

# SMOTE — SYNTHETIC MINORITY OVERSAMPLING
print("Before SMOTE:")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test:  {X_test.shape}, y_test:  {y_test.shape}")
print(y_train.value_counts().rename("train_class_counts"))

smote = SMOTE(k_neighbors=5, random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("\nAfter SMOTE:")
print(f"X_train_smote: {X_train_smote.shape}, y_train_smote: {y_train_smote.shape}")
print(y_train_smote.value_counts().rename("smote_train_class_counts"))

# MODEL DEFINITIONS
models = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced",
        max_iter=2_000,
        solver="lbfgs",
        random_state=42,
        n_jobs=-1,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=250,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
    "XGBoost": XGBClassifier(
        n_estimators=350,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=350,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    ),
}


# TRAINING & EVALUATION LOOP
results = {}

for model_name, model in models.items():
    print(f'\nTraining {model_name}...')

    # Fit on SMOTE-balanced training data
    model.fit(X_train_smote, y_train_smote)

    # Predict on the original test set
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Compute key metrics
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)

    # Print detailed per-class metrics
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"], digits=4))
    print(f"ROC-AUC: {roc_auc:.2f}")
    print(f"PR-AUC:  {pr_auc:.2f}")

    # Save confusion matrix visualisation
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=['Legitimate', 'Fraud'],
        cmap='Blues', values_format='d'
    )
    plt.title(f"{model_name} Confusion Matrix")
    plt.savefig(f"Visualizations/Confusion matrix/{model_name}_confusion_matrix.png")
    plt.close()

    # Cache results for later comparison
    results[model_name] = {
        'model': model,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc
    }

# ROC CURVE COMPARISON
plt.figure(figsize=(10, 7))
for model_name, result in results.items():
    fpr, tpr, _ = roc_curve(y_test, result['y_proba'])
    plt.plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUC={result['roc_auc']:.2f})")

plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
plt.title("ROC Curves")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("Visualizations/figures/roc_curves.png")
plt.close()

# PRECISION–RECALL CURVE COMPARISON
plt.figure(figsize=(10, 7))
for model_name, result in results.items():
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, result['y_proba'])
    plt.plot(recall_curve, precision_curve, linewidth=2, label=f"{model_name} (AP={result['pr_auc']:.2f})")

baseline = y_test.mean().iloc[0]
plt.axhline(baseline, linestyle="--", color="gray", label=f"Fraud baseline ({baseline:.2f})")
plt.title("Precision-Recall Curves")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig("Visualizations/figures/pr_curves.png")
plt.close()

# MODEL PERFORMANCE SUMMARY TABLE
summary = []

for model_name, result in results.items():
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, result['y_pred'], average='binary', zero_division=0)

    summary.append({
        'Model': model_name,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC-AUC': result['roc_auc'],
        'PR-AUC': result['pr_auc']
    })

summary_df = pd.DataFrame(summary).sort_values(by='PR-AUC', ascending=False).reset_index(drop=True)
print("\nModel Performance Summary:")
print(summary_df)

# DECISION-THRESHOLD TUNING
best_model_name = summary_df.loc[0, 'Model']
best_model = results[best_model_name]['model']
best_proba = results[best_model_name]['y_proba']

print(f"Best model: {best_model_name}")

thresholds = np.linspace(0.01, 0.99, 99)
f1_scores = []

for threshold in thresholds:
    thesholded_pred = (best_proba >= threshold).astype(int)
    f1_scores.append(f1_score(y_test, thesholded_pred, zero_division=0))

# Identify the threshold that yields the highest F1
best_threshold_index = int(np.argmax(f1_scores))
best_threshold = thresholds[best_threshold_index]
best_threshold_f1 = f1_scores[best_threshold_index]

# Plot F1 vs threshold ──
plt.figure(figsize=(10, 6))
plt.plot(thresholds, f1_scores, marker='o', markersize=3, linewidth=2)
plt.axvline(best_threshold, color='red', linestyle='--', label=f"Best threshold = {best_threshold:.2f}")
plt.title(f"F1 vs Threshold for {best_model_name}")
plt.xlabel("Decision Threshold")
plt.ylabel("Fraud-Class F1 Score")
plt.legend()
plt.tight_layout()
plt.savefig(f"Visualizations/figures/{best_model_name}_f1_threshold.png")
plt.close()

# Apply the optimal threshold and print the updated report 
optimal_pred = (best_proba >= best_threshold).astype(int)
print(f"Optimal threshold: {best_threshold:.2f}")
print(f"Best threshold F1: {best_threshold_f1:.2f}")
print(classification_report(y_test, optimal_pred, target_names=["Legitimate", "Fraud"], digits=4))

# Confusion matrix at optimal threshold
ConfusionMatrixDisplay.from_predictions(
    y_test,
    optimal_pred,
    display_labels=['Legitimate', 'Fraud'],
    cmap='Greens',
    values_format='d',
)
plt.title(f"{best_model_name} Confusion Matrix at Optimal Threshold")
plt.savefig(f"Visualizations/Confusion matrix/{best_model_name}_optimal_confusion_matrix.png")
plt.close()

# SHAP EXPLAINABILITY
if best_model_name not in {"XGBoost", "LightGBM"}:
    preferred_tree_models = [name for name in ["XGBoost", "LightGBM"] if name in results]
    shap_model_name = max(preferred_tree_models, key=lambda name: results[name]["pr_auc"])
    print(f"Best model is {best_model_name}; using {shap_model_name} for SHAP because it is the stronger tree booster.")
else:
    shap_model_name = best_model_name

shap_model = results[shap_model_name]['model']

# Stratified sub-sample of the test set for SHAP (faster computation)
shap_sample_size = min(5000, len(X_test))
shap_sample = X_test.copy()
shap_sample['Class'] = y_test.values
shap_sample = (
    shap_sample
    .groupby("Class", group_keys=False)
    .apply(lambda part: part.sample(
        n=min(len(part), max(1, int(shap_sample_size * len(part) / len(X_test)))),
        random_state=42,
    ))
)

# If the stratified sample came up short, top up with random remaining rows
if len(shap_sample) < shap_sample_size:
    remaining = X_test.drop(index=shap_sample.index).sample(
        n=min(shap_sample_size - len(shap_sample), len(X_test) - len(shap_sample)),
        random_state=42,
    )
    shap_sample = pd.concat([shap_sample.drop(columns='Class', errors='ignore'), remaining], axis=0)
else:
    shap_sample = shap_sample.drop(columns='Class')

# Compute SHAP values
explainer = shap.TreeExplainer(shap_model)
shap_values = explainer.shap_values(shap_sample)

# TreeExplainer may return a list [class-0 SHAP, class-1 SHAP]; we need fraud (class 1)
if isinstance(shap_values, list):
    shap_values_for_fraud = shap_values[1]
else:
    shap_values_for_fraud = shap_values

print(f"SHAP model: {shap_model_name}")
print(f"SHAP sample shape: {shap_sample.shape}")

# Beeswarm summary plot (shows feature impact direction and magnitude)
shap.summary_plot(shap_values_for_fraud, shap_sample, show =False, max_display=20)
plt.title(f"SHAP Summary Plot - {shap_model_name}")
plt.tight_layout()
plt.savefig(f"Visualizations/Shap/{shap_model_name}_shap_summary.png")
plt.close()

# Bar plot of mean absolute SHAP values (global feature importance)
shap.summary_plot(shap_values_for_fraud, shap_sample, plot_type="bar", show=False, max_display=20)
plt.title(f"Mean |SHAP| Feature Importance — {shap_model_name}")
plt.tight_layout()
plt.savefig(f"Visualizations/Shap/{shap_model_name}_shap_importance.png")
plt.close()

# TOP-5 FEATURE ANALYSIS
mean_abs_shap = pd.Series(
    np.abs(shap_values_for_fraud).mean(axis=0),
    index=shap_sample.columns,
).sort_values(ascending=False)

top_feature_records = []

for feature, importance in mean_abs_shap.head(5).items():
    feature_position = shap_sample.columns.get_loc(feature)
    shap_feature_values = shap_values_for_fraud[:, feature_position]

    correlation = np.corrcoef(shap_sample[feature], shap_feature_values)[0, 1]

    if np.isnan(correlation):
        direction = "mixed or non-linear effect"
    elif correlation > 0:
        direction = "higher values generally increase the fraud score"
    else:
        direction = "lower values generally increase the fraud score"

    y_test_flat = y_test.squeeze()
    top_feature_records.append({
        'Feature': feature,
        'Mean |SHAP|': importance,
        'Direction': direction,
        'Legitimate Median': X_test.loc[y_test_flat == 0, feature].median(),
        'Fraud Median': X_test.loc[y_test_flat == 1, feature].median(),
    })

top_features_df = pd.DataFrame(top_feature_records)
print("\nTop 5 Most Important Features Based on SHAP Values:")
print(top_features_df)

# MODEL PERSISTENCE
joblib.dump(best_model, f"Models/{best_model_name}_model.pkl")
joblib.dump(scaler, "Models/scaler.pkl")
print(f"Saved best model ({best_model_name}) and scaler")
