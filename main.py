# ================================
# AI Fairness Audit - Main Pipeline
# ================================

from src.data_loader import load_data
from src.preprocessing import preprocess
from src.model import train_model
from src.evaluate import evaluate_model
from src.fairness import demographic_parity, equal_opportunity
from src.fairlearn_analysis import compute_fairness_metrics, mitigate_bias

from sklearn.model_selection import train_test_split

import os
import pandas as pd

# ================================
# Ensure output directory exists
# ================================
if not os.path.exists("outputs"):
    os.makedirs("outputs")

# ================================
# 1. Load Data
# ================================
print("\nLoading dataset...")
df = load_data()

# ================================
# 2. Preprocess Data
# ================================
print("Preprocessing data...")
X, y, df = preprocess(df)

# ================================
# 3. Train/Test Split
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Sensitive feature
sensitive_train = df.loc[X_train.index, 'age_group']
sensitive_test = df.loc[X_test.index, 'age_group']

# ================================
# 4. Train Baseline Model
# ================================
print("\nTraining baseline model...")
model = train_model(X_train, y_train)

# ================================
# 5. Predictions
# ================================
y_pred = model.predict(X_test)

# ================================
# 6. Accuracy Evaluation
# ================================
accuracy = evaluate_model(y_test, y_pred)
print(f"\nBaseline Accuracy: {accuracy:.4f}")

# ================================
# 7. Basic Fairness Metrics
# ================================
df_test = X_test.copy()
df_test['y_true'] = y_test
df_test['y_pred'] = y_pred
df_test['age_group'] = sensitive_test

dp_basic = demographic_parity(df_test)
eo_young, eo_adult = equal_opportunity(df_test)

print("\nBasic Fairness Metrics:")
print(f"Demographic Parity Gap: {dp_basic:.4f}")
print(f"Equal Opportunity (Young): {eo_young:.4f}")
print(f"Equal Opportunity (Adult): {eo_adult:.4f}")

# ================================
# 8. Fairlearn Advanced Metrics
# ================================
dp_diff, eo_diff = compute_fairness_metrics(
    y_test, y_pred, sensitive_test
)

print("\nFairlearn Metrics (Before Mitigation):")
print(f"Demographic Parity Difference: {dp_diff:.4f}")
print(f"Equalized Odds Difference: {eo_diff:.4f}")

# ================================
# 9. Bias Mitigation
# ================================
print("\nApplying bias mitigation...")
mitigator = mitigate_bias(X_train, y_train, sensitive_train)

y_pred_mitigated = mitigator.predict(X_test)

# ================================
# 10. Evaluate After Mitigation
# ================================
accuracy_new = evaluate_model(y_test, y_pred_mitigated)

dp_diff_new, eo_diff_new = compute_fairness_metrics(
    y_test, y_pred_mitigated, sensitive_test
)

print("\nAfter Mitigation:")
print(f"Accuracy: {accuracy_new:.4f}")
print(f"Demographic Parity Difference: {dp_diff_new:.4f}")
print(f"Equalized Odds Difference: {eo_diff_new:.4f}")


print("\nPrediction Distribution After Mitigation:")
print(pd.Series(y_pred_mitigated).value_counts())

dist = pd.Series(y_pred_mitigated).value_counts(normalize=True)
print("\nPrediction Distribution (%):")
print(dist)

# ================================
# 11. Save Results
# ================================
with open("outputs/metrics.txt", "w") as f:
    f.write("=== Baseline ===\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"DP Diff: {dp_diff:.4f}\n")
    f.write(f"EO Diff: {eo_diff:.4f}\n\n")

    f.write("=== After Mitigation ===\n")
    f.write(f"Accuracy: {accuracy_new:.4f}\n")
    f.write(f"DP Diff: {dp_diff_new:.4f}\n")
    f.write(f"EO Diff: {eo_diff_new:.4f}\n")

print("\nResults saved to outputs/metrics.txt")

# ================================
# 12. Visualization (Optional)
# ================================
try:
    from src.visualization import plot_bias_comparison
    
    plot_bias_comparison(dp_diff, dp_diff_new)
    print("Bias comparison plot saved.")
    
except:
    print("Visualization module not found, skipping plots.")

# ================================
# END
# ================================