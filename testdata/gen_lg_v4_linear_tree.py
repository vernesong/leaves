#!/usr/bin/env python3
"""Generate LightGBM v4.6.0 test models for leaves — linear tree + JSON v4 support.

Requires: lightgbm >= 4.6.0

Output files:
  - lg_linear_tree_breast_cancer.model      text format (linear_tree)
  - lg_linear_tree_breast_cancer.json        JSON format v4 (linear_tree)
  - lg_linear_tree_breast_cancer_true_predictions.txt
  - lg_v4_gbdt_breast_cancer.json            JSON format v4 (non-linear, gbdt)
  - lg_v4_gbdt_breast_cancer_true_predictions.txt
  - breast_cancer_linear_tree_test.tsv       test features
"""

import numpy as np
import lightgbm as lgb
from sklearn import datasets
from sklearn.model_selection import train_test_split

print(f"LightGBM version: {lgb.__version__}")

X, y = datasets.load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

n_estimators = 10
d_train = lgb.Dataset(X_train, label=y_train)

# ── 1. Linear Tree model (text format) ───────────────────────────────────
print("\n=== Training linear_tree model (text format) ===")
params_lt = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'linear_tree': True,
    'linear_lambda': 1.0,
    'num_leaves': 8,
    'min_data_in_leaf': 5,
    'verbosity': -1,
}
clf_lt = lgb.train(params_lt, d_train, n_estimators)
y_pred_lt = clf_lt.predict(X_test, raw_score=True)  # raw scores, no sigmoid
clf_lt.save_model('lg_linear_tree_breast_cancer.model')
np.savetxt('lg_linear_tree_breast_cancer_true_predictions.txt', y_pred_lt)
print(f"  Saved: lg_linear_tree_breast_cancer.model")
print(f"  Predictions shape: {y_pred_lt.shape}")

# ── 2. Linear Tree model (JSON format v4) ────────────────────────────────
print("\n=== Dumping linear_tree model (JSON format v4) ===")
d_lt = clf_lt.dump_model()
import json
with open('lg_linear_tree_breast_cancer.json', 'w') as f:
    json.dump(d_lt, f, indent=1)
print(f"  Saved: lg_linear_tree_breast_cancer.json")
print(f"  Version: {d_lt.get('version', 'unknown')}")
print(f"  Has objective: {'objective' in d_lt}")

# Verify JSON structure
has_leaf_const = False
for ti in d_lt.get('tree_info', []):
    tree_str = json.dumps(ti)
    if '"leaf_const"' in tree_str:
        has_leaf_const = True
        break
print(f"  Linear tree detected in JSON: {has_leaf_const}")

# ── 3. Standard GBDT model (JSON format v4, non-linear) ──────────────────
print("\n=== Training gbdt model (JSON format v4, no linear tree) ===")
params_gbdt = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'num_leaves': 8,
    'min_data_in_leaf': 5,
    'verbosity': -1,
}
clf_gbdt = lgb.train(params_gbdt, d_train, n_estimators)
y_pred_gbdt = clf_gbdt.predict(X_test, raw_score=True)  # raw scores
d_gbdt = clf_gbdt.dump_model()
with open('lg_v4_gbdt_breast_cancer.json', 'w') as f:
    json.dump(d_gbdt, f, indent=1)
np.savetxt('lg_v4_gbdt_breast_cancer_true_predictions.txt', y_pred_gbdt)
print(f"  Saved: lg_v4_gbdt_breast_cancer.json")
print(f"  Version: {d_gbdt.get('version', 'unknown')}")

# ── 4. Save test data ────────────────────────────────────────────────────
np.savetxt('breast_cancer_linear_tree_test.tsv', X_test, delimiter='\t')
print(f"\nSaved test data: breast_cancer_linear_tree_test.tsv ({X_test.shape})")

print("\n✅ All test models generated successfully!")
