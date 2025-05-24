# -*- coding: utf-8 -*-
"""
Created on Fri May 23 19:48:13 2025

@author: ratho
"""

import pandas as pd
import joblib

# Load preprocessed test data
df_test = pd.read_csv("6_Output/test_preprocessed.csv")
X_test = df_test.drop(columns='Class')

# Load models
iso_model = joblib.load("5_Models/model_isolation_forest.pkl")
svm_model = joblib.load("5_Models/model_oneclass_svm.pkl")

# Make predictions
iso_preds = iso_model.predict(X_test)
svm_preds = svm_model.predict(X_test)

# Prepare results DataFrame
results = X_test.copy()
results['iso_pred'] = iso_preds
results['svm_pred'] = svm_preds

# Convert anomaly flags: -1 means fraud (1), else 0
results['iso_fraud'] = results['iso_pred'].apply(lambda x: 1 if x == -1 else 0)
results['svm_fraud'] = results['svm_pred'].apply(lambda x: 1 if x == -1 else 0)

# Save predictions to CSV
results.to_csv("6_Output/predicted_results.csv", index=False)
print("Predictions saved to output/predicted_results.csv")
