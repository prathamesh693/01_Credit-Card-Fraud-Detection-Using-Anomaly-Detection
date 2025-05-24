import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import joblib

# Load preprocessed training data
df_train = pd.read_csv("6_Output/train_preprocessed.csv")

# Separate features and target
X_train = df_train.drop(columns='Class')

# Train Isolation Forest
iso_model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
iso_model.fit(X_train)
joblib.dump(iso_model, '5_Models/model_isolation_forest.pkl')

# Train One-Class SVM
svm_model = OneClassSVM(kernel='rbf', gamma='scale', nu=0.01)
svm_model.fit(X_train)
joblib.dump(svm_model, '5_Models/model_oneclass_svm.pkl')

print("Both models trained and saved successfully.")
